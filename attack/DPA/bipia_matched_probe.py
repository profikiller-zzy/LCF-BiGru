#!/usr/bin/env python3
"""
BIPIA email matched-pair LCF prompt-injection probe.

Tests whether LCF's hidden-state anomaly score discriminates injected
emails from clean emails of the same format, when calibration and test
are drawn from the same deployment distribution.

Protocol
--------
100 base BIPIA emails, each appearing in 3 variants: clean / text-inject /
code-inject. Five-fold cross-validation: per fold, 80 emails are used as
calibration (clean-only) and 20 as test. Each email is in exactly one
fold's test set; aggregated metrics cover all 100 test pairs per model.

Per fold:
  1. Collect per-layer delta h_{l+1} - h_l at step-0 last-token position
     from the 80 cal clean emails.
  2. Per-dim μ, σ per layer from the cal.
  3. Per-layer diag_mahal scores: ||(delta - μ)/σ||_2.
  4. Per-layer z-score stats (cross-example) from the cal's score matrix.
  5. Ledoit-Wolf fit on the cal's z-score vectors.
  6. LOO threshold on the cal at the 90th percentile (target 10% FPR).
  7. Score 20 held-out test emails, one per variant (clean, text, code).

Metrics (aggregated across 5 folds, 100 test pairs per variant)
---------------------------------------------------------------
- Operational FPR / TPR (text, code) at each fold's own threshold
- Paired Δ = inject_score - clean_score per base email
- Cohen's d (paired), paired t-test, Wilcoxon, % pairs Δ > 0
- Length-delta diagnostic:
    * r(length_delta, score_delta): does paired score change track
      paired length change?
    * Length-controlled paired effect: regress score_delta on
      length_delta, report residual mean + Cohen's d
- Raw length correlation across pooled scores (sanity check)

Data collection is one-shot per model (300 forward passes). All fold
analysis then runs in numpy on cached deltas.

Environment
-----------
MODEL_PATH        HF repo id, default meta-llama/Meta-Llama-3-8B-Instruct
PROMPT_TEMPLATE   alpaca | qwen | gemma | llama3
TARGET_FPR        default 0.10
OUTPUT_DIR        default attack/DPA/bipia_matched_results/<model_name>
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.covariance import LedoitWolf

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "meta-llama/Meta-Llama-3-8B-Instruct")
PROMPT_TEMPLATE = os.environ.get("PROMPT_TEMPLATE", "alpaca")
TARGET_FPR = float(os.environ.get("TARGET_FPR", "0.10"))

DPA_ROOT = Path(__file__).resolve().parent
MODEL_NAME = os.path.basename(MODEL_PATH)
DEFAULT_POOL = DPA_ROOT / "data" / "test_data" / "poison" / "prompt_injection" / "bipia_email_matched_pool.json"
POOL_PATH = Path(os.environ.get("POOL_PATH", str(DEFAULT_POOL)))
# Derive domain name from the pool filename if not set explicitly
DOMAIN = os.environ.get("DOMAIN", POOL_PATH.stem.replace("_matched_pool", ""))
OUTPUT_DIR = Path(os.environ.get(
    "OUTPUT_DIR",
    str(DPA_ROOT / "bipia_matched_results" / DOMAIN / MODEL_NAME) if DOMAIN != "bipia_email"
    else str(DPA_ROOT / "bipia_matched_results" / MODEL_NAME),
))


# -----------------------------------------------------------------------------
# Prompt building (standard LCF templates)
# -----------------------------------------------------------------------------
def build_model_inputs(tokenizer, instruction, user_input="", prompt_template="alpaca"):
    instruction = (instruction or "").strip()
    user_input = (user_input or "").strip()
    if prompt_template == "alpaca":
        if user_input:
            prompt_text = f"### Instruction:\n{instruction}\n\n### Input:\n{user_input}\n\n### Response:\n"
        else:
            prompt_text = f"### Instruction:\n{instruction}\n\n### Response:\n"
    elif prompt_template == "qwen":
        content = f"{instruction}\n{user_input}" if user_input else instruction
        prompt_text = f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
    elif prompt_template == "gemma":
        content = f"{instruction}\n{user_input}" if user_input else instruction
        prompt_text = f"<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"
    elif prompt_template == "llama3":
        content = f"{instruction}\n{user_input}" if user_input else instruction
        prompt_text = (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{content}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    else:
        prompt_text = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096)
    return inputs, prompt_text


def load_model_and_tokenizer(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": torch.cuda.current_device()},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def get_num_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
        if hasattr(model.config, attr):
            return getattr(model.config, attr)
    raise ValueError("Cannot determine number of layers")


# -----------------------------------------------------------------------------
# Delta collection
# -----------------------------------------------------------------------------
@torch.no_grad()
def collect_deltas(model, tokenizer, entries, num_layers, label=""):
    n = len(entries)
    deltas = None
    char_lens = np.zeros(n, dtype=np.int32)
    tok_lens = np.zeros(n, dtype=np.int32)
    t0 = time.time()
    for i, ex in enumerate(entries):
        inputs, prompt_text = build_model_inputs(
            tokenizer, ex["instruction"], ex.get("input", ""), PROMPT_TEMPLATE,
        )
        char_lens[i] = len(prompt_text)
        input_ids = inputs["input_ids"].to(model.device)
        tok_lens[i] = input_ids.shape[1]
        out = model(input_ids=input_ids, output_hidden_states=True)
        hs = out.hidden_states  # (num_layers+1,) tensors
        if deltas is None:
            hidden_dim = hs[0].shape[-1]
            deltas = np.zeros((n, num_layers, hidden_dim), dtype=np.float32)
        for l in range(num_layers):
            d = (hs[l + 1][:, -1, :].float() - hs[l][:, -1, :].float()).squeeze(0).cpu().numpy()
            deltas[i, l, :] = d
        if (i + 1) % 25 == 0:
            print(f"    [{label}] {i+1}/{n}  ({time.time()-t0:.0f}s elapsed)", flush=True)
    return deltas, char_lens, tok_lens


# -----------------------------------------------------------------------------
# LCF scoring (per fold)
# -----------------------------------------------------------------------------
def fit_fold_cal(cal_deltas):
    """Given cal_deltas [n_cal, L, d], fit LCF stats and return a closure."""
    n_cal, num_layers, hidden_dim = cal_deltas.shape

    # Per-dim μ, σ per layer (from cal)
    mu = cal_deltas.mean(axis=0)                              # [L, d]
    sigma = np.maximum(cal_deltas.std(axis=0), 1e-8)          # [L, d]

    # Per-layer diag_mahal on cal
    cal_scores = np.linalg.norm((cal_deltas - mu) / sigma, axis=2)  # [n_cal, L]

    # Per-layer z-score stats
    lmu = cal_scores.mean(axis=0)
    lsig = np.maximum(cal_scores.std(axis=0), 1e-8)
    cal_z = (cal_scores - lmu) / lsig                          # [n_cal, L]

    # LW covariance fit
    mu_lw = cal_z.mean(axis=0)
    lw = LedoitWolf().fit(cal_z)
    prec = lw.precision_

    # LOO threshold
    loo = np.zeros(n_cal)
    for i in range(n_cal):
        mask = np.ones(n_cal, bool); mask[i] = False
        c = cal_z[mask]
        m = c.mean(axis=0)
        lwi = LedoitWolf().fit(c)
        d = cal_z[i] - m
        loo[i] = float(np.sqrt(d @ lwi.precision_ @ d))
    threshold = max(float(np.percentile(loo, 100 * (1 - TARGET_FPR))), 1.0)

    def score(deltas):
        """Score a batch of deltas [n, L, d] with this fold's cal statistics."""
        per_layer = np.linalg.norm((deltas - mu) / sigma, axis=2)    # [n, L]
        z = (per_layer - lmu) / lsig                                  # [n, L]
        diff = z - mu_lw
        lw_scores = np.sqrt(np.sum(diff @ prec * diff, axis=1))
        return lw_scores, z

    return score, threshold, loo


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------
def paired_stats(clean_scores, inject_scores):
    """Paired t-test + Wilcoxon + Cohen's d on within-pair deltas."""
    assert len(clean_scores) == len(inject_scores)
    deltas = inject_scores - clean_scores
    n = len(deltas)
    mean_d = float(deltas.mean())
    std_d = float(deltas.std(ddof=1)) if n > 1 else 0.0
    se_d = std_d / np.sqrt(n) if n > 1 else 0.0
    cohen_d = mean_d / std_d if std_d > 0 else float("nan")
    frac_pos = float((deltas > 0).mean())
    try:
        t, p_t = stats.ttest_rel(inject_scores, clean_scores)
    except Exception:
        t, p_t = float("nan"), float("nan")
    try:
        # Wilcoxon with zero handling
        w_stat, p_w = stats.wilcoxon(inject_scores, clean_scores, zero_method="pratt")
    except Exception:
        p_w = float("nan")
    return dict(
        n=n,
        mean=mean_d, std=std_d, se=se_d,
        cohen_d=cohen_d,
        frac_positive=frac_pos,
        t_stat=float(t), p_t=float(p_t), p_w=float(p_w),
    )


def length_controlled_effect(clean_scores, inject_scores, clean_lens, inject_lens):
    """
    Length-delta diagnostic: regress score_delta on length_delta,
    report residual mean + Cohen's d (length-controlled injection effect).
    """
    score_delta = inject_scores - clean_scores
    length_delta = (inject_lens - clean_lens).astype(float)
    n = len(score_delta)

    # Correlation between score_delta and length_delta
    if length_delta.std() < 1e-9 or score_delta.std() < 1e-9:
        r, p_r = float("nan"), float("nan")
    else:
        r, p_r = stats.pearsonr(length_delta, score_delta)

    # Linear regression score_delta = a * length_delta + b
    if length_delta.std() < 1e-9:
        a, b = 0.0, float(score_delta.mean())
        residuals = score_delta.copy()
    else:
        a, b = np.polyfit(length_delta, score_delta, 1)
        residuals = score_delta - (a * length_delta + b)
        # add back intercept so "residual mean" = intercept = length-zero injection effect
        residuals = residuals + b

    res_mean = float(residuals.mean())
    res_std = float(residuals.std(ddof=1)) if n > 1 else 0.0
    res_d = res_mean / res_std if res_std > 0 else float("nan")

    return dict(
        r_length_score=float(r),
        p_length_score=float(p_r),
        regression_slope=float(a),
        regression_intercept=float(b),
        residual_mean=res_mean,
        residual_std=res_std,
        residual_cohen_d=res_d,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=" * 72)
    print(f"BIPIA matched probe — {MODEL_NAME}")
    print(f"  template={PROMPT_TEMPLATE}  target_fpr={TARGET_FPR}")
    print(f"  pool={POOL_PATH}")
    print(f"  output={OUTPUT_DIR}")
    print("=" * 72)

    pool = json.load(open(POOL_PATH))
    clean_entries = pool["clean"]
    text_entries = pool["inject_text"]
    code_entries = pool["inject_code"]
    folds = pool["folds"]
    n_base = pool["n_base"]
    assert n_base == len(clean_entries) == len(text_entries) == len(code_entries)
    print(f"  base emails: {n_base}, folds: {len(folds)}")

    print("\nLoading model...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    num_layers = get_num_layers(model)
    print(f"  loaded in {time.time()-t0:.0f}s, {num_layers} layers")

    print("\nCollecting deltas...")
    t0 = time.time()
    clean_deltas, clean_clen, clean_tlen = collect_deltas(model, tokenizer, clean_entries, num_layers, "clean")
    text_deltas, text_clen, text_tlen = collect_deltas(model, tokenizer, text_entries, num_layers, "text-inject")
    code_deltas, code_clen, code_tlen = collect_deltas(model, tokenizer, code_entries, num_layers, "code-inject")
    print(f"  delta collection done in {time.time()-t0:.0f}s")
    print(f"  clean char-len mean={clean_clen.mean():.0f}  text={text_clen.mean():.0f}  code={code_clen.mean():.0f}")
    print(f"  clean tok-len  mean={clean_tlen.mean():.0f}  text={text_tlen.mean():.0f}  code={code_tlen.mean():.0f}")

    # Free GPU
    del model
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 5-fold analysis
    # -------------------------------------------------------------------------
    print("\n5-fold analysis...")
    all_clean_scores = np.full(n_base, np.nan)
    all_text_scores = np.full(n_base, np.nan)
    all_code_scores = np.full(n_base, np.nan)
    all_clean_z = np.full((n_base, num_layers), np.nan)
    all_text_z = np.full((n_base, num_layers), np.nan)
    all_code_z = np.full((n_base, num_layers), np.nan)
    fold_assignment = np.full(n_base, -1, dtype=int)
    fold_thresholds = np.zeros(len(folds))
    fold_details = []

    for fold in folds:
        k = fold["fold"]
        cal_ids = np.array(fold["cal"])
        test_ids = np.array(fold["test"])
        cal_deltas_k = clean_deltas[cal_ids]

        score_fn, threshold, loo_cal = fit_fold_cal(cal_deltas_k)
        fold_thresholds[k] = threshold

        # Score all entries with this fold's cal; only keep scores for test_ids
        clean_all_k, clean_z_k = score_fn(clean_deltas)
        text_all_k, text_z_k = score_fn(text_deltas)
        code_all_k, code_z_k = score_fn(code_deltas)

        for i in test_ids:
            all_clean_scores[i] = clean_all_k[i]
            all_text_scores[i] = text_all_k[i]
            all_code_scores[i] = code_all_k[i]
            all_clean_z[i] = clean_z_k[i]
            all_text_z[i] = text_z_k[i]
            all_code_z[i] = code_z_k[i]
            fold_assignment[i] = k

        # Per-fold metrics on the 20 test emails
        fc = clean_all_k[test_ids]
        ft = text_all_k[test_ids]
        fd = code_all_k[test_ids]
        fold_fpr = float((fc >= threshold).mean())
        fold_tpr_t = float((ft >= threshold).mean())
        fold_tpr_c = float((fd >= threshold).mean())

        fold_details.append({
            "fold": k,
            "n_cal": int(len(cal_ids)),
            "n_test": int(len(test_ids)),
            "threshold": float(threshold),
            "loo_mean": float(loo_cal.mean()),
            "loo_std": float(loo_cal.std()),
            "fpr": fold_fpr,
            "tpr_text": fold_tpr_t,
            "tpr_code": fold_tpr_c,
            "tpr_minus_fpr_text": fold_tpr_t - fold_fpr,
            "tpr_minus_fpr_code": fold_tpr_c - fold_fpr,
        })
        print(f"  fold {k}: thr={threshold:.3f}  FPR={fold_fpr:.1%}  "
              f"TPR_text={fold_tpr_t:.1%}  TPR_code={fold_tpr_c:.1%}")

    assert not np.isnan(all_clean_scores).any(), "some emails never tested"

    # -------------------------------------------------------------------------
    # Aggregated metrics
    # -------------------------------------------------------------------------
    per_email_thr = np.array([fold_thresholds[fold_assignment[i]] for i in range(n_base)])
    agg_fpr = float((all_clean_scores >= per_email_thr).mean())
    agg_tpr_text = float((all_text_scores >= per_email_thr).mean())
    agg_tpr_code = float((all_code_scores >= per_email_thr).mean())

    paired_text = paired_stats(all_clean_scores, all_text_scores)
    paired_code = paired_stats(all_clean_scores, all_code_scores)

    # Length-delta diagnostic (the critical one for this experiment)
    len_ctrl_text = length_controlled_effect(all_clean_scores, all_text_scores, clean_clen, text_clen)
    len_ctrl_code = length_controlled_effect(all_clean_scores, all_code_scores, clean_clen, code_clen)

    # Token-length diagnostic (alternative length metric, less noisy)
    len_ctrl_text_tok = length_controlled_effect(all_clean_scores, all_text_scores, clean_tlen, text_tlen)
    len_ctrl_code_tok = length_controlled_effect(all_clean_scores, all_code_scores, clean_tlen, code_tlen)

    # Pooled length correlation (across all 300 scores)
    all_scores = np.concatenate([all_clean_scores, all_text_scores, all_code_scores])
    all_char_lens = np.concatenate([clean_clen, text_clen, code_clen])
    all_tok_lens = np.concatenate([clean_tlen, text_tlen, code_tlen])
    pooled_r_char = float(stats.pearsonr(all_char_lens, all_scores)[0])
    pooled_r_tok = float(stats.pearsonr(all_tok_lens, all_scores)[0])

    # Within-split length correlation (controls for between-split shift)
    within_r_clean = float(stats.pearsonr(clean_clen, all_clean_scores)[0])
    within_r_text = float(stats.pearsonr(text_clen, all_text_scores)[0])
    within_r_code = float(stats.pearsonr(code_clen, all_code_scores)[0])

    # ---- Per-layer signal (top layers) ----
    # Mean z-score shift from clean to inject, averaged across all 100 test emails
    # (each email's z-score uses its own fold's cal, which is exactly the k-fold CV guarantee)
    assert not np.isnan(all_clean_z).any(), "some clean z-scores never written"
    assert not np.isnan(all_text_z).any(), "some text z-scores never written"
    assert not np.isnan(all_code_z).any(), "some code z-scores never written"

    text_layer_shift = (all_text_z - all_clean_z).mean(axis=0)  # [num_layers]
    code_layer_shift = (all_code_z - all_clean_z).mean(axis=0)  # [num_layers]

    top5_text = np.argsort(-text_layer_shift)[:5]
    top5_code = np.argsort(-code_layer_shift)[:5]

    # Also compute per-layer AUC (text vs clean) for a more robust ranking
    def per_layer_auc(clean_z, inject_z):
        n = clean_z.shape[0]
        aucs = np.zeros(clean_z.shape[1])
        for l in range(clean_z.shape[1]):
            labels = np.concatenate([np.zeros(n), np.ones(n)])
            scores = np.concatenate([clean_z[:, l], inject_z[:, l]])
            idx = np.argsort(-scores)
            ls = labels[idx]
            tp, auc = 0, 0.0
            for lab in ls:
                if lab == 1:
                    tp += 1
                else:
                    auc += tp
            aucs[l] = auc / (n * n) if n > 0 else 0.5
        return aucs

    text_layer_auc = per_layer_auc(all_clean_z, all_text_z)
    code_layer_auc = per_layer_auc(all_clean_z, all_code_z)
    top5_text_auc = np.argsort(-text_layer_auc)[:5]
    top5_code_auc = np.argsort(-code_layer_auc)[:5]

    # -------------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"AGGREGATED RESULTS — {MODEL_NAME}, n=100 test pairs (5-fold CV)")
    print("=" * 72)
    print(f"  Operational FPR on clean test:        {agg_fpr:>6.1%}")
    print(f"  Operational TPR text-inject:          {agg_tpr_text:>6.1%}   "
          f"(TPR-FPR = {agg_tpr_text-agg_fpr:+.1%})")
    print(f"  Operational TPR code-inject:          {agg_tpr_code:>6.1%}   "
          f"(TPR-FPR = {agg_tpr_code-agg_fpr:+.1%})")
    print()
    print(f"  Paired Δ text-inject:  mean={paired_text['mean']:+.3f} ± {paired_text['se']:.3f}  "
          f"d={paired_text['cohen_d']:+.3f}  %>0={100*paired_text['frac_positive']:.0f}%  "
          f"p_t={paired_text['p_t']:.3g}")
    print(f"  Paired Δ code-inject:  mean={paired_code['mean']:+.3f} ± {paired_code['se']:.3f}  "
          f"d={paired_code['cohen_d']:+.3f}  %>0={100*paired_code['frac_positive']:.0f}%  "
          f"p_t={paired_code['p_t']:.3g}")
    print()
    print(f"  Length-delta diagnostic (text, char-len):")
    print(f"    r(Δlength, Δscore) = {len_ctrl_text['r_length_score']:+.3f}  "
          f"(p={len_ctrl_text['p_length_score']:.3g})")
    print(f"    residual Cohen's d (length-controlled) = {len_ctrl_text['residual_cohen_d']:+.3f}")
    print(f"  Length-delta diagnostic (code, char-len):")
    print(f"    r(Δlength, Δscore) = {len_ctrl_code['r_length_score']:+.3f}  "
          f"(p={len_ctrl_code['p_length_score']:.3g})")
    print(f"    residual Cohen's d (length-controlled) = {len_ctrl_code['residual_cohen_d']:+.3f}")
    print()
    print(f"  Pooled length correlation (300 scores):")
    print(f"    chars:  r={pooled_r_char:+.3f}")
    print(f"    tokens: r={pooled_r_tok:+.3f}")
    print(f"  Within-split length correlation (n=100 each):")
    print(f"    clean r={within_r_clean:+.3f}  text r={within_r_text:+.3f}  code r={within_r_code:+.3f}")
    print()
    print(f"  Top-5 layers for text-inject (by mean z-score shift):")
    print(f"    {[(f'L{l}', round(float(text_layer_shift[l]), 3)) for l in top5_text]}")
    print(f"  Top-5 layers for text-inject (by per-layer AUC):")
    print(f"    {[(f'L{l}', round(float(text_layer_auc[l]), 3)) for l in top5_text_auc]}")
    print(f"  Top-5 layers for code-inject (by mean z-score shift):")
    print(f"    {[(f'L{l}', round(float(code_layer_shift[l]), 3)) for l in top5_code]}")
    print(f"  Top-5 layers for code-inject (by per-layer AUC):")
    print(f"    {[(f'L{l}', round(float(code_layer_auc[l]), 3)) for l in top5_code_auc]}")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "config": {
            "model_path": MODEL_PATH,
            "model_name": MODEL_NAME,
            "prompt_template": PROMPT_TEMPLATE,
            "target_fpr": TARGET_FPR,
            "num_layers": int(num_layers),
            "n_base_emails": int(n_base),
            "n_folds": len(folds),
        },
        "folds": fold_details,
        "aggregated": {
            "operational_fpr": agg_fpr,
            "operational_tpr_text": agg_tpr_text,
            "operational_tpr_code": agg_tpr_code,
            "tpr_minus_fpr_text": agg_tpr_text - agg_fpr,
            "tpr_minus_fpr_code": agg_tpr_code - agg_fpr,
            "paired_text": paired_text,
            "paired_code": paired_code,
            "length_controlled_text_char": len_ctrl_text,
            "length_controlled_code_char": len_ctrl_code,
            "length_controlled_text_tok": len_ctrl_text_tok,
            "length_controlled_code_tok": len_ctrl_code_tok,
            "pooled_length_r_char": pooled_r_char,
            "pooled_length_r_tok": pooled_r_tok,
            "within_split_length_r": {
                "clean": within_r_clean,
                "text": within_r_text,
                "code": within_r_code,
            },
            "mean_threshold": float(fold_thresholds.mean()),
            "std_threshold": float(fold_thresholds.std()),
            "per_layer": {
                "text_layer_shift": [float(x) for x in text_layer_shift],
                "code_layer_shift": [float(x) for x in code_layer_shift],
                "text_layer_auc": [float(x) for x in text_layer_auc],
                "code_layer_auc": [float(x) for x in code_layer_auc],
                "top5_text_by_shift": [int(l) for l in top5_text],
                "top5_text_by_auc": [int(l) for l in top5_text_auc],
                "top5_code_by_shift": [int(l) for l in top5_code],
                "top5_code_by_auc": [int(l) for l in top5_code_auc],
            },
        },
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    np.savez_compressed(
        OUTPUT_DIR / "scores.npz",
        clean_scores=all_clean_scores,
        text_scores=all_text_scores,
        code_scores=all_code_scores,
        clean_z=all_clean_z,
        text_z=all_text_z,
        code_z=all_code_z,
        clean_char_lens=clean_clen,
        text_char_lens=text_clen,
        code_char_lens=code_clen,
        clean_tok_lens=clean_tlen,
        text_tok_lens=text_tlen,
        code_tok_lens=code_tlen,
        fold_assignment=fold_assignment,
        fold_thresholds=fold_thresholds,
        per_email_thr=per_email_thr,
    )
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
