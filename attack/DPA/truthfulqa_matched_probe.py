#!/usr/bin/env python3
"""
TruthfulQA MC1 in-distribution LCF probe.

Tests whether LCF's prefill anomaly score discriminates questions the
model answers correctly from questions it answers incorrectly on
TruthfulQA MC1, when calibration is drawn from the SAME TruthfulQA
distribution (not sft_data).

Protocol
--------
817 TruthfulQA MC1 questions. For each question:
  - Collect hidden-state deltas at step-0 (one forward pass per question)
  - Score MC1 candidates (sum-log-p, one forward pass per candidate)
  - Label: correct if argmax(candidate scores) == gold

5-fold CV:
  - Per fold: ~654 questions for cal, ~163 for test
  - Cal includes BOTH correct and incorrect questions (the defender
    doesn't know which is which at calibration time)
  - Fit LCF on cal → LOO threshold → score test
  - Aggregate across folds → 817 total test predictions

Reports:
  - Group AUC (incorrect > correct): does wrong-answer Qs score higher?
  - Mean D gap: D(incorrect) - D(correct)
  - Per-layer AUC: which layers (if any) discriminate correct vs incorrect?
  - Length correlation diagnostic

Environment
-----------
MODEL_PATH      HF repo id
PROMPT_TEMPLATE alpaca | qwen | gemma | llama3
TARGET_FPR      default 0.10
"""
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.covariance import LedoitWolf

MODEL_PATH = os.environ.get("MODEL_PATH", "meta-llama/Meta-Llama-3-8B-Instruct")
PROMPT_TEMPLATE = os.environ.get("PROMPT_TEMPLATE", "alpaca")
TARGET_FPR = float(os.environ.get("TARGET_FPR", "0.10"))

DPA_ROOT = Path(__file__).resolve().parent
MODEL_NAME = os.path.basename(MODEL_PATH)
OUTPUT_DIR = Path(os.environ.get(
    "OUTPUT_DIR",
    str(DPA_ROOT / "truthfulqa_matched_results" / MODEL_NAME),
))

SEED = 42
N_FOLDS = 5


# -----------------------------------------------------------------------------
# Prompt building
# -----------------------------------------------------------------------------
def build_prompt(tokenizer, question, prompt_template="alpaca"):
    question = (question or "").strip()
    if prompt_template == "alpaca":
        prompt_text = f"### Instruction:\n{question}\n\n### Response:\n"
    elif prompt_template == "qwen":
        prompt_text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    elif prompt_template == "gemma":
        prompt_text = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
    elif prompt_template == "llama3":
        prompt_text = (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{question}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    else:
        prompt_text = f"### Instruction:\n{question}\n\n### Response:\n"
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
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
# Data loading
# -----------------------------------------------------------------------------
def load_truthfulqa_mc1():
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    items = []
    for ex in ds:
        choices = list(ex["mc1_targets"]["choices"])
        labels = list(ex["mc1_targets"]["labels"])
        correct_idx = int(np.argmax(labels))
        items.append({
            "question": ex["question"],
            "choices": choices,
            "correct_idx": correct_idx,
        })
    return items


# -----------------------------------------------------------------------------
# Phase 1: Collect deltas + MC1 scoring
# -----------------------------------------------------------------------------
@torch.no_grad()
def collect_deltas(model, tokenizer, questions, num_layers):
    n = len(questions)
    deltas = None
    char_lens = np.zeros(n, dtype=np.int32)
    t0 = time.time()
    for i, q in enumerate(questions):
        inputs, prompt_text = build_prompt(tokenizer, q["question"], PROMPT_TEMPLATE)
        char_lens[i] = len(prompt_text)
        input_ids = inputs["input_ids"].to(model.device)
        out = model(input_ids=input_ids, output_hidden_states=True)
        hs = out.hidden_states
        if deltas is None:
            hidden_dim = hs[0].shape[-1]
            deltas = np.zeros((n, num_layers, hidden_dim), dtype=np.float32)
        for l in range(num_layers):
            d = (hs[l + 1][:, -1, :].float() - hs[l][:, -1, :].float()).squeeze(0).cpu().numpy()
            deltas[i, l, :] = d
        if (i + 1) % 100 == 0:
            print(f"    deltas: {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    return deltas, char_lens


@torch.no_grad()
def score_choice(model, tokenizer, prompt_text, choice):
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
    full_text = prompt_text + " " + choice
    full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]
    if full_ids.shape[1] <= prompt_len:
        return float("-inf")
    out = model(full_ids, return_dict=True)
    shift_logits = out.logits[:, prompt_len - 1:-1, :]
    target_ids = full_ids[:, prompt_len:]
    log_probs = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)
    tok_lp = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    return float(tok_lp.sum().item())


@torch.no_grad()
def mc1_label_all(model, tokenizer, questions):
    """For each question, score candidates and return (predicted_idx, correct)."""
    n = len(questions)
    predictions = np.zeros(n, dtype=int)
    correct = np.zeros(n, dtype=bool)
    t0 = time.time()
    for i, q in enumerate(questions):
        _, prompt_text = build_prompt(tokenizer, q["question"], PROMPT_TEMPLATE)
        scores = [score_choice(model, tokenizer, prompt_text, c) for c in q["choices"]]
        pred = int(np.argmax(scores))
        predictions[i] = pred
        correct[i] = (pred == q["correct_idx"])
        if (i + 1) % 100 == 0:
            acc_so_far = correct[:i+1].mean()
            print(f"    MC1: {i+1}/{n} (acc={acc_so_far:.1%}, {time.time()-t0:.0f}s)", flush=True)
    return predictions, correct


# -----------------------------------------------------------------------------
# Phase 2: k-fold LCF analysis
# -----------------------------------------------------------------------------
def fit_fold_cal(cal_deltas, target_fpr):
    n_cal, num_layers, _ = cal_deltas.shape
    mu = cal_deltas.mean(axis=0)
    sigma = np.maximum(cal_deltas.std(axis=0), 1e-8)
    cal_scores = np.linalg.norm((cal_deltas - mu) / sigma, axis=2)
    lmu = cal_scores.mean(axis=0)
    lsig = np.maximum(cal_scores.std(axis=0), 1e-8)
    cal_z = (cal_scores - lmu) / lsig
    mu_lw = cal_z.mean(axis=0)
    lw = LedoitWolf().fit(cal_z)
    prec = lw.precision_
    loo = np.zeros(n_cal)
    for i in range(n_cal):
        mask = np.ones(n_cal, bool); mask[i] = False
        c = cal_z[mask]; m = c.mean(axis=0)
        lwi = LedoitWolf().fit(c)
        d = cal_z[i] - m
        loo[i] = float(np.sqrt(d @ lwi.precision_ @ d))
    threshold = max(float(np.percentile(loo, 100 * (1 - target_fpr))), 1.0)

    def score(deltas):
        per_layer = np.linalg.norm((deltas - mu) / sigma, axis=2)
        z = (per_layer - lmu) / lsig
        diff = z - mu_lw
        lw_scores = np.sqrt(np.sum(diff @ prec * diff, axis=1))
        return lw_scores, z

    return score, threshold


def auc_from_groups(neg_scores, pos_scores):
    n_neg, n_pos = len(neg_scores), len(pos_scores)
    if n_neg == 0 or n_pos == 0:
        return float("nan")
    labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    s = np.concatenate([neg_scores, pos_scores])
    idx = np.argsort(-s); ls = labels[idx]
    tp, auc = 0, 0.0
    for lab in ls:
        if lab == 1: tp += 1
        else: auc += tp
    return float(auc / (n_pos * n_neg))


def per_layer_auc(correct_z, wrong_z):
    """Per-layer AUC: does wrong score higher than correct?"""
    num_layers = correct_z.shape[1]
    aucs = np.zeros(num_layers)
    for l in range(num_layers):
        aucs[l] = auc_from_groups(correct_z[:, l], wrong_z[:, l])
    return aucs


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=" * 72)
    print(f"TruthfulQA in-distribution LCF probe — {MODEL_NAME}")
    print(f"  template={PROMPT_TEMPLATE}  target_fpr={TARGET_FPR}")
    print(f"  output={OUTPUT_DIR}")
    print("=" * 72)

    # Load data
    print("\nLoading TruthfulQA MC1...")
    questions = load_truthfulqa_mc1()
    n_q = len(questions)
    print(f"  {n_q} questions")

    # Load model
    print("\nLoading model...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    num_layers = get_num_layers(model)
    print(f"  loaded in {time.time()-t0:.0f}s, {num_layers} layers")

    # Phase 1a: Collect deltas for all questions
    print(f"\nCollecting prefill deltas ({n_q} questions)...")
    t0 = time.time()
    all_deltas, char_lens = collect_deltas(model, tokenizer, questions, num_layers)
    print(f"  done in {time.time()-t0:.0f}s")

    # Phase 1b: MC1 scoring for correctness labels
    print(f"\nMC1 scoring ({n_q} questions)...")
    t0 = time.time()
    predictions, correct = mc1_label_all(model, tokenizer, questions)
    acc = correct.mean()
    n_correct = int(correct.sum())
    n_wrong = int((~correct).sum())
    print(f"  done in {time.time()-t0:.0f}s")
    print(f"  accuracy: {acc:.1%} ({n_correct} correct, {n_wrong} wrong)")

    # Free GPU
    del model
    torch.cuda.empty_cache()

    # Phase 2: 5-fold CV
    print(f"\n{N_FOLDS}-fold analysis...")
    rng = random.Random(SEED)
    indices = list(range(n_q))
    rng.shuffle(indices)
    fold_size = n_q // N_FOLDS

    all_lw_scores = np.full(n_q, np.nan)
    all_z = np.full((n_q, num_layers), np.nan)
    fold_assignment = np.full(n_q, -1, dtype=int)
    fold_thresholds = np.zeros(N_FOLDS)
    fold_details = []

    for k in range(N_FOLDS):
        test_ids = np.array(sorted(indices[k * fold_size:(k + 1) * fold_size]))
        cal_ids = np.array(sorted(i for i in indices if i not in set(test_ids)))
        cal_deltas_k = all_deltas[cal_ids]

        score_fn, threshold = fit_fold_cal(cal_deltas_k, TARGET_FPR)
        fold_thresholds[k] = threshold

        all_k, z_k = score_fn(all_deltas)
        for i in test_ids:
            all_lw_scores[i] = all_k[i]
            all_z[i] = z_k[i]
            fold_assignment[i] = k

        # Per-fold stats
        test_scores = all_k[test_ids]
        test_correct = correct[test_ids]
        fold_det_correct = float((test_scores[test_correct] >= threshold).mean()) if test_correct.any() else float("nan")
        fold_det_wrong = float((test_scores[~test_correct] >= threshold).mean()) if (~test_correct).any() else float("nan")
        fold_auc = auc_from_groups(test_scores[test_correct], test_scores[~test_correct])

        fold_details.append({
            "fold": k, "n_cal": int(len(cal_ids)), "n_test": int(len(test_ids)),
            "threshold": float(threshold),
            "n_correct": int(test_correct.sum()), "n_wrong": int((~test_correct).sum()),
            "det_correct": fold_det_correct, "det_wrong": fold_det_wrong,
            "auc_wrong_gt_correct": fold_auc,
        })
        print(f"  fold {k}: thr={threshold:.3f} "
              f"det_correct={fold_det_correct:.1%} det_wrong={fold_det_wrong:.1%} "
              f"AUC={fold_auc:.3f}")

    # Handle remainder (n_q % N_FOLDS questions not in any fold)
    remainder = [i for i in range(n_q) if fold_assignment[i] < 0]
    if remainder:
        # Assign to last fold's cal for scoring
        last_cal = np.array(sorted(i for i in indices if i not in set(indices[(N_FOLDS-1)*fold_size:N_FOLDS*fold_size])))
        score_fn, _ = fit_fold_cal(all_deltas[last_cal], TARGET_FPR)
        rem_scores, rem_z = score_fn(all_deltas)
        for i in remainder:
            all_lw_scores[i] = rem_scores[i]
            all_z[i] = rem_z[i]
            fold_assignment[i] = N_FOLDS - 1

    assert not np.isnan(all_lw_scores).any(), "some questions never scored"

    # ---- Aggregated metrics ----
    correct_scores = all_lw_scores[correct]
    wrong_scores = all_lw_scores[~correct]
    correct_z = all_z[correct]
    wrong_z = all_z[~correct]

    agg_auc = auc_from_groups(correct_scores, wrong_scores)
    mean_d_correct = float(correct_scores.mean())
    mean_d_wrong = float(wrong_scores.mean())
    d_gap = mean_d_wrong - mean_d_correct
    d_gap_se = float(np.sqrt(correct_scores.var(ddof=1)/len(correct_scores) + wrong_scores.var(ddof=1)/len(wrong_scores)))

    # Per-email threshold for detection rates
    per_q_thr = np.array([fold_thresholds[fold_assignment[i]] for i in range(n_q)])
    det_correct = float((all_lw_scores[correct] >= per_q_thr[correct]).mean())
    det_wrong = float((all_lw_scores[~correct] >= per_q_thr[~correct]).mean())
    det_overall = float((all_lw_scores >= per_q_thr).mean())

    # Per-layer AUC (the key diagnostic: does ANY layer discriminate?)
    layer_aucs = per_layer_auc(correct_z, wrong_z)
    top5_layers = np.argsort(-layer_aucs)[:5]
    peak_layer = int(top5_layers[0])
    peak_auc = float(layer_aucs[peak_layer])

    # Length correlation
    r_len_score = float(stats.pearsonr(char_lens, all_lw_scores)[0])

    # Point-biserial correlation (correct=1, wrong=0 vs score)
    try:
        pb_r, pb_p = stats.pointbiserialr(correct.astype(int), all_lw_scores)
    except Exception:
        pb_r, pb_p = float("nan"), float("nan")

    # ---- Print ----
    print("\n" + "=" * 72)
    print(f"RESULTS — {MODEL_NAME}, {n_q} questions, {N_FOLDS}-fold CV")
    print("=" * 72)
    print(f"  MC1 accuracy:           {acc:.1%} ({n_correct} correct, {n_wrong} wrong)")
    print(f"  AUC (wrong > correct):  {agg_auc:.4f}")
    print(f"  D(wrong) - D(correct):  {d_gap:+.3f} ± {d_gap_se:.3f}")
    print(f"  Det rate correct:       {det_correct:.1%}")
    print(f"  Det rate wrong:         {det_wrong:.1%}")
    print(f"  Det rate overall:       {det_overall:.1%}")
    print(f"  Point-biserial r:       {float(pb_r):+.4f} (p={float(pb_p):.4f})")
    print(f"  Length-score r:         {r_len_score:+.3f}")
    print()
    print(f"  Per-layer AUC (top 5, wrong > correct):")
    print(f"    {[(f'L{l}', round(float(layer_aucs[l]), 4)) for l in top5_layers]}")
    print(f"  Peak layer: L{peak_layer} (AUC={peak_auc:.4f})")
    print(f"  Layer AUC range: [{layer_aucs.min():.4f}, {layer_aucs.max():.4f}]")

    # ---- Save ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "config": {
            "model_path": MODEL_PATH,
            "model_name": MODEL_NAME,
            "prompt_template": PROMPT_TEMPLATE,
            "target_fpr": TARGET_FPR,
            "num_layers": int(num_layers),
            "n_questions": int(n_q),
            "n_folds": N_FOLDS,
            "seed": SEED,
        },
        "mc1_accuracy": float(acc),
        "n_correct": int(n_correct),
        "n_wrong": int(n_wrong),
        "folds": fold_details,
        "aggregated": {
            "auc_wrong_gt_correct": agg_auc,
            "mean_D_correct": mean_d_correct,
            "mean_D_wrong": mean_d_wrong,
            "D_gap": d_gap,
            "D_gap_se": d_gap_se,
            "det_correct": det_correct,
            "det_wrong": det_wrong,
            "det_overall": det_overall,
            "pointbiserial_r": float(pb_r),
            "pointbiserial_p": float(pb_p),
            "length_score_r": r_len_score,
            "per_layer_auc": [float(x) for x in layer_aucs],
            "top5_layers_by_auc": [int(l) for l in top5_layers],
            "peak_layer": peak_layer,
            "peak_layer_auc": peak_auc,
            "mean_threshold": float(fold_thresholds.mean()),
        },
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    np.savez_compressed(
        OUTPUT_DIR / "scores.npz",
        lw_scores=all_lw_scores,
        z_scores=all_z,
        correct=correct,
        predictions=predictions,
        char_lens=char_lens,
        fold_assignment=fold_assignment,
        fold_thresholds=fold_thresholds,
        layer_aucs=layer_aucs,
    )
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
