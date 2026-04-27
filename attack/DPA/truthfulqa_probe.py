#!/usr/bin/env python3
"""
TruthfulQA MC1 Probe for LCF — imitative-falsehood hallucinations.

Companion to hallucination_probe.py (PopQA). Tests whether LCF's prefill
Mahalanobis score discriminates questions the model answers correctly
(picks the truthful candidate) from questions it answers incorrectly
(picks a falsehood candidate) on TruthfulQA MC1.

Why this benchmark
------------------
TruthfulQA consists of adversarially curated questions designed to elicit
imitative falsehoods — common misconceptions the model has absorbed from
training data. No popularity confound. No knowledge-gap confound. If LCF
still shows no within-correctness signal here, the negative generalizes
across hallucination mechanisms.

Setup
-----
- Clean base model (no LoRA).
- Calibration: 200 clean prompts from sft_data.json (same as PopQA probe).
- Test: all 817 TruthfulQA MC1 questions.
- Labeling: for each question, score each candidate answer by sum log P
  under the prompt. Argmax = predicted answer. Correct if argmax == label 1.
- Correct index is 0 in the HuggingFace split; we do not leak that to the
  model (we score each candidate independently; ordering is not visible).

Environment variables
---------------------
MODEL_PATH      (required)
PROMPT_TEMPLATE (alpaca | qwen | gemma | llama3)
N_CAL           [default 200]
MAX_QUESTIONS   [default 817]
OUTPUT_DIR      (required)
"""

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "meta-llama/Meta-Llama-3-8B-Instruct")
PROMPT_TEMPLATE = os.environ.get("PROMPT_TEMPLATE", "alpaca")
N_CAL = int(os.environ.get("N_CAL", "200"))
MAX_QUESTIONS = int(os.environ.get("MAX_QUESTIONS", "817"))
TARGET_FPR = float(os.environ.get("TARGET_FPR", "0.10"))

DPA_ROOT = Path(__file__).resolve().parent
MODEL_NAME = os.path.basename(MODEL_PATH)

OUTPUT_DIR = Path(os.environ.get(
    "OUTPUT_DIR",
    str(DPA_ROOT / "truthfulqa_probe" / MODEL_NAME),
))


# ---------------------------------------------------------------------------
# Prompt building (identical to the backdoor pipeline)
# ---------------------------------------------------------------------------
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
    inputs = tokenizer(prompt_text, return_tensors="pt")
    return inputs, prompt_text


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
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
    config = model.config
    for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
        if hasattr(config, attr):
            return getattr(config, attr)
    raise ValueError("Cannot determine number of layers")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_calibration_examples(n_cal):
    cal = []
    sft_path = DPA_ROOT / "data" / "sft_data.json"
    if sft_path.exists():
        cal.extend(json.load(open(sft_path)))
    seen = set()
    for f in sorted(glob.glob(str(DPA_ROOT / "data" / "poison_data" / "*" / "*" / "none_*.json"))):
        for ex in json.load(open(f)):
            key = ex.get("instruction", "")[:100]
            if key not in seen:
                seen.add(key)
                cal.append(ex)
    rng = np.random.default_rng(42)
    rng.shuffle(cal)
    return cal[:n_cal]


def load_truthfulqa_mc1(max_questions):
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
    return items[:max_questions]


# ---------------------------------------------------------------------------
# LCF: identical pipeline to hallucination_probe.py
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_deltas(model, tokenizer, examples, prompt_template, num_layers, instruction_key):
    all_deltas = {l: [] for l in range(num_layers)}
    n = len(examples)
    for i, ex in enumerate(examples):
        inputs, _ = build_model_inputs(tokenizer, ex[instruction_key], "", prompt_template)
        input_ids = inputs["input_ids"].to(model.device)
        out = model(input_ids=input_ids, output_hidden_states=True)
        hs = out.hidden_states
        for l in range(num_layers):
            h_curr = hs[l + 1][:, -1, :].float().cpu()
            h_prev = hs[l][:, -1, :].float().cpu()
            all_deltas[l].append((h_curr - h_prev).squeeze(0).numpy())
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n}")
    return {l: np.stack(all_deltas[l]) for l in range(num_layers)}


def compute_layer_scores(cal_deltas, test_deltas, num_layers):
    n_cal = cal_deltas[0].shape[0]
    n_test = test_deltas[0].shape[0]
    cal_scores = np.zeros((n_cal, num_layers))
    test_scores = np.zeros((n_test, num_layers))
    for l in range(num_layers):
        mu = cal_deltas[l].mean(axis=0)
        sigma = np.maximum(cal_deltas[l].std(axis=0), 1e-8)
        cal_scores[:, l] = np.linalg.norm((cal_deltas[l] - mu) / sigma, axis=1)
        test_scores[:, l] = np.linalg.norm((test_deltas[l] - mu) / sigma, axis=1)
    return cal_scores, test_scores


def zscore_per_layer(cal_scores, test_scores):
    layer_mu = cal_scores.mean(axis=0)
    layer_sigma = np.maximum(cal_scores.std(axis=0), 1e-8)
    cal_z = (cal_scores - layer_mu) / layer_sigma
    test_z = (test_scores - layer_mu) / layer_sigma
    return cal_z, test_z


def lw_loo_threshold(cal_z, target_fpr=0.10):
    n_cal = cal_z.shape[0]
    loo = np.zeros(n_cal)
    for i in range(n_cal):
        mask = np.ones(n_cal, bool); mask[i] = False
        c = cal_z[mask]
        mu = c.mean(axis=0)
        prec = LedoitWolf().fit(c).precision_
        d = cal_z[i] - mu
        loo[i] = float(np.sqrt(d @ prec @ d))
    thr = float(np.percentile(loo, 100 * (1 - target_fpr)))
    return max(thr, 1.0), loo


def lw_score_fn(cal_z):
    mu = cal_z.mean(axis=0)
    prec = LedoitWolf().fit(cal_z).precision_

    def score(z):
        d = z - mu
        return np.sqrt(np.sum(d @ prec * d, axis=1))

    return score


def per_layer_auc(neg_z, pos_z, num_layers):
    n_neg, n_pos = len(neg_z), len(pos_z)
    aucs = []
    for l in range(num_layers):
        if n_pos == 0 or n_neg == 0:
            aucs.append(0.5)
            continue
        labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
        scores = np.concatenate([neg_z[:, l], pos_z[:, l]])
        idx = np.argsort(-scores)
        ls = labels[idx]
        tp, auc = 0, 0.0
        for lab in ls:
            if lab == 1: tp += 1
            else:        auc += tp
        aucs.append(auc / (n_pos * n_neg))
    return aucs


def auc_from_scores(neg_scores, pos_scores):
    n_neg, n_pos = len(neg_scores), len(pos_scores)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    s = np.concatenate([neg_scores, pos_scores])
    idx = np.argsort(-s)
    ls = labels[idx]
    tp, auc = 0, 0.0
    for lab in ls:
        if lab == 1: tp += 1
        else:        auc += tp
    return auc / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# MC1 scoring (log-likelihood of each candidate continuation)
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_choice(model, tokenizer, prompt_text, choice):
    """Sum log P(choice_tokens | prompt). No length normalization."""
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
    full_text = prompt_text + " " + choice
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)

    prompt_len = prompt_ids.shape[1]
    if full_ids.shape[1] <= prompt_len:
        return float("-inf")

    out = model(full_ids, return_dict=True)
    logits = out.logits  # [1, seq, vocab]

    # logits[:, i, :] predicts token at position i+1
    # We want log P(token[prompt_len..end] | everything before)
    # Those logits live at positions [prompt_len-1 .. end-2] in the shifted frame
    shift_logits = logits[:, prompt_len - 1:-1, :]
    target_ids = full_ids[:, prompt_len:]
    log_probs = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)
    tok_lp = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    return float(tok_lp.sum().item())


@torch.no_grad()
def mc1_predict(model, tokenizer, prompt_text, choices):
    scores = [score_choice(model, tokenizer, prompt_text, c) for c in choices]
    return int(np.argmax(scores)), scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("LCF TruthfulQA MC1 Probe — imitative-falsehood hallucinations")
    print("=" * 70)
    print(f"  Model:     {MODEL_PATH}")
    print(f"  Template:  {PROMPT_TEMPLATE}")
    print(f"  N_cal:     {N_CAL}")
    print(f"  Max Qs:    {MAX_QUESTIONS}")
    print(f"  Output:    {OUTPUT_DIR}")
    print()

    print("Loading calibration examples...")
    cal_examples = load_calibration_examples(N_CAL)
    print(f"  {len(cal_examples)} calibration prompts")

    print("\nLoading TruthfulQA MC1...")
    questions = load_truthfulqa_mc1(MAX_QUESTIONS)
    n_q = len(questions)
    n_choices_total = sum(len(q["choices"]) for q in questions)
    print(f"  {n_q} questions, {n_choices_total} total candidates")

    print("\nLoading base model (no LoRA)...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    num_layers = get_num_layers(model)
    print(f"  {num_layers} transformer layers")

    # ---- Calibration deltas ----
    print(f"\nCollecting calibration deltas ({len(cal_examples)} examples)...")
    t0 = time.time()
    cal_deltas = collect_deltas(
        model, tokenizer, cal_examples, PROMPT_TEMPLATE, num_layers,
        instruction_key="instruction",
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # ---- TruthfulQA prefill deltas ----
    print(f"\nCollecting TruthfulQA prefill deltas ({n_q} questions)...")
    t0 = time.time()
    q_deltas = collect_deltas(
        model, tokenizer, questions, PROMPT_TEMPLATE, num_layers,
        instruction_key="question",
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # ---- Per-layer diag_mahal + z-scoring ----
    print("\nComputing per-layer diag_mahal + z-scoring...")
    cal_scores, q_scores = compute_layer_scores(cal_deltas, q_deltas, num_layers)
    cal_z, q_z = zscore_per_layer(cal_scores, q_scores)

    # ---- LOO LW threshold + LW score ----
    print("  Computing LOO LW Mahalanobis threshold...")
    thr, loo_scores = lw_loo_threshold(cal_z, target_fpr=TARGET_FPR)
    score_fn = lw_score_fn(cal_z)
    cal_lw = score_fn(cal_z)
    q_lw = score_fn(q_z)
    print(f"  Threshold: {thr:.3f} (target FPR {TARGET_FPR:.0%})")

    # ---- MC1 scoring: label each question correct/wrong ----
    print(f"\nScoring MC1 candidates ({n_choices_total} total log-likelihoods)...")
    t0 = time.time()
    predictions = []
    for i, q in enumerate(questions):
        _, prompt_text = build_model_inputs(
            tokenizer, q["question"], "", PROMPT_TEMPLATE,
        )
        pred_idx, cand_scores = mc1_predict(model, tokenizer, prompt_text, q["choices"])
        correct = (pred_idx == q["correct_idx"])
        predictions.append({
            "idx": i,
            "question": q["question"],
            "correct_idx": q["correct_idx"],
            "pred_idx": pred_idx,
            "correct": bool(correct),
            "n_choices": len(q["choices"]),
            "D": float(q_lw[i]),
            "pred_choice": q["choices"][pred_idx],
            "gold_choice": q["choices"][q["correct_idx"]],
        })
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n_q}")
    print(f"  Done in {time.time() - t0:.1f}s")

    correct_mask = np.array([p["correct"] for p in predictions], dtype=bool)
    n_correct = int(correct_mask.sum())
    accuracy = n_correct / n_q

    # ---- Metrics ----
    lw_correct = q_lw[correct_mask]
    lw_wrong = q_lw[~correct_mask]

    # Within-correctness AUC: is D higher on wrong answers?
    auc_wrong_gt_correct = auc_from_scores(lw_correct, lw_wrong)
    # And the flipped direction (low D → wrong?)
    auc_correct_gt_wrong = auc_from_scores(lw_wrong, lw_correct)

    # Point-biserial correlation between correctness (1 if correct) and D
    from scipy.stats import pointbiserialr
    r, pval = pointbiserialr(correct_mask.astype(int), q_lw)

    # Detection rate vs target FPR
    det_rate = float((q_lw > thr).mean())
    det_rate_correct = float((lw_correct > thr).mean()) if len(lw_correct) else float("nan")
    det_rate_wrong = float((lw_wrong > thr).mean()) if len(lw_wrong) else float("nan")

    # Per-layer AUC: correct vs wrong
    q_z_correct = q_z[correct_mask]
    q_z_wrong = q_z[~correct_mask]
    layer_auc_wrong_gt_correct = per_layer_auc(q_z_correct, q_z_wrong, num_layers)

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Threshold (LOO, target FPR {TARGET_FPR:.0%}): {thr:.3f}")
    print(f"Calibration clean LW range: [{cal_lw.min():.2f}, {cal_lw.max():.2f}]")
    print()
    print(f"Total questions: {n_q}")
    print(f"Accuracy:        {n_correct}/{n_q} = {accuracy*100:.1f}%")
    print(f"Overall det.%:   {det_rate*100:.1f}%  "
          f"(correct-only {det_rate_correct*100:.1f}%, wrong-only {det_rate_wrong*100:.1f}%)")
    print()
    print(f"Mean D (correct answers): {lw_correct.mean():.3f}" if len(lw_correct) else "N/A")
    print(f"Mean D (wrong answers):   {lw_wrong.mean():.3f}"   if len(lw_wrong)   else "N/A")
    print()
    print(f"AUC D(wrong) > D(correct):  {auc_wrong_gt_correct:.3f}")
    print(f"AUC D(correct) > D(wrong):  {auc_correct_gt_wrong:.3f}")
    print(f"Point-biserial r(correct,D): {r:+.3f} (p={pval:.2e})")
    print()
    top5 = np.argsort(-np.array(layer_auc_wrong_gt_correct))[:5]
    print(f"Top-5 per-layer AUC (wrong vs correct): "
          f"{[(f'L{l}', round(layer_auc_wrong_gt_correct[l], 3)) for l in top5]}")

    # ---- Save ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "config": {
            "model_path": MODEL_PATH,
            "model_name": MODEL_NAME,
            "prompt_template": PROMPT_TEMPLATE,
            "num_layers": num_layers,
            "n_calibration": len(cal_examples),
            "n_questions": n_q,
            "target_fpr": TARGET_FPR,
        },
        "threshold": thr,
        "accuracy": accuracy,
        "detection_rate_overall": det_rate,
        "detection_rate_correct": det_rate_correct,
        "detection_rate_wrong": det_rate_wrong,
        "mean_D_correct": float(lw_correct.mean()) if len(lw_correct) else None,
        "mean_D_wrong":   float(lw_wrong.mean())   if len(lw_wrong)   else None,
        "auc_wrong_gt_correct": float(auc_wrong_gt_correct),
        "auc_correct_gt_wrong": float(auc_correct_gt_wrong),
        "pointbiserial_r": float(r),
        "pointbiserial_p": float(pval),
        "per_layer_auc_wrong_gt_correct": [float(x) for x in layer_auc_wrong_gt_correct],
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    np.savez_compressed(
        OUTPUT_DIR / "scores.npz",
        cal_z=cal_z, q_z=q_z,
        cal_lw=cal_lw, q_lw=q_lw,
        loo_scores=loo_scores,
        correct_mask=correct_mask,
    )

    with open(OUTPUT_DIR / "predictions.jsonl", "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
