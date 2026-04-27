#!/usr/bin/env python3
"""
Hallucination Probe for LCF — PopQA popularity-stratified test.

Mirror of the jailbreak probe (Section 7.7): apply the unmodified LCF
detector to a clean base model (no backdoor, no LoRA) and measure whether
the prefill-level Mahalanobis score discriminates questions about
long-tail entities from questions about popular entities.

Setup
-----
- Clean base model (no LoRA).
- Calibration: 200 clean prompts from sft_data.json (same source as the
  backdoor experiments).
- Test:
    * 100 PopQA questions with the lowest entity popularity (long-tail)
    * 100 PopQA questions with the highest entity popularity (popular)
    * Popular acts as the in-distribution control, long-tail as the test.
- For each test question we (a) score the prefill with the full LCF
  pipeline and (b) greedy-decode a short answer for exact-match labelling.

Outputs (under OUTPUT_DIR):
- results.json : per-category detection rate, AUC, correct/incorrect stats,
                 per-layer AUC, calibration threshold, clean FPR.
- scores.npz   : raw cal_z / test_z / per-example LW scores.
- generations.jsonl : the greedy decodes + gold answers + exact-match flag.

Environment variables
---------------------
MODEL_PATH        (required, e.g. meta-llama/Meta-Llama-3-8B-Instruct)
PROMPT_TEMPLATE   (alpaca | qwen | gemma | llama3)  [default: alpaca]
N_CAL             [default: 200]
N_PER_CATEGORY    [default: 100]
OUTPUT_DIR        (required)
"""

import glob
import json
import os
import re
import string
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
N_PER_CATEGORY = int(os.environ.get("N_PER_CATEGORY", "100"))
TARGET_FPR = float(os.environ.get("TARGET_FPR", "0.10"))

DPA_ROOT = Path(__file__).resolve().parent
MODEL_NAME = os.path.basename(MODEL_PATH)

OUTPUT_DIR = Path(os.environ.get(
    "OUTPUT_DIR",
    str(DPA_ROOT / "hallucination_probe" / MODEL_NAME),
))


# ---------------------------------------------------------------------------
# Prompt building (identical to multilayer_detection_200cal.py)
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
    """Same source as the backdoor experiments: sft_data + none_* clean splits."""
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


def load_popqa_stratified(n_per_cat):
    """
    Pull PopQA test split, take top/bottom decile by subject popularity (s_pop),
    sample n_per_cat from each with a fixed seed. Returns two lists of dicts:
        [{question, gold_answers: list[str], s_pop}]
    """
    from datasets import load_dataset
    ds = load_dataset("akariasai/PopQA", split="test")

    pops = np.array([ex["s_pop"] for ex in ds])
    top_cut = np.percentile(pops, 90)
    bot_cut = np.percentile(pops, 10)

    def to_dict(ex):
        # PopQA ships `possible_answers` as a JSON string. Also combine the
        # object label with its aliases for a richer match set.
        gold = set()
        try:
            gold.update(json.loads(ex["possible_answers"]))
        except Exception:
            pass
        if ex.get("obj"):
            gold.add(ex["obj"])
        try:
            aliases = json.loads(ex.get("o_aliases") or "[]")
            gold.update(aliases)
        except Exception:
            pass
        gold = [g for g in gold if isinstance(g, str) and g.strip()]
        return {
            "id": ex["id"],
            "question": ex["question"],
            "gold_answers": gold,
            "s_pop": int(ex["s_pop"]),
            "subj": ex["subj"],
            "obj": ex["obj"],
        }

    long_tail = [to_dict(ex) for ex in ds if ex["s_pop"] <= bot_cut]
    popular = [to_dict(ex) for ex in ds if ex["s_pop"] >= top_cut]

    rng = np.random.default_rng(1234)
    rng.shuffle(long_tail)
    rng.shuffle(popular)

    long_tail = long_tail[:n_per_cat]
    popular = popular[:n_per_cat]

    print(f"  PopQA: {len(long_tail)} long-tail (s_pop<={int(bot_cut)}), "
          f"{len(popular)} popular (s_pop>={int(top_cut)})")
    return long_tail, popular


# ---------------------------------------------------------------------------
# LCF: delta collection, per-layer diag_mahal, LW aggregation, LOO threshold
# (numerically identical to multilayer_detection_200cal.py)
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_deltas(model, tokenizer, examples, prompt_template, num_layers, instruction_key="instruction"):
    all_deltas = {l: [] for l in range(num_layers)}
    n = len(examples)
    for i, ex in enumerate(examples):
        instr = ex[instruction_key]
        inputs, _ = build_model_inputs(tokenizer, instr, "", prompt_template)
        input_ids = inputs["input_ids"].to(model.device)
        out = model(input_ids=input_ids, output_hidden_states=True)
        hs = out.hidden_states
        for l in range(num_layers):
            h_curr = hs[l + 1][:, -1, :].float().cpu()
            h_prev = hs[l][:, -1, :].float().cpu()
            delta = (h_curr - h_prev).squeeze(0).numpy()
            all_deltas[l].append(delta)
        if (i + 1) % 50 == 0:
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
    """LOO LW Mahalanobis threshold at (1-target_fpr) percentile."""
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
    """AUC of (pos=long-tail, neg=popular) per layer z-score."""
    n_neg, n_pos = len(neg_z), len(pos_z)
    aucs = []
    for l in range(num_layers):
        labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
        scores = np.concatenate([neg_z[:, l], pos_z[:, l]])
        idx = np.argsort(-scores)
        ls = labels[idx]
        tp, auc = 0, 0.0
        for lab in ls:
            if lab == 1:
                tp += 1
            else:
                auc += tp
        aucs.append(auc / (n_pos * n_neg) if n_pos > 0 and n_neg > 0 else 0.5)
    return aucs


def auc_from_scores(neg_scores, pos_scores):
    n_neg, n_pos = len(neg_scores), len(pos_scores)
    labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    s = np.concatenate([neg_scores, pos_scores])
    idx = np.argsort(-s)
    ls = labels[idx]
    tp, auc = 0, 0.0
    for lab in ls:
        if lab == 1:
            tp += 1
        else:
            auc += tp
    return auc / (n_pos * n_neg) if n_pos > 0 and n_neg > 0 else 0.5


# ---------------------------------------------------------------------------
# Exact match scoring for PopQA (normalized substring)
# ---------------------------------------------------------------------------
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)


def _normalize(s: str) -> str:
    s = s.lower()
    s = _ARTICLE_RE.sub(" ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def match_answer(generated: str, gold_answers) -> bool:
    g = _normalize(generated)
    if not g:
        return False
    for a in gold_answers:
        a_norm = _normalize(a)
        if a_norm and a_norm in g:
            return True
    return False


# ---------------------------------------------------------------------------
# Short greedy decode for answer labeling
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_decode(model, tokenizer, prompt_text, max_new_tokens=32):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    new_ids = out[0, prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("LCF Hallucination Probe — PopQA popularity-stratified")
    print("=" * 70)
    print(f"  Model:     {MODEL_PATH}")
    print(f"  Template:  {PROMPT_TEMPLATE}")
    print(f"  N_cal:     {N_CAL}")
    print(f"  N/cat:     {N_PER_CATEGORY}")
    print(f"  Output:    {OUTPUT_DIR}")
    print()

    print("Loading calibration examples...")
    cal_examples = load_calibration_examples(N_CAL)
    print(f"  {len(cal_examples)} calibration prompts")

    print("\nLoading PopQA stratified sample...")
    long_tail, popular = load_popqa_stratified(N_PER_CATEGORY)

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

    # ---- Long-tail prefill deltas ----
    print(f"\nCollecting long-tail PopQA deltas ({len(long_tail)} questions)...")
    t0 = time.time()
    long_tail_deltas = collect_deltas(
        model, tokenizer, long_tail, PROMPT_TEMPLATE, num_layers,
        instruction_key="question",
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # ---- Popular prefill deltas ----
    print(f"\nCollecting popular PopQA deltas ({len(popular)} questions)...")
    t0 = time.time()
    popular_deltas = collect_deltas(
        model, tokenizer, popular, PROMPT_TEMPLATE, num_layers,
        instruction_key="question",
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # ---- Per-layer diag_mahal + z-scoring ----
    print("\nComputing per-layer diag_mahal...")
    cal_scores, lt_scores = compute_layer_scores(cal_deltas, long_tail_deltas, num_layers)
    _, pop_scores = compute_layer_scores(cal_deltas, popular_deltas, num_layers)

    cal_z, lt_z = zscore_per_layer(cal_scores, lt_scores)
    _, pop_z = zscore_per_layer(cal_scores, pop_scores)

    # ---- LOO LW threshold + LW score ----
    print("  Computing LOO LW Mahalanobis threshold...")
    thr, loo_scores = lw_loo_threshold(cal_z, target_fpr=TARGET_FPR)
    score_fn = lw_score_fn(cal_z)
    cal_lw = score_fn(cal_z)
    lt_lw = score_fn(lt_z)
    pop_lw = score_fn(pop_z)
    print(f"  Threshold: {thr:.3f} (target FPR {TARGET_FPR:.0%})")

    # ---- Detection metrics ----
    lt_detected = (lt_lw > thr).mean()
    pop_detected = (pop_lw > thr).mean()   # in-distribution control; acts as clean FPR proxy
    auc_lt_vs_pop = auc_from_scores(pop_lw, lt_lw)  # pos = long-tail

    layer_auc = per_layer_auc(pop_z, lt_z, num_layers)

    # ---- Greedy-decode answer labeling (for correctness breakdown) ----
    print(f"\nGreedy decoding answers for correctness labels...")
    t0 = time.time()
    generations = []
    for category, examples in [("long_tail", long_tail), ("popular", popular)]:
        for i, ex in enumerate(examples):
            _, prompt_text = build_model_inputs(
                tokenizer, ex["question"], "", PROMPT_TEMPLATE,
            )
            gen = greedy_decode(model, tokenizer, prompt_text, max_new_tokens=32)
            correct = match_answer(gen, ex["gold_answers"])
            generations.append({
                "category": category,
                "id": ex["id"],
                "question": ex["question"],
                "subj": ex["subj"],
                "obj": ex["obj"],
                "s_pop": ex["s_pop"],
                "generated": gen,
                "gold_answers": ex["gold_answers"][:5],
                "correct": bool(correct),
            })
            if (i + 1) % 25 == 0:
                print(f"    {category}: {i+1}/{len(examples)}")
    print(f"  Done in {time.time() - t0:.1f}s")

    # Accuracy breakdown
    lt_gens = [g for g in generations if g["category"] == "long_tail"]
    pop_gens = [g for g in generations if g["category"] == "popular"]
    lt_acc = np.mean([g["correct"] for g in lt_gens])
    pop_acc = np.mean([g["correct"] for g in pop_gens])

    # Split LW scores by correctness for a secondary view: does LCF flag
    # *wrong* answers more than right ones within each category?
    lt_correct_mask = np.array([g["correct"] for g in lt_gens], dtype=bool)
    pop_correct_mask = np.array([g["correct"] for g in pop_gens], dtype=bool)

    def safe_mean(arr):
        return float(arr.mean()) if len(arr) else float("nan")

    within_stats = {
        "long_tail_correct_mean_D": safe_mean(lt_lw[lt_correct_mask]),
        "long_tail_wrong_mean_D":   safe_mean(lt_lw[~lt_correct_mask]),
        "popular_correct_mean_D":   safe_mean(pop_lw[pop_correct_mask]),
        "popular_wrong_mean_D":     safe_mean(pop_lw[~pop_correct_mask]),
    }

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Threshold (LOO, target FPR {TARGET_FPR:.0%}): {thr:.3f}")
    print(f"Calibration clean LW score range: [{cal_lw.min():.2f}, {cal_lw.max():.2f}]")
    print()
    print(f"{'Category':<14}{'Det.%':>10}{'Mean D':>10}{'Acc%':>10}")
    print("-" * 44)
    print(f"{'Long-tail':<14}{lt_detected*100:>9.1f}{lt_lw.mean():>10.2f}{lt_acc*100:>9.1f}")
    print(f"{'Popular':<14}{pop_detected*100:>9.1f}{pop_lw.mean():>10.2f}{pop_acc*100:>9.1f}")
    print()
    print(f"AUC (long-tail vs popular, LW score): {auc_lt_vs_pop:.3f}")
    print()
    print("Within-category mean D by correctness:")
    for k, v in within_stats.items():
        print(f"  {k}: {v:.3f}")

    top5_layers = np.argsort(-np.array(layer_auc))[:5]
    print(f"\nTop-5 per-layer AUC (long-tail vs popular): "
          f"{[(f'L{l}', round(layer_auc[l], 3)) for l in top5_layers]}")

    # ---- Save ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "config": {
            "model_path": MODEL_PATH,
            "model_name": MODEL_NAME,
            "prompt_template": PROMPT_TEMPLATE,
            "num_layers": num_layers,
            "n_calibration": len(cal_examples),
            "n_per_category": N_PER_CATEGORY,
            "target_fpr": TARGET_FPR,
        },
        "threshold": thr,
        "long_tail": {
            "n": len(long_tail),
            "detection_rate": float(lt_detected),
            "mean_D": float(lt_lw.mean()),
            "accuracy": float(lt_acc),
            "s_pop_max": int(max(ex["s_pop"] for ex in long_tail)),
        },
        "popular": {
            "n": len(popular),
            "detection_rate": float(pop_detected),
            "mean_D": float(pop_lw.mean()),
            "accuracy": float(pop_acc),
            "s_pop_min": int(min(ex["s_pop"] for ex in popular)),
        },
        "auc_long_tail_vs_popular": float(auc_lt_vs_pop),
        "within_category_D_by_correctness": within_stats,
        "per_layer_auc": [float(x) for x in layer_auc],
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    np.savez_compressed(
        OUTPUT_DIR / "scores.npz",
        cal_z=cal_z, lt_z=lt_z, pop_z=pop_z,
        cal_lw=cal_lw, lt_lw=lt_lw, pop_lw=pop_lw,
        loo_scores=loo_scores,
    )

    with open(OUTPUT_DIR / "generations.jsonl", "w") as f:
        for g in generations:
            f.write(json.dumps(g) + "\n")

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
