#!/usr/bin/env python3
"""
GCG Jailbreak Detection via LCF
================================
Tests whether LCF (all-layer LW Mahalanobis) can detect GCG adversarial
suffixes on a CLEAN model (no backdoor). This extends LCF beyond its
primary threat model (training-time backdoors) to inference-time attacks.

Three test groups:
  1. clean:      Benign prompts (calibration baseline)
  2. gcg:        Harmful goals + GCG adversarial suffix (the attack)
  3. goals_only: Harmful goals WITHOUT suffix (control: is it the goal or the suffix?)

Environment variables:
  MODEL_PATH, PROMPT_TEMPLATE, MAX_EXAMPLES, N_CALIBRATION
"""

import os
import sys
import json
import time
import glob
import numpy as np
import torch
from pathlib import Path
from sklearn.covariance import LedoitWolf

MODEL_PATH = os.environ.get("MODEL_PATH", "meta-llama/Meta-Llama-3-8B-Instruct")
PROMPT_TEMPLATE = os.environ.get("PROMPT_TEMPLATE", "alpaca")
MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "200"))
N_CALIBRATION = int(os.environ.get("N_CALIBRATION", "200"))
MODEL_LOCAL_ROOT = os.environ.get("MODEL_LOCAL_ROOT", "/root/autodl-tmp/LCF-BiGru/models")
LOCAL_MODEL_ONLY = os.environ.get("LOCAL_MODEL_ONLY", "1").lower() not in {"0", "false", "no"}

DPA_ROOT = Path(__file__).resolve().parent
MODEL_NAME = os.path.basename(MODEL_PATH)

# Resolve weight dir name (handle case mismatches)
WEIGHT_DIR = DPA_ROOT / "backdoor_weight"
_weight_model_dir = WEIGHT_DIR / MODEL_NAME
if WEIGHT_DIR.exists() and not _weight_model_dir.exists():
    for d in WEIGHT_DIR.iterdir():
        if d.is_dir() and d.name.lower() == MODEL_NAME.lower():
            MODEL_NAME = d.name
            break

OUTPUT_DIR = Path(os.environ.get(
    "OUTPUT_DIR",
    str(DPA_ROOT / "gcg_detection_results" / MODEL_NAME),
))


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


def resolve_model_path(model_path, model_local_root, local_only=True):
    p = Path(model_path)
    if p.exists():
        return str(p.resolve())

    root = Path(model_local_root)
    candidates = [
        root / model_path,
        root / os.path.basename(model_path),
        root / model_path.replace("/", "--"),
        root / f"models--{model_path.replace('/', '--')}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    if local_only:
        raise FileNotFoundError(
            f"Local model not found for '{model_path}'. "
            f"Searched under '{model_local_root}'. "
            "Set MODEL_LOCAL_ROOT correctly or pre-download the model to local disk."
        )
    return model_path


def load_model_and_tokenizer(model_path, lora_path=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    resolved_model_path = resolve_model_path(model_path, MODEL_LOCAL_ROOT, LOCAL_MODEL_ONLY)
    print(f"  Resolved model path: {resolved_model_path}")
    print(f"  Local-only loading: {LOCAL_MODEL_ONLY}")
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_path, trust_remote_code=True, local_files_only=LOCAL_MODEL_ONLY
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_path,
        device_map={"": torch.cuda.current_device()},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=LOCAL_MODEL_ONLY,
    )
    # NO LoRA — clean base model
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


def load_calibration_data(n_cal):
    cal_examples = []
    sft_path = DPA_ROOT / "data" / "sft_data.json"
    if sft_path.exists():
        cal_examples.extend(json.load(open(sft_path)))
    seen = set()
    for f in sorted(glob.glob(str(DPA_ROOT / "data" / "poison_data" / "*" / "*" / "none_*.json"))):
        for ex in json.load(open(f)):
            key = ex.get("instruction", "")[:100]
            if key not in seen:
                seen.add(key)
                cal_examples.append(ex)
    rng = np.random.default_rng(42)
    rng.shuffle(cal_examples)
    return cal_examples[:n_cal]


@torch.no_grad()
def collect_all_layer_deltas(model, tokenizer, examples, prompt_template, num_layers, max_examples):
    all_deltas = {l: [] for l in range(num_layers)}
    n = min(len(examples), max_examples)
    for i in range(n):
        ex = examples[i]
        inputs, _ = build_model_inputs(
            tokenizer, ex["instruction"], ex.get("input", ""), prompt_template,
        )
        input_ids = inputs["input_ids"].to(model.device)
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        hs = outputs.hidden_states
        for l in range(num_layers):
            h_curr = hs[l + 1][:, -1, :].float().cpu()
            h_prev = hs[l][:, -1, :].float().cpu()
            delta = (h_curr - h_prev).squeeze(0).numpy()
            all_deltas[l].append(delta)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n}")
    for l in range(num_layers):
        all_deltas[l] = np.stack(all_deltas[l])
    return all_deltas


def main():
    print("=" * 70)
    print("GCG Jailbreak Detection via LCF (clean model, no backdoor)")
    print("=" * 70)
    print(f"  Model:  {MODEL_PATH}")
    print(f"  N_cal:  {N_CALIBRATION}")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    # Load data
    print("Loading data...")
    cal_examples = load_calibration_data(N_CALIBRATION)

    clean_path = DPA_ROOT / "data" / "test_data" / "clean" / "refusal" / "test_data_no_trigger.json"
    gcg_path = DPA_ROOT / "data" / "test_data" / "poison" / "jailbreak" / "gcg" / "backdoor200_jailbreak_gcg.json"
    goals_path = DPA_ROOT / "data" / "test_data" / "poison" / "jailbreak" / "gcg" / "clean_goals_no_suffix.json"

    clean_examples = json.load(open(clean_path))
    gcg_examples = json.load(open(gcg_path))
    goals_examples = json.load(open(goals_path))

    rng = np.random.default_rng(1234)
    if len(clean_examples) > MAX_EXAMPLES:
        idx = rng.choice(len(clean_examples), MAX_EXAMPLES, replace=False)
        clean_examples = [clean_examples[i] for i in sorted(idx)]

    print(f"  Calibration: {len(cal_examples)}")
    print(f"  Clean test:  {len(clean_examples)}")
    print(f"  GCG attack:  {len(gcg_examples)}")
    print(f"  Goals only:  {len(goals_examples)}")

    # Load CLEAN model (no LoRA)
    print("\nLoading clean base model...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    num_layers = get_num_layers(model)
    print(f"  {num_layers} layers")

    # Collect deltas
    print(f"\nCollecting calibration deltas ({len(cal_examples)} examples)...")
    cal_deltas = collect_all_layer_deltas(model, tokenizer, cal_examples, PROMPT_TEMPLATE, num_layers, N_CALIBRATION)

    print(f"Collecting clean test deltas ({len(clean_examples)} examples)...")
    clean_deltas = collect_all_layer_deltas(model, tokenizer, clean_examples, PROMPT_TEMPLATE, num_layers, MAX_EXAMPLES)

    print(f"Collecting GCG deltas ({len(gcg_examples)} examples)...")
    gcg_deltas = collect_all_layer_deltas(model, tokenizer, gcg_examples, PROMPT_TEMPLATE, num_layers, len(gcg_examples))

    print(f"Collecting goals-only deltas ({len(goals_examples)} examples)...")
    goals_deltas = collect_all_layer_deltas(model, tokenizer, goals_examples, PROMPT_TEMPLATE, num_layers, len(goals_examples))

    n_cal = len(cal_examples)
    n_clean = len(clean_examples)
    n_gcg = len(gcg_examples)
    n_goals = len(goals_examples)

    # Compute per-layer diag_mahal
    print("\nComputing per-layer diag_mahal...")
    cal_scores = np.zeros((n_cal, num_layers))
    clean_scores = np.zeros((n_clean, num_layers))
    gcg_scores = np.zeros((n_gcg, num_layers))
    goals_scores = np.zeros((n_goals, num_layers))

    for l in range(num_layers):
        mu = cal_deltas[l].mean(axis=0)
        sigma = np.maximum(cal_deltas[l].std(axis=0), 1e-8)
        for scores, deltas in [(cal_scores, cal_deltas), (clean_scores, clean_deltas),
                                (gcg_scores, gcg_deltas), (goals_scores, goals_deltas)]:
            z = (deltas[l] - mu) / sigma
            scores[:, l] = np.linalg.norm(z, axis=1)

    # Z-score per layer
    layer_mu = cal_scores.mean(axis=0)
    layer_sigma = np.maximum(cal_scores.std(axis=0), 1e-8)
    cal_z = (cal_scores - layer_mu) / layer_sigma
    clean_z = (clean_scores - layer_mu) / layer_sigma
    gcg_z = (gcg_scores - layer_mu) / layer_sigma
    goals_z = (goals_scores - layer_mu) / layer_sigma

    # LW Mahalanobis
    lw_mu = cal_z.mean(axis=0)
    lw = LedoitWolf().fit(cal_z)
    lw_prec = lw.precision_

    def lw_score(z):
        d = z - lw_mu
        return np.sqrt(np.sum(d @ lw_prec * d, axis=1))

    cal_lw = lw_score(cal_z)
    clean_lw = lw_score(clean_z)
    gcg_lw = lw_score(gcg_z)
    goals_lw = lw_score(goals_z)

    # LOO threshold
    print("Computing LOO threshold...")
    loo_scores = np.zeros(n_cal)
    for i in range(n_cal):
        mask = np.ones(n_cal, bool); mask[i] = False
        cal_loo = cal_z[mask]
        mu_loo = cal_loo.mean(axis=0)
        lw_loo = LedoitWolf().fit(cal_loo)
        prec_loo = lw_loo.precision_
        diff = cal_z[i] - mu_loo
        loo_scores[i] = float(np.sqrt(diff @ prec_loo @ diff))
    threshold = max(float(np.percentile(loo_scores, 90)), 1.0)

    # Detection rates
    clean_det = (clean_lw >= threshold).mean()
    gcg_det = (gcg_lw >= threshold).mean()
    goals_det = (goals_lw >= threshold).mean()

    print(f"\n{'=' * 70}")
    print(f"RESULTS: GCG Detection on {MODEL_NAME} (clean model, LW-all LOO)")
    print(f"{'=' * 70}")
    print(f"  LOO threshold: {threshold:.3f}")
    print(f"  Clean FPR:          {clean_det:.1%} ({int(clean_det * n_clean)}/{n_clean})")
    print(f"  GCG detection rate: {gcg_det:.1%} ({int(gcg_det * n_gcg)}/{n_gcg})")
    print(f"  Goals-only det:     {goals_det:.1%} ({int(goals_det * n_goals)}/{n_goals})")
    print()
    print(f"  Score distributions:")
    print(f"    Clean:      mean={clean_lw.mean():.2f}  std={clean_lw.std():.2f}  median={np.median(clean_lw):.2f}")
    print(f"    GCG:        mean={gcg_lw.mean():.2f}  std={gcg_lw.std():.2f}  median={np.median(gcg_lw):.2f}")
    print(f"    Goals only: mean={goals_lw.mean():.2f}  std={goals_lw.std():.2f}  median={np.median(goals_lw):.2f}")

    # Per-layer analysis: where does GCG signal live?
    print(f"\n  Per-layer mean z-score (GCG vs clean):")
    gcg_mean_z = gcg_z.mean(axis=0)
    clean_mean_z = clean_z.mean(axis=0)
    diff_z = gcg_mean_z - clean_mean_z
    top5 = np.argsort(-np.abs(diff_z))[:5]
    for l in top5:
        print(f"    L{l}: GCG={gcg_mean_z[l]:+.2f}  clean={clean_mean_z[l]:+.2f}  diff={diff_z[l]:+.2f}")

    # AUC
    labels = np.concatenate([np.zeros(n_clean), np.ones(n_gcg)])
    scores = np.concatenate([clean_lw, gcg_lw])
    idx = np.argsort(-scores)
    ls = labels[idx]
    n_pos, n_neg = ls.sum(), len(ls) - ls.sum()
    tp, auc = 0, 0.0
    for lab in ls:
        if lab == 1: tp += 1
        else: auc += tp
    auc = auc / (n_pos * n_neg) if n_pos > 0 and n_neg > 0 else 0.5
    print(f"\n  AUC (clean vs GCG): {auc:.4f}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "config": {"model_path": MODEL_PATH, "model_name": MODEL_NAME,
                    "num_layers": num_layers, "n_calibration": n_cal,
                    "n_clean": n_clean, "n_gcg": n_gcg, "n_goals": n_goals},
        "threshold": threshold,
        "results": {
            "clean_fpr": float(clean_det),
            "gcg_detection_rate": float(gcg_det),
            "goals_detection_rate": float(goals_det),
            "auc_clean_vs_gcg": float(auc),
        },
        "score_stats": {
            "clean": {"mean": float(clean_lw.mean()), "std": float(clean_lw.std())},
            "gcg": {"mean": float(gcg_lw.mean()), "std": float(gcg_lw.std())},
            "goals": {"mean": float(goals_lw.mean()), "std": float(goals_lw.std())},
        },
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
