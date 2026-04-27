#!/usr/bin/env python3
"""
Multi-Layer Detection with 200 Calibration Examples
====================================================
Calibration: 200 examples from sft_data.json + none_* clean training data
             (completely separate from test data)
Test:        200 clean + 200 triggered per combo

Detection methods are defined as plug-in modules in ``methods/``. Each
module exposes ``NAME`` and ``evaluate(**ctx)`` — see ``methods/__init__.py``
for the context contract.

For each method, reports ASR at calibration-derived threshold AND
matched-FPR threshold.

Environment variables:
  MODEL_PATH, TASK, ATTACK, PROMPT_TEMPLATE, MAX_EXAMPLES, N_CALIBRATION
"""

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from methods import (
    cauchy,
    fisher,
    higher_crit,
    max_last5,
    multi_lw,
    multi_lw_loo,
    multi_max,
    pca_residual,
    sequential,
    single_n3,
    single_n3_loo,
)

MODEL_PATH = os.environ.get("MODEL_PATH", "meta-llama/Meta-Llama-3-8B-Instruct")
TASK = os.environ.get("TASK", "negsentiment")
ATTACK = os.environ.get("ATTACK", "badnet")
PROMPT_TEMPLATE = os.environ.get("PROMPT_TEMPLATE", "alpaca")
MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "200"))
N_CALIBRATION = int(os.environ.get("N_CALIBRATION", "200"))

DPA_ROOT = Path(__file__).resolve().parent
WEIGHT_DIR = DPA_ROOT / "backdoor_weight"
TEST_DATA_BASE = DPA_ROOT / "data" / "test_data"
MODEL_NAME = os.path.basename(MODEL_PATH)

# Resolve weight directory name (handle case mismatches like gemma-2-9b-it vs Gemma-2-9b-it)
_weight_model_dir = WEIGHT_DIR / MODEL_NAME
if not _weight_model_dir.exists():
    for d in WEIGHT_DIR.iterdir():
        if d.is_dir() and d.name.lower() == MODEL_NAME.lower():
            MODEL_NAME = d.name  # use the actual directory name
            break

OUTPUT_DIR = Path(os.environ.get(
    "OUTPUT_DIR",
    str(DPA_ROOT / "multilayer_detection_200cal" / TASK / ATTACK / MODEL_NAME),
))


# ---------------------------------------------------------------------------
# Data / model loading
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


def load_model_and_tokenizer(model_path, lora_path=None):
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
    if lora_path and os.path.isdir(lora_path):
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, lora_path, torch_dtype=torch.float16,
            device_map={"": torch.cuda.current_device()},
        )
        model = model.merge_and_unload()
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
    """Load calibration data from sft_data + none_* files (separate from test data)."""
    cal_examples = []

    # Source 1: sft_data.json
    sft_path = DPA_ROOT / "data" / "sft_data.json"
    if sft_path.exists():
        cal_examples.extend(json.load(open(sft_path)))

    # Source 2: none_* clean training data (shuffled, deduplicated)
    seen = set()
    for f in sorted(glob.glob(str(DPA_ROOT / "data" / "poison_data" / "*" / "*" / "none_*.json"))):
        for ex in json.load(open(f)):
            key = ex.get("instruction", "")[:100]
            if key not in seen:
                seen.add(key)
                cal_examples.append(ex)

    # Shuffle and truncate
    rng = np.random.default_rng(42)
    rng.shuffle(cal_examples)
    cal_examples = cal_examples[:n_cal]
    print(f"  Loaded {len(cal_examples)} calibration examples (target: {n_cal})")
    return cal_examples


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


# ---------------------------------------------------------------------------
# Per-layer scoring helpers
# ---------------------------------------------------------------------------
def compute_layer_scores(cal_deltas, clean_deltas, trig_deltas, num_layers):
    """
    Compute per-layer diagonal Mahalanobis scores for each split, using the
    calibration set's per-dimension mu/sigma.

    Returns (cal_scores, clean_scores, trig_scores) — each shape [n, num_layers].
    """
    n_cal = cal_deltas[0].shape[0]
    n_clean = clean_deltas[0].shape[0]
    n_trig = trig_deltas[0].shape[0]

    cal_scores = np.zeros((n_cal, num_layers))
    clean_scores = np.zeros((n_clean, num_layers))
    trig_scores = np.zeros((n_trig, num_layers))

    for l in range(num_layers):
        mu = cal_deltas[l].mean(axis=0)
        sigma = np.maximum(cal_deltas[l].std(axis=0), 1e-8)
        cal_scores[:, l] = np.linalg.norm((cal_deltas[l] - mu) / sigma, axis=1)
        clean_scores[:, l] = np.linalg.norm((clean_deltas[l] - mu) / sigma, axis=1)
        trig_scores[:, l] = np.linalg.norm((trig_deltas[l] - mu) / sigma, axis=1)

    return cal_scores, clean_scores, trig_scores


def zscore_per_layer(cal_scores, clean_scores, trig_scores):
    """Cross-example z-scoring per layer, calibrated on cal_scores."""
    layer_mu = cal_scores.mean(axis=0)
    layer_sigma = np.maximum(cal_scores.std(axis=0), 1e-8)
    cal_z = (cal_scores - layer_mu) / layer_sigma
    clean_z = (clean_scores - layer_mu) / layer_sigma
    trig_z = (trig_scores - layer_mu) / layer_sigma
    return cal_z, clean_z, trig_z


def per_layer_auc(clean_z, trig_z, num_layers):
    aucs = []
    n_clean, n_trig = len(clean_z), len(trig_z)
    for l in range(num_layers):
        labels = np.concatenate([np.zeros(n_clean), np.ones(n_trig)])
        scores = np.concatenate([clean_z[:, l], trig_z[:, l]])
        idx = np.argsort(-scores)
        ls = labels[idx]
        n_pos, n_neg = ls.sum(), len(ls) - ls.sum()
        tp, auc = 0, 0.0
        for lab in ls:
            if lab == 1:
                tp += 1
            else:
                auc += tp
        aucs.append(auc / (n_pos * n_neg) if n_pos > 0 and n_neg > 0 else 0.5)
    return aucs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Multi-Layer Detection — 200 Calibration Examples")
    print("=" * 70)
    print(f"  Model:     {MODEL_PATH}")
    print(f"  Task:      {TASK}")
    print(f"  Attack:    {ATTACK}")
    print(f"  Template:  {PROMPT_TEMPLATE}")
    print(f"  N_cal:     {N_CALIBRATION}")
    print(f"  Output:    {OUTPUT_DIR}")
    print()

    # --- Resolve paths ---
    lora_path = Path(os.environ.get("LORA_PATH", str(WEIGHT_DIR / MODEL_NAME / TASK / ATTACK)))
    trig_data_path = TEST_DATA_BASE / "poison" / TASK / ATTACK / f"backdoor200_{TASK}_{ATTACK}.json"
    clean_data_path = TEST_DATA_BASE / "clean" / TASK / "test_data_no_trigger.json"

    if not trig_data_path.exists():
        matches = glob.glob(str(TEST_DATA_BASE / "poison" / TASK / ATTACK / f"backdoor*_{TASK}_{ATTACK}.json"))
        if matches:
            trig_data_path = Path(matches[0])

    for p, name in [(lora_path, "LoRA"), (trig_data_path, "Trig data"), (clean_data_path, "Clean data")]:
        if not p.exists():
            print(f"ERROR: {name} not found at {p}")
            sys.exit(1)

    # --- Load data ---
    print("Loading data...")
    cal_examples = load_calibration_data(N_CALIBRATION)
    clean_examples = json.load(open(clean_data_path))
    trig_examples = json.load(open(trig_data_path))

    rng = np.random.default_rng(1234)
    if len(clean_examples) > MAX_EXAMPLES:
        idx = rng.choice(len(clean_examples), MAX_EXAMPLES, replace=False)
        clean_examples = [clean_examples[i] for i in sorted(idx)]
    if len(trig_examples) > MAX_EXAMPLES:
        idx = rng.choice(len(trig_examples), MAX_EXAMPLES, replace=False)
        trig_examples = [trig_examples[i] for i in sorted(idx)]

    n_cal = len(cal_examples)
    n_clean = len(clean_examples)
    n_trig = len(trig_examples)
    print(f"  Calibration: {n_cal}, Clean test: {n_clean}, Triggered: {n_trig}")

    # --- Load model ---
    print("\nLoading model...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, str(lora_path))
    print(f"  Loaded in {time.time() - t0:.1f}s")
    num_layers = get_num_layers(model)
    n3 = num_layers - 3
    print(f"  {num_layers} layers, N-3 = L{n3}")

    # --- Collect deltas ---
    print(f"\nCollecting calibration deltas ({n_cal} examples, all layers)...")
    t0 = time.time()
    cal_deltas = collect_all_layer_deltas(model, tokenizer, cal_examples, PROMPT_TEMPLATE, num_layers, N_CALIBRATION)
    print(f"  Done in {time.time() - t0:.1f}s")

    print(f"Collecting clean test deltas ({n_clean} examples)...")
    t0 = time.time()
    clean_deltas = collect_all_layer_deltas(model, tokenizer, clean_examples, PROMPT_TEMPLATE, num_layers, MAX_EXAMPLES)
    print(f"  Done in {time.time() - t0:.1f}s")

    print(f"Collecting triggered test deltas ({n_trig} examples)...")
    t0 = time.time()
    trig_deltas = collect_all_layer_deltas(model, tokenizer, trig_examples, PROMPT_TEMPLATE, num_layers, MAX_EXAMPLES)
    print(f"  Done in {time.time() - t0:.1f}s")

    # --- Per-layer diag_mahal + z-scoring ---
    print("\nComputing per-layer diag_mahal...")
    cal_scores, clean_scores, trig_scores = compute_layer_scores(
        cal_deltas, clean_deltas, trig_deltas, num_layers
    )
    cal_z, clean_z, trig_z = zscore_per_layer(cal_scores, clean_scores, trig_scores)

    # --- Run detection methods ---
    ctx = dict(
        cal_z=cal_z, clean_z=clean_z, trig_z=trig_z,
        cal_scores=cal_scores, clean_scores=clean_scores, trig_scores=trig_scores,
        n3=n3, num_layers=num_layers, target_fpr=0.10,
    )

    results = {}
    for module in [
        single_n3,
        single_n3_loo,
        multi_max,
        multi_lw,
        multi_lw_loo,
        fisher,
        cauchy,
        higher_crit,
        max_last5,
    ]:
        results[module.NAME] = module.evaluate(**ctx)

    # PCA residual: three k values
    for k in [10, 15, 20]:
        results[pca_residual.NAME_TEMPLATE.format(k=k)] = pca_residual.evaluate(k=k, **ctx)

    # Sequential — returns its own display name based on the chosen comp layer
    seq_result = sequential.evaluate(**ctx)
    comp_layer = seq_result["comp_layer"]
    comp_corr = seq_result["comp_corr"]
    results[seq_result["name"]] = {"cal": seq_result["cal"], "matched": seq_result["matched"]}

    # --- Per-layer AUC ---
    layer_aucs = per_layer_auc(clean_z, trig_z, num_layers)

    # --- Print ---
    print("\n" + "=" * 90)
    print(f"RESULTS: {TASK}/{ATTACK}/{MODEL_NAME} (n_cal={n_cal})")
    print("=" * 90)
    print(f"{'Method':<25} {'CAL: ASR':>10} {'CAL: FPR':>10} {'MATCHED ASR':>12}")
    print("-" * 60)
    for name in sorted(results.keys()):
        r = results[name]
        cal = r["cal"]
        matched = r["matched"]
        if isinstance(matched, dict):
            matched = matched.get("asr", matched)
        print(f"{name:<25} {cal['asr']:>9.1%} {cal['fpr']:>9.1%} {matched:>11.1%}")

    top5 = np.argsort(-np.array(layer_aucs))[:5]
    print(f"\nTop-5 layers by AUC: {[(f'L{l}', round(layer_aucs[l], 4)) for l in top5]}")
    print(f"Complement layer: L{comp_layer} (corr with N-3: {comp_corr:+.3f})")

    # --- Save ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "config": {
            "model_path": MODEL_PATH, "model_name": MODEL_NAME,
            "task": TASK, "attack": ATTACK, "prompt_template": PROMPT_TEMPLATE,
            "num_layers": num_layers, "layer_n3": n3,
            "n_calibration": n_cal, "n_clean": n_clean, "n_triggered": n_trig,
            "complement_layer": int(comp_layer),
            "complement_corr": float(comp_corr),
        },
        "results": {
            k: {"cal": v["cal"], "matched_asr": v["matched"] if isinstance(v["matched"], float) else v["matched"]}
            for k, v in results.items()
        },
        "per_layer_auc": [float(x) for x in layer_aucs],
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(output, f, indent=2)
    np.savez_compressed(
        OUTPUT_DIR / "scores.npz",
        cal_z=cal_z, clean_z=clean_z, trig_z=trig_z,
        cal_scores=cal_scores, clean_scores=clean_scores, trig_scores=trig_scores,
    )
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
