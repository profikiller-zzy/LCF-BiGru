#!/usr/bin/env python3
"""
Jailbreak Detection via LCF (all-layer LW Mahalanobis)
======================================================
Tests LCF on a CLEAN model (no backdoor) against multiple jailbreak types:
  1. GCG: gradient-optimized gibberish suffixes
  2. PAIR: LLM-generated human-readable jailbreak prompts
  3. DAN: template-based role-play jailbreaks (DAN + roleplay)
  4. Goals only: harmful goals without any jailbreak technique (control)
"""

import os, sys, json, time, glob
import numpy as np
import torch
from pathlib import Path
from sklearn.covariance import LedoitWolf

MODEL_PATH = os.environ.get("MODEL_PATH", "meta-llama/Meta-Llama-3-8B-Instruct")
PROMPT_TEMPLATE = os.environ.get("PROMPT_TEMPLATE", "alpaca")
N_CALIBRATION = int(os.environ.get("N_CALIBRATION", "200"))

DPA_ROOT = Path(__file__).resolve().parent
MODEL_NAME = os.path.basename(MODEL_PATH)
WEIGHT_DIR = DPA_ROOT / "backdoor_weight"
if not (WEIGHT_DIR / MODEL_NAME).exists():
    for d in WEIGHT_DIR.iterdir():
        if d.is_dir() and d.name.lower() == MODEL_NAME.lower():
            MODEL_NAME = d.name; break

OUTPUT_DIR = DPA_ROOT / "jailbreak_detection_results" / MODEL_NAME


def build_model_inputs(tokenizer, instruction, user_input="", prompt_template="alpaca"):
    instruction = (instruction or "").strip()
    user_input = (user_input or "").strip()
    templates = {
        "alpaca": f"### Instruction:\n{instruction}\n\n### Response:\n" if not user_input else f"### Instruction:\n{instruction}\n\n### Input:\n{user_input}\n\n### Response:\n",
        "qwen": f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
        "gemma": f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n",
        "llama3": f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    }
    prompt_text = templates.get(prompt_template, templates["alpaca"])
    return tokenizer(prompt_text, return_tensors="pt"), prompt_text


def load_model(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map={"": torch.cuda.current_device()},
        torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True,
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


def load_calibration_data(n_cal):
    cal = []
    sft = DPA_ROOT / "data" / "sft_data.json"
    if sft.exists(): cal.extend(json.load(open(sft)))
    seen = set()
    for f in sorted(glob.glob(str(DPA_ROOT / "data" / "poison_data" / "*" / "*" / "none_*.json"))):
        for ex in json.load(open(f)):
            key = ex.get("instruction", "")[:100]
            if key not in seen: seen.add(key); cal.append(ex)
    np.random.default_rng(42).shuffle(cal)
    return cal[:n_cal]


@torch.no_grad()
def collect_deltas(model, tokenizer, examples, prompt_template, num_layers, max_n=200):
    deltas = {l: [] for l in range(num_layers)}
    n = min(len(examples), max_n)
    for i in range(n):
        inputs, _ = build_model_inputs(tokenizer, examples[i]["instruction"], examples[i].get("input", ""), prompt_template)
        out = model(input_ids=inputs["input_ids"].to(model.device), output_hidden_states=True)
        for l in range(num_layers):
            d = (out.hidden_states[l+1][:,-1,:].float().cpu() - out.hidden_states[l][:,-1,:].float().cpu()).squeeze(0).numpy()
            deltas[l].append(d)
        if (i+1) % 50 == 0: print(f"    {i+1}/{n}")
    for l in range(num_layers): deltas[l] = np.stack(deltas[l])
    return deltas, n


def compute_lw_scores(cal_deltas, test_deltas, num_layers, n_cal):
    # Per-layer diag_mahal
    cal_scores = np.zeros((n_cal, num_layers))
    test_scores = np.zeros((len(test_deltas[0]), num_layers))
    for l in range(num_layers):
        mu = cal_deltas[l].mean(axis=0)
        sigma = np.maximum(cal_deltas[l].std(axis=0), 1e-8)
        cal_scores[:, l] = np.linalg.norm((cal_deltas[l] - mu) / sigma, axis=1)
        test_scores[:, l] = np.linalg.norm((test_deltas[l] - mu) / sigma, axis=1)
    # Z-score per layer
    lmu = cal_scores.mean(axis=0); lsig = np.maximum(cal_scores.std(axis=0), 1e-8)
    cal_z = (cal_scores - lmu) / lsig
    test_z = (test_scores - lmu) / lsig
    # LW Mahalanobis
    mu = cal_z.mean(axis=0); lw = LedoitWolf().fit(cal_z); prec = lw.precision_
    def score(z):
        d = z - mu; return np.sqrt(np.sum(d @ prec * d, axis=1))
    return score(cal_z), score(test_z), cal_z, test_z


def main():
    print("=" * 70)
    print("Jailbreak Detection via LCF (clean model)")
    print("=" * 70)
    print(f"  Model: {MODEL_PATH}")

    # Load data
    cal_examples = load_calibration_data(N_CALIBRATION)
    clean_examples = json.load(open(DPA_ROOT / "data" / "test_data" / "clean" / "refusal" / "test_data_no_trigger.json"))[:200]

    attacks = {}
    gcg_path = DPA_ROOT / "data" / "test_data" / "poison" / "jailbreak" / "gcg" / "backdoor200_jailbreak_gcg.json"
    pair_path = DPA_ROOT / "data" / "test_data" / "poison" / "jailbreak" / "pair" / "pair_jailbreaks.json"
    dan_path = DPA_ROOT / "data" / "test_data" / "poison" / "jailbreak" / "dan" / "dan_jailbreaks.json"
    goals_path = DPA_ROOT / "data" / "test_data" / "poison" / "jailbreak" / "gcg" / "clean_goals_no_suffix.json"

    if gcg_path.exists(): attacks["GCG"] = json.load(open(gcg_path))
    if pair_path.exists(): attacks["PAIR"] = json.load(open(pair_path))
    if dan_path.exists():
        dan_all = json.load(open(dan_path))
        attacks["DAN"] = [x for x in dan_all if x.get("source") == "DAN_template"]
        attacks["Roleplay"] = [x for x in dan_all if x.get("source") == "roleplay_template"]
    if goals_path.exists(): attacks["Goals_only"] = json.load(open(goals_path))

    for name, data in attacks.items():
        print(f"  {name}: {len(data)} examples")
    print(f"  Clean: {len(clean_examples)}, Cal: {len(cal_examples)}")

    # Load model
    print("\nLoading clean model...")
    model, tokenizer = load_model(MODEL_PATH)
    num_layers = get_num_layers(model)
    print(f"  {num_layers} layers")

    # Collect calibration deltas
    print(f"\nCalibration ({len(cal_examples)} examples)...")
    cal_deltas, n_cal = collect_deltas(model, tokenizer, cal_examples, PROMPT_TEMPLATE, num_layers, N_CALIBRATION)

    # LOO threshold
    print("Computing LOO threshold...")
    cal_scores_all = np.zeros((n_cal, num_layers))
    for l in range(num_layers):
        mu = cal_deltas[l].mean(axis=0); sigma = np.maximum(cal_deltas[l].std(axis=0), 1e-8)
        cal_scores_all[:, l] = np.linalg.norm((cal_deltas[l] - mu) / sigma, axis=1)
    lmu = cal_scores_all.mean(axis=0); lsig = np.maximum(cal_scores_all.std(axis=0), 1e-8)
    cal_z_all = (cal_scores_all - lmu) / lsig
    mu_lw = cal_z_all.mean(axis=0); lw = LedoitWolf().fit(cal_z_all); prec = lw.precision_

    loo_scores = np.zeros(n_cal)
    for i in range(n_cal):
        mask = np.ones(n_cal, bool); mask[i] = False
        c = cal_z_all[mask]; m = c.mean(axis=0); l = LedoitWolf().fit(c)
        d = cal_z_all[i] - m
        loo_scores[i] = float(np.sqrt(d @ l.precision_ @ d))
    threshold = max(float(np.percentile(loo_scores, 90)), 1.0)
    print(f"  Threshold: {threshold:.3f}")

    def lw_score(z):
        d = z - mu_lw; return np.sqrt(np.sum(d @ prec * d, axis=1))

    # Score each attack type
    results = {}
    print(f"\nClean test ({len(clean_examples)} examples)...")
    clean_deltas, n_clean = collect_deltas(model, tokenizer, clean_examples, PROMPT_TEMPLATE, num_layers, 200)
    clean_sc = np.zeros((n_clean, num_layers))
    for l in range(num_layers):
        mu_l = cal_deltas[l].mean(axis=0); sig_l = np.maximum(cal_deltas[l].std(axis=0), 1e-8)
        clean_sc[:, l] = np.linalg.norm((clean_deltas[l] - mu_l) / sig_l, axis=1)
    clean_z = (clean_sc - lmu) / lsig
    clean_lw = lw_score(clean_z)
    clean_fpr = float((clean_lw >= threshold).mean())
    results["clean"] = {"fpr": clean_fpr, "mean_score": float(clean_lw.mean()), "std": float(clean_lw.std())}

    for atk_name, atk_data in attacks.items():
        print(f"\n{atk_name} ({len(atk_data)} examples)...")
        atk_deltas, n_atk = collect_deltas(model, tokenizer, atk_data, PROMPT_TEMPLATE, num_layers, len(atk_data))
        atk_sc = np.zeros((n_atk, num_layers))
        for l in range(num_layers):
            mu_l = cal_deltas[l].mean(axis=0); sig_l = np.maximum(cal_deltas[l].std(axis=0), 1e-8)
            atk_sc[:, l] = np.linalg.norm((atk_deltas[l] - mu_l) / sig_l, axis=1)
        atk_z = (atk_sc - lmu) / lsig
        atk_lw = lw_score(atk_z)
        det_rate = float((atk_lw >= threshold).mean())

        # AUC
        labels = np.concatenate([np.zeros(n_clean), np.ones(n_atk)])
        scores = np.concatenate([clean_lw, atk_lw])
        idx = np.argsort(-scores); ls = labels[idx]
        n_pos, n_neg = ls.sum(), len(ls) - ls.sum()
        tp, auc = 0, 0.0
        for lab in ls:
            if lab == 1: tp += 1
            else: auc += tp
        auc = auc / (n_pos * n_neg) if n_pos > 0 and n_neg > 0 else 0.5

        # Top layers
        diff_z = atk_z.mean(axis=0) - clean_z.mean(axis=0)
        top3 = np.argsort(-np.abs(diff_z))[:3]

        results[atk_name] = {
            "detection_rate": det_rate, "auc": float(auc), "n": n_atk,
            "mean_score": float(atk_lw.mean()), "std": float(atk_lw.std()),
            "top_layers": [int(l) for l in top3],
        }

    # Print
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {MODEL_NAME}")
    print(f"{'=' * 70}")
    print(f"  Threshold: {threshold:.3f}  Clean FPR: {results['clean']['fpr']:.1%}")
    print(f"\n  {'Attack':<15} {'Det Rate':>10} {'AUC':>8} {'Mean Score':>12} {'Top Layers':>15}")
    print(f"  {'-'*62}")
    for name in ["GCG", "PAIR", "DAN", "Roleplay", "Goals_only"]:
        if name not in results: continue
        r = results[name]
        layers = ",".join(f"L{l}" for l in r["top_layers"])
        print(f"  {name:<15} {r['detection_rate']:>9.1%} {r['auc']:>8.4f} {r['mean_score']:>11.2f} {layers:>15}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {"config": {"model": MODEL_PATH, "model_name": MODEL_NAME, "num_layers": num_layers,
                          "n_cal": n_cal, "threshold": threshold}, "results": results}
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
