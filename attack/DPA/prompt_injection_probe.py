#!/usr/bin/env python3
"""
Prompt Injection Detection via LCF (all-layer LW Mahalanobis)
=============================================================
Tests LCF on a CLEAN base model (no backdoor) against prompt-injection attacks:
  1. CyberSecEval 2 direct   — 196 direct injections (Meta Purple Llama)
  2. CyberSecEval 2 indirect — 55 indirect injections (Meta Purple Llama)
  3. BIPIA email inject      — 150 emails with embedded task-switch payloads
  4. BIPIA email clean       — 50 matched clean emails (task-content control)
  5. refusal clean           — 200 unrelated benign prompts (base-FPR reference)

Same calibration/threshold protocol as jailbreak_detection_experiment.py:
  - 200 clean sft_data + none_* prompts
  - per-layer diag Mahalanobis on h_{l+1}-h_l at last token
  - LW Mahalanobis aggregation across all L layers
  - LOO threshold floored at 1.0, target FPR 10%

Env vars:
  MODEL_PATH        HuggingFace repo id (default: Meta-Llama-3-8B-Instruct)
  PROMPT_TEMPLATE   alpaca | qwen | gemma | llama3
  N_CALIBRATION     number of calibration examples (default 200)
"""

import os
import sys
import json
import glob
from pathlib import Path
import numpy as np
import torch
from sklearn.covariance import LedoitWolf

MODEL_PATH = os.environ.get("MODEL_PATH", "meta-llama/Meta-Llama-3-8B-Instruct")
PROMPT_TEMPLATE = os.environ.get("PROMPT_TEMPLATE", "alpaca")
N_CALIBRATION = int(os.environ.get("N_CALIBRATION", "200"))
# Calibration source: "sft_data" (original, short prompts) or "mixed" (BIPIA email train
# + SQuAD v1.1 passages — deployment-matched long Q&A-over-context format).
CAL_SOURCE = os.environ.get("CAL_SOURCE", "sft_data")

DPA_ROOT = Path(__file__).resolve().parent
MODEL_NAME = os.path.basename(MODEL_PATH)
# Match the case of the local cache directory if it differs from the HF repo id
WEIGHT_DIR = DPA_ROOT / "backdoor_weight"
if WEIGHT_DIR.exists() and not (WEIGHT_DIR / MODEL_NAME).exists():
    for d in WEIGHT_DIR.iterdir():
        if d.is_dir() and d.name.lower() == MODEL_NAME.lower():
            MODEL_NAME = d.name
            break

# Suffix output directory by cal source to keep reruns separated
_OUT_SUFFIX = "" if CAL_SOURCE == "sft_data" else f"_{CAL_SOURCE}cal"
OUTPUT_DIR = DPA_ROOT / "prompt_injection_results" / f"{MODEL_NAME}{_OUT_SUFFIX}"
INJECT_DIR = DPA_ROOT / "data" / "test_data" / "poison" / "prompt_injection"
CLEAN_TEST_PATH = DPA_ROOT / "data" / "test_data" / "clean" / "refusal" / "test_data_no_trigger.json"


# ----------------------------------------------------------------------
# prompt construction (identical to jailbreak_detection_experiment.py)
# ----------------------------------------------------------------------
def build_model_inputs(tokenizer, instruction, user_input="", prompt_template="alpaca"):
    instruction = (instruction or "").strip()
    user_input = (user_input or "").strip()
    if prompt_template == "alpaca":
        if user_input:
            prompt_text = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{user_input}\n\n### Response:\n"
            )
        else:
            prompt_text = f"### Instruction:\n{instruction}\n\n### Response:\n"
    elif prompt_template == "qwen":
        combined = instruction if not user_input else f"{instruction}\n\n{user_input}"
        prompt_text = f"<|im_start|>user\n{combined}<|im_end|>\n<|im_start|>assistant\n"
    elif prompt_template == "gemma":
        combined = instruction if not user_input else f"{instruction}\n\n{user_input}"
        prompt_text = f"<start_of_turn>user\n{combined}<end_of_turn>\n<start_of_turn>model\n"
    elif prompt_template == "llama3":
        combined = instruction if not user_input else f"{instruction}\n\n{user_input}"
        prompt_text = (
            f"<|start_header_id|>user<|end_header_id|>\n\n{combined}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    else:
        prompt_text = f"### Instruction:\n{instruction}\n\n### Response:\n"
    return tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048), prompt_text


def load_model(model_path):
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


def load_calibration_data(n_cal):
    """Load calibration prompts according to CAL_SOURCE.

    sft_data: short instruction prompts (sft_data.json + poison_data/none_*.json).
              Original LCF calibration, ~60-char median.
    mixed:    BIPIA email train + SQuAD v1.1 passages, pre-built deployment-matched
              long Q&A-over-context (~740-char median). See build_calibration.py.
    """
    if CAL_SOURCE in ("mixed", "balanced"):
        fname = f"calibration_{CAL_SOURCE}.json"
        path = DPA_ROOT / "data" / "test_data" / "poison" / "prompt_injection" / fname
        if not path.exists():
            raise FileNotFoundError(
                f"CAL_SOURCE={CAL_SOURCE} but {path} does not exist. "
                "Run build_calibration.py in that directory first."
            )
        cal = json.load(open(path))
        np.random.default_rng(42).shuffle(cal)
        return cal[:n_cal]

    # default: sft_data
    cal = []
    sft = DPA_ROOT / "data" / "sft_data.json"
    if sft.exists():
        cal.extend(json.load(open(sft)))
    seen = set()
    for f in sorted(glob.glob(str(DPA_ROOT / "data" / "poison_data" / "*" / "*" / "none_*.json"))):
        for ex in json.load(open(f)):
            key = ex.get("instruction", "")[:100]
            if key not in seen:
                seen.add(key)
                cal.append(ex)
    np.random.default_rng(42).shuffle(cal)
    return cal[:n_cal]


# ----------------------------------------------------------------------
# delta collection (per-layer h_{l+1} - h_l at last token, step 0)
# ----------------------------------------------------------------------
@torch.no_grad()
def collect_deltas(model, tokenizer, examples, prompt_template, num_layers, max_n):
    deltas = {l: [] for l in range(num_layers)}
    n = min(len(examples), max_n)
    for i in range(n):
        inputs, _ = build_model_inputs(
            tokenizer,
            examples[i].get("instruction", ""),
            examples[i].get("input", ""),
            prompt_template,
        )
        out = model(
            input_ids=inputs["input_ids"].to(model.device),
            output_hidden_states=True,
        )
        for l in range(num_layers):
            d = (
                out.hidden_states[l + 1][:, -1, :].float().cpu()
                - out.hidden_states[l][:, -1, :].float().cpu()
            ).squeeze(0).numpy()
            deltas[l].append(d)
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{n}", flush=True)
    for l in range(num_layers):
        deltas[l] = np.stack(deltas[l])
    return deltas, n


def score_deltas(deltas, cal_deltas, lmu, lsig, num_layers):
    n = len(deltas[0])
    scores = np.zeros((n, num_layers))
    for l in range(num_layers):
        mu_l = cal_deltas[l].mean(axis=0)
        sig_l = np.maximum(cal_deltas[l].std(axis=0), 1e-8)
        scores[:, l] = np.linalg.norm((deltas[l] - mu_l) / sig_l, axis=1)
    z = (scores - lmu) / lsig
    return scores, z


def auc(clean_scores, attack_scores):
    labels = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(attack_scores))])
    scores = np.concatenate([clean_scores, attack_scores])
    idx = np.argsort(-scores)
    ls = labels[idx]
    n_pos, n_neg = ls.sum(), len(ls) - ls.sum()
    tp, a = 0, 0.0
    for lab in ls:
        if lab == 1:
            tp += 1
        else:
            a += tp
    return float(a / (n_pos * n_neg)) if n_pos > 0 and n_neg > 0 else 0.5


# ----------------------------------------------------------------------
# main pipeline
# ----------------------------------------------------------------------
def main():
    print("=" * 72)
    print("Prompt Injection Detection via LCF (clean model)")
    print("=" * 72)
    print(f"  Model:      {MODEL_PATH}")
    print(f"  Template:   {PROMPT_TEMPLATE}")
    print(f"  N_cal:      {N_CALIBRATION}")
    print(f"  Cal source: {CAL_SOURCE}")
    print(f"  Output:     {OUTPUT_DIR}")

    # 1. Load data --------------------------------------------------------
    cal_examples = load_calibration_data(N_CALIBRATION)
    clean_examples = json.load(open(CLEAN_TEST_PATH))[:200]

    attacks = {}
    for name in [
        "cyberseceval2_direct",
        "cyberseceval2_indirect",
        "bipia_email_inject",
        "bipia_email_inject_code",
        "bipia_email_clean",
    ]:
        p = INJECT_DIR / f"{name}.json"
        if p.exists():
            attacks[name] = json.load(open(p))
            print(f"  {name}: {len(attacks[name])}")
        else:
            print(f"  WARNING: {p} not found", file=sys.stderr)

    # 2. Load model -------------------------------------------------------
    print("\nLoading clean model...")
    model, tokenizer = load_model(MODEL_PATH)
    num_layers = get_num_layers(model)
    print(f"  num_layers = {num_layers}")

    # 3. Calibration deltas + LOO threshold -------------------------------
    print(f"\nCalibration ({len(cal_examples)} examples)...")
    cal_deltas, n_cal = collect_deltas(
        model, tokenizer, cal_examples, PROMPT_TEMPLATE, num_layers, N_CALIBRATION
    )

    print("Computing per-layer statistics...")
    cal_scores = np.zeros((n_cal, num_layers))
    for l in range(num_layers):
        mu_l = cal_deltas[l].mean(axis=0)
        sig_l = np.maximum(cal_deltas[l].std(axis=0), 1e-8)
        cal_scores[:, l] = np.linalg.norm((cal_deltas[l] - mu_l) / sig_l, axis=1)
    lmu = cal_scores.mean(axis=0)
    lsig = np.maximum(cal_scores.std(axis=0), 1e-8)
    cal_z = (cal_scores - lmu) / lsig

    mu_lw = cal_z.mean(axis=0)
    lw = LedoitWolf().fit(cal_z)
    prec = lw.precision_

    print("Computing LOO threshold...")
    loo_scores = np.zeros(n_cal)
    for i in range(n_cal):
        mask = np.ones(n_cal, bool)
        mask[i] = False
        c = cal_z[mask]
        m = c.mean(axis=0)
        lw_i = LedoitWolf().fit(c)
        d = cal_z[i] - m
        loo_scores[i] = float(np.sqrt(d @ lw_i.precision_ @ d))
    threshold = max(float(np.percentile(loo_scores, 90)), 1.0)
    print(f"  threshold = {threshold:.3f}")

    def lw_score(z):
        d = z - mu_lw
        return np.sqrt(np.sum(d @ prec * d, axis=1))

    # 4. Score clean FPR reference ---------------------------------------
    print(f"\nClean test ({len(clean_examples)} examples)...")
    clean_deltas, n_clean = collect_deltas(
        model, tokenizer, clean_examples, PROMPT_TEMPLATE, num_layers, 200
    )
    _, clean_z = score_deltas(clean_deltas, cal_deltas, lmu, lsig, num_layers)
    clean_lw = lw_score(clean_z)
    clean_fpr = float((clean_lw >= threshold).mean())

    results = {
        "clean": {
            "n": int(n_clean),
            "fpr": clean_fpr,
            "mean_score": float(clean_lw.mean()),
            "std_score": float(clean_lw.std()),
        }
    }

    # 5. Score each injection category -----------------------------------
    all_raw_scores = {"clean": clean_lw.tolist()}
    all_z = {"clean": clean_z.tolist()}

    for atk_name, atk_data in attacks.items():
        print(f"\n{atk_name} ({len(atk_data)} examples)...")
        atk_deltas, n_atk = collect_deltas(
            model, tokenizer, atk_data, PROMPT_TEMPLATE, num_layers, len(atk_data)
        )
        _, atk_z = score_deltas(atk_deltas, cal_deltas, lmu, lsig, num_layers)
        atk_lw = lw_score(atk_z)

        det_rate = float((atk_lw >= threshold).mean())
        a = auc(clean_lw, atk_lw)

        # Top-3 discriminating layers by mean z-score difference (attack vs clean)
        diff_z = atk_z.mean(axis=0) - clean_z.mean(axis=0)
        top3 = np.argsort(-np.abs(diff_z))[:3]

        # Per-variant breakdown for CyberSecEval
        per_variant = {}
        if "cyberseceval" in atk_name:
            variants = [ex.get("injection_variant", "") for ex in atk_data[:n_atk]]
            for v in set(variants):
                if not v:
                    continue
                mask = np.array([x == v for x in variants])
                per_variant[v] = {
                    "n": int(mask.sum()),
                    "det_rate": float((atk_lw[mask] >= threshold).mean()),
                    "mean_score": float(atk_lw[mask].mean()),
                }

        results[atk_name] = {
            "n": int(n_atk),
            "detection_rate": det_rate,
            "auc": a,
            "mean_score": float(atk_lw.mean()),
            "std_score": float(atk_lw.std()),
            "top_layers": [int(l) for l in top3],
            "per_variant": per_variant,
        }
        all_raw_scores[atk_name] = atk_lw.tolist()
        all_z[atk_name] = atk_z.tolist()

    # 6. Print summary ----------------------------------------------------
    print(f"\n{'=' * 72}")
    print(f"RESULTS: {MODEL_NAME}  (L={num_layers}, threshold={threshold:.3f})")
    print(f"{'=' * 72}")
    print(f"  Clean FPR (refusal test): {clean_fpr:.1%}\n")
    hdr = f"  {'Category':<26} {'N':>5} {'Det Rate':>10} {'AUC':>8} {'Mean':>9} {'TopL':>14}"
    print(hdr)
    print(f"  {'-' * 72}")
    for name in [
        "cyberseceval2_direct",
        "cyberseceval2_indirect",
        "bipia_email_inject",
        "bipia_email_inject_code",
        "bipia_email_clean",
    ]:
        if name not in results:
            continue
        r = results[name]
        layers = ",".join(f"L{l}" for l in r["top_layers"])
        print(
            f"  {name:<26} {r['n']:>5} {r['detection_rate']:>9.1%} "
            f"{r['auc']:>8.4f} {r['mean_score']:>9.2f} {layers:>14}"
        )

    # 7. Save -------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "config": {
            "model": MODEL_PATH,
            "model_name": MODEL_NAME,
            "num_layers": num_layers,
            "n_cal": int(n_cal),
            "threshold": float(threshold),
            "prompt_template": PROMPT_TEMPLATE,
            "cal_source": CAL_SOURCE,
        },
        "results": results,
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(output, f, indent=2)
    np.savez(
        OUTPUT_DIR / "scores.npz",
        **{k: np.array(v) for k, v in all_raw_scores.items()},
    )
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
