#!/usr/bin/env python3
"""
LLMScan baseline driver for jailbreak detection.

Mirrors the LCF jailbreak protocol (jailbreak_detection_experiment.py) so the
two methods are evaluated on the same data, then trains LLMScan's MLP detector
on a 70/30 stratified split per (model, attack) and reports both:
  (a) LLMScan-protocol numbers  : AUC / accuracy on the 30% held-out test
  (b) LCF-comparable numbers     : detection rate on the FULL attack set,
                                   FPR on the LCF clean test set
                                   (both using the MLP trained on the 70% split).
"""

from __future__ import annotations

import os
import sys
import json
import glob
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)

DPA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DPA_ROOT))

from baselines.llmscan.scanner import (  # noqa: E402
    LLMScanScanner,
    scan_dataset,
)

MODEL_PATH = os.environ.get("MODEL_PATH", "meta-llama/Meta-Llama-3-8B-Instruct")
PROMPT_TEMPLATE = os.environ.get("PROMPT_TEMPLATE", "alpaca")
N_CLEAN = int(os.environ.get("N_CLEAN", "200"))
MAX_TOKEN_POS = int(os.environ.get("MAX_TOKEN_POS", "64"))
SEED = int(os.environ.get("SEED", "0"))
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"

MODEL_NAME = os.path.basename(MODEL_PATH)
OUTPUT_TAG = os.environ.get("OUTPUT_TAG", "")
OUTPUT_DIR = DPA_ROOT / "jailbreak_baseline_llmscan_results" / (MODEL_NAME + OUTPUT_TAG)


def load_calibration_data(n_cal: int):
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


def load_lcf_clean_test(n: int = 200):
    path = DPA_ROOT / "data" / "test_data" / "clean" / "refusal" / "test_data_no_trigger.json"
    return json.load(open(path))[:n]


def load_jailbreak_attacks():
    base = DPA_ROOT / "data" / "test_data" / "poison" / "jailbreak"
    attacks: dict[str, list] = {}
    gcg = base / "gcg" / "backdoor200_jailbreak_gcg.json"
    pair = base / "pair" / "pair_jailbreaks.json"
    dan = base / "dan" / "dan_jailbreaks.json"
    goals = base / "gcg" / "clean_goals_no_suffix.json"
    if gcg.exists():
        attacks["GCG"] = json.load(open(gcg))
    if pair.exists():
        attacks["PAIR"] = json.load(open(pair))
    if dan.exists():
        dan_all = json.load(open(dan))
        attacks["DAN"] = [x for x in dan_all if x.get("source") == "DAN_template"]
        attacks["Roleplay"] = [x for x in dan_all if x.get("source") == "roleplay_template"]
    if goals.exists():
        attacks["Goals_only"] = json.load(open(goals))
    return attacks


def load_model(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": torch.cuda.current_device()},
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def stratified_split(n_clean: int, n_atk: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    clean_idx = rng.permutation(n_clean)
    atk_idx = rng.permutation(n_atk)
    nc_train = int(round(n_clean * 0.7))
    na_train = int(round(n_atk * 0.7))
    return (clean_idx[:nc_train], clean_idx[nc_train:],
            atk_idx[:na_train], atk_idx[na_train:])


def fit_and_eval(clean_feat, atk_feat, atk_full_feat, lcf_clean_feat, seed=0):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    n_c, n_a = len(clean_feat), len(atk_feat)
    if n_a < 4 or n_c < 4:
        return None
    c_train, c_test, a_train, a_test = stratified_split(n_c, n_a, seed=seed)

    X_train = np.concatenate([clean_feat[c_train], atk_feat[a_train]], axis=0)
    y_train = np.concatenate([np.zeros(len(c_train)), np.ones(len(a_train))])
    X_test = np.concatenate([clean_feat[c_test], atk_feat[a_test]], axis=0)
    y_test = np.concatenate([np.zeros(len(c_test)), np.ones(len(a_test))])

    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=seed,
        early_stopping=False,
    )
    mlp.fit(Xtr, y_train)
    test_proba = mlp.predict_proba(Xte)[:, 1]

    from sklearn.metrics import roc_auc_score, accuracy_score
    auc = float(roc_auc_score(y_test, test_proba)) if len(np.unique(y_test)) > 1 else 0.5
    acc = float(accuracy_score(y_test, (test_proba >= 0.5).astype(int)))

    full_atk_proba = mlp.predict_proba(scaler.transform(atk_full_feat))[:, 1]
    full_atk_det_rate = float((full_atk_proba >= 0.5).mean())

    lcf_clean_proba = mlp.predict_proba(scaler.transform(lcf_clean_feat))[:, 1]
    lcf_clean_fpr = float((lcf_clean_proba >= 0.5).mean())

    return {
        "auc_70_30": auc,
        "accuracy_70_30": acc,
        "n_train_clean": int(len(c_train)),
        "n_train_atk": int(len(a_train)),
        "n_test_clean": int(len(c_test)),
        "n_test_atk": int(len(a_test)),
        "full_attack_det_rate": full_atk_det_rate,
        "lcf_clean_fpr": lcf_clean_fpr,
    }


def main():
    print("=" * 70)
    print("LLMScan baseline (jailbreak)")
    print("=" * 70)
    print(f"  Model:         {MODEL_PATH}")
    print(f"  Template:      {PROMPT_TEMPLATE}")
    print(f"  N_CLEAN:       {N_CLEAN}")
    print(f"  MAX_TOKEN_POS: {MAX_TOKEN_POS}")
    print(f"  Output dir:    {OUTPUT_DIR}")
    print(f"  Smoke test:    {SMOKE_TEST}")

    cal = load_calibration_data(N_CLEAN)
    lcf_clean_test = load_lcf_clean_test(200)
    attacks = load_jailbreak_attacks()
    print(f"\n  Calibration:    {len(cal)} clean (used as MLP-train clean)")
    print(f"  LCF clean test: {len(lcf_clean_test)}")
    for n, d in attacks.items():
        print(f"  Attack {n}:      {len(d)}")

    if SMOKE_TEST:
        cal = cal[:6]
        lcf_clean_test = lcf_clean_test[:6]
        attacks = {k: v[:6] for k, v in attacks.items()}
        print("\n  SMOKE TEST: cap all sets to 6 examples\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feat_path = OUTPUT_DIR / "features.npz"

    cached: dict = {}
    if feat_path.exists():
        with np.load(feat_path) as zf:
            cached = {k: zf[k] for k in zf.files}
        print(f"\nResuming from {feat_path}: cached keys = {sorted(cached.keys())}", flush=True)

    def _save_features(cal_feat, lcf_clean_feat, atk_feats):
        payload = {}
        if cal_feat is not None:
            payload["cal"] = cal_feat
        if lcf_clean_feat is not None:
            payload["lcf_clean_test"] = lcf_clean_feat
        payload.update(atk_feats)
        np.savez(feat_path, **payload)

    print("\nLoading model...", flush=True)
    t0 = time.time()
    model, tokenizer = load_model(MODEL_PATH)
    scanner = LLMScanScanner(model, tokenizer, prompt_template=PROMPT_TEMPLATE)
    print(f"  loaded in {time.time() - t0:.1f}s ; layers={scanner.num_layers}", flush=True)

    if "cal" in cached:
        cal_feat = cached["cal"]
        print(f"\nSkipping calibration scan (cached, shape {cal_feat.shape})", flush=True)
    else:
        print(f"\nScanning calibration ({len(cal)})...", flush=True)
        t0 = time.time()
        cal_feat = scan_dataset(scanner, cal, PROMPT_TEMPLATE, max_token_positions=MAX_TOKEN_POS, label="cal")
        print(f"  done in {time.time() - t0:.1f}s ; feature shape {cal_feat.shape}", flush=True)
        _save_features(cal_feat, None, {})

    if "lcf_clean_test" in cached:
        lcf_clean_feat = cached["lcf_clean_test"]
        print(f"\nSkipping LCF clean test scan (cached, shape {lcf_clean_feat.shape})", flush=True)
    else:
        print(f"\nScanning LCF clean test ({len(lcf_clean_test)})...", flush=True)
        t0 = time.time()
        lcf_clean_feat = scan_dataset(scanner, lcf_clean_test, PROMPT_TEMPLATE, max_token_positions=MAX_TOKEN_POS, label="lcf_clean")
        print(f"  done in {time.time() - t0:.1f}s", flush=True)
        _save_features(cal_feat, lcf_clean_feat, {})

    atk_feats = {}
    for name, data in attacks.items():
        if not data:
            continue
        if name in cached:
            atk_feats[name] = cached[name]
            print(f"\nSkipping {name} scan (cached, shape {atk_feats[name].shape})", flush=True)
            continue
        print(f"\nScanning {name} ({len(data)})...", flush=True)
        t0 = time.time()
        atk_feats[name] = scan_dataset(scanner, data, PROMPT_TEMPLATE, max_token_positions=MAX_TOKEN_POS, label=name)
        print(f"  done in {time.time() - t0:.1f}s", flush=True)
        _save_features(cal_feat, lcf_clean_feat, atk_feats)
        print(f"  features.npz checkpointed", flush=True)

    print(f"\nAll features saved to {feat_path}", flush=True)

    finite_summary = {}
    for k, v in {"cal": cal_feat, "lcf_clean_test": lcf_clean_feat, **atk_feats}.items():
        n_nan = int(np.isnan(v).sum())
        n_inf = int(np.isinf(v).sum())
        finite_summary[k] = {"shape": list(v.shape), "n_nan": n_nan, "n_inf": n_inf,
                             "max_abs": float(np.nanmax(np.abs(v))) if v.size else 0.0}
    print(f"\n  feature finiteness: {finite_summary}", flush=True)

    results = {}
    for name, feat in atk_feats.items():
        try:
            r = fit_and_eval(cal_feat, feat, feat, lcf_clean_feat, seed=SEED)
            if r is None:
                results[name] = {"skipped": True, "reason": "too few examples"}
                continue
            results[name] = r
            print(
                f"  [{name}] auc={r['auc_70_30']:.3f} acc={r['accuracy_70_30']:.3f}"
                f"  full_atk_det={r['full_attack_det_rate']:.1%}  lcf_clean_fpr={r['lcf_clean_fpr']:.1%}",
                flush=True,
            )
        except Exception as e:
            results[name] = {"error": str(e), "type": type(e).__name__}
            print(f"  [{name}] FAILED: {type(e).__name__}: {e}", flush=True)

    out = {
        "config": {
            "model": MODEL_PATH,
            "model_name": MODEL_NAME,
            "prompt_template": PROMPT_TEMPLATE,
            "num_layers": scanner.num_layers,
            "n_clean": len(cal),
            "n_lcf_clean_test": len(lcf_clean_test),
            "max_token_positions": MAX_TOKEN_POS,
            "seed": SEED,
            "feature_dim": int(cal_feat.shape[1]),
            "attacks": {k: len(v) for k, v in attacks.items()},
            "feature_finiteness": finite_summary,
        },
        "results": results,
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results.json to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
