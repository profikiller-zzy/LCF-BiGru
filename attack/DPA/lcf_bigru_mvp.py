#!/usr/bin/env python3
"""
LCF-BiGRU MVP experiment.

This script keeps the original LCF pipeline intact and adds a supervised
trajectory detector on top of LCF-style hidden-state deltas:

    delta_l = h_{l+1} - h_l
    z_l = (delta_l - mu_l) / sigma_l
    r_l = PCA_l(z_l)
    x_l = concat(r_l, s_l, scalar_zscore_l, optional layer embedding)

It runs a compact Qwen2.5-7B-Instruct MVP by default:
  1. fit clean calibration stats and per-layer PCA;
  2. cache projected sequence features;
  3. evaluate LCF Ledoit-Wolf baseline;
  4. train BiGRU variants for k in {16,32,64}.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DPA_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DPA_ROOT.parent.parent
DEFAULT_MODEL_PATH = str(REPO_ROOT / "models" / "Qwen2.5-7B-Instruct")
SCHEMA_KEYS = ["instruction", "input", "output", "label", "prompt_type", "attack_type", "source_dataset"]


@dataclass
class Example:
    instruction: str
    user_input: str
    label: int
    attack_type: str
    source_dataset: str
    example_id: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_id(text: str, attack_type: str, source: str) -> str:
    import hashlib

    return hashlib.sha256(f"{attack_type}\n{source}\n{text}".encode("utf-8")).hexdigest()[:16]


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def row_to_example(row: dict) -> Example:
    instruction = (row.get("instruction") or "").strip()
    user_input = (row.get("input") or "").strip()
    attack_type = row.get("attack_type") or ("benign" if row.get("label") == 0 else "jailbreak")
    source = row.get("source_dataset") or "unknown"
    return Example(
        instruction=instruction,
        user_input=user_input,
        label=int(row.get("label", 1)),
        attack_type=attack_type,
        source_dataset=source,
        example_id=stable_id(instruction, attack_type, source),
    )


def load_prompt_library() -> tuple[list[Example], dict[str, list[Example]]]:
    benign_path = DPA_ROOT / "data" / "prompts" / "benign" / "benign_prompts.json"
    jailbreak_root = DPA_ROOT / "data" / "prompts" / "jailbreak"

    benign = [row_to_example(row) for row in load_json(benign_path) if (row.get("instruction") or "").strip()]
    attacks: dict[str, list[Example]] = {}
    for path in sorted(jailbreak_root.glob("*/*_prompts.json")):
        attack_type = path.parent.name
        rows = [row_to_example(row) for row in load_json(path) if (row.get("instruction") or "").strip()]
        attacks[attack_type] = rows
    return benign, attacks


def sample_splits(
    benign: list[Example],
    attacks: dict[str, list[Example]],
    *,
    n_calibration: int,
    train_per_attack: int,
    val_per_attack: int,
    test_per_attack: int,
    seed: int,
    include_attacks: list[str] | None = None,
) -> dict[str, list[Example]]:
    rng = random.Random(seed)
    benign_shuffled = list(benign)
    rng.shuffle(benign_shuffled)
    cal = benign_shuffled[:n_calibration]
    benign_pool = benign_shuffled[n_calibration:]

    chosen_attacks = include_attacks or sorted(attacks)
    train_pos: list[Example] = []
    val_pos: list[Example] = []
    test_pos: list[Example] = []

    for attack_type in chosen_attacks:
        rows = list(attacks.get(attack_type, []))
        rng.shuffle(rows)
        n_train = min(train_per_attack, len(rows))
        n_val = min(val_per_attack, max(0, len(rows) - n_train))
        n_test = min(test_per_attack, max(0, len(rows) - n_train - n_val))
        train_pos.extend(rows[:n_train])
        val_pos.extend(rows[n_train : n_train + n_val])
        test_pos.extend(rows[n_train + n_val : n_train + n_val + n_test])

    n_train_neg = len(train_pos)
    n_val_neg = len(val_pos)
    n_test_neg = len(test_pos)
    need = n_train_neg + n_val_neg + n_test_neg
    if len(benign_pool) < need:
        raise ValueError(f"Not enough benign prompts: need {need}, have {len(benign_pool)}")

    train_neg = benign_pool[:n_train_neg]
    val_neg = benign_pool[n_train_neg : n_train_neg + n_val_neg]
    test_neg = benign_pool[n_train_neg + n_val_neg : n_train_neg + n_val_neg + n_test_neg]

    splits = {
        "cal": cal,
        "train": train_neg + train_pos,
        "val": val_neg + val_pos,
        "test": test_neg + test_pos,
    }
    for name in ["train", "val", "test"]:
        rng.shuffle(splits[name])
    return splits


def _ratio_counts(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 0, 1

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    n_train = max(1, n_train)
    n_val = max(1, n_val)
    n_test = max(1, n_test)

    while n_train + n_val + n_test > n:
        if n_train >= n_val and n_train >= n_test and n_train > 1:
            n_train -= 1
        elif n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    while n_train + n_val + n_test < n:
        n_train += 1
    return n_train, n_val, n_test


def full_prompt_splits(
    benign: list[Example],
    attacks: dict[str, list[Example]],
    *,
    n_calibration: int,
    train_ratio: float,
    val_ratio: float,
    negative_ratio: float,
    seed: int,
    include_attacks: list[str] | None = None,
) -> dict[str, list[Example]]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("--train-ratio must be >0 and train+val must be <1")
    if negative_ratio <= 0:
        raise ValueError("--negative-ratio must be >0")

    rng = random.Random(seed)
    benign_shuffled = list(benign)
    rng.shuffle(benign_shuffled)
    cal = benign_shuffled[:n_calibration]
    benign_pool = benign_shuffled[n_calibration:]

    chosen_attacks = include_attacks or sorted(attacks)
    train_pos: list[Example] = []
    val_pos: list[Example] = []
    test_pos: list[Example] = []

    for attack_type in chosen_attacks:
        rows = list(attacks.get(attack_type, []))
        rng.shuffle(rows)
        n_train, n_val, n_test = _ratio_counts(len(rows), train_ratio, val_ratio)
        train_pos.extend(rows[:n_train])
        val_pos.extend(rows[n_train : n_train + n_val])
        test_pos.extend(rows[n_train + n_val : n_train + n_val + n_test])

    n_train_neg = int(round(len(train_pos) * negative_ratio))
    n_val_neg = int(round(len(val_pos) * negative_ratio))
    n_test_neg = int(round(len(test_pos) * negative_ratio))
    need = n_train_neg + n_val_neg + n_test_neg
    if len(benign_pool) < need:
        raise ValueError(f"Not enough benign prompts: need {need}, have {len(benign_pool)}")

    train_neg = benign_pool[:n_train_neg]
    val_neg = benign_pool[n_train_neg : n_train_neg + n_val_neg]
    test_neg = benign_pool[n_train_neg + n_val_neg : n_train_neg + n_val_neg + n_test_neg]

    splits = {
        "cal": cal,
        "train": train_neg + train_pos,
        "val": val_neg + val_pos,
        "test": test_neg + test_pos,
    }
    for name in ["train", "val", "test"]:
        rng.shuffle(splits[name])
    return splits


def build_model_inputs(tokenizer, instruction: str, user_input: str = "", prompt_template: str = "alpaca", max_input_tokens: int = 2048):
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
    elif prompt_template == "llama3":
        content = f"{instruction}\n{user_input}" if user_input else instruction
        prompt_text = (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{content}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    else:
        raise ValueError(f"Unsupported prompt template: {prompt_template}")

    old_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    tokenizer.truncation_side = old_side
    return inputs


def load_model_and_tokenizer(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Switch the server to GPU mode before feature extraction.")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": torch.cuda.current_device()},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer


def get_num_layers(model) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
        if hasattr(model.config, attr):
            return int(getattr(model.config, attr))
    raise ValueError("Cannot determine number of layers")


@torch.no_grad()
def collect_delta_one(model, tokenizer, ex: Example, prompt_template: str, max_input_tokens: int, num_layers: int) -> np.ndarray:
    inputs = build_model_inputs(tokenizer, ex.instruction, ex.user_input, prompt_template, max_input_tokens)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states
    deltas = []
    for layer in range(num_layers):
        h_curr = hs[layer + 1][:, -1, :].float()
        h_prev = hs[layer][:, -1, :].float()
        deltas.append((h_curr - h_prev).squeeze(0).cpu().numpy())
    return np.stack(deltas, axis=0).astype(np.float32)


def collect_calibration_deltas(model, tokenizer, examples: list[Example], args, num_layers: int) -> np.ndarray:
    rows = []
    t0 = time.time()
    for i, ex in enumerate(examples, 1):
        rows.append(collect_delta_one(model, tokenizer, ex, args.prompt_template, args.max_input_tokens, num_layers))
        if i % args.log_every == 0 or i == len(examples):
            print(f"  calibration deltas {i}/{len(examples)} ({time.time() - t0:.1f}s)", flush=True)
    return np.stack(rows, axis=0)


def fit_calibration(cal_deltas: np.ndarray, pca_dim: int, seed: int) -> dict:
    layer_mu = cal_deltas.mean(axis=0).astype(np.float32)
    layer_sigma = np.maximum(cal_deltas.std(axis=0), 1e-6).astype(np.float32)
    cal_z_dir = (cal_deltas - layer_mu[None, :, :]) / layer_sigma[None, :, :]
    cal_scores = np.linalg.norm(cal_z_dir, axis=2).astype(np.float32)
    score_mu = cal_scores.mean(axis=0).astype(np.float32)
    score_sigma = np.maximum(cal_scores.std(axis=0), 1e-6).astype(np.float32)
    cal_scalar_z = ((cal_scores - score_mu[None, :]) / score_sigma[None, :]).astype(np.float32)

    pcas = []
    num_layers = cal_deltas.shape[1]
    n_components = min(pca_dim, cal_deltas.shape[0] - 1, cal_deltas.shape[2])
    if n_components < pca_dim:
        raise ValueError(f"pca_dim={pca_dim} too large for calibration set; max is {n_components}")
    for layer in range(num_layers):
        pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=seed)
        pca.fit(cal_z_dir[:, layer, :])
        pcas.append(pca)
    return {
        "layer_mu": layer_mu,
        "layer_sigma": layer_sigma,
        "score_mu": score_mu,
        "score_sigma": score_sigma,
        "cal_scalar_z": cal_scalar_z,
        "pcas": pcas,
    }


def loo_lw_threshold(cal_z: np.ndarray, target_fpr: float) -> tuple[float, np.ndarray]:
    n_cal = cal_z.shape[0]
    loo_scores = np.zeros(n_cal, dtype=np.float32)
    for i in range(n_cal):
        mask = np.ones(n_cal, dtype=bool)
        mask[i] = False
        z = cal_z[mask]
        mu = z.mean(axis=0)
        prec = LedoitWolf().fit(z).precision_
        diff = cal_z[i] - mu
        loo_scores[i] = float(np.sqrt(diff @ prec @ diff))
    threshold = float(np.percentile(loo_scores, 100 * (1 - target_fpr)))
    return max(threshold, 1.0), loo_scores


def lw_scores(z: np.ndarray, cal_z: np.ndarray) -> np.ndarray:
    mu = cal_z.mean(axis=0)
    prec = LedoitWolf().fit(cal_z).precision_
    diff = z - mu[None, :]
    return np.sqrt(np.sum(diff @ prec * diff, axis=1)).astype(np.float32)


def project_delta(delta: np.ndarray, calibration: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = (delta - calibration["layer_mu"]) / calibration["layer_sigma"]
    scores = np.linalg.norm(z, axis=1).astype(np.float32)
    scalar_z = ((scores - calibration["score_mu"]) / calibration["score_sigma"]).astype(np.float32)
    projected = []
    for layer, pca in enumerate(calibration["pcas"]):
        projected.append(pca.transform(z[layer : layer + 1])[0])
    return np.stack(projected, axis=0).astype(np.float32), scores.astype(np.float32), scalar_z


def featurize_split(model, tokenizer, examples: list[Example], calibration: dict, args, num_layers: int, split_name: str) -> dict:
    r_rows = []
    score_rows = []
    scalar_z_rows = []
    labels = []
    attacks = []
    sources = []
    ids = []
    t0 = time.time()
    for i, ex in enumerate(examples, 1):
        delta = collect_delta_one(model, tokenizer, ex, args.prompt_template, args.max_input_tokens, num_layers)
        r, scores, scalar_z = project_delta(delta, calibration)
        r_rows.append(r)
        score_rows.append(scores[:, None])
        scalar_z_rows.append(scalar_z[:, None])
        labels.append(ex.label)
        attacks.append(ex.attack_type)
        sources.append(ex.source_dataset)
        ids.append(ex.example_id)
        if i % args.log_every == 0 or i == len(examples):
            print(f"  {split_name} features {i}/{len(examples)} ({time.time() - t0:.1f}s)", flush=True)
    return {
        "r": np.stack(r_rows, axis=0).astype(np.float32),
        "scores": np.stack(score_rows, axis=0).astype(np.float32),
        "scalar_z": np.stack(scalar_z_rows, axis=0).astype(np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "attack_types": np.asarray(attacks, dtype=object),
        "sources": np.asarray(sources, dtype=object),
        "example_ids": np.asarray(ids, dtype=object),
    }


def save_feature_cache(path: Path, payload: dict, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for split, data in payload.items():
        for key, value in data.items():
            arrays[f"{split}_{key}"] = value
    arrays["config_json"] = np.asarray(json.dumps(config), dtype=object)
    np.savez_compressed(path, **arrays)


def load_feature_cache(path: Path) -> tuple[dict, dict]:
    data = np.load(path, allow_pickle=True)
    splits: dict[str, dict] = {}
    for split in ["cal", "train", "val", "test"]:
        split_data = {}
        prefix = f"{split}_"
        for key in data.files:
            if key.startswith(prefix):
                split_data[key[len(prefix) :]] = data[key]
        if split_data:
            splits[split] = split_data
    config = json.loads(str(data["config_json"].item()))
    return splits, config


def build_features(args) -> Path:
    set_seed(args.seed)
    benign, attacks = load_prompt_library()
    include_attacks = [x.strip() for x in args.attacks.split(",") if x.strip()] if args.attacks else None
    if args.split_strategy == "full":
        splits = full_prompt_splits(
            benign,
            attacks,
            n_calibration=args.n_calibration,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            negative_ratio=args.negative_ratio,
            seed=args.seed,
            include_attacks=include_attacks,
        )
    else:
        splits = sample_splits(
            benign,
            attacks,
            n_calibration=args.n_calibration,
            train_per_attack=args.train_per_attack,
            val_per_attack=args.val_per_attack,
            test_per_attack=args.test_per_attack,
            seed=args.seed,
            include_attacks=include_attacks,
        )

    print("Split sizes:")
    for name, rows in splits.items():
        labels = {0: 0, 1: 0}
        for ex in rows:
            labels[ex.label] = labels.get(ex.label, 0) + 1
        attack_counts = {}
        for ex in rows:
            if ex.label == 1:
                attack_counts[ex.attack_type] = attack_counts.get(ex.attack_type, 0) + 1
        print(f"  {name}: n={len(rows)}, labels={labels}, attacks={attack_counts}", flush=True)

    model, tokenizer = load_model_and_tokenizer(args.model_path)
    num_layers = get_num_layers(model)
    print(f"Loaded model with {num_layers} layers on {model.device}", flush=True)

    cal_deltas = collect_calibration_deltas(model, tokenizer, splits["cal"], args, num_layers)
    hidden_dim = cal_deltas.shape[-1]
    print(f"Calibration delta shape: {cal_deltas.shape}", flush=True)
    calibration = fit_calibration(cal_deltas, args.pca_dim_max, args.seed)
    print("Fitted LCF stats and per-layer PCA", flush=True)

    payload = {
        "cal": {
            "r": np.zeros((len(splits["cal"]), num_layers, args.pca_dim_max), dtype=np.float32),
            "scores": np.linalg.norm(
                (cal_deltas - calibration["layer_mu"][None, :, :]) / calibration["layer_sigma"][None, :, :],
                axis=2,
            )[:, :, None].astype(np.float32),
            "scalar_z": calibration["cal_scalar_z"][:, :, None].astype(np.float32),
            "labels": np.zeros(len(splits["cal"]), dtype=np.int64),
            "attack_types": np.asarray(["benign"] * len(splits["cal"]), dtype=object),
            "sources": np.asarray([ex.source_dataset for ex in splits["cal"]], dtype=object),
            "example_ids": np.asarray([ex.example_id for ex in splits["cal"]], dtype=object),
        }
    }
    for i in range(len(splits["cal"])):
        projected, _, _ = project_delta(cal_deltas[i], calibration)
        payload["cal"]["r"][i] = projected

    for split in ["train", "val", "test"]:
        payload[split] = featurize_split(model, tokenizer, splits[split], calibration, args, num_layers, split)

    config = {
        "model_path": args.model_path,
        "prompt_template": args.prompt_template,
        "max_input_tokens": args.max_input_tokens,
        "n_calibration": args.n_calibration,
        "split_strategy": args.split_strategy,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "negative_ratio": args.negative_ratio,
        "train_per_attack": args.train_per_attack,
        "val_per_attack": args.val_per_attack,
        "test_per_attack": args.test_per_attack,
        "attacks": include_attacks or sorted(attacks),
        "seed": args.seed,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "pca_dim_max": args.pca_dim_max,
    }
    save_feature_cache(args.feature_cache, payload, config)
    print(f"Saved feature cache to {args.feature_cache}", flush=True)
    return args.feature_cache


def make_input(split_data: dict, k: int, mode: str, layer_embed_dim: int = 8) -> np.ndarray:
    if mode in {"scalar", "scalar_layer"}:
        parts = [split_data["scores"], split_data["scalar_z"]]
    else:
        parts = [split_data["r"][:, :, :k]]
    if mode in {"hybrid", "hybrid_layer"}:
        parts.append(split_data["scores"])
        parts.append(split_data["scalar_z"])
    x = np.concatenate(parts, axis=2).astype(np.float32)
    if mode in {"hybrid_layer", "scalar_layer"}:
        num_layers = x.shape[1]
        dim = layer_embed_dim
        positions = np.arange(num_layers, dtype=np.float32)[:, None]
        freqs = np.exp(np.arange(0, dim, 2, dtype=np.float32) * (-math.log(10000.0) / dim))
        emb = np.zeros((num_layers, dim), dtype=np.float32)
        emb[:, 0::2] = np.sin(positions * freqs)
        emb[:, 1::2] = np.cos(positions * freqs)
        emb = np.broadcast_to(emb[None, :, :], (x.shape[0], num_layers, dim))
        x = np.concatenate([x, emb], axis=2).astype(np.float32)
    return x


class BiGRUDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        h_cat = torch.cat([h[-2], h[-1]], dim=1)
        return self.head(h_cat).squeeze(1)


def standardize(train_x: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    std = np.maximum(train_x.reshape(-1, train_x.shape[-1]).std(axis=0), 1e-6)
    out = [((train_x - mean[None, None, :]) / std[None, None, :]).astype(np.float32)]
    for x in others:
        out.append(((x - mean[None, None, :]) / std[None, None, :]).astype(np.float32))
    return tuple(out)


def score_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float | None = None) -> dict:
    out = {
        "auroc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else 0.5,
        "auprc": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) == 2 else 0.0,
        "mean_score_pos": float(scores[y_true == 1].mean()) if np.any(y_true == 1) else None,
        "mean_score_neg": float(scores[y_true == 0].mean()) if np.any(y_true == 0) else None,
    }
    if threshold is not None:
        pred = scores >= threshold
        pos = y_true == 1
        neg = y_true == 0
        out.update(
            {
                "threshold": float(threshold),
                "tpr": float(pred[pos].mean()) if np.any(pos) else None,
                "fpr": float(pred[neg].mean()) if np.any(neg) else None,
            }
        )
    return out


def per_attack_tpr(y_true: np.ndarray, scores: np.ndarray, attacks: np.ndarray, threshold: float) -> dict[str, float]:
    out = {}
    for attack in sorted(set(attacks.tolist())):
        if attack == "benign":
            continue
        mask = (y_true == 1) & (attacks == attack)
        if np.any(mask):
            out[str(attack)] = float((scores[mask] >= threshold).mean())
    return out


@torch.no_grad()
def predict_model(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    scores = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    for (batch_x,) in loader:
        scores.append(model(batch_x.to(device)).detach().cpu().numpy())
    return np.concatenate(scores, axis=0)


def train_bigru_variant(
    splits: dict,
    *,
    k: int,
    mode: str,
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict:
    set_seed(seed)
    train_x = make_input(splits["train"], k, mode)
    val_x = make_input(splits["val"], k, mode)
    test_x = make_input(splits["test"], k, mode)
    train_x, val_x, test_x = standardize(train_x, val_x, test_x)

    train_y = splits["train"]["labels"].astype(np.float32)
    val_y = splits["val"]["labels"].astype(np.int64)
    test_y = splits["test"]["labels"].astype(np.int64)
    input_dim = train_x.shape[-1]

    model = BiGRUDetector(input_dim, hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos = float(train_y.sum())
    neg = float(len(train_y) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
    )

    best_state = None
    best_val = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_scores = predict_model(model, val_x, batch_size, device)
        val_auc = roc_auc_score(val_y, val_scores) if len(np.unique(val_y)) == 2 else 0.5
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_auroc": float(val_auc)})
        if val_auc > best_val:
            best_val = float(val_auc)
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val_scores = predict_model(model, val_x, batch_size, device)
    test_scores = predict_model(model, test_x, batch_size, device)
    threshold = float(np.percentile(val_scores[val_y == 0], 90))
    result = {
        "k": k,
        "mode": mode,
        "input_dim": input_dim,
        "best_val_auroc": best_val,
        "val": score_metrics(val_y, val_scores, threshold),
        "test": score_metrics(test_y, test_scores, threshold),
        "test_per_attack_tpr": per_attack_tpr(test_y, test_scores, splits["test"]["attack_types"], threshold),
        "history": history,
    }
    return result


def evaluate_lcf_baseline(splits: dict, target_fpr: float) -> dict:
    cal_z = splits["cal"]["scalar_z"].squeeze(-1)
    val_z = splits["val"]["scalar_z"].squeeze(-1)
    test_z = splits["test"]["scalar_z"].squeeze(-1)
    val_y = splits["val"]["labels"].astype(np.int64)
    test_y = splits["test"]["labels"].astype(np.int64)
    threshold, loo = loo_lw_threshold(cal_z, target_fpr)
    val_scores = lw_scores(val_z, cal_z)
    test_scores = lw_scores(test_z, cal_z)
    return {
        "name": "lcf_multi_lw_loo",
        "threshold_source": "calibration_loo",
        "threshold": float(threshold),
        "cal_loo_mean": float(loo.mean()),
        "cal_loo_std": float(loo.std()),
        "val": score_metrics(val_y, val_scores, threshold),
        "test": score_metrics(test_y, test_scores, threshold),
        "test_per_attack_tpr": per_attack_tpr(test_y, test_scores, splits["test"]["attack_types"], threshold),
    }


def run_training(args) -> dict:
    splits, feature_config = load_feature_cache(args.feature_cache)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu_train else "cpu")
    print(f"Training/eval device: {device}", flush=True)

    results = {
        "config": {
            "feature_config": feature_config,
            "ks": args.ks,
            "modes": args.modes,
            "hidden_dim": args.bigru_hidden_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "target_fpr": args.target_fpr,
        },
        "splits": {
            name: {
                "n": int(len(data["labels"])),
                "positives": int(data["labels"].sum()),
                "negatives": int((data["labels"] == 0).sum()),
            }
            for name, data in splits.items()
        },
        "lcf_baseline": evaluate_lcf_baseline(splits, args.target_fpr),
        "bigru": [],
    }

    seen_variants = set()
    for k in args.ks:
        for mode in args.modes:
            effective_k = 0 if mode in {"scalar", "scalar_layer"} else k
            variant_key = (mode, effective_k)
            if variant_key in seen_variants:
                continue
            seen_variants.add(variant_key)
            print(f"Training BiGRU variant mode={mode} k={k}", flush=True)
            t0 = time.time()
            result = train_bigru_variant(
                splits,
                k=effective_k,
                mode=mode,
                hidden_dim=args.bigru_hidden_dim,
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                device=device,
            )
            result["seconds"] = float(time.time() - t0)
            print(
                f"  done mode={mode} k={effective_k}: "
                f"val_auc={result['val']['auroc']:.4f} test_auc={result['test']['auroc']:.4f} "
                f"test_tpr@10fpr={result['test']['tpr']:.3f} test_fpr={result['test']['fpr']:.3f}",
                flush=True,
            )
            results["bigru"].append(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Saved results to {out_path}", flush=True)
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--prompt-template", default=os.environ.get("PROMPT_TEMPLATE", "alpaca"), choices=["alpaca", "qwen", "llama3"])
    parser.add_argument("--output-dir", type=Path, default=DPA_ROOT / "lcf_bigru_results" / "qwen2_5_7b_mvp")
    parser.add_argument("--feature-cache", type=Path, default=DPA_ROOT / "lcf_bigru_results" / "qwen2_5_7b_mvp" / "features_k64.npz")
    parser.add_argument("--stage", choices=["features", "train", "all"], default="all")
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-calibration", type=int, default=200)
    parser.add_argument("--split-strategy", choices=["quota", "full"], default="quota")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--train-per-attack", type=int, default=80)
    parser.add_argument("--val-per-attack", type=int, default=20)
    parser.add_argument("--test-per-attack", type=int, default=40)
    parser.add_argument("--attacks", default="", help="Comma-separated attack types. Empty means all prompt-library attacks.")
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--pca-dim-max", type=int, default=64)
    parser.add_argument("--ks", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["r", "hybrid", "hybrid_layer"],
        choices=["scalar", "scalar_layer", "r", "hybrid", "hybrid_layer"],
    )
    parser.add_argument("--bigru-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-fpr", type=float, default=0.10)
    parser.add_argument("--force-cpu-train", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in {"features", "all"}:
        if args.feature_cache.exists() and not args.overwrite_features:
            print(f"Feature cache exists; reusing {args.feature_cache}", flush=True)
        else:
            build_features(args)
    if args.stage in {"train", "all"}:
        run_training(args)


if __name__ == "__main__":
    main()
