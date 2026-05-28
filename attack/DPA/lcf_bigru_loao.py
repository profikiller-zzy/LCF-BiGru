#!/usr/bin/env python3
"""
LCF-BiGRU leave-one-attack-out experiments.

Two protocols are implemented:
  1. backdoor_only: train on six backdoor trigger families and test on the
     held-out seventh trigger family.
  2. unified: train on jailbreak prompts plus six backdoor trigger families,
     test on the held-out backdoor trigger family and on a jailbreak test split.

The script uses a clean calibration set for LCF statistics/PCA, caches features
once, and assembles all LOAO folds from that cache.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lcf_bigru_mvp import (
    BiGRUDetector,
    DEFAULT_MODEL_PATH,
    DPA_ROOT,
    Example,
    collect_calibration_deltas,
    collect_delta_one,
    fit_calibration,
    full_prompt_splits,
    load_model_and_tokenizer,
    load_prompt_library,
    loo_lw_threshold,
    lw_scores,
    make_input,
    per_attack_tpr,
    project_delta,
    score_metrics,
    set_seed,
    stable_id,
)


BACKDOOR_ATTACKS = ["badnet", "sleeper", "vpi", "mtba", "ctba", "stylebkd", "synbkd"]
BACKDOOR_TASKS = ["negsentiment", "refusal"]


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_example(row: dict, *, label: int, attack_type: str, source: str, row_id: int) -> Example:
    instruction = (row.get("instruction") or "").strip()
    user_input = (row.get("input") or "").strip()
    return Example(
        instruction=instruction,
        user_input=user_input,
        label=label,
        attack_type=attack_type,
        source_dataset=source,
        example_id=stable_id(f"{row_id}\n{instruction}", attack_type, source),
    )


def preferred_json(paths: list[Path]) -> Path:
    for path in paths:
        if path.name.startswith("backdoor200_"):
            return path
    return sorted(paths, key=lambda p: p.name)[-1]


def load_clean_calibration_pool() -> list[Example]:
    rows: list[Example] = []
    seen = set()

    sft_path = DPA_ROOT / "data" / "sft_data.json"
    if sft_path.exists():
        for i, row in enumerate(load_json(sft_path)):
            key = (row.get("instruction") or "")[:200]
            if key and key not in seen:
                seen.add(key)
                rows.append(make_example(row, label=0, attack_type="benign", source="sft_data", row_id=i))

    for path in sorted((DPA_ROOT / "data" / "poison_data").glob("*/*/none_*.json")):
        for i, row in enumerate(load_json(path)):
            key = (row.get("instruction") or "")[:200]
            if key and key not in seen:
                seen.add(key)
                rows.append(make_example(row, label=0, attack_type="benign", source=f"clean_cal:{path.parent}", row_id=i))
    return rows


def load_backdoor_train_examples() -> list[Example]:
    examples: list[Example] = []
    for task in BACKDOOR_TASKS:
        for attack in BACKDOOR_ATTACKS:
            paths = sorted((DPA_ROOT / "data" / "poison_data" / task / attack).glob(f"backdoor*_{task}_{attack}.json"))
            if not paths:
                raise FileNotFoundError(f"No train backdoor data for {task}/{attack}")
            path = paths[0]
            for i, row in enumerate(load_json(path)):
                examples.append(
                    make_example(row, label=1, attack_type=attack, source=f"backdoor_train:{task}:{attack}", row_id=i)
                )
    return examples


def load_backdoor_test_examples() -> list[Example]:
    examples: list[Example] = []
    for task in BACKDOOR_TASKS:
        for attack in BACKDOOR_ATTACKS:
            paths = sorted((DPA_ROOT / "data" / "test_data" / "poison" / task / attack).glob(f"backdoor*_{task}_{attack}.json"))
            if not paths:
                raise FileNotFoundError(f"No test backdoor data for {task}/{attack}")
            path = preferred_json(paths)
            for i, row in enumerate(load_json(path)):
                examples.append(
                    make_example(row, label=1, attack_type=attack, source=f"backdoor_test:{task}:{attack}", row_id=i)
                )
    return examples


def load_backdoor_clean_examples() -> tuple[list[Example], list[Example]]:
    train: list[Example] = []
    for path in sorted((DPA_ROOT / "data" / "poison_data").glob("*/*/none_*.json")):
        parts = path.parts
        task, attack = parts[-3], parts[-2]
        for i, row in enumerate(load_json(path)):
            train.append(make_example(row, label=0, attack_type="benign", source=f"clean_train:{task}:{attack}", row_id=i))

    test: list[Example] = []
    for task in BACKDOOR_TASKS:
        path = DPA_ROOT / "data" / "test_data" / "clean" / task / "test_data_no_trigger.json"
        for i, row in enumerate(load_json(path)):
            test.append(make_example(row, label=0, attack_type="benign", source=f"clean_test:{task}", row_id=i))
    return train, test


def split_indices_by_attack(examples: list[Example], val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    groups: dict[tuple[str, str], list[int]] = {}
    for i, ex in enumerate(examples):
        groups.setdefault((ex.attack_type, ex.source_dataset), []).append(i)

    for idxs in groups.values():
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_ratio))) if len(idxs) > 2 else 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def split_indices(examples: list[Example], val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = list(range(len(examples)))
    random.Random(seed).shuffle(idx)
    n_val = int(round(len(idx) * val_ratio))
    return np.asarray(idx[n_val:], dtype=np.int64), np.asarray(idx[:n_val], dtype=np.int64)


def featurize_examples(model, tokenizer, examples: list[Example], calibration: dict, args, num_layers: int, name: str) -> dict:
    r_rows, score_rows, scalar_z_rows = [], [], []
    labels, attacks, sources, ids = [], [], [], []
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
            print(f"  {name} features {i}/{len(examples)} ({time.time() - t0:.1f}s)", flush=True)
    return {
        "r": np.stack(r_rows, axis=0).astype(np.float32),
        "scores": np.stack(score_rows, axis=0).astype(np.float32),
        "scalar_z": np.stack(scalar_z_rows, axis=0).astype(np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "attack_types": np.asarray(attacks, dtype=object),
        "sources": np.asarray(sources, dtype=object),
        "example_ids": np.asarray(ids, dtype=object),
    }


def subset(data: dict, idx: np.ndarray) -> dict:
    return {key: value[idx] for key, value in data.items()}


def concat(parts: list[dict]) -> dict:
    keys = parts[0].keys()
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}


def shuffle_data(data: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data["labels"]))
    return subset(data, idx)


def take(data: dict, n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data["labels"]))[:n]
    return subset(data, idx)


def save_cache(path: Path, payload: dict, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"config_json": np.asarray(json.dumps(config), dtype=object)}
    for name, data in payload.items():
        for key, value in data.items():
            arrays[f"{name}_{key}"] = value
    np.savez_compressed(path, **arrays)


def load_cache(path: Path) -> tuple[dict, dict]:
    data = np.load(path, allow_pickle=True)
    payload: dict[str, dict] = {}
    for key in data.files:
        if key == "config_json":
            continue
        name, field = key.split("_", 1)
        payload.setdefault(name, {})[field] = data[key]
    return payload, json.loads(str(data["config_json"].item()))


def build_feature_cache(args) -> Path:
    set_seed(args.seed)
    clean_cal = load_clean_calibration_pool()
    random.Random(args.seed).shuffle(clean_cal)
    clean_cal = clean_cal[: args.n_calibration]

    bd_train = load_backdoor_train_examples()
    bd_test = load_backdoor_test_examples()
    clean_train, clean_test = load_backdoor_clean_examples()

    benign, jailbreak_attacks = load_prompt_library()
    jb_splits = full_prompt_splits(
        benign,
        jailbreak_attacks,
        n_calibration=0,
        train_ratio=args.jailbreak_train_ratio,
        val_ratio=args.jailbreak_val_ratio,
        negative_ratio=1.0,
        seed=args.seed,
    )

    print("Dataset sizes:", flush=True)
    print(f"  calibration clean: {len(clean_cal)}", flush=True)
    print(f"  backdoor train positives: {len(bd_train)}", flush=True)
    print(f"  backdoor test positives: {len(bd_test)}", flush=True)
    print(f"  backdoor clean train pool: {len(clean_train)}", flush=True)
    print(f"  backdoor clean test: {len(clean_test)}", flush=True)
    for name in ["train", "val", "test"]:
        rows = jb_splits[name]
        print(f"  jailbreak {name}: n={len(rows)} pos={sum(x.label for x in rows)}", flush=True)

    model, tokenizer = load_model_and_tokenizer(args.model_path)
    num_layers = len(model.model.layers) if hasattr(model, "model") and hasattr(model.model, "layers") else int(model.config.num_hidden_layers)
    print(f"Loaded model with {num_layers} layers on {model.device}", flush=True)

    cal_deltas = collect_calibration_deltas(model, tokenizer, clean_cal, args, num_layers)
    calibration = fit_calibration(cal_deltas, args.pca_dim_max, args.seed)
    cal_projected = []
    for i in range(len(clean_cal)):
        projected, _, _ = project_delta(cal_deltas[i], calibration)
        cal_projected.append(projected)
    payload = {
        "cal": {
            "r": np.stack(cal_projected, axis=0).astype(np.float32),
            "scores": np.linalg.norm(
                (cal_deltas - calibration["layer_mu"][None, :, :]) / calibration["layer_sigma"][None, :, :],
                axis=2,
            )[:, :, None].astype(np.float32),
            "scalar_z": calibration["cal_scalar_z"][:, :, None].astype(np.float32),
            "labels": np.zeros(len(clean_cal), dtype=np.int64),
            "attack_types": np.asarray(["benign"] * len(clean_cal), dtype=object),
            "sources": np.asarray([ex.source_dataset for ex in clean_cal], dtype=object),
            "example_ids": np.asarray([ex.example_id for ex in clean_cal], dtype=object),
        },
        "bdtrain": featurize_examples(model, tokenizer, bd_train, calibration, args, num_layers, "bdtrain"),
        "bdtest": featurize_examples(model, tokenizer, bd_test, calibration, args, num_layers, "bdtest"),
        "cleantrain": featurize_examples(model, tokenizer, clean_train, calibration, args, num_layers, "cleantrain"),
        "cleantest": featurize_examples(model, tokenizer, clean_test, calibration, args, num_layers, "cleantest"),
        "jbtrain": featurize_examples(model, tokenizer, jb_splits["train"], calibration, args, num_layers, "jbtrain"),
        "jbval": featurize_examples(model, tokenizer, jb_splits["val"], calibration, args, num_layers, "jbval"),
        "jbtest": featurize_examples(model, tokenizer, jb_splits["test"], calibration, args, num_layers, "jbtest"),
    }
    config = {
        "model_path": args.model_path,
        "prompt_template": args.prompt_template,
        "max_input_tokens": args.max_input_tokens,
        "n_calibration": args.n_calibration,
        "pca_dim_max": args.pca_dim_max,
        "seed": args.seed,
        "num_layers": num_layers,
        "attacks": BACKDOOR_ATTACKS,
        "tasks": BACKDOOR_TASKS,
        "jailbreak_train_ratio": args.jailbreak_train_ratio,
        "jailbreak_val_ratio": args.jailbreak_val_ratio,
    }
    save_cache(args.feature_cache, payload, config)
    print(f"Saved feature cache to {args.feature_cache}", flush=True)
    return args.feature_cache


def standardize_from_train(train_x: np.ndarray, arrays: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    flat = train_x.reshape(-1, train_x.shape[-1])
    mean = flat.mean(axis=0)
    std = np.maximum(flat.std(axis=0), 1e-6)
    train_out = ((train_x - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    other_out = [((x - mean[None, None, :]) / std[None, None, :]).astype(np.float32) for x in arrays]
    return train_out, other_out


@torch.no_grad()
def predict_model(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    scores = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    for (batch_x,) in loader:
        scores.append(model(batch_x.to(device)).detach().cpu().numpy())
    return np.concatenate(scores, axis=0)


def train_and_evaluate(
    train: dict,
    val: dict,
    eval_sets: dict[str, dict],
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
    train_x = make_input(train, k, mode)
    val_x = make_input(val, k, mode)
    eval_x = {name: make_input(data, k, mode) for name, data in eval_sets.items()}
    train_x, others = standardize_from_train(train_x, [val_x] + list(eval_x.values()))
    val_x = others[0]
    eval_x = dict(zip(eval_x.keys(), others[1:]))

    train_y = train["labels"].astype(np.float32)
    val_y = val["labels"].astype(np.int64)
    model = BiGRUDetector(train_x.shape[-1], hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos = float(train_y.sum())
    neg = float(len(train_y) - pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device))
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=batch_size, shuffle=True)

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
    threshold = float(np.percentile(val_scores[val_y == 0], 90))
    result = {
        "mode": mode,
        "k": k,
        "input_dim": int(train_x.shape[-1]),
        "best_val_auroc": best_val,
        "val": score_metrics(val_y, val_scores, threshold),
        "eval": {},
        "history": history,
    }
    for name, data in eval_sets.items():
        y = data["labels"].astype(np.int64)
        scores = predict_model(model, eval_x[name], batch_size, device)
        result["eval"][name] = score_metrics(y, scores, threshold)
        result["eval"][name]["per_attack_tpr"] = per_attack_tpr(y, scores, data["attack_types"], threshold)
    return result


def lcf_eval(cal: dict, eval_sets: dict[str, dict], target_fpr: float) -> dict:
    cal_z = cal["scalar_z"].squeeze(-1)
    threshold, loo = loo_lw_threshold(cal_z, target_fpr)
    out = {"threshold": threshold, "cal_loo_mean": float(loo.mean()), "cal_loo_std": float(loo.std()), "eval": {}}
    for name, data in eval_sets.items():
        z = data["scalar_z"].squeeze(-1)
        y = data["labels"].astype(np.int64)
        scores = lw_scores(z, cal_z)
        out["eval"][name] = score_metrics(y, scores, threshold)
        out["eval"][name]["per_attack_tpr"] = per_attack_tpr(y, scores, data["attack_types"], threshold)
    return out


def build_fold(payload: dict, heldout: str, seed: int, unified: bool, bd_val_ratio: float) -> tuple[dict, dict, dict[str, dict]]:
    bd = payload["bdtrain"]
    bd_train_idx_all, bd_val_idx_all = split_indices_by_attack(
        [Example("", "", int(label), str(atk), str(src), str(i)) for i, (label, atk, src) in enumerate(zip(bd["labels"], bd["attack_types"], bd["sources"]))],
        bd_val_ratio,
        seed,
    )
    bd_train_mask = bd["attack_types"][bd_train_idx_all] != heldout
    bd_val_mask = bd["attack_types"][bd_val_idx_all] != heldout
    bd_train_pos = subset(bd, bd_train_idx_all[bd_train_mask])
    bd_val_pos = subset(bd, bd_val_idx_all[bd_val_mask])

    clean_train_idx, clean_val_idx = split_indices(
        [Example("", "", 0, "benign", str(src), str(i)) for i, src in enumerate(payload["cleantrain"]["sources"])],
        bd_val_ratio,
        seed,
    )
    clean_train_pool = subset(payload["cleantrain"], clean_train_idx)
    clean_val_pool = subset(payload["cleantrain"], clean_val_idx)

    bd_test = payload["bdtest"]
    heldout_test = subset(bd_test, np.where(bd_test["attack_types"] == heldout)[0])
    backdoor_eval = shuffle_data(concat([heldout_test, payload["cleantest"]]), seed + 10)

    if not unified:
        train_neg = take(clean_train_pool, len(bd_train_pos["labels"]), seed + 20)
        val_neg = take(clean_val_pool, len(bd_val_pos["labels"]), seed + 21)
        train = shuffle_data(concat([bd_train_pos, train_neg]), seed + 22)
        val = shuffle_data(concat([bd_val_pos, val_neg]), seed + 23)
        return train, val, {"heldout_backdoor": backdoor_eval}

    jb_train_pos = subset(payload["jbtrain"], np.where(payload["jbtrain"]["labels"] == 1)[0])
    jb_train_neg = subset(payload["jbtrain"], np.where(payload["jbtrain"]["labels"] == 0)[0])
    jb_val_pos = subset(payload["jbval"], np.where(payload["jbval"]["labels"] == 1)[0])
    jb_val_neg = subset(payload["jbval"], np.where(payload["jbval"]["labels"] == 0)[0])

    train_pos = concat([bd_train_pos, jb_train_pos])
    val_pos = concat([bd_val_pos, jb_val_pos])
    train_neg_pool = concat([clean_train_pool, jb_train_neg])
    val_neg_pool = concat([clean_val_pool, jb_val_neg])
    train_neg = take(train_neg_pool, len(train_pos["labels"]), seed + 30)
    val_neg = take(val_neg_pool, len(val_pos["labels"]), seed + 31)
    train = shuffle_data(concat([train_pos, train_neg]), seed + 32)
    val = shuffle_data(concat([val_pos, val_neg]), seed + 33)
    jailbreak_eval = shuffle_data(payload["jbtest"], seed + 34)
    return train, val, {"heldout_backdoor": backdoor_eval, "jailbreak": jailbreak_eval}


def run_training(args) -> dict:
    payload, cache_config = load_cache(args.feature_cache)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu_train else "cpu")
    print(f"Training/eval device: {device}", flush=True)

    output = {
        "config": {
            "cache_config": cache_config,
            "protocols": args.protocols,
            "heldout_attacks": BACKDOOR_ATTACKS,
            "modes": args.modes,
            "ks": args.ks,
            "hidden_dim": args.bigru_hidden_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "target_fpr": args.target_fpr,
        },
        "folds": {},
    }

    for protocol in args.protocols:
        unified = protocol == "unified"
        output["folds"][protocol] = {}
        for heldout in BACKDOOR_ATTACKS:
            print(f"Protocol={protocol} heldout={heldout}", flush=True)
            train, val, eval_sets = build_fold(payload, heldout, args.seed + BACKDOOR_ATTACKS.index(heldout), unified, args.bd_val_ratio)
            fold = {
                "sizes": {
                    "train": {"n": int(len(train["labels"])), "positives": int(train["labels"].sum())},
                    "val": {"n": int(len(val["labels"])), "positives": int(val["labels"].sum())},
                    **{name: {"n": int(len(data["labels"])), "positives": int(data["labels"].sum())} for name, data in eval_sets.items()},
                },
                "lcf_baseline": lcf_eval(payload["cal"], eval_sets, args.target_fpr),
                "bigru": [],
            }
            seen = set()
            for k in args.ks:
                for mode in args.modes:
                    effective_k = 0 if mode in {"scalar", "scalar_layer"} else k
                    key = (mode, effective_k)
                    if key in seen:
                        continue
                    seen.add(key)
                    t0 = time.time()
                    result = train_and_evaluate(
                        train,
                        val,
                        eval_sets,
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
                    summary = result["eval"]["heldout_backdoor"]
                    print(
                        f"  {mode} k={effective_k}: heldout_auc={summary['auroc']:.4f} "
                        f"tpr={summary['tpr']:.3f} fpr={summary['fpr']:.3f}",
                        flush=True,
                    )
                    fold["bigru"].append(result)
            output["folds"][protocol][heldout] = fold

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"Saved results to {out_path}", flush=True)
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt-template", default="alpaca", choices=["alpaca", "qwen", "llama3"])
    parser.add_argument("--output-dir", type=Path, default=DPA_ROOT / "lcf_bigru_results" / "qwen2_5_7b_loao")
    parser.add_argument("--feature-cache", type=Path, default=DPA_ROOT / "lcf_bigru_results" / "qwen2_5_7b_loao" / "features_loao.npz")
    parser.add_argument("--stage", choices=["features", "train", "all"], default="all")
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-calibration", type=int, default=200)
    parser.add_argument("--bd-val-ratio", type=float, default=0.15)
    parser.add_argument("--jailbreak-train-ratio", type=float, default=0.70)
    parser.add_argument("--jailbreak-val-ratio", type=float, default=0.15)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--pca-dim-max", type=int, default=64)
    parser.add_argument("--protocols", nargs="+", default=["backdoor_only", "unified"], choices=["backdoor_only", "unified"])
    parser.add_argument("--ks", type=int, nargs="+", default=[64])
    parser.add_argument("--modes", nargs="+", default=["scalar_layer", "hybrid_layer"], choices=["scalar", "scalar_layer", "r", "hybrid", "hybrid_layer"])
    parser.add_argument("--bigru-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-fpr", type=float, default=0.10)
    parser.add_argument("--force-cpu-train", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in {"features", "all"}:
        if args.feature_cache.exists() and not args.overwrite_features:
            print(f"Feature cache exists; reusing {args.feature_cache}", flush=True)
        else:
            build_feature_cache(args)
    if args.stage in {"train", "all"}:
        run_training(args)


if __name__ == "__main__":
    main()
