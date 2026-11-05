"""One CLI for validation, frozen feature extraction, fitting and evaluation."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .backbone import FrozenBackbone, FrozenBackboneConfig, StateCache, protocol_hash
from .data import canonical_family, load_dataset, read_jsonl
from .evaluation import grouped_metrics
from .pipeline import Detector, MethodConfig, fit_detector, json_bytes


def _configuration(path: str) -> tuple[FrozenBackboneConfig, MethodConfig]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(value) - {"backbone", "method"}:
        raise ValueError("Configuration accepts backbone and method sections only")
    return FrozenBackboneConfig(**value.get("backbone", {})), MethodConfig.from_dict(value.get("method", {}))


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(json_bytes(value))


def _cache_protocol(cache_root: Path) -> dict[str, Any]:
    return json.loads((cache_root / "protocol.json").read_text(encoding="utf-8"))


def _load_states(cache_root: Path, records: list[dict[str, Any]], protocol: dict[str, Any]) -> np.ndarray:
    if not records:
        raise ValueError("The selected dataset role is empty")
    cache = StateCache(cache_root / "rows")
    values = []
    for row in records:
        state = cache.load(row["id"], row["text"], protocol,
                           expected_shape=(len(protocol["candidate_layers"]), protocol["hidden_size"]))
        if state is None:
            raise FileNotFoundError(f"Missing state for {row['id']}; run extract with the same data/protocol first")
        values.append(state)
    return np.stack(values)


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_dataset(args.data)
    filtered = dataset.without_attacks(args.holdout)
    return {"valid": True, "rows": {name: len(rows) for name, rows in filtered.splits.items()},
            "risk_pairs": len(filtered.risk_pairs), "manipulation_pairs": len(filtered.manipulation_pairs),
            "attack_families": sorted({canonical_family(row.get("attack_family")) for row in dataset.records
                                      if row["kind"] == "jailbreak"}),
            "outer_holdout": canonical_family(args.holdout) if args.holdout else None,
            "independent_confirmation_available": bool(dataset.splits.get("confirm"))}


def _extract(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_dataset(args.data)
    config, _ = _configuration(args.config)
    if args.model_path:
        config = replace(config, model_name_or_path=args.model_path)
    if args.device:
        config = replace(config, device=args.device)
    cache_root = Path(args.cache)
    cache_root.mkdir(parents=True, exist_ok=True)
    backbone = FrozenBackbone(config)
    backbone.extract([])
    protocol = backbone.get_protocol()
    protocol_path = cache_root / "protocol.json"
    if protocol_path.exists():
        if protocol_hash(_cache_protocol(cache_root)) != protocol_hash(protocol):
            raise ValueError("Cache directory belongs to another extraction protocol; choose a new directory")
    else:
        _write_new(protocol_path, protocol)
    cache = StateCache(cache_root / "rows")
    extracted = reused = 0
    names = args.splits or ["train", "select", "calibrate", "diagnose"]
    for name in names:
        if name not in dataset.splits:
            raise ValueError(f"Dataset has no {name} split")
        pending = []
        for row in dataset.splits[name]:
            if cache.load(row["id"], row["text"], protocol) is None:
                pending.append(row)
            else:
                reused += 1
            if len(pending) == config.batch_size:
                states = backbone.extract([item["text"] for item in pending])
                for item, state in zip(pending, states):
                    cache.save(item["id"], item["text"], protocol, state)
                extracted += len(pending)
                pending = []
        if pending:
            states = backbone.extract([item["text"] for item in pending])
            for item, state in zip(pending, states):
                cache.save(item["id"], item["text"], protocol, state)
            extracted += len(pending)
        print(f"{name}: state cache ready", file=sys.stderr, flush=True)
    return {"extracted": extracted, "reused": reused, "protocol_sha256": protocol_hash(protocol),
            "cache": str(cache_root.resolve()), "candidate_layers": protocol["candidate_layers"]}


def _train(args: argparse.Namespace) -> dict[str, Any]:
    if Path(args.output).exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    dataset = load_dataset(args.data)
    config, method = _configuration(args.config)
    cache_root = Path(args.cache)
    protocol = _cache_protocol(cache_root)
    if list(config.candidate_layers) != protocol["candidate_layers"]:
        raise ValueError("Method configuration and cached candidate-layer numbering differ")
    for name in ("dtype", "max_length"):
        if name in protocol.get("parameters", {}) and getattr(config, name) != protocol["parameters"][name]:
            raise ValueError(f"Configuration {name} disagrees with the cached extraction protocol")
    states = {split: _load_states(cache_root, dataset.splits[split], protocol) for split in ("train", "select", "calibrate")}
    model = fit_detector(dataset, states, protocol["candidate_layers"], method,
                         holdout=args.holdout, extraction_protocol=protocol)
    model.save(args.output)
    return {"model": str(Path(args.output).resolve()), "risk_layers": model.risk_layers,
            "manipulation_layers": model.manipulation_layers, "features": len(model.fusion.coef),
            "calibration": model.metadata["calibration"], "outer_holdout": model.metadata["outer_holdout"],
            "confirmation_claim": False}


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    model = Detector.load(args.model)
    dataset = load_dataset(args.data)
    if args.split not in dataset.splits:
        raise ValueError(f"No {args.split} dataset; the project ships exposed development diagnostics only")
    records = dataset.splits[args.split]
    cache_root = Path(args.cache)
    protocol = _cache_protocol(cache_root)
    if protocol_hash(protocol) != protocol_hash(model.metadata["extraction_protocol"]):
        raise ValueError("Evaluation cache extraction protocol differs from the fitted detector")
    states = _load_states(cache_root, records, protocol)
    positions = [protocol["candidate_layers"].index(int(layer)) for layer in model.bank.layer_ids]
    logits = model.decision_function(states[:, positions])
    result = {"split": args.split, "threshold": model.threshold, "decision_rule": ">=", "score_domain": "logit",
              "metrics": grouped_metrics(records, logits, model.threshold),
              "independent_confirmation": args.split == "confirm",
              "scores": [{"id": row["id"], "logit": float(score), "risk": bool(score >= model.threshold)}
                         for row, score in zip(records, logits)]}
    holdout = model.metadata.get("outer_holdout")
    if holdout:
        mask = np.asarray([row["kind"] == "jailbreak" and canonical_family(row.get("attack_family")) == holdout
                           for row in records])
        result["held_out_family"] = holdout
        if mask.any():
            result["held_out_family_metrics"] = grouped_metrics([row for row, flag in zip(records, mask) if flag],
                                                               logits[mask], model.threshold)
        else:
            result["held_out_family_metrics"] = None
    _write_new(Path(args.output), result)
    return {key: value for key, value in result.items() if key != "scores"}


def _predict(args: argparse.Namespace) -> dict[str, Any]:
    model = Detector.load(args.model)
    expected = model.metadata.get("extraction_protocol")
    if not expected:
        raise ValueError("This detector has no real backbone protocol and cannot accept raw-text inference")
    configuration = dict(expected["parameters"])
    configuration["candidate_layers"] = tuple(int(layer) for layer in model.bank.layer_ids)
    checkpoint = expected.get("checkpoint", {})
    if checkpoint.get("kind") == "hub_commit":
        configuration["revision"] = checkpoint["commit"]
    if args.model_path:
        configuration["model_name_or_path"] = args.model_path
    if args.device:
        configuration["device"] = args.device
    configuration["batch_size"] = args.batch_size
    records = read_jsonl(Path(args.input)) if args.input else [{"id": "request", "text": args.text}]
    if not records:
        raise ValueError("Prediction input must not be empty")
    for index, row in enumerate(records):
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"Prediction row {index} needs nonempty text")
    backbone = FrozenBackbone(FrozenBackboneConfig(**configuration))
    backbone.extract([])
    actual = backbone.get_protocol()
    from .backbone import assert_compatible_protocol
    assert_compatible_protocol(expected, actual)
    logits = model.decision_function(backbone.extract([row["text"] for row in records]))
    result = {"threshold": model.threshold, "score_domain": "logit", "decision_rule": ">=",
              "predictions": [{"id": row.get("id", str(index)), "logit": float(score),
                               "risk": bool(score >= model.threshold)}
                              for index, (row, score) in enumerate(zip(records, logits))]}
    if args.output:
        _write_new(Path(args.output), result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dual-concept complementary cross-layer risk detector")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Validate data hashes, pairs and split isolation")
    validate.add_argument("--data", default="data")
    validate.add_argument("--holdout")
    validate.set_defaults(run=_validate)
    extract = sub.add_parser("extract", help="Extract frozen last-valid-token states without generating answers")
    extract.add_argument("--data", default="data")
    extract.add_argument("--config", default="configs/qwen2_5_7b.json")
    extract.add_argument("--cache", default="artifacts/states")
    extract.add_argument("--splits", nargs="+", choices=("train", "select", "calibrate", "diagnose", "confirm"))
    extract.add_argument("--model-path")
    extract.add_argument("--device")
    extract.set_defaults(run=_extract)
    train = sub.add_parser("train", help="Fit concepts, grouped layer selection, sparse head and safety threshold")
    train.add_argument("--data", default="data")
    train.add_argument("--config", default="configs/qwen2_5_7b.json")
    train.add_argument("--cache", default="artifacts/states")
    train.add_argument("--output", required=True)
    train.add_argument("--holdout")
    train.set_defaults(run=_train)
    evaluate = sub.add_parser("evaluate", help="Evaluate a frozen model; never recalibrate")
    evaluate.add_argument("--data", default="data")
    evaluate.add_argument("--cache", default="artifacts/states")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--split", choices=("diagnose", "confirm"), default="diagnose")
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(run=_evaluate)
    predict = sub.add_parser("predict", help="Single-forward raw-text inference with a frozen detector")
    predict.add_argument("--model", required=True)
    source = predict.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--input", help="JSONL records with text and optional id")
    predict.add_argument("--model-path")
    predict.add_argument("--device")
    predict.add_argument("--batch-size", type=int, default=1)
    predict.add_argument("--output")
    predict.set_defaults(run=_predict)
    args = parser.parse_args(argv)
    try:
        result = args.run(args)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json_bytes(result).decode(), end="")
    return 0
