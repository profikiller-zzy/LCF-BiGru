#!/usr/bin/env python3
"""Frozen DCCF operating points calibrated on safe material only.

No detector fitting, model download, text generation, or diagnostic tuning occurs.
The shipped diagnose material remains exposed development data with candidate
labels. Every attack group uses the same model's diagnostic safety controls.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Callable, Sequence

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from dual_concept.backbone import FrozenBackboneConfig, StateCache, protocol_hash
from dual_concept.data import Dataset, canonical_family, load_dataset, sha256_file
from dual_concept.evaluation import binary_metrics, calibrate_threshold
from dual_concept.pipeline import Detector, MethodConfig, json_bytes


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def fixed_alphas(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not np.isfinite(value) or not 0 <= value < 1 for value in result):
        raise ValueError("Predeclared calibration targets must lie in [0,1)")
    if len(set(result)) != len(result):
        raise ValueError("Calibration targets must not contain duplicates")
    return result


def checked_scores(records: Sequence[dict], score: Callable) -> np.ndarray:
    result = np.asarray(score(records), dtype=np.float64)
    if result.shape != (len(records),) or not np.isfinite(result).all():
        raise ValueError("Frozen scorer returned malformed or nonfinite logits")
    return result


def operating_points(dataset: Dataset, score: Callable,
                     alphas: Sequence[float] = (.01, .05)) -> dict:
    """Freeze ALL thresholds before accessing any diagnostic records/scores."""
    targets = fixed_alphas(alphas)
    calibration = dataset.splits["calibrate"]
    if not calibration or any(row["kind"] != "safe" or row["label"] != 0 for row in calibration):
        raise ValueError("Operating-point calibration requires safe-only records")
    calibration_scores = checked_scores(calibration, score)
    frozen = [calibrate_threshold(calibration_scores, alpha) for alpha in targets]

    # Do not move this access above the safe-only threshold construction.
    records = dataset.splits["diagnose"]
    if not records:
        raise ValueError("No diagnostic records are available")
    logits = checked_scores(records, score)
    labels = np.asarray([row["label"] for row in records])
    safe = np.asarray([row["kind"] == "safe" for row in records])
    if not safe.any():
        raise ValueError("Diagnostic AUROC needs common safe controls")
    families = sorted({canonical_family(row.get("attack_family")) for row in records
                       if row["kind"] == "jailbreak"})
    points = []
    for calibrated in frozen:
        threshold = calibrated["threshold"]
        safety = binary_metrics(labels[safe], logits[safe], threshold)
        point = {"calibration": calibrated, "overall": binary_metrics(labels, logits, threshold),
                 "safe_controls": {"n": int(safe.sum()), "fp": safety["fp"],
                                   "tn": safety["tn"], "fpr": safety["fpr"]},
                 "by_kind": {}, "by_attack_family": {}}

        def positive_group(positive):
            count = int(positive.sum())
            if not count:
                return None
            mask = safe | positive
            paired = binary_metrics(labels[mask], logits[mask], threshold)
            return {"n": count, "tp": paired["tp"], "fn": paired["fn"],
                    "tpr": paired["tpr"], "auroc_vs_common_safe": paired["auroc"],
                    "common_safe_n": int(safe.sum())}

        for kind in ("harmful", "jailbreak"):
            point["by_kind"][kind] = positive_group(np.asarray([row["kind"] == kind for row in records]))
        for family in families:
            point["by_attack_family"][family] = positive_group(np.asarray([
                row["kind"] == "jailbreak" and canonical_family(row.get("attack_family")) == family
                for row in records]))
        points.append(point)
    return {"alphas_predeclared": list(targets), "operating_points": points,
            "calibration_scores": [{"id": row["id"], "logit": float(value)}
                                   for row, value in zip(calibration, calibration_scores)],
            "diagnostic_scores": [{"id": row["id"], "logit": float(value)}
                                  for row, value in zip(records, logits)],
            "exposure": sorted({row["exposure"] for row in records}),
            "label_status": sorted({row["label_status"] for row in records}),
            "decision_rule": ">=", "score_domain": "logit",
            "threshold_policy": "all thresholds fixed from calibrate safe logits before diagnostic scoring",
            "auroc_policy": "positive group versus this model's same diagnostic safe controls; no positive-only AUROC",
            "fpr_policy": "calibration target is an empirical budget, not a diagnostic or future-data FPR guarantee",
            "independent_confirmation": False, "asr_reported": False}


class CacheScorer:
    def __init__(self, model: Detector, cache: Path, protocol: dict):
        expected = model.metadata.get("extraction_protocol")
        if not expected or protocol_hash(expected) != protocol_hash(protocol):
            raise ValueError("Cache extraction protocol differs from the trained detector")
        self.model, self.protocol = model, protocol
        if not (cache / "rows").is_dir():
            raise FileNotFoundError("Complete state-cache rows directory is absent")
        self.cache = StateCache(cache / "rows")
        layers = list(protocol["candidate_layers"])
        self.positions = [layers.index(int(layer)) for layer in model.bank.layer_ids]
        self.fingerprints: dict[str, str] = {}

    def __call__(self, records):
        selected = []
        for row in records:
            state = self.cache.load(row["id"], row["text"], self.protocol)
            if state is None:
                raise FileNotFoundError("Missing immutable cached state for " + row["id"])
            path = self.cache.path_for(row["id"])
            self.fingerprints[str(path)] = sha256_file(path)
            selected.append(state[self.positions])
        return self.model.decision_function(np.stack(selected))


def run(args, output: Path) -> dict:
    dataset = load_dataset(args.data)
    model = Detector.load(args.model)
    configuration = json.loads(args.config.read_text(encoding="utf-8"))
    if asdict(MethodConfig.from_dict(configuration.get("method", {}))) != model.metadata.get("method_config"):
        raise ValueError("Configuration method differs from the trained detector")
    protocol = json.loads((args.cache / "protocol.json").read_text(encoding="utf-8"))
    configured = asdict(FrozenBackboneConfig(**configuration.get("backbone", {})))
    runtime = {"model_name_or_path", "revision", "device", "batch_size", "local_files_only"}
    saved = asdict(FrozenBackboneConfig(**protocol["parameters"])) if "parameters" in protocol else None
    if saved is None or json_bytes({key: value for key, value in configured.items() if key not in runtime}) != json_bytes(
            {key: value for key, value in saved.items() if key not in runtime}):
        raise ValueError("Configuration feature semantics differ from the cached extraction protocol")
    inputs = [args.config, args.data / "manifest.json", args.model / "model.json",
              args.model / "parameters.npz", args.cache / "protocol.json", Path(__file__).resolve()]
    inputs += [args.data / name for name in dataset.manifest.get("output_sha256", dataset.manifest.get("outputs_sha256", {}))]
    fingerprints = {str(path): sha256_file(path) for path in inputs}
    plan = {"alphas": list(fixed_alphas(args.alphas)), "input_sha256": fingerprints,
            "model": str(args.model), "data": str(args.data), "cache": str(args.cache),
            "config": str(args.config), "output": str(output), "fit_parameters": False,
            "diagnostic_tuning": False, "split": "diagnose", "python": sys.version,
            "numpy_version": np.__version__,
            "scoring_source_sha256": {path.name: sha256_file(path) for path in (REPOSITORY / "dual_concept").glob("*.py")}}
    write_json(output / "plan.json", plan)
    scorer = CacheScorer(model, args.cache, protocol)
    result = operating_points(dataset, scorer, args.alphas)
    result.update(model=str(args.model), model_sha256=fingerprints[str(args.model / "model.json")],
                  extraction_protocol_sha256=protocol_hash(protocol),
                  outer_holdout=model.metadata.get("outer_holdout"),
                  original_model_threshold=model.threshold,
                  original_model_calibration=model.metadata.get("calibration"),
                  plan_sha256=hashlib.sha256(json_bytes(plan)).hexdigest())
    heldout = result["outer_holdout"]
    if heldout:
        for point in result["operating_points"]:
            point["held_out_family"] = heldout
            point["held_out_family_metrics"] = point["by_attack_family"].get(heldout)
    elif args.include_geometry:
        result["reference_model_diagnostics"] = {
            "source": "unchanged saved training concepts and internal select diagnostics",
            "concepts": model.metadata.get("concepts"),
            "risk_selection": model.metadata.get("risk_selection"),
            "manipulation_selection": model.metadata.get("manipulation_selection"),
            "geometry_is_not_semantic_separation": True}
    all_fingerprints = {**fingerprints, **scorer.fingerprints}
    for name, expected in all_fingerprints.items():
        if sha256_file(Path(name)) != expected:
            raise ValueError("An input or cached state changed during diagnostics: " + name)
    result["scored_cache_sha256"] = scorer.fingerprints
    write_json(output / "diagnostics.json", result)
    return {"status": "succeeded", "operating_points": len(result["operating_points"]),
            "outer_holdout": heldout, "independent_confirmation": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "data", "cache", "config", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--alphas", type=float, nargs="+", default=[.01, .05])
    parser.add_argument("--include-geometry", action="store_true",
                        help="Reference saved full-model geometry; does not establish semantic separation")
    args = parser.parse_args(argv)
    for name in ("model", "data", "cache", "config", "output"):
        setattr(args, name, getattr(args, name).resolve())
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(json.dumps({"status": "setup_failed", "error": str(error)}), file=sys.stderr)
        return 2
    report = {"status": "running", "started_at": now(), "generates_answers": False,
              "fits_detector": False, "independent_confirmation": False}
    write_json(args.output / "run.json", report)
    code = 0
    try:
        report.update(run(args, args.output))
    except (Exception, KeyboardInterrupt) as error:
        failure = {"error_type": type(error).__name__, "error": str(error),
                   "traceback": traceback.format_exc()}
        write_json(args.output / "failure.json", failure)
        report.update(status="failed", failure=failure)
        code = 130 if isinstance(error, KeyboardInterrupt) else 1
    report["finished_at"] = now()
    write_json(args.output / "run.json", report)
    print(json.dumps({key: report.get(key) for key in ("status", "operating_points", "outer_holdout")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
