#!/usr/bin/env python3
"""Refit three predefined post-selection readout controls on a frozen baseline.

Concepts, anchors and both independently selected layer lists remain frozen.
These are post-selection readout ablations, not single-concept layer reselection.
No backbone is loaded and no request/label is synthesized. All requested heads
are fitted, safely calibrated and saved before diagnostic state arrays are read.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from dual_concept.backbone import StateCache, protocol_hash
from dual_concept.concepts import ConceptBank, concept_scores
from dual_concept.data import canonical_family, load_dataset, sha256_file as _sha256_file
from dual_concept.evaluation import calibrate_threshold, grouped_metrics
from dual_concept.fusion import build_feature_matrix, fit_sparse_fusion
from dual_concept.pipeline import Detector, MethodConfig, json_bytes


FAMILIES = ("full", "autodan", "drattack", "gcg", "pair_tap", "saa")
CONTROLS = ("risk_readout_only", "manipulation_readout_only", "no_group_penalty")
READOUT_NOTE = (
    "Post-selection readout-head ablation: baseline concepts, anchors and both "
    "independently selected layer lists are frozen; this is not single-concept reselection.")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    return _sha256_file(Path(path))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def hash_json(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def role_identity(rows):
    return {"n": len(rows), "ids_sha256": hash_json([row["id"] for row in rows])}


def bank_identity(bank):
    return hash_json({name: {"shape": list(array.shape), "dtype": str(array.dtype),
                            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
                      for name, array in bank.arrays().items()})


def refit_head(baseline, train_rows, train_states, calibration_rows,
               calibration_states, control, *, source=None):
    """Fit only a readout head/scaler on supplied outer-filtered training rows.

    Arrays have the baseline bank's physical layer order. No select, diagnose or
    confirm arrays are accepted. Baseline role identities bind row ordering and
    exclude accidental use of an outer-held-out attack during refitting.
    """
    if control not in CONTROLS:
        raise ValueError("Unknown predefined readout control")
    holdout = baseline.metadata.get("outer_holdout")
    for name, rows in (("train", train_rows), ("calibrate", calibration_rows)):
        if not rows or any(row["split"] != name for row in rows):
            raise ValueError("Readout fitting requires the exact train/calibrate roles")
        if role_identity(rows) != baseline.metadata.get("fit_roles", {}).get(name):
            raise ValueError("Readout rows differ from the baseline's frozen " + name + " identity")
    if holdout and any(row["kind"] == "jailbreak" and canonical_family(row.get("attack_family")) == holdout
                       for row in train_rows):
        raise ValueError("Outer-held-out attacks cannot fit an ablation head")
    if any(row["kind"] != "safe" or row["label"] != 0 for row in calibration_rows):
        raise ValueError("Readout threshold calibration is safety-only")
    if {row["id"] for row in train_rows} & {row["id"] for row in calibration_rows}:
        raise ValueError("Training and safety calibration identities overlap")
    method = MethodConfig.from_dict(baseline.metadata["method_config"])
    if control == "no_group_penalty":
        method = replace(method, fusion=replace(method.fusion, regularization=0.0))
    train_states, calibration_states = np.asarray(train_states), np.asarray(calibration_states)
    expected = (len(baseline.bank.layer_ids), baseline.bank.risk_direction.shape[1])
    for name, rows, states in (("train", train_rows, train_states),
                               ("calibrate", calibration_rows, calibration_states)):
        if states.shape != (len(rows), *expected) or not np.isfinite(states).all():
            raise ValueError("Malformed or misaligned " + name + " readout states")
    risk, manipulation = concept_scores(train_states, baseline.bank)
    features, layers, branches = build_feature_matrix(
        risk, manipulation, baseline.risk_layers, baseline.manipulation_layers,
        baseline.bank.layer_ids)
    removed = (branches == "manipulation" if control == "risk_readout_only" else
               branches == "risk" if control == "manipulation_readout_only" else
               np.zeros(len(branches), dtype=bool))
    # Zero only the removed readout columns. The standard scaler consequently
    # stores mean=0, scale=1 there and optimization must retain exact zero weights.
    fitting_features = features.copy()
    fitting_features[:, removed] = 0.0
    fusion = fit_sparse_fusion(fitting_features,
                               np.asarray([row["label"] for row in train_rows]),
                               layers, config=method.fusion)
    if not fusion.diagnostics.get("converged"):
        raise RuntimeError("Ablation fusion did not converge; fixed hyperparameters were not changed")
    if np.any(fusion.coef[removed] != 0.0):
        raise RuntimeError("Removed branch acquired nonzero coefficients; refusing to freeze a head")
    if np.any(fusion.mean[removed] != 0.0) or np.any(fusion.scale[removed] != 1.0):
        raise RuntimeError("Removed branch scaler disagrees with zero-column fitting")
    bank = ConceptBank(**baseline.bank.arrays(), diagnostics=copy.deepcopy(baseline.bank.diagnostics))
    model = Detector(bank, list(baseline.risk_layers), list(baseline.manipulation_layers), fusion, 0.0)
    calibration = calibrate_threshold(model.decision_function(calibration_states), method.target_fpr)
    model.threshold = calibration["threshold"]
    model.metadata = copy.deepcopy(baseline.metadata)
    model.metadata.update(method_config=asdict(method), fusion=copy.deepcopy(fusion.diagnostics),
                          calibration=calibration, confirmation_claim=False,
                          label_status="candidate_not_independently_validated")
    model.metadata["readout_ablation"] = {
        "control": control, "stage": "post_selection_readout_head", "note": READOUT_NOTE,
        "concepts_refitted": False, "layers_reselected": False,
        "risk_layers": list(baseline.risk_layers),
        "manipulation_layers": list(baseline.manipulation_layers),
        "baseline_bank_sha256": bank_identity(baseline.bank),
        "removed_branch_columns": np.flatnonzero(removed).tolist(),
        "removed_coefficients_exact_zero": True,
        "fit_roles": {"train": role_identity(train_rows), "calibrate": role_identity(calibration_rows)},
        "source": copy.deepcopy(source or {}),
        "diagnose_used_for_fitting": False,
    }
    if bank_identity(model.bank) != bank_identity(baseline.bank):
        raise RuntimeError("Ablation unexpectedly changed the frozen concept bank")
    return model


class AblationRunner:
    def __init__(self, baseline_output, data, cache, output, families=None):
        self.baseline_output, self.data, self.cache, self.output = [Path(p).resolve()
            for p in (baseline_output, data, cache, output)]
        self.families = list(FAMILIES if families is None else families)
        if not self.families or len(set(self.families)) != len(self.families) or set(self.families) - set(FAMILIES):
            raise ValueError("families must be unique predefined outer cases")
        self.cache_hashes = {}
        self.state_memory = {}
        self.runs = {}

    def _assert_inputs(self):
        for path, expected in self.input_hashes.items():
            if sha256_file(path) != expected:
                raise ValueError("Frozen ablation input changed: " + path)

    def _states(self, records, *, role):
        values, fingerprints = [], {}
        cache = StateCache(self.cache / "rows")
        for row in records:
            if row["split"] != role:
                raise ValueError("Requested cached rows crossed their declared role")
            path = cache.path_for(row["id"])
            key = str(path)
            current = sha256_file(path)
            if key in self.cache_hashes and self.cache_hashes[key] != current:
                raise ValueError("Cached state changed during ablations: " + key)
            self.cache_hashes[key] = current
            fingerprints[key] = current
            if row["id"] not in self.state_memory:
                value = cache.load(row["id"], row["text"], self.protocol,
                                   expected_shape=(len(self.protocol["candidate_layers"]), self.protocol["hidden_size"]))
                if value is None:
                    raise FileNotFoundError("Missing cached state: " + row["id"])
                value.setflags(write=False)
                self.state_memory[row["id"]] = value
            values.append(self.state_memory[row["id"]])
        if not values:
            raise ValueError("The requested cache role is empty")
        return np.stack(values), fingerprints

    def _initialize(self):
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError("Choose a new/empty ablation output directory; previous evidence is preserved")
        self.output.mkdir(parents=True, exist_ok=True)
        self._output_accepted = True
        baseline_plan = read(self.baseline_output / "plan.json")
        specification = baseline_plan["specification"]
        if baseline_plan["plan_sha256"] != hash_json(specification):
            raise ValueError("Baseline plan hash does not match its specification")
        checkpoint = read(self.baseline_output / "checkpoint.json")
        if checkpoint["plan_sha256"] != baseline_plan["plan_sha256"]:
            raise ValueError("Baseline checkpoint/plan identities disagree")
        if Path(specification["data"]).resolve() != self.data or Path(specification["cache"]).resolve() != self.cache:
            raise ValueError("Ablations must use the baseline data and cached-state locations")
        self.dataset = load_dataset(self.data)
        self.protocol = read(self.cache / "protocol.json")
        self.source_configuration = read(specification["config"])
        source_method = asdict(MethodConfig.from_dict(self.source_configuration.get("method", {})))
        self.input_hashes = {str(self.baseline_output / name): sha256_file(self.baseline_output / name)
                             for name in ("plan.json", "checkpoint.json")}
        self.input_hashes.update({str(self.cache / "protocol.json"): sha256_file(self.cache / "protocol.json"),
                                 str(Path(specification["config"])): sha256_file(specification["config"]),
                                 str(self.data / "manifest.json"): sha256_file(self.data / "manifest.json"),
                                 str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve())})
        # Baseline manifest/code/config hashes are checked as bytes. Diagnostic
        # feature arrays and evaluation results are not opened during fitting.
        for path, expected in specification["input_sha256"].items():
            if sha256_file(path) != expected:
                raise ValueError("Baseline source/input fingerprint changed: " + path)
            self.input_hashes[path] = expected
        extraction = checkpoint["steps"].get("extract", {})
        if extraction.get("status") != "succeeded":
            raise ValueError("A completed first extraction is required before ablations")
        self.baselines = {}
        for family in self.families:
            train = checkpoint["steps"].get(family + ".train", {})
            evaluation = checkpoint["steps"].get(family + ".evaluate", {})
            if train.get("status") != "succeeded" or evaluation.get("status") != "succeeded":
                raise ValueError("Baseline train/evaluation is incomplete: " + family)
            model_path = Path(train["artifact"]).resolve()
            for path, expected in train["artifact_sha256"].items():
                if sha256_file(path) != expected:
                    raise ValueError("Baseline trained artifact changed: " + path)
                self.input_hashes[path] = expected
            baseline = Detector.load(model_path)
            expected_holdout = None if family == "full" else family
            if baseline.metadata.get("outer_holdout") != expected_holdout:
                raise ValueError("Baseline outer holdout disagrees with case " + family)
            if baseline.metadata.get("method_config") != source_method:
                raise ValueError("Baseline method hyperparameters disagree with the source configuration")
            if protocol_hash(baseline.metadata["extraction_protocol"]) != protocol_hash(self.protocol):
                raise ValueError("Baseline model and readout cache protocols differ")
            self.baselines[family] = (baseline, model_path)
        self.specification = {"schema_version": 1, "baseline_output": str(self.baseline_output),
            "baseline_plan_sha256": baseline_plan["plan_sha256"], "data": str(self.data), "cache": str(self.cache),
            "output": str(self.output), "families": self.families, "controls": list(CONTROLS),
            "input_sha256": self.input_hashes, "numpy_version": np.__version__,
            "protocol_sha256": protocol_hash(self.protocol), "note": READOUT_NOTE,
            "fit_policy": "outer-filtered train only; safety-only calibration; freeze all heads before reading diagnostic states",
            "hyperparameter_policy": "baseline method hyperparameters; no_group_penalty changes regularization to zero only",
            "independent_confirmation": False, "label_status": "candidate_not_independently_validated"}
        self.plan_hash = hash_json(self.specification)
        write_json(self.output / "plan.json", {"created_at": now(), "plan_sha256": self.plan_hash,
                                               "specification": self.specification})

    def _flush(self, *, fitted=False):
        failed = [name for name, item in self.runs.items() if item.get("status") == "failed"]
        global_failure = getattr(self, "global_failure", None)
        if global_failure:
            failed.append("__global__")
        status = ("failed" if global_failure else "running" if not fitted or
                  any(item["status"] == "running" for item in self.runs.values()) else
                  "completed_with_failures" if failed else "succeeded")
        result = {"schema_version": 1, "updated_at": now(), "plan_sha256": getattr(self, "plan_hash", None),
                  "status": status, "valid": global_failure is None,
                  "fit_stage_complete": fitted, "note": READOUT_NOTE, "runs": self.runs,
                  "failed_runs": failed, "global_failure": global_failure,
                  "independent_confirmation": False,
                  "label_status": "candidate_not_independently_validated"}
        write_json(self.output / "checkpoint.json", result)
        write_json(self.output / "summary.json", result)
        return result

    def run(self):
        try:
            return self._run()
        except (Exception, KeyboardInterrupt) as error:
            # Integrity failures can occur after individual heads succeeded.
            # Persist that the overall snapshot is invalid rather than leaving
            # a success-looking summary behind a nonzero process exit.
            if getattr(self, "_output_accepted", False):
                self.global_failure = {"error_type": type(error).__name__, "error": str(error),
                                       "traceback": traceback.format_exc(), "failed_at": now()}
                write_json(self.output / "failure.json", self.global_failure)
                self._flush(fitted=getattr(self, "fit_stage_complete", False))
            raise

    def _run(self):
        self._initialize()
        for family in self.families:
            baseline, baseline_path = self.baselines[family]
            filtered = self.dataset.without_attacks(baseline.metadata.get("outer_holdout"))
            train_rows, calibration_rows = filtered.splits["train"], filtered.splits["calibrate"]
            train_h, train_hashes = self._states(train_rows, role="train")
            calibration_h, calibration_hashes = self._states(calibration_rows, role="calibrate")
            positions = [self.protocol["candidate_layers"].index(int(layer)) for layer in baseline.bank.layer_ids]
            source = {"baseline_model": str(baseline_path),
                      "baseline_model_sha256": sha256_file(baseline_path / "model.json"),
                      "baseline_parameters_sha256": sha256_file(baseline_path / "parameters.npz"),
                      "cache_protocol_sha256": protocol_hash(self.protocol),
                      "train_cache_sha256": train_hashes, "calibration_cache_sha256": calibration_hashes}
            for control in CONTROLS:
                key = family + "/" + control
                directory = self.output / "runs" / family / control
                directory.mkdir(parents=True)
                item = {"outer_case": family, "outer_holdout": baseline.metadata.get("outer_holdout"),
                        "control": control, "status": "running", "fit_status": "running", "evaluate_status": "not_run",
                        "started_at": now(), "directory": str(directory)}
                write_json(directory / "source.json", source)
                item["source_sha256"] = sha256_file(directory / "source.json")
                self.runs[key] = item
                self._flush()
                print(f"[{now()}] {key} fit", flush=True)
                try:
                    self._assert_inputs()
                    model = refit_head(baseline, train_rows, train_h[:, positions], calibration_rows,
                                       calibration_h[:, positions], control, source=source)
                    self._assert_inputs()
                    model.save(directory / "model")
                    configuration = copy.deepcopy(self.source_configuration)
                    configuration["method"] = model.metadata["method_config"]
                    write_json(directory / "config.json", configuration)
                    item.update(fit_status="succeeded", model=str(directory / "model"),
                                config=str(directory / "config.json"), frozen_at=now(),
                                model_sha256=sha256_file(directory / "model/model.json"),
                                parameters_sha256=sha256_file(directory / "model/parameters.npz"))
                except (OSError, ValueError, RuntimeError) as error:
                    failure = {"stage": "fit", "error_type": type(error).__name__,
                               "error": str(error), "traceback": traceback.format_exc()}
                    write_json(directory / "failure.json", failure)
                    item.update(status="failed", fit_status="failed", failure=failure)
                finally:
                    write_json(directory / "step.json", item)
                    self._flush()
        # This phase boundary is intentional: no diagnostic feature array has
        # been loaded while any head configuration remained to be fitted.
        self._assert_inputs()
        self.fit_stage_complete = True
        self._flush(fitted=True)
        if not any(item["fit_status"] == "succeeded" for item in self.runs.values()):
            return self._flush(fitted=True)
        diagnostic_rows = self.dataset.splits["diagnose"]
        diagnostic_h, diagnostic_hashes = self._states(diagnostic_rows, role="diagnose")
        write_json(self.output / "cache-input-sha256.json", {
            "fit_cache_sha256": {path: digest for path, digest in self.cache_hashes.items() if path not in diagnostic_hashes},
            "diagnostic_cache_sha256": diagnostic_hashes})
        for key, item in self.runs.items():
            if item["fit_status"] != "succeeded":
                continue
            directory = Path(item["directory"])
            print(f"[{now()}] {key} frozen evaluate", flush=True)
            try:
                self._assert_inputs()
                if sha256_file(Path(item["model"]) / "model.json") != item["model_sha256"]:
                    raise ValueError("A frozen ablation model changed before evaluation")
                model = Detector.load(item["model"])
                positions = [self.protocol["candidate_layers"].index(int(layer)) for layer in model.bank.layer_ids]
                logits = model.decision_function(diagnostic_h[:, positions])
                result = {"split": "diagnose", "threshold": model.threshold, "decision_rule": ">=",
                          "score_domain": "logit", "independent_confirmation": False,
                          "readout_ablation": item["control"], "ablation_note": READOUT_NOTE,
                          "metrics": grouped_metrics(diagnostic_rows, logits, model.threshold),
                          "scores": [{"id": row["id"], "logit": float(value), "risk": bool(value >= model.threshold)}
                                     for row, value in zip(diagnostic_rows, logits)]}
                holdout = model.metadata.get("outer_holdout")
                if holdout:
                    mask = np.asarray([row["kind"] == "jailbreak" and canonical_family(row.get("attack_family")) == holdout
                                       for row in diagnostic_rows])
                    result["held_out_family"] = holdout
                    result["held_out_family_metrics"] = (grouped_metrics([row for row, flag in zip(diagnostic_rows, mask) if flag],
                                                                          logits[mask], model.threshold) if mask.any() else None)
                self._assert_inputs()
                write_json(directory / "evaluation.json", result)
                item.update(status="succeeded", evaluate_status="succeeded", evaluation=str(directory / "evaluation.json"),
                            evaluation_sha256=sha256_file(directory / "evaluation.json"), finished_at=now())
            except (OSError, ValueError, RuntimeError) as error:
                failure = {"stage": "evaluate", "error_type": type(error).__name__,
                           "error": str(error), "traceback": traceback.format_exc()}
                write_json(directory / "failure.json", failure)
                item.update(status="failed", evaluate_status="failed", failure=failure)
            finally:
                write_json(directory / "step.json", item)
                self._flush(fitted=True)
        for path, expected in self.cache_hashes.items():
            if sha256_file(path) != expected:
                raise ValueError("A readout cached state changed before completion: " + path)
        return self._flush(fitted=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-output", "data", "cache", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    args = parser.parse_args(argv)
    try:
        result = AblationRunner(args.baseline_output, args.data, args.cache, args.output, args.families).run()
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print("Readout ablation error: " + str(error), file=sys.stderr)
        return 2
    print(json_bytes(result).decode(), end="")
    return 1 if result["failed_runs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
