"""Safe-only operating-point tests; no LLM, model download or invented results."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from dual_concept.backbone import FrozenBackboneConfig, StateCache
from dual_concept.concepts import ConceptBank
from dual_concept.evaluation import calibrate_threshold
from dual_concept.fusion import SparseFusionModel
from dual_concept.pipeline import Detector, MethodConfig
from dual_concept.data import sha256_file
from tools import run_dccf_diagnostics as diagnostics


def row(ident, kind, *, split="diagnose", family=""):
    return {"id": ident, "text": "offline fixture " + ident, "kind": kind,
            "label": int(kind != "safe"), "split": split, "attack_family": family,
            "exposure": "synthetic_test_only", "label_status": "synthetic_test_only"}


def dataset():
    return SimpleNamespace(splits={
        "calibrate": [row("c" + str(i), "safe", split="calibrate") for i in range(20)],
        "diagnose": [row("safe0", "safe"), row("safe1", "safe"),
                     row("j0", "jailbreak", family="GCG"),
                     row("j1", "jailbreak", family="nanoGCG"), row("h", "harmful")]})


def fixture_scorer(rows):
    values = {"safe0": -1., "safe1": 1., "j0": .5, "j1": 1.5, "h": 2.}
    return np.asarray([values.get(item["id"], 0.) for item in rows])


class DiagnosticsTests(unittest.TestCase):
    def test_all_thresholds_are_fixed_before_any_diagnostic_records_are_accessed(self):
        data = dataset()
        events = []
        count = [0]

        class GuardedSplits(dict):
            def __getitem__(self, name):
                if name == "diagnose":
                    self_test.assertEqual(count[0], 2, "diagnose accessed before both safe thresholds were frozen")
                    events.append("diagnose_records")
                return super().__getitem__(name)

        self_test = self
        data.splits = GuardedSplits(data.splits)
        def calibrated(scores, alpha):
            self.assertEqual(len(scores), 20)
            self.assertTrue(np.all(scores == 0))
            count[0] += 1
            events.append("calibrate")
            return calibrate_threshold(scores, alpha)
        def scored(rows):
            events.append("score_" + rows[0]["split"])
            return fixture_scorer(rows)
        with mock.patch.object(diagnostics, "calibrate_threshold", side_effect=calibrated):
            result = diagnostics.operating_points(data, scored, [.01, .05])
        self.assertEqual(events, ["score_calibrate", "calibrate", "calibrate", "diagnose_records", "score_diagnose"])
        self.assertFalse(result["independent_confirmation"])

    def test_arbitrary_diagnostic_score_changes_cannot_change_thresholds(self):
        data = dataset()
        first = diagnostics.operating_points(data, fixture_scorer)
        def changed(rows):
            if rows[0]["split"] == "diagnose":
                return np.linspace(-1e6, 1e6, len(rows))
            return fixture_scorer(rows)
        second = diagnostics.operating_points(data, changed)
        self.assertEqual([p["calibration"] for p in first["operating_points"]],
                         [p["calibration"] for p in second["operating_points"]])
        self.assertNotEqual(first["operating_points"][0]["overall"], second["operating_points"][0]["overall"])

    def test_family_auroc_uses_common_safe_controls_and_canonical_families(self):
        point = diagnostics.operating_points(dataset(), fixture_scorer)["operating_points"][0]
        family = point["by_attack_family"]["gcg"]
        self.assertEqual(set(point["by_attack_family"]), {"gcg"})
        self.assertEqual((family["tp"], family["fn"], family["n"]), (2, 0, 2))
        self.assertEqual(family["common_safe_n"], 2)
        self.assertEqual(family["auroc_vs_common_safe"], .75)
        self.assertEqual(point["safe_controls"], {"n": 2, "fp": 1, "tn": 1, "fpr": .5})

    def test_inclusive_decision_and_conservative_calibration_ties(self):
        data = dataset()
        threshold = float(np.nextafter(0., np.inf))
        def scored(rows):
            if rows[0]["split"] == "calibrate":
                return np.zeros(len(rows))
            return np.asarray([-1., 0., threshold, 0., 1.])
        result = diagnostics.operating_points(data, scored)
        for point in result["operating_points"]:
            self.assertEqual(point["calibration"]["threshold"], threshold)
            self.assertEqual(point["calibration"]["observed_fp"], 0)
            self.assertEqual(point["by_attack_family"]["gcg"]["tp"], 1)
            self.assertEqual(point["by_attack_family"]["gcg"]["fn"], 1)
        self.assertEqual(result["decision_rule"], ">=")

    def test_non_safe_calibration_and_invalid_alpha_are_rejected(self):
        data = dataset()
        data.splits["calibrate"][0]["kind"] = "harmful"
        with self.assertRaisesRegex(ValueError, "safe-only"):
            diagnostics.operating_points(data, fixture_scorer)
        for alphas in ([], [.01, .01], [-.01], [1.], [float("nan")]):
            with self.subTest(alphas=alphas), self.assertRaises(ValueError):
                diagnostics.fixed_alphas(alphas)

    def test_cached_scorer_is_bound_to_saved_protocol_and_rejects_changed_text(self):
        bank = ConceptBank(np.array([1]), np.array([[1., 0.]]), np.array([[0., 1.]]),
                           np.array([[0., 0.]]), np.array([[1., 0.]]))
        fusion = SparseFusionModel(np.zeros(2), np.ones(2), np.ones(2), 0., np.array([1, 1]), .01)
        protocol = {"candidate_layers": [1, 2], "hidden_size": 2}
        model = Detector(bank, [1], [1], fusion, 0., {"extraction_protocol": protocol})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = StateCache(root / "rows")
            record = row("cache", "safe")
            states = np.asarray([[2., 3.], [5., 7.]], dtype=np.float32)
            cache.save(record["id"], record["text"], protocol, states)
            scorer = diagnostics.CacheScorer(model, root, protocol)
            np.testing.assert_array_equal(scorer([record]), model.decision_function(states[None, :1]))
            self.assertEqual(len(scorer.fingerprints), 1)
            with self.assertRaisesRegex(ValueError, "protocol"):
                diagnostics.CacheScorer(model, root, {**protocol, "revision": "changed"})
            with self.assertRaisesRegex(ValueError, "text"):
                scorer([{**record, "text": "different request"}])

    def test_cli_failure_preserves_evidence_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root / "failed"
            argv = ["--model", str(root / "missing"), "--data", str(root / "missing"),
                    "--cache", str(root / "missing"), "--config", str(root / "missing"),
                    "--output", str(output)]
            self.assertEqual(diagnostics.main(argv), 1)
            saved = (output / "failure.json").read_bytes()
            self.assertEqual(json.loads((output / "run.json").read_text())["status"], "failed")
            self.assertFalse((output / "diagnostics.json").exists())
            self.assertEqual(diagnostics.main(argv), 2)
            self.assertEqual((output / "failure.json").read_bytes(), saved)

    def test_offline_cli_reads_bound_npz_artifacts_without_fitting_or_mutating_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data_root = root / "data"; cache_root = root / "cache"
            model_root = root / "frozen"; output = root / "diagnostics"
            (data_root / "development").mkdir(parents=True)
            (data_root / "pairs").mkdir()
            data = dataset()
            train = [row("tb", "safe", split="train"), row("th", "harmful", split="train"),
                     row("tj", "jailbreak", split="train", family="GCG")]
            for record in train:
                record["group_id"] = "training_goal"
            train[-1]["original_id"] = "th"
            for name in ("calibrate", "diagnose"):
                for record in data.splits[name]:
                    record["group_id"] = record["id"]
            for record in data.splits["diagnose"]:
                if record["kind"] in ("harmful", "jailbreak"):
                    record["group_id"] = "diagnostic_goal"
                if record["kind"] == "jailbreak":
                    record["original_id"] = "h"
            roles = {"train": train, "select": [], **data.splits}
            for name, records in roles.items():
                (data_root / "development" / (name + ".jsonl")).write_text(
                    "".join(json.dumps(record) + "\n" for record in records))
            for branch, negative, positive in (("risk", "tb", "th"), ("manipulation", "th", "tj")):
                (data_root / "pairs" / (branch + ".jsonl")).write_text(
                    json.dumps({"split": "train", "negative_id": negative, "positive_id": positive}) + "\n")
            hashes = {str(path.relative_to(data_root)): sha256_file(path) for path in data_root.rglob("*.jsonl")}
            (data_root / "manifest.json").write_text(json.dumps({"output_sha256": hashes}))
            configuration = {"backbone": asdict(FrozenBackboneConfig(candidate_layers=(1, 2))),
                             "method": asdict(MethodConfig())}
            config = root / "config.json"; config.write_text(json.dumps(configuration))
            protocol = {"candidate_layers": [1, 2], "hidden_size": 2,
                        "parameters": configuration["backbone"], "synthetic_test_only": True}
            cache = StateCache(cache_root / "rows")
            (cache_root / "protocol.json").write_text(json.dumps(protocol))
            for records in data.splits.values():
                for index, record in enumerate(records):
                    cache.save(record["id"], record["text"], protocol,
                               np.asarray([[index + 1., .2], [index + 2., .4]], dtype=np.float32))
            bank = ConceptBank(np.array([1]), np.array([[1., 0.]]), np.array([[0., 1.]]),
                               np.zeros((1, 2)), np.array([[1., 0.]]))
            fusion = SparseFusionModel(np.zeros(2), np.ones(2), np.ones(2), 0., np.array([1, 1]), .01)
            model = Detector(bank, [1], [1], fusion, 0., {"method_config": asdict(MethodConfig()),
                             "extraction_protocol": protocol, "outer_holdout": "gcg"})
            model.save(model_root)
            inputs = {path: sha256_file(path) for path in root.rglob("*") if path.is_file()}
            argv = ["--model", str(model_root), "--data", str(data_root), "--cache", str(cache_root),
                    "--config", str(config), "--output", str(output)]
            with mock.patch("dual_concept.pipeline.fit_detector", side_effect=AssertionError("No refitting")):
                self.assertEqual(diagnostics.main(argv), 0)
            report = json.loads((output / "diagnostics.json").read_text())
            self.assertEqual(report["outer_holdout"], "gcg")
            self.assertEqual(report["alphas_predeclared"], [.01, .05])
            self.assertFalse(report["independent_confirmation"])
            self.assertEqual(len(report["scored_cache_sha256"]), 25)
            for point in report["operating_points"]:
                self.assertEqual(point["calibration"]["n_safe"], 20)
                self.assertEqual(point["safe_controls"]["n"], 2)
                self.assertIsNotNone(point["held_out_family_metrics"]["auroc_vs_common_safe"])
            self.assertEqual(Detector.load(model_root).threshold, 0.)
            for path, digest in inputs.items():
                self.assertEqual(sha256_file(path), digest)


if __name__ == "__main__":
    unittest.main()
