"""Readout-head controls, split boundaries and real failure preservation."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from dual_concept.backbone import StateCache
from dual_concept.concepts import ConceptBank
from dual_concept.data import canonical_family
from dual_concept.fusion import SparseFusionModel
from dual_concept.pipeline import Detector, MethodConfig, fit_detector
from tools.run_dccf_ablations import (
    AblationRunner, CONTROLS, FAMILIES, bank_identity, hash_json, refit_head,
    role_identity, sha256_file, write_json,
)
from test_pipeline import LAYERS, PROTOCOL, synthetic_dataset


def noisy_readout_fixture():
    """Overlapping noisy labels keep the unpenalized logistic optimum finite."""
    random = np.random.RandomState(413)
    train_h = random.normal(size=(240, 2, 3))
    calibration_h = random.normal(size=(60, 2, 3))
    labels = random.binomial(1, .5 + .15 * np.tanh(train_h[:, 0, 0]))
    rows = [{"id": f"train:{i}", "text": f"Synthetic {i}", "split": "train",
             "kind": "harmful" if label else "safe", "label": int(label),
             "attack_family": ""} for i, label in enumerate(labels)]
    calibration = [{"id": f"calibrate:{i}", "text": f"Synthetic calibration {i}",
                    "split": "calibrate", "kind": "safe", "label": 0,
                    "attack_family": ""} for i in range(len(calibration_h))]
    bank = ConceptBank(np.array([1, 2]), np.array([[1., 0., 0.]] * 2),
                       np.array([[0., 1., 0.]] * 2), np.zeros((2, 3)),
                       np.array([[.4, 0., 0.]] * 2))
    fusion = SparseFusionModel(np.zeros(4), np.ones(4), np.zeros(4), 0.,
                               np.array([1, 2, 1, 2]), .01, {"converged": True})
    metadata = {"outer_holdout": "gcg", "method_config": asdict(MethodConfig()),
                "fit_roles": {"train": role_identity(rows), "calibrate": role_identity(calibration)},
                "extraction_protocol": {"candidate_layers": [1, 2], "hidden_size": 3, "synthetic_test": True},
                "risk_selection": {"synthetic_test": "frozen risk choices"},
                "manipulation_selection": {"synthetic_test": "frozen manipulation choices"}}
    return Detector(bank, [1, 2], [1, 2], fusion, 0., metadata), rows, train_h, calibration, calibration_h


class ReadoutControlTests(unittest.TestCase):
    def test_real_three_control_fits_keep_bank_and_layers_and_use_only_requested_penalty(self):
        baseline, rows, train, calibration, safe = noisy_readout_fixture()
        baseline_bank = bank_identity(baseline.bank)
        baseline_metadata = copy.deepcopy(baseline.metadata)
        for control in CONTROLS:
            with self.subTest(control=control), \
                 mock.patch("dual_concept.pipeline.fit_concepts", side_effect=AssertionError("No concept refit")), \
                 mock.patch("dual_concept.pipeline.select_layers", side_effect=AssertionError("No layer reselection")):
                model = refit_head(baseline, rows, train, calibration, safe, control)
                self.assertEqual(bank_identity(model.bank), baseline_bank)
                self.assertEqual(model.risk_layers, baseline.risk_layers)
                self.assertEqual(model.manipulation_layers, baseline.manipulation_layers)
                self.assertTrue(model.fusion.diagnostics["converged"])
                self.assertLessEqual(model.metadata["calibration"]["observed_fp"], 3)
                self.assertEqual(model.metadata["readout_ablation"]["stage"], "post_selection_readout_head")
                self.assertFalse(model.metadata["readout_ablation"]["layers_reselected"])
                expected = copy.deepcopy(baseline_metadata["method_config"])
                if control == "no_group_penalty":
                    expected["fusion"]["regularization"] = 0.0
                self.assertEqual(model.metadata["method_config"], expected)
        self.assertEqual(baseline.metadata, baseline_metadata)
        np.testing.assert_array_equal(baseline.fusion.coef, np.zeros(4))

    def test_removed_branch_is_exact_zero_and_arbitrary_readout_changes_do_not_change_logits(self):
        baseline, rows, train, calibration, safe = noisy_readout_fixture()
        random = np.random.RandomState(19)
        risk, manipulation = random.normal(size=(2, 5, 2))
        for control in CONTROLS[:2]:
            model = refit_head(baseline, rows, train, calibration, safe, control)
            removed = model.metadata["readout_ablation"]["removed_branch_columns"]
            np.testing.assert_array_equal(model.fusion.coef[removed], 0.)
            np.testing.assert_array_equal(model.fusion.mean[removed], 0.)
            np.testing.assert_array_equal(model.fusion.scale[removed], 1.)
            with tempfile.TemporaryDirectory() as folder:
                model.save(Path(folder) / "model")
                restored = Detector.load(Path(folder) / "model")
                changed = (risk, manipulation * 1e9 + 15) if control == "risk_readout_only" else (risk * -1e8, manipulation)
                with mock.patch("dual_concept.pipeline.concept_scores", return_value=(risk, manipulation)):
                    logits = restored.decision_function(np.zeros((5, 2, 3)))
                with mock.patch("dual_concept.pipeline.concept_scores", return_value=changed):
                    np.testing.assert_array_equal(logits, restored.decision_function(np.zeros((5, 2, 3))))

    def test_calibration_changes_threshold_but_never_scaler_or_head(self):
        baseline, rows, train, calibration, safe = noisy_readout_fixture()
        first = refit_head(baseline, rows, train, calibration, safe, "risk_readout_only")
        altered = np.broadcast_to(np.array([[[1., 0., 0.], [1., 0., 0.]]]), safe.shape).copy()
        second = refit_head(baseline, rows, train, calibration, altered, "risk_readout_only")
        for name in ("mean", "scale", "coef"):
            np.testing.assert_array_equal(getattr(first.fusion, name), getattr(second.fusion, name))
        self.assertEqual(first.fusion.intercept, second.fusion.intercept)
        self.assertNotEqual(first.threshold, second.threshold)

    def test_heldout_wrong_order_and_non_safe_calibration_are_rejected(self):
        baseline, rows, train, calibration, safe = noisy_readout_fixture()
        with self.assertRaisesRegex(ValueError, "identity"):
            refit_head(baseline, rows[::-1], train[::-1], calibration, safe, "risk_readout_only")
        attacks = copy.deepcopy(rows)
        attacks[0].update(kind="jailbreak", label=1, attack_family="nanoGCG")
        with self.assertRaisesRegex(ValueError, "held-out"):
            refit_head(baseline, attacks, train, calibration, safe, "risk_readout_only")
        unsafe = copy.deepcopy(calibration)
        unsafe[0].update(kind="harmful", label=1)
        with self.assertRaisesRegex(ValueError, "safety-only"):
            refit_head(baseline, rows, train, unsafe, safe, "risk_readout_only")

    def test_nonconvergence_is_not_frozen_and_is_not_retried_with_other_hyperparameters(self):
        baseline, rows, train, calibration, safe = noisy_readout_fixture()
        nonconverged = copy.deepcopy(baseline.fusion)
        nonconverged.diagnostics["converged"] = False
        with mock.patch("tools.run_dccf_ablations.fit_sparse_fusion", return_value=nonconverged) as fit:
            with self.assertRaisesRegex(RuntimeError, "did not converge"):
                refit_head(baseline, rows, train, calibration, safe, "no_group_penalty")
            self.assertEqual(fit.call_count, 1)
            self.assertEqual(fit.call_args.kwargs["config"].regularization, 0.)
            self.assertEqual(fit.call_args.kwargs["config"].max_iter, 10000)


class AblationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data, self.cache, self.baseline = [self.root / name for name in ("data", "cache", "baseline")]
        with mock.patch("test_pipeline.FAMILIES", {"GCG": -.5, "PAIR": 0., "AutoDAN": .5,
                                                    "DRATTACK": -.25, "SAA": .25}):
            dataset, states = synthetic_dataset()
        (self.data / "development").mkdir(parents=True)
        (self.data / "pairs").mkdir()
        self.cache.mkdir()
        write_json(self.data / "manifest.json", {"output_sha256": {}})
        for split, records in dataset.splits.items():
            (self.data / "development" / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
        for branch, pairs in (("risk", dataset.risk_pairs), ("manipulation", dataset.manipulation_pairs)):
            (self.data / "pairs" / f"{branch}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in pairs))
        write_json(self.cache / "protocol.json", PROTOCOL)
        cache = StateCache(self.cache / "rows")
        for split, records in dataset.splits.items():
            for row, value in zip(records, states[split]):
                cache.save(row["id"], row["text"], PROTOCOL, value)
        self.config = self.root / "config.json"
        write_json(self.config, {"backbone": {"candidate_layers": LAYERS}, "method": {}})
        self.baseline.mkdir()
        checkpoint = {"steps": {"extract": {"status": "succeeded"}}}
        self.models = {}
        for family in FAMILIES:
            model = fit_detector(dataset, states, LAYERS, holdout=None if family == "full" else family,
                                 extraction_protocol=PROTOCOL)
            directory = self.baseline / family / "model"
            model.save(directory)
            self.models[family] = directory
            checkpoint["steps"][family + ".train"] = {"status": "succeeded", "artifact": str(directory),
                "artifact_sha256": {str(directory / name): sha256_file(directory / name)
                                    for name in ("model.json", "parameters.npz")}}
            checkpoint["steps"][family + ".evaluate"] = {"status": "succeeded"}
        specification = {"data": str(self.data), "cache": str(self.cache), "config": str(self.config),
                         "input_sha256": {str(self.config): sha256_file(self.config)}}
        plan_hash = hash_json(specification)
        checkpoint["plan_sha256"] = plan_hash
        write_json(self.baseline / "plan.json", {"plan_sha256": plan_hash, "specification": specification})
        write_json(self.baseline / "checkpoint.json", checkpoint)

    def runner(self, output="controls"):
        return AblationRunner(self.baseline, self.data, self.cache, self.root / output)

    def test_eighteen_cases_freeze_before_diagnostics_and_keep_real_fit_failures(self):
        runner = self.runner()
        original_states = runner._states
        original_refit = refit_head
        events = []
        def load(records, *, role):
            events.append(role)
            if role == "diagnose":
                self.assertEqual(len(runner.runs), 18)
                self.assertTrue(all(item["fit_status"] in ("succeeded", "failed") for item in runner.runs.values()))
            return original_states(records, role=role)
        def refit(baseline, rows, train, calibration, safe, control, **kwargs):
            if control == "no_group_penalty":
                raise RuntimeError("Synthetic test-only nonconvergence")
            if baseline.metadata["outer_holdout"]:
                self.assertFalse(any(row["kind"] == "jailbreak" and
                                     canonical_family(row["attack_family"]) == baseline.metadata["outer_holdout"]
                                     for row in rows))
            return original_refit(baseline, rows, train, calibration, safe, control, **kwargs)
        with mock.patch.object(runner, "_states", side_effect=load), \
             mock.patch("tools.run_dccf_ablations.refit_head", side_effect=refit):
            result = runner.run()
        self.assertEqual(len(result["runs"]), 18)
        self.assertEqual(len(result["failed_runs"]), 6)
        self.assertEqual(events[-1], "diagnose")
        self.assertEqual(events.count("diagnose"), 1)
        for name, item in result["runs"].items():
            directory = Path(item["directory"])
            if item["control"] == "no_group_penalty":
                self.assertEqual(item["status"], "failed")
                self.assertTrue((directory / "failure.json").is_file())
                self.assertFalse((directory / "model").exists())
                self.assertFalse((directory / "evaluation.json").exists())
            else:
                self.assertEqual(item["status"], "succeeded", name)
                evaluation = json.loads((directory / "evaluation.json").read_text())
                self.assertFalse(evaluation["independent_confirmation"])
                self.assertEqual(evaluation["decision_rule"], ">=")
                model = Detector.load(directory / "model")
                configuration = json.loads((directory / "config.json").read_text())
                self.assertEqual(configuration["method"], model.metadata["method_config"])
                self.assertIn("train_cache_sha256", model.metadata["readout_ablation"]["source"])
        with self.assertRaises(FileExistsError):
            self.runner().run()

    def test_all_fit_failures_never_read_diagnostic_states(self):
        runner = self.runner()
        original = runner._states
        def load(records, *, role):
            if role == "diagnose":
                raise AssertionError("No frozen models exist to evaluate")
            return original(records, role=role)
        with mock.patch.object(runner, "_states", side_effect=load), \
             mock.patch("tools.run_dccf_ablations.refit_head", side_effect=RuntimeError("Synthetic all-fit failure")):
            result = runner.run()
        self.assertEqual(len(result["failed_runs"]), 18)
        self.assertFalse((runner.output / "cache-input-sha256.json").exists())

    def test_tampered_baseline_is_rejected_before_head_fitting(self):
        with (self.models["full"] / "parameters.npz").open("ab") as stream:
            stream.write(b"tampered")
        with mock.patch("tools.run_dccf_ablations.refit_head") as refit:
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                self.runner().run()
            refit.assert_not_called()

    def test_diagnostic_cache_failure_is_recorded_as_global_invalidity(self):
        runner = self.runner()
        original = runner._states
        def load(records, *, role):
            if role == "diagnose":
                raise FileNotFoundError("Synthetic diagnostic cache absent")
            return original(records, role=role)
        with mock.patch.object(runner, "_states", side_effect=load):
            with self.assertRaisesRegex(FileNotFoundError, "cache absent"):
                runner.run()
        result = json.loads((runner.output / "summary.json").read_text())
        self.assertFalse(result["valid"])
        self.assertIn("__global__", result["failed_runs"])
        self.assertTrue(result["fit_stage_complete"])
        self.assertTrue((runner.output / "failure.json").is_file())

    def test_final_cache_hash_failure_never_leaves_an_all_success_summary(self):
        runner = self.runner()
        original = runner._states
        def load(records, *, role):
            values, fingerprints = original(records, role=role)
            if role == "diagnose":
                with Path(next(iter(fingerprints))).open("ab") as stream:
                    stream.write(b"synthetic tampering after diagnostic load")
            return values, fingerprints
        with mock.patch.object(runner, "_states", side_effect=load):
            with self.assertRaisesRegex(ValueError, "cached state changed"):
                runner.run()
        result = json.loads((runner.output / "summary.json").read_text())
        self.assertFalse(result["valid"])
        self.assertEqual(result["status"], "failed")
        self.assertIn("__global__", result["failed_runs"])


if __name__ == "__main__":
    unittest.main()
