"""Low-dimensional numerical end-to-end checks, without loading an LLM."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from dual_concept import cli
from dual_concept.concepts import fit_concepts
from dual_concept.data import Dataset, load_dataset
from dual_concept.pipeline import Detector, fit_detector
from dual_concept.selection import select_layers


LAYERS = [1, 7, 27]
FAMILIES = {"GCG": -0.5, "PAIR": 0.0, "AutoDAN": 0.5}
PROTOCOL = {"candidate_layers": LAYERS, "hidden_size": 4, "synthetic_test": True}


def synthetic_dataset():
    """Risk is coordinate 0; manipulation occupies coordinates 1 and 2.

    Each family has a different manipulation direction. Coordinate 3 is a
    small goal-specific nuisance shared by matched requests, so paired changes
    isolate the two concepts while inputs themselves are not identical.
    """
    splits = {name: [] for name in ("train", "select", "calibrate", "diagnose", "confirm")}
    states = {name: [] for name in splits}
    risk_pairs, manipulation_pairs = [], []
    marker = 0

    def add(split, kind, identifier, group, value, family="", original=None):
        row = {"id": identifier, "text": f"Synthetic request {identifier}", "group_id": group,
               "kind": kind, "label": int(kind != "safe"), "split": split,
               "attack_family": family, "label_status": "synthetic_test_only",
               "exposure": "independent_confirmation" if split == "confirm" else "historical_development",
               "source": {"packaging": "plain"}}
        if original is not None:
            row["original_id"] = original
        splits[split].append(row)
        states[split].append(np.stack([np.asarray(value) * scale for scale in (1.0, 1.3, .8)]))

    for split, count in (("train", 3), ("select", 2), ("diagnose", 1), ("confirm", 1)):
        for family, tilt in FAMILIES.items():
            for task in range(count):
                marker += 1
                group = f"{split}:{family}:{task}"
                safe_id, harmful_id, attack_id = [f"{group}:{kind}" for kind in ("b", "h", "j")]
                nuisance = marker * .001
                add(split, "safe", safe_id, group, [0., 0., 0., nuisance])
                add(split, "harmful", harmful_id, group, [3., 0., 0., nuisance])
                add(split, "jailbreak", attack_id, group, [3., 4., 4. * tilt, nuisance], family, harmful_id)
                if split == "train":
                    risk_pairs.append({"negative_id": safe_id, "positive_id": harmful_id, "split": split})
                manipulation_pairs.append({"negative_id": harmful_id, "positive_id": attack_id, "split": split})
    for number in range(20):
        add("calibrate", "safe", f"calibration:{number}", f"calibration:{number}",
            [0., 0., 0., number * .0001])
    data = Dataset(splits, risk_pairs, manipulation_pairs, {"output_sha256": {}})
    data.validate()
    return data, {name: np.asarray(value, dtype=np.float64) for name, value in states.items()}


class PipelineTests(unittest.TestCase):
    def assert_same_predictive_parameters(self, first, second):
        self.assertEqual(first.risk_layers, second.risk_layers)
        self.assertEqual(first.manipulation_layers, second.manipulation_layers)
        for name, array in first.bank.arrays().items():
            np.testing.assert_array_equal(array, second.bank.arrays()[name])
        for name, array in first.fusion.arrays().items():
            np.testing.assert_array_equal(array, second.fusion.arrays()[name])

    def test_numerical_training_finds_both_concepts_and_detects_new_requests(self):
        data, states = synthetic_dataset()
        model = fit_detector(data, states, LAYERS, extraction_protocol=PROTOCOL)
        self.assertTrue(model.fusion.diagnostics["converged"])
        self.assertEqual(len(model.risk_layers), 2)
        self.assertEqual(len(model.manipulation_layers), 2)
        np.testing.assert_allclose(model.bank.risk_direction[:, 0], 1., atol=1e-12)
        positions = [LAYERS.index(int(layer)) for layer in model.bank.layer_ids]
        predicted = model.predict(states["confirm"][:, positions])
        expected = np.array([row["label"] for row in data.splits["confirm"]], dtype=bool)
        np.testing.assert_array_equal(predicted, expected)
        self.assertEqual(model.metadata["calibration"]["n_safe"], 20)
        self.assertLessEqual(model.metadata["calibration"]["observed_fp"], 1)
        self.assertFalse(model.metadata["confirmation_claim"])

    def test_outer_family_holdout_and_internal_banks_are_actually_refitted(self):
        data, states = synthetic_dataset()
        calls = []

        def inspect_bank(*args, **kwargs):
            values, _, manipulation_pairs = args[:3]
            positives = np.asarray(manipulation_pairs)[:, 1]
            # The first physical layer is unscaled. Family-specific tilt is
            # the third coordinate divided by manipulation amplitude four.
            tilts = set(np.round(values[positives, 0, 2] / 4., 12).tolist())
            bank = fit_concepts(*args, **kwargs)
            calls.append((tilts, bank))
            return bank

        with mock.patch("dual_concept.pipeline.fit_concepts", side_effect=inspect_bank):
            model = fit_detector(data, states, LAYERS, holdout="nanoGCG")
        self.assertEqual(model.metadata["outer_holdout"], "gcg")
        self.assertEqual([fold["held_out_family"] for fold in model.metadata["internal_folds"]],
                         ["autodan", "pair_tap"])
        self.assertEqual(len(calls), 3)  # Two internal refits and one final training bank.
        self.assertEqual([call[0] for call in calls], [{0.0}, {0.5}, {0.0, 0.5}])
        self.assertGreater(np.linalg.norm(calls[0][1].manipulation_direction -
                                         calls[1][1].manipulation_direction), .1)
        self.assertEqual(model.metadata["fit_roles"]["train"]["n"], 24)
        self.assertEqual(model.metadata["fit_roles"]["select"]["n"], 16)
        self.assertEqual(len(data.splits["diagnose"]), 9)

    def test_calibration_changes_only_threshold_and_diagnostic_arrays_are_never_read(self):
        data, states = synthetic_dataset()
        baseline = fit_detector(data, states, LAYERS)

        class ForbiddenDiagnosticArray:
            def __array__(self, *args, **kwargs):
                raise AssertionError("Diagnostic states must not be read while fitting")

        changed = dict(states)
        changed["diagnose"] = ForbiddenDiagnosticArray()
        changed["confirm"] = ForbiddenDiagnosticArray()
        extreme = np.stack([np.array([1e9, 1e9, 0., 0.]) * scale for scale in (1.0, 1.3, .8)])
        changed["calibrate"] = np.broadcast_to(extreme, states["calibrate"].shape).copy()
        stressed = fit_detector(data, changed, LAYERS)
        self.assert_same_predictive_parameters(baseline, stressed)
        self.assertNotEqual(baseline.threshold, stressed.threshold)
        self.assertEqual(baseline.metadata["risk_selection"], stressed.metadata["risk_selection"])
        self.assertEqual(baseline.metadata["manipulation_selection"], stressed.metadata["manipulation_selection"])

    def test_migrated_schema_preserves_safe_packaging_groups_for_selection(self):
        data = load_dataset(Path(__file__).resolve().parents[1] / "data")
        states = {}
        for split, rows in data.splits.items():
            values = []
            for row in rows:
                # Exercise the real migrated metadata with controlled states;
                # this checks grouped selection, not real detection performance.
                nuisance = sum(row["group_id"].encode("utf-8")) * 1e-7
                vector = np.array([3. * (row["kind"] != "safe"),
                                   4. * (row["kind"] == "jailbreak"), 0., nuisance])
                values.append(np.stack([vector * scale for scale in (1., 1.3, .8)]))
            states[split] = np.asarray(values)
        observed = []

        def inspect_folds(folds, *args, **kwargs):
            for fold in folds:
                groups, counts = np.unique(fold["safe_groups"][fold["safe_mask"]], return_counts=True)
                observed.append(dict(zip(groups.tolist(), counts.tolist())))
            return select_layers(folds, *args, **kwargs)

        with mock.patch("dual_concept.pipeline.select_layers", side_effect=inspect_folds):
            fit_detector(data, states, LAYERS)
        self.assertTrue(observed)
        self.assertTrue(all(groups == {"source_original": 159, "plain": 40, "roleplay": 40, "email": 40}
                            for groups in observed))

    def test_save_load_preserves_logits_and_rejects_numeric_tampering(self):
        data, states = synthetic_dataset()
        model = fit_detector(data, states, LAYERS)
        positions = [LAYERS.index(int(layer)) for layer in model.bank.layer_ids]
        queries = states["diagnose"][:, positions]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "frozen"
            model.save(target)
            restored = Detector.load(target)
            np.testing.assert_array_equal(restored.decision_function(queries), model.decision_function(queries))
            np.testing.assert_array_equal(restored.predict(queries), model.predict(queries))
            self.assertEqual(restored.threshold, model.threshold)
            with self.assertRaises(FileExistsError):
                model.save(target)
            with (target / "parameters.npz").open("ab") as stream:
                stream.write(b"tampered numeric artifact")
            with self.assertRaisesRegex(ValueError, "parameters have changed"):
                Detector.load(target)

    def test_saved_layer_order_mismatch_is_rejected(self):
        data, states = synthetic_dataset()
        model = fit_detector(data, states, LAYERS)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "frozen"
            model.save(target)
            metadata_path = target / "model.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["risk_layers"] = metadata["risk_layers"][::-1]
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "layer ordering"):
                Detector.load(target)

    def test_independent_cli_evaluation_uses_frozen_threshold_without_recalibration(self):
        data, states = synthetic_dataset()
        model = fit_detector(data, states, LAYERS, extraction_protocol=PROTOCOL)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model.save(root / "frozen")
            args = Namespace(model=str(root / "frozen"), data="unused", cache="unused", split="confirm",
                             output=str(root / "evaluation.json"))
            with mock.patch("dual_concept.cli.load_dataset", return_value=data), \
                 mock.patch("dual_concept.cli._cache_protocol", return_value=PROTOCOL), \
                 mock.patch("dual_concept.cli._load_states", return_value=states["confirm"]), \
                 mock.patch("dual_concept.pipeline.calibrate_threshold",
                            side_effect=AssertionError("Evaluation must not recalibrate")), \
                 mock.patch("dual_concept.evaluation.calibrate_threshold",
                            side_effect=AssertionError("Evaluation must not recalibrate")):
                result = cli._evaluate(args)
            output = json.loads((root / "evaluation.json").read_text())
            self.assertTrue(result["independent_confirmation"])
            self.assertEqual(output["threshold"], model.threshold)
            self.assertEqual(output["metrics"]["overall"]["tpr"], 1.)
            self.assertEqual(output["metrics"]["overall"]["fpr"], 0.)
            self.assertEqual(Detector.load(root / "frozen").threshold, model.threshold)


if __name__ == "__main__":
    unittest.main()
