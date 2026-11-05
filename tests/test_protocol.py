"""Split, pairing, outer-holdout and calibration protocol regression checks."""
import unittest

import numpy as np

from dual_concept.data import Dataset, canonical_family, normalized_text
from dual_concept.evaluation import binary_metrics, calibrate_threshold
from dual_concept.selection import binary_auroc


def record(identifier, split, kind, group, family="", original=None):
    row = {"id": identifier, "text": f"Unique request for {identifier}", "split": split,
           "group_id": group, "kind": kind, "label": 0 if kind == "safe" else 1,
           "attack_family": family, "exposure": "historical_development",
           "label_status": "source_derived_candidate"}
    if original is not None:
        row["original_id"] = original
    return row


def protocol_dataset():
    train = [record("tb", "train", "safe", "training-a"),
             record("th", "train", "harmful", "training-a"),
             record("tj", "train", "jailbreak", "training-a", "nanoGCG", "th"),
             record("tb2", "train", "safe", "training-b"),
             record("th2", "train", "harmful", "training-b"),
             record("tj2", "train", "jailbreak", "training-b", "PAIR", "th2")]
    selection = [record("sb", "select", "safe", "selection-a"),
                 record("sh", "select", "harmful", "selection-a"),
                 record("sj", "select", "jailbreak", "selection-a", "GCG", "sh")]
    diagnostic = [record("db", "diagnose", "safe", "diagnostic-a"),
                  record("dh", "diagnose", "harmful", "diagnostic-a"),
                  record("dj", "diagnose", "jailbreak", "diagnostic-a", "GCG", "dh")]
    return Dataset(
        {"train": train, "select": selection,
         "calibrate": [record("cb", "calibrate", "safe", "calibration-a")],
         "diagnose": diagnostic},
        [{"negative_id": "tb", "positive_id": "th", "split": "train"},
         {"negative_id": "tb2", "positive_id": "th2", "split": "train"}],
        [{"negative_id": "th", "positive_id": "tj", "split": "train"},
         {"negative_id": "th2", "positive_id": "tj2", "split": "train"},
         {"negative_id": "sh", "positive_id": "sj", "split": "select"},
         {"negative_id": "dh", "positive_id": "dj", "split": "diagnose"}],
        {},
    )


class DatasetProtocolTests(unittest.TestCase):
    def test_valid_development_dataset(self):
        protocol_dataset().validate()

    def test_same_goal_component_cannot_cross_splits(self):
        data = protocol_dataset()
        data.splits["select"][0]["group_id"] = "training-a"
        with self.assertRaisesRegex(ValueError, "Cross-split goal component"):
            data.validate()

    def test_nfkc_case_and_whitespace_text_leakage_is_rejected(self):
        self.assertEqual(normalized_text("  ＡＢＣ\tX  "), "abc x")
        data = protocol_dataset()
        data.splits["train"][0]["text"] = "  ＡＢＣ\tX  "
        data.splits["select"][0]["text"] = "abc  x"
        with self.assertRaisesRegex(ValueError, "Cross-split normalized text"):
            data.validate()

    def test_risk_pair_requires_safe_to_original_harmful_endpoints(self):
        data = protocol_dataset()
        data.risk_pairs[0]["positive_id"] = "tj"
        with self.assertRaisesRegex(ValueError, "Wrong endpoint kinds"):
            data.validate()

    def test_risk_pairs_are_training_only_and_cannot_cross_splits(self):
        data = protocol_dataset()
        data.risk_pairs[0] = {"negative_id": "sb", "positive_id": "sh", "split": "select"}
        with self.assertRaisesRegex(ValueError, "training-only"):
            data.validate()
        data = protocol_dataset()
        data.risk_pairs[0]["negative_id"] = "sb"
        with self.assertRaisesRegex(ValueError, "inside one split"):
            data.validate()

    def test_pair_group_components_must_agree(self):
        for branch in ("risk", "manipulation"):
            with self.subTest(branch=branch):
                data = protocol_dataset()
                row_id = "tb" if branch == "risk" else "tj"
                next(row for row in data.records if row["id"] == row_id)["group_id"] = "unrelated-task"
                with self.assertRaises(ValueError):
                    data.validate()

    def test_manipulation_requires_declared_same_split_original(self):
        data = protocol_dataset()
        data.splits["train"][2]["original_id"] = "th2"
        with self.assertRaisesRegex(ValueError, "original_id"):
            data.validate()
        data = protocol_dataset()
        data.manipulation_pairs = [pair for pair in data.manipulation_pairs if pair["positive_id"] != "sj"]
        data.splits["select"][2]["original_id"] = "th"
        with self.assertRaisesRegex(ValueError, "same-split"):
            data.validate()

    def test_outer_holdout_removes_fit_pairs_but_preserves_diagnostics(self):
        data = protocol_dataset()
        data.validate()
        heldout = data.without_attacks("GCG/nanoGCG")
        ids = {row["id"] for row in heldout.records}
        self.assertNotIn("tj", ids)
        self.assertNotIn("sj", ids)
        self.assertIn("dj", ids)
        self.assertIn("th", ids)
        self.assertIn("sh", ids)
        self.assertIn("tj2", ids)
        self.assertEqual(heldout.manifest["outer_holdout"], "gcg")
        self.assertEqual({pair["positive_id"] for pair in heldout.manipulation_pairs}, {"tj2", "dj"})
        self.assertEqual(len(heldout.risk_pairs), 2)
        self.assertEqual(len(data.splits["train"]), 6)  # No mutation of the source dataset.

    def test_outer_holdout_keeps_originals_with_source_family_metadata(self):
        data = protocol_dataset()
        next(row for row in data.records if row["id"] == "th")["attack_family"] = "GCG"
        data.validate()
        heldout = data.without_attacks("nanoGCG")
        self.assertIn("th", {row["id"] for row in heldout.splits["train"]})

    def test_deployment_calibration_accepts_only_safe_rows(self):
        data = protocol_dataset()
        data.splits["calibrate"] = [record("ch", "calibrate", "harmful", "calibration-harm")]
        with self.assertRaisesRegex(ValueError, "safety-only"):
            data.validate()

    def test_confirmation_requires_explicit_independence(self):
        data = protocol_dataset()
        data.splits["confirm"] = [record("new", "confirm", "safe", "confirmation-a")]
        with self.assertRaisesRegex(ValueError, "independent_confirmation"):
            data.validate()
        data.splits["confirm"][0]["exposure"] = "independent_confirmation"
        data.validate()


class CalibrationProtocolTests(unittest.TestCase):
    def test_small_sample_floor_budget_and_greater_equal_rule(self):
        scores = np.arange(19, dtype=float)
        result = calibrate_threshold(scores, .05)
        self.assertEqual(result["fp_budget"], 0)
        self.assertEqual(result["observed_fp"], 0)
        self.assertEqual(result["threshold"], np.nextafter(18., np.inf))
        scores = np.arange(20, dtype=float)
        result = calibrate_threshold(scores, .05)
        self.assertEqual(result["fp_budget"], 1)
        self.assertEqual(result["observed_fp"], 1)
        self.assertEqual(int((scores >= result["threshold"]).sum()), 1)
        metrics = binary_metrics([0, 1], [result["threshold"], result["threshold"]], result["threshold"])
        self.assertEqual((metrics["fp"], metrics["tp"]), (1, 1))

    def test_boundary_ties_are_excluded_conservatively(self):
        scores = np.array([10., 10.] + list(range(18)))
        result = calibrate_threshold(scores, .05)
        self.assertLessEqual(result["observed_fp"], 1)
        # A top-score tie straddling the available budget cannot be split.
        result = calibrate_threshold(np.array([5., 5.] + [0.] * 18), .05)
        self.assertEqual(result["observed_fp"], 0)
        self.assertGreater(result["threshold"], 5.)

    def test_calibration_bad_inputs_and_unrepresentable_boundary_are_rejected(self):
        for scores, target in [([], .05), ([np.nan], .05), ([0], 1), ([0], -.1),
                               ([np.finfo(float).max], 0)]:
            with self.subTest(scores=scores, target=target):
                with np.errstate(over="ignore"):
                    with self.assertRaises(ValueError):
                        calibrate_threshold(scores, target)

    def test_auroc_uses_average_ranks_for_ties_and_single_class_is_undefined(self):
        labels, scores = np.array([0, 1, 0, 1]), np.array([.2, .5, .5, .9])
        self.assertEqual(binary_auroc(labels, scores), .875)
        self.assertEqual(binary_metrics(labels, scores, .5)["auroc"], .875)
        self.assertIsNone(binary_metrics([0, 0], [.1, .2], .5)["auroc"])
        self.assertIsNone(binary_metrics([0, 0], [.1, .2], .5)["tpr"])
        with self.assertRaises(ValueError):
            binary_auroc(np.zeros(2), np.ones(2))

    def test_family_aliases_are_withheld_as_one_family(self):
        self.assertEqual(canonical_family("nano_GCG"), "gcg")
        self.assertEqual(canonical_family("TAP"), canonical_family("PAIR"))


if __name__ == "__main__":
    unittest.main()
