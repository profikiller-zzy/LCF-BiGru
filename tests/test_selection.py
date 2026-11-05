import unittest

import numpy as np

from dual_concept.selection import SelectionConfig, binary_auroc, select_layers


class SelectionTests(unittest.TestCase):
    def fold(self, group="attack"):
        scores = np.array([[1., .4, 1.], [1., .4, 1.], [.4, 1., .4], [.4, 1., .4],
                           [.1, .1, .8], [.1, .1, .8], [.1, .1, .1], [.1, .1, .1]])
        return {"scores": scores, "labels": np.array([1] * 4 + [0] * 4),
                "safe_mask": np.array([False] * 4 + [True] * 4),
                "attack_groups": np.array([group] * 4 + [""] * 4),
                "thresholds": np.array([.5, .5, .5])}

    def test_auroc_ties_and_auxiliary_safe_controls(self):
        self.assertEqual(binary_auroc(np.array([0, 1]), np.array([.2, .2])), .5)
        fold = self.fold()
        # Manipulation negatives can be harmful originals. Auxiliary safe rows
        # remain outside branch AUROC while still constraining added false flags.
        fold["safe_mask"][:] = False
        fold["safe_mask"][6:] = True
        fold["labels"][6:] = -1
        selected = select_layers([fold], [1, 2, 3])
        self.assertEqual(selected.layers, [1, 2])

    def test_complement_recovers_misses_and_rejects_extra_false_positives(self):
        result = select_layers([self.fold("a"), self.fold("b")], [3, 7, 8])
        self.assertEqual(result.layers, [3, 7])
        self.assertEqual(result.layers_by_budget, {2: [3, 7]})
        self.assertEqual(result.diagnostics["final_coverage"]["worst_tpr"], 1.)
        self.assertEqual(result.diagnostics["final_coverage"]["macro_safe_fpr"], 0.)
        self.assertEqual(result.diagnostics["stop_reason"], "no_positive_incremental_utility")
        self.assertGreater(result.diagnostics["trace"][1]["mean_tpr_gain"], 0)

    def test_worst_attack_group_filter_does_not_hide_poor_generalization(self):
        first, second = self.fold("a"), self.fold("b")
        second["scores"][:4, 1:] = 0  # Only layer 1 transfers to group b.
        with self.assertRaisesRegex(ValueError, "Only 1 layers"):
            select_layers([first, second], [1, 2, 3])

    def test_thresholds_must_be_supplied_and_safe_labels_are_distinct(self):
        fold = self.fold()
        del fold["thresholds"]
        with self.assertRaises(KeyError):
            select_layers([fold], [1, 2, 3])
        fold = self.fold()
        fold["safe_mask"][0] = True
        with self.assertRaises(ValueError):
            select_layers([fold], [1, 2, 3])
        with self.assertRaises(ValueError):
            SelectionConfig(min_layers=1)


if __name__ == "__main__":
    unittest.main()
