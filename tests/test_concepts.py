import unittest

import numpy as np

from dual_concept.concepts import ConceptBank, anchored_cosine, concept_scores, fit_concepts, rank_one_direction


class ConceptTests(unittest.TestCase):
    def test_rank_one_matches_svd_for_both_matrix_shapes(self):
        rng = np.random.default_rng(51)
        for shape in [(4, 9), (9, 4), (1, 4)]:
            delta = rng.normal(size=shape) + 0.3
            direction, detail = rank_one_direction(delta)
            expected = np.linalg.svd(delta, full_matrices=False)[2][0]
            self.assertAlmostEqual(abs(float(direction @ expected)), 1.0, places=10)
            self.assertGreaterEqual(float(direction @ delta.mean(axis=0)), -1e-12)
            self.assertLess(detail["eigen_residual"], 1e-10)

    def test_zero_matrix_rejected_and_zero_mean_sign_deterministic(self):
        with self.assertRaises(ValueError):
            rank_one_direction(np.zeros((2, 3)))
        direction, detail = rank_one_direction(np.array([[-2., 0.], [2., 0.]]))
        np.testing.assert_allclose(direction, [1, 0])
        self.assertTrue(detail["sign_tie"])

    def test_anchors_pair_order_readout_and_serialization(self):
        # Two independent concepts; translation of the whole training/query
        # space cannot alter the anchored scores.
        states = np.array([[[0., 0.]], [[2., 0.]], [[2., 3.]],
                           [[0., 1.]], [[2., 1.]], [[2., 4.]]])
        kwargs = dict(risk_pairs=[[0, 1], [3, 4]], manipulation_pairs=[[1, 2], [4, 5]],
                      safe_indices=[0, 3], original_harmful_indices=[1, 4], layer_ids=[7])
        bank = fit_concepts(states, **kwargs)
        np.testing.assert_allclose(bank.safe_anchor, [[0, .5]])
        np.testing.assert_allclose(bank.harmful_anchor, [[2, .5]])
        np.testing.assert_allclose(bank.risk_direction, [[1, 0]])
        np.testing.assert_allclose(bank.manipulation_direction, [[0, 1]])
        translated = fit_concepts(states + 100, **kwargs)
        for actual, expected in zip(concept_scores(states, bank), concept_scores(states + 100, translated)):
            np.testing.assert_allclose(actual, expected)
        reloaded = ConceptBank(**bank.arrays())
        np.testing.assert_allclose(concept_scores(states, bank)[0], concept_scores(states, reloaded)[0])
        np.testing.assert_array_equal(anchored_cosine(bank.safe_anchor[None], bank.safe_anchor,
                                                   bank.risk_direction), [[0]])

    def test_bad_pair_indices_and_nonfinite_queries_rejected(self):
        states = np.ones((3, 1, 2))
        with self.assertRaises(ValueError):
            fit_concepts(states, [[0, 3]], [[1, 2]], [0], [1])
        with self.assertRaises(ValueError):
            anchored_cosine(np.full((1, 1, 2), np.nan), np.ones((1, 2)), np.ones((1, 2)))


if __name__ == "__main__":
    unittest.main()
