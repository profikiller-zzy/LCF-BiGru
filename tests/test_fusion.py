import unittest

import numpy as np

from dual_concept.fusion import SparseFusionModel, build_feature_matrix, fit_sparse_fusion


class FusionTests(unittest.TestCase):
    def test_features_share_groups_by_physical_layer(self):
        risk = np.array([[1., 2., 3.], [4., 5., 6.]])
        manipulation = risk + 10
        x, groups, branches = build_feature_matrix(risk, manipulation, [7, 3], [7, 8], [3, 7, 8])
        np.testing.assert_array_equal(x, [[2, 1, 12, 13], [5, 4, 15, 16]])
        np.testing.assert_array_equal(groups, [7, 3, 7, 8])
        self.assertEqual(branches.tolist(), ["risk", "risk", "manipulation", "manipulation"])

    def test_solution_satisfies_group_lasso_kkt_conditions(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(800, 5))
        logits = 1.3 * x[:, 0] - .8 * x[:, 2] + .1
        y = (rng.random(len(x)) < 1 / (1 + np.exp(-logits))).astype(int)
        groups = np.array([3, 7, 3, 8, 8])
        model = fit_sparse_fusion(x, y, groups, regularization=.06, tol=1e-9)
        self.assertTrue(model.diagnostics["converged"])
        z = (x - model.mean) / model.scale
        residual = model.predict_proba(x) - y
        gradient = z.T @ residual / len(x)
        self.assertLess(abs(residual.mean()), 1e-8)
        for group in np.unique(groups):
            mask = groups == group
            norm = np.linalg.norm(model.coef[mask])
            if norm:
                np.testing.assert_allclose(gradient[mask] + .06 * model.coef[mask] / norm, 0, atol=1e-8)
            else:
                self.assertLessEqual(np.linalg.norm(gradient[mask]), .06 + 1e-8)
        history = model.diagnostics["objective_history"]
        self.assertTrue(np.all(np.diff(history) <= 1e-10))
        self.assertEqual(model.diagnostics["active_layers"], [3])

    def test_large_penalty_zeroes_groups_without_penalizing_intercept(self):
        x = np.arange(40.).reshape(20, 2)
        y = np.array([1] * 5 + [0] * 15)
        model = fit_sparse_fusion(x, y, [1, 1], regularization=10)
        np.testing.assert_array_equal(model.coef, [0, 0])
        self.assertAlmostEqual(model.intercept, np.log(1 / 3), places=12)

    def test_training_scaler_is_frozen_for_queries_and_roundtrip(self):
        x = np.array([[0., 3.], [1., 3.], [2., 3.], [3., 3.]])
        model = fit_sparse_fusion(x, [0, 0, 1, 1], [2, 3], regularization=.1)
        self.assertEqual(model.scale[1], 1.)
        query = np.array([[100., 3.]])
        prediction = model.decision_function(query)[0]
        np.testing.assert_allclose(model.decision_function(np.concatenate((x, query)))[-1], prediction)
        restored = SparseFusionModel(**model.arrays())
        np.testing.assert_allclose(restored.decision_function(query), [prediction])
        with self.assertRaises(ValueError):
            fit_sparse_fusion(x, [0, 0, 0, 0], [1, 2])


if __name__ == "__main__":
    unittest.main()
