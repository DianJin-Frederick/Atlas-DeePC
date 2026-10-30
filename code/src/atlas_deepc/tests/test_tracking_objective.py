"""Objective checks without running environments or changing result data."""

import unittest
from unittest.mock import patch

import cvxpy as cp
import numpy as np

from atlas_deepc.baselines import baseline_atlas_deepc as baseline
from atlas_deepc.control.expert_pipeline_search import DeePCController, validate_objective_options


class TrackingObjectiveTest(unittest.TestCase):
    def controller(self):
        return DeePCController(np.array([[3.]]), np.array([[2.]]),
                               7., 1., cp.CLARABEL, dimensions=(1, 1, 1, 2))

    def test_baseline_defaults(self):
        config = baseline.default_config()
        self.assertEqual((config.lam_g, config.lam_du), (0., 0.))

    def test_three_term_formula_and_no_coefficient_cost(self):
        controller = self.controller()
        u, y, sigma, g = cp.Variable((2, 1)), cp.Variable((2, 1)), cp.Variable((1, 1)), cp.Variable((6, 1))
        u.value, y.value, sigma.value = np.array([[.2], [-.3]]), np.array([[1.], [2.]]), np.array([[.4]])
        reference = np.array([[.5], [1.]])
        cost = controller.tracking_cost(u, y, sigma, reference)
        expected = 3. * np.sum((y.value-reference)**2) + 2. * np.sum(u.value**2) + 7. * np.sum(sigma.value**2)
        self.assertAlmostEqual(np.asarray(cost.value).item(), expected)
        self.assertEqual({v.id for v in cost.variables()}, {u.id, y.id, sigma.id})
        self.assertFalse(hasattr(controller, "legacy_cost"))
        for name in ("lam_g", "lam_du", "Q_terminal"):
            self.assertFalse(hasattr(controller, name))

    def test_retired_penalties_are_not_silently_ignored(self):
        validate_objective_options(0., 0., np.zeros((1, 1)))
        for options in ({"lam_g": .05}, {"lam_du": 5.}, {"Q_terminal": np.eye(1)}):
            with self.assertRaisesRegex(ValueError, "frozen historical source"):
                validate_objective_options(**options)
            with self.assertRaises(TypeError):
                DeePCController(np.eye(1), np.eye(1), 7., 1., cp.CLARABEL, **options)

    def test_solution_scoring_matches_controller_objective(self):
        from atlas_deepc.atlas import contextual_cluster_atlas_experiment as contextual

        controller = self.controller()
        u = np.array([[.2], [-.3]])
        y = np.array([[1.], [2.]])
        sigma = np.array([[.4]])
        reference = np.array([[.5], [1.]])
        trajectory = np.vstack([[[0.]], u, sigma, y])
        with patch.multiple(contextual.pipe, M=1, P=1, T_INI=1, T_F=2):
            value = contextual._solution_objective(
                controller, np.eye(6), np.zeros((1, 1)), reference, u, y, trajectory)
        expected = controller.tracking_cost(u, y, sigma, reference).value
        self.assertAlmostEqual(value, np.asarray(expected).item())

    def test_solver_preserves_input_bounds_and_history_slack(self):
        controller = self.controller()
        basis = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                          [0., 0., 1., 0.], [0., 0., 0., 1.],
                          [0., 1., 0., 1.], [0., 0., 1., 1.]])
        result = controller.solve(basis, np.zeros((1, 1)), np.zeros((1, 1)),
                                  np.full((2, 1), 10.))
        self.assertIn(result[3], ("optimal", "optimal_inaccurate"))
        np.testing.assert_allclose(result[0], 1., atol=1e-5)
        np.testing.assert_allclose(result[1], 1. + 54./13., atol=1e-5)


if __name__ == "__main__":
    unittest.main()
