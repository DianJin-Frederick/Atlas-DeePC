"""Checks for the paper/upstream GS-DeePC Reacher port."""
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import cvxpy as cp
import numpy as np

from atlas_deepc.benchmarks import gain_scheduled_deepc_reacher as gs


class GainScheduledTests(unittest.TestCase):
    def test_matched_default_horizons(self):
        self.assertEqual((gs.Config().tini, gs.Config().horizon), (2, 10))
        self.assertIn("tini2_tf10", gs.OUTPUT.name)

    def test_scheduler_uses_second_joint(self):
        y = np.zeros((2, 8))
        q = np.array([-.7, 1.2])
        y[:, 1], y[:, 3] = np.cos(q), np.sin(q)
        np.testing.assert_allclose(gs.scheduling_variable(y), q)
        y[:, 6:] = 500  # Cartesian reference/output does not choose the region.
        np.testing.assert_allclose(gs.scheduling_variable(y), q)

    def test_composite_hysteresis_jumps_and_endpoints(self):
        edges = np.arange(6.)
        self.assertEqual(gs.select_composite(1.5, edges), 0)
        self.assertEqual(gs.select_composite(1.9, edges, 0), 0)
        self.assertEqual(gs.select_composite(2., edges, 0), 1)
        self.assertEqual(gs.select_composite(1.01, edges, 1), 1)
        self.assertEqual(gs.select_composite(.99, edges, 1), 0)
        self.assertEqual(gs.select_composite(4.9, edges, 0), 3)
        self.assertEqual(gs.select_composite(.1, edges, 3), 0)
        self.assertEqual(gs.select_composite(5., edges), 3)
        np.testing.assert_array_equal(gs.region_labels([0,1,5], edges), [0,1,4])
        with self.assertRaises(ValueError):
            gs.select_composite(5.1, edges, 3)

    def test_raw_mosaic_provenance_no_crossings(self):
        rng = np.random.default_rng(12)
        u = rng.normal(size=(4, 60, 2))
        y = rng.normal(size=(4, 60, 8))
        q = np.tile(np.repeat([-1., 1., -1., 1.], 15), (4, 1))
        y[:, :, 1], y[:, :, 3] = np.cos(q), np.sin(q)
        c = replace(gs.Config(), tini=2, horizon=2, regions=2, columns_per_region=20)
        lib = gs.Library(u, y, c)
        self.assertEqual(lib.composite_u.shape, (1, 8, 40))
        for region, ids in enumerate(lib.provenance):
            for column, (ep, start) in enumerate(ids):
                labels = gs.region_labels(q[ep, start:start+4], lib.edges)
                self.assertTrue(np.all(labels == region))
                np.testing.assert_array_equal(lib.hu[region, :, column], u[ep, start:start+4].ravel())
                np.testing.assert_array_equal(lib.hy[region, :, column], y[ep, start:start+4, 6:8].ravel())
        np.testing.assert_array_equal(lib.composite_u[0, :, :20], lib.hu[0])
        np.testing.assert_array_equal(lib.composite_u[0, :, 20:], lib.hu[1])
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            gs.Library(u, y, replace(c, columns_per_region=10000))

    def make_controller(self):
        c = replace(gs.Config(), tini=1, horizon=1, regions=2, columns_per_region=4,
                    q_xy=3., r=.2, lambda_g=2., lambda_ini=7.)
        # Independent raw data equations make a non-unit-sum solution possible.
        bank = np.eye(8)
        lib = SimpleNamespace(config=c, edges=np.array([-np.pi, 0, np.pi]),
                              composite_u=bank[:4][None], composite_y=bank[4:][None])
        return gs.Controller(lib)

    def test_qp_cost_terminal_equality_no_affine_constraint(self):
        ctrl = self.make_controller()
        history = np.zeros((1, 8)); history[0, 1] = 1
        history[0, 6:8] = [.05, -.02]
        action, result = ctrl.action(np.array([[.1, -.1]]), history, np.array([.1, .05]))
        self.assertFalse(result['fallback'])
        self.assertTrue(ctrl.problem.is_qp())
        np.testing.assert_allclose(result['planned_y'][-1], [.1,.05], atol=1e-6)
        self.assertGreater(abs(result['g_coefficients'].sum()-1), .1)
        g, sigma = result['g_coefficients'], result['slack']
        expected = (3*np.sum((result['planned_y']-[.1,.05])**2)
                    + .2*np.sum(result['planned_u']**2) + 2*np.sum(g**2) + 7*np.sum(sigma**2))
        self.assertAlmostEqual(result['objective'], expected, places=6)
        self.assertLessEqual(np.max(np.abs(action)), 1)

    def test_failed_solve_explicit_zero_fallback(self):
        ctrl = self.make_controller()
        history = np.zeros((1, 8)); history[0, 1] = 1
        with patch.object(cp.Problem, 'solve', side_effect=cp.error.SolverError('test')):
            action, row = ctrl.action(np.zeros((1, 2)), history, np.zeros(2))
        np.testing.assert_array_equal(action, np.zeros(2))
        self.assertTrue(row['fallback'])
        self.assertTrue(np.isnan(row['pred_xy']).all())

    def test_smoke_not_success_estimate(self):
        a = dict(distance=np.zeros(8), fallback=np.zeros(8, bool),
                 region_switched=np.zeros(8, bool), stage_control_cost=np.zeros(8), u=np.zeros((8, 2)))
        row = gs.rollout_summary(a, 211, 8)
        self.assertFalse(row['success_evaluable'])
        self.assertIsNone(row['mean_last_100'])
        self.assertFalse(row['strict_success'])


if __name__ == '__main__':
    unittest.main()
