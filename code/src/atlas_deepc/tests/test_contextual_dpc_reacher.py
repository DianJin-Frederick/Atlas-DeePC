"""Small mathematical/protocol checks; no development sweep."""
import unittest
from unittest.mock import patch

import numpy as np

from atlas_deepc.benchmarks import contextual_dpc_reacher as ctx


class ContextualProtocol(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(16)
        self.u = rng.uniform(-1, 1, (3, 30, 2))
        self.y = rng.normal(size=(3, 30, 8))
        self.config = ctx.Config(columns=30)

    def test_recovery_and_episode_boundaries(self):
        y10 = np.concatenate([self.y, np.zeros((3, 30, 2))], axis=-1)
        h = np.vstack([ctx.windows(self.u, 12), ctx.windows(y10, 12)])
        u, y = ctx.recover_episodes(h, episode_length=30)
        np.testing.assert_array_equal(u, self.u)
        np.testing.assert_array_equal(y, self.y)
        h[0, 1] += 1
        with self.assertRaises(ValueError):
            ctx.recover_episodes(h, episode_length=30)

    def test_incremental_windows_normalization_and_no_svd(self):
        with patch.object(np.linalg, "svd", side_effect=AssertionError("SVD forbidden")):
            lib = ctx.Library(self.u, self.y, self.config)
            d = np.vstack([lib.dp, lib.df]).reshape(12, 2, -1)
            z = np.vstack([lib.zp, lib.zf]).reshape(12, 10, -1)
            np.testing.assert_allclose(d[:, :, 0] * lib.du_scale + lib.du_mean, np.diff(self.u[0, :13], axis=0))
            expected = np.c_[self.y[0, 1:13], self.u[0, 1:13]]
            np.testing.assert_allclose(z[:, :, 0] * lib.z_scale + lib.z_mean, expected)
            self.assertEqual(lib.meta["incremental_columns"], 3 * 18)
            # New episode's first window, not a boundary-crossing window.
            np.testing.assert_allclose(d[:, :, 18] * lib.du_scale + lib.du_mean, np.diff(self.u[1, :13], axis=0))

    def test_selector_matches_equations(self):
        lib = ctx.Library(self.u, self.y, self.config)
        p, r = np.arange(20) / 20., np.arange(100) / 100.
        dp = np.linalg.norm(lib.zp - p[:, None], axis=0)
        df = np.sqrt(np.sum((lib.zf-r[:, None])**2 * lib.future_weight[:, None], axis=0))
        expected = (dp-dp.mean())/dp.std() + (df-df.mean())/df.std()
        idx, scores = lib.select(p, r)
        np.testing.assert_array_equal(idx, np.argsort(expected, kind="stable")[:30])
        np.testing.assert_allclose(scores, expected[idx], atol=1e-12)
        self.assertEqual(len(np.unique(idx)), 30)

    def test_solver_raw_coefficients_and_increment_identity(self):
        lib = ctx.Library(self.u, self.y, self.config)
        action, record = ctx.compute_action(lib, self.u[0, :3], self.y[0, 1:3],
                                           self.y[0, 3, 6:8], -np.ones(2), np.ones(2))
        self.assertFalse(record["fallback"], record["solver_error"])
        self.assertEqual(record["g_coefficients"].shape, (30,))
        g, idx = record["g_coefficients"], record["selected_indices"]
        self.assertAlmostEqual(g.sum(), 1, places=5)
        du = (lib.df[:, idx] @ g).reshape(10, 2)*lib.du_scale + lib.du_mean
        predicted_z = (lib.zf[:, idx] @ g).reshape(10, 10)*lib.z_scale + lib.z_mean
        np.testing.assert_allclose(record["planned_u"], self.u[0, 2] + np.cumsum(du, axis=0), atol=1e-6)
        np.testing.assert_allclose(predicted_z[:, 8:], record["planned_u"], atol=1e-5)
        np.testing.assert_allclose(action, record["planned_u"][0], atol=1e-6)
        self.assertTrue(np.all(np.abs(action) <= 1))
        self.assertLess(record["max_constraint_violation"], self.config.accept_residual)
        reference = np.zeros((10, 10))
        reference[:, 6:8] = self.y[0, 3, 6:8]
        normalized_prediction = (predicted_z-lib.z_mean)/lib.z_scale
        normalized_reference = (reference-lib.z_mean)/lib.z_scale
        expected_objective = (
            np.sum((normalized_prediction-normalized_reference)**2 * lib.q)
            + self.config.r * np.sum(((record["planned_u"]-lib.u_mean)/lib.u_scale)**2)
            + self.config.r * np.sum((du/lib.u_scale)**2)
            + self.config.lambda_g_bar * 30 * np.linalg.norm(g)
            + self.config.lambda_sigma * np.sum(np.abs(record["slack"])))
        self.assertAlmostEqual(record["objective"], expected_objective, places=5)
        subtotal = sum(record[k] for k in ("selection_time_s", "matrix_preparation_time_s", "solve_time_s"))
        self.assertGreaterEqual(record["control_compute_time_s"], subtotal)

    def test_solver_failure_is_visible_zero_action(self):
        lib = ctx.Library(self.u, self.y, self.config)
        with patch.object(ctx.cp.Problem, "solve", side_effect=ctx.cp.error.SolverError("test failure")):
            action, row = ctx.compute_action(lib, self.u[0, :3], self.y[0, 1:3], np.zeros(2), -np.ones(2), np.ones(2))
        np.testing.assert_array_equal(action, np.zeros(2))
        self.assertTrue(row["fallback"])
        self.assertEqual(row["solver_error"], "test failure")
        self.assertTrue(np.isnan(row["pred"]).all())


if __name__ == "__main__":
    unittest.main()
