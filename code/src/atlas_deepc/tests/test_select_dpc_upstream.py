"""Integration checks against a three-step pilot and untouched upstream calls.

Run in tmp/select_dpc_upstream_env after the stable three-step smoke command.
"""
import json
import unittest
from unittest.mock import patch

import cvxpy as cp
import gymnasium as gym
import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.benchmarks import select_dpc_upstream as run


class UpstreamSelectDPC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dc, cls.args, cls.scheduler = run.load_upstream()
        path = run.ROOT / "tmp/select_dpc_upstream_smoke_original_stable_v2/seed211_control_results.npz"
        if not path.exists():
            raise unittest.SkipTest("Run the stable three-step pilot first")
        with np.load(path) as trace:
            cls.trace = {k: trace[k] for k in trace.files}

    def test_original_settings_and_trace_alignment(self):
        t = self.trace
        self.assertEqual(t["y"].shape, (3, 8))
        self.assertEqual(t["selected_indices"].shape, (3, 5, 200))
        self.assertEqual(t["inner_status"].shape, (3, 5))
        self.assertEqual(self.args.deepc_dims.T_fut, 15)
        np.testing.assert_array_equal(np.diag(self.args.controller_costs.Q), [0, 0, 0, 0, 40000, 40000, 10, 10])
        self.assertEqual(self.args.controller_costs.regularizer_cost_g_pi, 10000)
        np.testing.assert_allclose(t["ref"][:, 6:8], np.tile(t["initial_observation"][4:6], (3, 1)))
        np.testing.assert_allclose(t["y"][:, 6:8], t["observation_after"][:, 4:6]+t["observation_after"][:, 8:10])
        np.testing.assert_allclose(t["distance"], np.linalg.norm(t["y"][:, 6:8]-t["target_xy"], axis=1))
        np.testing.assert_allclose(t["prediction_error_xy"], np.linalg.norm(t["pred"][:, 6:8]-t["y"][:, 6:8], axis=1))
        np.testing.assert_allclose(t["g_coefficients"], t["inner_g"][:, -1], equal_nan=True)
        good = np.isin(t["inner_status"], ["optimal", "optimal_inaccurate"])
        self.assertTrue(good.all(), t["inner_status"])
        np.testing.assert_allclose(t["inner_g"].sum(axis=-1), 1, atol=1e-5)
        self.assertLess(np.max(t["inner_max_constraint_violation"]), 1e-4)
        self.assertTrue((np.diff(t["selected_scores"], axis=-1) >= -1e-12).all())
        stages = sum(t[k] for k in ("selection_time_s", "matrix_preparation_time_s", "solve_time_s"))
        self.assertTrue(np.all(t["control_compute_time_s"] >= stages))
        row = json.loads(t["summary"].item())
        self.assertAlmostEqual(row["closed_loop_control_cost"], t["stage_control_cost"].sum())

    def test_logging_preserves_original_actions(self):
        # No selection, QP-construction or compute_action hooks in this replay.
        # Only the same explicitly requested Clarabel solver adapter is applied.
        solve = cp.Problem.solve

        def clarabel(problem, *args, **kwargs):
            kwargs["solver"] = "CLARABEL"
            kwargs.pop("eps_abs")
            kwargs.pop("eps_rel")
            kwargs.update(run.CLARABEL_OPTIONS)
            return solve(problem, *args, **kwargs)

        env = gym.make("Reacher-v4", max_episode_steps=3)
        obs, _ = env.reset(seed=211)
        controller = self.dc.SelectDeePC(self.args, self.dc.LkSelector(), 200, 5)
        scheduler = self.scheduler()
        try:
            with threadpool_limits(limits=1), patch.object(cp.Problem, "solve", clarabel):
                for step in range(3):
                    action = controller.compute_action(run.native_output(obs), scheduler(obs))
                    np.testing.assert_allclose(action, self.trace["u"][step], atol=1e-9, rtol=1e-9)
                    obs, *_ = env.step(action)
                    np.testing.assert_allclose(obs, self.trace["observation_after"][step], atol=1e-9, rtol=1e-9)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
