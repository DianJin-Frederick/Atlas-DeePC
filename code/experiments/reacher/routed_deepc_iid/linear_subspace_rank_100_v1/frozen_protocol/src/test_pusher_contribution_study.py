"""Small deterministic tests for the contribution-study endpoint definitions."""

import unittest

import numpy as np

from pusher_atlas.config import PusherAtlasConfig
from pusher_contribution_study import metrics


def synthetic_result(distances, condition="nominal", error=""):
    n = len(distances)
    result = {
        "distance": np.asarray(distances), "dt": np.array(.05), "seed": np.array(1),
        "method": np.array("A_policy"), "condition": np.array(condition),
        "error": np.array(error), "action": np.zeros((n, 7)), "raw_action": np.zeros((n, 7)),
        "status": np.full(n, "policy_only"), "fallback": np.zeros(n, bool),
        "steady_timing_mask": np.arange(n) >= 20,
        "control_compute_time_s": np.full(n, .001), "environment_step_time_s": np.full(n, .0001),
        "actuator_violation": np.zeros(n), "joint_violation_rad": np.zeros(n),
        "trial_wall_time_s": np.array(1.0), "setup_time_s": np.array(0.0),
        "disturbance_pre_distance": np.array(.05), "disturbance_post_distance": np.array(.15),
    }
    return result


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.cfg = PusherAtlasConfig()

    def test_final_eight_boundary_is_success(self):
        r = synthetic_result([.2] * 232 + [.08] * 8)
        m = metrics(r, self.cfg, {})
        self.assertTrue(m["success"])
        self.assertAlmostEqual(m["settling_time_s"], 233 * .05)

    def test_seven_samples_is_not_success(self):
        r = synthetic_result([.2] * 233 + [.07] * 7)
        m = metrics(r, self.cfg, {})
        self.assertFalse(m["success"])
        self.assertTrue(m["reached_goal"])
        self.assertEqual(m["capped_settling_time_s"], 12.0)

    def test_earlier_hold_not_final_hold(self):
        r = synthetic_result([.01] * 239 + [.09])
        self.assertFalse(metrics(r, self.cfg, {})["success"])

    def test_exception_or_short_trial_remains_failure(self):
        for r in (synthetic_result([.01] * 239), synthetic_result([.01] * 240, error="failure")):
            m = metrics(r, self.cfg, {})
            self.assertFalse(m["complete"])
            self.assertFalse(m["success"])

    def test_recovery_requires_established_then_broken_success(self):
        r = synthetic_result([.01] * 120 + [.2] * 10 + [.01] * 110, "disturbance")
        m = metrics(r, self.cfg, {"step": 120})
        self.assertTrue(m["recovery_eligible"])
        self.assertTrue(m["recovery_success"])
        self.assertAlmostEqual(m["post_disturbance_settling_time_s"], .55)
        r["disturbance_post_distance"] = np.array(.07)
        self.assertFalse(metrics(r, self.cfg, {"step": 120})["recovery_eligible"])

    def test_torque_clipping_and_joint_excess_are_separate(self):
        r = synthetic_result([.01] * 240)
        r["method"] = np.array("C_combined")
        r["status"] = np.full(240, "optimal:osqp")
        r["raw_action"][0, 0] = 2.01
        r["joint_violation_rad"][1] = .01
        m = metrics(r, self.cfg, {})
        self.assertEqual(m["raw_qp_torque_exceedance_steps"], 1)
        self.assertEqual(m["torque_violation_steps"], 0)
        self.assertEqual(m["joint_limit_violation_steps"], 1)


if __name__ == "__main__":
    unittest.main()
