"""Focused trial diagnostics do not change or overstate controller results."""
import unittest
import numpy as np
import select_dpc_focused as focused
import select_dpc_development as study


class FocusedTests(unittest.TestCase):
    def test_waypoints_advance_by_proximity_and_end_at_original_target(self):
        schedule = focused.WaypointSchedule([[.1, 0.], [.15, 0.]], [.2, 0.])
        ref, index = schedule.reference([0., 0.], 0)
        self.assertEqual(index, 0)
        np.testing.assert_allclose(ref[4:6], [.1, 0.])
        _, index = schedule.reference([.102, 0.], 1)
        self.assertEqual(index, 1)
        ref, index = schedule.reference([.152, 0.], 2)
        self.assertEqual(index, 2)
        np.testing.assert_allclose(ref[4:6], [.2, 0.])
        _, index = schedule.reference([.2, 0.], 100)
        self.assertEqual(index, 2)

    def test_timeout_is_recorded_not_claimed_as_reached(self):
        schedule = focused.WaypointSchedule([[.1, 0.], [.15, 0.]], [.2, 0.], max_steps=40)
        self.assertEqual(schedule.reference([0., 0.], 39)[1], 0)
        self.assertEqual(schedule.reference([0., 0.], 40)[1], 1)
        self.assertEqual(schedule.reference([0., 0.], 80)[1], 2)
        self.assertEqual([e["reason"] for e in schedule.events], ["timeout", "timeout"])

    def test_200_step_metrics_use_final_target_not_waypoints(self):
        trace = self.trace(200)
        trace["ref"] = [np.array([0., 0., 0., 0., .1, 0., 0., 0.])] * 200
        _, row = focused.snapshot(trace, [{}] * 200, np.zeros(2), study.default_plan(), 200)
        self.assertTrue(row["success"])
        self.assertAlmostEqual(row["common_task_cost"], 0.)
        self.assertAlmostEqual(row["active_reference_task_cost"], 200000.)

    def test_uniform_scaling_keeps_structure_and_relative_weights(self):
        config = study.SelectConfig()
        scaled = focused.scaled_config(config, .01)
        for key in ("q_xy", "q_velocity", "r", "lambda_g1", "lambda_projection", "lambda_slack"):
            self.assertAlmostEqual(getattr(scaled, key), .01 * getattr(config, key))
        for key in ("columns", "past", "horizon", "iterations", "input_limit"):
            self.assertEqual(getattr(scaled, key), getattr(config, key))
        for scale in (0., -1., float("nan")):
            with self.assertRaises(ValueError):
                focused.scaled_config(config, scale)

    def trace(self, length):
        return {"y": [np.zeros(10)] * length, "u": [np.zeros(2)] * length,
                "control_compute_time_s": [.02] * length,
                "statuses": [["optimal"] * 5] * length}

    def test_partial_success_is_not_completed_success(self):
        _, row = focused.snapshot(self.trace(100), [{}] * 100, np.zeros(2), study.default_plan(), 320)
        self.assertIsNone(row["success"])
        self.assertTrue(row["success_100"])

    def test_completed_success_and_outlier(self):
        trace = self.trace(320)
        _, row = focused.snapshot(trace, [{}] * 320, np.zeros(2), study.default_plan(), 320)
        self.assertTrue(row["success"])
        trace["y"][-1] = np.array([0., 0., 0., 0., 0., 0., .021, 0., 0., 0.])
        _, row = focused.snapshot(trace, [{}] * 320, np.zeros(2), study.default_plan(), 320)
        self.assertFalse(row["success"])
        self.assertAlmostEqual(row["last30_max_error_mm"], 21.)


if __name__ == "__main__":
    unittest.main()
