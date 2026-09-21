import unittest
import numpy as np
import select_dpc_waypoint_dev100 as panel
from select_dpc_focused import WaypointSchedule


class PanelTests(unittest.TestCase):
    def test_frozen_single_setting_panel(self):
        p = panel.default_plan()
        self.assertEqual(p["seeds"], list(range(211, 311)))
        self.assertEqual(p["steps"], 200)
        self.assertEqual(p["parameters"]["columns"], 200)
        self.assertEqual(p["parameters"]["lambda_projection"], 1e6)
        self.assertEqual(p["objective_scale"], .01)

    def test_points_follow_each_seeds_geometry(self):
        np.testing.assert_allclose(panel.intermediate_points([0, 0], [.3, -.3]), [[.1, -.1], [.2, -.2]])
        np.testing.assert_allclose(panel.intermediate_points([.1, .2], [.1, .2]), [[.1, .2], [.1, .2]])
        with self.assertRaises(ValueError):
            panel.intermediate_points([float("nan"), 0], [0, 0])

    def test_final_reference_no_later_than_step80(self):
        schedule = WaypointSchedule(panel.intermediate_points([0, 0], [.3, 0]), [.3, 0], max_steps=40)
        self.assertEqual(schedule.reference([0, 0], 39)[1], 0)
        self.assertEqual(schedule.reference([0, 0], 40)[1], 1)
        ref, i = schedule.reference([0, 0], 80)
        self.assertEqual(i, 2)
        np.testing.assert_allclose(ref[4:6], [.3, 0])

    def test_partial_rate_is_not_a_complete_100_seed_result(self):
        row = dict(success=False, success_30=True, success_100=False, final100_mm=12.,
                   common_task_cost=100., active_reference_task_cost=50., unsolved_subproblems=3,
                   subproblems=1000)
        r = panel.aggregate([row], 100)
        self.assertFalse(r["complete"])
        self.assertEqual(r["completed_seeds"], 1)
        self.assertEqual(r["strict_successes"], 0)
        self.assertEqual(r["last30_successes"], 1)
        self.assertIsNone(panel.aggregate([], 100)["strict_success_rate_completed"])
        self.assertTrue(panel.aggregate([row] * 100, 100)["complete"])


if __name__ == "__main__":
    unittest.main()
