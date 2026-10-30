"""Shared reference helpers do not depend on the retired stitched controller."""

from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from atlas_deepc.atlas import goal_reference as goals
from atlas_deepc.common.paths import source_path


class GoalReferenceTests(unittest.TestCase):
    def test_target_lookup_seed_offsets_and_cache(self):
        goals.offline_environment_targets.cache_clear()
        self.addCleanup(goals.offline_environment_targets.cache_clear)
        with patch.object(goals, "environment_target", side_effect=lambda s: [s, -s]) as target:
            first = goals.offline_environment_targets(3, 100)
            np.testing.assert_array_equal(first, [[100, -100], [101, -101], [102, -102]])
            self.assertIs(first, goals.offline_environment_targets(3, 100))
            self.assertEqual(target.call_count, 3)

    def test_branch_selection_for_both_output_dimensions(self):
        for dimension in (8, 10):
            with self.subTest(dimension=dimension), ExitStack() as stack:
                for name, value in {"M": 2, "L": 2, "P": dimension}.items():
                    stack.enter_context(patch.object(goals.pipe, name, value))
                stack.enter_context(patch.object(goals, "DEFAULT_EPISODE_LENGTH", 3))
                stack.enter_context(patch.object(goals, "offline_environment_targets", return_value=np.ones((2, 2))))
                stack.enter_context(patch.object(goals, "environment_target", return_value=np.ones(2)))
                H = np.zeros(((2 + dimension) * 2, 4))
                outputs = H[4:].reshape(2, dimension, 4)
                outputs[0, :4] = [[1, 1, -1, -1]] * 4
                outputs[0, 4:6] = 3
                outputs[0, 6] = [0.1, 0.2, 0.01, 0.02]
                if dimension == 10:
                    outputs[0, 8:10] = [[1, 1, -1, -1]] * 2
                current = outputs[0, :, 0].copy()
                config = SimpleNamespace(goal_chart_candidates=4, goal_charts=2)
                reference, metadata = goals.chart_compatible_goal(H, np.zeros(2), current, config)
                self.assertEqual(metadata["data_goal_index"], 0)
                np.testing.assert_array_equal(reference[4:8], np.zeros(4))
                config.goal_branch_selection = "closest_xy"
                _, metadata = goals.chart_compatible_goal(H, np.zeros(2), current, config)
                self.assertEqual(metadata["data_goal_index"], 2)
                config.goal_chart_override = 2
                with self.assertRaisesRegex(ValueError, "goal_chart_override"):
                    goals.chart_compatible_goal(H, np.zeros(2), current, config)
                config.goal_source_target_min_distance = 2
                with self.assertRaisesRegex(RuntimeError, "fair goal candidates"):
                    goals.chart_compatible_goal(H, np.zeros(2), current, config)

    def test_incomplete_episodes_are_rejected(self):
        H = np.zeros(((goals.pipe.M + goals.pipe.P) * goals.pipe.L, 1))
        with self.assertRaisesRegex(ValueError, "complete episodes"):
            goals.chart_compatible_goal(H, np.zeros(2), np.zeros(goals.pipe.P), SimpleNamespace())

    def test_source_registry_points_to_shared_module(self):
        self.assertEqual(source_path("goal_reference.py").resolve(), source_path("goal_reference").resolve())
        self.assertTrue(source_path("goal_reference.py").is_file())


if __name__ == "__main__":
    unittest.main()
