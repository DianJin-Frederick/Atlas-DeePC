"""Focused checks for candidate selection and exact cached graph routing."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import unittest
from unittest.mock import patch

import numpy as np

from atlas_deepc.atlas import candidate_goal_routing_experiment as candidate
from atlas_deepc.control import expert_pipeline_search as pipe
from atlas_deepc.atlas import routed_deepc_experiment as routed
class CandidateRoutingTests(unittest.TestCase):
    def setUp(self):
        counts = np.array([[0, 4, 4], [0, 0, 4], [4, 0, 0]])
        center_distance = np.array([[0, 2, 8], [2, 0, 2], [2, 2, 0]], dtype=float)
        self.atlas = {"transition_counts": counts, "center_distance": center_distance,
                      "centers": np.zeros((3, 1))}

    def test_cached_routes_match_baseline(self):
        tables = candidate.shortest_path_tables(self.atlas, 3)
        policy = object.__new__(candidate.CandidateGoalPolicy)
        policy.atlas = self.atlas
        policy.distances, policy.predecessors = tables
        for source in range(3):
            for destination in range(3):
                self.assertEqual(policy.path(source, destination), routed.shortest_chart_path(self.atlas, source, destination, 3))
        self.assertEqual(policy.path(0, 2), [0, 1, 2])
        self.assertEqual(policy.distances[0, 2], 2.0)

    def test_candidate_time_alignment_ties_and_reselection(self):
        columns = routed.WINDOWS_PER_EPISODE
        H = np.zeros(((pipe.M + pipe.P) * pipe.L, columns))
        outputs = H[pipe.M * pipe.L:].reshape(pipe.L, pipe.P, columns)
        outputs[0, 6] = 100.0  # Selection must not use this older observation.
        outputs[pipe.T_INI - 1, 6] = np.arange(columns) * .001
        outputs[pipe.T_INI - 1, 0] = np.arange(columns)
        self.atlas["labels"] = np.array([2, 1, 1] + [0] * (columns - 3))
        config = routed.RoutedDeePCConfig()
        with patch.object(candidate, "offline_environment_targets", return_value=np.array([[1., 1.]])):
            policy = candidate.CandidateGoalPolicy(H, self.atlas, config, 3, candidate.shortest_path_tables(self.atlas, 3))
        policy.prepare(np.array([0., 0.]))
        np.testing.assert_array_equal(policy.candidates, [0, 1, 2])
        np.testing.assert_array_equal(policy.indices, [0, 1])
        goal, meta, chart, cost = policy.select(0)
        self.assertEqual(chart, 1)
        self.assertEqual(meta["data_goal_index"], 1)
        self.assertEqual(cost, 1.)
        self.assertEqual(goal[0], 1.)
        np.testing.assert_array_equal(goal[4:8], np.zeros(4))
        _, meta, chart, cost = policy.select(2)
        self.assertEqual(chart, 2)
        self.assertEqual(meta["data_goal_index"], 0)
        self.assertEqual(cost, 0.)
        self.assertGreaterEqual(meta["data_goal_source_target_distance"], config.goal_source_target_min_distance)

    def test_no_reachable_candidates_is_explicit(self):
        policy = object.__new__(candidate.CandidateGoalPolicy)
        policy.distances = np.array([[0, np.inf], [np.inf, 0]])
        policy.charts = np.array([1])
        with self.assertRaisesRegex(RuntimeError, "No candidate chart reachable"):
            policy.select(0)

    def test_fixed_nearest_goal_routes_without_reselecting(self):
        columns = routed.WINDOWS_PER_EPISODE
        H = np.zeros(((pipe.M + pipe.P) * pipe.L, columns))
        outputs = H[pipe.M * pipe.L:].reshape(pipe.L, pipe.P, columns)
        outputs[pipe.T_INI - 1, 6] = np.arange(columns) * .001
        self.atlas["labels"] = np.array([2] + [1] * (columns - 1))
        config = routed.RoutedDeePCConfig()
        with patch.object(candidate, "offline_environment_targets", return_value=np.array([[1., 1.]])):
            policy = candidate.FixedNearestGoalPolicy(
                H, self.atlas, config, candidate.shortest_path_tables(self.atlas, 3))
        policy.prepare(np.zeros(2))
        for current in (0, 1, 2, 0):
            goal, metadata, chart, cost = policy.select(current)
            self.assertEqual(metadata["data_goal_index"], 0)
            self.assertEqual(chart, 2)
            np.testing.assert_array_equal(goal, policy.fixed_selection[0])
            self.assertEqual(cost, policy.distances[current, 2])
        self.assertEqual(policy.path(0, 2), [0, 1, 2])
        self.assertEqual(policy.path(1, 2), [1, 2])
        self.assertEqual(policy.path(2, 2), [2])
        policy.distances[0, 2] = np.inf
        self.assertEqual(policy.path(0, 2), [])
        self.assertEqual(policy.select(0)[1]["data_goal_index"], 0)


if __name__ == "__main__":
    unittest.main()
