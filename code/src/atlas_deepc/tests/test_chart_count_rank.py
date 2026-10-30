import unittest
import numpy as np

from atlas_deepc.ablations import chart_count_rank as study
from atlas_deepc.ablations.chordal_graph_cost import chordal_distances, graph_costs
from atlas_deepc.atlas.candidate_goal_routing_experiment import shortest_path_tables


class ChartCountRankTests(unittest.TestCase):
    def test_single_chart_graph(self):
        basis = np.eye(144)[:, :60]
        costs = graph_costs(chordal_distances([basis]), np.array([[100]]), 3)
        self.assertTrue(np.isinf(costs[0, 0]))
        distances, _ = shortest_path_tables({"transition_counts": np.array([[100]]),
                                             "center_distance": np.zeros((1, 1)),
                                             "routing_edge_costs": costs}, 3)
        self.assertEqual(distances[0, 0], 0)

    def test_boundary_search(self):
        rows = [{"rank": r, "successes": 100-r, "median_rollout_rmse_mm": 1} for r in (48, 54, 60)]
        self.assertEqual(study.extend_ranks(rows), [42])
        rows[1]["successes"] = 100
        self.assertEqual(study.extend_ranks(rows), [])
        rows[2]["successes"] = 101
        self.assertEqual(study.extend_ranks(rows), [66])

    def test_only_k_and_rank_change(self):
        from dataclasses import asdict
        original = study.previous.control_signature(10, "graph")["config"]
        value = {k: v for k, v in asdict(study.config_for(1, 54)).items() if k != "seed"}
        expected = dict(original, clusters=1, svd_rank=54)
        self.assertEqual(value, expected)
        self.assertIsNone(study.reused_directory(1, 60))
        self.assertIsNotNone(study.reused_directory(48, 60))


if __name__ == "__main__":
    unittest.main()
