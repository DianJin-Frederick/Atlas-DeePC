import unittest

import numpy as np

from atlas_deepc.ablations.chordal_graph_cost import chordal_distances, graph_costs
from atlas_deepc.atlas.candidate_goal_routing_experiment import shortest_path_tables
from atlas_deepc.atlas.routed_deepc_experiment import shortest_chart_path


class ChordalGraphTests(unittest.TestCase):
    def test_rotation_invariance_and_projector_formula(self):
        rng = np.random.default_rng(10)
        u = np.linalg.qr(rng.normal(size=(12, 4)))[0]
        v = np.linalg.qr(rng.normal(size=(12, 4)))[0]
        rotation = np.linalg.qr(rng.normal(size=(4, 4)))[0]
        distances = chordal_distances([u, v, u @ rotation])
        expected = np.linalg.norm(u @ u.T - v @ v.T, "fro") / np.sqrt(2)
        self.assertAlmostEqual(distances[0, 1], expected)
        self.assertLess(distances[0, 2], 1e-7)
        self.assertAlmostEqual(distances[0, 1], distances[2, 1])

    def test_orthogonal_and_invalid_bases(self):
        eye = np.eye(6)
        self.assertAlmostEqual(chordal_distances([eye[:, :3], eye[:, 3:]])[0, 1], np.sqrt(3))
        with self.assertRaises(ValueError):
            chordal_distances([2 * eye[:, :3]])

    def test_counts_edges_and_both_routing_paths(self):
        counts = np.array([[10, 4, 4], [0, 10, 9], [2, 0, 10]])
        distances = np.array([[0., 2., 8.], [2., 0., 3.], [8., 3., 0.]])
        costs = graph_costs(distances, counts, 3)
        self.assertEqual(costs[0, 1], 1.)
        self.assertEqual(costs[1, 2], 1.)
        self.assertTrue(np.isinf(costs[2, 0]))
        self.assertTrue(np.all(np.isinf(np.diag(costs))))
        atlas = {"transition_counts": counts, "center_distance": distances,
                 "routing_edge_costs": costs}
        tables = shortest_path_tables(atlas, 3)
        self.assertEqual(tables[0][0, 2], 2.)
        self.assertEqual(shortest_chart_path(atlas, 0, 2, 3), [0, 1, 2])

    def test_zero_chordal_edge_is_retained(self):
        counts = np.array([[0, 4, 0], [0, 0, 4], [0, 0, 0]])
        distances = np.zeros((3, 3))
        atlas = {"transition_counts": counts, "center_distance": distances,
                 "routing_edge_costs": graph_costs(distances, counts, 3)}
        self.assertEqual(shortest_path_tables(atlas, 3)[0][0, 2], 0.)
        self.assertEqual(shortest_chart_path(atlas, 0, 2, 3), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
