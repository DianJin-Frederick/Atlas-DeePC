"""The renderer must reconstruct routes with the experiment's saved metric."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from atlas_deepc.reporting.render_routed_deepc_dashboard import load_dashboard_atlas


class DashboardChordalTests(unittest.TestCase):
    def test_saved_chordal_costs_are_loaded(self):
        atlas = {"labels": np.array([0, 1]), "transition_counts": np.array([[0, 5], [7, 0]])}
        costs = np.array([[np.inf, .3], [.4, np.inf]])
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            np.savez(folder / "graph_geometry.npz", **atlas, chordal_cost=costs)
            with patch("atlas_deepc.reporting.render_routed_deepc_dashboard.load_atlas", return_value=atlas):
                result, geometry = load_dashboard_atlas(folder / "seed211.npz", folder / "atlas.npz",
                    {"reference_mode": "graph", "graph_cost_method": "chordal"})
            np.testing.assert_array_equal(result["routing_edge_costs"], costs)
            self.assertEqual(geometry, folder / "graph_geometry.npz")

    def test_missing_chordal_geometry_does_not_fall_back_to_centroids(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with patch("atlas_deepc.reporting.render_routed_deepc_dashboard.load_atlas", return_value={}):
                with self.assertRaises(FileNotFoundError):
                    load_dashboard_atlas(folder / "seed211.npz", folder / "atlas.npz",
                        {"reference_mode": "graph", "graph_cost_method": "chordal"})

    def test_no_graph_needs_no_graph_geometry(self):
        with patch("atlas_deepc.reporting.render_routed_deepc_dashboard.load_atlas", return_value={}):
            result, geometry = load_dashboard_atlas("seed211.npz", "atlas.npz", {"reference_mode": "terminal"})
        self.assertEqual(result, {})
        self.assertIsNone(geometry)
