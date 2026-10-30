"""Validate geometry and constant-reference ablations before long panels."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from atlas_deepc.common.paths import project_root

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.control import expert_pipeline_search as pipe
from atlas_deepc.atlas import routed_deepc_experiment as routed
from atlas_deepc.atlas.candidate_goal_routing_experiment import CandidateGoalPolicy, shortest_path_tables


class BasisRoutingTests(unittest.TestCase):
    def test_terminal_only_retains_iterative_candidate_selection(self):
        root = project_root()
        H = np.load(root / "data/select_iid_offline_hankel.npy", mmap_mode="r")
        atlas = routed.load_atlas(root / "experiments/routed_deepc_iid/clustering_jointconfig48_fresh211_310/behavioral_atlas.npz")
        cfg = routed.RoutedDeePCConfig(steps=4, seed=211)
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=1):
            cache = routed.precompute_chart_matrix_cache(H, atlas, cfg, geometry="linear")
            policy = CandidateGoalPolicy(H, atlas, cfg, 10, shortest_path_tables(atlas, cfg.edge_count_threshold))
            path = Path(folder) / "terminal_iterative.npz"
            with patch.object(policy, "select", wraps=policy.select) as select, patch.object(
                    policy, "path", side_effect=AssertionError("Waypoint route must not be reconstructed")):
                routed.run_routed_deepc(H, atlas, cfg, matrix_cache=cache, goal_policy=policy,
                                       reference_mode="terminal", save_path=path)
            self.assertEqual(select.call_count, cfg.steps)
            with np.load(path) as trace:
                for reference, index in zip(trace["ref"], trace["goal_index"]):
                    expected = policy.references[np.flatnonzero(policy.indices == index)[0]]
                    np.testing.assert_array_equal(reference, expected)
                self.assertTrue(np.all(trace["path_length"] == 1))

    def test_uncentered_svd_and_same_bound_rule(self):
        rng = np.random.default_rng(14)
        X = rng.normal(size=(20, 50)) + np.linspace(1, 10, 20)[:, None]
        atlas = {"labels": np.zeros(50, dtype=int)}
        cfg = routed.RoutedDeePCConfig(svd_rank=8)
        with threadpool_limits(limits=1):
            affine, _ = routed.chart_matrix(X, atlas, [0], cfg)
            linear, _ = routed.chart_matrix(X, atlas, [0], cfg, geometry="linear")
        np.testing.assert_allclose(affine["offset"], X.mean(axis=1, keepdims=True))
        np.testing.assert_array_equal(linear["offset"], np.zeros((20, 1)))
        expected = np.linalg.svd(X, full_matrices=False)[0][:, :8]
        basis = linear["basis"]
        np.testing.assert_allclose(basis @ basis.T, expected @ expected.T, atol=1e-10)
        coordinates = basis.T @ X
        np.testing.assert_allclose(linear["g_min"], coordinates.mean(axis=1, keepdims=True) - 3 * coordinates.std(axis=1, keepdims=True))
        np.testing.assert_allclose(linear["g_max"], coordinates.mean(axis=1, keepdims=True) + 3 * coordinates.std(axis=1, keepdims=True))

    def test_fixed_terminal_skips_graph_and_keeps_same_goal(self):
        root = project_root()
        H = np.load(root / "data/select_iid_offline_hankel.npy", mmap_mode="r")
        atlas = routed.load_atlas(root / "experiments/routed_deepc_iid/clustering_jointconfig48_fresh211_310/behavioral_atlas.npz")
        cfg = routed.RoutedDeePCConfig(steps=3, seed=211)
        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=1):
            cache = routed.precompute_chart_matrix_cache(H, atlas, cfg)
            baseline = routed.run_routed_deepc(H, atlas, cfg, matrix_cache=cache)
            path = Path(directory) / "terminal.npz"
            with patch.object(routed, "shortest_chart_path", side_effect=AssertionError("Graph called in terminal mode")):
                terminal = routed.run_routed_deepc(H, atlas, cfg, matrix_cache=cache, reference_mode="terminal", save_path=path)
            self.assertEqual(baseline["data_goal_index"], terminal["data_goal_index"])
            with np.load(path) as result:
                np.testing.assert_array_equal(result["ref"], np.tile(result["final_goal"], (cfg.steps, 1)))
                self.assertTrue(np.all(result["goal_index"] == terminal["data_goal_index"]))


if __name__ == "__main__":
    unittest.main()
