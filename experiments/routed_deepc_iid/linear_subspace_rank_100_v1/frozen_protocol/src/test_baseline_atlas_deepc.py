import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

import baseline_atlas_deepc as baseline
import routed_deepc_experiment as routed


class BaselineTests(unittest.TestCase):
    def test_combination_and_per_step_candidate_selection(self):
        H = np.load(baseline.DATA_PATH, mmap_mode="r")
        atlas = routed.load_atlas(baseline.ATLAS_PATH)
        config = replace(baseline.default_config(), steps=4)
        calls = []
        original = baseline.CandidateGoalPolicy.select

        def select(policy, chart):
            result = original(policy, chart)
            self.assertEqual(policy.candidate_count, 10)
            self.assertEqual(result[-1], np.min(policy.distances[chart, policy.charts]))
            calls.append(chart)
            return result

        with threadpool_limits(limits=1), tempfile.TemporaryDirectory() as folder:
            controller = baseline.BaselineAtlasDeePC(H, atlas, config)
            self.assertEqual(len(controller.matrix_cache), 48)
            for matrix, _ in controller.matrix_cache.values():
                self.assertFalse(np.any(matrix["offset"]))
                self.assertEqual(matrix["basis"].shape[1], 60)
            with patch.object(baseline.CandidateGoalPolicy, "select", select):
                row = controller.run(211, Path(folder) / "trace.npz")
            self.assertEqual(len(calls), 4)
            self.assertEqual(row["matrix_geometry"], "linear")
            self.assertTrue(row["dynamic_terminal_reference"])
            self.assertFalse(row["fitted_dynamics_model"])
            self.assertEqual(row["unsolved_steps"], 0)

    def test_rejects_legacy_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "seed211_control_results.npz").touch()
            with self.assertRaisesRegex(ValueError, "Unversioned"):
                baseline.run_panel([211], folder)


if __name__ == "__main__":
    unittest.main()
