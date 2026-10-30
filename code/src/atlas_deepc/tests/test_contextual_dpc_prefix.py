"""Prefix conversion regression checks, without environment or solver calls."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from atlas_deepc.reporting.contextual_dpc_prefix import derive, prefix_arrays, rescore


class ContextualPrefixTests(unittest.TestCase):
    def test_prefix_preserves_warmup_and_recomputes_tail(self):
        y = np.zeros((320, 8))
        y[300:, 6] = .1  # Failure after step 300 must not affect the retained tail.
        a = dict(y=y, ref=np.zeros((320, 8)), u=np.zeros((320, 2)),
                 final_goal=np.zeros(8), target_xy=np.zeros(2),
                 initial_observation=np.zeros(10), warmup_u=np.zeros((12, 2)),
                 warmup_y=np.zeros((12, 8)), fallback=np.zeros(320, dtype=bool),
                 prediction_error_xy=np.zeros(320), distance=y[:, 6].copy(),
                 control_compute_time_s=np.ones(320),
                 stage_tracking_cost=y[:, 6]**2, stage_input_cost=np.zeros(320),
                 stage_smoothing_cost=np.zeros(320), stage_control_cost=y[:, 6]**2)
        prefix = prefix_arrays(a, 320, 300)
        self.assertEqual(prefix["warmup_y"].shape, (12, 8))
        self.assertEqual(prefix["y"].shape, (300, 8))
        norm = dict(Q_diagonal=np.r_[np.ones(8), np.zeros(2)],
                    z_scale=np.ones(10), z_mean=np.zeros(10))
        row = rescore(prefix, dict(seed=211, steps=320, wall_seconds=330.),
                      dict(config=dict(r=.1)), norm, 300)
        self.assertTrue(row["strict_success"])
        self.assertEqual(row["mean_last_100"], 0.)
        self.assertEqual(row["closed_loop_control_cost"], 0.)
        self.assertEqual(row["retained_control_compute_seconds"], 300.)
        self.assertIsNone(row["wall_seconds"])
        self.assertEqual(row["original_wall_seconds"], 330.)
        self.assertFalse(row["rerun"])
        self.assertEqual(a["y"].shape, (320, 8))
        self.assertTrue(np.all(a["distance"][300:] == .1))

    def test_unknown_shape_or_length_rejected(self):
        with self.assertRaises(ValueError):
            prefix_arrays(dict(unknown=np.zeros(3)), 320, 300)
        for length in (99, 320, 400):
            with self.assertRaises(ValueError):
                prefix_arrays(dict(y=np.zeros((320, 8))), 320, length)

    def test_existing_outputs_not_overwritten(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                derive(Path(tmp) / "source", Path(tmp))
            with self.assertRaises(ValueError):
                derive(Path(tmp), Path(tmp))


if __name__ == "__main__":
    unittest.main()
