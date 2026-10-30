"""Saved-prefix conversion checks: never launch a controller."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from atlas_deepc.reporting.select_dpc_prefix import derive, prefix_arrays


class PrefixConversionTests(unittest.TestCase):
    def test_step_axes_and_static_metadata(self):
        source = {"y": np.arange(320*8).reshape(320, 8),
                  "planned_y": np.zeros((320, 15, 8)),
                  "inner_g": np.zeros((320, 5, 200)),
                  "status": np.full(320, "optimal"),
                  "initial_observation": np.arange(11),
                  "target_xy": np.arange(2), "final_goal": np.arange(8)}
        result = prefix_arrays(source, 320, 300)
        for key in ("y", "planned_y", "inner_g", "status"):
            np.testing.assert_array_equal(result[key], source[key][:300])
        for key in ("initial_observation", "target_xy", "final_goal"):
            np.testing.assert_array_equal(result[key], source[key])
        self.assertEqual(source["y"].shape, (320, 8))
        result["y"][0, 0] = -1
        self.assertEqual(source["y"][0, 0], 0)

    def test_reject_unknown_shapes_or_invalid_lengths(self):
        with self.assertRaises(ValueError):
            prefix_arrays({"unknown": np.zeros(3)}, 320, 300)
        for steps in (99, 320, 400):
            with self.assertRaises(ValueError):
                prefix_arrays({"y": np.zeros((320, 8))}, 320, steps)

    def test_never_overwrite(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "existing"
            output.mkdir()
            with self.assertRaises(ValueError):
                derive(Path(tmp) / "source", output)
            with self.assertRaises(ValueError):
                derive(output, output)


if __name__ == "__main__":
    unittest.main()
