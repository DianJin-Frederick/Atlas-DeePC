
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.noise import measurement_noise_rank62_ablation as experiment
class Rank62NoiseTests(unittest.TestCase):
    def test_config_is_local(self):
        self.assertEqual(experiment.experiment_config().svd_rank, 62)
        self.assertEqual(experiment.experiment_config().lam_g, .05)
        self.assertEqual(experiment.rank59.experiment_config().svd_rank, 59)
        self.assertEqual(experiment.noise.experiment_config().svd_rank, 60)

    def test_frozen_atlas_and_paired_noise(self):
        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=1):
            for ratio in (0., .03):
                manifest = experiment.prepare_level(ratio, directory)
                self.assertTrue(manifest["nested_basis_prefix_verified"])
                self.assertTrue(manifest["same_chart_memberships_and_graph_verified"])
                folder = Path(directory) / experiment.level_name(ratio)
                experiment.initialize_worker(ratio, folder)
                controller = experiment.noise._controller
                self.assertEqual(controller.matrix_cache[(0,)][0]["basis"].shape, (144, 62))
                controller.config = replace(controller.config, steps=8)
                clean_path = Path(directory) / "clean.npz"
                if ratio == 0:
                    controller.run(212, clean_path)
                row = experiment.run_seed(212, folder)
                self.assertEqual(row["svd_rank"], 62)
                with np.load(row["result_path"], allow_pickle=True) as current:
                    if ratio == 0:
                        with np.load(clean_path, allow_pickle=True) as clean:
                            for key in ("y", "u", "ref", "chart", "goal_index", "distance"):
                                np.testing.assert_array_equal(current[key], clean[key])
                    else:
                        self.assertFalse(np.array_equal(current["y"], current["y_measured"]))
            self.assertEqual(experiment.noise.experiment_config().svd_rank, 60)


if __name__ == "__main__":
    unittest.main()
