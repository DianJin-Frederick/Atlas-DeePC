import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

import numpy as np
from threadpoolctl import threadpool_limits

import measurement_noise_rank59_ablation as experiment


class Rank59NoiseTests(unittest.TestCase):
    def test_config_is_local(self):
        self.assertEqual(experiment.experiment_config().svd_rank, 59)
        self.assertEqual(experiment.experiment_config().lam_g, .05)
        self.assertEqual(experiment.noise.experiment_config().svd_rank, 60)
        self.assertEqual(experiment.noise.experiment_config().lam_g, 0)

    def test_zero_noise_and_noisy_rollouts(self):
        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=1):
            for ratio in (0., .05):
                artifact = experiment.prepare_level(ratio, directory)
                folder = Path(directory) / experiment.level_name(ratio)
                experiment.initialize_worker(ratio, folder)
                controller = experiment.noise._controller
                self.assertEqual(controller.matrix_cache[(0,)][0]["basis"].shape, (144, 59))
                controller.config = replace(controller.config, steps=8)
                clean_path = Path(directory) / "clean.npz"
                if ratio == 0:
                    controller.run(212, clean_path)
                row = experiment.run_seed(212, folder)
                with np.load(row["result_path"], allow_pickle=True) as trace:
                    self.assertEqual(json.loads(trace["summary"].item())["noise_version"], experiment.VERSION)
                    np.testing.assert_allclose(trace["distance"], np.linalg.norm(
                        trace["y"][:, 6:8]-row["exact_target_xy"], axis=1))
                    if ratio == 0:
                        with np.load(clean_path, allow_pickle=True) as clean:
                            for key in ("y", "u", "ref", "chart", "goal_index", "distance"):
                                np.testing.assert_array_equal(trace[key], clean[key])
                    else:
                        self.assertFalse(np.array_equal(trace["y"], trace["y_measured"]))
                self.assertEqual(artifact["signature"]["config"]["lam_g"], .05)
            self.assertEqual(experiment.noise.experiment_config().svd_rank, 60)


if __name__ == "__main__":
    unittest.main()
