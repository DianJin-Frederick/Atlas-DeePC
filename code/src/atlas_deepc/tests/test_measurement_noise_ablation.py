
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import tempfile
import unittest
from pathlib import Path
from dataclasses import replace

import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.noise import measurement_noise_ablation as experiment
class NoiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.H = np.load(experiment.baseline.DATA_PATH, mmap_mode="r")

    def test_only_coefficient_regularization_changes(self):
        from dataclasses import asdict
        original = asdict(experiment.baseline.default_config())
        actual = asdict(experiment.experiment_config())
        self.assertEqual(actual.pop("lam_g"), 0.0)
        original.pop("lam_g")
        self.assertEqual(actual, original)

    def test_offline_overlap_and_zero_identity(self):
        clean, scale, _ = experiment.noisy_library(self.H, 0)
        np.testing.assert_array_equal(clean, self.H)
        one, _, noise = experiment.noisy_library(self.H, .01)
        two, _, doubled = experiment.noisy_library(self.H, .02)
        np.testing.assert_array_equal(doubled, 2*noise)
        np.testing.assert_array_equal(one[:24], self.H[:24])
        np.testing.assert_allclose(np.sqrt(np.mean(noise**2, axis=(0, 1)))/scale, .01, rtol=.02)
        self.assertEqual(experiment.episode_outputs(two).shape, (200, 200, 10))

    def test_online_noise_reproducible_and_independent_of_truth(self):
        output = lambda obs: np.arange(10, dtype=float)
        sigma = np.linspace(.01, .1, 10)
        a = experiment.NoisyOutputObserver(211, sigma, output)
        b = experiment.NoisyOutputObserver(211, 2*sigma, output)
        a(np.zeros(10)); b(np.zeros(10))
        np.testing.assert_allclose(np.array(b.measured)-b.true, 2*(np.array(a.measured)-a.true))
        self.assertFalse(np.array_equal(a.true, a.measured))

    def test_zero_noise_rollout_matches_baseline(self):
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=1):
            experiment.prepare_level(0, folder)
            level = Path(folder) / experiment.level_name(0)
            experiment.initialize_worker(0, level)
            experiment._controller.config = replace(experiment._controller.config, steps=8)
            expected = Path(folder) / "clean.npz"
            experiment._controller.run(212, expected)
            row = experiment.run_seed(212, level)
            self.assertEqual(row["lam_g"], 0.0)
            with np.load(expected, allow_pickle=True) as clean, np.load(row["result_path"], allow_pickle=True) as noisy:
                for key in ("y", "u", "ref", "chart", "goal_index", "distance"):
                    np.testing.assert_array_equal(clean[key], noisy[key])
                np.testing.assert_array_equal(noisy["y"], noisy["y_measured"])
            experiment._sigma = np.full(10, .01)
            experiment._ratio = .01
            row = experiment.run_seed(213, level)
            with np.load(row["result_path"], allow_pickle=True) as trace:
                self.assertFalse(np.array_equal(trace["y"], trace["y_measured"]))
                np.testing.assert_allclose(trace["distance"], np.linalg.norm(trace["y"][:, 6:8]-row["exact_target_xy"], axis=1))


if __name__ == "__main__":
    unittest.main()
