"""Synthetic checks only: no simulator, optimization, atlas fit or rollouts."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import unittest
import numpy as np
from atlas_deepc.noise import measurement_norm_noise_core as noise
class MeasurementNormNoiseTests(unittest.TestCase):
    def test_zero_and_scaling(self):
        y = np.arange(10, dtype=float)
        np.testing.assert_array_equal(noise.measurement_sigma(np.zeros(10), .1), [0.])
        np.testing.assert_array_equal(noise.measurement_sigma(y, 0), [0.])
        np.testing.assert_allclose(noise.measurement_sigma(2*y, .05), 2*noise.measurement_sigma(y, .05))
        self.assertAlmostEqual(float(noise.measurement_sigma(y, .1).item())**2*10, .01*np.dot(y,y))

    def test_invalid_input(self):
        for ratio in [-1, np.nan, np.inf]:
            with self.assertRaises(ValueError):noise.measurement_sigma(np.ones(10), ratio)
        for y in [np.ones(9), np.full(10, np.nan)]:
            with self.assertRaises(ValueError):noise.measurement_sigma(y, .1)

    def test_expected_power_and_gaussian_draws(self):
        # Detect missing sqrt(dimension), clipping, or normalizing each random vector.
        y = np.arange(1, 11, dtype=float)
        z = np.random.default_rng(23).normal(size=(100000,10))
        eps = noise.measurement_sigma(y, .05)*z
        ratios = np.sum(eps**2, axis=1)/np.dot(y,y)
        self.assertAlmostEqual(ratios.mean(), .05**2, delta=.05**2*.01)
        self.assertGreater(ratios.std(), .001)

    def test_overlaps_inputs_and_paired_levels(self):
        t = noise.routed.WINDOWS_PER_EPISODE + noise.pipe.L - 1
        y = np.random.default_rng(7).normal(size=(2,t,10))
        y[:,0]=0
        outputs = noise.output_hankel(y)
        inputs = np.random.default_rng(8).normal(size=(noise.pipe.M*noise.pipe.L, outputs.shape[1]))
        H = np.vstack([inputs,outputs])
        clean, _, zero = noise.noisy_library(H,0)
        first, sig, eps = noise.noisy_library(H,.02)
        second, _, eps2 = noise.noisy_library(H,.04)
        np.testing.assert_array_equal(clean,H)
        np.testing.assert_array_equal(zero,np.zeros_like(y))
        np.testing.assert_array_equal(first[:len(inputs)],inputs)
        np.testing.assert_array_equal(eps[:,0],0)
        np.testing.assert_allclose(eps2,2*eps)
        np.testing.assert_array_equal(noise.episode_outputs(first),y+eps)
        np.testing.assert_array_equal(sig,noise.measurement_sigma(y,.02))

    def test_online_follows_current_measurement_and_pairs_z(self):
        obs=np.zeros(11)
        first=noise.NoisyOutputObserver(211,.05,lambda _: np.ones(10))
        second=noise.NoisyOutputObserver(211,.05,lambda _: np.full(10,2.))
        a,b=first(obs),second(obs)
        np.testing.assert_array_equal(first.standard_normal,second.standard_normal)
        np.testing.assert_allclose(b,2*a)
        zero=noise.NoisyOutputObserver(211,.05,lambda _: np.zeros(10))
        np.testing.assert_array_equal(zero(obs),np.zeros(10))
        self.assertAlmostEqual(first.sigmas[0],.05)
        # Offline and online use the exact same sigma definition.
        self.assertAlmostEqual(first.sigmas[0],noise.measurement_sigma(first.true[0],.05).item())

    def test_rank_settings_are_local(self):
        from atlas_deepc.noise import measurement_norm_noise_rank59 as r59
        from atlas_deepc.noise import measurement_norm_noise_rank62 as r62
        original=noise.experiment_config
        with r59.settings():
            self.assertEqual(noise.experiment_config().svd_rank,59)
            self.assertEqual(noise.experiment_config().lam_g,.05)
        self.assertIs(noise.experiment_config,original)
        with r62.settings():self.assertEqual(noise.experiment_config().svd_rank,62)
        self.assertIs(noise.experiment_config,original)
        sig=r59.signature()
        self.assertEqual(sig['noise_model'],'measurement_norm_gaussian_v1')
        self.assertEqual(sig['solver'],'CLARABEL')

    def test_trace_finalization_with_synthetic_controller(self):
        # Exercise the sensor wrapper and checkpoint writer without MuJoCo or a QP.
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        class SyntheticController:
            config = SimpleNamespace(steps=320)
            def run(self, seed, path):
                obs = np.zeros(11)
                obs[:2] = 1
                obs[8:10] = [.08, .04]
                sensed = np.array([noise.pipe.reacher_output_from_obs(obs) for _ in range(323)])
                y = sensed[-320:]
                row = {'steps':320, 'seed':seed, 'closed_loop_cost':1.}
                np.savez_compressed(path, y=y, distance=np.linalg.norm(y[:,6:8],axis=1),
                    pred=y, prediction_error_xy=np.zeros(320), summary=np.array(json.dumps(row)))
                return row
        with tempfile.TemporaryDirectory() as temp:
            with patch.multiple(noise, _controller=SyntheticController(), _ratio=.05):
                row=noise.run_seed(211,temp)
            path=Path(row['result_path'])
            self.assertTrue(path.is_file())
            self.assertFalse((Path(temp)/'.seed211_pending.npz').exists())
            with np.load(path) as trace:
                self.assertEqual(trace['sensor_true'].shape,(323,10))
                self.assertEqual(trace['sensor_noise_sigma'].shape,(323,))
                np.testing.assert_array_equal(trace['y'],trace['sensor_true'][-320:])
                np.testing.assert_allclose(trace['sensor_measured'],trace['sensor_true']+
                    trace['sensor_noise_sigma'][:,None]*trace['sensor_standard_normal'])
                saved=json.loads(trace['summary'].item())
                self.assertEqual(saved['noise_model'],'measurement_norm_gaussian_v1')
                self.assertAlmostEqual(saved['mean_last_100'],np.hypot(.08,.04))
                self.assertIn('closed_loop_cost',saved['measurement_based_costs'])


if __name__=='__main__':unittest.main()
