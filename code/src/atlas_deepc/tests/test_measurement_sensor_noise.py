"""Synthetic tests only; no simulator, solver, atlas fit, or evaluation rollout."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import unittest
from unittest.mock import patch
import numpy as np
from atlas_deepc.noise import measurement_sensor_noise_core as noise
def outputs(q, v=None, p=None):
    q=np.asarray(q); y=np.zeros(q.shape[:-1]+(10,))
    y[...,0:2]=np.cos(q);y[...,2:4]=np.sin(q)
    y[...,8]=np.cos(q.sum(axis=-1));y[...,9]=np.sin(q.sum(axis=-1))
    if v is not None:y[...,4:6]=v
    if p is not None:y[...,6:8]=p
    return y


class SensorNoiseTests(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(noise.os.environ);self.env.start();noise.configure(.1,1.,.01)
    def tearDown(self):self.env.stop()

    def test_fixed_scales_and_zero_readings(self):
        z=np.ones(6)
        for degree in [0,6,350]:
            clean=outputs(np.deg2rad([degree,degree]))
            measured=noise.corrupt_outputs(clean,1,z)
            expected=outputs(np.deg2rad([degree+.1,degree+.1]), [.01,.01], [.001,.001])
            np.testing.assert_allclose(measured,expected,atol=1e-15)
        np.testing.assert_allclose(noise.measurement_sigma(1),[np.deg2rad(.1)]*2+[.01]*2+[.001]*2)

    def test_periodic_identities_and_shared_angle_errors(self):
        rng=np.random.default_rng(5);q=rng.uniform(-np.pi,np.pi,(1000,2));z=rng.normal(size=(1000,6))
        measured=noise.corrupt_outputs(outputs(q),10,z)
        np.testing.assert_allclose(measured[:,:2]**2+measured[:,2:4]**2,1,atol=1e-15)
        np.testing.assert_allclose(measured[:,8],measured[:,0]*measured[:,1]-measured[:,2]*measured[:,3],atol=1e-15)
        np.testing.assert_allclose(measured[:,9],measured[:,2]*measured[:,1]+measured[:,0]*measured[:,3],atol=1e-15)

    def test_zero_identity_and_input_validation(self):
        y=outputs([.7,2.]);z=np.ones(6)
        np.testing.assert_array_equal(noise.corrupt_outputs(y,0,z),y)
        for m in [-1,np.nan,np.inf]:
            with self.assertRaises(ValueError):noise.measurement_sigma(m)
        with self.assertRaises(ValueError):noise.corrupt_outputs(y,1,np.ones(10))
        with self.assertRaises(ValueError):noise.configure(-1,1,.01)

    def test_offline_overlap_and_unchanged_inputs(self):
        rng=np.random.default_rng(7);t=noise.routed.WINDOWS_PER_EPISODE+noise.pipe.L-1
        clean=outputs(rng.normal(size=(2,t,2)),rng.normal(size=(2,t,2)),rng.normal(size=(2,t,2)))
        Hout=noise.output_hankel(clean);Hin=rng.normal(size=(noise.pipe.M*noise.pipe.L,Hout.shape[1]));H=np.vstack([Hin,Hout])
        noisy,sigma,delta=noise.noisy_library(H,2)
        z=np.random.default_rng(noise.OFFLINE_NOISE_SEED).normal(size=(2,t,6))
        np.testing.assert_array_equal(noisy[:len(Hin)],Hin)
        np.testing.assert_array_equal(noise.episode_outputs(noisy),noise.corrupt_outputs(clean,2,z))
        np.testing.assert_array_equal(delta,z*sigma)
        np.testing.assert_array_equal(noise.noisy_library(H,0)[0],H)

    def test_online_matches_offline_model_and_pairs_draws(self):
        y=outputs([.1,.2],[0,3],[.02,.1]);obs=np.zeros(11)
        a=noise.NoisyOutputObserver(211,1,lambda _:y)
        b=noise.NoisyOutputObserver(211,2,lambda _:y)
        ma,mb=a(obs),b(obs)
        np.testing.assert_array_equal(a.standard_normal,b.standard_normal)
        np.testing.assert_array_equal(ma,noise.corrupt_outputs(y,1,a.standard_normal[0]))
        np.testing.assert_array_equal(mb,noise.corrupt_outputs(y,2,b.standard_normal[0]))
        np.testing.assert_allclose(mb[4:]-y[4:],noise.corrupt_outputs(y,2,b.standard_normal[0])[4:]-y[4:])

    def test_independent_position_and_velocity_not_forward_kinematics(self):
        y=outputs([.2,.4],[1,2],[.03,.04]);z=np.array([1.,-2,0,0,0,0])
        changed=noise.corrupt_outputs(y,1,z)
        np.testing.assert_array_equal(changed[4:8],y[4:8])
        self.assertFalse(np.array_equal(changed[:4],y[:4]))

    def test_settings_signature(self):
        from atlas_deepc.noise import measurement_sensor_noise_rank59 as r59
        from atlas_deepc.noise import measurement_sensor_noise_rank62 as r62
        with r59.settings():self.assertEqual(noise.experiment_config().svd_rank,59)
        with r62.settings():self.assertEqual(noise.experiment_config().svd_rank,62)
        sig=r59.signature();self.assertEqual(sig['noise_model'],'physical_sensor_gaussian_v1')
        noise.configure(.2,2,.02)
        self.assertNotEqual(sig,r59.signature())

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
                self.assertEqual(trace['sensor_noise_sigma'].shape,(323,6))
                np.testing.assert_array_equal(trace['y'],trace['sensor_true'][-320:])
                np.testing.assert_allclose(trace['sensor_measured'],noise.corrupt_outputs(trace['sensor_true'],.05,trace['sensor_standard_normal']))
                saved=json.loads(trace['summary'].item())
                self.assertEqual(saved['noise_model'],'physical_sensor_gaussian_v1')
                self.assertAlmostEqual(saved['mean_last_100'],np.hypot(.08,.04))
                self.assertIn('closed_loop_cost',saved['measurement_based_costs'])

if __name__=='__main__':unittest.main()
