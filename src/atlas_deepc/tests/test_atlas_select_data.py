"""Dataset conversion and isolation checks; no full experiment is launched."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from atlas_deepc.benchmarks import atlas_select_data as study


class OriginalDataTests(unittest.TestCase):
    def observations(self, n=200, target=(.05,.08)):
        raw=np.zeros((n,11))
        raw[:,0:2]=1
        raw[:,4:6]=target
        raw[:,6]=np.arange(n)/100
        raw[:,7]=-np.arange(n)/200
        raw[:,8]=np.arange(n)/1000
        raw[:,9]=-.03
        return raw

    def test_mapping_alignment_and_window_count(self):
        u=np.column_stack([np.arange(200)/2000,-np.arange(200)/2000])
        raw=self.observations()
        h,target=study.episode_hankel(u,raw)
        self.assertEqual(h.shape,(144,189))
        np.testing.assert_allclose(target,raw[0,4:6])
        for start in (0,1,188):
            np.testing.assert_allclose(h[:24,start].reshape(12,2),u[start:start+12])
            y=h[24:,start].reshape(12,10)
            np.testing.assert_allclose(y[:,4:6],raw[start:start+12,6:8])
            np.testing.assert_allclose(y[:,6:8],raw[start:start+12,4:6]+raw[start:start+12,8:10])
            np.testing.assert_allclose(y[:,8],1)
            np.testing.assert_allclose(y[:,9],0)

    def test_reject_wrong_data(self):
        with self.assertRaises(ValueError):
            study.episode_hankel(np.ones((200,2)),self.observations())
        with self.assertRaises(ValueError):
            study.episode_hankel(np.zeros((200,2)),self.observations()[:,:8])
        raw=self.observations();raw[1,4]=.9
        with self.assertRaises(AssertionError):
            study.episode_hankel(np.zeros((200,2)),raw)

    def test_reference_uses_recorded_targets_and_actual_ids(self):
        a,_=study.episode_hankel(np.zeros((200,2)),self.observations(target=(0,0)))
        b,_=study.episode_hankel(np.zeros((200,2)),self.observations(target=(.2,.2)))
        h=np.concatenate([a,b],axis=1)
        h8=study.experiment.remove_joint_sum_channels(h)
        atlas={"labels":np.zeros(378,dtype=int)}
        with patch.object(study.goals,"offline_environment_targets",side_effect=AssertionError("Wrong dataset targets")):
            policy=study.OriginalDataGoal(h8,atlas,study.config(),np.array([[0,0],[.2,.2]]),[0,100])
            policy.prepare([0,0])
        goal,metadata,_,_=policy.select(0)
        self.assertGreaterEqual(metadata['data_goal_index'],189)  # episode 0 is ineligible
        self.assertEqual(metadata['data_goal_source_seed'],100)  # not Atlas DEFAULT_DATA_SEED
        np.testing.assert_allclose(goal[4:8],0)
        self.assertGreater(metadata['data_goal_source_target_distance'],.06)

    def test_defaults_and_readonly_empty_report(self):
        cfg=study.config()
        self.assertEqual((cfg.clusters,cfg.svd_rank,cfg.steps),(64,53,300))
        self.assertEqual((cfg.lam_g,cfg.lam_du,cfg.lam_ini),(0,0,1e5))
        self.assertEqual(study.SEEDS,tuple(range(211,311)))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'absent'
            self.assertEqual(study.report(path)['completed'],0)
            self.assertFalse(path.exists())

    def test_duplicate_worker_and_bad_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            with study.run_lock(d):
                with self.assertRaises(RuntimeError):
                    with study.run_lock(d):
                        pass
        for kwargs in ({'batch_size':0},{'block_start':210},{'block_stop':311}):
            with self.assertRaises(ValueError):
                study.run(**kwargs)


if __name__=='__main__':
    unittest.main()
