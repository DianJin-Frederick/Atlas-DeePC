"""Model-independent Reacher dataset checks, without simulation rollouts."""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from atlas_deepc.common import reacher_data as data
from atlas_deepc.common.paths import source_path


class ReacherDataTests(unittest.TestCase):
    def setUp(self):
        self.episode_len = data.pipe.L + 2
        self.columns = 2 * (self.episode_len - data.pipe.L + 1)
        self.H = np.zeros(((data.pipe.M + data.pipe.P) * data.pipe.L, self.columns))

    def audit(self, H=None, seeds=(200,)):
        return data.audit_iid_library(
            self.H if H is None else H, data_seed=100, episodes=2,
            episode_len=self.episode_len, eval_seeds=seeds,
        )

    def test_target_lookup_uses_seed_and_closes_environment(self):
        env = Mock()
        observation = np.arange(10, dtype=float)
        env.reset.return_value = (observation, {})
        with patch.object(data.gym, "make", return_value=env) as make:
            target = data.environment_target(123)
        make.assert_called_once_with(data.pipe.REACHER_ENV_ID)
        env.reset.assert_called_once_with(seed=123)
        env.close.assert_called_once_with()
        np.testing.assert_array_equal(target, [4, 5])
        observation[4] = 99
        self.assertEqual(target[0], 4)

    def test_audit_reports_disjoint_seeds_and_target_distance(self):
        with patch.object(data, "environment_target", side_effect=lambda seed: np.array([seed, 1.])):
            report = self.audit()
        self.assertEqual(report["data_seed_range"], [100, 101])
        self.assertEqual(report["offline_stabilized_episode_count"], 0)
        self.assertEqual(report["held_out_targets"]["200"]["nearest_offline_seed"], 101)
        self.assertEqual(report["held_out_targets"]["200"]["nearest_offline_target_distance"], 99)

    def test_audit_rejects_overlapping_seeds(self):
        with patch.object(data, "environment_target", return_value=np.ones(2)):
            with self.assertRaisesRegex(RuntimeError, "offline collection seed range"):
                self.audit(seeds=(100,))

    def test_audit_rejects_stabilized_offline_episodes(self):
        with patch.object(data, "environment_target", return_value=np.zeros(2)):
            with self.assertRaisesRegex(RuntimeError, "stabilize at their targets"):
                self.audit()

    def test_audit_checks_column_count_before_target_lookup(self):
        with patch.object(data, "environment_target") as target:
            with self.assertRaisesRegex(ValueError, "columns for the audit"):
                self.audit(H=self.H[:, :-1])
            target.assert_not_called()

    def test_shared_source_is_registered(self):
        self.assertEqual(source_path("reacher_data.py").name, "reacher_data.py")
        self.assertTrue(source_path("reacher_data.py").is_file())


if __name__ == "__main__":
    unittest.main()
