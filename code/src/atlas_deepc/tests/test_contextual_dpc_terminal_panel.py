"""Evaluation launcher/resume tests with synthetic traces, never simulations."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from atlas_deepc.benchmarks import contextual_dpc_terminal_panel as panel


class EvaluationPanelTests(unittest.TestCase):
    def test_frozen_parameters(self):
        config, _ = panel.frozen_config()
        self.assertEqual((config.tini, config.horizon, config.columns), (2, 10, 200))
        self.assertEqual((config.q_joint, config.q_dq, config.q_xy), (.01, .0001, 100.))
        self.assertEqual(panel.SEEDS, list(range(10001, 10501)))
        self.assertEqual(panel.STEPS, 300)
        self.assertEqual((config.r, config.lambda_g_bar, config.lambda_sigma), (.1, .1, 100.))

    def test_reference_protocol_rejects_wrong_panel(self):
        manifest = dict(output_dimension=8, reference_mode='terminal',
            terminal_reference_policy='fixed_nearest_eligible_fingertip',
            atlas_sha256=panel.REFERENCE_ATLAS_SHA256, evaluation_seeds=panel.SEEDS,
            data_sha256='dataset', config=dict(clusters=64, svd_rank=53, steps=300,
                                               goal_source_target_min_distance=.06))
        with patch.object(panel.case.base, 'digest', return_value='dataset'):
            panel.validate_reference_manifest(manifest)
            for key, value in [('clusters', 48), ('svd_rank', 52), ('steps', 320),
                               ('goal_source_target_min_distance', .05)]:
                changed = copy.deepcopy(manifest)
                changed['config'][key] = value
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    panel.validate_reference_manifest(changed)
            for key, value in [('output_dimension', 2), ('reference_mode', 'guided'),
                               ('atlas_sha256', 'old-atlas'), ('evaluation_seeds', [211]),
                               ('data_sha256', 'different-data')]:
                changed = copy.deepcopy(manifest)
                changed[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    panel.validate_reference_manifest(changed)

    def test_resume_and_reject_changed_manifest(self):
        refs = np.zeros((2, 8))
        lib = SimpleNamespace(du_mean=np.zeros(2),du_scale=np.ones(2),z_mean=np.zeros(10),
                              z_scale=np.ones(10),q=np.ones(10),meta={})
        def fake_rollout(lib, seed, steps, directory, reference, setup):
            np.savez(directory/f'seed{seed}_control_results.npz',final_goal=reference,
                     summary=np.array(json.dumps(dict(seed=seed,steps=steps))))
        def fake_summary(directory, seeds):
            return dict(completed=len(list(directory.glob('seed*_control_results.npz'))),successes=0)
        with TemporaryDirectory() as tmp, patch.object(panel, 'SEEDS', [10001,10002]), \
             patch.object(panel, 'read_references', return_value=(refs, {'test':True})), \
             patch.object(panel.case.Library, 'from_path', return_value=lib), \
             patch.object(panel.case.base, 'digest', return_value='test-hash'), \
             patch.object(panel.case, 'run_seed', side_effect=fake_rollout) as rollout, \
             patch.object(panel.case, 'summarize', side_effect=fake_summary):
            directory=Path(tmp)
            panel.run(directory)
            self.assertEqual(rollout.call_count,2)
            self.assertTrue(all(call.args[2] == 300 for call in rollout.call_args_list))
            self.assertEqual(json.loads((directory/'manifest.json').read_text())['steps'], 300)
            panel.run(directory)
            self.assertEqual(rollout.call_count,2)

            self.assertEqual(json.loads((directory/'progress.json').read_text())['state'],'complete')
            manifest=json.loads((directory/'manifest.json').read_text())
            manifest['config']['q_xy']=999
            (directory/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'Configuration/source/provenance changed'):
                panel.run(directory)
            self.assertEqual(rollout.call_count,2)

    def test_resume_rejects_320_step_cache(self):
        refs = np.zeros((1, 8))
        lib = SimpleNamespace(du_mean=np.zeros(2), du_scale=np.ones(2), z_mean=np.zeros(10),
                              z_scale=np.ones(10), q=np.ones(10), meta={})
        with TemporaryDirectory() as tmp, patch.object(panel, 'SEEDS', [10001]), \
             patch.object(panel, 'read_references', return_value=(refs, {'test': True})), \
             patch.object(panel.case.Library, 'from_path', return_value=lib), \
             patch.object(panel.case.base, 'digest', return_value='test-hash'), \
             patch.object(panel.case, 'run_seed') as rollout:
            directory = Path(tmp)
            # Create the current manifest without creating a completed trace.
            with patch.object(panel.case, 'summarize', return_value=dict(completed=0, successes=0)):
                panel.run(directory)
            rollout.reset_mock()
            np.savez(directory/'seed10001_control_results.npz', final_goal=refs[0],
                     summary=np.array(json.dumps(dict(seed=10001, steps=320))))
            with self.assertRaisesRegex(ValueError, 'Incompatible saved rollout'):
                panel.run(directory)
            rollout.assert_not_called()
            self.assertEqual(json.loads((directory/'progress.json').read_text())['state'], 'interrupted')


if __name__=='__main__':
    unittest.main()
