"""Compact storage preserves rank selection and never removes winner traces."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from atlas_deepc.ablations import compact_large_k_rank as study


class CompactLargeKTest(unittest.TestCase):
    def test_only_chart_count_and_rank_change(self):
        old_k, old_seeds = study.original.K, study.original.SEEDS
        for k in (80, 96):
            with study.study_context(k):
                for rank in (28, 62, 70):
                    config = {key: value for key, value in asdict(study.original.config_for(rank)).items()
                              if key != 'seed'}
                    for p, mode in study.four.CASES:
                        study.original.validate_config(config, p, mode, rank)
                    self.assertEqual(config['clusters'], k)
                    self.assertEqual(config['lam_g'], 0.)
                    self.assertEqual(config['lam_du'], 0.)
        self.assertEqual(study.original.K, old_k)
        self.assertIs(study.original.SEEDS, old_seeds)
        self.assertEqual(len(study.RANKS)*len(study.SEEDS)*4*len(study.COUNTS), 34400)

    def make_panel(self, root):
        directory = root / 'sweep/rank062/output8/terminal'
        directory.mkdir(parents=True)
        rows = [dict(seed=211, success_30=True, success_100=True, mean_last_100=.001, unsolved_steps=0),
                dict(seed=212, success_30=False, success_100=False, mean_last_100=.020, unsolved_steps=1)]
        study.four.save_json(directory/'summary.json', dict(seeds=[211, 212], results=rows))
        for name in ('seed211_control_results.npz', 'seed212_control_results.npz', 'selection_geometry.npz'):
            np.savez_compressed(directory/name, value=np.arange(10))
        unrelated = directory/'unrelated.npz'
        np.savez_compressed(unrelated, value=np.arange(3))
        record = dict(seeds=2, successes=1, failed_seeds=[212],
            summary_sha256=study.four.file_digest(directory/'summary.json'),
            per_seed_rmse_mm={'211': 20., '212': 100.}, median_rollout_rmse_mm=60.)
        return directory, record

    def test_compaction_is_idempotent_and_keeps_seed_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, record = self.make_panel(Path(temp))
            summary = (directory/'summary.json').read_bytes()
            study.compact_panel(directory, record)
            rows = json.loads((directory/'seed_metrics.json').read_text())['seeds']
            self.assertEqual([r['success'] for r in rows], [True, False])
            self.assertEqual([r['final100_mean_error_mm'] for r in rows], [1., 20.])
            self.assertEqual([r['rollout_rmse_mm'] for r in rows], [20., 100.])
            self.assertFalse(list(directory.glob('seed*_control_results.npz')))
            self.assertFalse((directory/'selection_geometry.npz').exists())
            self.assertTrue((directory/'unrelated.npz').exists())
            self.assertEqual((directory/'summary.json').read_bytes(), summary)
            study.compact_panel(directory, record)

    def test_corrupted_compact_metrics_block_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, record = self.make_panel(Path(temp))
            study.compact_panel(directory, record)
            (directory/'seed_metrics.json').write_text('{}')
            with self.assertRaises(AssertionError):
                study.compact_panel(directory, record)

    def test_interrupted_deletion_resumes_from_published_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, record = self.make_panel(Path(temp))
            original_unlink = Path.unlink

            def interrupted(path, *args, **kwargs):
                if path.name == 'seed212_control_results.npz':
                    raise InterruptedError('simulated interruption')
                return original_unlink(path, *args, **kwargs)

            with patch.object(Path, 'unlink', interrupted), self.assertRaises(InterruptedError):
                study.compact_panel(directory, record)
            self.assertFalse((directory/'seed211_control_results.npz').exists())
            self.assertTrue((directory/'seed212_control_results.npz').exists())
            self.assertTrue((directory/'seed_metrics.json').exists())
            study.compact_panel(directory, record)
            self.assertFalse(list(directory.glob('seed*_control_results.npz')))

    def test_incorrect_success_count_prevents_deletion(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, record = self.make_panel(Path(temp))
            record['successes'] = 2
            with self.assertRaises(AssertionError):
                study.compact_panel(directory, record)
            self.assertEqual(len(list(directory.glob('seed*_control_results.npz'))), 2)

    def test_serial_traces_cannot_be_compacted(self):
        with self.assertRaisesRegex(ValueError, 'retain their full traces'):
            study.compact_panel(Path('/tmp/selected_serial/rank062/output8/terminal'), {})


if __name__ == '__main__':
    unittest.main()
