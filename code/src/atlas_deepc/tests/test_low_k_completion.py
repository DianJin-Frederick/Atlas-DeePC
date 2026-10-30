"""Protocol, metric and rank-selection checks for compact low-K completion."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from atlas_deepc.ablations import low_k_completion as study


class LowKCompletionTest(unittest.TestCase):
    def test_current_baseline_only_k_rank_and_steps_change(self):
        base = asdict(study.runner.experiment.baseline.default_config())
        for k in study.COUNTS:
            config = asdict(study.config_for(k, 53))
            self.assertEqual(config, dict(base, clusters=k, svd_rank=53, steps=300))
            self.assertEqual(config['lam_g'], 0)
            self.assertEqual(config['lam_du'], 0)
        self.assertEqual(study.SEEDS, tuple(range(211, 311)))

    def test_exhaustive_ranks_and_tie_breaking(self):
        self.assertEqual(study.RANKS, tuple(range(28, 71)))
        rows = [dict(rank=r, successes=10) for r in study.RANKS]
        self.assertEqual(study.best_record(rows)['rank'], 28)
        rows[-1]['successes'] = 11
        self.assertEqual(study.best_record(rows)['rank'], 70)
        self.assertEqual(len(study.RANKS)*len(study.COUNTS)*len(study.CASES)*len(study.SEEDS), 90300)

    def test_success_uses_300_steps_without_storing_errors(self):
        p, n = 8, 300
        y = np.zeros((n, p))
        y[:200, 6] = .1
        trace = dict(y=y, distance=y[:, 6], final_goal=np.zeros(p), ref=np.zeros((n, p)),
            goal_index=np.zeros(n), chart=np.zeros(n), selection_windows=np.zeros((n, 120)),
            selection_residuals=np.zeros((n, 1)), status=np.array(['optimal']*n))
        self.assertIs(study.seed_metrics(trace, 211, p, 'terminal', 1), True)
        summary = study.summarize(74, 100, 1, 53, p, 'terminal')
        self.assertEqual(summary['success_rate'], 74.)
        self.assertNotIn('median_rollout_rmse_mm', summary)
        trace['distance'] = np.zeros(320)
        with self.assertRaises(AssertionError):
            study.seed_metrics(trace, 211, p, 'terminal', 1)

    def test_pruning_only_removes_losing_checkpoints_after_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            winner = dict(rank=53, successes=80)
            for rank in (28, 53, 70):
                study.save_json(root/f'rank{rank:03d}.json', dict(rank=rank))
            study.save_json(root/'unrelated.json', {})
            with self.assertRaises(FileNotFoundError):
                study.prune_losing_ranks(root, winner)
            self.assertEqual(len(list(root.glob('rank*.json'))), 3)
            study.save_json(root/'selection.json', dict(complete=True, winner=winner))
            study.prune_losing_ranks(root, winner)
            study.prune_losing_ranks(root, winner)
            self.assertEqual([p.name for p in root.glob('rank*.json')], ['rank053.json'])
            self.assertTrue((root/'unrelated.json').exists())


if __name__ == '__main__':
    unittest.main()
