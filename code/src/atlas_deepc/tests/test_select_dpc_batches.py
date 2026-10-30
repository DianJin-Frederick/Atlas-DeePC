"""Batch orchestration tests: synthetic traces, no simulations or solvers."""
from contextlib import redirect_stdout
import fcntl
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from atlas_deepc.benchmarks import select_dpc_upstream as run
from atlas_deepc.common.metrics import true_error_metrics


class SelectBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name)/'results'
        self.cli = SimpleNamespace(seed_start=10001, seed_stop=10005, steps=300,
            block_start=None, block_stop=None, max_new_seeds=2, output_dir=self.output)
        costs = SimpleNamespace(Q=np.eye(8), R=np.eye(2), regularizer_cost_g_1=1.,
            regularizer_cost_g_pi=1., slack_cost=1.)
        args = SimpleNamespace(trajectory_data=None, controller_costs=costs)
        for target, kwargs in [
            ('load_upstream', dict(return_value=(None, args, None))),
            ('audit_baseline_data', dict(return_value={'matches_baseline': False})),
            ('digest', dict(return_value='unchanged-test-source')),
            ('run_seed', dict(side_effect=self.fake_seed)),
        ]:
            mocker = patch.object(run, target, **kwargs)
            mocker.start()
            self.addCleanup(mocker.stop)
        version = patch.object(run.importlib.metadata, 'version', return_value='test-version')
        version.start()
        self.addCleanup(version.stop)
        self.calls = []

    def fake_seed(self, dc, args, scheduler, seed, steps, path, progress):
        self.calls.append(seed)
        distance = np.full(steps, .005)
        row = dict(seed=seed, steps=steps, experiment_version=run.VERSION,
            result_path=str(path.resolve()), strict_success=True,
            target_above_original_height_constraint=False,
            closed_loop_reference_cost=float(steps), median_control_compute_time_ms=1600.,
            **true_error_metrics(distance))
        np.savez(path, summary=np.array(json.dumps(row)), distance=distance,
            y=np.zeros((steps, 8)), u=np.zeros((steps, 2)), ref=np.zeros((steps, 8)),
            control_compute_time_s=np.full(steps, 1.6))
        return row

    def invoke(self):
        with redirect_stdout(io.StringIO()):
            run.main(self.cli)
        return json.loads((self.output/'summary.json').read_text()), json.loads((self.output/'progress.json').read_text())

    def test_automatic_batches_resume_and_full_completion(self):
        report, progress = self.invoke()
        self.assertEqual(self.calls, [10001, 10002])
        self.assertEqual(report['total'], 2)
        self.assertFalse(report['complete'])
        self.assertEqual(progress['state'], 'batch_complete')
        self.assertEqual(progress['next_missing_seed'], 10003)
        manifest = (self.output/'manifest.json').read_bytes()
        report, _ = self.invoke()
        self.assertEqual(self.calls, [10001, 10002, 10003, 10004])
        self.assertEqual(report['total'], 4)
        report, progress = self.invoke()
        self.assertEqual(self.calls, list(range(10001, 10006)))
        self.assertTrue(report['complete'])
        self.assertEqual(report['successes'], 5)
        self.assertEqual(progress['state'], 'complete')
        self.assertIsNone(progress['next_missing_seed'])
        self.assertEqual((self.output/'manifest.json').read_bytes(), manifest)
        self.invoke()  # Already-complete panel must never run again.
        self.assertEqual(len(self.calls), 5)

    def test_noncontiguous_blocks_and_overlap_share_statistics(self):
        self.cli.block_start, self.cli.block_stop = 10004, 10005
        report, progress = self.invoke()
        self.assertEqual(report['completed_seeds'], [10004, 10005])
        self.assertTrue(progress['block_complete'])
        self.assertFalse(report['complete'])
        self.invoke()
        self.assertEqual(self.calls, [10004, 10005])
        self.cli.block_start, self.cli.block_stop = 10001, 10004
        self.cli.max_new_seeds = 3
        report, progress = self.invoke()
        self.assertEqual(self.calls, [10004, 10005, 10001, 10002, 10003])
        self.assertEqual(report['completed_seeds'], list(range(10001, 10006)))
        self.assertTrue(report['complete'])
        self.assertEqual(progress['state'], 'complete')
        self.assertAlmostEqual(report['timing_discard_first5']['median_ms'], 1600.)

    def test_interruption_preserves_completed_seeds(self):
        def interrupted(*args):
            if args[3] == 10002:
                raise KeyboardInterrupt
            return self.fake_seed(*args)
        with patch.object(run, 'run_seed', side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.invoke()
        progress = json.loads((self.output/'progress.json').read_text())
        self.assertEqual(progress['state'], 'interrupted')
        self.assertEqual(progress['completed'], 1)
        self.assertEqual(progress['seed'], 10002)
        self.invoke()
        self.assertEqual(self.calls, [10001, 10002, 10003])

    def test_changed_horizon_rejected_without_new_seeds(self):
        self.invoke()
        self.cli.steps = 320
        with self.assertRaisesRegex(ValueError, 'Incompatible run manifest'):
            self.invoke()
        self.assertEqual(self.calls, [10001, 10002])

    def test_incompatible_cached_trace_rejected(self):
        self.invoke()
        self.fake_seed(None, None, None, 10001, 320,
                       self.output/'seed10001_control_results.npz', None)
        with self.assertRaisesRegex(ValueError, 'Incompatible saved rollout'):
            self.invoke()

    def test_invalid_blocks_rejected_before_output_creation(self):
        for start, stop, limit in [(10000, 10002, 2), (10004, 10006, 2),
                                   (10004, 10003, 2), (10001, None, 2), (None, None, 0)]:
            self.cli.block_start, self.cli.block_stop, self.cli.max_new_seeds = start, stop, limit
            with self.subTest(start=start, stop=stop, limit=limit), self.assertRaises(ValueError):
                self.invoke()
            self.assertFalse(self.output.exists())

    def test_active_lock_rejects_duplicate_worker(self):
        self.invoke()
        with (self.output/'run.lock').open('r') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.invoke()
        self.assertEqual(self.calls, [10001, 10002])


if __name__ == '__main__':
    unittest.main()
