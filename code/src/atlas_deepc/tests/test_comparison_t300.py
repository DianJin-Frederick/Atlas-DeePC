"""Exercise real 300-step rollout/logging paths with cheap stub controllers.

These test horizon boundaries and statistics, not control quality or runtime.
Real controller formulations are covered by the separate controller tests.
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from atlas_deepc.benchmarks import contextual_dpc_atlas_terminal as ctx
from atlas_deepc.benchmarks import contextual_dpc_terminal_panel as panel
from atlas_deepc.benchmarks import select_dpc_upstream as select


class Comparison300Tests(unittest.TestCase):
    def check_trace(self, path):
        with np.load(path) as saved:
            a = {key: saved[key] for key in saved.files}
        row = json.loads(a['summary'].item())
        self.assertEqual(row['steps'], 300)
        for key in ('y', 'u', 'ref', 'distance', 'control_compute_time_s', 'stage_control_cost'):
            self.assertEqual(len(a[key]), 300, key)
        for n in (30, 100):
            self.assertAlmostEqual(row[f'mean_last_{n}'], a['distance'][-n:].mean())
            self.assertAlmostEqual(row[f'max_last_{n}'], a['distance'][-n:].max())
        self.assertEqual(row['strict_success'], bool(row['success_30'] and row['success_100']))
        self.assertAlmostEqual(row['closed_loop_control_cost'], a['stage_control_cost'].sum())
        return a, row

    def test_contextual_300_after_12_warmup_steps(self):
        config, _ = panel.frozen_config()
        lib = SimpleNamespace(config=config, q=np.r_[[.01]*4, [.0001]*2, [100.]*2, [0.]*2],
                              z_scale=np.ones(10), u_mean=np.zeros(2), u_scale=np.ones(2))
        env = ctx.gym.make('Reacher-v5')
        try:
            obs, _ = env.reset(seed=10001)
        finally:
            env.close()
        goal = np.r_[[1., 1., 0., 0., 0., 0.], obs[4:6]]
        calls = []
        def action(*args):
            calls.append(1)
            return np.zeros(2), dict(pred=np.zeros(8), fallback=False,
                status='test_stub', control_compute_time_s=len(calls)/1000,
                selection_time_s=.001, matrix_preparation_time_s=.002, solve_time_s=.003)
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch.object(ctx, 'compute_action', side_effect=action), \
             patch.object(ctx.gym, 'make', wraps=ctx.gym.make) as make:
            directory = Path(tmp)
            ctx.run_seed(lib, 10001, 300, directory, goal)
            make.assert_called_once_with('Reacher-v5', max_episode_steps=312)
            self.assertEqual(len(calls), 300)
            a, row = self.check_trace(directory/'seed10001_control_results.npz')
            self.assertEqual(a['warmup_y'].shape, (12, 8))
            np.testing.assert_array_equal(a['ref'], np.tile(goal, (300, 1)))
            report = ctx.summarize(directory, [10001])
            self.assertTrue(report['complete'])
            expected = 1000*a['control_compute_time_s'][5:]
            self.assertEqual(len(expected), 295)
            self.assertAlmostEqual(report['modular_timing_steps6_onwards']['control_compute_time_s']['mean_ms'], expected.mean())
            self.assertAlmostEqual(row['timing']['median_ms'], 1000*np.median(a['control_compute_time_s']))

    def test_select_300_without_warmup(self):
        watch = SimpleNamespace(selector=lambda original: original, install=lambda stack: None,
                                reset=lambda: None)
        calls = []
        def action(state, reference):
            calls.append(1)
            watch.selection = [(0., np.arange(200), np.zeros(200)) for _ in range(5)]
            watch.construction = [0.]*5
            watch.solve = [0.]*5
            watch.inner = [dict(pred=np.tile(reference, (15, 1)), planned_u=np.zeros((15, 2)),
                status='optimal', g=np.full(200, 1/200), objective=0., max_constraint_violation=0.)
                for _ in range(5)]
            return np.zeros(2)
        dc = SimpleNamespace(LkSelector=lambda: None,
                             SelectDeePC=lambda *args: SimpleNamespace(compute_action=action))
        args = SimpleNamespace(controller_costs=SimpleNamespace(Q=np.eye(8), R=np.eye(2)))
        def scheduler():
            return lambda obs: np.r_[[1., 1., 0., 0.], obs[4:6], [0., 0.]]
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch.object(select, 'Instrumentation', return_value=watch), \
             patch.object(select.gym, 'make', wraps=select.gym.make) as make:
            directory = Path(tmp)
            path = directory/'seed10001_control_results.npz'
            row = select.run_seed(dc, args, scheduler, 10001, 300, path, directory/'progress.json')
            make.assert_called_once_with('Reacher-v4', max_episode_steps=300)
            self.assertEqual(len(calls), 300)
            a, _ = self.check_trace(path)
            self.assertEqual(a['selected_indices'].shape, (300, 5, 200))
            report = select.summarize([row], [10001], directory, {'comparison_caveats': []})
            self.assertTrue(report['complete'])
            expected = 1000*a['control_compute_time_s'][5:]
            self.assertEqual(len(expected), 295)
            self.assertAlmostEqual(report['timing_discard_first5']['median_ms'], np.median(expected))


if __name__ == '__main__':
    unittest.main()
