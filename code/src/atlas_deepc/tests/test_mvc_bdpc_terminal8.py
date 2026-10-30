"""p=8 MVC dimensions, objective, rollout records and notebook launch safety."""
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cvxpy as cp
import numpy as np

from atlas_deepc.benchmarks import mvc_bdpc_terminal8 as mvc


class MVCTerminal8Tests(unittest.TestCase):
    def library(self):
        rng=np.random.default_rng(5)
        lib=mvc.Library.__new__(mvc.Library)
        lib.config=replace(mvc.Config(),past=2,clusters=1,q_joint=3.,q_dq=7.,q_xy=11.)
        lib.mean=np.arange(10)*.01; lib.scale=np.arange(10)*.1+1
        lib.models=rng.normal(size=(1,30,40))
        lib.centers=rng.normal(size=(1,20,10));lib.representatives=np.array([0])
        return lib

    def test_eight_output_objective_and_slack(self):
        lib=self.library();past=np.zeros((2,10));goal=np.arange(8)*.02
        problem,g,slack,un,yn=mvc.build_problem(lib,0,past,goal)
        self.assertEqual(slack.shape,(16,));self.assertEqual(yn.shape,(8,))
        self.assertEqual(len(problem.constraints),1)
        self.assertTrue(problem.is_dcp());self.assertFalse(problem.is_qp())
        problem.solve(solver='CLARABEL')
        self.assertIn(problem.status,('optimal','optimal_inaccurate'))
        yref=(goal-lib.mean[2:])/lib.scale[2:]
        expected=(np.linalg.norm(np.sqrt(lib.config.output_weights)*(yn.value-yref))
                  +np.sqrt(lib.config.r)*np.linalg.norm(un.value+lib.mean[:2]/lib.scale[:2])
                  +lib.config.lambda_epsilon*np.linalg.norm(slack.value))
        self.assertAlmostEqual(problem.value,expected,places=6)
        h=lib.models[0].reshape(3,10,40)
        np.testing.assert_allclose(yn.value,h[-1,2:]@g.value+lib.centers[0,-1,2:])
        with self.assertRaises(ValueError):mvc.build_problem(lib,0,past,np.zeros(2))

    def test_selection_uses_past_only(self):
        lib=self.library();past=np.ones((2,10))
        old=lib.select(past)
        lib.centers[:,-1]=10000
        for a,b in zip(old,lib.select(past)):np.testing.assert_array_equal(a,b)
        with self.assertRaises(ValueError):lib.select(np.ones((2,4)))

    def test_rejected_solve_shapes_and_no_fallback(self):
        lib=self.library()
        with patch.object(cp.Problem,'solve',side_effect=cp.error.SolverError('test')):
            action,row=mvc.compute_action(lib,np.zeros((2,2)),np.ones((2,8)),np.zeros(8))
        self.assertFalse(row['accepted']);self.assertTrue(np.isnan(action).all())
        self.assertEqual(row['pred_y'].shape,(8,));self.assertEqual(row['past_output_slack'].shape,(16,))

    def test_rollout_records_p8_and_300_step_statistics(self):
        lib=self.library();goal=np.array([1.,1.,0.,0.,0.,0.,.1,.1])
        def action(lib,uh,yh,ref):
            self.assertEqual(yh.shape,(2,8));np.testing.assert_array_equal(ref,goal)
            return np.zeros(2),dict(accepted=True,pred_y=goal.copy(),pred_xy=goal[6:8].copy(),
                cluster=0,control_compute_time_s=.01,command_u=np.zeros(2))
        with tempfile.TemporaryDirectory() as tmp,patch.object(mvc,'compute_action',side_effect=action),patch('builtins.print'):
            row=mvc.run_seed(lib,211,300,Path(tmp),goal)
            self.assertEqual(row['completed_steps'],300)
            with np.load(Path(tmp)/'seed211_control_results.npz') as a:
                self.assertEqual(a['ref'].shape,(300,8));self.assertEqual(a['y'].shape,(300,8))
                self.assertEqual(a['warmup_y'].shape,(12,8))
                np.testing.assert_allclose(a['ref'],np.tile(goal,(300,1)))
                self.assertAlmostEqual(row['mean_last_100'],a['distance'][-100:].mean())
                self.assertAlmostEqual(row['max_last_30'],a['distance'][-30:].max())
            summary=mvc.summarize(Path(tmp),[211]);self.assertEqual(summary['full_rollouts'],1)

    def test_notebook_defaults_and_cli_entrypoint(self):
        path=mvc.ROOT/'notebooks/reacher/comparisons/mvc_bdpc_reacher.ipynb'
        notebook=json.loads(path.read_text())
        for cell in notebook['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),str(path),'exec')
        settings=''.join(notebook['cells'][4]['source'])
        self.assertIn('211, 310, 300',settings)
        self.assertIn('mvc_bdpc_terminal8 as mvc',settings)
        self.assertIn('RUN_EXPERIMENT = False',settings)
        result=subprocess.run([sys.executable,'-m','atlas_deepc.benchmarks.mvc_bdpc_terminal8','--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0);self.assertIn('--prepare-only',result.stdout)


if __name__=='__main__':unittest.main()
