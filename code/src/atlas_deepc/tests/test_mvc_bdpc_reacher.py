"""Equation/protocol tests, independent of full Reacher development experiments."""
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import cvxpy as cp
import numpy as np

from atlas_deepc.benchmarks import mvc_bdpc_reacher as mvc


class MVCPaperTests(unittest.TestCase):
    def test_hankel_and_original_width_after_svd(self):
        a=np.arange(40.).reshape(10,4)
        h=mvc.hankel(a,3)
        self.assertEqual(h.shape,(12,8))
        np.testing.assert_array_equal(h[:,2],a[2:5].ravel())
        q,low,s=mvc.truncated_subspace(h,.01)
        self.assertEqual(low.shape,h.shape)
        np.testing.assert_allclose(q.T@q,np.eye(q.shape[1]),atol=1e-12)
        self.assertEqual(q.shape[1],np.sum(s>s[0]*.01))

    def test_principal_angle_metric(self):
        e=np.eye(4)
        self.assertAlmostEqual(mvc.subspace_distance(e[:,:2],e[:,:2]),0)
        self.assertAlmostEqual(mvc.subspace_distance(e[:,:1],e[:,1:2]),np.pi/2)
        self.assertAlmostEqual(mvc.subspace_distance(e[:,:1],e[:,:2]),np.pi/2)
        t=.3
        v=(e[:,0]*np.cos(t)+e[:,1]*np.sin(t))[:,None]
        self.assertAlmostEqual(mvc.subspace_distance(e[:,:1],v),t)

    def test_nmf_normalization_consensus_and_representatives(self):
        rng=np.random.default_rng(9)
        x=[rng.uniform(.01,1,(12,12)) for _ in range(2)]
        c=replace(mvc.Config(),clusters=3,nmf_tol=1e-5,nmf_inner=100,nmf_outer=500)
        result=mvc.joint_nmf(x,c)
        self.assertTrue(result['converged'])
        np.testing.assert_allclose(result['views'].sum(axis=(1,2)),1)
        np.testing.assert_allclose(result['U'].sum(axis=1),1,atol=1e-12)
        np.testing.assert_allclose(result['consensus'],result['V'].mean(axis=0))
        np.testing.assert_array_equal(result['representatives'],result['consensus'].argmax(axis=0))
        self.assertTrue(np.all(np.diff(result['objective'])<=1e-10))
        self.assertTrue(np.all(result['V']>=0))

    def test_matching_uses_past_segments_and_eq9(self):
        lib=mvc.Library.__new__(mvc.Library)
        lib.config=replace(mvc.Config(),past=2,clusters=2)
        rng=np.random.default_rng(7)
        lib.centers=rng.normal(size=(2,8,4));past=rng.normal(size=(2,4))
        selected,d1,d2,scores=lib.select(past)
        np.testing.assert_allclose(d1,np.linalg.norm((lib.centers[:,-3:-1]-past).reshape(2,-1),axis=1))
        np.testing.assert_allclose(scores,.1*d1/d1.sum()+.1*d2/d2.sum())
        self.assertEqual(selected,np.argmin(scores))
        lib.centers[:,-1]=10000 # Future data must not influence online matching.
        np.testing.assert_array_equal(lib.select(past)[3],scores)

    def test_equation11_norms_offsets_and_constraints(self):
        rng=np.random.default_rng(11)
        c=replace(mvc.Config(),past=2,clusters=1)
        lib=SimpleNamespace(config=c,models=rng.normal(size=(1,12,15)),
                            centers=rng.normal(size=(1,9,4)),mean=np.array([.1,.2,.3,.4]),scale=np.array([1.,2.,3.,4.]))
        past=rng.normal(size=(2,4)); target=np.array([.2,-.1])
        problem,g,slack,un,yn=mvc.build_problem(lib,0,past,target)
        self.assertTrue(problem.is_dcp());self.assertFalse(problem.is_qp())
        self.assertEqual(g.shape,(15,))
        self.assertEqual(len(problem.constraints),1) # No hidden box/sum/g constraints.
        problem.solve(solver='CLARABEL')
        self.assertIn(problem.status,['optimal','optimal_inaccurate'])
        h=lib.models[0].reshape(3,4,15); center=lib.centers[0,-3:]
        np.testing.assert_allclose(h[:-1,:2].reshape(4,15)@g.value,(past[:,:2]-center[:-1,:2]).ravel(),atol=1e-7)
        np.testing.assert_allclose(un.value,h[-1,:2]@g.value+center[-1,:2])
        np.testing.assert_allclose(yn.value,h[-1,2:]@g.value+center[-1,2:])
        expected=(np.sqrt(c.q_xy)*np.linalg.norm(yn.value-(target-lib.mean[2:])/lib.scale[2:])
                  +np.sqrt(c.r)*np.linalg.norm(un.value+lib.mean[:2]/lib.scale[:2])
                  +c.lambda_epsilon*np.linalg.norm(slack.value))
        self.assertAlmostEqual(problem.value,expected,places=7)

    def test_zero_view_is_not_silently_replaced(self):
        with self.assertRaises(ValueError):
            mvc.joint_nmf([np.ones((5,5)),np.zeros((5,5))],mvc.Config())

    def test_rejected_solve_has_no_fallback_action(self):
        rng=np.random.default_rng(6)
        lib=mvc.Library.__new__(mvc.Library)
        lib.config=replace(mvc.Config(),past=2,clusters=1)
        lib.mean=np.zeros(4);lib.scale=np.ones(4)
        lib.models=rng.normal(size=(1,12,15));lib.centers=rng.normal(size=(1,9,4));lib.representatives=np.array([0])
        with patch.object(cp.Problem,'solve',side_effect=cp.error.SolverError('test')):
            action,row=mvc.compute_action(lib,np.zeros((2,2)),np.ones((2,2)),np.zeros(2))
        self.assertFalse(row['accepted']);self.assertTrue(np.isnan(action).all())
        self.assertEqual(row['solver_error'],'test')


if __name__=='__main__': unittest.main()
