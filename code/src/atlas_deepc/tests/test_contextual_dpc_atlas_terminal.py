"""Reference/weight extension must leave the contextual formulation intact."""
import unittest
from unittest.mock import patch
import numpy as np
from atlas_deepc.benchmarks import contextual_dpc_reacher as base
from atlas_deepc.benchmarks import contextual_dpc_atlas_terminal as full


class TerminalReferenceTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(16)
        self.u=rng.uniform(-1,1,(3,30,2));self.y=rng.normal(size=(3,30,8))

    def test_xy_only_regression(self):
        c=full.Config(columns=30,q_joint=0.,q_dq=0.)
        original=base.Library(self.u,self.y,c)
        extension=full.Library(self.u,self.y,c)
        ref=self.y[0,3].copy()
        args=(self.u[0,:3],self.y[0,1:3])
        u0,r0=base.compute_action(original,*args,ref[6:8],-np.ones(2),np.ones(2))
        u1,r1=full.compute_action(extension,*args,ref,-np.ones(2),np.ones(2))
        np.testing.assert_array_equal(r0['selected_indices'],r1['selected_indices'])
        for k in ('g_coefficients','planned_y','planned_u','objective'):
            np.testing.assert_allclose(r0[k],r1[k],rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(u0,u1,atol=1e-7)

    def test_full_output_score_and_objective(self):
        c=full.Config(columns=30,q_joint=10.,q_dq=.1)
        with patch.object(np.linalg,'svd',side_effect=AssertionError('SVD forbidden')):
            lib=full.Library(self.u,self.y,c)
        ref=self.y[0,3].copy()
        uh,yh=self.u[0,:3],self.y[0,1:3]
        action,row=full.compute_action(lib,uh,yh,ref,-np.ones(2),np.ones(2))
        self.assertFalse(row['fallback'])
        normalized=(np.tile(np.r_[ref,0.,0.],(10,1))-lib.z_mean)/lib.z_scale
        p=((np.c_[yh,uh[-2:]]-lib.z_mean)/lib.z_scale).ravel()
        d1=np.linalg.norm(lib.zp-p[:,None],axis=0)
        d2=np.sqrt(np.sum((lib.zf-normalized.ravel()[:,None])**2*lib.future_weight[:,None],axis=0))
        score=(d1-d1.mean())/d1.std()+(d2-d2.mean())/d2.std()
        np.testing.assert_array_equal(row['selected_indices'],np.argsort(score,kind='stable')[:30])
        idx,g=row['selected_indices'],row['g_coefficients']
        z=(lib.zf[:,idx]@g).reshape(10,10)
        du=(lib.df[:,idx]@g).reshape(10,2)*lib.du_scale+lib.du_mean
        objective=np.sum((z-normalized)**2*lib.q)+c.r*(np.sum(((row['planned_u']-lib.u_mean)/lib.u_scale)**2)+np.sum((du/lib.u_scale)**2))+c.lambda_g_bar*c.columns*np.linalg.norm(g)+c.lambda_sigma*np.abs(row['slack']).sum()
        self.assertAlmostEqual(row['objective'],objective,places=5)
        self.assertAlmostEqual(g.sum(),1.,places=5)
        np.testing.assert_allclose(row['planned_u'],uh[-1]+np.cumsum(du,axis=0),atol=1e-6)
        self.assertLess(row['max_constraint_violation'],c.accept_residual)
        self.assertTrue(np.all(lib.q[:8]>0))
        np.testing.assert_array_equal(lib.q[8:],np.zeros(2))


if __name__=='__main__':unittest.main()
