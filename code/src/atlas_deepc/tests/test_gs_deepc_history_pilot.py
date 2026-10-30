"""Check the known-input audit against an independent CVXPY formulation."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import cvxpy as cp
import numpy as np

from atlas_deepc.benchmarks import gs_deepc_history_pilot as pilot


class HistoryAuditTests(unittest.TestCase):
    def test_paired_future_alignment_and_qp_equivalence(self):
        rng=np.random.default_rng(21)
        u=rng.normal(size=(1,12,2)); y=rng.normal(size=(1,12,8))
        y[:,:,1]=1.; y[:,:,3]=0.
        ids=[(0,4),(0,8)]
        for tini in (2,3,4):
            c=replace(pilot.config(tini),horizon=2,regions=2,columns_per_region=15)
            hu=rng.normal(size=(2*(tini+2),30)); hy=rng.normal(size=hu.shape)
            lib=SimpleNamespace(config=c,edges=np.array([-np.pi,0,np.pi]),
                composite_u=hu[None],composite_y=hy[None])
            scores,one,horizon=pilot.fixed_input_prediction(lib,u,y,'q2',ids)
            for j,(e,t) in enumerate(ids):
                g=cp.Variable(30)
                yp=hy[:2*tini]; yf=hy[2*tini:]
                problem=cp.Problem(cp.Minimize(c.lambda_g*cp.sum_squares(g)+
                    c.lambda_ini*cp.sum_squares(yp@g-y[e,t-tini:t,6:8].ravel())),
                    [hu@g==u[e,t-tini:t+2].ravel()])
                problem.solve(solver='CLARABEL')
                self.assertIn(problem.status,('optimal','optimal_inaccurate'))
                err=np.linalg.norm((yf@g.value).reshape(2,2)-y[e,t:t+2,6:8],axis=1)
                np.testing.assert_allclose(one[j],err[0],rtol=1e-5,atol=1e-6)
                np.testing.assert_allclose(horizon[j],np.sqrt(np.mean(err**2)),rtol=1e-5,atol=1e-6)
            self.assertLess(scores['max_input_residual'],1e-6)


if __name__=='__main__':
    unittest.main()
