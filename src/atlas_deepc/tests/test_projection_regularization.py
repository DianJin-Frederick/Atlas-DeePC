import tempfile
import unittest
from pathlib import Path
import cvxpy as cp
import numpy as np

from atlas_deepc.ablations import projection_regularization as study


class ProjectionTests(unittest.TestCase):
    def test_correct_rows_and_projector_identities(self):
        b=np.linalg.qr(np.random.default_rng(8).normal(size=(120,53)))[0]
        d=study.projection_geometry(b)
        np.testing.assert_allclose(d['M'],b[np.r_[0:4,24:40,4:24]])
        self.assertEqual(d['M'].shape,(40,53))  # no 1^T row
        self.assertEqual(d['nullity'],13)
        pi=d['projector'];null=d['null_basis']
        np.testing.assert_allclose(pi,np.linalg.pinv(d['M'],rcond=study.RTOL)@d['M'],atol=1e-12)
        np.testing.assert_allclose(pi@pi,pi,atol=1e-12)
        np.testing.assert_allclose(d['M']@null,0,atol=1e-12)
        np.testing.assert_allclose(null@null.T+pi,np.eye(53),atol=1e-12)
        g=np.arange(53,dtype=float)
        self.assertAlmostEqual(np.sum((null.T@g)**2),np.sum(((np.eye(53)-pi)@g)**2),places=9)

    def test_future_outputs_excluded(self):
        rng=np.random.default_rng(3)
        b=rng.normal(size=(120,53));other=b.copy();other[40:]*=1000
        a,c=study.projection_geometry(b),study.projection_geometry(other)
        np.testing.assert_array_equal(a['projector'],c['projector'])

    def test_orthogonal_basis_rotation_invariance(self):
        rng=np.random.default_rng(4)
        b=np.linalg.qr(rng.normal(size=(120,53)))[0]
        rotation=np.linalg.qr(rng.normal(size=(53,53)))[0]
        g=rng.normal(size=(53,1))
        a=study.projection_geometry(b);c=study.projection_geometry(b@rotation)
        self.assertAlmostEqual(float(np.sum((a['null_basis'].T@g)**2)),
                               float(np.sum((c['null_basis'].T@rotation.T@g)**2)),places=10)

    def test_full_column_rank_has_zero_penalty(self):
        b=np.eye(120)[:,:20]
        d=study.projection_geometry(b)
        self.assertEqual(d['null_basis'].shape,(20,0))
        np.testing.assert_allclose(d['projector'],np.eye(20))

    def test_solver_zero_parity_and_positive_penalty(self):
        dims=(1,1,1,2);b=np.eye(6)
        geo=study.projection_geometry(b,dims)
        matrix={'offset':np.zeros((6,1)),'basis':b,'projection_null_basis':geo['null_basis']}
        args=(np.eye(1),np.eye(1)*.1,1.,1.,cp.CLARABEL)
        u0=y0=np.zeros((1,1));ref=np.ones((2,1))
        baseline=study.BASE_CONTROLLER(*args,dimensions=dims).solve(matrix,u0,y0,ref)
        zero=study.ProjectionController(*args,dimensions=dims,lambda_pi=0).solve(matrix,u0,y0,ref)
        penalized=study.ProjectionController(*args,dimensions=dims,lambda_pi=9).solve(matrix,u0,y0,ref)
        for x,y in zip(baseline[:3],zero[:3]):np.testing.assert_array_equal(x,y)
        np.testing.assert_allclose(penalized[1],.1,atol=1e-7)
        self.assertLess(np.linalg.norm(geo['null_basis'].T@penalized[2]),np.linalg.norm(geo['null_basis'].T@zero[2]))
        changed=dict(matrix,g_sum=1.)
        with self.assertRaises(ValueError):
            study.ProjectionController(*args,dimensions=dims,lambda_pi=1).solve(changed,u0,y0,ref)

    def test_validation_and_no_results(self):
        for value in [-1,np.nan,np.inf]:
            with self.assertRaises(ValueError):study.check_lambda(value)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'absent'
            result=study.sweep_report(root,[0,1])
            self.assertFalse(root.exists())
            self.assertEqual([r['completed'] for r in result['panels']],[0,0])
        with self.assertRaises(ValueError):study.run(root=study.ROOT/'inside_project')


if __name__=='__main__':unittest.main()
