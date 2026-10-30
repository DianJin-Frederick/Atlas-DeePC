"""Numerical and provenance checks against fitted, evaluation-independent artifacts."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import unittest
import numpy as np
from scipy.stats import multivariate_normal
from scipy import sparse
from threadpoolctl import threadpool_limits
from atlas_deepc.ablations import clustering_methods_experiment as e
class FittedAtlasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limit=threadpool_limits(limits=1)
        e.routed.pipe.P=10
        cls.sig=e.prepare()
        cls.H=np.load(e.DATA,mmap_mode='r')
        with np.load(e.RESULT/'common_features.npz') as a:
            cls.X=a['features'];cls.mean=a['mean'];cls.scale=a['scale'];cls.landmarks=a['landmark_indices']
        cls.models={m:e.fit_method(m,cls.H,np.column_stack([np.zeros((len(cls.X),4)),cls.X]),
            cls.mean,cls.scale,cls.landmarks,cls.sig) for m in e.METHODS}

    def test_exact_feature_definition(self):
        Y=self.H[e.routed.pipe.M*e.routed.pipe.L:].reshape(e.routed.pipe.L,10,self.H.shape[1])
        raw=Y[e.routed.pipe.T_INI-1,[0,1,2,3,8,9],:].T
        np.testing.assert_allclose(self.X,(raw-self.mean[4:])/self.scale[4:],rtol=0,atol=0)
        self.assertEqual(e.file_digest(e.DATA),self.sig['data_sha256'])

    def test_reload_memberships_and_basis_bounds(self):
        for method,(gate,atlas,cache,meta) in self.models.items():
            self.assertEqual(len(cache),48)
            np.testing.assert_array_equal(gate.predict(self.X[::757]),atlas['labels'][::757])
            for chart in (0,23,47):
                expected,indices=e.routed.chart_matrix(self.H,atlas,(chart,),e.CONFIG,geometry='linear')
                actual,support=cache[(chart,)]
                np.testing.assert_array_equal(support,indices)
                for key in ('basis','offset','g_min','g_max'):
                    np.testing.assert_array_equal(actual[key],expected[key])
            with np.load(e.RESULT/method/'memberships.npz') as a:
                np.testing.assert_array_equal(a['column_indices'],np.arange(self.H.shape[1]))
                supports=np.concatenate([a[f'chart_{k:02d}'] for k in range(48)])
                np.testing.assert_array_equal(np.sort(supports),np.arange(self.H.shape[1]))

    def test_ppca_likelihood_against_independent_gaussian_density(self):
        gate=self.models['mppca'][0]
        X=self.X[::5000]
        expected=np.column_stack([multivariate_normal.logpdf(X,mean=gate.centers[k],cov=gate.covariances[k])+np.log(gate.weights[k]) for k in range(48)])
        np.testing.assert_allclose(gate.log_posterior(X),expected,rtol=1e-9,atol=1e-8)
        for k in range(48):
            W=gate.arrays['loadings'][k];v=gate.arrays['noise_variances'][k]
            np.testing.assert_allclose(gate.covariances[k],W@W.T+v*np.eye(6))
        self.assertTrue(np.all(np.diff(gate.diagnostics['mean_log_likelihood'])>=-1e-7))

    def test_graph_eigenfunctions(self):
        for method in ('spectral','diffusion'):
            gate=self.models[method][0]
            D=sparse.diags(gate.density**(-gate.alpha))
            A=D@gate.graph@D
            P=sparse.diags(1/np.asarray(A.sum(axis=1)).ravel())@A
            np.testing.assert_allclose(P@gate.psi,gate.psi*gate.eigenvalues,rtol=1e-5,atol=1e-8)
            a=gate.transform(self.X[[211,872,23010]])
            b=np.vstack([gate.transform(x[None,:]) for x in self.X[[211,872,23010]]])
            np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)


if __name__=='__main__':
    unittest.main()
