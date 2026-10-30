"""Regression checks for the retained fixed paper controllers and their data."""
import unittest
import json
import numpy as np
import cvxpy as cp
from threadpoolctl import threadpool_limits
from atlas_deepc.f1tenth import atlas, contextual, task
from atlas_deepc.f1tenth.config import AtlasConfig, ContextualConfig, TI, TF, Q_PHYSICAL, R_PHYSICAL, LIMIT, PAPER_SEEDS
from atlas_deepc.f1tenth.controller import DeePCController

class F1TenthPaperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=threadpool_limits(limits=1);cls.threads.__enter__()
        cls.record=task.collect();cls.a=atlas.fit(64);cls.lib=contextual.Library(cls.record)
    @classmethod
    def tearDownClass(cls):cls.threads.__exit__(None,None,None)

    def window(self,index):
        e,start=self.lib.meta[index];u=self.record['U'][e];y=self.record['Y'][e]
        return u[start-1:start+TI],y[start:start+TI],y[start+TI:start+TI+TF]

    def test_full_window_projection_and_causal_alignment(self):
        H,features,meta=atlas.build_library()
        pol=atlas.Atlas(self.a,AtlasConfig())
        for index in (80,900,2100):
            e,start=meta[index];u=self.record['U'][e];y=self.record['Y'][e]
            uh=u[start-1:start+TI+TF];yh=y[start:start+TI+TF]
            ref=y[start+TI+TF:start+TI+2*TF]
            pol.action(uh,yh,ref,AtlasConfig())
            np.testing.assert_allclose(pol.selection_window,H[:,index],atol=1e-12)
            expected=np.array([np.linalg.norm(H[:,index]-b@np.linalg.lstsq(b,H[:,index],rcond=None)[0])
                               for b in self.a['bases'][:,:,:40]])
            np.testing.assert_allclose(pol.selector.residual_history[-1],expected,atol=1e-8)
            # A different future reference must not affect history-only selection.
            _,one=pol.action(uh,yh,ref,AtlasConfig())
            _,two=pol.action(uh,yh,ref+1,AtlasConfig())
            self.assertEqual(one['chart'],two['chart'])
        with self.assertRaisesRegex(ValueError,'requires L observed'):
            pol.action(uh[-6:],yh[-5:],ref,AtlasConfig())

    def test_shared_warmup_and_saved_selection(self):
        import tempfile
        from pathlib import Path
        pol=atlas.Atlas(self.a,AtlasConfig())
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'atlas';c=Path(tmp)/'contextual'
            task.rollout(pol,AtlasConfig(),511,steps=2,save=a)
            task.rollout(self.lib,ContextualConfig(),511,steps=2,save=c)
            with np.load(a/'rollout.npz') as x,np.load(c/'rollout.npz') as y:
                for name in ('warmup_states','warmup_inputs','initial_state'):
                    np.testing.assert_array_equal(x[name],y[name])
                np.testing.assert_array_equal(x['measured_history'][:TI+TF],y['measured_history'][:TI+TF])
                first=task.frame(x['measured_history'][:TI+TF],x['measured_history'][TI-1,[0,1,3]])
                np.testing.assert_array_equal(x['selection_windows'][0],np.r_[x['input_history'][1:1+TI+TF].ravel(),first.ravel()])
                self.assertLess(np.linalg.norm(x['initial_state'][:2]-task.PATH.xy[task.PATH.localize(x['initial_state'][:2],0)]),.5)

    def test_partial_projection_alignment_and_rank_degeneracy(self):
        from atlas_deepc.f1tenth.partial_selection import METHOD, PAST_ROWS, past_matrix, RANK_RTOL
        H,_,meta=atlas.build_library()
        for rank in (24,40,42):
            cfg=AtlasConfig(rank=rank)
            pol=atlas.Atlas(self.a,cfg,selection_method=METHOD)
            for index in (80,900,2100):
                e,start=meta[index];u=self.record['U'][e];y=self.record['Y'][e]
                uh=u[start-1:start+TI];yh=y[start:start+TI];ref=y[start+TI:start+TI+TF]
                _,info=pol.action(uh,yh,ref,cfg)
                np.testing.assert_allclose(pol.selection_window,H[PAST_ROWS,index],atol=1e-12)
                expected=[]
                for b in self.a['bases'][:,:,:rank]:
                    p=past_matrix(b);w=pol.selection_window
                    expected.append(np.linalg.norm(w-p@np.linalg.lstsq(p,w,rcond=RANK_RTOL)[0]))
                np.testing.assert_allclose(pol.selector.residual_history[-1],expected,atol=1e-8)
                if rank>=40:
                    np.testing.assert_array_equal(pol.selector.ranks,[27,27])
                    self.assertLess(pol.selector.residual_history[-1][1],1e-10)
                # The query uses the recent past only, independent of older history.
                pol.action(np.vstack([np.ones((10,2))*99,uh]),np.vstack([np.ones((10,4))*99,yh]),ref,cfg)
                np.testing.assert_allclose(pol.selection_window,H[PAST_ROWS,index],atol=1e-12)
            full=atlas.Atlas(self.a,cfg)
            for a,b in zip(pol.parts,full.parts):
                for x,z in zip(a[:5],b[:5]):np.testing.assert_array_equal(x,z)
                for x,z in zip(a[5:],b[5:]):np.testing.assert_array_equal(x.toarray(),z.toarray())

    def test_partial_full_past_images_tie_exactly(self):
        from atlas_deepc.f1tenth.partial_selection import PastWindowProjectionSelector, ANCHOR_ROWS
        bases=atlas.fit(28,12)['bases']
        selector=PastWindowProjectionSelector(bases)
        rng=np.random.default_rng(91)
        for _ in range(10):
            window=rng.normal(size=30);window[ANCHOR_ROWS]=0.
            selected=selector.select(window)
            self.assertEqual(selected,0)
            self.assertEqual(selector.tie_history[-1],12)
            self.assertLess(max(selector.residual_history[-1]),1e-12)

    def test_offline_noise_before_windows_and_clustering(self):
        from atlas_deepc.f1tenth.config import DEFAULT_NOISE_SIGMA, OFFLINE_NOISE_SEED
        clean=task.collect_clean()
        scale=clean['Y'].reshape(-1,4).std(axis=0)
        expected=clean['Y']+np.random.default_rng(OFFLINE_NOISE_SEED).normal(size=clean['Y'].shape)*(DEFAULT_NOISE_SIGMA*scale)
        np.testing.assert_array_equal(self.record['Y'],expected)
        np.testing.assert_array_equal(task.collect()['Y'],expected)
        for key in ('U','X'):np.testing.assert_array_equal(self.record[key],clean[key])
        H,features,meta=atlas.build_library()
        for index in (80,81,900,2100):
            e,start=meta[index];y=expected[e,start:start+TI+TF]
            aligned=task.frame(y,y[TI-1,[0,1,3]])
            np.testing.assert_array_equal(H[2*(TI+TF):,index],aligned.ravel())
            self.assertEqual(features[index,0],expected[e,start+TI-1,2])
            contextual=(np.c_[aligned,self.record['U'][e,start:start+TI+TF]]-self.lib.z_mean)/self.lib.z_scale
            np.testing.assert_allclose(self.lib.zp[:,index],contextual[:TI].ravel(),atol=1e-12)

    def test_noise_ratio_scaling_and_zero_control(self):
        import tempfile
        from pathlib import Path
        clean=task.collect_clean()
        with tempfile.TemporaryDirectory() as tmp:
            zero=task.collect(Path(tmp)/'zero.npz',sigma=0.)
            low=task.collect(Path(tmp)/'low.npz',sigma=.01)
            high=task.collect(Path(tmp)/'high.npz',sigma=.05)
            np.testing.assert_array_equal(zero['Y'],clean['Y'])
            np.testing.assert_allclose(high['Y']-clean['Y'],5*(low['Y']-clean['Y']),atol=1e-13)
            np.testing.assert_allclose(high['noise_std']/high['signal_scale'],.05)
            state=clean['X'][0,0]
            a=task.measured(state,np.random.default_rng(99),noise_std=high['noise_std'])
            expected=state[[0,1,3,4]]+np.random.default_rng(99).normal(size=4)*high['noise_std']
            np.testing.assert_array_equal(a,expected)

    def test_atlas_matches_explicit_three_term_qp(self):
        for rank in (24,40,44,64):
            with self.subTest(rank=rank):
                self.check_atlas_qp(rank)
        for k,rank in ((1,40),(5,28),(15,26)):
            with self.subTest(clusters=k,rank=rank):
                self.check_atlas_qp(rank,k,atlas.fit(rank,k))

    def check_atlas_qp(self,rank,clusters=2,bases=None):
        cfg=AtlasConfig(clusters=clusters,rank=rank);pol=atlas.Atlas(self.a if bases is None else bases,cfg)
        for index in [80,900,2100]:
            e,start=self.lib.meta[index];u0=self.record['U'][e];y0=self.record['Y'][e]
            uh=u0[start-1:start+TI+TF];yh=y0[start:start+TI+TF];ref=y0[start+TI+TF:start+TI+2*TF]
            if rank==24:
                # Arbitrary excitation histories can require enormous coefficients
                # at low rank. Zero history is exactly feasible in every linear chart.
                uh=np.zeros_like(uh);yh=np.zeros_like(yh)
            action,info=pol.action(uh,yh,ref,cfg)
            self.assertTrue(info['accepted'])
            B,u,y,target,g_direct=pol.last
            oracle=DeePCController(np.diag(Q_PHYSICAL),np.diag(R_PHYSICAL),cfg.lam_ini,LIMIT,'CLARABEL',dimensions=(2,4,TI,TF))
            problem,uv,_,_=oracle.build_problem(B,u[:,None],y[:,None],target[:,None])
            problem.solve(solver='CLARABEL',tol_feas=1e-10,tol_gap_abs=1e-10,tol_gap_rel=1e-12,max_threads=1,max_iter=10000)
            self.assertIn(problem.status,('optimal','optimal_inaccurate'))
            # Low-rank saturated cases can be ill-conditioned in the explicit
            # slack formulation; check objective agreement as well as action.
            np.testing.assert_allclose(action,uv.value[:2,0],atol=1e-4,rtol=1e-4)
            predicted=B@g_direct
            cost=np.sum(np.tile(R_PHYSICAL,TF)*predicted[2*TI:2*(TI+TF)]**2)
            cost+=cfg.lam_ini*np.sum((predicted[2*(TI+TF):2*(TI+TF)+4*TI]-y)**2)
            cost+=np.sum(np.tile(Q_PHYSICAL,TF)*(predicted[-4*TF:]-target)**2)
            np.testing.assert_allclose(cost,problem.value,rtol=1e-6,atol=1e-5)

    def test_contextual_explicit_cost_and_increment_identity(self):
        for columns in (30,60,90):
            with self.subTest(columns=columns):
                self.check_contextual_cost(columns)

    def check_contextual_cost(self, columns):
        lib=self.lib;cfg=ContextualConfig(columns=columns);uh,yh,ref=self.window(123)
        pose=yh[-1,[0,1,3]]
        knownpast=(np.c_[task.frame(yh,pose),uh[-TI:]]-lib.z_mean)/lib.z_scale
        np.testing.assert_allclose(lib.zp[:,123],knownpast.ravel(),atol=1e-12)
        np.testing.assert_allclose(lib.dp[:,123],((np.diff(uh,axis=0)-lib.du_mean)/lib.du_scale).ravel(),atol=1e-12)
        ix,P,q,Z,z,U,E,b=lib.problem_arrays(uh,yh,ref,cfg)
        target=(np.c_[task.frame(ref,pose),np.zeros((TF,2))]-lib.z_mean)/lib.z_scale
        g=cp.Variable(cfg.columns)
        future=cp.reshape(lib.zf[:,ix]@g,(TF,6),order='C')
        inp=cp.reshape(U@g,(TF,2),order='C')
        delta=cp.vstack([inp[0:1]-uh[-1],inp[1:]-inp[:-1]])
        cost=cp.sum_squares(cp.multiply(future-target,np.sqrt(lib.Q)))
        cost+=cp.sum_squares(cp.multiply((inp-lib.u_mean)/lib.u_scale,np.sqrt(lib.R)))
        cost+=cp.sum_squares(cp.multiply(delta/lib.u_scale,np.sqrt(lib.R)))
        cost+=cfg.lambda_sigma*cp.norm(Z@g-z,1)+cfg.lambda_g*cp.norm(g,2)
        problem=cp.Problem(cp.Minimize(cost),[E@g==b,inp<=LIMIT,inp>=-LIMIT])
        problem.solve(solver='CLARABEL',tol_feas=1e-9,tol_gap_abs=1e-9,tol_gap_rel=1e-9,max_threads=1)
        self.assertEqual(problem.status,'optimal')
        actual,info=lib.action(uh,yh,ref,cfg);self.assertTrue(info['accepted'])
        reconstructed=np.diff(np.vstack([uh[-1],inp.value]),axis=0)/lib.u_scale
        np.testing.assert_allclose((lib.duf[:,ix]@g.value).reshape(TF,2),reconstructed,atol=1e-9)
        gv=g.value
        assembled=.5*gv@P@gv+q@gv+np.sum(lib.future_weight*target.ravel()**2)
        assembled+=cfg.lambda_sigma*np.sum(abs(Z@gv-z))+cfg.lambda_g*np.linalg.norm(gv)
        np.testing.assert_allclose(assembled,problem.value,atol=1e-8,rtol=1e-8)
        np.testing.assert_allclose(actual,inp.value[0],atol=2e-3,rtol=2e-3)

if __name__=='__main__':unittest.main()
