"""KDE-BDPC offline geometry, density/interpolation and rollout storage.

The only controller is the one-step Eq.34 implementation in paper.py.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import cvxpy as cp
import gymnasium as gym
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from comparison_methods.common import reacher_references as shared
from atlas_deepc.common.metrics import true_error_metrics

ROOT = Path(__file__).resolve().parents[3]
DATA = shared.DATA
SSD = shared.RESULT_STORE
PAPER = Path('/Users/kumiko/Library/CloudStorage/OneDrive-UW-Madison/EECS/Research_Projects/atlas_DeePC/related_work/KDE-BDPC.pdf')
SEEDS = tuple(range(211, 311))
STEPS = 300


class PlanningError(RuntimeError):
    """No verified connection; never substitute a different controller."""


@dataclass(frozen=True)
class Config:
    past: int = 2
    future: int = 1
    warmup: int = 12
    clusters: int = 4
    lambda_view1: float = .1
    lambda_view2: float = .1
    svd_rtol: float = .01
    nmf_seed: int = 42
    nmf_tol: float = 1e-6
    nmf_outer: int = 10000
    nmf_inner: int = 500
    beta_kde: float = 1.5   # Covariance B=beta_kde*I, NOT standard deviation.
    beta_bound: float = 1. # Explicit, untuned Reacher radius in standardized coordinates.
    beta_s: float = .8
    q_joint: float = 4000.
    q_dq: float = 100.
    q_xy: float = 400.
    r: float = .1
    lambda_epsilon: float = 100.
    path_max_iter: int = 100
    path_tol: float = 1e-7
    eigen_tol: float = 1e-10
    solver_tol: float = 1e-8
    solver_max_iter: int = 1000
    accept_residual: float = 1e-5

    @property
    def length(self):
        return self.past + self.future

    @property
    def output_weights(self):
        return np.array([self.q_joint]*4 + [self.q_dq]*2 + [self.q_xy]*2)

    def validate(self):
        ints = ('past','future','warmup','clusters','nmf_outer','nmf_inner',
                'path_max_iter','solver_max_iter')
        if any(type(getattr(self,k)) is not int or getattr(self,k)<1 for k in ints):
            raise ValueError('Positive integer horizons, counts and iteration limits required')
        if type(self.nmf_seed) is not int or self.nmf_seed < 0:
            raise ValueError('Invalid NMF seed')
        if self.warmup < self.length:
            raise ValueError('Interpolation needs a full measured length-L window; warmup must be >= L')
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Nonfinite configuration')
        if not 0 < self.svd_rtol < 1 or not 0 < self.beta_s < 1:
            raise ValueError('Need 0 < SVD cutoff and density ratio threshold < 1')
        for k in ('lambda_view1','lambda_view2','nmf_tol','beta_kde','beta_bound',
                  'q_joint','q_dq','q_xy','r','lambda_epsilon','path_tol','eigen_tol',
                  'solver_tol','accept_residual'):
            if getattr(self,k) <= 0:
                raise ValueError(f'{k} must be positive')


def save_npz(path, **arrays):
    path = Path(path)
    with path.with_suffix('.tmp').open('wb') as f:
        np.savez_compressed(f, **arrays)
    path.with_suffix('.tmp').replace(path)


def basis_from_samples(samples, rtol):
    """Left singular basis of a column matrix via its small Gram matrix.

    Samples are rows. Eigenspectrum is squared singular spectrum; threshold
    is rtol**2. No column subsampling or extra state identification.
    """
    gram = samples.T @ samples
    values, vectors = np.linalg.eigh((gram+gram.T)/2)
    values = np.maximum(values[::-1], 0.)
    vectors = vectors[:, ::-1]
    cutoff = max(rtol**2, np.finfo(float).eps*max(samples.shape))*values[0]
    keep = values > cutoff
    return vectors[:, keep], np.sqrt(values)


def chordal_squared(a, b):
    """Eq. 8: |r-s| + sum sin(theta)^2 = max(r,s) - tr(Pa Pb)."""
    return max(0., float(max(a.shape[1],b.shape[1])-np.sum((a.T@b)**2)))


def interpolated_basis(bases, weights, minimum_rank, maximum_rank, tol=1e-10):
    """Theorem 3 / Eq. 24: exact dimension sweep, not a fixed-rank average."""
    weights = np.asarray(weights, float)
    if len(bases)!=len(weights) or np.any(weights<0) or not np.isclose(weights.sum(),1):
        raise ValueError('Invalid interpolation weights')
    dimension = bases[0].shape[0]
    if not 1 <= minimum_rank <= maximum_rank < dimension:
        raise ValueError('Invalid interpolation rank range')
    projector = sum(w*(f@f.T) for w,f in zip(weights,bases))
    vals, vecs = np.linalg.eigh((projector+projector.T)/2)
    vals, vecs = vals[::-1], vecs[:, ::-1]
    ranks = np.arange(minimum_rank, maximum_rank+1)
    cluster_ranks = np.array([f.shape[1] for f in bases])
    costs = np.array([weights@np.maximum(cluster_ranks,r)-vals[:r].sum() for r in ranks])
    # Theorem 3 assumes positive retained eigenvalues and a boundary eigengap.
    admissible = np.array([vals[r-1]>tol and vals[r-1]-vals[r]>tol for r in ranks])
    if not admissible.any():
        raise PlanningError('No admissible rank with positive eigenvalues and an eigengap')
    idx = int(np.argmin(np.where(admissible,costs,np.inf)))
    rank = int(ranks[idx])
    return vecs[:,:rank], {'rank':rank,'cost':float(costs[idx]),
                           'candidate_ranks':ranks,'candidate_costs':costs,
                           'admissible':admissible,'eigenvalues':vals}


class Library:
    def __init__(self, u, y, config=Config()):
        config.validate(); self.config = config
        u,y = np.asarray(u,float),np.asarray(y,float)
        if (u.ndim!=3 or y.ndim!=3 or u.shape[:2]!=y.shape[:2] or
                u.shape[2]!=2 or y.shape[2]!=8 or u.shape[1]<config.length or len(u)<config.clusters):
            raise ValueError('Expected equal-length episodes with 2 inputs and 8 outputs')
        raw = np.concatenate([u,y],axis=2)
        if not np.isfinite(raw).all(): raise ValueError('Nonfinite offline data')
        self.mean,self.scale = raw.mean((0,1)),raw.std((0,1))
        if np.any(self.scale<=1e-12): raise ValueError('Zero-variance channel')
        trajectories = (raw-self.mean)/self.scale
        windows = [shared.hankel(x,config.length).T for x in trajectories]
        episode_bases = [basis_from_samples(x,config.svd_rtol)[0] for x in windows]
        d1 = squareform(pdist(trajectories.reshape(len(u),-1)))
        d2 = np.zeros_like(d1)
        for i in range(len(u)):
            for j in range(i): d2[i,j]=d2[j,i]=np.sqrt(chordal_squared(episode_bases[i],episode_bases[j]))
        nmf = shared.joint_nmf([d1,d2],config)
        if not nmf['converged']: raise RuntimeError('NMF did not converge; no model accepted')
        self.labels = nmf['labels']
        # Eq.9 hard labels define the trajectory sets. Empty NMF columns are
        # reported as inactive, never filled using unrelated representative data.
        self.active_labels = np.unique(self.labels)
        self.samples,self.bases,self.centers = [],[],[]
        spectra=[];sizes=[];columns=[]
        for label in self.active_labels:
            members = np.flatnonzero(self.labels==label)
            samples = np.concatenate([windows[j] for j in members])
            # Section 2.3: average ALL length-L segments, then subtract that
            # common L-length center from each segment before pooling.
            center = samples.mean(axis=0)
            basis,spectrum = basis_from_samples(samples-center,config.svd_rtol)
            if basis.shape[1]==0: raise ValueError('Zero-rank centered cluster')
            self.samples.append(samples);self.bases.append(basis);self.centers.append(center)
            spectra.append(spectrum);sizes.append(len(members));columns.append(len(samples))
        self.centers = np.array(self.centers)
        self.graph = {}; self.graph_failures = {}
        self._density_cache()
        self.audit = {'episodes':len(u),'episode_length':u.shape[1],
            'past':config.past,'future':config.future,'interpolation_history':config.length,
            'requested_clusters':config.clusters,'active_clusters':len(self.bases),
            'inactive_nmf_columns':sorted(set(range(config.clusters))-set(self.active_labels.tolist())),
            'cluster_episode_counts':sizes,'cluster_hankel_columns':columns,
            'cluster_ranks':[f.shape[1] for f in self.bases],
            'cluster_singular_values':[s.tolist() for s in spectra],
            'all_offline_samples_used':True,'nmf_converged':True,
            'nmf_iterations':len(nmf['objective'])-1,'nmf_final_objective':float(nmf['objective'][-1]),
            'center':'Mean of every length-L segment in each hard-assigned cluster',
            'kde':'All original standardized length-L samples; isotropic covariance beta_kde*I',
            'basis':'Low-rank orthonormal basis of pooled center-subtracted length-L segments',
            'interpolation_rank_range':[2*config.length+1,10*config.length-1]}

    def _density_cache(self):
        self.sample_norms = [np.einsum('ij,ij->i',x,x) for x in self.samples]
        self.sample_means = np.array([x.mean(0) for x in self.samples])

    def log_density(self, point, cluster, gradient=False):
        point = np.asarray(point,float)
        x = self.samples[cluster]
        distances = np.maximum(self.sample_norms[cluster]+point@point-2*x@point,0.)
        exponent = -distances/(2*self.config.beta_kde)
        normalizer = logsumexp(exponent)
        value = float(normalizer-np.log(len(x))-len(point)/2*np.log(2*np.pi*self.config.beta_kde))
        if not gradient: return value
        weights = np.exp(exponent-normalizer)
        return value,(weights@x-point)/self.config.beta_kde

    def membership(self, point):
        logs = np.array([self.log_density(point,i) for i in range(len(self.bases))])
        # Tiny floor prevents floating underflow, not a density truncation.
        weights = np.maximum(np.exp(logs-logsumexp(logs)),np.finfo(float).tiny)
        return weights/weights.sum()

    def interpolate(self, point):
        weights = self.membership(point)
        f,info = interpolated_basis(self.bases,weights,2*self.config.length+1,
                                    10*self.config.length-1,self.config.eigen_tol)
        return f,weights,info

    def entropy(self, point):
        w = self.membership(point)
        return float(-w@np.log(w))

    def first_cluster_step(self, source, target):
        """Eq.27: printed sum-log Gaussian objective; true KDE-ratio constraint.

        SLSQP solves this nonconvex constraint locally. No global-optimality
        claim; convergence AND density feasibility are checked independently.
        """
        c = self.centers[source];f = self.bases[source]
        mean = self.sample_means[target]
        threshold = self.log_density(c,source)+np.log(self.config.beta_s)
        def objective(g):
            d=c+f@g-mean
            return .5*d@d/self.config.beta_kde
        def jac(g): return f.T@(c+f@g-mean)/self.config.beta_kde
        def constraint(g): return self.log_density(c+f@g,source)-threshold
        def cjac(g): return f.T@self.log_density(c+f@g,source,True)[1]
        solution=minimize(objective,np.zeros(f.shape[1]),jac=jac,method='SLSQP',
            constraints=[{'type':'ineq','fun':constraint,'jac':cjac}],
            options={'maxiter':self.config.path_max_iter,'ftol':self.config.path_tol})
        if not solution.success or not np.isfinite(solution.x).all() or constraint(solution.x)<-self.config.accept_residual:
            raise PlanningError(f'Eq.27 connection rejected: {solution.message}')
        return c+f@solution.x


    def prepare_graph(self):
        self.graph={};self.graph_failures={}
        for i in range(len(self.bases)):
            for j in range(len(self.bases)):
                if i==j:continue
                key=f'{i}:{j}'
                try:self.graph[key]=self.connection(self.centers[i],j,source=i)
                except PlanningError as exc:self.graph_failures[key]=str(exc)
        self.audit['valid_directed_edges']=len(self.graph)
        self.audit['failed_directed_edges']=self.graph_failures
        # Incomplete connectivity is explicit, never repaired with straight lines.
        return self.graph


    def save(self,path):
        arrays={'mean':self.mean,'scale':self.scale,'centers':self.centers,
                'labels':self.labels,'active_labels':self.active_labels,
                'config':np.array(json.dumps(asdict(self.config))),
                'audit':np.array(json.dumps(self.audit)),
                'graph':np.array(json.dumps(self.graph)),
                'graph_failures':np.array(json.dumps(self.graph_failures))}
        for i,(samples,basis) in enumerate(zip(self.samples,self.bases)):
            arrays[f'samples_{i}']=samples;arrays[f'basis_{i}']=basis
        save_npz(path,**arrays)

    @classmethod
    def load(cls,path,config):
        with np.load(path,allow_pickle=False) as z:
            if json.loads(z['config'].item())!=asdict(config):raise ValueError('Cached KDE configuration mismatch')
            lib=cls.__new__(cls);lib.config=config
            for k in ('mean','scale','centers','labels','active_labels'):setattr(lib,k,z[k].copy())
            for k in ('audit','graph','graph_failures'):setattr(lib,k,json.loads(z[k].item()))
            lib.samples=[z[f'samples_{i}'].copy() for i in range(len(lib.centers))]
            lib.bases=[z[f'basis_{i}'].copy() for i in range(len(lib.centers))]
        lib._density_cache()
        return lib


def run_seed(lib,seed,steps,output,reference8,*,action_fn):
    c=lib.config;started=time.perf_counter();records=[];details=[];failure=None
    env=gym.make('Reacher-v5',max_episode_steps=c.warmup+steps)
    uh=[];yh=[]
    try:
        obs,_=env.reset(seed=seed);initial=obs.copy()
        if not np.allclose(reference8[6:8],obs[4:6],atol=1e-12,rtol=0):raise ValueError('Seed/reference mismatch')
        for _ in range(c.warmup):
            action=np.zeros(2);obs,_,terminated,truncated,_=env.step(action)
            if terminated or truncated:raise RuntimeError('Warmup ended early')
            uh.append(action);yh.append(shared.utils.output_from_obs(obs))
        warmup_u,warmup_y=np.array(uh),np.array(yh)
        initial_cluster=int(np.argmax(lib.membership(((np.c_[uh[-c.length:],yh[-c.length:]]-lib.mean)/lib.scale).ravel())))
        for step in range(steps):
            attempted=time.perf_counter()
            try:
                command,row=action_fn(lib,np.array(uh[-c.length:]),np.array(yh[-c.length:]),reference8,initial_cluster)
            except PlanningError as exc:
                failure={'step':step,'message':str(exc),'attempt_compute_time_s':time.perf_counter()-attempted};break
            action=np.clip(command,-1,1);before=obs.copy()
            obs,_,terminated,truncated,_=env.step(action)
            if terminated or (truncated and step!=steps-1):raise RuntimeError('Unexpected environment termination')
            y=shared.utils.output_from_obs(obs)
            details.append({k:row.pop(k) for k in ('route','g','failed_connections')})
            details[-1]['g']=details[-1]['g'].tolist()
            row.update(u=action,y=y,ref=reference8.copy(),distance=float(np.linalg.norm(y[6:8]-reference8[6:8])),
                prediction_error_xy=float(np.linalg.norm(row['pred_y'][6:8]-y[6:8])),
                action_clip=float(np.max(np.abs(command-action))),observation_before=before,observation_after=obs)
            records.append(row);uh.append(action);yh.append(y)
            if step==0 or (step+1)%20==0 or step==steps-1:
                shared.utils.write_json(output/'progress.json',{'state':'running','seed':seed,'step':step+1,'steps':steps})
                print(f'seed {seed} step {step+1}/{steps}: error={row["distance"]*1000:.2f} mm, rank={row["rank"]}',flush=True)
    finally:env.close()
    arrays={k:np.array([x[k] for x in records]) for k in records[0]} if records else {}
    distance=arrays.get('distance',np.empty(0));metrics=true_error_metrics(distance) if len(distance) else {}
    for window in (30,100):
        if len(distance)<window:
            for key in (f'mean_last_{window}',f'max_last_{window}',f'success_{window}'):
                metrics.pop(key,None)
    complete=len(records)==steps and failure is None
    evaluable=complete and steps>=100
    summary={'seed':seed,'requested_steps':steps,'completed_steps':len(records),'complete':complete,
        'past':c.past,'future':c.future,'interpolation_history':c.length,'failure':failure,
        'success_evaluable':evaluable,'strict_success':bool(evaluable and metrics['success_30'] and metrics['success_100']),
        **metrics,'wall_seconds':time.perf_counter()-started}
    save_npz(output/f'seed{seed}_control_results.npz',**arrays,summary=np.array(json.dumps(summary)),
             details=np.array(json.dumps(details)),final_goal=reference8,initial_observation=initial,
             warmup_u=warmup_u,warmup_y=warmup_y)
    return summary


def summarize(output,seeds):
    rows=[];times=[]
    for seed in seeds:
        path=output/f'seed{seed}_control_results.npz'
        if not path.exists():continue
        with np.load(path,allow_pickle=False) as z:
            rows.append(json.loads(z['summary'].item()))
            if 'control_compute_time_s' in z:times.extend(z['control_compute_time_s'].tolist())
    errors=[r['mean_last_100']*1000 for r in rows if r['complete'] and r['requested_steps']>=100]
    result={'finished_seeds':len(rows),'requested_seeds':len(seeds),'panel_finished':len(rows)==len(seeds),
        'full_rollouts':sum(r['complete'] for r in rows),'failed_rollouts':sum(not r['complete'] for r in rows),
        'successes':sum(r['strict_success'] for r in rows) if rows and all(r['requested_steps']>=100 for r in rows) else None,
        'success_rate_finished':float(np.mean([r['strict_success'] for r in rows])) if rows and all(r['requested_steps']>=100 for r in rows) else None,
        'median_final100_error_mm_full_rollouts':float(np.median(errors)) if errors else None,
        'recorded_control_steps':len(times),'median_control_compute_ms':float(np.median(times)*1000) if times else None,
        'timing_definition':'Pooled all accepted steps: membership, rank interpolation, online graph connections/search, model construction and solve. Excludes offline prep, reference setup, warmup, environment and disk. Failed attempts separately recorded.',
        'results':rows}
    shared.utils.write_json(output/'summary.json',result)
    return result


