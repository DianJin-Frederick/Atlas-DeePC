"""Equation-based Reacher adaptation of Han et al., ACC 2024, pp. 5100–5105.

This is not author code. See the notebook for the under-specified conventions.
No Atlas controller, routing, contextual sampling, affine-sum or g penalty.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import time
import warnings

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import cvxpy as cp
import gymnasium as gym
import numpy as np
from scipy.spatial.distance import pdist, squareform
from threadpoolctl import threadpool_limits

# Only data decoding, environment output mapping and file/metric utilities are
# reused. No contextual controller, representation, selection or cost is reused.
from atlas_deepc.benchmarks import contextual_dpc_reacher as utils
from atlas_deepc.common.metrics import true_error_metrics

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/reacher/select_iid_offline_hankel.npy"
OUTPUT = ROOT / "experiments/reacher/mvc_bdpc_xy_211_310_v1"
VERSION = "han_acc2024_mvc_bdpc_reacher_xy_v1"


@dataclass(frozen=True)
class Config:
    past: int = 6             # Paper example L=7, L_f=1.
    clusters: int = 5
    lambda_view1: float = .1
    lambda_view2: float = .1
    q_xy: float = 400.        # Reacher starting choice, not authors' Reacher tuning.
    r: float = 1.
    lambda_epsilon: float = 1400.
    svd_rtol: float = 1e-2   # Offline-spectrum choice; paper leaves this cutoff unspecified.
    nmf_seed: int = 11
    nmf_tol: float = 1e-6
    nmf_outer: int = 3000
    nmf_inner: int = 500
    warmup: int = 12
    solver_tol: float = 1e-8
    solver_max_iter: int = 1000
    accept_residual: float = 1e-4

    def validate(self):
        if self.past < 1 or self.warmup < self.past or self.clusters < 1:
            raise ValueError("Need positive past/clusters and warmup >= past")
        if not 0 <= self.svd_rtol < 1 or not 0 < self.nmf_tol < 1:
            raise ValueError("Invalid SVD cutoff or NMF tolerance")
        if min(self.lambda_view1, self.lambda_view2, self.q_xy, self.r,
               self.lambda_epsilon, self.solver_tol, self.accept_residual) <= 0:
            raise ValueError("Weights and tolerances must be positive")
        if min(self.nmf_outer, self.nmf_inner, self.solver_max_iter) < 1:
            raise ValueError("Iteration limits must be positive")


def hankel(samples, length):
    """Eq. (4), one whole recorded trajectory, time-major/channel-minor rows."""
    a = np.asarray(samples, dtype=float)
    if a.ndim != 2 or not 1 <= length <= len(a):
        raise ValueError("Invalid trajectory or Hankel depth")
    return np.stack([a[t:t+length].reshape(-1) for t in range(len(a)-length+1)], axis=1)


def truncated_subspace(matrix, rtol):
    """Keep original column count; truncate singular values, not g coordinates."""
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    threshold = max(rtol, np.finfo(float).eps*max(matrix.shape)) * (s[0] if len(s) else 0.)
    rank = int(np.count_nonzero(s > threshold))
    return u[:, :rank], (u[:, :rank]*s[:rank]) @ vt[:rank], s


def subspace_distance(a, b):
    """Eq. (8), including the unequal-dimension penalty (not chordal distance)."""
    if a.shape[0] != b.shape[0]:
        raise ValueError("Subspaces must have the same ambient dimension")
    cosines = np.linalg.svd(a.T @ b, compute_uv=False)
    # Exact equal spaces can otherwise produce artificial angles around 1e-8.
    cosines[np.abs(cosines-1) < 1e-12] = 1.
    theta = np.arccos(np.clip(cosines, 0, 1))
    return float(np.sqrt(abs(a.shape[1]-b.shape[1])*np.pi**2/4 + theta @ theta))


def nmf_objective(views, us, vs, consensus, weights):
    return float(sum(np.sum((x-u@v.T)**2)+lam*np.sum((v-consensus)**2)
                     for x,u,v,lam in zip(views,us,vs,weights)))


def joint_nmf(raw_views, config):
    """Algorithm 1: coupled multiplicative updates and consensus, not K-means."""
    c = config
    weights = np.array([c.lambda_view1, c.lambda_view2])
    if len(raw_views) != 2:
        raise ValueError("This protocol has two views")
    views = []
    for x in raw_views:
        if not np.isfinite(x).all() or np.min(x) < 0 or np.sum(x) <= 1e-12:
            raise ValueError("A view is invalid/degenerate; inspect singular spectra/cutoff, do not silently drop a view")
        # Explicit convention: entrywise L1 norm, consistent with nonnegative NMF data.
        views.append(x / x.sum())
    n = views[0].shape[1]
    if any(x.shape[1] != n for x in views) or c.clusters > n:
        raise ValueError("Incompatible views or cluster count")
    rng = np.random.default_rng(c.nmf_seed)
    us = [rng.uniform(.5, 1.5, (x.shape[0], c.clusters)) for x in views]
    us = [u/u.sum(axis=0) for u in us]
    vs = [rng.uniform(.5, 1.5, (n, c.clusters))/(n*c.clusters) for _ in views]
    consensus = np.average(np.stack(vs), axis=0, weights=weights)
    history = [nmf_objective(views, us, vs, consensus, weights)]
    converged = False
    inner_counts = []
    floor = np.finfo(float).tiny
    for outer in range(c.nmf_outer):
        counts = []
        inner_converged = []
        for view, (x, lam) in enumerate(zip(views, weights)):
            u, v = us[view], vs[view]
            previous = np.sum((x-u@v.T)**2)+lam*np.sum((v-consensus)**2)
            inner_done = False
            for inner in range(c.nmf_inner):
                # Algorithm 1 step 3, with Q=diag(column sums of U).
                numerator = x @ v + lam*np.sum(v*consensus, axis=0)[None,:]
                denominator = u @ (v.T@v) + lam*u.sum(axis=0)[None,:]*np.sum(v*v, axis=0)[None,:]
                u *= numerator/np.maximum(denominator, floor)
                q = u.sum(axis=0)
                if np.any(q <= floor):
                    raise ValueError("NMF collapsed a factor; choose a different explicit initialization")
                u /= q
                v *= q
                # Algorithm 1 step 5, after step 4 has normalized U.
                v *= (x.T@u + lam*consensus)/np.maximum(v@(u.T@u) + lam*v, floor)
                current = np.sum((x-u@v.T)**2)+lam*np.sum((v-consensus)**2)
                if abs(current-previous) <= c.nmf_tol*max(previous, 1e-15):
                    inner_done = True
                    break
                previous = current
            counts.append(inner+1)
            inner_converged.append(inner_done)
            us[view], vs[view] = u, v
        consensus = np.average(np.stack(vs), axis=0, weights=weights)
        current = nmf_objective(views, us, vs, consensus, weights)
        if current > history[-1]+1e-10*max(history[-1], 1e-12):
            raise RuntimeError("NMF objective increased; stop rather than accept an unchecked clustering")
        history.append(current)
        inner_counts.append(counts)
        if all(inner_converged) and abs(history[-1]-history[-2]) <= c.nmf_tol*max(history[-2], 1e-15):
            converged = True
            break
    return {"views": np.stack(views), "U": np.stack(us), "V": np.stack(vs),
            "consensus": consensus, "labels": np.argmax(consensus, axis=1),
            # Paper III-A: maximum consensus loading per column, NOT an arithmetic centroid.
            "representatives": np.argmax(consensus, axis=0), "objective": np.array(history),
            "inner_iterations": np.array(inner_counts), "converged": converged}


class Library:
    def __init__(self, u, y_xy, config):
        config.validate()
        self.config = config
        u, y_xy = np.asarray(u), np.asarray(y_xy)
        if u.ndim != 3 or u.shape != y_xy.shape or u.shape[-1] != 2 or u.shape[1] <= config.past:
            raise ValueError("Expected equal-length episodes with 2 inputs and 2 fingertip outputs")
        if not np.isfinite(u).all() or not np.isfinite(y_xy).all():
            raise ValueError("Nonfinite offline samples")
        self.mean = np.r_[u.mean(axis=(0,1)), y_xy.mean(axis=(0,1))]
        self.scale = np.r_[u.std(axis=(0,1)), y_xy.std(axis=(0,1))]
        if np.any(self.scale <= 1e-12):
            raise ValueError("Offline channel has zero variance")
        trajectories = (np.concatenate([u,y_xy], axis=-1)-self.mean)/self.scale
        self.trajectories = trajectories
        length = config.past+1
        bases, matrices, spectra = [], [], []
        for trajectory in trajectories:
            basis, matrix, spectrum = truncated_subspace(hankel(trajectory,length), config.svd_rtol)
            bases.append(basis); matrices.append(matrix); spectra.append(spectrum)
        self.spectra = np.stack(spectra)
        self.ranks = np.array([b.shape[1] for b in bases])
        d1 = squareform(pdist(trajectories.reshape(len(u),-1), metric="euclidean"))
        d2 = np.zeros_like(d1)
        for i in range(len(u)):
            for j in range(i):
                d2[i,j] = d2[j,i] = subspace_distance(bases[i],bases[j])
        self.distances = np.stack([d1,d2])
        self.nmf = joint_nmf([d1,d2],config)
        if not self.nmf["converged"]:
            raise RuntimeError("Joint NMF did not converge within the declared limits; increase explicit limits, no silent acceptance")
        self.representatives = self.nmf["representatives"]
        self.centers = trajectories[self.representatives]
        self.models = np.stack([matrices[i] for i in self.representatives])
        self.audit = {"episodes": len(u), "episode_length": u.shape[1], "physical_outputs": "fingertip x/y",
                      "all_original_samples_used": True, "g_columns": self.models.shape[-1],
                      "hankel_depth": length, "hankel_ambient_dimension": length*4,
                      "svd_ranks": self.ranks.tolist(), "svd_rtol": config.svd_rtol,
                      "representative_episode_indices": self.representatives.tolist(),
                      "distinct_representatives": len(np.unique(self.representatives)),
                      "cluster_sizes": np.bincount(self.nmf["labels"],minlength=config.clusters).tolist(),
                      "nmf_converged": True, "nmf_outer_iterations": len(self.nmf["objective"])-1,
                      "nmf_final_objective": float(self.nmf["objective"][-1]),
                      "full_row_rank_episodes": int(np.sum(self.ranks==length*4)),
                      "view2_max_distance": float(d2.max())}
        if len(np.unique(self.representatives)) < config.clusters:
            warnings.warn("Repeated representatives under the paper's argmax rule; retained without deduplication")

    @classmethod
    def from_path(cls, path, config):
        u,y = utils.recover_episodes(np.load(path,mmap_mode="r"))
        return cls(u,y[:,:,6:8],config)

    def save(self, path):
        arrays = {k:getattr(self,k) for k in ["mean","scale","trajectories","spectra","ranks","distances",
                                             "representatives","centers","models"]}
        arrays.update({"nmf_"+k:v for k,v in self.nmf.items()})
        arrays["audit"] = np.array(json.dumps(self.audit))
        temporary = Path(path).with_suffix(".tmp")
        with temporary.open("wb") as f:
            np.savez_compressed(f,**arrays)
        temporary.replace(path)

    @classmethod
    def load(cls,path,config):
        lib = cls.__new__(cls); lib.config=config
        with np.load(path,allow_pickle=False) as a:
            for k in ["mean","scale","trajectories","spectra","ranks","distances","representatives","centers","models"]:
                setattr(lib,k,a[k].copy())
            lib.nmf = {k[4:]:a[k].copy() for k in a.files if k.startswith("nmf_")}
            lib.audit = json.loads(a["audit"].item())
        return lib

    def select(self, past):
        """Eq. (9), past only; never use reference/future observations to select."""
        c = self.config
        if past.shape != (c.past,4):
            raise ValueError("Wrong history shape")
        centers = self.centers[:,-c.past-1:-1,:]
        d1 = np.linalg.norm((centers-past).reshape(c.clusters,-1),axis=1)
        # Explicit missing-detail convention: H_{L_p} of the two L_p-sample
        # past segments. These are one-column Hankels, compared in one ambient space.
        basis = truncated_subspace(hankel(past,c.past),c.svd_rtol)[0]
        d2 = np.array([subspace_distance(basis,truncated_subspace(hankel(x,c.past),c.svd_rtol)[0]) for x in centers])
        def relative(d):
            return d/d.sum() if d.sum()>1e-12 else np.zeros_like(d)
        score = c.lambda_view1*relative(d1)+c.lambda_view2*relative(d2)
        return int(np.argmin(score)), d1, d2, score


def build_problem(lib, cluster, past, target):
    """Eq. (11), norms as printed; no added constraints or regularization."""
    c=lib.config
    h=lib.models[cluster].reshape(c.past+1,4,-1)
    center=lib.centers[cluster,-c.past-1:]
    g=cp.Variable(h.shape[-1],name="g")
    up=h[:-1,:2,:].reshape(2*c.past,-1)
    yp=h[:-1,2:,:].reshape(2*c.past,-1)
    uf,yf=h[-1,:2,:],h[-1,2:,:]
    slack=yp@g-(past[:,2:]-center[:-1,2:]).ravel()
    predicted_u=uf@g+center[-1,:2]
    predicted_y=yf@g+center[-1,2:]
    yref=(target-lib.mean[2:])/lib.scale[2:]
    uref=(np.zeros(2)-lib.mean[:2])/lib.scale[:2]
    objective=(cp.norm(np.sqrt(c.q_xy)*(predicted_y-yref),2)
               +cp.norm(np.sqrt(c.r)*(predicted_u-uref),2)+c.lambda_epsilon*cp.norm(slack,2))
    constraints=[up@g==(past[:,:2]-center[:-1,:2]).ravel()]
    problem=cp.Problem(cp.Minimize(objective),constraints)
    return problem,g,slack,predicted_u,predicted_y


def compute_action(lib,u_history,y_history,target):
    started=time.perf_counter()
    past=(np.c_[u_history,y_history]-lib.mean)/lib.scale
    selection_start=time.perf_counter()
    cluster,d1,d2,scores=lib.select(past)
    selection_s=time.perf_counter()-selection_start
    preparation_start=time.perf_counter()
    problem,g,slack,un,yn=build_problem(lib,cluster,past,target)
    preparation_s=time.perf_counter()-preparation_start
    solve_start=time.perf_counter(); error=""
    try:
        problem.solve(solver="CLARABEL",max_threads=1,max_iter=lib.config.solver_max_iter,
                      tol_gap_abs=lib.config.solver_tol,tol_gap_rel=lib.config.solver_tol,
                      tol_feas=lib.config.solver_tol,warm_start=False)
    except cp.error.SolverError as exc:
        error=str(exc)
    solve_s=time.perf_counter()-solve_start
    accepted=not error and problem.status in ("optimal","optimal_inaccurate")
    residual=float("nan")
    if accepted:
        residual=float(max(np.max(x.violation()) for x in problem.constraints))
        accepted=bool(np.isfinite(g.value).all() and np.isfinite(un.value).all()
                      and np.isfinite(yn.value).all() and residual<=lib.config.accept_residual)
    command=np.asarray(un.value)*lib.scale[:2]+lib.mean[:2] if accepted else np.full(2,np.nan)
    predicted=np.asarray(yn.value)*lib.scale[2:]+lib.mean[2:] if accepted else np.full(2,np.nan)
    return command,{"accepted":accepted,"status":str(problem.status or "solver_error"),"solver_error":error,
        "max_constraint_violation":residual,"cluster":cluster,"representative":int(lib.representatives[cluster]),
        "distance_view1":d1,"distance_view2":d2,"matching_error":scores,
        "g_coefficients":g.value.copy() if accepted else np.full(lib.models.shape[-1],np.nan),
        "past_output_slack":slack.value.copy() if accepted else np.full(2*lib.config.past,np.nan),
        "pred_xy":predicted,"command_u":command.copy(),"objective":float(problem.value) if accepted else float("nan"),
        "selection_time_s":selection_s,"matrix_preparation_time_s":preparation_s,"solve_time_s":solve_s,
        "control_compute_time_s":time.perf_counter()-started}


def run_seed(lib,seed,steps,output):
    started=time.perf_counter(); c=lib.config
    env=gym.make("Reacher-v5",max_episode_steps=c.warmup+steps)
    records,uh,yh=[],[],[]; failure=None
    try:
        obs,_=env.reset(seed=seed); initial=obs.copy(); target=obs[4:6].copy()
        for _ in range(c.warmup):
            u=np.zeros(2); obs,_,term,trunc,_=env.step(u)
            if term or trunc: raise RuntimeError("Warmup ended early")
            uh.append(u); yh.append(utils.output_from_obs(obs)[6:8])
        warmup_u,warmup_y=np.array(uh),np.array(yh)
        for step in range(steps):
            command,row=compute_action(lib,np.array(uh[-c.past:]),np.array(yh[-c.past:]),target)
            if not row["accepted"]:
                # Never manufacture a fallback controller or count a short run as success.
                failure={"step":step,"status":row["status"],"solver_error":row["solver_error"]}
                break
            # Eq. (11) has no bounds. Reacher's physical actuator saturates;
            # record both commands and applied actions rather than hiding saturation.
            action=np.clip(command,env.action_space.low,env.action_space.high)
            before=obs.copy(); obs,_,term,trunc,_=env.step(action)
            if term or (trunc and step!=steps-1): raise RuntimeError("Unexpected environment termination")
            y8=utils.output_from_obs(obs); xy=y8[6:8]
            tracking=np.sqrt(c.q_xy)*np.linalg.norm((xy-target)/lib.scale[2:])
            effort=np.sqrt(c.r)*np.linalg.norm(action/lib.scale[:2])
            row.update(u=action,y=y8,ref_xy=target.copy(),distance=np.linalg.norm(xy-target),
                prediction_error_xy=np.linalg.norm(row["pred_xy"]-xy),
                action_clip=float(np.max(np.abs(command-action))),observation_before=before,observation_after=obs,
                stage_tracking_cost=tracking,stage_input_cost=effort,stage_control_cost=tracking+effort)
            records.append(row); uh.append(action); yh.append(xy)
            if step==0 or (step+1)%20==0 or step==steps-1:
                utils.write_json(output/"progress.json",{"state":"running","seed":seed,"step":step+1,"steps":steps})
                print(f"seed {seed} step {step+1}/{steps}: {1000*row['distance']:.2f} mm, cluster {row['cluster']}, clip {row['action_clip']:.3g}",flush=True)
    finally:
        env.close()
    arrays={k:np.array([row[k] for row in records]) for k in records[0]} if records else {}
    distance=arrays.get("distance",np.empty(0))
    complete=len(records)==steps and failure is None
    metrics=true_error_metrics(distance) if len(distance) else {}
    for window in (30,100):
        if len(distance)<window:
            for key in (f'mean_last_{window}',f'max_last_{window}',f'success_{window}'):
                metrics.pop(key,None)
    evaluable=complete and steps>=100
    summary={"seed":seed,"requested_steps":steps,"completed_steps":len(records),"complete":complete,
             "failure":failure,"success_evaluable":evaluable,
             "strict_success":bool(evaluable and metrics["success_30"] and metrics["success_100"]),
             **metrics,"saturated_steps":int(sum(row['action_clip']>1e-8 for row in records)),
             "closed_loop_control_cost":float(sum(row['stage_control_cost'] for row in records)),
             "rollout_rmse_mm":float(1000*np.sqrt(np.mean(distance**2))) if complete else None,
             "wall_seconds":time.perf_counter()-started}
    temp=output/f"seed{seed}_control_results.tmp"
    with temp.open("wb") as f:
        np.savez_compressed(f,**arrays,initial_observation=initial,target_xy=target,warmup_u=warmup_u,warmup_y=warmup_y,
                            summary=np.array(json.dumps(summary)))
    temp.replace(output/f"seed{seed}_control_results.npz")
    return summary


def summarize(output,seeds):
    rows=[]; times={}
    for seed in seeds:
        path=Path(output)/f"seed{seed}_control_results.npz"
        if not path.exists(): continue
        with np.load(path,allow_pickle=False) as a:
            rows.append(json.loads(a["summary"].item()))
            for key in ("selection_time_s","matrix_preparation_time_s","solve_time_s","control_compute_time_s"):
                if key in a: times.setdefault(key,[]).append(a[key].copy())
    n=len(rows); eligible=bool(n and all(r['requested_steps']>=100 for r in rows))
    successes=sum(r['strict_success'] for r in rows)
    completed=[r for r in rows if r['complete']]
    final100_rows=[r for r in completed if r['completed_steps']>=100]
    payload={"finished_seeds":n,"requested_seeds":len(seeds),"panel_finished":n==len(seeds),
             "full_rollouts":len(completed),"failed_rollouts":n-len(completed),"results":rows,
             "successes":successes if eligible else None,"success_rate":successes/n if eligible else None,
             "metric_note":"Early aborted rollouts count as unsuccessful; error/RMSE aggregates below require full rollouts and list their count.",
             "median_final100_error_mm_full_rollouts":float(np.median([r['mean_last_100'] for r in final100_rows])*1000) if final100_rows else None,
             "timing_all_steps":{k:utils.timing_stats(np.concatenate(v)) for k,v in times.items()},
             "timing_discard_first5":{k:utils.timing_stats(np.concatenate([a[5:] for a in v if len(a)>5]))
                                     for k,v in times.items() if any(len(a)>5 for a in v)}}
    utils.write_json(Path(output)/"summary.json",payload)
    return payload


def prepare(config,output,data=DATA):
    config.validate(); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    signature={"version":VERSION,"config":asdict(config),"data_sha256":utils.digest(data),
               "paper":"Han et al., Big Data-driven Predictive Control Using Multi-view Clustering, ACC 2024, pp.5100–5105",
               "paper_sha256":"53e4347222752e981d8fac360ebdfc4f6741739eb1670fe7de4b1f809d508975",
               "source_sha256":utils.digest(__file__),"utilities_sha256":utils.digest(utils.__file__),
               "output_channels":[6,7],"environment":"Reacher-v5","solver":"CLARABEL",
               "workers":1,"threads":1,"horizon":1,"reference":"fixed final Cartesian target; physical u_ref=0",
               "cost":"unsquared weighted Euclidean output/input norms + unsquared L2 past-output slack",
               "constraints":"past input equality only; no coefficient penalty, sum constraint or actuator bounds",
               "actuation":"physical saturation at ±1 outside optimizer, command/applied inputs saved; abort on rejected solve",
               "conventions":{"view_normalization":"entrywise L1","svd":"relative cutoff, original Hankel width retained",
                    "online_angle":"H_past of past-length segments, one-column spaces",
                    "model":"truncated H_L(w_center) in global z-score coordinates, used on center-relative deviations as Eq.10/11 print"},
               "packages":{p:importlib.metadata.version(p) for p in ["numpy","scipy","cvxpy","clarabel","gymnasium","mujoco"]}}
    path=output/"manifest.json"
    if path.exists():
        if json.loads(path.read_text())!=signature:
            raise ValueError("Source/data/settings changed: choose a fresh output folder; existing results preserved")
    else:
        if list(output.glob("seed*_control_results.npz")) or (output/"offline_model.npz").exists():
            raise ValueError("Refusing unversioned existing artifacts")
        utils.write_json(path,signature)
    cache=output/"offline_model.npz"
    if cache.exists(): return Library.load(cache,config)
    start=time.perf_counter(); lib=Library.from_path(data,config)
    lib.audit["offline_setup_seconds"]=time.perf_counter()-start
    lib.save(cache); utils.write_json(output/"offline_audit.json",lib.audit)
    print(json.dumps({k:v for k,v in lib.audit.items() if k!='svd_ranks'},indent=2),flush=True)
    return lib


def run(config=Config(),output=OUTPUT,data=DATA,seed_start=211,seed_stop=310,steps=300,prepare_only=False):
    import fcntl
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if steps<1 or seed_stop<seed_start: raise ValueError("Invalid seed range or steps")
    with (output/"run.lock").open("a+") as lock, threadpool_limits(limits=1):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lib=prepare(config,output,data)
        if prepare_only: return lib.audit
        seeds=list(range(seed_start,seed_stop+1))
        panel_path=output/"panel.json";panel={"seeds":seeds,"steps":steps}
        if panel_path.exists() and json.loads(panel_path.read_text())!=panel:
            raise ValueError("Panel changed; choose a new result folder")
        utils.write_json(panel_path,panel)
        try:
            for seed in seeds:
                path=output/f"seed{seed}_control_results.npz"
                if not path.exists(): run_seed(lib,seed,steps,output)
                else:
                    with np.load(path) as a: row=json.loads(a['summary'].item())
                    if row['seed']!=seed or row['requested_steps']!=steps: raise ValueError("Incompatible saved seed")
                payload=summarize(output,seeds)
                print(f"{payload['finished_seeds']}/{len(seeds)} finished; successes={payload['successes']}, aborted={payload['failed_rollouts']}",flush=True)
            utils.write_json(output/"progress.json",{"state":"complete", "finished_seeds":len(seeds)})
            return payload
        except BaseException as exc:
            utils.write_json(output/"progress.json",{"state":"interrupted" if isinstance(exc,KeyboardInterrupt) else "error","message":str(exc)})
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--data",type=Path,default=DATA)
    parser.add_argument("--seed-start",type=int,default=211)
    parser.add_argument("--seed-stop",type=int,default=310)
    parser.add_argument("--steps",type=int,default=300)
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    config=Config(**json.loads(args.config.read_text())) if args.config else Config()
    run(config,args.output,args.data,args.seed_start,args.seed_stop,args.steps,args.prepare_only)


if __name__=="__main__": main()
