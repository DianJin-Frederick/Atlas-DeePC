"""Eight-output development variant (seeds 211--310, 300 steps).

Equation-based Reacher adaptation of Han et al., ACC 2024, pp. 5100–5105.

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
OUTPUT = ROOT / "experiments/reacher/mvc_bdpc_p8_211_310_t300_v1"
VERSION = "han_acc2024_mvc_bdpc_reacher_p8_development_v1"


@dataclass(frozen=True)
class Config:
    past: int = 6             # Paper example L=7, L_f=1.
    clusters: int = 5
    lambda_view1: float = .1
    lambda_view2: float = .1
    q_xy: float = 400.        # Reacher starting choice, not authors' Reacher tuning.
    q_joint: float = 400.     # Extend the original scalar 400 I to eight standardized outputs.
    q_dq: float = 400.        # Explicit development defaults, not tuned or paper Reacher values.
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

    @property
    def output_weights(self):
        return np.array([self.q_joint]*4 + [self.q_dq]*2 + [self.q_xy]*2)

    def validate(self):
        if self.past < 1 or self.warmup < self.past or self.clusters < 1:
            raise ValueError("Need positive past/clusters and warmup >= past")
        if not 0 <= self.svd_rtol < 1 or not 0 < self.nmf_tol < 1:
            raise ValueError("Invalid SVD cutoff or NMF tolerance")
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Configuration must be finite")
        if min(self.lambda_view1, self.lambda_view2, self.q_joint, self.q_dq, self.q_xy, self.r,
               self.lambda_epsilon, self.solver_tol, self.accept_residual) <= 0:
            raise ValueError("Weights and tolerances must be positive")
        if min(self.nmf_outer, self.nmf_inner, self.solver_max_iter) < 1:
            raise ValueError("Iteration limits must be positive")


# Share the unchanged paper algorithms with the preserved XY implementation.
from atlas_deepc.benchmarks import mvc_bdpc_reacher as legacy
hankel = legacy.hankel
truncated_subspace = legacy.truncated_subspace
subspace_distance = legacy.subspace_distance
joint_nmf = legacy.joint_nmf
summarize = legacy.summarize

class Library:
    def __init__(self, u, y8, config):
        config.validate()
        self.config = config
        u, y8 = np.asarray(u), np.asarray(y8)
        if u.ndim != 3 or y8.ndim != 3 or u.shape[:2] != y8.shape[:2] or u.shape[-1] != 2 or y8.shape[-1] != 8 or u.shape[1] <= config.past:
            raise ValueError("Expected equal-length episodes with 2 inputs and 8 physical outputs")
        if not np.isfinite(u).all() or not np.isfinite(y8).all():
            raise ValueError("Nonfinite offline samples")
        self.mean = np.r_[u.mean(axis=(0,1)), y8.mean(axis=(0,1))]
        self.scale = np.r_[u.std(axis=(0,1)), y8.std(axis=(0,1))]
        if np.any(self.scale <= 1e-12):
            raise ValueError("Offline channel has zero variance")
        trajectories = (np.concatenate([u,y8], axis=-1)-self.mean)/self.scale
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
        self.audit = {"episodes": len(u), "episode_length": u.shape[1], "physical_outputs": ["cos_q1","cos_q2","sin_q1","sin_q2","dq1","dq2","fingertip_x","fingertip_y"],
                      "all_original_samples_used": True, "g_columns": self.models.shape[-1],
                      "hankel_depth": length, "hankel_ambient_dimension": length*10,
                      "svd_ranks": self.ranks.tolist(), "svd_rtol": config.svd_rtol,
                      "representative_episode_indices": self.representatives.tolist(),
                      "distinct_representatives": len(np.unique(self.representatives)),
                      "cluster_sizes": np.bincount(self.nmf["labels"],minlength=config.clusters).tolist(),
                      "nmf_converged": True, "nmf_outer_iterations": len(self.nmf["objective"])-1,
                      "nmf_final_objective": float(self.nmf["objective"][-1]),
                      "full_row_rank_episodes": int(np.sum(self.ranks==length*10)),
                      "view2_max_distance": float(d2.max())}
        self.audit["model_geometry"]=self.geometry_audit()
        if all(r["past_rank"]==r["past_plus_future_u_rank"] for r in self.audit["model_geometry"]):
            warnings.warn("No independent future-input direction at fixed past in any model; p=8 alone has not removed this limitation")
        if len(np.unique(self.representatives)) < config.clusters:
            warnings.warn("Repeated representatives under the paper's argmax rule; retained without deduplication")

    @classmethod
    def from_path(cls, path, config):
        u,y = utils.recover_episodes(np.load(path,mmap_mode="r"))
        return cls(u,y,config)

    def geometry_audit(self):
        """Independent future-input directions after fixing the full past."""
        rows=[]
        for i,model in enumerate(self.models):
            h=model.reshape(self.config.past+1,10,-1)
            hp=h[:-1].reshape(self.config.past*10,-1); uf=h[-1,:2]
            _,singular,vt=np.linalg.svd(hp,full_matrices=False)
            basis=vt[singular>1e-8]
            residual=uf-(uf@basis.T)@basis
            rows.append({"cluster":i,"past_rank":int(np.sum(singular>1e-8)),
                "past_plus_future_u_rank":int(np.linalg.matrix_rank(np.vstack([hp,uf]),tol=1e-8)),
                "future_input_nullspace_relative_norm":float(np.linalg.norm(residual)/max(np.linalg.norm(uf),1e-15))})
        return rows

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
        if past.shape != (c.past,10):
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
    target=np.asarray(target,dtype=float)
    if target.shape!=(8,) or not np.isfinite(target).all():
        raise ValueError("Expected a finite p=8 reference")
    h=lib.models[cluster].reshape(c.past+1,10,-1)
    center=lib.centers[cluster,-c.past-1:]
    g=cp.Variable(h.shape[-1],name="g")
    up=h[:-1,:2,:].reshape(2*c.past,-1)
    yp=h[:-1,2:,:].reshape(8*c.past,-1)
    uf,yf=h[-1,:2,:],h[-1,2:,:]
    slack=yp@g-(past[:,2:]-center[:-1,2:]).ravel()
    predicted_u=uf@g+center[-1,:2]
    predicted_y=yf@g+center[-1,2:]
    yref=(target-lib.mean[2:])/lib.scale[2:]
    uref=(np.zeros(2)-lib.mean[:2])/lib.scale[:2]
    objective=(cp.norm(cp.multiply(np.sqrt(c.output_weights),predicted_y-yref),2)
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
    predicted=np.asarray(yn.value)*lib.scale[2:]+lib.mean[2:] if accepted else np.full(8,np.nan)
    return command,{"accepted":accepted,"status":str(problem.status or "solver_error"),"solver_error":error,
        "max_constraint_violation":residual,"cluster":cluster,"representative":int(lib.representatives[cluster]),
        "distance_view1":d1,"distance_view2":d2,"matching_error":scores,
        "g_coefficients":g.value.copy() if accepted else np.full(lib.models.shape[-1],np.nan),
        "past_output_slack":slack.value.copy() if accepted else np.full(8*lib.config.past,np.nan),
        "pred_y":predicted,"pred_xy":predicted[6:8],"command_u":command.copy(),"objective":float(problem.value) if accepted else float("nan"),
        "selection_time_s":selection_s,"matrix_preparation_time_s":preparation_s,"solve_time_s":solve_s,
        "control_compute_time_s":time.perf_counter()-started}


def run_seed(lib,seed,steps,output,reference8):
    reference8=np.asarray(reference8,dtype=float)
    if reference8.shape!=(8,) or not np.isfinite(reference8).all():
        raise ValueError("Expected a finite p=8 reference")
    started=time.perf_counter(); c=lib.config
    env=gym.make("Reacher-v5",max_episode_steps=c.warmup+steps)
    records,uh,yh=[],[],[]; failure=None
    try:
        obs,_=env.reset(seed=seed); initial=obs.copy(); target=obs[4:6].copy()
        for _ in range(c.warmup):
            u=np.zeros(2); obs,_,term,trunc,_=env.step(u)
            if term or trunc: raise RuntimeError("Warmup ended early")
            uh.append(u); yh.append(utils.output_from_obs(obs))
        warmup_u,warmup_y=np.array(uh),np.array(yh)
        for step in range(steps):
            command,row=compute_action(lib,np.array(uh[-c.past:]),np.array(yh[-c.past:]),reference8)
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
            tracking=np.linalg.norm(np.sqrt(c.output_weights)*(y8-reference8)/lib.scale[2:])
            effort=np.sqrt(c.r)*np.linalg.norm(action/lib.scale[:2])
            row.update(u=action,y=y8,ref=reference8.copy(),ref_xy=reference8[6:8].copy(),distance=np.linalg.norm(xy-reference8[6:8]),
                prediction_error_y=np.linalg.norm(row["pred_y"]-y8),
                prediction_error_xy=np.linalg.norm(row["pred_xy"]-xy),
                action_clip=float(np.max(np.abs(command-action))),observation_before=before,observation_after=obs,
                stage_tracking_cost=tracking,stage_input_cost=effort,stage_control_cost=tracking+effort)
            records.append(row); uh.append(action); yh.append(y8)
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
        np.savez_compressed(f,**arrays,initial_observation=initial,target_xy=target,final_goal=reference8,warmup_u=warmup_u,warmup_y=warmup_y,
                            summary=np.array(json.dumps(summary)))
    temp.replace(output/f"seed{seed}_control_results.npz")
    return summary


REFERENCE_SOURCE = ROOT / "experiments/reacher/four_baselines_k64_rank_projection_chordal_v1/selected_serial/rank053/output8/terminal"
REFERENCE_SUMMARY_SHA256 = "e6e3179ed68a04ee8d853c6259139d5d213aabb670687f52273114c2131daf19"
REFERENCE_MANIFEST_SHA256 = "193cacc035044fd4bb41fa7718279db9d7f2f0999be30b98638e18ca79b254a9"


def read_references(source=REFERENCE_SOURCE,data=DATA):
    """Recover fixed development goals from saved indices, NOT test rollouts.

    The old traces were deleted. The retained summary stores each selected
    offline column. The original policy uses that column's latest past posture,
    zero velocity, and the seeded environment target (not the sample's XY).
    No controller simulation, graph routing, reference selection or tuning.
    """
    source=Path(source)
    manifest_path=source/"manifest.json"; summary_path=source/"summary.json"
    if utils.digest(manifest_path)!=REFERENCE_MANIFEST_SHA256 or utils.digest(summary_path)!=REFERENCE_SUMMARY_SHA256:
        raise ValueError("Development reference source changed; refusing unverified references")
    manifest=json.loads(manifest_path.read_text())
    if (manifest["output_dimension"]!=8 or manifest["config"]["clusters"]!=64
            or manifest["config"]["svd_rank"]!=53
            or manifest["terminal_reference_policy"]!="fixed_nearest_eligible_fingertip"
            or utils.digest(data)!=manifest["data_sha256"]):
        raise ValueError("Wrong p8/K64/rank53 reference source or offline dataset")
    rows=json.loads(summary_path.read_text())["results"]
    if len(rows)!=100 or sorted(r["seed"] for r in rows)!=list(range(211,311)):
        raise ValueError("Reference source must contain exactly development seeds 211--310")
    h=np.load(data,mmap_mode="r")
    if h.shape!=(144,37800):
        raise ValueError("Unexpected offline Hankel layout")
    samples=h[24:].reshape(12,10,h.shape[1])[1].T[:,:8]
    references={}; indices={}
    env=gym.make("Reacher-v5")
    try:
        for row in rows:
            seed=int(row["seed"]); index=int(row["data_goal_index"])
            if not 0<=index<len(samples):
                raise ValueError("Invalid saved goal column")
            obs,_=env.reset(seed=seed); target=obs[4:6].copy()
            goal=samples[index].copy()
            if not np.isfinite(goal).all() or not np.allclose(goal[:2]**2+goal[2:4]**2,1,atol=1e-10):
                raise ValueError("Invalid offline posture")
            if not np.isclose(np.linalg.norm(goal[6:8]-target),row["data_goal_xy_error"],atol=1e-10,rtol=0):
                raise ValueError("Saved goal index and seeded target do not agree")
            goal[4:6]=0.; goal[6:8]=target
            references[seed]=goal; indices[str(seed)]=index
    finally:
        env.close()
    return references,{"source_summary":str(summary_path.relative_to(ROOT)),
        "source_summary_sha256":REFERENCE_SUMMARY_SHA256,
        "source_manifest_sha256":REFERENCE_MANIFEST_SHA256,
        "offline_data_sha256":manifest["data_sha256"],
        "atlas_sha256":manifest["atlas_sha256"],"saved_goal_indices":indices,
        "method":"Reconstructed from saved development column indices: H_y at past index 1, velocity=0, XY=seeded target",
        "control_timing_excludes_reference_setup":True}


def prepare(config,output,data=DATA,reference_provenance=None):
    config.validate(); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    signature={"version":VERSION,"config":asdict(config),"data_sha256":utils.digest(data),
               "paper":"Han et al., Big Data-driven Predictive Control Using Multi-view Clustering, ACC 2024, pp.5100–5105",
               "paper_sha256":"53e4347222752e981d8fac360ebdfc4f6741739eb1670fe7de4b1f809d508975",
               "source_sha256":utils.digest(__file__),"legacy_algorithms_sha256":utils.digest(legacy.__file__),
               "utilities_sha256":utils.digest(utils.__file__),"reference_provenance":reference_provenance,
               "output_channels":list(range(8)),"environment":"Reacher-v5","solver":"CLARABEL",
               "workers":1,"threads":1,"horizon":1,"reference":"fixed reconstructed K64/rank53 development p8 goal; physical u_ref=0",
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
    if not 211 <= seed_start <= seed_stop <= 310 or not 1 <= steps <= 300:
        raise ValueError("Development seeds must be within 211--310; steps within 1--300 (300 for the panel)")
    references,reference_provenance=read_references(data=data)
    with (output/"run.lock").open("a+") as lock, threadpool_limits(limits=1):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lib=prepare(config,output,data,reference_provenance)
        if prepare_only: return lib.audit
        seeds=list(range(seed_start,seed_stop+1))
        panel_path=output/"panel.json";panel={"seeds":seeds,"steps":steps}
        if panel_path.exists() and json.loads(panel_path.read_text())!=panel:
            raise ValueError("Panel changed; choose a new result folder")
        utils.write_json(panel_path,panel)
        try:
            for seed in seeds:
                path=output/f"seed{seed}_control_results.npz"
                if not path.exists(): run_seed(lib,seed,steps,output,references[seed])
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


if __name__ == "__main__":
    main()


