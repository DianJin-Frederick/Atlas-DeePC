"""Projection regularization in Atlas' reduced linear-chart coordinates.

Own Reacher data, p=8, K=64, rank=53, no graph; serial resumable lambda panels.
Uses the existing full-trace runner with process-local adapters only. The original
controller, comparison notebook, and historical results are never modified.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib.metadata
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import cvxpy as cp
import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.benchmarks import atlas_select_data as storage
from atlas_deepc.common.reacher_data import DEFAULT_DATA_SEED
from atlas_deepc.atlas.goal_reference import offline_environment_targets

experiment = storage.experiment
pipe = experiment.routed.pipe
ROOT = storage.ROOT
DATA = ROOT / "data/reacher/select_iid_offline_hankel.npy"  # OUR data despite legacy filename.
OUTPUT = Path.home() / "Projects/atlas_proj_reg_data/own_data_p8_k64_r53_t300_v1"
LAMBDAS = (0., 1., 10., 100., 1000., 10000.)
RTOL = 1e-10
VERSION = "atlas_projection_regularization_v1"
BASE_CONTROLLER = pipe.DeePCController
BASE_LOGGER = storage.comparison.logging_controller
BASE_GEOMETRY = experiment.routed.precompute_chart_matrix_cache


def check_lambda(value):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError("lambda_pi must be finite and nonnegative")
    return value


def panel_directory(root, value):
    return Path(root) / ("lambda_" + format(check_lambda(value), ".12g"))


def projection_geometry(basis, dimensions=(2,8,2,10), rtol=RTOL):
    """M=[B_up; B_yp; B_uf], Pi=M^dagger M, I-Pi=N N^T.

    Storage order is [U_p; U_f; Y_p; Y_f], NOT the row order in the screenshot.
    No ones row: g parameterizes a linear SVD chart, not affine raw columns.
    """
    b = np.asarray(basis, dtype=float)
    m,p,ti,tf = dimensions
    if b.ndim != 2 or b.shape[0] != (m+p)*(ti+tf) or not np.isfinite(b).all():
        raise ValueError("Invalid chart basis/dimensions")
    if not 0 < rtol < 1:
        raise ValueError("Invalid projector rank tolerance")
    rows = np.r_[0:m*ti, m*(ti+tf):m*(ti+tf)+p*ti, m*ti:m*(ti+tf)]
    M = b[rows]
    _, singular, vt = np.linalg.svd(M, full_matrices=True)
    threshold = rtol*singular[0] if singular.size else 0.
    rank = int(np.sum(singular > threshold))
    null = vt[rank:].T
    projection = vt[:rank].T @ vt[:rank]
    return dict(M=M, null_basis=null, projector=projection, singular_values=singular,
                rank=rank, nullity=b.shape[1]-rank, threshold=threshold)


class ProjectionController(BASE_CONTROLLER):
    """Same Atlas tracking/slack QP, plus lambda_pi ||N^T g||_2^2."""
    def __init__(self, *args, lambda_pi=0., **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_pi = check_lambda(lambda_pi)

    def solve(self, A, u_ini, y_ini, y_ref):
        # Exact baseline solve path for the control case, not a perturbed QP.
        if self.lambda_pi == 0:
            return super().solve(A, u_ini, y_ini, y_ref)
        m,p,ti,tf = self.dimensions or (pipe.M,pipe.P,pipe.T_INI,pipe.T_F)
        offset, basis, lower, upper = pipe.unpack_basis(A)
        if np.any(offset) or not isinstance(A,dict) or A.get("g_sum") is not None:
            raise ValueError("This ablation requires a zero-offset linear chart without an affine sum constraint")
        if lower is not None or upper is not None or A.get("g_nonnegative") or A.get("g_negative_weight",0):
            raise ValueError("Do not combine projection ablation with other coefficient penalties/bounds")
        if self.Q.shape != (p,p) or self.R.shape != (m,m):
            raise ValueError("Cost dimensions do not match the controller")
        null = A["projection_null_basis"]  # precomputed offline, never an online SVD
        u,y = cp.Variable((m*tf,1)),cp.Variable((p*tf,1))
        g,sigma = cp.Variable((basis.shape[1],1)),cp.Variable((p*ti,1))
        cost = self.tracking_cost(u,y,sigma,y_ref)
        if null.shape[1]:
            cost += self.lambda_pi*cp.sum_squares(null.T@g)
        limit=np.asarray(self.u_limit,dtype=float)
        if limit.ndim:
            limit=np.tile(limit.reshape(m),tf).reshape(-1,1)
        constraints=[cp.vstack([u_ini,u,y_ini+sigma,y])==basis@g,u<=limit,u>=-limit]
        problem=cp.Problem(cp.Minimize(cost),constraints)
        try:
            problem.solve(solver=self.solver,verbose=False,max_iter=10000)
        except Exception as exc:
            return None,None,None,f"crashed: {exc}",np.inf
        if problem.status not in ("optimal","optimal_inaccurate"):
            return None,None,None,problem.status,np.inf
        residual=np.linalg.norm(basis@g.value-np.vstack([u_ini,u.value,y_ini,y.value]))
        return u.value,y.value,g.value,problem.status,float(residual)


def logging_controller(_original, records, value):
    """Retain both the unregularized objective and the added penalty."""
    class Configured(ProjectionController):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,lambda_pi=value,**kwargs)
    class Logged(BASE_LOGGER(Configured,records)):
        def solve(self,A,*args,**kwargs):
            answer=super().solve(A,*args,**kwargs)
            g=answer[2]
            raw=float(np.sum((A['projection_null_basis'].T@g)**2)) if g is not None else np.nan
            row=records[-1]
            row.update(objective_without_projection=row['objective'],projection_norm_squared=raw,
                       projection_penalty=value*raw,lambda_pi=value,
                       projection_M_rank=A['projection_M_rank'],projection_nullity=A['projection_null_basis'].shape[1])
            row['objective']=row['objective_without_projection']+row['projection_penalty']
            return answer
    return Logged


class OwnDataGoal(storage.OriginalDataGoal):
    def prepare(self,target):
        super().prepare(target)
        self.fixed_selection[1]['data_goal_source_dataset']='Atlas original IID, inputs +/-1'


def prepare_assets(root, data=DATA):
    """Freeze own data, one shared atlas and exact geometry outside the project."""
    root, data=Path(root).resolve(),Path(data).resolve()
    if root.is_relative_to(ROOT.parent.resolve()):
        raise ValueError("Choose an output directory outside the atlas_deepc project")
    if not data.is_file():
        raise FileNotFoundError(f"Our offline dataset is unavailable: {data}; no dataset substitution is allowed")
    shared=root/'shared'
    shared.mkdir(parents=True,exist_ok=True)
    digest=storage.file_digest(data)
    config=storage.config()
    identity=dict(version=VERSION,data_sha256=digest,adapter_sha256=storage.file_digest(__file__),
                  storage_adapter_sha256=storage.file_digest(storage.__file__),
                  config={k:v for k,v in storage.asdict(config).items() if k!='seed'},rtol=RTOL,
                  packages={n:importlib.metadata.version(n) for n in
                            ('numpy','scipy','scikit-learn','cvxpy','clarabel','gymnasium','mujoco')})
    manifest=shared/'assets.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        if saved['identity']!=identity:
            raise ValueError("Frozen assets/configuration changed; use a fresh output root")
        for name,checksum in saved['files_sha256'].items():
            if storage.file_digest(shared/name)!=checksum:
                raise ValueError(f"Asset checksum mismatch: {name}")
        return shared,saved
    local=shared/'own_offline_hankel.npy'
    if local.exists() and storage.file_digest(local)!=digest:
        raise ValueError("Different existing offline data")
    if not local.exists():
        temporary=local.with_suffix('.pending.npy')
        shutil.copy2(data,temporary)
        temporary.replace(local)
    source=np.load(local,mmap_mode='r')
    if source.shape!=(144,37800) or not np.isfinite(source).all():
        raise ValueError("Expected our 200-episode length-12 offline data")
    # Fail explicitly if the Select-DPC +/-0.1 library is supplied by mistake.
    if np.max(np.abs(source[:24]))<=.100001 or np.max(np.abs(source[:24]))>1.000001:
        raise ValueError("Expected our +/-1 exploration data, not Select-DPC's +/-0.1 dataset")
    atlas_path,_=experiment.prepare_atlas(config,shared/'atlas',local,output_dimension=8)
    with experiment.output_context(8),threadpool_limits(limits=1):
        H=experiment.load_hankel(local,8)
        atlas=experiment.routed.load_atlas(atlas_path)
        cache=BASE_GEOMETRY(H,atlas,config,geometry='linear')
        bases=np.stack([cache[(k,)][0]['basis'] for k in range(64)])
        if bases.shape!=(64,120,53):
            raise ValueError("All charts must support rank 53")
        targets=offline_environment_targets(200)
    np.savez_compressed(shared/'source_episodes.npz',targets=targets,episode_ids=np.arange(DEFAULT_DATA_SEED,DEFAULT_DATA_SEED+200))
    diagnostics=[]
    projectors=[]
    for k,b in enumerate(bases):
        geo=projection_geometry(b)
        projectors.append(geo['projector'])
        diagnostics.append(dict(chart=k,rank=53,M_rank=geo['rank'],nullity=geo['nullity'],
                                threshold=geo['threshold'],singular_values=geo['singular_values'].tolist(),
                                annihilation_residual=float(np.linalg.norm(geo['M']@geo['null_basis']))))
    np.savez_compressed(shared/'controller_geometry.npz',bases=bases,projectors=projectors)
    storage.save_json(shared/'projection_diagnostics.json',diagnostics)
    files=['own_offline_hankel.npy','source_episodes.npz','controller_geometry.npz',
           'projection_diagnostics.json','atlas/behavioral_atlas.npz','atlas/atlas_manifest.json']
    saved=dict(identity=identity,files_sha256={f:storage.file_digest(shared/f) for f in files},
               source_dataset=str(data),source_seed_range=[2026,2225],
               M_definition='[B_up; B_yp; B_uf], no ones row; penalty on reduced g, not raw-column coefficients')
    storage.save_json(manifest,saved)
    return shared,saved


def panel_prepare(directory,steps,shared,assets,value):
    """Scoped replacement for storage.prepare; never loads Select-DPC data."""
    directory=Path(directory)
    data=shared/'own_offline_hankel.npy'
    atlas=shared/'atlas/behavioral_atlas.npz'
    cfg=storage.config(steps)
    signature=experiment.execution_signature(cfg,data,atlas,8,'terminal',
                chart_selection_method='full_window_projection',warmup_steps=12)
    signature.update(lambda_pi=value,projection_rtol=RTOL,ones_row=False,
                     adapter_sha256=storage.file_digest(__file__),assets_sha256=storage.file_digest(shared/'assets.json'),
                     storage_adapter_sha256=storage.file_digest(storage.__file__),
                     logger_sha256=storage.file_digest(storage.comparison.__file__),
                     goal_policy_sha256=storage.file_digest(storage.goals.__file__))
    protocol=dict(version=VERSION,steps=steps,seeds=list(storage.SEEDS),signature=signature,
                  data_source='our original IID dataset',reference='identical fixed-nearest rule for every lambda',
                  regularizer='lambda_pi * ||N_k.T g||_2^2, N_k spans ker([B_up;B_yp;B_uf])',
                  note='Reduced orthonormal-basis penalty, not numerically equivalent to regularizing raw-column coefficients')
    manifest=directory/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=protocol:
            raise ValueError("Lambda/protocol changed; use a fresh output directory")
    elif any(directory.glob('seed*_control_results.npz')):
        raise ValueError("Unversioned results")
    else:
        storage.save_json(manifest,protocol)
    source=shared/'source_episodes.npz'
    dest=directory/'source_episodes.npz'
    if dest.exists() and storage.file_digest(dest)!=storage.file_digest(source):
        raise ValueError("Reference provenance changed")
    if not dest.exists():
        shutil.copy2(source,dest)
    return cfg,data,atlas


def frozen_cache(shared,H,atlas,*args,**kwargs):
    with np.load(shared/'controller_geometry.npz') as a:
        bases=a['bases'].copy()
    result={}
    for k,b in enumerate(bases):
        geo=projection_geometry(b)
        result[(k,)]=(dict(offset=np.zeros((120,1)),basis=b,projection_null_basis=geo['null_basis'],
                          projection_M_rank=geo['rank']),np.flatnonzero(atlas['labels']==k))
    return result


def summarize_panel(directory):
    result=storage.report(directory)
    if not (Path(directory)/'manifest.json').exists():
        return result
    manifest=json.loads((Path(directory)/'manifest.json').read_text())
    value=manifest['signature']['lambda_pi']
    result['lambda_pi']=value
    norm_values=[]
    for row in result['results']:
        with np.load(row['result_path'],allow_pickle=True) as a:
            np.testing.assert_allclose(a['lambda_pi'],value)
            np.testing.assert_allclose(a['projection_penalty'],value*a['projection_norm_squared'],equal_nan=True)
            np.testing.assert_allclose(a['objective'],a['objective_without_projection']+a['projection_penalty'],equal_nan=True)
            norm_values.extend(a['projection_norm_squared'][np.isfinite(a['projection_norm_squared'])].tolist())
    result['median_projection_norm_squared']=float(np.median(norm_values)) if norm_values else None
    return result


def sweep_report(root=OUTPUT,lambdas=LAMBDAS):
    rows=[]
    paired=[]
    for value in lambdas:
        result=summarize_panel(panel_directory(root,value))
        online=next((x for x in result.get('timing',{}).get('modules',[]) if x['module']=='Total online control'),{})
        rows.append(dict(lambda_pi=float(value),completed=result['completed'],requested=100,
                         successes=result.get('successes'),success_rate=result.get('success_rate'),
                         median_final100_error_mm=result.get('median_final100_error_mm'),
                         median_rollout_rmse_mm=result.get('median_rollout_rmse_mm'),
                         median_online_ms=online.get('median_ms'),p95_online_ms=online.get('p95_ms'),
                         unsolved_steps=sum(x['unsolved_steps'] for x in result['results']),
                         median_projection_norm_squared=result.get('median_projection_norm_squared')))
        if value==0:
            continue
        for r in result['results']:
            baseline=panel_directory(root,0)/f"seed{r['seed']}_control_results.npz"
            if not baseline.exists():
                continue
            with np.load(baseline,allow_pickle=True) as a,np.load(r['result_path'],allow_pickle=True) as b:
                np.testing.assert_allclose(a['final_goal'],b['final_goal'],rtol=0,atol=1e-12)
                np.testing.assert_allclose(a['warmup_y'],b['warmup_y'],rtol=0,atol=1e-12)
                if len(a['distance'])!=len(b['distance']):
                    raise ValueError("Cannot compare different rollout lengths")
                paired.append(dict(lambda_pi=float(value),seed=r['seed'],
                    delta_final100_mm=float(1000*(b['distance'][-100:].mean()-a['distance'][-100:].mean()))))
    return dict(panels=rows,paired_deltas=paired)


def run(root=OUTPUT,*,lambdas=LAMBDAS,steps=300,batch_size=5,block_start=211,block_stop=310,prepare_only=False,data=DATA):
    root=Path(root).resolve()
    storage.config(steps)
    if batch_size<1 or not 211<=block_start<=block_stop<=310:
        raise ValueError('Positive batch size and a development-seed block within 211--310 required')
    values=[check_lambda(x) for x in lambdas]
    if not values or len({panel_directory(root,x) for x in values})!=len(values):
        raise ValueError("Use distinct, nonempty lambda values")
    if root.is_relative_to(ROOT.parent.resolve()):
        raise ValueError("Save ablation data outside the project")
    with storage.run_lock(root),threadpool_limits(limits=1):
        shared,assets=prepare_assets(root,data)
        if prepare_only:
            return sweep_report(root,values)
        for value in values:
            with ExitStack() as stack:
                stack.enter_context(patch.object(storage,'prepare',lambda d,s:panel_prepare(d,s,shared,assets,value)))
                stack.enter_context(patch.object(storage,'OriginalDataGoal',OwnDataGoal))
                stack.enter_context(patch.object(storage.comparison,'logging_controller',lambda o,r:logging_controller(o,r,value)))
                stack.enter_context(patch.object(experiment.routed,'precompute_chart_matrix_cache',lambda *a,**kw:frozen_cache(shared,*a,**kw)))
                storage.run(panel_directory(root,value),steps=steps,batch_size=batch_size,
                            block_start=block_start,block_stop=block_stop)
            result=summarize_panel(panel_directory(root,value))
            storage.save_json(panel_directory(root,value)/'summary.json',result)
        result=sweep_report(root,values)
        storage.save_json(root/'sweep_summary.json',result)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--lambdas',type=float,nargs='+',default=LAMBDAS)
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--batch-size',type=int,default=5)
    parser.add_argument('--block-start',type=int,default=211)
    parser.add_argument('--block-stop',type=int,default=310)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    run(args.output,lambdas=args.lambdas,steps=args.steps,batch_size=args.batch_size,
        block_start=args.block_start,block_stop=args.block_stop,prepare_only=args.prepare_only,data=args.data)


if __name__=='__main__':
    main()
