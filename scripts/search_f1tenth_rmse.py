"""Broad K/d search at fixed noiseless Ti=8, Tf=10; explicit input convention."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from atlas_deepc.f1tenth import config
config.TI,config.TF=8,10
config.WARMUP_STEPS=15
# Search support only: keep clustering, SVD, selection and QP equations fixed.
config.ATLAS_CLUSTER_COUNTS=tuple(range(1,49))
from atlas_deepc.f1tenth import atlas,task,partial_selection
from atlas_deepc.f1tenth.replay import verify
from atlas_deepc.f1tenth.paths import COMMANDED_OFFLINE_DATA_FILE,CLEAN_OFFLINE_DATA_FILE,MAP_BASE,WAYPOINT_FILE
import numpy as np
from threadpoolctl import threadpool_limits


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-mode',choices=['commanded','applied'],required=True)
    p.add_argument('--seed',type=int,default=542)
    p.add_argument('--target-rmse-cm',type=float,default=2.1)
    a=p.parse_args()
    mode='commanded' if a.input_mode=='commanded' else task.INPUT_HISTORY_CONVENTION
    record=task.collect_commanded_clean() if a.input_mode=='commanded' else task.collect_clean()
    datafile=COMMANDED_OFFLINE_DATA_FILE if a.input_mode=='commanded' else CLEAN_OFFLINE_DATA_FILE
    cluster_order=[13,12,8,5,3,2,10,6,4,14,7,9,11,16,20,24,18,22,15,17,19,21,23]
    rank_order=[42,43,36,34,32,28,24,40,38,37,35,33,30,26,22,20,18,16,44,41,39,31,29,27,25,23,21,19,17]
    # K=1 is a diagnostic baseline; require K>1 for the search target.
    candidates=[(1,d) for d in (32,36,40,42,44)]
    candidates += [(k,d) for d in rank_order for k in cluster_order]
    candidates += [(k,d) for d in rank_order for k in range(25,49)]
    files=list((ROOT/'src/atlas_deepc/f1tenth').glob('*.py'))+[
        ROOT/'src/atlas_deepc/atlas/kmeans.py',ROOT/'src/atlas_deepc/control/expert_pipeline_search.py',
        Path(__file__),datafile,WAYPOINT_FILE,MAP_BASE.with_suffix('.png'),MAP_BASE.with_suffix('.yaml')]
    protocol=dict(t_ini=8,t_f=10,dt=.1,warmup_steps=15,noise_sigma=0.,seed=a.seed,
        input_history_convention=mode,candidates=candidates,target_rmse_cm=a.target_rmse_cm,
        stopping_rule='First verified collision-free K>1 lap at or below target RMSE; otherwise exhaust feasible grid. K=1 reported as baseline only. No claim of optimality.',
        fixed='Noiseless observations, original offline episodes, seed-11 speed clustering, uncentered SVD, past-history projection, prescribed numerical tie rule, Q/R, lambda_ini=100, limits, original simulator including limiter, reference and initial conditions. Only K and d vary.',
        benchmark=dict(K=12,d=40,t_ini=5,t_f=10,sigma=.001,seed=542,rmse_cm=1.7448100398372233,
                       note='Saved paper example. Different history length and noise level; a numerical RMSE target, not a matched robustness comparison.'),
        hashes={str(p.relative_to(ROOT)):task.digest(p) for p in files})
    token=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()[:12]
    output=ROOT/'experiments/f1tenth/history_rmse_search'/f'{a.input_mode}_sigma0_seed{a.seed}_{token}'
    output.mkdir(parents=True,exist_ok=True)
    task.write(output/'protocol.json',protocol)
    for f in files:
        if f.suffix=='.py':
            dst=output/'source_snapshot'/f.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dst)
    ledger_file=output/'search_ledger.json'
    ledger=json.loads(ledger_file.read_text()) if ledger_file.exists() else []
    if any(r.get('target_reached') for r in ledger):print(f'RESULT_DIRECTORY={output}');return
    finished={(r['K'],r['d']) for r in ledger};fitted_cache={};skipped=[]
    def save_status(state):
        good=[r for r in ledger if r.get('verified_success') and r['K']>1]
        best=min(good,key=lambda r:r['rmse_cm']) if good else None
        task.write(output/'status.json',dict(status=state,evaluated=len(ledger),skipped=len(skipped),
                                            collision_free_candidates=len(good),best=best))
        if best:task.write(output/'best.json',best)
    print(f'RESULT_DIRECTORY={output}',flush=True)
    save_status('running')
    with threadpool_limits(limits=1):
        library=atlas.build_library(record)
        for k,d in candidates:
            if (k,d) in finished:continue
            if k not in fitted_cache:
                labels,_=atlas.fit_kmeans(library[1],k,11)
                minimum=int(np.bincount(labels,minlength=k).min())
                fitted_cache[k]=atlas.fit(min(44,minimum),k,library) if minimum>=16 else None
                if fitted_cache[k] is not None:np.savez_compressed(output/f'atlas_k{k}.npz',**fitted_cache[k])
            fitted=fitted_cache[k]
            if fitted is None or d>fitted['bases'].shape[-1]:
                skipped.append(dict(K=k,d=d,reason='Rank exceeds smallest chart membership'))
                task.write(output/'skipped.json',skipped);continue
            cfg=config.AtlasConfig(clusters=k,rank=d)
            policy=atlas.Atlas(fitted,cfg,partial_selection.METHOD)
            folder=output/f'k{k}_d{d}'
            summary=task.rollout(policy,cfg,a.seed,save=folder,noise_sigma=0.,input_history_convention=mode)
            calls=json.loads((folder/'calls.json').read_text())
            choices=np.array([c['chart'] for c in calls],dtype=int)
            residuals=np.array(policy.selector.residual_history)
            selected=residuals[np.arange(len(choices)),choices]
            row=dict(K=k,d=d,steps=summary['steps'],termination=summary['termination'],
                geometric_lap=summary['lap_completed'],lap_fraction=summary['lap_fraction'],rmse_cm=summary['rmse_cm'],
                verified_success=False,collision_replay=None,history_ranks=policy.selector.ranks.tolist(),
                charts_visited=len(np.unique(choices)),chart_counts=np.bincount(choices,minlength=k).tolist(),
                switches=int(np.count_nonzero(np.diff(choices))),
                tie_fraction=float(np.mean(np.array(policy.selector.tie_history)>1)),
                selected_residual_median=float(np.median(selected)),selected_residual_max=float(np.max(selected)),
                median_ms=(summary['timing'] or {}).get('median_ms'),folder=str(folder.relative_to(output)))
            if summary['lap_completed']:
                replay=verify(folder);row.update(collision_replay=replay,verified_success=bool(replay['verified']))
            row['target_reached']=bool(k>1 and row['verified_success'] and row['rmse_cm']<=a.target_rmse_cm)
            ledger.append(row);finished.add((k,d))
            task.write(ledger_file,ledger);save_status('complete' if row['target_reached'] else 'running')
            print(f"#{len(ledger):03d} K={k:2d} d={d:2d} steps={row['steps']:3d} lap={row['lap_fraction']:.3f} success={row['verified_success']} RMSE={row['rmse_cm']} charts={row['charts_visited']} switches={row['switches']}",flush=True)
            if row['target_reached']:
                print(f'TARGET_REACHED K={k}, d={d}, RMSE={row["rmse_cm"]:.6f} cm',flush=True)
                return
    save_status('exhausted')

if __name__=='__main__':main()
