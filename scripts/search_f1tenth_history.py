"""Tune only K and d for the noiseless T_ini=8, T_f=10, seed-542 trial."""
import hashlib
import json
from pathlib import Path
import sys
from dataclasses import asdict

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from atlas_deepc.f1tenth import config
config.TI,config.TF=8,10
config.WARMUP_STEPS=15
from atlas_deepc.f1tenth import atlas,task,partial_selection
from atlas_deepc.f1tenth.replay import verify
from atlas_deepc.f1tenth.paths import CLEAN_OFFLINE_DATA_FILE,MAP_BASE,WAYPOINT_FILE
import numpy as np
from threadpoolctl import threadpool_limits


def main():
    cluster_order=[12,8,5,3,2,16,20,24,10,6,4,14,18,22,7,9,11,13,15,17,19,21,23]
    rank_order=[44,42,36,32,28,24,20,16,38,40,34,30,26,22,18]
    # Broad initial grid across K, then remaining ranks if needed.
    candidates=[(k,d) for d in rank_order for k in cluster_order]
    candidates += [(k,d) for d in range(17,44,2) for k in cluster_order]
    task.collect_clean()
    files=list((ROOT/'src/atlas_deepc/f1tenth').glob('*.py'))+[
        ROOT/'src/atlas_deepc/atlas/kmeans.py',ROOT/'src/atlas_deepc/control/expert_pipeline_search.py',
        Path(__file__), CLEAN_OFFLINE_DATA_FILE,WAYPOINT_FILE,MAP_BASE.with_suffix('.png'),MAP_BASE.with_suffix('.yaml')]
    protocol=dict(input_history_convention=task.INPUT_HISTORY_CONVENTION,t_ini=8,t_f=10,noise_sigma=0.,seed=542,warmup_steps=15,dt=.1,
        candidates=candidates,stopping_rule='First replay-verified collision-free full lap with more than one visited chart; ranks strictly below effective history dimension 45.',
        fixed='Same offline data, speed clustering, seed 11, partial-history projection, numerical tie rule, QP, lambda_ini=100, physical Q/R, limits, reference, initialization and seed. Only K and d varied.',
        selection_method=partial_selection.METHOD,
        hashes={str(p.relative_to(ROOT)):task.digest(p) for p in files})
    tag=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()[:12]
    output=ROOT/'experiments/f1tenth/history_projection_search'/f'ti8_tf10_sigma0_seed542_{tag}'
    task.write(output/'protocol.json',protocol)
    ledger_file=output/'search_ledger.json'
    ledger=json.loads(ledger_file.read_text()) if ledger_file.exists() else []
    if any(r.get('qualified_success') for r in ledger):
        print(f'SEARCH_DIRECTORY={output}',flush=True);return
    task.write(output/'status.json',dict(status='running',evaluated=len(ledger)))
    finished={(r['K'],r['d']) for r in ledger}
    fitted_cache={}
    with threadpool_limits(limits=1):
        library=atlas.build_library(task.collect_clean())
        for k,d in candidates:
            if (k,d) in finished:continue
            if k not in fitted_cache:
                labels,_=atlas.fit_kmeans(library[1],k,11)
                minimum=int(np.bincount(labels,minlength=k).min())
                fitted_cache[k]=atlas.fit(min(44,minimum),k,library) if minimum>=16 else None
                if fitted_cache[k] is not None:np.savez_compressed(output/f'atlas_k{k}.npz',**fitted_cache[k])
            fitted=fitted_cache[k]
            if fitted is None or d>fitted['bases'].shape[-1]:
                continue
            cfg=config.AtlasConfig(clusters=k,rank=d)
            policy=atlas.Atlas(fitted,cfg,partial_selection.METHOD)
            folder=output/f'k{k}_d{d}'
            summary=task.rollout(policy,cfg,542,save=folder,noise_sigma=0.)
            calls=json.loads((folder/'calls.json').read_text())
            selected=np.array([r['chart'] for r in calls])
            residuals=np.array(policy.selector.residual_history)
            row=dict(K=k,d=d,steps=summary['steps'],lap_fraction=summary['lap_fraction'],
                geometric_lap=summary['lap_completed'],termination=summary['termination'],rmse_cm=summary['rmse_cm'],
                charts_visited=len(set(selected.tolist())),chart_ids_one_based=(np.unique(selected)+1).tolist(),
                switches=int(np.count_nonzero(np.diff(selected))),
                tie_fraction=float(np.mean(np.array(policy.selector.tie_history)>1)),
                history_ranks=policy.selector.ranks.tolist(),
                selected_residual_median=float(np.median(residuals[np.arange(len(selected)),selected])),
                median_ms=(summary['timing'] or {}).get('median_ms'),
                collision_replay=None,verified_success=False,qualified_success=False,folder=str(folder.relative_to(output)))
            if summary['lap_completed']:
                replay=verify(folder)
                row.update(collision_replay=replay,verified_success=bool(replay['verified']))
                row['qualified_success']=row['verified_success'] and row['charts_visited']>1
            ledger.append(row);finished.add((k,d))
            task.write(ledger_file,ledger)
            task.write(output/'status.json',dict(status='complete' if row['qualified_success'] else 'running',evaluated=len(ledger),selected=row if row['qualified_success'] else None))
            print(f"K={k:2d} d={d:2d}: steps={row['steps']:3d}, lap={row['lap_fraction']:.3f}, success={row['verified_success']}, charts={row['charts_visited']}, switches={row['switches']}, RMSE={row['rmse_cm']} cm",flush=True)
            if row['qualified_success']:
                task.write(output/'selected.json',row)
                print(f'SEARCH_DIRECTORY={output}',flush=True)
                return
    task.write(output/'status.json',dict(status='exhausted',evaluated=len(ledger)))
    print(f'SEARCH_DIRECTORY={output}',flush=True)

if __name__=='__main__':main()
