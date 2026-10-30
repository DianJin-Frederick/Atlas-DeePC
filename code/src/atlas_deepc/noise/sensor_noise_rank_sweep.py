"""Ranks 55--65, six fixed physical-sensor noise levels, 100 paired seeds."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from atlas_deepc.common.paths import source_path, project_root
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
ROOT=project_root()
RESULT=ROOT/'experiments/routed_deepc_iid/physical_sensor_ranks55_65_100_v1'
RANKS=list(range(55,66)); LEVELS=[0,1,2,3,4,5]; SEEDS=list(range(211,311))
NEW_RANKS=[r for r in RANKS if r not in (59,62)]
VERSION='physical_sensor_ranks55_65_100_v1'
SOURCES=['measurement_sensor_noise_core.py','measurement_sensor_noise_rank59.py',
         'measurement_sensor_noise_rank62.py','baseline_atlas_deepc.py','routed_deepc_experiment.py',
         'candidate_goal_routing_experiment.py','expert_pipeline_search.py','baseline_1000_benchmark.py',
         'reacher_data.py','goal_reference.py']


def utc():return datetime.now(timezone.utc).isoformat()
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(p)
def level_name(level):return f'multiplier_{level:.4f}'.replace('.','p')
def source_folder(rank,level):return ROOT/f'experiments/routed_deepc_iid/measurement_sensor_noise_rank{rank}_lamg005_100_v1'/level_name(level)
def folder(rank,level):return RESULT/f'rank{rank:03d}'/level_name(level)
def path_for(rank,level,seed):return (source_folder(rank,level) if rank in (59,62) else folder(rank,level))/f'seed{seed}_control_results.npz'


@contextmanager
def lock():
    import fcntl
    RESULT.mkdir(parents=True,exist_ok=True)
    with (RESULT/'run.lock').open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A sweep is already running; do not launch a duplicate')
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def freeze():
    frozen=RESULT/'frozen_protocol';manifest=RESULT/'manifest.json'
    packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','cvxpy','clarabel','gymnasium','mujoco','scikit-learn']}
    if manifest.exists():
        m=read(manifest)
        assert m['packages']==packages,'Numerical environment changed'
        assert m['runner_sha256']==digest(__file__),'Sweep runner changed'
        for rel,sha in m['frozen_sha256'].items():assert digest(frozen/rel)==sha,rel
        return m
    # Freeze code and clean data before importing the numerical implementation.
    # New snapshots preserve package-relative imports and all shared helpers.
    shutil.copytree(ROOT/'src/atlas_deepc', frozen/'src/atlas_deepc',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'), dirs_exist_ok=True)
    data_rel='data/select_iid_offline_hankel.npy'
    dest=frozen/data_rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/data_rel,dest)
    clean_rel='experiments/routed_deepc_iid/clustering_jointconfig48_fresh211_310/behavioral_atlas.npz'
    dest=frozen/clean_rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/clean_rel,dest)
    for level in LEVELS:
        for name in ['artifact_manifest.json','behavioral_atlas.npz','chart_cache.npz']:
            dest=frozen/'noise_atlases'/level_name(level)/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source_folder(59,level)/name,dest)
    m={'version':VERSION,'ranks':RANKS,'levels':LEVELS,'seeds':SEEDS,'packages':packages,
       'runner_sha256':digest(__file__),'created_utc':utc(),'workers':1,'numerical_threads':1,
       'sensor_settings':{'angle_sd_deg':.1,'position_sd_mm':1.,'velocity_sd_rad_s':.01},
       'reuse_ranks':[59,62],'new_ranks':NEW_RANKS,'expected_rollouts':6600,'new_rollouts':5400,
       'order':'level, seed, rotating new-rank order within each seed',
       'frozen_sha256':{str(p.relative_to(frozen)):digest(p) for p in sorted(frozen.rglob('*')) if p.is_file()}}
    write(manifest,m);return m


def modules():
    frozen_src = RESULT/'frozen_protocol/src'
    sys.path.insert(0,str(frozen_src))
    import numpy as np
    if (frozen_src/'atlas_deepc').is_dir():
        # This dedicated sweep process must not mix live and snapshotted packages.
        for name in list(sys.modules):
            if name == 'atlas_deepc' or name.startswith('atlas_deepc.'):
                del sys.modules[name]
        from atlas_deepc.noise import measurement_sensor_noise_core as noise
        from atlas_deepc.noise import measurement_sensor_noise_rank59 as old
        from atlas_deepc.benchmarks.baseline_1000_benchmark import summarize_timing
    else:
        # Pre-reorganization archives retain their original flat source layout.
        import measurement_sensor_noise_core as noise
        import measurement_sensor_noise_rank59 as old
        from baseline_1000_benchmark import summarize_timing
    noise.configure(.1,1.,.01)
    return np,noise,old,summarize_timing


def validate_trace(path,rank,level,seed,np,noise,expected_config,reference=None):
    with np.load(path,allow_pickle=True) as t:
        row=json.loads(t['summary'].item())
        for k,v in asdict(replace(expected_config,seed=seed,svd_rank=rank)).items():assert row[k]==v,(path,k)
        assert row['noise_model']=='physical_sensor_gaussian_v1' and row['noise_multiplier']==level
        assert row['sensor_settings']==noise.sensor_settings()
        assert row['graph_waypoints_enabled'] and row['candidate_count']==10 and row['matrix_geometry']=='linear'
        assert row['evaluation_uses_true_outputs'] and not row['noise_on_inputs']
        assert t['sensor_true'].shape==(303,10) and t['distance'].shape==(300,)
        z=np.random.default_rng(np.random.SeedSequence([noise.ONLINE_NOISE_SEED,seed])).normal(size=(303,6))
        np.testing.assert_array_equal(t['sensor_standard_normal'],z)
        np.testing.assert_allclose(t['sensor_measured'],noise.corrupt_outputs(t['sensor_true'],level,z),rtol=0,atol=1e-14)
        np.testing.assert_allclose(t['sensor_noise_sigma'],np.broadcast_to(noise.measurement_sigma(level),(303,6)))
        np.testing.assert_array_equal(t['y'],t['sensor_true'][-300:])
        np.testing.assert_array_equal(t['y_measured'],t['sensor_measured'][-300:])
        distance=np.linalg.norm(t['y'][:,6:8]-row['exact_target_xy'],axis=1)
        np.testing.assert_allclose(t['distance'],distance)
        assert np.isfinite(distance).all()
        for window in (30,100):
            tail=distance[-window:]
            np.testing.assert_allclose([row[f'mean_last_{window}'],row[f'max_last_{window}']],[tail.mean(),tail.max()])
            assert row[f'success_{window}']==bool(tail.mean()<.01 and tail.max()<.02)
        from collections import Counter
        assert dict(Counter(str(v) for v in t['status']))==row['statuses']
        assert sum(row['statuses'].values())==300
        assert row['unsolved_steps']==sum(v for k,v in row['statuses'].items() if k not in ('optimal','optimal_inaccurate'))
        for key in ['localization_time_s','goal_selection_time_s','routing_time_s','matrix_preparation_time_s','solve_time_s','control_compute_time_s']:
            assert t[key].shape==(300,) and np.isfinite(t[key]).all() and (t[key]>=0).all()
        if reference is not None:
            with np.load(reference,allow_pickle=True) as ref:
                r=json.loads(ref['summary'].item())
                np.testing.assert_array_equal(row['exact_target_xy'],r['exact_target_xy'])
                np.testing.assert_array_equal(t['sensor_true'][0],ref['sensor_true'][0])
    row['result_path']=str(path.resolve())
    return row


def audit_old(m,np,noise,old):
    path=RESULT/'historical_audit.json'
    if path.exists():
        audit=read(path)
        for rel,sha in audit['sha256'].items():assert digest(ROOT/rel)==sha,rel
        return audit
    hashes={};counts={}
    config=old.experiment_config()
    for level in LEVELS:
        a=read(source_folder(59,level)/'artifact_manifest.json')
        sig=a['signature']
        assert sig['sensor_settings']==noise.sensor_settings() and sig['config']=={k:v for k,v in asdict(config).items() if k!='seed'}
        assert sig['data_sha256']==digest(noise.baseline.DATA_PATH)
        assert sig['clean_atlas_sha256']==digest(noise.baseline.ATLAS_PATH)
        for name,sha in sig['source_sha256'].items():assert digest(RESULT/'frozen_protocol/src'/name)==sha,name
        for rank in (59,62):
            src=source_folder(rank,level);art=read(src/'artifact_manifest.json')
            assert art['signature']['sensor_settings']==noise.sensor_settings()
            cfg=dict(sig['config'],svd_rank=rank);assert art['signature']['config']==cfg
            if rank==62:
                assert art['signature']['rank59_artifact_manifest_sha256']==digest(source_folder(59,level)/'artifact_manifest.json')
                assert art['signature']['rank62_runner_sha256']==digest(RESULT/'frozen_protocol/src/measurement_sensor_noise_rank62.py')
            for name,sha in art['artifact_sha256'].items():assert digest(src/name)==sha
            for name in ['artifact_manifest.json','summary.json','behavioral_atlas.npz','chart_cache.npz']:
                p=src/name;hashes[str(p.relative_to(ROOT))]=digest(p)
            summary=read(src/'summary.json');assert summary['seeds']==SEEDS and len(summary['results'])==100
            count=0
            for seed in SEEDS:
                p=path_for(rank,level,seed)
                row=validate_trace(p,rank,level,seed,np,noise,config,path_for(59,level,seed))
                saved=summary['results'][seed-211]
                assert {k:v for k,v in row.items() if k!='result_path'}=={k:v for k,v in saved.items() if k!='result_path'}
                hashes[str(p.relative_to(ROOT))]=digest(p);count+=1
            counts[f'{rank}/{level}']=count
    audit={'checked_utc':utc(),'valid_rollouts':1200,'counts':counts,'sha256':hashes}
    write(path,audit);return audit


def prepare(level,np,noise,old):
    start=time.perf_counter();base=RESULT/'offline'/level_name(level);base.mkdir(parents=True,exist_ok=True)
    H,sigma,eps=noise.noisy_library(np.load(noise.baseline.DATA_PATH,mmap_mode='r'),level)
    source=RESULT/'frozen_protocol/noise_atlases'/level_name(level)
    atlas=noise.routed.load_atlas(source/'behavioral_atlas.npz')
    cache_path=base/'rank65_cache.npz';meta=base/'manifest.json'
    if meta.exists():
        saved=read(meta);assert digest(cache_path)==saved['cache_sha256']
    else:
        controller=noise.baseline.BaselineAtlasDeePC(H,atlas,replace(old.experiment_config(),svd_rank=65))
        matrices=[controller.matrix_cache[(k,)][0] for k in range(48)]
        cache={key:np.stack([v[key] for v in matrices]) for key in ['basis','offset','g_min','g_max']}
        assert cache['basis'].shape==(48,144,65) and not np.any(cache['offset'])
        for rank in (59,62):
            with np.load(source_folder(rank,level)/'chart_cache.npz') as prior:
                np.testing.assert_allclose(cache['basis'][:,:,:rank],prior['basis'],rtol=1e-10,atol=1e-10)
                for key in ['g_min','g_max']:np.testing.assert_allclose(cache[key][:,:rank],prior[key],rtol=1e-10,atol=1e-10)
        np.savez_compressed(cache_path,**cache)
        saved={'level':level,'rank':65,'cache_sha256':digest(cache_path),'atlas_sha256':digest(source/'behavioral_atlas.npz'),
               'source_atlas_reused':True,'nested_prefix_59_62_verified':True,'setup_seconds':time.perf_counter()-start,'utc':utc()}
        write(meta,saved)
    with np.load(cache_path) as z:cache={k:z[k] for k in z.files}
    # Prefix slices preserve the original SVD coefficient bounds exactly.
    controllers={}
    routes=noise.shortest_path_tables(atlas,old.experiment_config().edge_count_threshold)
    for rank in NEW_RANKS:
        c=object.__new__(noise.baseline.BaselineAtlasDeePC)
        c.config=replace(old.experiment_config(),svd_rank=rank);c.H=H;c.atlas=atlas;c.route_tables=routes
        c.matrix_cache={(k,):({'basis':cache['basis'][k,:,:rank],'offset':cache['offset'][k],
                              'g_min':cache['g_min'][k,:rank],'g_max':cache['g_max'][k,:rank]},
                             np.flatnonzero(atlas['labels']==k)) for k in range(48)}
        controllers[rank]=c
    return controllers


def aggregate(np,noise,old,summarize_timing,require_complete=False):
    panels=[];completed=0;failures=0
    for level in LEVELS:
        for rank in RANKS:
            rows=[];paths=[]
            for seed in SEEDS:
                p=path_for(rank,level,seed)
                if p.exists():
                    row=validate_trace(p,rank,level,seed,np,noise,old.experiment_config(),path_for(59,level,seed))
                    rows.append(row);paths.append(p)
            completed+=len(rows)
            if not rows:continue
            errors=np.array([r['mean_last_100'] for r in rows])*1000
            failed=sum(r['unsolved_steps'] for r in rows);failures+=failed
            panel={'rank':rank,'multiplier':level,'n':len(rows),'complete':len(rows)==100,
                   'successes':sum(r['success_30'] and r['success_100'] for r in rows),
                   'median_error_mm':float(np.median(errors)),'p90_error_mm':float(np.percentile(errors,90)),
                   'p95_error_mm':float(np.percentile(errors,95)),'p99_error_mm':float(np.percentile(errors,99)),
                   'unsolved_qps':failed,'session':'historical' if rank in (59,62) else 'expanded',
                   'timing':summarize_timing(paths),'results':rows}
            write(folder(rank,level)/'summary.json',panel);panels.append(panel)
    result={'version':VERSION,'updated_utc':utc(),'ranks':RANKS,'levels':LEVELS,'seeds':SEEDS,'rollouts':completed,
            'complete':completed==6600,'unsolved_qps':failures,'panels':panels,
            'timing_note':'Ranks 59/62 reused from earlier sessions; other ranks use one new serial session. Cross-session timing differences are not attributable to rank alone.'}
    if require_complete:assert result['complete']
    write(RESULT/'summary.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    with lock():
        m=freeze();np,noise,old,timing=modules()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            audit_old(m,np,noise,old)
            if args.report_only:aggregate(np,noise,old,timing);return
            write(RESULT/'launch.json',{'pid':os.getpid(),'started_utc':utc(),'arguments':sys.argv,'python':sys.executable})
            controllers=None;new_done=sum(path_for(r,l,s).exists() for r in NEW_RANKS for l in LEVELS for s in SEEDS)
            for level in LEVELS:
                controllers=prepare(level,np,noise,old)
                print(f'Prepared level {level}x: same noisy data/atlas, nested rank65 cache verified',flush=True)
                if args.preflight:continue
                for index,seed in enumerate(SEEDS):
                    order=NEW_RANKS[index%len(NEW_RANKS):]+NEW_RANKS[:index%len(NEW_RANKS)]
                    for rank in order:
                        p=path_for(rank,level,seed);p.parent.mkdir(parents=True,exist_ok=True)
                        if p.exists():validate_trace(p,rank,level,seed,np,noise,old.experiment_config(),path_for(59,level,seed));continue
                        noise._controller=controllers[rank];noise._ratio=level;noise.VERSION=VERSION
                        start=time.perf_counter();noise.run_seed(seed,p.parent)
                        row=validate_trace(p,rank,level,seed,np,noise,old.experiment_config(),path_for(59,level,seed))
                        new_done+=1
                        write(RESULT/'progress.json',{'utc':utc(),'pid':os.getpid(),'new_completed':new_done,'reused':1200,'completed':new_done+1200,'total':6600,
                                                     'last_rank':rank,'last_level':level,'last_seed':seed,'last_rollout_seconds':time.perf_counter()-start})
                        if new_done%10==0 or new_done<10:print(f'{new_done}/5400 new | level={level} rank={rank} seed={seed} success={row["success_30"] and row["success_100"]}',flush=True)
                    if index%10==0:freeze()  # Recheck frozen data/code hashes and numerical versions.
                controllers=None
                if not args.preflight:aggregate(np,noise,old,timing)
            if args.preflight:
                write(RESULT/'preflight.json',{'passed':True,'utc':utc(),'historical_rollouts':1200,'levels':LEVELS,'rank65_caches_verified':True})
                print('PREFLIGHT PASSED; no new rollouts run',flush=True);return
            freeze();audit_old(m,np,noise,old)
            write(RESULT/'invariance_final.json',{'passed':True,'utc':utc(),'frozen_protocol_and_historical_checks_passed':True})
            aggregate(np,noise,old,timing,True)
            print('COMPLETE: 6600 validated rollouts; no 10x panels',flush=True)

if __name__=='__main__':main()
