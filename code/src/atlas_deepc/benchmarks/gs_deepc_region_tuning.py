"""Tune existing GS-DeePC region-count/weight parameters, with no new mechanism."""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.benchmarks import gs_deepc_scheduler_pilot as pilot

gs=pilot.gs
ROOT=gs.ROOT/'experiments/reacher/gs_deepc_region_tuning_tf10_v1'


def run(name, scheduler, regions, seeds, q=1000., r=5., lambda_g=10., lambda_ini=1e6):
    import fcntl
    if Path(name).name!=name or name in ('','.','..'):
        raise ValueError('Use a simple experiment name')
    if scheduler not in ('q2','link2') or not seeds or len(set(seeds))!=len(seeds) or not all(211<=s<=310 for s in seeds):
        raise ValueError('Use existing scheduler and unique development seeds 211--310')
    c=replace(gs.Config(),regions=regions,q_xy=q,r=r,lambda_g=lambda_g,lambda_ini=lambda_ini)
    c.validate()
    directory=ROOT/name
    directory.mkdir(parents=True,exist_ok=True)
    with (ROOT/'tuning.lock').open('a+') as lock,threadpool_limits(limits=1):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=dict(config=asdict(c),scheduler=scheduler,seeds=seeds,steps=300,
            source_sha256=gs.utils.digest(__file__),controller_sha256=gs.utils.digest(gs.__file__),
            adapter_sha256=gs.utils.digest(pilot.__file__),utilities_sha256=gs.utils.digest(gs.utils.__file__),
            metrics_sha256=gs.utils.digest(gs.ROOT/'src/atlas_deepc/common/metrics.py'),
            data_sha256=gs.utils.digest(gs.DATA),workers=1,threads=1,
            packages={p:importlib.metadata.version(p) for p in ('numpy','scipy','cvxpy','clarabel','gymnasium','mujoco')},
            scope='Existing uniform-region count and quadratic weights only. Unmodified GS controller, raw 200+200 columns, T_ini=2, T_f=10, fixed final XY target.')
        path=directory/'manifest.json'
        if path.exists() and json.loads(path.read_text())!=manifest:
            raise ValueError('Incompatible cached experiment')
        gs.utils.write_json(path,manifest)
        u,y=gs.utils.recover_episodes(np.load(gs.DATA,mmap_mode='r'))
        try:
            lib=gs.Library(u,pilot.proxy_observation(y,scheduler),c)
        except ValueError as exc:
            gs.utils.write_json(directory/'data_rejection.json',dict(reason=str(exc)))
            raise
        lib.save(directory/'offline_library.npz')
        gs.utils.write_json(directory/'data_audit.json',lib.audit)
        ctrl=pilot.ScheduledController(lib,scheduler)
        for seed in seeds:
            if not (directory/f'seed{seed}_control_results.npz').exists():
                with (directory/f'seed{seed}.log').open('w') as log,redirect_stdout(log):
                    gs.run_seed(ctrl,seed,300,directory)
            result=gs.summarize(directory,seeds)
            row=next(r for r in result['results'] if r['seed']==seed)
            print(json.dumps(dict(name=name,seed=seed,success=row['strict_success'],
                final100_mm=1000*row['mean_last_100'],fallback_steps=row['fallback_steps'],
                completed=result['completed'],successes=result['successes'])),flush=True)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True)
    p.add_argument('--scheduler',choices=['q2','link2'],default='link2')
    p.add_argument('--regions',type=int,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[211,222])
    p.add_argument('--q',type=float,default=1000.)
    p.add_argument('--r',type=float,default=5.)
    p.add_argument('--lambda-g',type=float,default=10.)
    p.add_argument('--lambda-ini',type=float,default=1e6)
    run(**vars(p.parse_args()))
