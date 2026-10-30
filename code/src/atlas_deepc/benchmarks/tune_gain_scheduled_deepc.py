"""Bounded, explicit weight-only GS-DeePC development pilots; no automatic grid."""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.benchmarks import gain_scheduled_deepc_reacher as gs

ROOT = gs.ROOT / 'experiments/reacher/gs_deepc_control_tuning_tf10_v1'


def report(directory, seeds):
    result = gs.summarize(directory, seeds)
    rows = result['results']
    errors = [1000*r['mean_last_100'] for r in rows]
    predictions, saturation = [], []
    for row in rows:
        with np.load(directory/f"seed{row['seed']}_control_results.npz") as a:
            finite = a['prediction_error_xy'][np.isfinite(a['prediction_error_xy'])]
            predictions.extend((1000*finite).tolist())
            saturation.extend(np.any(np.abs(a['u']) >= .999, axis=1).tolist())
    result.update(mean_final100_mm=float(np.mean(errors)) if errors else None,
        median_final100_mm=float(np.median(errors)) if errors else None,
        mean_prediction_mm=float(np.mean(predictions)) if predictions else None,
        saturation_fraction=float(np.mean(saturation)) if saturation else None,
        selection_rule='Strict success count, then mean final-100 error; never compare differently weighted native costs')
    gs.utils.write_json(directory/'tuning_summary.json', result)
    return result


def run(name, seeds, steps, q, r, lambda_g, lambda_ini):
    import fcntl
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Use a simple experiment name')
    if not seeds or len(set(seeds)) != len(seeds) or not all(211<=s<=310 for s in seeds) or steps<100:
        raise ValueError('Unique development seeds 211--310 and >=100 steps required')
    ROOT.mkdir(parents=True, exist_ok=True)
    directory = ROOT/name
    directory.mkdir(exist_ok=True)
    config = replace(gs.Config(), q_xy=q, r=r, lambda_g=lambda_g, lambda_ini=lambda_ini)
    with (ROOT/'tuning.lock').open('a+') as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        library = gs.Library.from_path(config=config)
        # Check that data-selection/model settings do not depend on the weights.
        for key, value in asdict(gs.Config()).items():
            if key not in ('q_xy','r','lambda_g','lambda_ini'):
                assert getattr(config,key)==value
        manifest = dict(config=asdict(config), steps=steps, seeds=seeds,
            controller_sha256=gs.utils.digest(gs.__file__), tuner_sha256=gs.utils.digest(__file__),
            utilities_sha256=gs.utils.digest(gs.utils.__file__), data_sha256=gs.utils.digest(gs.DATA),
            packages={p:importlib.metadata.version(p) for p in ('numpy','scipy','cvxpy','clarabel','gymnasium','mujoco')},
            purpose='Weight-only development tuning; raw 400-column composite model, no graph or formulation changes',
            workers=1, threads=1)
        path=directory/'manifest.json'
        if path.exists() and json.loads(path.read_text())!=manifest:
            raise ValueError('Incompatible cached experiment; use a new name')
        gs.utils.write_json(path,manifest)
        gs.utils.write_json(directory/'data_audit.json',library.audit)
        library.save(directory/'offline_library.npz')
        controller=gs.Controller(library)
        started=time.perf_counter()
        for seed in seeds:
            if not (directory/f'seed{seed}_control_results.npz').exists():
                with (directory/f'seed{seed}.log').open('w') as log, redirect_stdout(log):
                    gs.run_seed(controller,seed,steps,directory)
            result=report(directory,seeds)
            row=next(r for r in result['results'] if r['seed']==seed)
            print(json.dumps(dict(name=name,seed=seed,success=row['strict_success'],
                final100_mm=1000*row['mean_last_100'],fallback_steps=row['fallback_steps'],
                completed=result['completed'],successes=result['successes'],
                mean_final100_mm=result['mean_final100_mm'],elapsed_s=time.perf_counter()-started)),flush=True)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[211])
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--q',type=float,default=100.)
    p.add_argument('--r',type=float,default=.05)
    p.add_argument('--lambda-g',type=float,default=1e3)
    p.add_argument('--lambda-ini',type=float,default=1e6)
    run(**vars(p.parse_args()))
