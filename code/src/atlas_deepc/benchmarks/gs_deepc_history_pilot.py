"""Separate history-length diagnostic; never changes the matched T_ini=2 baseline."""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import importlib.metadata
import json

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from atlas_deepc.benchmarks import gs_deepc_scheduler_pilot as scheduler_pilot

gs = scheduler_pilot.gs
ROOT = gs.ROOT/'experiments/reacher/gs_deepc_history_pilot_tf10_v1'


def config(tini):
    return replace(gs.Config(), tini=tini, q_xy=1000., r=5., lambda_g=10., lambda_ini=1e6)


def fingerprints():
    return dict(data_sha256=gs.utils.digest(gs.DATA), controller_sha256=gs.utils.digest(gs.__file__),
        adapter_sha256=gs.utils.digest(scheduler_pilot.__file__), source_sha256=gs.utils.digest(__file__),
        utilities_sha256=gs.utils.digest(gs.utils.__file__),
        metrics_sha256=gs.utils.digest(gs.ROOT/'src/atlas_deepc/common/metrics.py'))


def fixed_input_prediction(lib, u, y, scheduler, ids):
    """Known-input regularized predictor diagnostic, NOT the tracking QP.

    IDs specify (episode, first FUTURE sample), identical across histories.
    Minimize lambda_g ||g||² + lambda_ini ||Yp g - yp||² subject to Hu g = u.
    Do not use target, terminal equality, output bounds, or future-output cost.
    """
    c = lib.config
    rho = gs.scheduling_variable(scheduler_pilot.proxy_observation(y, scheduler))
    one_step, horizon, residuals = np.empty(len(ids)), np.empty(len(ids)), []
    for i, (hu,hy) in enumerate(zip(lib.composite_u,lib.composite_y)):
        indices = [j for j,(e,t) in enumerate(ids) if gs.select_composite(float(rho[e,t-1]),lib.edges)==i]
        if not indices:
            continue
        samples = [ids[j] for j in indices]
        past = np.column_stack([y[e,t-c.tini:t,6:8].ravel() for e,t in samples])
        inputs = np.column_stack([u[e,t-c.tini:t+c.horizon].ravel() for e,t in samples])
        yp, yf = hy[:2*c.tini],hy[2*c.tini:]
        n, m = hu.shape[1],hu.shape[0]
        h = c.lambda_g*np.eye(n)+c.lambda_ini*yp.T@yp
        kkt = np.block([[h,hu.T],[hu,np.zeros((m,m))]])
        g = solve(kkt,np.vstack([c.lambda_ini*yp.T@past,inputs]),assume_a='sym')[:n]
        residuals.append(float(np.max(abs(hu@g-inputs))))
        pred = (yf@g).T.reshape(-1,c.horizon,2)
        truth = np.stack([y[e,t:t+c.horizon,6:8] for e,t in samples])
        err = np.linalg.norm(pred-truth,axis=2)
        one_step[indices] = err[:,0]
        horizon[indices] = np.sqrt(np.mean(err**2,axis=1))
    assert np.isfinite(one_step).all() and np.isfinite(horizon).all()
    assert max(residuals)<1e-6
    return dict(mean_one_step_mm=float(1000*one_step.mean()),
        mean_horizon_rmse_mm=float(1000*horizon.mean()),
        median_horizon_rmse_mm=float(1000*np.median(horizon)),
        max_input_residual=max(residuals)), one_step, horizon


def audit():
    ROOT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        u,y = gs.utils.recover_episodes(np.load(gs.DATA,mmap_mode='r'))
        libraries, unavailable = {},[]
        for scheduler in ('q2','link2'):
            for tini in (2,3,4,6,8):
                try:
                    lib = gs.Library(u,scheduler_pilot.proxy_observation(y,scheduler),config(tini))
                except ValueError as exc:
                    unavailable.append(dict(scheduler=scheduler,tini=tini,reason=str(exc)))
                    continue
                libraries[scheduler,tini] = lib
        used = set()
        for lib in libraries.values():
            used.update(lib.provenance[:,:,0].ravel().tolist())
        unused = sorted(set(range(len(u)))-used)
        if not unused:
            raise ValueError('No common unused offline episodes for paired prediction audit')
        max_tini = max(t for _,t in libraries)
        ids = [(e,t) for e in unused for t in range(max_tini,u.shape[1]-10+1,max_tini+10)]
        rows, arrays = [],dict(episode_future_start=np.asarray(ids))
        for (scheduler,tini),lib in libraries.items():
            scores,one,horizon = fixed_input_prediction(lib,u,y,scheduler,ids)
            name = f'{scheduler}_tini{tini}'
            arrays[name+'_one_step_m'],arrays[name+'_horizon_rmse_m'] = one,horizon
            rows.append(dict(scheduler=scheduler,tini=tini,**scores,
                valid_columns_per_region=lib.audit['valid_columns_per_region']))
            lib.save(ROOT/f'{name}_offline_library.npz')
        result = dict(fingerprints=fingerprints(), horizon=10, configs=[asdict(config(t)) for t in sorted(set(t for _,t in libraries))],
            paired_unused_episodes_zero_based=unused, windows=len(ids), results=rows, unavailable=unavailable,
            diagnostic=fixed_input_prediction.__doc__,
            caveat='History length changes retained windows as well as past information; not a pure observability test. No raw-column reduction or fallback bank.')
        np.savez_compressed(ROOT/'prediction_windows.npz',**arrays)
        gs.utils.write_json(ROOT/'prediction_audit.json',result)
        print(json.dumps(result,indent=2),flush=True)
        return result


def run(scheduler,tini,seeds):
    import fcntl
    if tini not in (3,4) or not seeds or len(set(seeds))!=len(seeds) or not all(211<=s<=310 for s in seeds):
        raise ValueError('Bounded pilot: T_ini=3/4, unique development seeds 211--310')
    ROOT.mkdir(parents=True,exist_ok=True)
    directory=ROOT/f"{scheduler}_tini{tini}_seeds{'_'.join(map(str,seeds))}"
    directory.mkdir(exist_ok=True)
    with (ROOT/'pilot.lock').open('a+') as lock,threadpool_limits(limits=1):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c=config(tini)
        manifest=dict(**fingerprints(),config=asdict(c),scheduler=scheduler,seeds=seeds,steps=300,
            threads=1,workers=1,packages={p:importlib.metadata.version(p) for p in ('numpy','scipy','cvxpy','clarabel','gymnasium','mujoco')})
        path=directory/'manifest.json'
        if path.exists() and json.loads(path.read_text())!=manifest:
            raise ValueError('Incompatible cached experiment')
        gs.utils.write_json(path,manifest)
        u,y=gs.utils.recover_episodes(np.load(gs.DATA,mmap_mode='r'))
        lib=gs.Library(u,scheduler_pilot.proxy_observation(y,scheduler),c)
        lib.save(directory/'offline_library.npz')
        gs.utils.write_json(directory/'data_audit.json',lib.audit)
        ctrl=scheduler_pilot.ScheduledController(lib,scheduler)
        for seed in seeds:
            if not (directory/f'seed{seed}_control_results.npz').exists():
                with (directory/f'seed{seed}.log').open('w') as log,redirect_stdout(log):
                    gs.run_seed(ctrl,seed,300,directory)
            result=gs.summarize(directory,seeds)
            print(json.dumps(dict(scheduler=scheduler,tini=tini,completed=result['completed'],
                successes=result['successes'],last=result['results'][-1])),flush=True)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',action='store_true')
    p.add_argument('--scheduler',choices=['q2','link2'],default='link2')
    p.add_argument('--tini',type=int,default=3)
    p.add_argument('--seeds',type=int,nargs='+',default=[211,222])
    a=p.parse_args()
    audit() if a.audit else run(a.scheduler,a.tini,a.seeds)
