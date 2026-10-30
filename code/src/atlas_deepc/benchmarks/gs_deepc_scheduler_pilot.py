"""Controlled scalar-scheduler experiments, separate from the frozen GS port.

Only the scalar scheduling coordinate changes. A proxy observation passes that
coordinate through the original q2 interface; raw inputs and fingertip outputs
are untouched. The original controller and its QP are reused without modification.
"""
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

ROOT = gs.ROOT / 'experiments/reacher/gs_deepc_scheduler_pilot_tf10_v1'


def proxy_observation(y8, scheduler):
    y = np.asarray(y8)
    if scheduler == 'q2':
        return y
    proxy = y.copy()
    q1 = np.arctan2(y[..., 2], y[..., 0])
    if scheduler == 'q1':
        rho = q1
    elif scheduler == 'link2':
        rho = q1 + np.arctan2(y[..., 3], y[..., 1])
    else:
        raise ValueError(scheduler)
    proxy[..., 1], proxy[..., 3] = np.cos(rho), np.sin(rho)
    return proxy


class ScheduledController(gs.Controller):
    def __init__(self, library, scheduler):
        super().__init__(library)
        self.scheduler = scheduler

    def action(self, u_history, y8_history, target):
        start = time.perf_counter()
        proxy = proxy_observation(y8_history, self.scheduler)
        overhead = time.perf_counter() - start
        action, row = super().action(u_history, proxy, target)
        row['scheduler_value'] = row.pop('scheduler_q2')
        row['scheduler_q2'] = float(gs.scheduling_variable(y8_history[-1]))
        row['selection_time_s'] += overhead
        row['control_compute_time_s'] = time.perf_counter() - start
        return action, row


def audit_geometry(library, y8):
    rows = []
    length = library.config.tini + library.config.horizon
    for i, ids in enumerate(library.provenance):
        samples = np.concatenate([y8[e, t:t+length] for e, t in ids])
        q1 = np.arctan2(samples[:, 2], samples[:, 0])
        rows.append(dict(region=i,
            q1_circular_resultant=float(abs(np.mean(np.exp(1j*q1)))),
            xy_std_mm=(1000*np.std(samples[:, 6:8], axis=0)).tolist(),
            source_episode_min=int(ids[:, 0].min()), source_episode_max=int(ids[:, 0].max())))
    return dict(regions=rows, caveat='Broad geometry is a diagnostic, not proof of predictive invalidity. Overlapping windows are not independent samples.')


def run(scheduler, seeds, steps):
    import fcntl
    if steps < 100 or not seeds or len(set(seeds)) != len(seeds) or not all(211 <= s <= 310 for s in seeds):
        raise ValueError('Use unique development seeds 211--310 and >=100 steps')
    config = replace(gs.Config(), q_xy=1000., r=5., lambda_g=10., lambda_ini=1e6)
    directory = ROOT / scheduler
    directory.mkdir(parents=True, exist_ok=True)
    with (ROOT/'pilot.lock').open('a+') as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = dict(scheduler=scheduler, config=asdict(config), seeds=seeds, steps=steps,
            data_sha256=gs.utils.digest(gs.DATA), controller_sha256=gs.utils.digest(gs.__file__),
            adapter_sha256=gs.utils.digest(__file__), utilities_sha256=gs.utils.digest(gs.utils.__file__),
            packages={p: importlib.metadata.version(p) for p in ('numpy','scipy','cvxpy','clarabel','gymnasium','mujoco')},
            workers=1, threads=1, purpose='Scalar scheduler-only Reacher adaptation pilot; no QP changes')
        manifest_path = directory/'manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Incompatible cached experiment')
        gs.utils.write_json(manifest_path, manifest)
        u, y8 = gs.utils.recover_episodes(np.load(gs.DATA, mmap_mode='r'))
        library = gs.Library(u, proxy_observation(y8, scheduler), config)
        # Independently verify the proxy has not altered any retained raw data.
        length = config.tini + config.horizon
        for i, ids in enumerate(library.provenance):
            np.testing.assert_array_equal(library.hu[i], np.column_stack([u[e,t:t+length].ravel() for e,t in ids]))
            np.testing.assert_array_equal(library.hy[i], np.column_stack([y8[e,t:t+length,6:8].ravel() for e,t in ids]))
        library.save(directory/'offline_library.npz')
        gs.utils.write_json(directory/'data_audit.json', library.audit)
        gs.utils.write_json(directory/'geometry_audit.json', audit_geometry(library, y8))
        controller = ScheduledController(library, scheduler)
        for seed in seeds:
            if not (directory/f'seed{seed}_control_results.npz').exists():
                with (directory/f'seed{seed}.log').open('w') as log, redirect_stdout(log):
                    gs.run_seed(controller, seed, steps, directory)
            result = gs.summarize(directory, seeds)
            print(json.dumps(dict(scheduler=scheduler, completed=result['completed'],
                successes=result['successes'], last=result['results'][-1])), flush=True)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scheduler', choices=['q2','q1','link2'], required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[211,222])
    parser.add_argument('--steps', type=int, default=300)
    run(**vars(parser.parse_args()))
