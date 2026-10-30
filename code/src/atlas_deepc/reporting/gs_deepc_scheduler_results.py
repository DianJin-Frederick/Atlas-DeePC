"""Saved scheduler-pilot results and a separate fixed-input prediction audit."""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from atlas_deepc.benchmarks import gs_deepc_scheduler_pilot as pilot

gs = pilot.gs
TUNING = gs.ROOT/'experiments/reacher/gs_deepc_control_tuning_tf10_v1'
SOURCES = {
    'q2': [TUNING/'q1000_r5_g10_two_seed_320', TUNING/'validation_q1000_r5_g10_extra6'],
    'q1': [pilot.ROOT/'q1', pilot.ROOT/'validation_extra6/q1'],
    'link2': [pilot.ROOT/'link2', pilot.ROOT/'validation_extra6/link2'],
}
SEEDS = [211,219,222,237,258,281,294,305]


def verify_sources():
    configs = []
    for directories in SOURCES.values():
        for directory in directories:
            m = json.loads((directory/'manifest.json').read_text())
            assert m['controller_sha256'] == gs.utils.digest(gs.__file__)
            assert m['utilities_sha256'] == gs.utils.digest(gs.utils.__file__)
            assert m['data_sha256'] == gs.utils.digest(gs.DATA)
            assert m['steps'] == 320
            configs.append(m['config'])
    assert all(c == configs[0] for c in configs)
    return configs[0]


def prediction_audit(config):
    """Ridge/slack predictor with all future inputs given, not a control QP.

    min lambda_g ||g||^2 + lambda_ini ||Yp g - yp||^2, Hu g = u.
    No target, terminal equality or future-output penalty: this isolates bank
    predictive information, and is explicitly NOT a modified rollout controller.
    """
    u, y8 = gs.utils.recover_episodes(np.load(gs.DATA, mmap_mode='r'))
    tini, length = config['tini'], config['tini'] + config['horizon']
    banks = {}
    for name, directories in SOURCES.items():
        with np.load(directories[0]/'offline_library.npz') as a:
            banks[name] = {k:a[k].copy() for k in a.files}
    first_test_episode = 1 + max(int(b['provenance'][:,:,0].max()) for b in banks.values())
    if first_test_episode >= len(u):
        raise ValueError('No common held-out offline episodes')
    ids = [(e,t) for e in range(first_test_episode,len(u)) for t in range(0,u.shape[1]-length+1,length)]
    answer = dict(first_test_episode_zero_based=first_test_episode,
        episodes=len(u)-first_test_episode, windows=len(ids), nonoverlapping_windows=True,
        description=prediction_audit.__doc__, schedulers={})
    for name, bank in banks.items():
        rho = gs.scheduling_variable(pilot.proxy_observation(y8,name))
        grouped = [[] for _ in bank['composite_u']]
        for e,t in ids:
            i = gs.select_composite(float(rho[e,t+tini-1]),bank['edges'])
            grouped[i].append((e,t))
        first_errors, horizon_errors, persistence, inside = [], [], [], []
        max_input_residual = 0.
        for i, samples in enumerate(grouped):
            if not samples:
                continue
            hu, hy = bank['composite_u'][i], bank['composite_y'][i]
            yp_matrix, yf_matrix = hy[:2*tini],hy[2*tini:]
            n, m = hu.shape[1], hu.shape[0]
            h = config['lambda_g']*np.eye(n) + config['lambda_ini']*yp_matrix.T@yp_matrix
            kkt = np.block([[h,hu.T],[hu,np.zeros((m,m))]])
            past = np.column_stack([y8[e,t:t+tini,6:8].ravel() for e,t in samples])
            inputs = np.column_stack([u[e,t:t+length].ravel() for e,t in samples])
            rhs = np.vstack([config['lambda_ini']*yp_matrix.T@past,inputs])
            g = solve(kkt,rhs,assume_a='sym')[:n]
            max_input_residual = max(max_input_residual,float(np.max(abs(hu@g-inputs))))
            predictions = (yf_matrix@g).T.reshape(-1,config['horizon'],2)
            truth = np.stack([y8[e,t+tini:t+length,6:8] for e,t in samples])
            error = np.linalg.norm(predictions-truth,axis=2)
            first_errors.extend(error[:,0]); horizon_errors.extend(np.sqrt(np.mean(error**2,axis=1)))
            persistence.extend(np.linalg.norm(truth[:,0]-past.T[:,-2:],axis=1))
            inside.extend([bool(np.all((rho[e,t:t+length]>=bank['edges'][i]) &
                (rho[e,t:t+length]<bank['edges'][i+2]))) for e,t in samples])
        first_errors, horizon_errors = np.array(first_errors), np.array(horizon_errors)
        inside = np.array(inside)
        answer['schedulers'][name] = dict(mean_one_step_mm=float(1000*np.mean(first_errors)),
            median_one_step_mm=float(1000*np.median(first_errors)),
            mean_horizon_rmse_mm=float(1000*np.mean(horizon_errors)),
            mean_persistence_one_step_mm=float(1000*np.mean(persistence)),
            windows_staying_in_composite=int(inside.sum()),
            mean_one_step_staying_mm=float(1000*np.mean(first_errors[inside])) if inside.any() else None,
            mean_horizon_rmse_staying_mm=float(1000*np.mean(horizon_errors[inside])) if inside.any() else None,
            max_input_constraint_residual=max_input_residual)
    return answer


def export():
    with threadpool_limits(limits=1):
        config = verify_sources()
        result = dict(config=config, seeds=SEEDS, steps=320, schedulers=[],
            caveat='Eight development seeds, not a held-out success-rate estimate; scheduler choice also changes which raw windows are retained.')
        for name, directories in SOURCES.items():
            rows, predictions = {}, []
            for directory in directories:
                for seed in SEEDS:
                    path = directory/f'seed{seed}_control_results.npz'
                    if not path.exists():
                        continue
                    assert seed not in rows
                    with np.load(path) as a:
                        rows[seed] = json.loads(a['summary'].item())
                        predictions.extend(a['prediction_error_xy'].tolist())
            result['schedulers'].append(dict(scheduler=name, completed=len(rows),
                successes=sum(r['strict_success'] for r in rows.values()),
                mean_final100_mm=float(1000*np.mean([r['mean_last_100'] for r in rows.values()])),
                mean_closed_loop_prediction_mm=float(1000*np.mean(predictions)),
                fallback_steps=sum(r['fallback_steps'] for r in rows.values()),
                results=[rows[s] for s in sorted(rows)]))
        result['fixed_input_prediction_audit'] = prediction_audit(config)
        gs.utils.write_json(pilot.ROOT/'scheduler_report.json',result)
        print(json.dumps(result,indent=2))
        return result


if __name__ == '__main__':
    export()
