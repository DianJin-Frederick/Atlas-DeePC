"""Read-only experiment audit and lossless error/timing export for KDE-BDPC.

Never runs a simulator or controller. Original manifests and traces are retained.
Repeated appearances in screening stages are not counted as new rollouts.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

PARAMETERS = {
    'clusters': 'Requested cluster count K',
    'nmf_seed': 'NMF initialization seed',
    'lambda_view1': 'MVC consensus penalty, view 1',
    'lambda_view2': 'MVC consensus penalty, view 2',
    'svd_rtol': 'Relative offline SVD cutoff',
    'beta_kde': 'KDE covariance multiplier',
    'beta_bound': 'Local-step / coefficient radius',
    'beta_s': 'Connection density-ratio threshold',
    'lambda_epsilon': 'Past-mismatch penalty',
    'q_joint': 'Joint-output tracking weight',
    'q_dq': 'Velocity tracking weight',
    'q_xy': 'Fingertip tracking weight',
    'r': 'Input tracking weight',
}
TUNING_ROOTS = ('kde_bdpc_tuning_20261101_v1', 'kde_bdpc_cluster_tuning_20261101_v1')
PANEL = 'kde_bdpc_p8_k4_consensus_dev100_t300_v1'


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def scan(experiment_root):
    root = Path(experiment_root)
    if not root.is_dir():
        raise FileNotFoundError(f'External experiment directory unavailable: {root}')
    dirs = [(d, 'tuning') for name in TUNING_ROOTS for d in sorted((root/name).glob('*'))
            if d.is_dir() and (d/'manifest.json').exists()]
    if (root/PANEL/'manifest.json').exists():
        dirs.append((root/PANEL, 'separate_100_seed_panel'))
    trials, configurations, inventory, distances, times, prediction, offsets = [], [], [], [], [], [], [0]
    stage_rows = []
    for name in TUNING_ROOTS:
        for path in sorted((root/name).glob('*/protocol.json')):
            protocol = read_json(path)
            stage_rows.append(dict(stage=str(path.parent.relative_to(root)),
                                   seeds=protocol['seeds'], steps=protocol['steps'],
                                   candidates=list(protocol['configs'])))
            inventory.append(dict(path=str(path), sha256=digest(path)))
    for directory, group in dirs:
        manifest_path = directory/'manifest.json'
        manifest = read_json(manifest_path)
        config = manifest['config']
        candidate = str(directory.relative_to(root))
        inventory.append(dict(path=str(manifest_path), sha256=digest(manifest_path)))
        failure_path = directory/'offline_failure.json'
        offline_failure = read_json(failure_path)['message'] if failure_path.exists() else None
        audit_path = directory/'offline_audit.json'
        audit = read_json(audit_path) if audit_path.exists() else {}
        for path in (failure_path, audit_path):
            if path.exists():
                inventory.append(dict(path=str(path), sha256=digest(path)))
        local_rows, local_times = [], []
        for path in sorted(directory.glob('seed*_control_results.npz')):
            with np.load(path, allow_pickle=False) as z:
                saved = json.loads(z['summary'].item())
                n = int(saved['completed_steps'])
                requested = int(saved['requested_steps'])
                error = z['distance'].copy() if n else np.empty(0)
                timing = z['control_compute_time_s'].copy() if n else np.empty(0)
                if error.shape != (n,) or timing.shape != (n,):
                    raise ValueError(f'Invalid trace lengths: {path}')
                if not np.isfinite(error).all() or not np.isfinite(timing).all() or np.any(error < 0) or np.any(timing <= 0):
                    raise ValueError(f'Invalid error/timing values: {path}')
                if n:
                    np.testing.assert_allclose(error, np.linalg.norm(z['y'][:, 6:8]-z['final_goal'][6:8], axis=1))
                complete = n == requested and saved['failure'] is None
                if complete != saved['complete']:
                    raise ValueError(f'Invalid completion status: {path}')
                evaluable = complete and requested >= 100
                success = bool(evaluable and all(np.mean(error[-w:]) < .01 and np.max(error[-w:]) < .02 for w in (30, 100)))
                if success != saved['strict_success']:
                    raise ValueError(f'Invalid success status: {path}')
                pred = z['prediction_error_xy'].copy() if n and 'prediction_error_xy' in z else np.full(n, np.nan)
                row = dict(rollout_id=len(trials), group=group, candidate=candidate,
                    seed=int(saved['seed']), requested_steps=requested, completed_steps=n,
                    complete=complete, success_evaluable=evaluable, strict_success=success,
                    last_error_mm=float(error[-1]*1000) if n else None,
                    mean_final100_error_mm=float(np.mean(error[-100:])*1000) if evaluable else None,
                    max_final100_error_mm=float(np.max(error[-100:])*1000) if evaluable else None,
                    median_online_ms=float(np.median(timing)*1000) if n else None,
                    failure=saved['failure'], trace_path=str(path), trace_sha256=digest(path))
                trials.append(row); local_rows.append(row); local_times.extend(timing.tolist())
                distances.append(error); times.append(timing); prediction.append(pred); offsets.append(offsets[-1]+n)
                inventory.append(dict(path=str(path), sha256=row['trace_sha256']))
        full = [r for r in local_rows if r['complete']]
        errors = [r['mean_final100_error_mm'] for r in full if r['success_evaluable']]
        configurations.append(dict(group=group, candidate=candidate, config=config,
            attempted=bool(local_rows or offline_failure), offline_failure=offline_failure,
            requested_clusters=config['clusters'], active_clusters=audit.get('active_clusters'),
            seeds=[r['seed'] for r in local_rows], requested_steps=sorted({r['requested_steps'] for r in local_rows}),
            saved_rollouts=len(local_rows), full_rollouts=len(full), aborted_rollouts=len(local_rows)-len(full),
            successes=sum(r['strict_success'] for r in local_rows),
            mean_final100_error_mm=float(np.mean(errors)) if errors else None,
            median_online_ms=float(np.median(local_times)*1000) if local_times else None,
            failure_reasons=sorted({r['failure']['message'] for r in local_rows if r['failure']})))
    attempted = [r for r in configurations if r['group']=='tuning' and r['attempted']]
    ranges = []
    for key, label in PARAMETERS.items():
        values = sorted({r['config'][key] for r in attempted})
        completed = sorted({r['config'][key] for r in attempted if r['full_rollouts']})
        ranges.append(dict(parameter=key, meaning=label, attempted_values=values,
                           values_with_completed_rollouts=completed,
                           minimum=min(values) if values else None, maximum=max(values) if values else None))
    tuning_trials = [r for r in trials if r['group']=='tuning']
    result = dict(schema_version=1, ranges=ranges, configurations=configurations, rollouts=trials, stages=stage_rows,
        tuning_totals=dict(candidate_directories=len(attempted),
            unique_configurations=len({json.dumps(r['config'], sort_keys=True) for r in attempted}),
            saved_rollouts=len(tuning_trials), full_rollouts=sum(r['complete'] for r in tuning_trials),
            aborted_rollouts=sum(not r['complete'] for r in tuning_trials),
            offline_rejections=sum(r['offline_failure'] is not None for r in attempted)),
        inventory=inventory,
        interpretation='Observed values, not a Cartesian grid or exhaustive search. Compare only matched seeds/horizons. '
            'Offline rejection and planning/solver aborts are separate from completed tracking failures. '
            'Hybrid KDE-Atlas results excluded. The 100-seed panel is separate, not pooled into tuning totals.',
        timing_definition='Measured accepted online steps: membership, interpolation, online graph construction/search, '
            'matrix preparation and optimization. Excludes offline work, warmup, simulation and disk I/O. '
            'Median pools steps within a candidate; failed-attempt timing is in each rollout failure record.',
        array_definition='rollout_offsets delimit each rollout_id in distance_m, control_compute_time_s, '
            'prediction_error_xy_m. Original full traces remain at trace_path. Missing prediction measurements are NaN.')
    arrays = dict(rollout_offsets=np.array(offsets, dtype=np.int64),
                  distance_m=np.concatenate(distances) if distances else np.empty(0),
                  control_compute_time_s=np.concatenate(times) if times else np.empty(0),
                  prediction_error_xy_m=np.concatenate(prediction) if prediction else np.empty(0))
    return result, arrays


def export_snapshot(result, arrays, destination):
    """Write a content-addressed report; never alter source experiment files."""
    encoded = json.dumps(result, sort_keys=True, allow_nan=False, indent=2)
    signature = hashlib.sha256(encoded.encode()).hexdigest()[:16]
    out = Path(destination)/f'snapshot_{signature}'
    out.mkdir(parents=True, exist_ok=True)
    report_path = out/'report.json'
    if report_path.exists() and report_path.read_text() != encoded:
        raise ValueError('Snapshot content mismatch')
    if not report_path.exists():
        report_path.write_text(encoded)
    data_path = out/'tracking_and_timing.npz'
    if data_path.exists():
        with np.load(data_path) as saved:
            for key, value in arrays.items():
                np.testing.assert_array_equal(saved[key], value)
    else:
        np.savez_compressed(data_path, **arrays)
    return out
