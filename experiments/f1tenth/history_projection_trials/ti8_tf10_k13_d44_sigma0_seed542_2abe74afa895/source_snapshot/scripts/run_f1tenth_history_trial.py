"""Isolated noiseless horizon trial using the retained Atlas controller and selector.

Run in a fresh process: set horizon constants before importing dependent modules.
The paper defaults on disk, data, QP, warmup, and selection algorithm are unchanged.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--t-ini', type=int, default=8)
    p.add_argument('--t-f', type=int, default=10)
    p.add_argument('--clusters', type=int, default=12)
    p.add_argument('--rank', type=int, default=40)
    p.add_argument('--seed', type=int, default=542)
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--output-root', type=Path, default=ROOT/'experiments/f1tenth/history_projection_trials')
    a = p.parse_args()
    if min(a.t_ini, a.t_f, a.steps) < 1:
        p.error('Horizons and step budget must be positive')
    # This process imports the dependent modules only AFTER setting horizons.
    from atlas_deepc.f1tenth import config
    config.TI, config.TF = a.t_ini, a.t_f
    config.WARMUP_STEPS = max(15, a.t_ini)
    from atlas_deepc.f1tenth import task, atlas, partial_selection
    from atlas_deepc.f1tenth.replay import verify
    from atlas_deepc.f1tenth.paths import CLEAN_OFFLINE_DATA_FILE, MAP_BASE, WAYPOINT_FILE
    import numpy as np
    from threadpoolctl import threadpool_limits

    cfg = config.AtlasConfig(clusters=a.clusters, rank=a.rank)
    task.collect_clean()  # Build versioned applied-input data before hashing it.
    sources = list((ROOT/'src/atlas_deepc/f1tenth').glob('*.py')) + [
        ROOT/'src/atlas_deepc/atlas/kmeans.py',
        ROOT/'src/atlas_deepc/control/expert_pipeline_search.py', Path(__file__)]
    assets = [CLEAN_OFFLINE_DATA_FILE, WAYPOINT_FILE, MAP_BASE.with_suffix('.png'), MAP_BASE.with_suffix('.yaml')]
    protocol = dict(input_history_convention=task.INPUT_HISTORY_CONVENTION,t_ini=a.t_ini, t_f=a.t_f, dt=config.DT, config=asdict(cfg), seed=a.seed,
        maximum_steps=a.steps, noise_sigma=0., warmup_steps=config.WARMUP_STEPS,
        chart_selection_method=partial_selection.METHOD,
        clustering='Measured speed at history anchor; K-means seed 11; uncentered full-window SVD.',
        selection='Latest T_ini input/output pairs, aligned to last history pose; Euclidean projection residual on history rows of each full basis; smallest code chart ID breaks ties.',
        noise='Clean offline observations and zero measurement standard deviations online. Original random initial position/yaw perturbation remains; zero-scaled Gaussian draws preserve the original RNG sequence.',
        qp='Unchanged physical Q/R and lambda_ini=100; no coefficient regularization or coefficient bounds.',
        Q=config.Q_PHYSICAL.tolist(), R=config.R_PHYSICAL.tolist(),
        runtime_override='Horizon constants set in this subprocess before dependent imports. Paper config file untouched.',
        hashes={str(f.relative_to(ROOT)): task.digest(f) for f in sources + assets})
    token = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    output = a.output_root / f'ti{a.t_ini}_tf{a.t_f}_k{a.clusters}_d{a.rank}_sigma0_seed{a.seed}_{token}'
    if (output/'status.json').is_file() and json.loads((output/'status.json').read_text())['status'] == 'complete':
        for name, expected in json.loads((output/'checksums.json').read_text()).items():
            if task.digest(output/name) != expected:
                raise ValueError(f'Cached artifact checksum mismatch: {name}')
        print('Using verified cached trial.', flush=True)
        print(f'RESULT_DIRECTORY={output}')
        return
    task.write(output/'protocol.json', protocol)
    task.write(output/'status.json', {'status':'running'})
    for f in sources:
        dest = output/'source_snapshot'/f.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dest)
    with threadpool_limits(limits=1):
        clean = task.collect_clean()
        library = atlas.build_library(clean)
        H, features, metadata = library
        fitted = atlas.fit(a.rank, a.clusters, library)
        policy = atlas.Atlas(fitted, cfg, partial_selection.METHOD)
        np.savez_compressed(output/'atlas.npz', **fitted)
        print(f'H={H.shape}, chart history ranks={policy.selector.ranks.tolist()}', flush=True)
        folder = output/'rollout'
        task.rollout(policy, cfg, a.seed, steps=a.steps, save=folder, noise_sigma=0.)
        collision = verify(folder)
        summary = json.loads((folder/'summary.json').read_text())
        calls = json.loads((folder/'calls.json').read_text())
        with np.load(folder/'rollout.npz') as saved:
            trace = {k:saved[k].copy() for k in saved.files}
        windows = trace['selection_windows']
        residuals = trace['selection_residuals']
        chosen = np.array([c['chart'] for c in calls], dtype=int)
        assert windows.shape == (len(calls), 6*a.t_ini)
        assert residuals.shape == (len(calls), a.clusters)
        # Check the saved selector against independent least-squares projections.
        independent = np.column_stack([
            np.linalg.norm(windows.T - b @ np.linalg.lstsq(b, windows.T, rcond=1e-10)[0], axis=0)
            for b in fitted['bases'][:, partial_selection.PAST_ROWS, :]])
        np.testing.assert_allclose(residuals, independent, rtol=1e-6, atol=1e-9)
        tolerance = 100*np.finfo(float).eps*np.maximum(1., np.linalg.norm(windows, axis=1))
        tied = residuals <= residuals.min(axis=1)[:, None]+tolerance[:, None]
        np.testing.assert_array_equal(chosen, tied.argmax(axis=1))
        np.testing.assert_array_equal(trace['selection_tie_count'], tied.sum(axis=1))
        # Reconstruct every selection window from the recorded observed history.
        for i, window in enumerate(windows):
            stop = config.WARMUP_STEPS+i
            hist = trace['measured_history'][stop-a.t_ini:stop]
            past_u = trace['input_history'][stop-a.t_ini+1:stop+1]
            expected = np.r_[past_u.ravel(), task.frame(hist, hist[-1, [0,1,3]], anchor_index=-1).ravel()]
            np.testing.assert_allclose(window, expected, rtol=0, atol=1e-12)
        # Check that histories contain applied inputs, including warmup, and that
        # commands reproduce every saved transition with the unchanged plant.
        states=np.vstack((trace['warmup_states'],trace['X'].reshape(-1,7)))
        applied=np.vstack((trace['warmup_inputs'],trace['U'].reshape(-1,2)))
        commands=np.vstack((trace['warmup_commanded_inputs'],trace['U_commanded'].reshape(-1,2)))
        next_states=np.vstack((states[1:],trace['final_state']))
        np.testing.assert_array_equal(trace['input_history'][1:],applied)
        np.testing.assert_allclose(np.diff(np.r_[states[:,3],trace['final_state'][3]]),config.DT*applied[:,0],atol=1e-12,rtol=0)
        for state,cmd,u,nxt in zip(states,commands,applied,next_states):
            predicted,actual,_=task.step(state,cmd,return_applied=True)
            np.testing.assert_allclose(predicted,nxt,atol=1e-12,rtol=0)
            np.testing.assert_allclose(actual,u,atol=1e-12,rtol=0)
        # Verify the complete measurement stream is noise-free.
        truth = np.vstack((trace['warmup_states'], trace['X']))[:, [0,1,3,4]]
        np.testing.assert_array_equal(trace['measured_history'], truth)
        selected = residuals[np.arange(len(chosen)), chosen]
        counts = np.bincount(chosen, minlength=a.clusters)
        diagnostics = dict(H_shape=list(H.shape), basis_shape=list(fitted['bases'].shape),
            history_shape=[6*a.t_ini, a.rank], effective_history_dimension=6*a.t_ini-3,
            past_ranks=policy.selector.ranks.tolist(), chart_counts=counts.tolist(),
            chart_ids_zero_based=np.flatnonzero(counts).tolist(),
            chart_ids_one_based=(np.flatnonzero(counts)+1).tolist(),
            switches=int(np.count_nonzero(np.diff(chosen))),
            switch_control_steps=(np.flatnonzero(np.diff(chosen))+1).tolist(),
            tie_fraction=float(np.mean(trace['selection_tie_count']>1)),
            selected_residual_min=float(selected.min()), selected_residual_max=float(selected.max()),
            selected_residual_median=float(np.median(selected)),
            maximum_chart_residual_spread=float(np.max(np.ptp(residuals,axis=1))),
            member_counts=fitted['member_count'].tolist(),
            applied_input_validation=dict(verified=True,limited_control_steps=int(np.count_nonzero(abs(trace['U'][:,0]-trace['U_commanded'][:,0])>1e-10))),
            validation='Applied input history/replay, independent least-squares residual, argmin/tie rule, observed-window reconstruction, and noiseless observations checked.',
            maximum_projection_reconstruction_error=float(np.max(np.abs(residuals-independent))),
            collision_replay=collision)
        np.savez_compressed(output/'selection_over_time.npz', step=np.arange(len(chosen)),
            time_s=np.arange(len(chosen))*config.DT, chart_zero_based=chosen,
            chart_one_based=chosen+1, residuals=residuals,
            tie_count=trace['selection_tie_count'], accepted=np.array([c['accepted'] for c in calls]))
        task.write(output/'diagnostics.json', diagnostics)
        report = dict(protocol=protocol, summary=summary, diagnostics=diagnostics)
        task.write(output/'report.json', report)
        task.write(output/'checksums.json', {str(f.relative_to(output)):task.digest(f)
                   for f in sorted(output.rglob('*')) if f.is_file() and f.name not in ('checksums.json','status.json')})
        task.write(output/'status.json', {'status':'complete'})
        print(json.dumps(dict(summary=summary,diagnostics=diagnostics), indent=2), flush=True)
    print(f'RESULT_DIRECTORY={output}')


if __name__ == '__main__':
    main()
