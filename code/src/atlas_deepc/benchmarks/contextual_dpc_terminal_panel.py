"""Frozen-weight contextual DPC evaluation using saved unguided Atlas goals.

Separate launcher so the 211--310 controller and result signatures stay intact.
No Atlas simulation or controller solve is run here.
"""
import argparse
from contextlib import redirect_stdout
import fcntl
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits
from atlas_deepc.benchmarks import contextual_dpc_atlas_terminal as case

ATLAS = case.base.ROOT / 'experiments/reacher/four_baselines_500_mixed_k_t300_v1/output8/terminal'
OUTPUT = case.base.ROOT / 'experiments/reacher/contextual_dpc_k64_rank53_terminal8_10001_10500_t300_v1'
SEEDS = list(range(10001, 10501))
STEPS = 300
# Freeze the user-selected reference source, not the contextual controller's weights.
REFERENCE_MANIFEST_SHA256 = 'f8a4440a88b97e9d91d793cf6651140804d722539b38bbe11937db0998b2e6a0'
REFERENCE_ATLAS_SHA256 = '8a69d7a46685bd6da0c0d68467f7b3bc5341d4d42db60dbfc07e7f89a0ba6f20'


def frozen_config():
    path = case.ROOT / 'tuning_selection.json'
    selected = json.loads(path.read_text())
    config = case.Config(**selected['config'])
    expected = case.Config(q_xy=100., q_joint=.01, q_dq=.0001, r=.1,
                           lambda_g_bar=.1, lambda_sigma=100.)
    if config != expected or not selected['frozen_before_full_panel']:
        raise ValueError('The frozen development configuration changed; do not retune on evaluation seeds.')
    return config, path


def validate_reference_manifest(manifest):
    expected = dict(output_dimension=8, reference_mode='terminal',
                    terminal_reference_policy='fixed_nearest_eligible_fingertip',
                    atlas_sha256=REFERENCE_ATLAS_SHA256, evaluation_seeds=SEEDS)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f'Atlas reference protocol mismatch: {key}')
    for key, value in dict(clusters=64, svd_rank=53, steps=STEPS,
                           goal_source_target_min_distance=.06).items():
        if manifest['config'].get(key) != value:
            raise ValueError(f'Atlas reference configuration mismatch: {key}')
    if manifest['data_sha256'] != case.base.digest(case.base.DATA):
        raise ValueError('Offline dataset changed')


def read_references(atlas=ATLAS):
    """Read and validate all goals before creating a cache or starting a rollout."""
    manifest_path = atlas / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    validate_reference_manifest(manifest)
    if case.base.digest(manifest_path) != REFERENCE_MANIFEST_SHA256:
        raise ValueError('The frozen K=64, rank-53 reference manifest changed')
    references, sources = [], []
    for seed in SEEDS:
        path = atlas / f'seed{seed}_control_results.npz'
        with np.load(path) as a:
            goal = a['final_goal'].copy()
            if goal.shape != (8,) or not np.isfinite(goal).all():
                raise ValueError(f'Invalid reference for seed {seed}')
            np.testing.assert_allclose(a['ref'], np.tile(goal, (STEPS, 1)), rtol=0, atol=1e-12)
            np.testing.assert_array_equal(goal[4:6], np.zeros(2))
            row = json.loads(a['summary'].item())
            if row['seed'] != seed or row['steps'] != STEPS:
                raise ValueError(f'Incompatible source rollout for seed {seed}')
        references.append(goal)
        sources.append(dict(seed=seed, file=str(path.relative_to(case.base.ROOT)),
                            sha256=case.base.digest(path), original_atlas_goal_setup_ms=row.get('goal_setup_time_ms')))
    provenance = dict(source_manifest=str(manifest_path.relative_to(case.base.ROOT)),
        atlas_manifest_sha256=case.base.digest(manifest_path), sources=sources,
        reference='Exact fixed K=64, rank-53, p=8 unguided Atlas final_goal; no graph, actions, outcomes or reference updates used',
        timing='Reference loading and original Atlas construction excluded from online step time; recorded separately.')
    return np.stack(references), provenance


def preflight():
    config, _ = frozen_config()
    references, _ = read_references()
    if 'CLARABEL' not in case.cp.installed_solvers():
        raise RuntimeError('Clarabel is unavailable')
    print(f'Validated {len(references)} fixed K=64, rank-53, eight-output references for seeds 10001--10500, {STEPS} steps.')
    print('Frozen config:', case.asdict(config))
    print('No controller rollout was run.')


def run(output=OUTPUT):
    config, selection_path = frozen_config()
    references, provenance = read_references()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'run.lock').open('a+') as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signature = dict(seeds=SEEDS, steps=STEPS, config=case.asdict(config), workers=1, threads=1,
            solver='CLARABEL', svd=False, source_sha256=case.base.digest(__file__),
            controller_sha256=case.base.digest(case.__file__), base_sha256=case.base.digest(case.base.__file__),
            metrics_sha256=case.base.digest(Path(case.base.__file__).parents[1]/'common/metrics.py'),
            data_sha256=case.base.digest(case.base.DATA), selection_sha256=case.base.digest(selection_path),
            reference_provenance=provenance,
            packages={p:importlib.metadata.version(p) for p in ('numpy','scipy','cvxpy','clarabel','gymnasium','mujoco')})
        manifest = output / 'manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != signature:
                raise ValueError('Configuration/source/provenance changed; use a fresh output directory')
        elif list(output.glob('seed*_control_results.npz')):
            raise ValueError('Unversioned rollout cache; use a fresh directory')
        case.write_json(manifest, signature)
        bundle = output / 'references.npz'
        if bundle.exists():
            with np.load(bundle) as a:
                np.testing.assert_array_equal(a['seeds'], SEEDS)
                np.testing.assert_array_equal(a['references'], references)
        else:
            np.savez_compressed(bundle, seeds=SEEDS, references=references)
        case.write_json(output/'reference_provenance.json', provenance)
        lib = case.Library.from_path(case.base.DATA, config)
        np.savez_compressed(output/'normalization.npz', du_mean=lib.du_mean, du_scale=lib.du_scale,
                            z_mean=lib.z_mean, z_scale=lib.z_scale, Q_diagonal=lib.q)
        case.write_json(output/'data_audit.json', lib.meta)
        try:
            for seed in SEEDS:
                started = time.perf_counter()
                with np.load(bundle) as a:
                    reference = a['references'][seed-SEEDS[0]].copy()
                setup_s = time.perf_counter()-started
                path = output/f'seed{seed}_control_results.npz'
                if not path.exists():
                    with (output/f'seed{seed}.log').open('w') as log, redirect_stdout(log):
                        case.run_seed(lib, seed, STEPS, output, reference, setup_s)
                else:
                    with np.load(path) as a:
                        saved = json.loads(a['summary'].item())
                        if saved['seed'] != seed or saved['steps'] != STEPS:
                            raise ValueError('Incompatible saved rollout')
                        np.testing.assert_array_equal(a['final_goal'], reference)
                report = case.summarize(output, SEEDS)
                report['note'] = ('Evaluation seeds 10001--10500, 300 steps; weights frozen on development seeds. '
                    'Exact fixed K=64, rank-53, p=8 targets; costs use steps 1--300, '
                    'final error windows end at step 300, pooled trimmed timing uses steps 6--300. '
                    'Reference loading/construction excluded from online step time. No SVD; '
                    'native costs are not comparable across different output weights.')
                case.write_json(output/'summary.json', report)
                print(f"{report['completed']}/500 completed, {report['successes']} successes; last seed {seed}", flush=True)
        except BaseException:
            case.write_json(output/'progress.json', dict(state='interrupted', seed=seed,
                note='Completed seeds preserved; interrupted seed restarts on resume.'))
            raise
        case.write_json(output/'progress.json', dict(state='complete', completed=report['completed']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--preflight', action='store_true', help='Validate inputs only; never run control')
    args = parser.parse_args()
    preflight() if args.preflight else run(args.output_dir)
