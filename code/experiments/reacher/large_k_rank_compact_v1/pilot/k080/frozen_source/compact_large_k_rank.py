"""Storage-bounded K=80/96 four-method rank sweeps using the K=64 runner."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np

from atlas_deepc.ablations import k64_rank as original
from atlas_deepc.reporting.four_case_tracking_table import format_table

four = original.four
ROOT = original.ROOT
RESULT_ROOT = ROOT / 'experiments/reacher/large_k_rank_compact_v1'
NOTEBOOK = ROOT / 'notebooks/reacher/ablations/ablation_linear_subspace_rank_k80_k96.ipynb'
COUNTS = (80, 96)
RANKS = tuple(range(28, 71))
SEEDS = tuple(range(211, 311))
ORIGINAL_HASHES = original.code_hashes
STORAGE_POLICY = 'Validate full panel, retain per-seed metrics, discard sweep traces; fully rerun selected ranks.'


def code_hashes():
    return {**ORIGINAL_HASHES(), Path(__file__).name: four.file_digest(__file__)}


@contextmanager
def study_context(k, seeds=SEEDS):
    if k not in COUNTS:
        raise ValueError('This study supports K=80 and K=96')
    with patch.object(original, 'K', k), patch.object(original, 'SEEDS', list(seeds)), \
            patch.object(original.previous, 'SEEDS', list(seeds)), \
            patch.object(original, 'code_hashes', code_hashes):
        yield


def count_root(root, k):
    return Path(root) / f'k{k:03d}'


def prepare(root=RESULT_ROOT, counts=COUNTS, ranks=RANKS, seeds=SEEDS, workers=4):
    root = Path(root)
    for k in counts:
        directory = count_root(root, k)
        directory.mkdir(parents=True, exist_ok=True)
        with study_context(k, seeds):
            config = original.config_for()
            values = {key: value for key, value in asdict(config).items() if key != 'seed'}
            for p, mode in four.CASES:
                original.validate_config(values, p, mode, 60)
            data_hash = four.file_digest(four.DATA_PATH)
            assert data_hash == original.previous.control_signature(10, 'graph')['data_sha256']
            manifest = dict(charts=k, ranks=list(ranks), seeds=list(seeds), config_rank60=values,
                workers=workers, numerical_threads_per_worker=1, source_sha256=code_hashes(),
                data_sha256=data_hash, selection='full_window_projection', graph_cost='chordal',
                selection_rule=original.previous.SELECTION_RULE, storage_policy=STORAGE_POLICY,
                objective='tracking + input effort + initial-output slack',
                caveat='Exploratory development seeds, not held-out evaluation.')
            path = directory / 'manifest.json'
            if path.exists():
                if json.loads(path.read_text()) != manifest:
                    raise ValueError('Protocol/source changed; choose a fresh result root')
            else:
                four.save_json(path, manifest)
                for name in manifest['source_sha256']:
                    if name == Path(__file__).name:
                        source = Path(__file__)
                    elif name.startswith('common/'):
                        source = ROOT / 'src/atlas_deepc' / name
                    elif (Path(original.__file__).parent / name).is_file():
                        source = Path(original.__file__).parent / name
                    else:
                        source = original.source_path(name)
                    destination = directory / 'frozen_source' / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
            inventory = {}
            for p in (8, 10):
                atlas_path, _ = four.experiment.prepare_atlas(config,
                    original.atlas_path(directory, p).parent, four.DATA_PATH, output_dimension=p)
                atlas = four.experiment.routed.load_atlas(atlas_path)
                sizes = np.bincount(atlas['labels'].astype(int), minlength=k)
                assert len(sizes) == k and sizes.sum() == 37800 and sizes.min() >= max(ranks)
                assert atlas['medoid_states'].shape == (k, p)
                assert np.all(atlas['feature_mean'] == 0) and np.all(atlas['feature_scale'] == 1)
                inventory[str(p)] = dict(sha256=four.file_digest(atlas_path),
                                        chart_sizes=sizes.tolist(), min_columns=int(sizes.min()))
                # Serial staging avoids simultaneous graph/terminal workers publishing one atlas.
                for rank in ranks:
                    target = original.panel_directory(directory, rank, p, 'terminal').parent / 'behavioral_atlas.npz'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        temporary = target.with_suffix('.pending.npz')
                        shutil.copy2(atlas_path, temporary)
                        temporary.replace(target)
                    assert four.file_digest(target) == inventory[str(p)]['sha256']
            four.save_json(directory / 'atlas_inventory.json', inventory)


def compact_panel(directory, record):
    """Publish validated compact metrics before deleting only this panel's trace files."""
    directory = Path(directory)
    if 'sweep' not in directory.parts:
        raise ValueError('Selected-rank reruns must retain their full traces')
    marker_path = directory / 'compact_storage.json'
    metrics_path = directory / 'seed_metrics.json'
    assert four.file_digest(directory / 'summary.json') == record['summary_sha256']
    if marker_path.exists():
        marker = json.loads(marker_path.read_text())
        assert marker['summary_sha256'] == record['summary_sha256']
        assert marker['metrics_sha256'] == four.file_digest(metrics_path)
    else:
        panel = json.loads((directory / 'summary.json').read_text())
        seeds = panel['seeds']
        assert len(seeds) == record['seeds'] and len(set(seeds)) == len(seeds)
        by_seed = {row['seed']: row for row in panel['results']}
        assert set(by_seed) == set(seeds)
        compact = []
        for seed in seeds:
            row = by_seed[seed]
            compact.append(dict(seed=seed, success=bool(four.strict_success(row)),
                final100_mean_error_mm=1000*row['mean_last_100'],
                rollout_rmse_mm=record['per_seed_rmse_mm'][str(seed)],
                unsolved_steps=row['unsolved_steps']))
        assert sum(row['success'] for row in compact) == record['successes']
        assert [row['seed'] for row in compact if not row['success']] == record['failed_seeds']
        np.testing.assert_allclose(np.median([row['rollout_rmse_mm'] for row in compact]),
                                   record['median_rollout_rmse_mm'])
        four.save_json(metrics_path, dict(seeds=compact, summary_sha256=record['summary_sha256'],
                                         storage_policy=STORAGE_POLICY))
        paths = [directory / f'seed{seed}_control_results.npz' for seed in seeds]
        paths += [directory / name for name in ('selection_geometry.npz', 'graph_geometry.npz')
                  if (directory / name).exists()]
        files = [dict(name=path.name, bytes=path.stat().st_size, sha256=four.file_digest(path))
                 for path in paths]
        marker = dict(summary_sha256=record['summary_sha256'],
                      metrics_sha256=four.file_digest(metrics_path), files=files,
                      note='Original summary trace paths are provenance only; traces intentionally discarded.')
        four.save_json(marker_path, marker)
    for item in marker['files']:
        if Path(item['name']).name != item['name'] or not item['name'].endswith('.npz'):
            raise ValueError('Unsafe compact-storage filename')
        path = directory / item['name']
        if path.exists():
            if path.is_symlink() or four.file_digest(path) != item['sha256']:
                raise ValueError(f'Trace changed before compaction: {path}')
            path.unlink()


def run_case(root, k, rank, p, mode, workers=4, serial=False):
    directory = count_root(root, k)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if rank not in manifest['ranks']:
        raise ValueError('Rank not in frozen protocol')
    with study_context(k, manifest['seeds']):
        record = original.run_case(directory, rank, p, mode, workers, serial=serial)
    if not serial:
        compact_panel(original.panel_directory(directory, rank, p, mode), record)
    return record


def report(root=RESULT_ROOT, counts=COUNTS):
    payload = {}
    for k in counts:
        directory = count_root(root, k)
        manifest_path = directory / 'manifest.json'
        if not manifest_path.exists():
            payload[str(k)] = dict(complete=False, completed_rollouts=0, records=[])
            continue
        manifest = json.loads(manifest_path.read_text())
        records = [json.loads(p.read_text()) for p in sorted((directory / 'panel_metrics').glob('*.json'))]
        expected = {(r, p, mode) for r in manifest['ranks'] for p, mode in four.CASES}
        observed = {(r['rank'], r['output_dimension'], r['reference_mode']) for r in records}
        assert observed <= expected
        best = [original.previous.best_record([r for r in records
                if (r['output_dimension'], r['reference_mode']) == case])
                for case in four.CASES if any((r['output_dimension'], r['reference_mode']) == case for r in records)]
        result = dict(complete=observed == expected, completed_rollouts=sum(r['seeds'] for r in records),
            requested_rollouts=len(expected)*len(manifest['seeds']), records=records,
            best_completed_ranks=best, missing_panels=sorted(expected-observed), storage_policy=STORAGE_POLICY)
        four.save_json(directory / 'summary.json', result)
        payload[str(k)] = result
    return payload


def run_all(root=RESULT_ROOT, counts=COUNTS, ranks=RANKS, seeds=SEEDS, workers=4):
    root = Path(root)
    prepare(root, counts, ranks, seeds, workers)
    for k in counts:
        with ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            for rank in sorted(ranks, key=lambda r: (abs(r-62), r)):
                jobs = [pool.submit(run_case, root, k, rank, p, mode, workers) for p, mode in four.CASES]
                for future in as_completed(jobs):
                    row = future.result()
                    print(f"K={k} rank={rank} p={row['output_dimension']} {row['reference_mode']}: "
                          f"{row['successes']}/{row['seeds']}; metrics saved, sweep traces discarded", flush=True)
                    report(root, counts)
        result = report(root, counts)[str(k)]
        assert result['complete']
        selected = []
        for winner in result['best_completed_ranks']:
            row = run_case(root, k, winner['rank'], winner['output_dimension'],
                           winner['reference_mode'], workers, serial=True)
            assert row['failed_seeds'] == winner['failed_seeds']
            for key in ('median_error_mm', 'median_rollout_rmse_mm', 'p99_error_mm'):
                np.testing.assert_allclose(row[key], winner[key], rtol=1e-6, atol=1e-5)
            selected.append(row)
        directory = count_root(root, k)
        four.save_json(directory / 'best_rank_table.json', dict(columns=selected,
            charts=k, ranks_tested=list(ranks), seeds=list(seeds), selection_rule=original.previous.SELECTION_RULE))
        caption = (f'Best observed ranks with $K={k}$ on {len(seeds)} development seeds. '
                   'Ranks maximize success, then minimize median rollout RMSE, then rank. '
                   'Detailed data and serial timings come from fresh selected-rank reruns. '
                   'Selection and reporting share exploratory seeds, not a held-out test set.')
        (directory / 'table_best_ranks.tex').write_text(format_table(
            selected, caption, f'tab:best-ranks-k{k}-compact', include_rank=True))
    return report(root, counts)
