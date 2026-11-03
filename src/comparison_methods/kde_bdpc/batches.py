"""Resumable 9 x 100 development panel; parallelize clusters, not shared files.

The controller is unchanged. Each process holds its candidate lock, loads one
offline library, and runs a bounded seed block serially. Existing terminal
attempts (including setup rejections) are never automatically rerun.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from comparison_methods.kde_bdpc import paper as study

CLUSTERS = (2, 4, 8, 16, 32, 48, 64, 80, 96)
SEEDS = tuple(range(211, 311))
STEPS = 300
OUTPUT = study.OUTPUT.parent/'kde_bdpc_eq34_l3_clusters9_dev100_t300_v1'
SCREEN = study.OUTPUT.parent/'kde_bdpc_eq34_l3_clusters9_seed211_t150_v1'
ROOT = Path(__file__).resolve().parents[3]
write = study.core.shared.utils.write_json


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


@contextmanager
def lock_file(path):
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another worker owns {path}; refusing duplicate work') from exc
        yield


def require_idle_screen(source):
    """Read-only check: do not overlap the earlier single-seed screen."""
    if source is None or not (source/'sweep.lock').exists():
        return
    with (source/'sweep.lock').open('r') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('The single-seed cluster sweep is still active. Wait for it to finish before starting batches.') from exc


def requested_seeds(start=None, stop=None):
    if (start is None) != (stop is None):
        raise ValueError('Supply both block-start and block-stop, or neither')
    if start is None:
        return list(SEEDS)
    if not SEEDS[0] <= start <= stop <= SEEDS[-1]:
        raise ValueError('Seed blocks must be within 211--310')
    return list(range(start, stop+1))


def select_pending(done, start=None, stop=None, batch_size=10):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Batch size must be a positive integer (per cluster)')
    return [s for s in requested_seeds(start, stop) if s not in done][:batch_size]


def identity(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Protocol changed: {path}. Use a new output directory.')
    else:
        write(path, value)


def signature(config, provenance):
    # Same immutable controller identity as study.run; no source modifications.
    return dict(version='kde_bdpc_eq34_one_sample_v1', config=asdict(config),
        conventions=study.CONVENTIONS, data_sha256=study.core.shared.utils.digest(study.core.DATA),
        reference_provenance=provenance, paper_sha256=study.core.shared.utils.digest(study.core.PAPER),
        sources={str(p): study.core.shared.utils.digest(p) for p in
                 (study.__file__, study.core.__file__, study.core.shared.__file__,
                  study.core.shared.clustering.__file__, study.core.shared.utils.__file__,
                  study.minimum_mean_route.__file__)},
        packages={n: importlib.metadata.version(n) for n in
                  ('numpy', 'scipy', 'cvxpy', 'clarabel', 'gymnasium', 'mujoco')}, geometry_reuse=None)


def trace_row(path, seed):
    with study.np.load(path, allow_pickle=False) as trace:
        row = json.loads(trace['summary'].item())
        if (row['seed'], row['requested_steps'], row['past'], row['future']) != (seed, STEPS, 2, 1):
            raise ValueError(f'Trace protocol mismatch: {path}')
        n = row['completed_steps']
        if not 0 <= n <= STEPS or row['complete'] != (n == STEPS and row['failure'] is None):
            raise ValueError(f'Invalid trace completion: {path}')
        if n:
            if trace['distance'].shape != (n,) or trace['control_compute_time_s'].shape != (n,):
                raise ValueError(f'Incomplete trace arrays: {path}')
            if not study.np.isfinite(trace['distance']).all() or not study.np.isfinite(trace['control_compute_time_s']).all():
                raise ValueError(f'Nonfinite trace arrays: {path}')
    return row


def completed_rows(folder):
    return {seed: trace_row(folder/f'seed{seed}_control_results.npz', seed)
            for seed in SEEDS if (folder/f'seed{seed}_control_results.npz').exists()}


def atomic_copy(source, target):
    if target.exists():
        if study.core.shared.utils.digest(source) != study.core.shared.utils.digest(target):
            raise ValueError(f'Refusing to overwrite different data: {target}')
        return
    temp = target.with_suffix(target.suffix+'.importing')
    shutil.copyfile(source, temp)
    temp.replace(target)


def import_screen(source, target, expected):
    """Reuse verified geometry/graph only; never import 150-step rollout results."""
    if source is None or not (source/'offline_model.npz').exists():
        return
    # The parent checks sweep.lock. Also reject an independently active candidate.
    with (source/'run.lock').open('r') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Source candidate is active: {source}') from exc
        if read(source/'manifest.json') != expected:
            raise ValueError('Screen manifest differs from current controller/data; refusing reuse')
        panel = read(source/'panel.json')
        if panel and panel != dict(seeds=[211], steps=150):
            raise ValueError('Unexpected source panel')
        names = ['offline_model.npz', 'offline_audit.json']
        provenance = dict(source=str(source), reuse='offline_model_and_graph_only',
                          rollout_results_reused=False, target_steps=STEPS,
                          files={n: study.core.shared.utils.digest(source/n) for n in names})
        identity(target/'screen_import.json', provenance)
        for name in names:
            atomic_copy(source/name, target/name)


def summarize(folder):
    completed_rows(folder)  # Reject traces from a different rollout length before aggregation.
    result = study.core.summarize(folder, SEEDS)
    rows = result['results']
    setup = sum((r.get('failure') or {}).get('message', '').startswith('Outside Sec.7.1 setup:') for r in rows)
    result.update(setup_rejections=setup, other_aborted_rollouts=result['failed_rollouts']-setup,
                  pending_rollouts=len(SEEDS)-len(rows),
                  tracking_success_rate_completed=(result['successes']/result['full_rollouts'] if result['full_rollouts'] else None),
                  requested_steps_per_rollout=STEPS,
                  reporting_note='Setup rejections are not completed tracking trials. Only 300-step-protocol attempts are included. Timing may reflect parallel CPU contention; see runtime metadata.')
    write(folder/'summary.json', result)
    return result


def worker(args):
    folder = args.output/f'k{args.worker_cluster:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    config = replace(study.Config(), clusters=args.worker_cluster)
    with lock_file(folder/'batch_candidate.lock'), study.core.threadpool_limits(limits=1):
        refs, provenance = study.core.shared.read_references()
        expected = signature(config, provenance)
        identity(folder/'batch_protocol.json', dict(seeds=list(SEEDS), steps=STEPS,
                 config=asdict(config), runner_sha256=study.core.shared.utils.digest(__file__)))
        if not (folder/'manifest.json').exists() and any(folder.glob('*.npz')):
            raise ValueError('Refusing unversioned existing data')
        identity(folder/'manifest.json', expected)
        identity(folder/'panel.json', dict(seeds=list(SEEDS), steps=STEPS))
        if (folder/'batch_error.json').exists():
            raise RuntimeError('Previous unexpected error requires inspection; not automatically retrying')
        if (folder/'offline_rejected.json').exists():
            return
        if args.reuse_screen is not None:
            import_screen(args.reuse_screen/folder.name, folder, expected)
        try:
            study.run(config, folder, seeds=SEEDS, steps=STEPS, prepare_only=True)
        except RuntimeError as exc:
            if str(exc) != 'NMF did not converge; no model accepted':
                raise
            write(folder/'offline_rejected.json', dict(message=str(exc), blocked_seeds=list(SEEDS)))
            return
        with lock_file(folder/'run.lock'):
            rows = completed_rows(folder)
            pending = select_pending(rows, args.block_start, args.block_stop, args.batch_size)
            write(folder/'last_batch.json', dict(started_at=utc(), planned_seeds=pending,
                  max_parallel_workers=args.workers, timing_context='parallel across cluster counts; one BLAS thread per worker'))
            lib = study.Library.load(folder/'offline_model.npz', config) if pending else None
            completed_now = 0
            try:
                for seed in pending:
                    if (args.output/'STOP_AFTER_CURRENT').exists():
                        break
                    print(f'{utc()} K={config.clusters} seed={seed}', flush=True)
                    study.core.run_seed(lib, seed, STEPS, folder, refs[seed], action_fn=study.compute_action)
                    trace_row(folder/f'seed{seed}_control_results.npz', seed)
                    write(folder/f'seed{seed}_runtime.json', dict(batch_id=args.batch_id,
                          max_parallel_workers=args.workers, pid=os.getpid(), completed_at=utc(),
                          blas_threads=1, solver='CLARABEL', timing_comparable_to_serial=False if args.workers > 1 else True))
                    completed_now += 1
                    summarize(folder)
            except BaseException as exc:
                write(folder/'batch_error.json', dict(message=str(exc), type=type(exc).__name__, time=utc()))
                raise
            result = summarize(folder)
            write(folder/'progress.json', dict(state='complete' if result['panel_finished'] else 'batch_complete',
                  finished_seeds=result['finished_seeds'], requested_seeds=100,
                  completed_this_batch=completed_now, pending_seeds=result['pending_rollouts'],
                  stopped_by_request=(args.output/'STOP_AFTER_CURRENT').exists()))


def snapshot(output=OUTPUT):
    rows = []
    for k in CLUSTERS:
        folder = Path(output)/f'k{k:02d}'
        summary, progress, audit = (read(folder/n) for n in ('summary.json','progress.json','offline_audit.json'))
        rejected = read(folder/'offline_rejected.json')
        rows.append(dict(clusters=k, active_clusters=audit.get('active_clusters'),
            state='offline_rejected' if rejected else progress.get('state', 'not_started'),
            processed=summary.get('finished_seeds', 0), complete=summary.get('full_rollouts', 0),
            successes=summary.get('successes'), setup_rejections=summary.get('setup_rejections', 0),
            other_aborts=summary.get('other_aborted_rollouts', 0), blocked=100 if rejected else 0,
            pending=0 if rejected else 100-summary.get('finished_seeds', 0),
            final100_error_mm=summary.get('median_final100_error_mm_full_rollouts'),
            median_online_ms=summary.get('median_control_compute_ms')))
    return rows


def validate(args):
    requested_seeds(args.block_start, args.block_stop)
    select_pending({}, args.block_start, args.block_stop, args.batch_size)
    if type(args.workers) is not int or not 1 <= args.workers <= len(CLUSTERS):
        raise ValueError('Workers must be between 1 and 9; start with 2 on a laptop')
    if not args.clusters or len(set(args.clusters)) != len(args.clusters) or not set(args.clusters) <= set(CLUSTERS):
        raise ValueError(f'Choose distinct cluster counts from {CLUSTERS}')
    if args.worker_cluster is not None and args.worker_cluster not in args.clusters:
        raise ValueError('Worker cluster not in requested clusters')
    args.output = args.output.resolve()
    if not args.output.is_relative_to((study.core.SSD/'experiments/reacher').resolve()):
        raise ValueError('Use the external-drive experiment directory')
    if args.output == SCREEN.resolve():
        raise ValueError('Do not overwrite the historical single-seed screen')


def main(args):
    validate(args)
    if args.dry_run:
        # Read-only: no directory creation, simulation, model preparation or imports.
        plans = []
        for k in args.clusters:
            done = completed_rows(args.output/f'k{k:02d}')
            plans.append(dict(clusters=k, saved_attempts=len(done),
                 next_seeds=select_pending(done, args.block_start, args.block_stop, args.batch_size)))
        print(json.dumps(dict(total_panel_rollouts=len(CLUSTERS)*len(SEEDS), max_steps=len(CLUSTERS)*len(SEEDS)*STEPS,
             workers=args.workers, plans=plans,
             note='Only verified offline models/graphs are reused from the 150-step screen; all 300-step rollout attempts start fresh.'), indent=2))
        return
    if not study.core.SSD.is_dir():
        raise FileNotFoundError('Connect Crucial X10')
    require_idle_screen(args.reuse_screen)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker_cluster is not None:
        worker(args)
        return
    with lock_file(args.output/'batch.lock'):
        identity(args.output/'protocol.json', dict(clusters=list(CLUSTERS), seeds=list(SEEDS), steps=STEPS,
                 configs={str(k): asdict(replace(study.Config(), clusters=k)) for k in CLUSTERS},
                 runner_sha256=study.core.shared.utils.digest(__file__),
                 purpose='Development screening, not held-out evaluation'))
        if (args.output/'STOP_AFTER_CURRENT').exists():
            print('Stop request is present. Explicitly clear it before a new batch.', flush=True)
            return
        batch_id = args.batch_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        queued, active, failures = list(args.clusters), {}, []
        write(args.output/f'batch_{batch_id}.json', dict(started_at=utc(), clusters=args.clusters,
              block_start=args.block_start, block_stop=args.block_stop, batch_size=args.batch_size,
              workers=args.workers, imported_screen=str(args.reuse_screen) if args.reuse_screen else None))
        try:
            while queued or active:
                stopping = (args.output/'STOP_AFTER_CURRENT').exists() or bool(failures)
                if stopping:
                    queued.clear()
                while queued and len(active) < args.workers:
                    k = queued.pop(0)
                    command = [sys.executable, '-u', '-m', __spec__.name if __spec__ else 'comparison_methods.kde_bdpc.batches',
                               '--output', str(args.output), '--worker-cluster', str(k),
                               '--batch-size', str(args.batch_size), '--workers', str(args.workers), '--batch-id', batch_id]
                    if args.block_start is not None:
                        command += ['--block-start', str(args.block_start), '--block-stop', str(args.block_stop)]
                    command += ['--reuse-screen', str(args.reuse_screen)] if args.reuse_screen else ['--no-reuse-screen']
                    log = (args.output/f'k{k:02d}_batch.log').open('a')
                    try:
                        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    except BaseException:
                        log.close()
                        raise
                    active[k] = (child, log)
                for k, (child, log) in list(active.items()):
                    code = child.poll()
                    if code is not None:
                        log.close(); del active[k]
                        if code:
                            failures.append(dict(clusters=k, exit_code=code))
                            # Let other workers finish their current rollout, then stop.
                            (args.output/'STOP_AFTER_CURRENT').touch()
                write(args.output/'batch_progress.json', dict(state='running', batch_id=batch_id,
                      active={str(k): child.pid for k,(child,log) in active.items()}, queued=queued, failures=failures))
                write(args.output/'panel_summary.json', snapshot(args.output))
                if queued or active:
                    time.sleep(1)
        except KeyboardInterrupt:
            (args.output/'STOP_AFTER_CURRENT').touch()
            print('Stop requested; workers will finish their current rollout.', flush=True)
            for child, log in active.values():
                child.wait(); log.close()
        except BaseException:
            (args.output/'STOP_AFTER_CURRENT').touch()
            for child, log in active.values():
                child.wait(); log.close()
            raise
        finally:
            write(args.output/'panel_summary.json', snapshot(args.output))
        write(args.output/'batch_progress.json', dict(state='error' if failures else 'batch_complete',
              failures=failures, batch_id=batch_id, finished_at=utc(),
              stop_requested=(args.output/'STOP_AFTER_CURRENT').exists()))
        if failures:
            raise RuntimeError(f'Worker errors: {failures}; inspect per-cluster logs before resuming')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--clusters', type=int, nargs='+', default=list(CLUSTERS))
    p.add_argument('--batch-size', type=int, default=10, help='Maximum NEW seed attempts per cluster this invocation')
    p.add_argument('--block-start', type=int)
    p.add_argument('--block-stop', type=int)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--dry-run', action='store_true')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--reuse-screen', type=Path, default=SCREEN)
    group.add_argument('--no-reuse-screen', action='store_const', const=None, dest='reuse_screen')
    p.add_argument('--worker-cluster', type=int, help=argparse.SUPPRESS)
    p.add_argument('--batch-id', help=argparse.SUPPRESS)
    return p


if __name__ == '__main__':
    main(parser().parse_args())
