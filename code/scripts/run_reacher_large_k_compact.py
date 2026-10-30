"""Run/resume the compact K=80/96 sweep, with an exclusive lock and status file."""

import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/atlas_deepc_mpl')

from atlas_deepc.ablations import compact_large_k_rank as study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=study.RESULT_ROOT)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--pilot', action='store_true', help='Separate one-seed/rank-62 storage-validation run')
    args = parser.parse_args()
    root = args.root / 'pilot' if args.pilot else args.root
    ranks, seeds = ((62,), (211,)) if args.pilot else (study.RANKS, study.SEEDS)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('This sweep is already running')
        status = dict(pid=os.getpid(), status='preparing' if args.prepare_only else 'running',
            started_utc=datetime.now(timezone.utc).isoformat(), workers=args.workers,
            requested_sweep_rollouts=len(study.COUNTS)*len(ranks)*4*len(seeds),
            selected_serial_rollouts=len(study.COUNTS)*4*len(seeds), storage_policy=study.STORAGE_POLICY)
        study.four.save_json(root / 'run_status.json', status)
        try:
            if args.prepare_only:
                study.prepare(root, ranks=ranks, seeds=seeds, workers=args.workers)
                status['status'] = 'prepared_not_started'
            else:
                study.run_all(root, ranks=ranks, seeds=seeds, workers=args.workers)
                status['status'] = 'complete'
        except BaseException as exc:
            status.update(status='stopped', error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            status['updated_utc'] = datetime.now(timezone.utc).isoformat()
            study.four.save_json(root / 'run_status.json', status)


if __name__ == '__main__':
    main()
