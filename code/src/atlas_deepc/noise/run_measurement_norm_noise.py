"""Local launcher for the measurement-norm noise study (1,400 rollouts)."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"


from atlas_deepc.common.files import exclusive_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ratios", type=float, nargs="+", default=[0, .01, .02, .03, .04, .05, .10])
    args = parser.parse_args()
    if args.workers < 1 or len(set(args.ratios)) != len(args.ratios) or not set(args.ratios) <= {0, .01, .02, .03, .04, .05, .10}:
        parser.error("Use positive workers and distinct supported noise levels")
    import cvxpy as cp
    from atlas_deepc.noise import measurement_norm_noise_rank59 as rank59
    from atlas_deepc.noise import measurement_norm_noise_rank62 as rank62
    assert "CLARABEL" in cp.installed_solvers(), "Install Clarabel in this Python environment"
    for path in (rank59.baseline.DATA_PATH, rank59.baseline.ATLAS_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Copy the project data and atlas with the project: {path}")
    with exclusive_lock(rank59.RESULT_ROOT.parent / "measurement_norm_noise_study.lock"):
        print(f"Python: {sys.executable}; {args.workers} worker(s); 100 seeds per level per rank", flush=True)
        rank59.run_all(args.ratios, workers=args.workers)
        rank62.run_all(args.ratios, workers=args.workers)
    print("Both rank panels complete. Run the notebook report cells.", flush=True)


if __name__ == "__main__":
    main()
