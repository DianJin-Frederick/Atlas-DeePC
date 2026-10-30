"""Run one resumable batch of the three missing low-K Reacher methods."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

from atlas_deepc.ablations import low_k_completion as study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=study.RESULT_ROOT)
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Maximum NEW rollouts this invocation; may stop within a rank panel")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel rank panels; not a latency benchmark")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    result = (study.report(args.root) if args.report_only else
              study.run_batch(args.root, args.batch_size, args.workers))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
