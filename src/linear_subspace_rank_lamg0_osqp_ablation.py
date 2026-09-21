"""Launch the isolated lambda_g=0, OSQP rank/seed grid."""

import fcntl
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments/routed_deepc_iid/linear_subspace_rank_lamg0_osqp_100_v1"
RUNNER = RESULT / "frozen_protocol/src/linear_subspace_rank_ablation.py"


if __name__ == "__main__":
    assert RUNNER.is_file(), "The isolated source snapshot must exist before launch."
    lock = (RESULT / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("This lambda_g=0 OSQP sweep already has an active launcher.")
    os.set_inheritable(lock.fileno(), True)
    (RESULT / "launch.json").write_text(json.dumps({
        "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "runner": str(RUNNER), "arguments": sys.argv[1:],
    }, indent=2))
    os.execv(sys.executable, [sys.executable, "-u", str(RUNNER), *sys.argv[1:]])
