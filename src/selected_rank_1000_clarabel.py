"""Locked launcher for the frozen serial, paired 1,000-seed experiment."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments/routed_deepc_iid/selected_ranks_1000_clarabel_v1"
RUNNER = RESULT / "frozen_protocol/src/selected_rank_1000_experiment.py"

if __name__ == "__main__":
    lock = (RESULT / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("The selected-rank experiment is already running.")
    os.set_inheritable(lock.fileno(), True)
    (RESULT / "launch.json").write_text(json.dumps({"pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "runner": str(RUNNER),
        "arguments": sys.argv[1:]}, indent=2))
    os.execv(sys.executable, [sys.executable, "-u", str(RUNNER), *sys.argv[1:]])
