"""Run the resumable K=64 sweep with a process lock and persistent status."""

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

from atlas_deepc.ablations import k64_rank as study


def stage_panel_atlases(root):
    """Publish shared atlases before graph and terminal workers can read them."""
    root = Path(root)
    inventory = json.loads((root / "atlas_inventory.json").read_text())
    for dimension in (8, 10):
        source = study.atlas_path(root, dimension)
        expected = inventory[str(dimension)]["sha256"]
        if study.four.file_digest(source) != expected:
            raise ValueError(f"Frozen atlas checksum mismatch: {source}")
        for rank in study.RANKS:
            destination = study.panel_directory(root, rank, dimension, "terminal").parent / "behavioral_atlas.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if study.four.file_digest(destination) != expected:
                    raise ValueError(f"Existing panel atlas checksum mismatch: {destination}")
                continue
            descriptor, name = tempfile.mkstemp(dir=destination.parent, suffix=".pending.npz")
            os.close(descriptor)
            temporary = Path(name)
            try:
                shutil.copy2(source, temporary)
                if study.four.file_digest(temporary) != expected:
                    raise ValueError(f"Staged atlas checksum mismatch: {temporary}")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)


def main():
    root = study.RESULT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A K=64 rank sweep is already running")
        status = {"pid": os.getpid(), "status": "running",
                  "started_utc": datetime.now(timezone.utc).isoformat(),
                  "workers": 4, "requested_sweep_rollouts": 17200,
                  "selected_serial_rollouts": 400,
                  "resume_command": f"{sys.executable} {Path(__file__).resolve()}"}
        study.four.save_json(root / "run_status.json", status)
        try:
            study.prepare(root, workers=4)
            stage_panel_atlases(root)
            print("Shared atlases verified. Resuming saved panels with four workers.", flush=True)
            study.run_all(root, workers=4)
        except BaseException as exc:
            status.update(status="stopped", error=str(exc), traceback=traceback.format_exc())
            raise
        else:
            status["status"] = "complete"
        finally:
            status["updated_utc"] = datetime.now(timezone.utc).isoformat()
            study.four.save_json(root / "run_status.json", status)


if __name__ == "__main__":
    main()
