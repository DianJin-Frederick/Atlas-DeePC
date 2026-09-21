"""Fresh single-worker OSQP timing runs for the baseline notebook."""

from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
from unittest.mock import patch
from uuid import uuid4

import baseline_atlas_deepc as baseline
import baseline_osqp_comparison as experiment
from candidate_goal_routing_experiment import file_digest


RESULT_ROOT = baseline.RESULT_ROOT / "local_osqp_benchmarks"
_initialize = experiment.initialize_worker


def initialize_benchmark_worker(output_dir):
    _initialize(output_dir)
    controller = experiment._controller
    warmup = baseline.BaselineAtlasDeePC(
        controller.H, controller.atlas, replace(controller.config, steps=10))
    with experiment.use_osqp([]):
        warmup.run(210)


def run_fresh_panel(seeds, parent_dir=RESULT_ROOT, workers=1):
    if workers != 1:
        raise ValueError("Local latency benchmarks require WORKERS=1")
    seeds = list(map(int, seeds))
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and unique")
    started = datetime.now(timezone.utc)
    output_dir = Path(parent_dir) / (started.strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        "kind": "fresh_single_worker_osqp", "workers": 1, "numerical_threads": 1,
        "warmup_seed": 210, "warmup_steps": 10, "discard_initial_steps_per_rollout": 5,
        "reused_rollouts": 0, "started_utc": started.isoformat(),
        "host": platform.node(), "platform": platform.platform(), "python": platform.python_version(),
        "load_average_at_start": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "benchmark_source_sha256": file_digest(Path(__file__)),
        "timing_caveat": "Single rollout worker; stop unrelated heavy jobs for an uncontended latency measurement.",
    }
    (output_dir / "benchmark_protocol.json").write_text(json.dumps(protocol, indent=2))
    # The initializer is a top-level function, so spawned workers can import it.
    with patch.object(experiment, "initialize_worker", initialize_benchmark_worker):
        payload = experiment.run_panel(seeds, output_dir, workers=1)
    protocol["completed_utc"] = datetime.now(timezone.utc).isoformat()
    payload["benchmark_protocol"] = protocol
    (output_dir / "benchmark_protocol.json").write_text(json.dumps(protocol, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(211, 1211)))
    parser.add_argument("--parent-dir", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()
    print(run_fresh_panel(args.seeds, args.parent_dir), flush=True)
