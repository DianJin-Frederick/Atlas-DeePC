"""Small, resumable development panels for the unchanged contextual controller.

Parallel panels select parameters by control performance, NOT computation time.
Results retain raw trajectories for common-weight evaluation. No automatic grid.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
import fcntl
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.benchmarks import contextual_dpc_reacher as c

ROOT = c.ROOT / "experiments/reacher/contextual_dpc_tuning_v1"


def worker(config, seed, steps, directory):
    directory = Path(directory)
    job = directory / "jobs" / f"seed{seed}"
    job.mkdir(parents=True, exist_ok=True)
    with (job / "run.log").open("a", buffering=1) as log, redirect_stdout(log), redirect_stderr(log), threadpool_limits(limits=1):
        lib = c.Library.from_path(c.DATA, config)
        result = c.run_seed(lib, seed, steps, job)
        (job / f"seed{seed}_control_results.npz").replace(directory / f"seed{seed}_control_results.npz")
    return result


def panel(directory, seeds):
    rows = []
    for seed in seeds:
        path = directory / f"seed{seed}_control_results.npz"
        if path.exists():
            with np.load(path) as a:
                rows.append(json.loads(a["summary"].item()))
    result = {"requested_seeds": seeds, "completed": len(rows), "results": rows,
              "complete": len(rows) == len(seeds), "timing_valid_for_serial_comparison": False}
    if rows:
        result.update(successes=sum(r["strict_success"] for r in rows),
            success_rate=float(np.mean([r["strict_success"] for r in rows])),
            median_final100_mm=float(np.median([r["mean_last_100"] for r in rows])*1000),
            mean_final100_mm=float(np.mean([r["mean_last_100"] for r in rows])*1000),
            mean_xy_squared_error_sum_m2=float(np.mean([r["xy_squared_error_sum_m2"] for r in rows])),
            fallback_steps=sum(r["fallback_steps"] for r in rows))
    return result


def run(name, config, seeds, steps=300, workers=2):
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Use a simple candidate directory name")
    if not seeds or len(set(seeds)) != len(seeds) or not all(211 <= s <= 310 for s in seeds):
        raise ValueError("Unique development seeds in 211--310 required")
    if steps < 100 or workers not in (1, 2, 3):
        raise ValueError("At least 100 steps and 1--3 workers required")
    directory = ROOT / name
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signature = {"config": asdict(config), "steps": steps, "controller_sha256": c.digest(c.__file__),
            "tuner_sha256": c.digest(__file__), "data_sha256": c.digest(c.DATA), "workers": workers,
            "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "cvxpy", "clarabel", "gymnasium", "mujoco")},
            "purpose": "Development tuning; do not report parallel timings as serial latency",
            "selection_rule": "Strict success, then mean final100 error; no cross-weight native-cost ranking"}
        path = directory / "manifest.json"
        if path.exists():
            if json.loads(path.read_text()) != signature:
                raise ValueError("Incompatible candidate cache; use a new name")
        else:
            c.write_json(path, signature)
        c.write_json(directory / "requested_seeds.json", seeds)
        remaining = [s for s in seeds if not (directory / f"seed{s}_control_results.npz").exists()]
        print(f"Candidate {name}: {config}; {len(remaining)}/{len(seeds)} new seeds; {workers} workers", flush=True)
        begun = time.perf_counter()
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
                pending = {executor.submit(worker, config, seed, steps, directory): seed for seed in remaining}
                for future in as_completed(pending):
                    row = future.result()
                    summary = panel(directory, seeds)
                    c.write_json(directory / "summary.json", summary)
                    print(f"{name}: {summary['completed']}/{len(seeds)}, {summary['successes']} successes; seed {row['seed']} success={row['strict_success']}, final100={1000*row['mean_last_100']:.3f} mm, fallbacks={row['fallback_steps']}", flush=True)
            summary = panel(directory, seeds)
            summary["invocation_wall_seconds"] = time.perf_counter() - begun
            c.write_json(directory / "summary.json", summary)
            print(json.dumps({k: v for k, v in summary.items() if k not in ("results", "requested_seeds")}, indent=2), flush=True)
        except BaseException as exc:
            c.write_json(directory / "interruption.json", {"type": type(exc).__name__, "message": str(exc), "updated_unix": time.time()})
            raise
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--seeds", default="211,222,258,281", help="Comma-separated development seeds, or all")
    p.add_argument("--q", type=float, required=True)
    p.add_argument("--r", type=float, default=.1)
    p.add_argument("--lambda-g-bar", type=float, default=.1)
    p.add_argument("--lambda-sigma", type=float, default=100.)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    seeds = list(range(211, 311)) if args.seeds == "all" else [int(s) for s in args.seeds.split(",")]
    run(args.name, c.Config(q_xy=args.q, r=args.r, lambda_g_bar=args.lambda_g_bar, lambda_sigma=args.lambda_sigma), seeds, workers=args.workers)
