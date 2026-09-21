"""Portable, resumable lambda-g sweep using the graph-waypoint baseline.

Importing this module never runs experiments. The notebook launches this file
as a separate process so multiprocessing also works on Windows and macOS.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import platform
import tempfile
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import cvxpy as cp
import numpy as np
from threadpoolctl import threadpool_limits

import baseline_atlas_deepc as baseline
import routed_deepc_experiment as routed
from candidate_goal_routing_experiment import file_digest

VERSION = "baseline_linear_n10_graph_clarabel_lam_g_v1"
EXPECTED_DATA = "75c6e2e8376c00ac1fc032ecaaa31b9a3eb830b671f68a18e559b22f9263c45b"
EXPECTED_ATLAS = "b8641b1c376a2f4af5dcf9415a5d591a36e99bb223b1c0f7c17390de16803007"
SOURCES = (Path(__file__).name, "baseline_atlas_deepc.py", "routed_deepc_experiment.py",
           "candidate_goal_routing_experiment.py", "expert_pipeline_search.py",
           "stitched_atlas_mpc_experiment.py", "local_affine_atlas_experiment.py")
PACKAGES = ("numpy", "scipy", "cvxpy", "clarabel", "gymnasium", "mujoco", "scikit-learn")


def default_config():
    # The published notebook explicitly overrides the module's historical default.
    return replace(baseline.default_config(), lam_g=0.05)


def lambda_directory_name(index, value):
    return f"lam_g_{index:03d}_{float(value):.12e}".replace("+", "p").replace("-", "m")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def exclusive_run(output):
    """OS lock releases on process exit, including interrupted runs."""
    with (output / "run.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            if not stream.read(1):
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def make_signature(values, seeds, data_path, atlas_path, workers):
    values, seeds = list(map(float, values)), list(map(int, seeds))
    if not values or len(set(values)) != len(values) or any(not np.isfinite(v) or v < 0 for v in values):
        raise ValueError("Lambda values must be unique, finite and nonnegative")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or 2026 <= s <= 2225 for s in seeds):
        raise ValueError("Use unique nonnegative testing seeds outside offline seeds 2026–2225")
    if int(workers) != workers or workers < 1:
        raise ValueError("workers must be a positive integer")
    if "CLARABEL" not in cp.installed_solvers():
        raise RuntimeError("Install Clarabel in the notebook's Python environment")
    data_hash, atlas_hash = file_digest(data_path), file_digest(atlas_path)
    if (data_hash, atlas_hash) != (EXPECTED_DATA, EXPECTED_ATLAS):
        raise ValueError("Use the original baseline offline data and atlas; file hashes differ")
    config = default_config()
    assert config.clusters == 48 and config.svd_rank == 60 and config.steps == 320
    return {"version": VERSION, "base_config": asdict(config), "parameter_values": values,
            "seeds": seeds, "solver": "CLARABEL", "solver_options": {"max_iter": 10000},
            "matrix_geometry": "linear", "candidate_count": 10, "reference_mode": "graph",
            "data_sha256": data_hash, "atlas_sha256": atlas_hash,
            "source_sha256": {name: file_digest(Path(__file__).with_name(name)) for name in SOURCES},
            "packages": {name: version(name) for name in PACKAGES},
            "workers": int(workers), "numerical_threads_per_worker": 1,
            "timing_discard_initial_steps": 5, "warmup_seed": 210, "warmup_steps": 10,
            "order": "rotate lambda values within each seed"}


def validate_row(row, lam_g, seed):
    expected = asdict(replace(default_config(), lam_g=float(lam_g), seed=int(seed)))
    if any(row.get(k) != v for k, v in expected.items()):
        raise ValueError("Checkpoint configuration does not match the requested baseline")
    assert row["parameter_ablation_version"] == VERSION
    assert row["matrix_geometry"] == "linear" and row["candidate_count"] == 10
    assert row["dynamic_terminal_reference"] and row["graph_waypoints_enabled"]
    assert row["reference_mode"] == "graph" and row["solver"] == "CLARABEL"
    assert row["terminal_reference_policy"] == "iterative_minimum_route_cost"
    assert sum(row["statuses"].values()) == 320
    assert row["minimum_selected_goal_source_separation"] >= .06
    assert len(row["exact_target_xy"]) == 2 and np.isfinite(row["exact_target_xy"]).all()
    for window in (30, 100):
        mean, maximum = row[f"mean_last_{window}"], row[f"max_last_{window}"]
        assert np.isfinite(mean) and np.isfinite(maximum) and mean >= 0 and maximum >= 0
        # Averaging nearly constant errors can round a few ulps above max().
        # Tolerate only reduction roundoff; retain the measured values and the
        # exact success thresholds below without clipping or recomputing them.
        tolerance = 8 * np.finfo(float).eps * max(mean, maximum)
        assert mean <= maximum + tolerance, (
            f"Invalid final-{window} error statistics for seed={seed}, lam_g={lam_g}: "
            f"mean={mean!r}, maximum={maximum!r}, tolerance={tolerance!r}"
        )
        assert row[f"success_{window}"] == (mean < .01 and maximum < .02)
    return row


def load_lam_g_sweep(output_root):
    """Read atomic checkpoints, including interrupted partial panels."""
    root = Path(output_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["signature"]["version"] != VERSION:
        raise ValueError("Historical control-parameter results cannot be reused here")
    sig = manifest["signature"]
    payloads, targets = {}, {}
    for index, lam in enumerate(sig["parameter_values"]):
        directory = root / lambda_directory_name(index, lam)
        expected_paths = {directory / f"seed{seed}.json" for seed in sig["seeds"]}
        if set(directory.glob("seed*.json")) - expected_paths:
            raise ValueError(f"Unexpected seeds in {directory}")
        rows = []
        for seed in sig["seeds"]:
            path = directory / f"seed{seed}.json"
            if not path.exists():
                continue
            row = validate_row(json.loads(path.read_text(encoding="utf-8")), lam, seed)
            if seed in targets:
                np.testing.assert_array_equal(row["exact_target_xy"], targets[seed])
            targets[seed] = row["exact_target_xy"]
            rows.append(row)
        payloads[lam] = {"config": asdict(replace(default_config(), lam_g=lam)),
                         "complete": len(rows) == len(sig["seeds"]), "results": rows}
    return manifest, payloads


_CONTROLLER = _LIMIT = None


def freeze(value):
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
    elif isinstance(value, dict):
        for v in value.values():
            freeze(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            freeze(v)


def initialize_worker(data_path, atlas_path):
    global _CONTROLLER, _LIMIT
    _LIMIT = threadpool_limits(limits=1)
    H = np.load(data_path, mmap_mode="r")
    atlas = routed.load_atlas(Path(atlas_path))
    _CONTROLLER = baseline.BaselineAtlasDeePC(H, atlas, default_config())
    assert H.shape == (144, 37800) and not H.flags.writeable
    assert len(_CONTROLLER.matrix_cache) == 48
    for matrix, _ in _CONTROLLER.matrix_cache.values():
        assert matrix["basis"].shape == (144, 60)
        assert not np.any(matrix["offset"]) and "g_sum" not in matrix
    freeze(atlas)
    freeze(_CONTROLLER.matrix_cache)
    freeze(_CONTROLLER.route_tables)
    _CONTROLLER.config = replace(default_config(), steps=10)
    _CONTROLLER.run(210, reference_mode="graph")
    _CONTROLLER.config = default_config()


def run_one(task):
    lam, seed = task
    # Lambda changes only the online objective; share the same offline matrices.
    _CONTROLLER.config = replace(default_config(), lam_g=lam)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="atlas_lam_g_") as directory:
        path = Path(directory) / "trace.npz"
        row = _CONTROLLER.run(seed, path, reference_mode="graph")
        with np.load(path, allow_pickle=True) as trace:
            assert trace["distance"].shape == (320,)
            target = trace["final_goal"].reshape(-1)[6:8]
            np.testing.assert_allclose(trace["distance"], np.linalg.norm(trace["y"][:, 6:8] - target, axis=1))
            row["exact_target_xy"] = target.tolist()
            # Common physical target across lambda values; joint references may differ.
            row["closed_loop_task_cost"] = float(row["q_xy"] * np.sum(trace["distance"]**2)
                                                 + row["r"] * np.sum(trace["u"]**2))
            for source, label in (("control_compute_time_s", "control_compute"), ("solve_time_s", "solve")):
                values = 1000 * trace[source][5:]
                assert np.isfinite(values).all() and (values >= 0).all()
                for q, name in ((50, "median"), (95, "p95"), (99, "p99")):
                    row[f"{name}_{label}_time_ms"] = float(np.percentile(values, q))
    row.update(parameter_ablation_version=VERSION, solver="CLARABEL", timed_steps=315,
               rollout_wall_seconds=time.perf_counter() - start)
    return validate_row(row, lam, seed)


def run_sweep(settings):
    data, atlas, output = (Path(settings[k]).expanduser().resolve() for k in ("data_path", "atlas_path", "output_root"))
    signature = make_signature(settings["lam_g_values"], settings["seeds"], data, atlas, settings["workers"])
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(output):
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["signature"] != signature:
                raise ValueError("Incompatible results: use a new output folder for changed settings, sources or packages")
        else:
            if any(output.glob("lam_g_*/*")):
                raise ValueError("Unversioned results: choose an empty output folder")
            manifest = {"signature": signature, "complete": False}
            write_json(manifest_path, manifest)
        _, payloads = load_lam_g_sweep(output)
        done = {(lam, row["seed"]) for lam, payload in payloads.items() for row in payload["results"]}
        targets = {row["seed"]: row["exact_target_xy"] for payload in payloads.values() for row in payload["results"]}
        values, seeds = signature["parameter_values"], signature["seeds"]
        tasks = [(values[i], seed) for j, seed in enumerate(seeds)
                 for i in [(k+j) % len(values) for k in range(len(values))]
                 if (values[i], seed) not in done]
        if not tasks:
            manifest["complete"] = True
            write_json(manifest_path, manifest)
            print("All requested rollouts are already complete.", flush=True)
            return
        session = {"started_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
                   "python": platform.python_version(), "pending": len(tasks), "new_rollouts": 0}
        session_path = output / ("session_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
        write_json(session_path, session)
        print(f"Resuming {len(done)} saved; {len(tasks)} missing rollouts; {settings['workers']} worker(s).", flush=True)
        executor = None
        try:
            if settings["workers"] == 1:
                initialize_worker(str(data), str(atlas))
                results = map(run_one, tasks)
            else:
                executor = ProcessPoolExecutor(max_workers=settings["workers"],
                    mp_context=multiprocessing.get_context("spawn"), initializer=initialize_worker,
                    initargs=(str(data), str(atlas)))
                results = executor.map(run_one, tasks, chunksize=1)
            for (lam, seed), row in zip(tasks, results):
                if seed in targets:
                    np.testing.assert_array_equal(row["exact_target_xy"], targets[seed])
                targets[seed] = row["exact_target_xy"]
                directory = output / lambda_directory_name(values.index(lam), lam)
                directory.mkdir(exist_ok=True)
                write_json(directory / f"seed{seed}.json", row)
                done.add((lam, seed))
                session["new_rollouts"] += 1
                if session["new_rollouts"] % 25 == 0 or len(done) == len(values)*len(seeds):
                    if make_signature(values, seeds, data, atlas, settings["workers"]) != signature:
                        raise RuntimeError("Offline data or numerical sources changed during the run")
                    write_json(session_path, session)
                    print(f"{len(done)}/{len(values)*len(seeds)} | lambda={lam:.6g}, seed={seed}, success={row['success_30'] and row['success_100']}", flush=True)
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            write_json(session_path, session)
        _, payloads = load_lam_g_sweep(output)
        manifest["complete"] = all(p["complete"] for p in payloads.values())
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        print("Sweep complete. Run the notebook's analysis cells.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    run_sweep(json.loads(args.settings.read_text(encoding="utf-8")))
