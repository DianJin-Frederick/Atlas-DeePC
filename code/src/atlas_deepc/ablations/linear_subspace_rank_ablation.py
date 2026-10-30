"""Dense rank-only sweep of the current baseline, with compact checkpoints."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from atlas_deepc.common.paths import source_hashes, source_path

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile
import time
from datetime import datetime, timezone

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.baselines import baseline_atlas_deepc as baseline
from atlas_deepc.control import expert_pipeline_search as pipe
from atlas_deepc.atlas import routed_deepc_experiment as routed
from atlas_deepc.benchmarks.baseline_1000_benchmark import summarize_timing
from atlas_deepc.atlas.candidate_goal_routing_experiment import TIMING_KEYS, file_digest, strict_success

VERSION = "linear_subspace_rank_100_v1"
RESULT_ROOT = baseline.ROOT / "experiments/routed_deepc_iid" / VERSION
RANKS = tuple(range(20, 81))
SEEDS = tuple(range(211, 311))
_H = _atlas = _controller = _thread_limit = None


def write_json(path, payload):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def rank_config(rank):
    return replace(baseline.default_config(), svd_rank=int(rank))


def prepare():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    signature = {
        "version": VERSION, "base_config": asdict(baseline.default_config()),
        "data_sha256": file_digest(baseline.DATA_PATH), "atlas_sha256": file_digest(baseline.ATLAS_PATH),
        "matrix_geometry": "linear", "candidate_count": 10, "output_dimension": 10,
        "noise_sd_ratio": 0.0,
        "source_sha256": source_hashes((
            "linear_subspace_rank_ablation.py", "baseline_atlas_deepc.py",
            "routed_deepc_experiment.py", "candidate_goal_routing_experiment.py",
            "expert_pipeline_search.py", "goal_reference.py",
            "baseline_1000_benchmark.py")),
    }
    path = RESULT_ROOT / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != signature:
            raise ValueError("Incompatible rank experiment; choose a new result directory")
    else:
        if any(RESULT_ROOT.glob("rank*/seed*.npz")):
            raise ValueError("Cannot reuse unversioned rank checkpoints")
        write_json(path, signature)
    audit_path = RESULT_ROOT / "fairness_audit.json"
    if not audit_path.exists():
        pipe.P = 10
        audit = routed.audit_iid_library(np.load(baseline.DATA_PATH), eval_seeds=SEEDS)
        write_json(audit_path, audit)
    return signature


def initialize_worker():
    global _H, _atlas, _controller, _thread_limit
    _thread_limit = threadpool_limits(limits=1)
    pipe.P = 10
    _H = np.load(baseline.DATA_PATH, mmap_mode="r")
    _atlas = routed.load_atlas(baseline.ATLAS_PATH)
    _controller = None


def controller_for_rank(rank):
    global _controller
    if _controller is None or _controller.config.svd_rank != rank:
        _controller = baseline.BaselineAtlasDeePC(_H, _atlas, rank_config(rank))
        for matrix, _ in _controller.matrix_cache.values():
            assert matrix["basis"].shape == (144, rank)
            assert not np.any(matrix["offset"])
            assert "g_sum" not in matrix
    return _controller


def checkpoint_path(rank, seed):
    return RESULT_ROOT / f"rank{rank:03d}" / f"seed{seed}.npz"


def validate_row(row, rank, seed):
    expected = asdict(replace(rank_config(rank), seed=seed))
    assert all(row[key] == value for key, value in expected.items())
    assert row["rank_ablation_version"] == VERSION
    assert row["baseline_version"] == baseline.BASELINE_VERSION
    assert row["matrix_geometry"] == "linear" and row["candidate_count"] == 10
    assert row["terminal_reference_policy"] == "iterative_minimum_route_cost"
    assert row["output_dimension"] == 10 and row["noise_sd_ratio"] == 0.0
    assert row["minimum_selected_goal_source_separation"] >= .06
    assert not row["fitted_dynamics_model"] and not row["online_nearest_trajectory_retrieval"]


def load_checkpoint(path, rank, seed):
    with np.load(path, allow_pickle=False) as trace:
        row = json.loads(trace["summary"].item())
        validate_row(row, rank, seed)
        for key in ("distance", *TIMING_KEYS):
            values = trace[key]
            assert values.shape == (row["steps"],)
            assert np.all(np.isfinite(values)) and np.all(values >= 0)
        for window in (30, 100):
            error = trace["distance"][-window:]
            np.testing.assert_allclose(row[f"mean_last_{window}"], error.mean())
            np.testing.assert_allclose(row[f"max_last_{window}"], error.max())
            assert row[f"success_{window}"] == bool(error.mean() < .01 and error.max() < .02)
    return row


def compact_trace(source, destination, rank, seed):
    with np.load(source, allow_pickle=True) as trace:
        row = json.loads(trace["summary"].item())
        assert trace["y"].shape == (row["steps"], 10)
        target = trace["final_goal"].reshape(-1)[6:8]
        np.testing.assert_allclose(trace["distance"], np.linalg.norm(trace["y"][:, 6:8]-target, axis=1))
        arrays = {key: trace[key] for key in ("distance", *TIMING_KEYS)}
        row.update(rank_ablation_version=VERSION, output_dimension=10, noise_sd_ratio=0.0,
                   exact_target_xy=target.tolist(), result_path=str(destination),
                   ever_reached_10mm=bool(np.min(trace["distance"]) < .01),
                   tail_time_below_10mm=float(np.mean(trace["distance"][-100:] < .01)))
    validate_row(row, rank, seed)
    # An interrupted write must never masquerade as a completed checkpoint.
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays, summary=np.array(json.dumps(row)))
    load_checkpoint(temporary, rank, seed)
    temporary.replace(destination)
    return row


def run_one(rank, seed):
    path = checkpoint_path(rank, seed)
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        row = load_checkpoint(path, rank, seed)
    else:
        controller = controller_for_rank(rank)
        with tempfile.TemporaryDirectory(prefix="atlas_rank_") as directory:
            source = Path(directory) / "trace.npz"
            controller.run(seed, source)
            row = compact_trace(source, path, rank, seed)
        validate_row(row, rank, seed)
    return rank, seed


def summarize_rank(rank, seeds):
    rows = []
    paths = [checkpoint_path(rank, seed) for seed in seeds]
    for seed, path in zip(seeds, paths):
        row = load_checkpoint(path, rank, seed)
        rows.append(row)
    timing = summarize_timing(paths)
    payload = {"version": VERSION, "rank": rank, "seeds": list(seeds), "results": rows,
               "successes": sum(strict_success(row) for row in rows),
               "median_tail100_mm": float(1000*np.median([row["mean_last_100"] for row in rows])),
               "p90_tail100_mm": float(1000*np.percentile([row["mean_last_100"] for row in rows], 90)),
               "unsolved_qps": sum(row["unsolved_steps"] for row in rows), "timing": timing}
    write_json(RESULT_ROOT / f"rank{rank:03d}" / "summary.json", payload)
    return {key: payload[key] for key in ("rank", "successes", "median_tail100_mm", "p90_tail100_mm", "unsolved_qps")}


def run_all(ranks=RANKS, seeds=SEEDS, workers=6):
    ranks, seeds = list(map(int, ranks)), list(map(int, seeds))
    if not ranks or len(set(ranks)) != len(ranks) or min(ranks) < 1 or max(ranks) >= 144:
        raise ValueError("Ranks must be unique integers between 1 and 143")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique")
    if workers < 1:
        raise ValueError("workers must be positive")
    signature = prepare()
    order = ([60] if 60 in ranks else []) + [rank for rank in ranks if rank != 60]
    completed = {}
    started = time.monotonic()
    session = {"started_utc": datetime.now(timezone.utc).isoformat(), "workers": workers,
               "numerical_threads": 1, "ranks": ranks, "seeds": seeds,
               "load_average_at_start": os.getloadavg()}
    write_json(RESULT_ROOT / f"session_{time.time_ns()}.json", session)
    print(f"Ranks {ranks}; {len(seeds)} seeds/rank; {workers} workers; rank 60 first", flush=True)
    pool = ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker)
    pending = set()
    try:
        for rank in order:
            # Only one job per worker is outstanding. A failure cannot drain a
            # pre-submitted 6,100-job queue; at most current rollouts may finish.
            remaining = iter(seeds)
            pending = {pool.submit(run_one, rank, seed) for seed in [next(remaining, None) for _ in range(min(workers, len(seeds)))]}
            count = 0
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                # Inspect the entire finished batch before submitting more work.
                for future in done:
                    actual_rank, seed = future.result()
                    assert actual_rank == rank and seed in seeds
                    count += 1
                for _ in done:
                    seed = next(remaining, None)
                    if seed is not None:
                        pending.add(pool.submit(run_one, rank, seed))
                if count % 10 == 0 or count == len(seeds):
                    print(f"rank {rank}: {count}/{len(seeds)}; elapsed {time.monotonic()-started:.0f}s", flush=True)
            completed[rank] = summarize_rank(rank, seeds)
            print(f"RANK COMPLETE {completed[rank]}", flush=True)
            write_json(RESULT_ROOT / "summary.json", {
                    "version": VERSION, "signature": signature, "ranks": ranks, "seeds": seeds,
                    "workers": workers, "numerical_threads": 1,
                    "timing_note": "Shared-computer wall clock; rank-grouped jobs, first five steps excluded; QP includes construction and solve",
                    "completed": [completed[key] for key in sorted(completed)],
                    "complete": len(completed) == len(ranks),
                })
    except BaseException:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    assert len(completed) == len(ranks)
    return completed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    args = parser.parse_args()
    run_all(args.ranks, workers=args.workers)
