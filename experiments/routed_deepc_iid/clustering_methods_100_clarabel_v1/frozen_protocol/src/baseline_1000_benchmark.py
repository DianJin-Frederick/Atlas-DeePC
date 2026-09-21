"""Single-method baseline evaluation with resumable rollouts and stage timings."""

import argparse
import gc
import json
import os
import platform
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

import numpy as np
from threadpoolctl import threadpool_limits

import baseline_atlas_deepc as baseline
import routed_deepc_experiment as routed
from candidate_goal_routing_experiment import file_digest
from stitched_atlas_mpc_experiment import offline_environment_targets


STAGES = {
    "Chart localization": "localization_time_s",
    "Terminal-reference selection": "goal_selection_time_s",
    "Graph routing": "routing_time_s",
    "Linear subspace retrieval": "matrix_preparation_time_s",
    "QP construction and solve": "solve_time_s",
}
DISCARD = 5


def summarize_timing(paths):
    collected = {name: [] for name in [*STAGES, "Other control-loop work", "Total online control"]}
    per_seed = []
    for path in paths:
        with np.load(path, allow_pickle=True) as trace:
            row = json.loads(trace["summary"].item())
            arrays = {name: 1000*np.asarray(trace[key], dtype=float)[DISCARD:] for name, key in STAGES.items()}
            total = 1000*np.asarray(trace["control_compute_time_s"], dtype=float)[DISCARD:]
            if not len(total) or not all(np.all(np.isfinite(v)) and np.all(v >= 0) for v in arrays.values()):
                raise ValueError(f"Invalid stage timing: {path}")
            other = total - sum(arrays.values())
            if np.min(other) < -1e-6:
                raise ValueError(f"Stage timers exceed their total: {path}")
            arrays["Other control-loop work"] = np.maximum(other, 0)
            arrays["Total online control"] = total
            for name, values in arrays.items():
                collected[name].append(values)
            per_seed.append({"seed": row["seed"], "one_time_goal_setup_ms": row["goal_setup_time_ms"],
                             "median_total_ms": float(np.median(total)),
                             "mean_total_ms": float(np.mean(total))})
    pooled = {name: np.concatenate(values) for name, values in collected.items()}
    mean_total = float(pooled["Total online control"].mean())
    modules = [{"module": name, "mean_ms": float(values.mean()),
                "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
                "p99_ms": float(np.percentile(values, 99)),
                "share_of_mean_total": float(values.mean()/mean_total)} for name, values in pooled.items()]
    setup = np.array([r["one_time_goal_setup_ms"] for r in per_seed])
    return {"modules": modules, "per_seed": per_seed,
            "timed_steps": len(pooled["Total online control"]),
            "discard_initial_steps_per_rollout": DISCARD,
            "median_of_rollout_medians_ms": float(np.median([r["median_total_ms"] for r in per_seed])),
            "one_time_goal_setup_ms": {"mean": float(setup.mean()), "median": float(np.median(setup)),
                                        "p95": float(np.percentile(setup, 95))}}


def run_panel(seeds, output_dir=None, config=None):
    seeds = list(map(int, seeds))
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and unique")
    config = baseline.default_config() if config is None else config
    output_dir = baseline.RESULT_ROOT / "evaluation_1000" if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = {
        "baseline_version": baseline.BASELINE_VERSION, "matrix_geometry": "linear", "candidate_count": 10,
        "config": {k: v for k, v in asdict(config).items() if k != "seed"},
        "data_sha256": file_digest(baseline.DATA_PATH), "atlas_sha256": file_digest(baseline.ATLAS_PATH),
        "source_sha256": {name: file_digest(baseline.ROOT / "src" / name) for name in (
            "baseline_atlas_deepc.py", "routed_deepc_experiment.py", "candidate_goal_routing_experiment.py",
            "expert_pipeline_search.py", "stitched_atlas_mpc_experiment.py")},
        "timing_protocol": "single_process_single_numerical_thread_warmup10_discard5",
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != signature:
            raise ValueError("Incompatible benchmark cache; choose a new output directory")
    elif any(output_dir.glob("seed*_control_results.npz")):
        raise ValueError("Unversioned rollouts cannot be reused for this benchmark")
    manifest_path.write_text(json.dumps(signature, indent=2))
    session = {"started_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
               "workers": 1, "numerical_threads": 1, "load_average_at_start": os.getloadavg(),
               "timing_caveat": "Local wall-clock measurements on a shared computer, not isolated from other jobs."}
    with threadpool_limits(limits=1):
        started = time.perf_counter()
        H = np.load(baseline.DATA_PATH, mmap_mode="r")
        data_load = time.perf_counter()-started
        started = time.perf_counter()
        atlas = routed.load_atlas(baseline.ATLAS_PATH)
        atlas_load = time.perf_counter()-started
        started = time.perf_counter()
        offline_environment_targets(H.shape[1] // routed.WINDOWS_PER_EPISODE)
        metadata_setup = time.perf_counter()-started
        started = time.perf_counter()
        controller = baseline.BaselineAtlasDeePC(H, atlas, config)
        chart_setup = time.perf_counter()-started
        session["offline_setup_seconds"] = {"data_loading": data_load, "atlas_loading": atlas_load,
            "source_episode_metadata": metadata_setup, "linear_chart_cache_and_all_pairs_routes": chart_setup}
        warmup = baseline.BaselineAtlasDeePC(H, atlas, replace(config, steps=10))
        warmup.run(210)
        session_path = output_dir / f"timing_session_{time.time_ns()}.json"
        session_path.write_text(json.dumps(session, indent=2))
        rows, paths = [], []
        print(f"Warm-up complete; evaluating {len(seeds)} baseline seeds only.", flush=True)
        for i, seed in enumerate(seeds):
            path = output_dir / f"seed{seed}_control_results.npz"
            if path.exists():
                with np.load(path, allow_pickle=True) as trace:
                    row = json.loads(trace["summary"].item())
                assert row["baseline_version"] == baseline.BASELINE_VERSION and row["seed"] == seed
            else:
                gc.collect()
                row = controller.run(seed, path)
            row["result_path"] = str(path.resolve())
            rows.append(row)
            paths.append(path)
            if (i+1) % 10 == 0 or i+1 == len(seeds):
                count = sum(r["success_30"] and r["success_100"] for r in rows)
                print(f"{i+1}/{len(seeds)} complete; successes={count}; latest seed={seed}", flush=True)
        timing = summarize_timing(paths)
        timing["sessions"] = [json.loads(p.read_text()) for p in sorted(output_dir.glob("timing_session_*.json"))]
        (output_dir / "timing_summary.json").write_text(json.dumps(timing, indent=2))
        payload = {"method": baseline.BASELINE_VERSION, "signature": signature,
                   "config": asdict(config), "results": rows,
                   "fairness_audit": routed.audit_iid_library(H, eval_seeds=seeds),
                   "timing_summary_path": str(output_dir / "timing_summary.json")}
        (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps(timing["modules"], indent=2), flush=True)
        return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=211)
    parser.add_argument("--seed-stop", type=int, default=1210)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_panel(range(args.seed_start, args.seed_stop+1), args.output_dir)
