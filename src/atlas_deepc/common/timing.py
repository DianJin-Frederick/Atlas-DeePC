"""Stage-timing summaries shared by the current Reacher evaluation runners."""

import json
import os

# Preserve the numerical-thread policy used by the existing serial runners.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

import numpy as np


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
