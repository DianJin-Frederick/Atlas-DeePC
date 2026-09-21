"""Controlled serial timing for contextual-cluster Atlas basis ranks."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

import contextual_cluster_atlas_experiment as experiment
import routed_deepc_experiment as routed
from contextual_chart_expansion_experiment import baseline_config


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/select_iid_offline_hankel.npy"
BASELINE_DIR = ROOT / "experiments/routed_deepc_iid/clustering_jointconfig48_fresh211_310"
RESULT_ROOT = ROOT / "experiments/routed_deepc_iid"
RANK_DIRS = {
    60: RESULT_ROOT / "contextual_cluster_eval211_710_global128_triggered",
    47: RESULT_ROOT / "contextual_cluster_rank_eval47_211_710",
    45: RESULT_ROOT / "contextual_cluster_rank_eval45_211_710",
    40: RESULT_ROOT / "contextual_cluster_rank_eval40_211_710",
}
OUTPUT_PATH = RANK_DIRS[47] / "controlled_serial_rank_timing.json"
ROLLOUT_DIR = RANK_DIRS[47] / "controlled_serial_rank_rollouts"
SEEDS = (232, 277, 299)


def milliseconds(values, mask=None):
    values = 1000 * np.asarray(values, dtype=float)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    return values


def summarize(rank, scope, parts):
    joined = {
        key: np.concatenate(values) if values else np.empty(0)
        for key, values in parts.items()
    }
    return {
        "rank": rank,
        "scope": scope,
        "control steps": int(joined["control"].size),
        "median control (ms)": float(np.median(joined["control"])),
        "p95 control (ms)": float(np.percentile(joined["control"], 95)),
        "median prototype score (ms)": float(np.median(joined["selection"])),
        "median QP (ms)": float(np.median(joined["qp"])),
        "p95 QP (ms)": float(np.percentile(joined["qp"], 95)),
    }


def main():
    ROLLOUT_DIR.mkdir(parents=True, exist_ok=True)
    H = np.load(DATA_PATH)
    baseline_payload = json.loads((BASELINE_DIR / "summary.json").read_text())
    template = baseline_config(baseline_payload)
    atlas = routed.load_atlas(BASELINE_DIR / "behavioral_atlas.npz")
    artifacts = {
        rank: experiment.load_contextual_cluster_atlas(
            directory / "contextual_cluster_atlas.npz"
        )
        for rank, directory in RANK_DIRS.items()
    }
    cluster_configs = {
        rank: experiment.ContextualClusterConfig(
            **json.loads(artifact["cluster_config"].item())
        )
        for rank, artifact in artifacts.items()
    }
    physical_caches = {
        rank: experiment.physical_matrix_cache_from_artifact(artifact)
        for rank, artifact in artifacts.items()
    }
    all_parts = {
        rank: {key: [] for key in ("control", "selection", "qp")}
        for rank in RANK_DIRS
    }
    active_parts = {
        rank: {key: [] for key in ("control", "selection", "qp")}
        for rank in RANK_DIRS
    }
    outcomes = []
    ranks = list(RANK_DIRS)
    for seed_index, seed in enumerate(SEEDS):
        order = ranks[seed_index:] + ranks[:seed_index]
        for rank in order:
            config = routed.RoutedDeePCConfig(
                **{**asdict(template), "seed": seed, "svd_rank": rank}
            )
            path = ROLLOUT_DIR / f"rank{rank}_seed{seed}.npz"
            summary = experiment.run_contextual_cluster_atlas(
                H, atlas, artifacts[rank], config, cluster_configs[rank],
                save_path=path, physical_matrix_cache=physical_caches[rank],
            )
            with np.load(path, allow_pickle=True) as data:
                active = np.asarray(data["contextual_active"], dtype=bool)
                values = {
                    "control": milliseconds(data["control_compute_time_s"]),
                    "selection": milliseconds(data["cluster_scoring_time_s"]),
                    "qp": milliseconds(data["solve_time_s"]),
                }
                for key, value in values.items():
                    all_parts[rank][key].append(value)
                    if np.any(active):
                        active_parts[rank][key].append(value[active])
            outcomes.append({
                "seed": seed,
                "rank": rank,
                "strict success": bool(summary["success_30"] and summary["success_100"]),
                "triggered": bool(summary["contextual_triggered"]),
                "contextual usage fraction": float(summary["contextual_usage_fraction"]),
            })

    methods = []
    for rank in ranks:
        methods.append(summarize(rank, "all steps", all_parts[rank]))
        methods.append(summarize(rank, "context-active steps", active_parts[rank]))
    payload = {
        "seeds": list(SEEDS),
        "rank order rotated by seed": True,
        "protocol": "serial ranks in one process on identical seeds and 320-step horizon",
        "methods": methods,
        "outcomes": outcomes,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
