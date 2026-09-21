"""Controlled serial latency comparison for contextual versus clustered context."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

import contextual_cluster_atlas_experiment as clustered
import routed_deepc_experiment as routed
from contextual_chart_expansion_experiment import baseline_config
from online_selected_deepc_comparison import run_online_selected_deepc


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/select_iid_offline_hankel.npy"
BASELINE_DIR = ROOT / "experiments/routed_deepc_iid/clustering_jointconfig48_fresh211_310"
CLUSTER_DIR = ROOT / "experiments/routed_deepc_iid/contextual_cluster_eval211_710_global128_triggered"
OUTPUT_PATH = CLUSTER_DIR / "controlled_serial_timing.json"
ROLLOUT_DIR = CLUSTER_DIR / "controlled_serial_rollouts"
SEEDS = (232, 277, 299)


def stage(values, mask=None):
    values = 1000 * np.asarray(values, dtype=float)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    return values


def summarize(name, control, selection, svd, qp, steps):
    return {
        "method": name,
        "control steps": int(steps),
        "median control (ms)": float(np.median(control)),
        "p95 control (ms)": float(np.percentile(control, 95)),
        "median selection (ms)": float(np.median(selection)),
        "median SVD (ms)": float(np.median(svd)),
        "median QP (ms)": float(np.median(qp)),
    }


def main():
    ROLLOUT_DIR.mkdir(parents=True, exist_ok=True)
    H = np.load(DATA_PATH)
    baseline_payload = json.loads((BASELINE_DIR / "summary.json").read_text())
    control_template = baseline_config(baseline_payload)
    atlas = routed.load_atlas(BASELINE_DIR / "behavioral_atlas.npz")
    artifact = clustered.load_contextual_cluster_atlas(CLUSTER_DIR / "contextual_cluster_atlas.npz")
    cluster_config = clustered.ContextualClusterConfig(
        **json.loads(artifact["cluster_config"].item())
    )
    physical_cache = clustered.physical_matrix_cache_from_artifact(artifact)

    contextual_parts = {key: [] for key in ("control", "selection", "svd", "qp")}
    cluster_parts = {key: [] for key in ("control", "selection", "svd", "qp")}
    active_parts = {key: [] for key in ("control", "selection", "svd", "qp")}
    outcomes = []
    for seed in SEEDS:
        config = routed.RoutedDeePCConfig(**{**asdict(control_template), "seed": seed})
        contextual_path = ROLLOUT_DIR / f"contextual_seed{seed}.npz"
        cluster_path = ROLLOUT_DIR / f"cluster_seed{seed}.npz"
        contextual_summary = run_online_selected_deepc(
            H, atlas, config, selected_columns=200, selected_model="rank_matched",
            selection_method="contextual", reference_mode="graph_routed",
            save_path=contextual_path,
        )
        cluster_summary = clustered.run_contextual_cluster_atlas(
            H, atlas, artifact, config, cluster_config,
            save_path=cluster_path, physical_matrix_cache=physical_cache,
        )
        with np.load(contextual_path, allow_pickle=True) as data:
            contextual_parts["control"].append(stage(data["control_compute_time_s"]))
            contextual_parts["selection"].append(stage(data["selection_time_s"]))
            contextual_parts["svd"].append(stage(data["svd_rebuild_time_s"]))
            contextual_parts["qp"].append(stage(data["solve_time_s"]))
        with np.load(cluster_path, allow_pickle=True) as data:
            active = np.asarray(data["contextual_active"], dtype=bool)
            values = {
                "control": stage(data["control_compute_time_s"]),
                "selection": stage(data["cluster_scoring_time_s"]),
                "svd": np.zeros(active.size),
                "qp": stage(data["solve_time_s"]),
            }
            for key, array in values.items():
                cluster_parts[key].append(array)
                if np.any(active):
                    active_parts[key].append(array[active])
        outcomes.append({
            "seed": seed,
            "contextual strict": bool(contextual_summary["success_30"] and contextual_summary["success_100"]),
            "cluster strict": bool(cluster_summary["success_30"] and cluster_summary["success_100"]),
            "cluster triggered": bool(cluster_summary["contextual_triggered"]),
            "cluster usage fraction": float(cluster_summary["contextual_usage_fraction"]),
        })

    def joined(parts, key):
        return np.concatenate(parts[key]) if parts[key] else np.empty(0)

    methods = [
        summarize(
            "Contextual-aware DPC: all steps",
            *[joined(contextual_parts, key) for key in ("control", "selection", "svd", "qp")],
            steps=joined(contextual_parts, "control").size,
        ),
        summarize(
            "Contextual-cluster Atlas: all steps",
            *[joined(cluster_parts, key) for key in ("control", "selection", "svd", "qp")],
            steps=joined(cluster_parts, "control").size,
        ),
    ]
    if active_parts["control"]:
        methods.append(
            summarize(
                "Contextual-cluster Atlas: active steps",
                *[joined(active_parts, key) for key in ("control", "selection", "svd", "qp")],
                steps=joined(active_parts, "control").size,
            )
        )
    payload = {
        "seeds": list(SEEDS),
        "protocol": "serial methods on the same process, machine, seeds, and 320-step horizon",
        "methods": methods,
        "outcomes": outcomes,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
