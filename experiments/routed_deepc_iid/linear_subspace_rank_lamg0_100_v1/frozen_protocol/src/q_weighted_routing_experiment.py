"""Paired ablation of baseline and Q-weighted graph routing.

The frozen 48-chart output-gated atlas is reused without refitting. For every
admissible directed edge, transition-supported offline columns are scored by
the normalized Q-weighted distance between their future-output block and the
destination chart's repeated medoid reference. This future cost augments only
the graph edge weight; localization, goal construction, chart matrices, DeePC
costs and constraints, and the rollout protocol remain unchanged.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

import expert_pipeline_search as pipe
import routed_deepc_experiment as routed
from local_affine_atlas_experiment import audit_iid_library
from online_selected_deepc_comparison import prepare_contextual_library


RESULT_ROOT = Path("experiments/routed_deepc_iid")
DEFAULT_DATA_PATH = Path("data/select_iid_offline_hankel.npy")
DEFAULT_BASELINE_DIR = RESULT_ROOT / "clustering_jointconfig48_fresh211_310"
DEFAULT_OUTPUT_NAME = "q_weighted_routing211_310"


def output_cost(config: routed.RoutedDeePCConfig) -> np.ndarray:
    return np.diag(
        [
            config.q_data_goal_joint,
            config.q_data_goal_joint,
            config.q_data_goal_joint,
            config.q_data_goal_joint,
            config.q_dq,
            config.q_dq,
            config.q_xy,
            config.q_xy,
            config.q_data_goal_joint,
            config.q_data_goal_joint,
        ]
    )


def transition_support_indices(atlas: dict, columns: int) -> dict[tuple[int, int], np.ndarray]:
    """Return offline columns followed by each observed one-step chart edge."""
    labels = np.asarray(atlas["labels"], dtype=int)
    if labels.size != columns:
        raise ValueError("Atlas labels and Hankel columns have different sizes.")
    support: dict[tuple[int, int], list[int]] = {}
    episodes = columns // routed.WINDOWS_PER_EPISODE
    if episodes * routed.WINDOWS_PER_EPISODE != columns:
        raise ValueError("Hankel columns do not contain complete offline episodes.")
    for episode in range(episodes):
        start = episode * routed.WINDOWS_PER_EPISODE
        stop = start + routed.WINDOWS_PER_EPISODE
        for index in range(start, stop - 1):
            edge = (int(labels[index]), int(labels[index + 1]))
            if edge[0] != edge[1]:
                support.setdefault(edge, []).append(index)
    return {
        edge: np.asarray(indices, dtype=int)
        for edge, indices in support.items()
    }


def build_q_weighted_routing_atlas(
    H: np.ndarray,
    atlas: dict,
    config: routed.RoutedDeePCConfig,
    top_m: int = 20,
    beta: float = 1.0,
) -> tuple[dict, dict]:
    """Attach nonnegative Q-weighted edge costs to a copy of the frozen atlas."""
    if top_m <= 0:
        raise ValueError("top_m must be positive.")
    if beta < 0:
        raise ValueError("beta must be nonnegative.")

    counts = np.asarray(atlas["transition_counts"], dtype=int)
    centers = np.asarray(atlas["center_distance"], dtype=float)
    support = transition_support_indices(atlas, H.shape[1])
    library = prepare_contextual_library(H)
    Y_f_normalized = library["Y_f_normalized"]
    output_mean = library["output_mean"]
    output_scale = library["output_scale"]
    future_weights = np.tile(np.diag(output_cost(config)), pipe.T_F).reshape(-1, 1)

    edge_rows = []
    for source in range(config.clusters):
        destinations = np.where(counts[source] >= config.edge_count_threshold)[0]
        for destination_value in destinations:
            destination = int(destination_value)
            if destination == source:
                continue
            indices = support.get((source, destination))
            if indices is None or indices.size != counts[source, destination]:
                raise RuntimeError(
                    f"Transition support mismatch for edge {source}->{destination}."
                )
            waypoint = np.asarray(atlas["medoid_states"][destination], dtype=float)
            reference_normalized = (
                (np.tile(waypoint, (pipe.T_F, 1)) - output_mean[None, :])
                / output_scale[None, :]
            ).reshape(pipe.P * pipe.T_F, 1)
            delta = Y_f_normalized[:, indices] - reference_normalized
            distances = np.sqrt(np.sum(future_weights * delta * delta, axis=0))
            retained = min(top_m, distances.size)
            best = np.partition(distances, retained - 1)[:retained]
            base_cost = centers[source, destination] / np.sqrt(
                float(counts[source, destination])
            )
            edge_rows.append(
                {
                    "source": source,
                    "destination": destination,
                    "transition_count": int(counts[source, destination]),
                    "retained_columns": int(retained),
                    "base_cost": float(base_cost),
                    "future_q_cost": float(np.mean(best)),
                }
            )

    if not edge_rows:
        raise RuntimeError("No admissible non-self graph edges were found.")
    base_scale = max(float(np.median([row["base_cost"] for row in edge_rows])), 1e-12)
    future_scale = max(
        float(np.median([row["future_q_cost"] for row in edge_rows])), 1e-12
    )
    routing_costs = np.full(counts.shape, np.inf, dtype=float)
    for row in edge_rows:
        normalized_base = row["base_cost"] / base_scale
        normalized_future = row["future_q_cost"] / future_scale
        cost = normalized_base + beta * normalized_future
        routing_costs[row["source"], row["destination"]] = cost
        row["normalized_base_cost"] = float(normalized_base)
        row["normalized_future_q_cost"] = float(normalized_future)
        row["q_weighted_edge_cost"] = float(cost)

    q_atlas = dict(atlas)
    q_atlas["routing_edge_costs"] = routing_costs
    metadata = {
        "routing_version": 1,
        "routing_method": "q_weighted_future_output_edges",
        "edge_formula": (
            "base/median(base) + beta * top-m-mean(||Y_f-r_destination||_Q)"
            "/median(future_cost)"
        ),
        "transition_support": "observed consecutive offline chart labels",
        "future_reference": "destination chart medoid repeated over T_f",
        "output_normalization": "per output channel over the offline library",
        "top_m": int(top_m),
        "beta": float(beta),
        "base_cost_scale": base_scale,
        "future_q_cost_scale": future_scale,
        "admissible_directed_edges": len(edge_rows),
        "edge_table": edge_rows,
    }
    return q_atlas, metadata


def load_saved_summary(path: Path) -> dict:
    result = np.load(path, allow_pickle=True)
    return json.loads(result["summary"].item())


def baseline_config(baseline_payload: dict) -> routed.RoutedDeePCConfig:
    allowed = {field.name for field in fields(routed.RoutedDeePCConfig)}
    values = {
        key: value
        for key, value in baseline_payload["config"].items()
        if key in allowed
    }
    return routed.RoutedDeePCConfig(**values)


def run_q_weighted_panel(
    seeds: list[int],
    output_name: str = DEFAULT_OUTPUT_NAME,
    data_path: Path = DEFAULT_DATA_PATH,
    baseline_dir: Path = DEFAULT_BASELINE_DIR,
    top_m: int = 20,
    beta: float = 1.0,
    overwrite: bool = False,
) -> dict:
    """Run or resume the Q-weighted routing panel one seed at a time."""
    if not seeds:
        raise ValueError("At least one seed is required.")
    baseline_payload = json.loads((baseline_dir / "summary.json").read_text())
    config = baseline_config(baseline_payload)
    if (
        config.clusters != 48
        or config.clustering_input_weight != 0.0
        or config.clustering_feature_mode != "plant_joint_configuration"
        or config.formulation != "svd_current"
        or config.svd_rank != 60
    ):
        raise ValueError("The baseline is not the frozen 48-chart output-only atlas.")
    H = np.load(data_path)
    atlas = routed.load_atlas(baseline_dir / "behavioral_atlas.npz")
    q_atlas, routing_metadata = build_q_weighted_routing_atlas(
        H, atlas, config, top_m=top_m, beta=beta
    )

    output_dir = RESULT_ROOT / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "q_weighted_behavioral_atlas.npz", **q_atlas)
    (output_dir / "routing_metadata.json").write_text(
        json.dumps(routing_metadata, indent=2)
    )
    audit = audit_iid_library(H, eval_seeds=list(seeds))

    results = []
    for seed in seeds:
        result_path = output_dir / f"seed{seed}_control_results.npz"
        if result_path.exists() and not overwrite:
            summary = load_saved_summary(result_path)
            compatible = (
                summary.get("routing_method") == routing_metadata["routing_method"]
                and summary.get("routing_version") == routing_metadata["routing_version"]
                and summary.get("routing_top_m") == int(top_m)
                and np.isclose(summary.get("routing_beta", np.nan), beta)
            )
            if not compatible:
                raise RuntimeError(
                    f"Existing result {result_path} uses different routing settings; "
                    "choose a new output name or set overwrite=True."
                )
            print("loaded q-weighted routing seed", seed, flush=True)
        else:
            run_config = routed.RoutedDeePCConfig(
                **{**asdict(config), "seed": int(seed)}
            )
            summary = routed.run_routed_deepc(
                H,
                q_atlas,
                run_config,
                save_path=result_path,
                summary_metadata={
                    "routing_version": routing_metadata["routing_version"],
                    "routing_method": routing_metadata["routing_method"],
                    "routing_top_m": int(top_m),
                    "routing_beta": float(beta),
                    "routing_base_cost_scale": routing_metadata["base_cost_scale"],
                    "routing_future_q_cost_scale": routing_metadata[
                        "future_q_cost_scale"
                    ],
                },
            )
            print(
                "q-weighted routing seed", seed,
                "mean30_mm", f"{1000 * summary['mean_last_30']:.3f}",
                "success", summary["success_30"] and summary["success_100"],
                flush=True,
            )
        results.append(summary)
        payload = {
            "method": "q_weighted_graph_routed_fixed_chart_deepc",
            "baseline_source": str(baseline_dir),
            "config": asdict(config),
            "routing": routing_metadata,
            "fairness_audit": audit,
            "results": results,
        }
        (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    return payload
