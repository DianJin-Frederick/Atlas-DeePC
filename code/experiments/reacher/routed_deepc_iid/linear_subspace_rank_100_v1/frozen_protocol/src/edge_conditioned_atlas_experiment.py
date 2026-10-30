"""Held-out ablation for graph-edge-conditioned Atlas-DeePC bases.

For each directed graph edge k -> j, cache a rank-60 affine basis from the top
source-chart Hankel columns whose normalized future outputs best match chart
j's medoid waypoint.  Online routing, reference generation, DeePC costs,
constraints, horizons, and solver are otherwise identical to the frozen
48-chart joint-configuration Atlas baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import cvxpy as cp
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import expert_pipeline_search as pipe
from online_selected_deepc_comparison import (
    build_controller,
    prepare_contextual_library,
    selected_affine_basis,
    timing_summary,
)
from routed_deepc_experiment import (
    DEFAULT_DATA_PATH,
    RoutedDeePCConfig,
    assign_atlas_charts,
    chart_matrix,
    load_atlas,
    online_behavioral_feature,
    shortest_chart_path,
)
from stitched_atlas_mpc_experiment import chart_compatible_goal


DEFAULT_ATLAS_PATH = Path(
    "experiments/routed_deepc_iid/"
    "clustering_jointconfig48_benchmark111_210/behavioral_atlas.npz"
)
DEFAULT_BASELINE_SUMMARY = Path(
    "experiments/routed_deepc_iid/"
    "clustering_jointconfig48_benchmark111_210/summary.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/edge_conditioned_atlas/heldout111_210"
)


def output_cost(config: RoutedDeePCConfig) -> np.ndarray:
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


def prepare_model_caches(
    H: np.ndarray,
    atlas: dict,
    config: RoutedDeePCConfig,
    edge_columns: int,
) -> tuple[dict, dict, dict, dict]:
    """Build every current-chart and admissible directed-edge basis offline."""
    start = time.perf_counter()
    current_models: dict[int, tuple[dict, np.ndarray]] = {}
    for chart_id in range(config.clusters):
        current_models[chart_id] = chart_matrix(H, atlas, (chart_id,), config)

    library = prepare_contextual_library(H)
    Y_f_normalized = library["Y_f_normalized"]
    output_mean = library["output_mean"]
    output_scale = library["output_scale"]
    weights = np.tile(np.diag(output_cost(config)), pipe.T_F).reshape(-1, 1)
    labels = np.asarray(atlas["labels"], dtype=int)
    counts = np.asarray(atlas["transition_counts"], dtype=int)
    edge_models: dict[tuple[int, int], tuple[dict, np.ndarray]] = {}
    selected_sizes = []

    for source in range(config.clusters):
        candidates = np.where(labels == source)[0]
        destinations = np.where(counts[source] >= config.edge_count_threshold)[0]
        for destination_value in destinations:
            destination = int(destination_value)
            if destination == source:
                continue
            waypoint = np.asarray(atlas["medoid_states"][destination], dtype=float)
            reference_normalized = (
                (np.tile(waypoint, (pipe.T_F, 1)) - output_mean[None, :])
                / output_scale[None, :]
            ).reshape(pipe.P * pipe.T_F, 1)
            delta = Y_f_normalized[:, candidates] - reference_normalized
            future_distance = np.sqrt(np.sum(weights * delta**2, axis=0))
            order = np.argsort(future_distance, kind="stable")
            selected = candidates[order[: min(edge_columns, candidates.size)]]
            if selected.size <= config.svd_rank:
                raise RuntimeError(
                    f"edge {source}->{destination} has only {selected.size} selected columns"
                )
            edge_models[(source, destination)] = (
                selected_affine_basis(H, selected, config),
                selected,
            )
            selected_sizes.append(int(selected.size))

    elapsed = time.perf_counter() - start
    metadata = {
        "offline_cache_build_time_s": float(elapsed),
        "current_chart_model_count": len(current_models),
        "directed_edge_model_count": len(edge_models),
        "edge_columns_requested": int(edge_columns),
        "edge_selected_columns_min": int(np.min(selected_sizes)),
        "edge_selected_columns_median": float(np.median(selected_sizes)),
        "edge_selected_columns_max": int(np.max(selected_sizes)),
        "edge_score": (
            "Q-weighted normalized future-output distance to the destination-chart "
            "medoid waypoint, evaluated only within the source chart"
        ),
    }
    reference_context = {
        "Y_f_normalized": Y_f_normalized,
        "output_mean": output_mean,
        "output_scale": output_scale,
        "future_weights": weights,
        "labels": labels,
        "edge_columns": int(edge_columns),
    }
    return current_models, edge_models, reference_context, metadata


def reference_conditioned_model(
    H: np.ndarray,
    reference_context: dict,
    chart_id: int,
    reference: np.ndarray,
    config: RoutedDeePCConfig,
) -> tuple[dict, np.ndarray]:
    """Build one terminal-chart basis for the graph-generated final goal."""
    candidates = np.where(reference_context["labels"] == chart_id)[0]
    reference_normalized = (
        (
            np.tile(np.asarray(reference, dtype=float), (pipe.T_F, 1))
            - reference_context["output_mean"][None, :]
        )
        / reference_context["output_scale"][None, :]
    ).reshape(pipe.P * pipe.T_F, 1)
    delta = reference_context["Y_f_normalized"][:, candidates] - reference_normalized
    distance = np.sqrt(
        np.sum(reference_context["future_weights"] * delta**2, axis=0)
    )
    selected = candidates[
        np.argsort(distance, kind="stable")[: reference_context["edge_columns"]]
    ]
    return selected_affine_basis(H, selected, config), selected


def run_edge_conditioned_atlas(
    H: np.ndarray,
    atlas: dict,
    config: RoutedDeePCConfig,
    current_models: dict[int, tuple[dict, np.ndarray]],
    edge_models: dict[tuple[int, int], tuple[dict, np.ndarray]],
    reference_context: dict,
    cache_metadata: dict,
    save_path: Path | None = None,
) -> dict:
    controller = build_controller(config)
    env = gym.make(
        pipe.REACHER_ENV_ID,
        max_episode_steps=config.steps + pipe.T_INI + 5,
    )
    obs, _ = env.reset(seed=config.seed)
    target_xy = np.asarray(obs[4:6], dtype=float).copy()
    initial_y = pipe.reacher_output_from_obs(obs)
    goal, goal_metadata = chart_compatible_goal(H, target_xy, initial_y, config)
    goal_index = int(goal_metadata["data_goal_index"])
    goal_chart = int(atlas["labels"][goal_index])
    terminal_model_start = time.perf_counter()
    terminal_model = reference_conditioned_model(
        H,
        reference_context,
        goal_chart,
        goal,
        config,
    )
    terminal_model_setup_time = time.perf_counter() - terminal_model_start

    u_history: list[np.ndarray] = []
    y_history: list[np.ndarray] = []
    for _ in range(pipe.T_INI):
        action = np.zeros(pipe.M)
        obs, *_ = env.step(action)
        u_history.append(action.copy())
        y_history.append(pipe.reacher_output_from_obs(obs))

    target_reference_latched = False
    records: dict[str, list] = {
        key: []
        for key in (
            "y",
            "u",
            "pred",
            "ref",
            "distance",
            "prediction_error_xy",
            "status",
            "chart",
            "goal_chart",
            "next_chart",
            "path_length",
            "edge_model_used",
            "terminal_model_used",
            "model_columns",
            "localization_time_s",
            "routing_time_s",
            "matrix_preparation_time_s",
            "solve_time_s",
            "control_compute_time_s",
        )
    }

    for _ in range(config.steps):
        control_start = time.perf_counter()
        u_ini = np.asarray(u_history[-pipe.T_INI :]).reshape(
            pipe.M * pipe.T_INI, 1
        )
        y_ini = np.asarray(y_history[-pipe.T_INI :]).reshape(
            pipe.P * pipe.T_INI, 1
        )

        stage = time.perf_counter()
        feature = online_behavioral_feature(
            u_ini,
            y_ini,
            atlas["feature_mean"],
            atlas["feature_scale"],
            config.clustering_input_weight,
            config.clustering_feature_mode,
            config.clustering_configuration_weight,
            config.clustering_velocity_weight,
            config.clustering_fingertip_weight,
        )
        current_chart = int(assign_atlas_charts(atlas, feature)[0])
        localization_time = time.perf_counter() - stage

        stage = time.perf_counter()
        path = shortest_chart_path(
            atlas,
            current_chart,
            goal_chart,
            config.edge_count_threshold,
        ) or [current_chart]
        next_chart = path[1] if len(path) > 1 else current_chart
        distance_before = float(
            np.linalg.norm(y_history[-1][6:8] - target_xy)
        )
        if distance_before < config.target_switch_distance:
            target_reference_latched = True
        use_target = (
            target_reference_latched
            if config.latch_target_reference
            else distance_before < config.target_switch_distance
        )
        active_reference = (
            goal
            if use_target or current_chart == goal_chart
            else atlas["medoid_states"][next_chart]
        )
        reference = np.tile(active_reference.reshape(pipe.P, 1), (pipe.T_F, 1))
        routing_time = time.perf_counter() - stage

        stage = time.perf_counter()
        edge_key = (current_chart, next_chart)
        use_edge_model = (
            current_chart != next_chart
            and edge_key in edge_models
        )
        use_terminal_model = use_target and current_chart == goal_chart
        if use_terminal_model:
            matrix, selected = terminal_model
        elif use_edge_model:
            matrix, selected = edge_models[edge_key]
        else:
            matrix, selected = current_models[current_chart]
        matrix_time = time.perf_counter() - stage

        stage = time.perf_counter()
        u_solution, y_solution, _g, status, _ = controller.solve(
            matrix,
            u_ini,
            y_ini,
            reference,
        )
        solve_time = time.perf_counter() - stage
        if u_solution is None:
            action = np.zeros(pipe.M)
            prediction = np.full(pipe.P, np.nan)
        else:
            action = np.clip(
                u_solution[: pipe.M, 0],
                -config.u_limit,
                config.u_limit,
            )
            prediction = np.asarray(y_solution[: pipe.P, 0]).reshape(pipe.P)
        compute_time = time.perf_counter() - control_start

        obs, *_ = env.step(action)
        y_now = pipe.reacher_output_from_obs(obs)
        u_history.append(action.copy())
        y_history.append(y_now.copy())
        records["y"].append(y_now)
        records["u"].append(action)
        records["pred"].append(prediction)
        records["ref"].append(active_reference)
        records["distance"].append(np.linalg.norm(y_now[6:8] - target_xy))
        records["prediction_error_xy"].append(
            np.linalg.norm(prediction[6:8] - y_now[6:8])
        )
        records["status"].append(str(status))
        records["chart"].append(current_chart)
        records["goal_chart"].append(goal_chart)
        records["next_chart"].append(next_chart)
        records["path_length"].append(len(path))
        records["edge_model_used"].append(use_edge_model)
        records["terminal_model_used"].append(use_terminal_model)
        records["model_columns"].append(selected.size)
        records["localization_time_s"].append(localization_time)
        records["routing_time_s"].append(routing_time)
        records["matrix_preparation_time_s"].append(matrix_time)
        records["solve_time_s"].append(solve_time)
        records["control_compute_time_s"].append(compute_time)
    env.close()

    arrays = {
        key: np.asarray(value, dtype=object if key == "status" else None)
        for key, value in records.items()
    }
    distance = arrays["distance"]
    tail_30 = distance[-30:]
    tail_100 = distance[-min(100, distance.size) :]
    summary = {
        **asdict(config),
        "method": "graph_routed_edge_conditioned_atlas_deepc",
        "minimum": float(distance.min()),
        "minimum_step": int(distance.argmin()),
        "last": float(distance[-1]),
        "mean_last_30": float(tail_30.mean()),
        "max_last_30": float(tail_30.max()),
        "mean_last_100": float(tail_100.mean()),
        "max_last_100": float(tail_100.max()),
        "success_30": bool(tail_30.mean() < 0.01 and tail_30.max() < 0.02),
        "success_100": bool(
            tail_100.mean() < 0.01 and tail_100.max() < 0.02
        ),
        "mean_prediction_error_xy": float(
            np.nanmean(arrays["prediction_error_xy"])
        ),
        "edge_model_usage_fraction": float(
            np.mean(arrays["edge_model_used"])
        ),
        "terminal_model_usage_fraction": float(
            np.mean(arrays["terminal_model_used"])
        ),
        "terminal_model_setup_time_ms": float(1000.0 * terminal_model_setup_time),
        "median_model_columns": float(np.median(arrays["model_columns"])),
        "statuses": dict(Counter(arrays["status"].tolist())),
        "charts_visited": int(np.unique(arrays["chart"]).size),
        "qp_count": config.steps,
        "online_full_library_selection": False,
        "cached_edge_models": True,
        **timing_summary(arrays["localization_time_s"], "localization"),
        **timing_summary(arrays["routing_time_s"], "routing"),
        **timing_summary(
            arrays["matrix_preparation_time_s"], "matrix_preparation"
        ),
        **timing_summary(arrays["solve_time_s"], "solve"),
        **timing_summary(arrays["control_compute_time_s"], "control_compute"),
        **cache_metadata,
        **goal_metadata,
    }
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path,
            **arrays,
            summary=np.array(json.dumps(summary), dtype=object),
        )
    return summary


_WORKER_H = None
_WORKER_ATLAS = None
_WORKER_CONFIG = None
_WORKER_CURRENT_MODELS = None
_WORKER_EDGE_MODELS = None
_WORKER_REFERENCE_CONTEXT = None
_WORKER_CACHE_METADATA = None
_WORKER_OUTPUT_DIR = None


def initialize_worker(
    data_path: str,
    atlas_path: str,
    config_payload: dict,
    edge_columns: int,
    output_dir: str,
) -> None:
    global _WORKER_H, _WORKER_ATLAS, _WORKER_CONFIG
    global _WORKER_CURRENT_MODELS, _WORKER_EDGE_MODELS, _WORKER_REFERENCE_CONTEXT
    global _WORKER_CACHE_METADATA, _WORKER_OUTPUT_DIR
    _WORKER_H = np.load(data_path)
    _WORKER_ATLAS = load_atlas(Path(atlas_path))
    _WORKER_CONFIG = RoutedDeePCConfig(**config_payload)
    (
        _WORKER_CURRENT_MODELS,
        _WORKER_EDGE_MODELS,
        _WORKER_REFERENCE_CONTEXT,
        _WORKER_CACHE_METADATA,
    ) = prepare_model_caches(
        _WORKER_H,
        _WORKER_ATLAS,
        _WORKER_CONFIG,
        edge_columns,
    )
    _WORKER_OUTPUT_DIR = Path(output_dir)


def run_worker(seed: int) -> dict:
    config = replace(_WORKER_CONFIG, seed=int(seed))
    save_path = _WORKER_OUTPUT_DIR / f"seed{seed}_control_results.npz"
    return run_edge_conditioned_atlas(
        _WORKER_H,
        _WORKER_ATLAS,
        config,
        _WORKER_CURRENT_MODELS,
        _WORKER_EDGE_MODELS,
        _WORKER_REFERENCE_CONTEXT,
        _WORKER_CACHE_METADATA,
        save_path=save_path,
    )


def load_saved_summary(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return json.loads(str(data["summary"].item()))


def exact_mcnemar_p(repaired: int, regressed: int) -> float:
    discordant = repaired + regressed
    if discordant == 0:
        return 1.0
    lower = min(repaired, regressed)
    probability = sum(
        math.comb(discordant, k) for k in range(lower + 1)
    ) / (2**discordant)
    return float(min(1.0, 2.0 * probability))


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [float(center - radius), float(center + radius)]


def make_figure(
    output_dir: Path,
    seeds: list[int],
    baseline_by_seed: dict[int, dict],
    edge_by_seed: dict[int, dict],
) -> None:
    baseline_tail = np.array([baseline_by_seed[s]["mean_last_100"] for s in seeds])
    edge_tail = np.array([edge_by_seed[s]["mean_last_100"] for s in seeds])
    baseline_prediction = np.array(
        [baseline_by_seed[s]["mean_prediction_error_xy"] for s in seeds]
    )
    edge_prediction = np.array(
        [edge_by_seed[s]["mean_prediction_error_xy"] for s in seeds]
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.7))
    axes[0].scatter(
        1000.0 * baseline_tail,
        1000.0 * edge_tail,
        s=18,
        alpha=0.75,
    )
    limit = 1000.0 * max(np.max(baseline_tail), np.max(edge_tail))
    axes[0].plot([0, limit], [0, limit], "k--", linewidth=1)
    axes[0].set(
        title="Paired final-100 error",
        xlabel="Current-chart Atlas [mm]",
        ylabel="Edge-conditioned Atlas [mm]",
    )
    axes[1].scatter(
        1000.0 * baseline_prediction,
        1000.0 * edge_prediction,
        s=18,
        alpha=0.75,
        color="tab:green",
    )
    limit = 1000.0 * max(np.max(baseline_prediction), np.max(edge_prediction))
    axes[1].plot([0, limit], [0, limit], "k--", linewidth=1)
    axes[1].set(
        title="Paired prediction error",
        xlabel="Current-chart Atlas [mm]",
        ylabel="Edge-conditioned Atlas [mm]",
    )
    success_counts = [
        sum(baseline_by_seed[s]["success_30"] for s in seeds),
        sum(edge_by_seed[s]["success_30"] for s in seeds),
    ]
    axes[2].bar(
        ["Current chart", "Edge conditioned"],
        success_counts,
        color=["tab:blue", "tab:green"],
    )
    axes[2].set(title="Strict successes", ylabel=f"Seeds out of {len(seeds)}", ylim=(0, len(seeds)))
    for index, value in enumerate(success_counts):
        axes[2].text(index, value + 0.5, str(value), ha="center")
    for axis in axes:
        axis.grid(alpha=0.22)
    fig.suptitle("Held-out seeds 111–210: edge-conditioned Atlas ablation")
    fig.tight_layout()
    fig.savefig(output_dir / "edge_conditioned_ablation.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / "edge_conditioned_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_stop + 1))
    config = RoutedDeePCConfig(
        clusters=48,
        clustering_input_weight=0.0,
        clustering_method="minibatch_kmeans",
        clustering_feature_mode="plant_joint_configuration",
        formulation="svd_current",
        svd_rank=args.rank,
        steps=args.steps,
        seed=seeds[0],
    )

    results_by_seed: dict[int, dict] = {}
    pending = []
    for seed in seeds:
        result_path = args.output_dir / f"seed{seed}_control_results.npz"
        if args.resume and result_path.exists():
            try:
                results_by_seed[seed] = load_saved_summary(result_path)
                continue
            except Exception:
                pass
        pending.append(seed)

    if pending:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize_worker,
            initargs=(
                str(args.data_path),
                str(args.atlas_path),
                asdict(config),
                args.edge_columns,
                str(args.output_dir),
            ),
        ) as executor:
            futures = {executor.submit(run_worker, seed): seed for seed in pending}
            completed = len(seeds) - len(pending)
            for future in as_completed(futures):
                seed = futures[future]
                results_by_seed[seed] = future.result()
                completed += 1
                result = results_by_seed[seed]
                print(
                    f"completed {completed}/{len(seeds)} seed={seed} "
                    f"success={result['success_30'] and result['success_100']} "
                    f"tail={1000 * result['mean_last_100']:.2f} mm",
                    flush=True,
                )

    baseline = json.loads(args.baseline_summary.read_text())
    baseline_by_seed = {
        int(result["seed"]): result for result in baseline["results"]
    }
    missing = sorted(set(seeds) - set(baseline_by_seed))
    if missing:
        raise RuntimeError(f"baseline summary is missing seeds {missing}")
    edge_by_seed = {seed: results_by_seed[seed] for seed in seeds}

    baseline_success = np.array(
        [
            baseline_by_seed[s]["success_30"]
            and baseline_by_seed[s]["success_100"]
            for s in seeds
        ],
        dtype=bool,
    )
    edge_success = np.array(
        [
            edge_by_seed[s]["success_30"]
            and edge_by_seed[s]["success_100"]
            for s in seeds
        ],
        dtype=bool,
    )
    repaired_seeds = [
        seed
        for seed, old, new in zip(seeds, baseline_success, edge_success)
        if not old and new
    ]
    regressed_seeds = [
        seed
        for seed, old, new in zip(seeds, baseline_success, edge_success)
        if old and not new
    ]
    shared_failures = [
        seed
        for seed, old, new in zip(seeds, baseline_success, edge_success)
        if not old and not new
    ]
    baseline_tail = np.array(
        [baseline_by_seed[s]["mean_last_100"] for s in seeds]
    )
    edge_tail = np.array([edge_by_seed[s]["mean_last_100"] for s in seeds])
    baseline_prediction = np.array(
        [baseline_by_seed[s]["mean_prediction_error_xy"] for s in seeds]
    )
    edge_prediction = np.array(
        [edge_by_seed[s]["mean_prediction_error_xy"] for s in seeds]
    )

    edge_success_count = int(np.sum(edge_success))
    baseline_success_count = int(np.sum(baseline_success))
    payload = {
        "benchmark": "edge-conditioned versus current-chart Atlas-DeePC",
        "seeds": seeds,
        "config": asdict(config),
        "edge_columns": args.edge_columns,
        "baseline_summary": str(args.baseline_summary.resolve()),
        "fairness": {
            "same_seed_panel": True,
            "same_offline_data_and_48_chart_atlas": True,
            "same_graph_router_and_reference_policy": True,
            "same_controller_horizons_costs_constraints_rank_and_solver": True,
            "only_intended_change": (
                "current-chart cached basis versus cached directed-edge basis during routing "
                "and one goal-conditioned terminal-chart basis built before control"
            ),
            "target_phase_model": (
                "baseline uses the current chart; the candidate uses one rank-60 goal-chart "
                "basis selected once from the graph-generated final reference"
            ),
        },
        "comparison": {
            "baseline_successes": baseline_success_count,
            "edge_conditioned_successes": edge_success_count,
            "baseline_success_rate": baseline_success_count / len(seeds),
            "edge_conditioned_success_rate": edge_success_count / len(seeds),
            "baseline_wilson_95": wilson_interval(baseline_success_count, len(seeds)),
            "edge_conditioned_wilson_95": wilson_interval(edge_success_count, len(seeds)),
            "repaired_seeds": repaired_seeds,
            "regressed_seeds": regressed_seeds,
            "shared_failure_seeds": shared_failures,
            "exact_mcnemar_p": exact_mcnemar_p(
                len(repaired_seeds), len(regressed_seeds)
            ),
            "median_baseline_final100_mm": float(1000 * np.median(baseline_tail)),
            "median_edge_final100_mm": float(1000 * np.median(edge_tail)),
            "median_paired_final100_change_mm": float(
                1000 * np.median(edge_tail - baseline_tail)
            ),
            "median_baseline_prediction_error_mm": float(
                1000 * np.median(baseline_prediction)
            ),
            "median_edge_prediction_error_mm": float(
                1000 * np.median(edge_prediction)
            ),
            "median_paired_prediction_change_mm": float(
                1000 * np.median(edge_prediction - baseline_prediction)
            ),
            "median_edge_control_compute_ms": float(
                np.median(
                    [edge_by_seed[s]["median_control_compute_ms"] for s in seeds]
                )
            ),
            "median_edge_usage_fraction": float(
                np.median(
                    [edge_by_seed[s]["edge_model_usage_fraction"] for s in seeds]
                )
            ),
            "median_terminal_model_usage_fraction": float(
                np.median(
                    [
                        edge_by_seed[s]["terminal_model_usage_fraction"]
                        for s in seeds
                    ]
                )
            ),
        },
        "cache_metadata": edge_by_seed[seeds[0]],
        "results": [edge_by_seed[seed] for seed in seeds],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    make_figure(args.output_dir, seeds, baseline_by_seed, edge_by_seed)
    print(json.dumps(payload["comparison"], indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--atlas-path", type=Path, default=DEFAULT_ATLAS_PATH)
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=DEFAULT_BASELINE_SUMMARY,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed-start", type=int, default=111)
    parser.add_argument("--seed-stop", type=int, default=210)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--edge-columns", type=int, default=200)
    parser.add_argument("--rank", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
