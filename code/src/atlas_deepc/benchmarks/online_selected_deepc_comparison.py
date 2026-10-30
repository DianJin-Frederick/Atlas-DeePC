"""Controlled Atlas-chart versus online trajectory-selection DeePC ablation.

All variants retain the graph router, waypoint policy, Hankel library, output
representation, horizons, costs, constraints, CLARABEL solver, and one QP per
control step. The runner supports two online-selection representations:

* Atlas: columns belonging to the localized chart (cached chart basis).
* Rank matched: select the N closest full trajectory windows using either the
  Select-DPC full-trajectory L1 score or the Contextual Sampling output/reference
  score, then rebuild the same rank-60 affine representation used by Atlas.
* Raw: optimize one coefficient per selected column with ``sum(g)=1``.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cvxpy as cp
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from atlas_deepc.control import expert_pipeline_search as pipe
from atlas_deepc.atlas.routed_deepc_experiment import (
    DEFAULT_DATA_PATH,
    RoutedDeePCConfig,
    assign_charts,
    load_atlas,
    online_behavioral_feature,
    run_routed_deepc,
    shortest_chart_path,
)
from atlas_deepc.atlas.goal_reference import chart_compatible_goal


DEFAULT_ATLAS_PATH = Path(
    "experiments/routed_deepc_iid/heldout100_frozen_target_separated/behavioral_atlas.npz"
)
DEFAULT_OUTPUT_DIR = Path("experiments/online_selected_deepc_comparison")


def build_controller(config: RoutedDeePCConfig) -> pipe.DeePCController:
    Q = np.diag(
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
    pipe.validate_objective_options(config.lam_g, config.lam_du)
    return pipe.DeePCController(
        Q,
        config.r * np.eye(pipe.M),
        config.lam_ini,
        config.u_limit,
        cp.CLARABEL,
    )


def select_relevant_columns(
    H_u: np.ndarray,
    H_y: np.ndarray,
    query_u: np.ndarray,
    query_y: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select-DPC's raw full-trajectory L1 ranking."""
    distances = np.sum(np.abs(H_u - query_u), axis=0)
    distances += np.sum(np.abs(H_y - query_y), axis=0)
    selected = np.argsort(distances, kind="stable")[:count]
    return selected, distances


def prepare_contextual_library(H: np.ndarray) -> dict[str, np.ndarray]:
    """Precompute the output normalization used by Contextual Sampling.

    The paper normalizes each output channel once from the training data, then
    evaluates past-output and future-reference distances online.  This
    preprocessing is deliberately outside the online timing boundary.
    """
    _U_p, _U_f, Y_p, Y_f = pipe.partition_hankel_rows(H)
    Y_all = H[pipe.M * pipe.L :].reshape(pipe.L, pipe.P, H.shape[1])
    output_mean = Y_all.mean(axis=(0, 2))
    output_scale = np.maximum(Y_all.std(axis=(0, 2)), 1e-8)
    Y_p_normalized = (
        (Y_p.reshape(pipe.T_INI, pipe.P, H.shape[1]) - output_mean[None, :, None])
        / output_scale[None, :, None]
    ).reshape(pipe.P * pipe.T_INI, H.shape[1])
    Y_f_normalized = (
        (Y_f.reshape(pipe.T_F, pipe.P, H.shape[1]) - output_mean[None, :, None])
        / output_scale[None, :, None]
    ).reshape(pipe.P * pipe.T_F, H.shape[1])
    return {
        "Y_p_normalized": Y_p_normalized,
        "Y_f_normalized": Y_f_normalized,
        "output_mean": output_mean,
        "output_scale": output_scale,
    }


def contextual_distance_components(
    library: dict[str, np.ndarray],
    y_ini: np.ndarray,
    reference: np.ndarray,
    output_cost: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return standardized past and Q-weighted future contextual distances."""
    output_mean = library["output_mean"]
    output_scale = library["output_scale"]
    y_ini_normalized = (
        (np.asarray(y_ini).reshape(pipe.T_INI, pipe.P) - output_mean[None, :])
        / output_scale[None, :]
    ).reshape(pipe.P * pipe.T_INI, 1)
    reference_normalized = (
        (np.asarray(reference).reshape(pipe.T_F, pipe.P) - output_mean[None, :])
        / output_scale[None, :]
    ).reshape(pipe.P * pipe.T_F, 1)

    past_distance = np.linalg.norm(
        library["Y_p_normalized"] - y_ini_normalized,
        axis=0,
    )
    q_diagonal = np.diag(np.asarray(output_cost, dtype=float))
    future_weights = np.tile(q_diagonal, pipe.T_F).reshape(-1, 1)
    future_delta = library["Y_f_normalized"] - reference_normalized
    future_distance = np.sqrt(np.sum(future_weights * future_delta**2, axis=0))

    def standardize_distance(values: np.ndarray) -> np.ndarray:
        scale = max(float(np.std(values)), 1e-12)
        return (values - float(np.mean(values))) / scale

    return standardize_distance(past_distance), standardize_distance(future_distance)


def select_contextual_columns(
    library: dict[str, np.ndarray],
    y_ini: np.ndarray,
    reference: np.ndarray,
    output_cost: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Contextual Sampling equations (5)--(7) from arXiv:2503.23890v2.

    Rank each complete Hankel column using the standardized sum of its distance
    to the measured past outputs and its Q-weighted distance to the graph-routed
    future reference.  The selected indices still retrieve complete U/Y columns.
    """
    past_score, future_score = contextual_distance_components(
        library, y_ini, reference, output_cost
    )
    contextual_score = past_score + future_score
    selected = np.argsort(contextual_score, kind="stable")[:count]
    return selected, contextual_score


def selected_affine_basis(
    H: np.ndarray,
    selected: np.ndarray,
    config: RoutedDeePCConfig,
) -> dict[str, np.ndarray]:
    """Apply the exact Atlas affine rank/bound construction to selected columns."""
    X = np.asarray(H[:, selected], dtype=float)
    center = X.mean(axis=1, keepdims=True)
    rank = min(config.svd_rank, X.shape[0] - 1, X.shape[1] - 1)
    basis = pipe.fixed_rank_left_svd_basis(X - center, rank)
    coordinates = basis.T @ (X - center)
    coordinate_mean = coordinates.mean(axis=1, keepdims=True)
    coordinate_scale = np.maximum(coordinates.std(axis=1, keepdims=True), 1e-8)
    return {
        "offset": center,
        "basis": basis,
        "g_min": coordinate_mean - config.svd_bound_scale * coordinate_scale,
        "g_max": coordinate_mean + config.svd_bound_scale * coordinate_scale,
    }


def selected_raw_columns(H: np.ndarray, selected: np.ndarray) -> dict:
    """Use the selected Hankel columns directly with an affine sum constraint."""
    return {
        "offset": np.zeros((H.shape[0], 1)),
        "basis": np.asarray(H[:, selected], dtype=float),
        "g_sum": 1.0,
    }


def timing_summary(values: np.ndarray, name: str) -> dict[str, float]:
    milliseconds = 1000 * np.asarray(values, dtype=float)
    steady = milliseconds[min(5, milliseconds.size) :]
    if not steady.size:
        steady = milliseconds
    return {
        f"mean_{name}_ms": float(np.mean(milliseconds)),
        f"median_{name}_ms": float(np.median(milliseconds)),
        f"p95_{name}_ms": float(np.percentile(milliseconds, 95)),
        f"p99_{name}_ms": float(np.percentile(milliseconds, 99)),
        f"max_{name}_ms": float(np.max(milliseconds)),
        f"steady_median_{name}_ms": float(np.median(steady)),
        f"steady_p95_{name}_ms": float(np.percentile(steady, 95)),
    }


def run_online_selected_deepc(
    H: np.ndarray,
    atlas: dict,
    config: RoutedDeePCConfig,
    selected_columns: int,
    selected_model: str = "raw",
    selection_method: str = "select_dpc",
    reference_mode: str = "graph_routed",
    save_path: Path | None = None,
) -> dict:
    if selected_model not in ("raw", "rank_matched"):
        raise ValueError("selected_model must be 'raw' or 'rank_matched'")
    if selection_method not in ("select_dpc", "contextual"):
        raise ValueError("selection_method must be 'select_dpc' or 'contextual'")
    if reference_mode not in ("graph_routed", "final_only"):
        raise ValueError("reference_mode must be 'graph_routed' or 'final_only'")
    if selected_model == "rank_matched" and selected_columns <= config.svd_rank:
        raise ValueError("selected_columns must exceed svd_rank for a rank-matched affine basis")
    if selected_columns > H.shape[1]:
        raise ValueError("selected_columns exceeds the Hankel library size")

    H_u = H[: pipe.M * pipe.L]
    H_y = H[pipe.M * pipe.L :]
    contextual_library = prepare_contextual_library(H)
    output_cost = np.diag(
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
    controller = build_controller(config)
    env = gym.make(pipe.REACHER_ENV_ID, max_episode_steps=config.steps + pipe.T_INI + 5)
    obs, _ = env.reset(seed=config.seed)
    target_xy = np.asarray(obs[4:6], dtype=float).copy()
    initial_y = pipe.reacher_output_from_obs(obs)
    goal, goal_metadata = chart_compatible_goal(H, target_xy, initial_y, config)
    goal_index = int(goal_metadata["data_goal_index"])
    goal_chart = int(atlas["labels"][goal_index])

    u_history: list[np.ndarray] = []
    y_history: list[np.ndarray] = []
    for _ in range(pipe.T_INI):
        action = np.zeros(pipe.M)
        obs, *_ = env.step(action)
        u_history.append(action.copy())
        y_history.append(pipe.reacher_output_from_obs(obs))

    planned_u = np.zeros((pipe.M * pipe.T_F, 1))
    planned_y = np.tile(y_history[-1].reshape(pipe.P, 1), (pipe.T_F, 1))
    previous_selection: np.ndarray | None = None
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
            "selected_indices",
            "selected_overlap_fraction",
            "selected_distance_median",
            "localization_time_s",
            "routing_time_s",
            "selection_time_s",
            "svd_rebuild_time_s",
            "matrix_preparation_time_s",
            "solve_time_s",
            "control_compute_time_s",
        )
    }

    for step in range(config.steps):
        control_start = time.perf_counter()
        u_ini = np.asarray(u_history[-pipe.T_INI :]).reshape(pipe.M * pipe.T_INI, 1)
        y_ini = np.asarray(y_history[-pipe.T_INI :]).reshape(pipe.P * pipe.T_INI, 1)

        if reference_mode == "graph_routed":
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
            current_chart = int(assign_charts(atlas["centers"], feature)[0])
            localization_time = time.perf_counter() - stage

            stage = time.perf_counter()
            path = shortest_chart_path(
                atlas, current_chart, goal_chart, config.edge_count_threshold
            ) or [current_chart]
            next_chart = path[1] if len(path) > 1 else current_chart
            distance_before = float(np.linalg.norm(y_history[-1][6:8] - target_xy))
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
            routing_time = time.perf_counter() - stage
        else:
            # The final-only ablation deliberately performs no online chart
            # localization or shortest-path routing.  It uses the same final
            # graph-compatible goal as every other method from step zero.
            current_chart = -1
            next_chart = goal_chart
            path = [goal_chart]
            active_reference = goal
            localization_time = 0.0
            routing_time = 0.0
        reference = np.tile(active_reference.reshape(pipe.P, 1), (pipe.T_F, 1))

        query_u = np.vstack([u_ini, planned_u])
        query_y = np.vstack([y_ini, planned_y])
        stage = time.perf_counter()
        if selection_method == "select_dpc":
            selected, distances = select_relevant_columns(
                H_u, H_y, query_u, query_y, selected_columns
            )
        else:
            selected, distances = select_contextual_columns(
                contextual_library,
                y_ini,
                reference,
                output_cost,
                selected_columns,
            )
        selection_time = time.perf_counter() - stage

        stage = time.perf_counter()
        matrix = (
            selected_raw_columns(H, selected)
            if selected_model == "raw"
            else selected_affine_basis(H, selected, config)
        )
        matrix_time = time.perf_counter() - stage
        svd_rebuild_time = matrix_time if selected_model == "rank_matched" else 0.0

        stage = time.perf_counter()
        u_solution, y_solution, g_solution, status, _ = controller.solve(
            matrix, u_ini, y_ini, reference
        )
        solve_time = time.perf_counter() - stage
        if u_solution is None:
            action = np.zeros(pipe.M)
            prediction = np.full(pipe.P, np.nan)
        else:
            action = np.clip(u_solution[: pipe.M, 0], -config.u_limit, config.u_limit)
            prediction = np.asarray(y_solution[: pipe.P, 0]).reshape(pipe.P)
            planned_u = np.asarray(u_solution, dtype=float).reshape(-1, 1)
            planned_y = np.asarray(y_solution, dtype=float).reshape(-1, 1)
        compute_time = time.perf_counter() - control_start

        obs, *_ = env.step(action)
        y_now = pipe.reacher_output_from_obs(obs)
        u_history.append(action.copy())
        y_history.append(y_now.copy())
        overlap = np.nan
        if previous_selection is not None:
            overlap = np.intersect1d(previous_selection, selected).size / selected_columns
        previous_selection = selected.copy()

        records["y"].append(y_now)
        records["u"].append(action)
        records["pred"].append(prediction)
        records["ref"].append(active_reference)
        records["distance"].append(np.linalg.norm(y_now[6:8] - target_xy))
        records["prediction_error_xy"].append(np.linalg.norm(prediction[6:8] - y_now[6:8]))
        records["status"].append(str(status))
        records["chart"].append(current_chart)
        records["goal_chart"].append(goal_chart)
        records["next_chart"].append(next_chart)
        records["path_length"].append(len(path))
        records["selected_indices"].append(selected)
        records["selected_overlap_fraction"].append(overlap)
        records["selected_distance_median"].append(float(np.median(distances[selected])))
        records["localization_time_s"].append(localization_time)
        records["routing_time_s"].append(routing_time)
        records["selection_time_s"].append(selection_time)
        records["svd_rebuild_time_s"].append(svd_rebuild_time)
        records["matrix_preparation_time_s"].append(matrix_time)
        records["solve_time_s"].append(solve_time)
        records["control_compute_time_s"].append(compute_time)
        if (step + 1) % 20 == 0:
            print(
                f"online selection seed={config.seed}: {step + 1}/{config.steps}, "
                f"distance={1000 * records['distance'][-1]:.2f} mm, status={status}",
                flush=True,
            )
    env.close()

    arrays = {
        key: np.asarray(value, dtype=object if key == "status" else None)
        for key, value in records.items()
    }
    distance = arrays["distance"]
    tail_30 = distance[-30:]
    tail_100 = distance[-min(100, distance.size) :]
    reference_cost = pipe.closed_loop_control_cost(
        arrays["y"],
        arrays["u"],
        arrays["ref"],
        output_cost,
        config.r * np.eye(pipe.M),
    )
    final_goal_cost = pipe.closed_loop_control_cost(
        arrays["y"],
        arrays["u"],
        goal,
        output_cost,
        config.r * np.eye(pipe.M),
        reference_description="common final graph-compatible goal",
    )
    summary = {
        **asdict(config),
        "method": f"{reference_mode}_{selection_method}_selected_{selected_model}_deepc",
        "selection_method": selection_method,
        "reference_mode": reference_mode,
        "graph_routing_used": reference_mode == "graph_routed",
        "selected_columns": selected_columns,
        "selected_model": selected_model,
        "optimized_coefficient_dimension": (
            selected_columns if selected_model == "raw" else config.svd_rank
        ),
        "qp_count": config.steps,
        "minimum": float(distance.min()),
        "minimum_step": int(distance.argmin()),
        "last": float(distance[-1]),
        "mean_last_30": float(tail_30.mean()),
        "max_last_30": float(tail_30.max()),
        "mean_last_100": float(tail_100.mean()),
        "max_last_100": float(tail_100.max()),
        "success_30": bool(tail_30.mean() < 0.01 and tail_30.max() < 0.02),
        "success_100": bool(tail_100.mean() < 0.01 and tail_100.max() < 0.02),
        "mean_prediction_error_xy": float(np.nanmean(arrays["prediction_error_xy"])),
        "mean_selected_overlap_fraction": float(
            np.nanmean(arrays["selected_overlap_fraction"])
        ),
        "statuses": dict(Counter(arrays["status"].tolist())),
        "charts_visited": int(np.unique(arrays["chart"]).size),
        "library_columns": int(H.shape[1]),
        "selector": (
            "raw L1 over complete measured-past/previous-prediction trajectories"
            if selection_method == "select_dpc"
            else (
                "Contextual Sampling v2: standardized past-output distance plus "
                f"Q-weighted future {reference_mode.replace('_', '-')} reference distance"
            )
        ),
        "local_model": (
            "raw selected Hankel columns with sum(g)=1"
            if selected_model == "raw"
            else "same affine fixed-rank SVD and coefficient bounds as Atlas chart_matrix"
        ),
        "fitted_dynamics_model": False,
        "online_nearest_trajectory_retrieval": True,
        **reference_cost,
        "closed_loop_reference_cost": reference_cost["closed_loop_control_cost"],
        "closed_loop_reference_tracking_cost": reference_cost["closed_loop_tracking_cost"],
        "mean_closed_loop_reference_stage_cost": reference_cost["mean_closed_loop_stage_cost"],
        "closed_loop_final_goal_cost": final_goal_cost["closed_loop_control_cost"],
        "closed_loop_final_goal_tracking_cost": final_goal_cost["closed_loop_tracking_cost"],
        "mean_closed_loop_final_goal_stage_cost": final_goal_cost["mean_closed_loop_stage_cost"],
        "closed_loop_final_goal_cost_definition": final_goal_cost["closed_loop_cost_definition"],
        **timing_summary(arrays["localization_time_s"], "localization"),
        **timing_summary(arrays["routing_time_s"], "routing"),
        **timing_summary(arrays["selection_time_s"], "selection"),
        **timing_summary(arrays["svd_rebuild_time_s"], "svd_rebuild"),
        **timing_summary(
            arrays["routing_time_s"] + arrays["selection_time_s"],
            "routing_plus_selection",
        ),
        **timing_summary(arrays["matrix_preparation_time_s"], "matrix_preparation"),
        **timing_summary(arrays["solve_time_s"], "solve"),
        **timing_summary(arrays["control_compute_time_s"], "control_compute"),
        "control_compute_includes_svd_rebuild": selected_model == "rank_matched",
        **goal_metadata,
    }
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path,
            **arrays,
            final_goal=goal,
            summary=np.array(json.dumps(summary), dtype=object),
        )
    return summary


def make_figure(output_dir: Path, atlas_path: Path, selected_path: Path) -> None:
    atlas = np.load(atlas_path, allow_pickle=True)
    selected = np.load(selected_path, allow_pickle=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))
    for label, data, color in (
        ("Atlas chart", atlas, "tab:blue"),
        ("Online top-N", selected, "tab:purple"),
    ):
        axes[0].plot(1000 * data["distance"], label=label, color=color)
        axes[1].plot(1000 * data["prediction_error_xy"], label=label, color=color)
        axes[2].plot(1000 * data["control_compute_time_s"], label=label, color=color)
    axes[0].set(title="Target tracking", xlabel="Control step", ylabel="Error [mm]")
    axes[1].set(title="One-step prediction", xlabel="Control step", ylabel="XY error [mm]")
    axes[2].set(title="End-to-end online compute", xlabel="Control step", ylabel="Time [ms]", yscale="log")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("Controlled ablation: cached Atlas chart vs online relevant trajectories")
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / "comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    H = np.load(args.data_path)
    atlas = load_atlas(args.atlas_path)
    config = RoutedDeePCConfig(steps=args.steps, seed=args.seed, svd_rank=args.rank)
    atlas_path = args.output_dir / "atlas_control_results.npz"
    selected_path = args.output_dir / "online_selected_control_results.npz"
    print("running Atlas baseline", flush=True)
    atlas_summary = run_routed_deepc(H, atlas, config, save_path=atlas_path)
    atlas_summary.update(
        method="graph_routed_atlas_chart_deepc",
        selected_columns="localized chart membership",
        optimized_coefficient_dimension=config.svd_rank,
        qp_count=config.steps,
        library_columns=int(H.shape[1]),
    )
    print("running online trajectory-selection ablation", flush=True)
    selected_summary = run_online_selected_deepc(
        H,
        atlas,
        config,
        args.selected_columns,
        args.selected_model,
        args.selection_method,
        args.reference_mode,
        save_path=selected_path,
    )
    latency_ratio = (
        selected_summary["median_control_compute_ms"]
        / atlas_summary["median_control_compute_time_ms"]
    )
    payload = {
        "benchmark": "controlled graph-routed DeePC local-column mechanism ablation",
        "shared": {
            "data_path": str(args.data_path.resolve()),
            "atlas_path": str(args.atlas_path.resolve()),
            "environment": pipe.REACHER_ENV_ID,
            "seed": args.seed,
            "steps": args.steps,
            "T_ini": pipe.T_INI,
            "T_f": pipe.T_F,
            "output_dimension": pipe.P,
            "input_dimension": pipe.M,
            "rank": args.rank,
            "solver": "CLARABEL",
            "qp_per_step": 1,
            "router_waypoint_costs_constraints_regularization": "identical",
        },
        "only_intended_change": {
            "atlas": "localized chart column membership, chart basis cached",
            "online_selection": (
                f"top {args.selected_columns} raw-L1 full-trajectory Hankel columns; "
                + (
                    "used directly with sum(g)=1"
                    if args.selected_model == "raw"
                    else "compressed to the same affine rank/bounds rebuilt online"
                )
            ),
        },
        "results": [atlas_summary, selected_summary],
        "comparison": {
            "median_latency_ratio_online_selection_over_atlas": float(latency_ratio),
            "atlas_success": bool(atlas_summary["success_30"] and atlas_summary["success_100"]),
            "online_selection_success": bool(
                selected_summary["success_30"] and selected_summary["success_100"]
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    make_figure(args.output_dir, atlas_path, selected_path)
    print(json.dumps(payload["comparison"], indent=2), flush=True)
    for result in payload["results"]:
        median_key = (
            "median_control_compute_time_ms"
            if "median_control_compute_time_ms" in result
            else "median_control_compute_ms"
        )
        print(
            f"{result['method']}: median={result[median_key]:.2f} ms, "
            f"final={1000 * result['last']:.2f} mm, "
            f"success={result['success_30'] and result['success_100']}",
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--atlas-path", type=Path, default=DEFAULT_ATLAS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--selected-columns", type=int, default=200)
    parser.add_argument(
        "--selected-model",
        choices=("raw", "rank_matched"),
        default="raw",
        help="Use 200 raw coefficients or compress the selected set to the Atlas rank.",
    )
    parser.add_argument(
        "--selection-method",
        choices=("select_dpc", "contextual"),
        default="select_dpc",
    )
    parser.add_argument(
        "--reference-mode",
        choices=("graph_routed", "final_only"),
        default="graph_routed",
    )
    parser.add_argument("--rank", type=int, default=60)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
