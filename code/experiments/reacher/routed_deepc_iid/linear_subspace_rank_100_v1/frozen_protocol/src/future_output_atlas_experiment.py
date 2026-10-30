"""Reference-conditioned Atlas-DeePC using offline Y_f mode clusters.

The physical atlas is unchanged: its charts are fitted from the current plant
configuration in Y_p.  This module adds a second offline description to every
physical chart by clustering that chart's Q-weighted future-output block Y_f.
At runtime, a terminal stagnation trigger scores only cached Y_f prototypes in
the current chart and its observed outgoing graph neighbors.  The selected
current/neighbor chart-pair DeePC basis is also cached offline.

No online Hankel-column ranking, online SVD, fitted dynamics model, IK, or
fallback controller is used.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import gymnasium as gym
import numpy as np
from sklearn.cluster import KMeans

import expert_pipeline_search as pipe
import routed_deepc_experiment as routed
from contextual_chart_expansion_experiment import baseline_config, output_cost
from local_affine_atlas_experiment import audit_iid_library
from online_selected_deepc_comparison import (
    build_controller,
    prepare_contextual_library,
    timing_summary,
)
from stitched_atlas_mpc_experiment import chart_compatible_goal


DEFAULT_DATA_PATH = Path("data/select_iid_offline_hankel.npy")


@dataclass(frozen=True)
class FutureOutputAtlasConfig:
    modes_per_chart: int = 8
    mode_cluster_seed: int = 37
    error_threshold: float = 0.01
    stall_window: int = 15
    minimum_progress: float = 0.001
    score_margin: float = 0.0
    edge_count_threshold: int = 3
    require_alternate_after_stall: bool = True


def _matrix_to_arrays(matrix: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    return (
        np.asarray(matrix["offset"], dtype=float),
        np.asarray(matrix["basis"], dtype=float),
        np.asarray(matrix["g_min"], dtype=float),
        np.asarray(matrix["g_max"], dtype=float),
    )


def build_future_output_atlas(
    H: np.ndarray,
    atlas: dict,
    control_config: routed.RoutedDeePCConfig,
    future_config: FutureOutputAtlasConfig,
) -> dict[str, np.ndarray]:
    """Fit chart-local Y_f modes and all admissible cached chart-pair bases."""
    labels = np.asarray(atlas["labels"], dtype=int)
    chart_count = int(atlas["centers"].shape[0])
    library = prepare_contextual_library(H)
    q_diagonal = np.diag(output_cost(control_config))
    sqrt_future_weights = np.sqrt(np.tile(q_diagonal, pipe.T_F))
    weighted_future = (
        library["Y_f_normalized"].T * sqrt_future_weights[None, :]
    )

    prototypes = np.empty(
        (chart_count, future_config.modes_per_chart, weighted_future.shape[1]),
        dtype=float,
    )
    prototype_counts = np.zeros(
        (chart_count, future_config.modes_per_chart), dtype=int
    )
    for chart in range(chart_count):
        members = weighted_future[labels == chart]
        if members.shape[0] < future_config.modes_per_chart:
            raise RuntimeError(
                f"Chart {chart} has {members.shape[0]} columns, fewer than "
                f"{future_config.modes_per_chart} requested Y_f modes."
            )
        model = KMeans(
            n_clusters=future_config.modes_per_chart,
            random_state=future_config.mode_cluster_seed + chart,
            n_init=5,
            max_iter=300,
            algorithm="lloyd",
        )
        mode_labels = model.fit_predict(members)
        prototypes[chart] = model.cluster_centers_
        prototype_counts[chart] = np.bincount(
            mode_labels, minlength=future_config.modes_per_chart
        )

    counts = np.asarray(atlas["transition_counts"], dtype=int)
    keys = [(chart, chart) for chart in range(chart_count)]
    keys.extend(
        (source, destination)
        for source in range(chart_count)
        for destination in range(source + 1, chart_count)
        if (
            counts[source, destination] >= future_config.edge_count_threshold
            or counts[destination, source] >= future_config.edge_count_threshold
        )
    )
    offsets = []
    bases = []
    g_mins = []
    g_maxs = []
    matrix_columns = []
    for source, destination in keys:
        chart_ids = (source,) if source == destination else (source, destination)
        matrix, selected = routed.chart_matrix(H, atlas, chart_ids, control_config)
        offset, basis, g_min, g_max = _matrix_to_arrays(matrix)
        offsets.append(offset)
        bases.append(basis)
        g_mins.append(g_min)
        g_maxs.append(g_max)
        matrix_columns.append(selected.size)

    return {
        "prototypes": prototypes,
        "prototype_counts": prototype_counts,
        "output_mean": library["output_mean"],
        "output_scale": library["output_scale"],
        "sqrt_future_weights": sqrt_future_weights,
        "matrix_keys": np.asarray(keys, dtype=int),
        "matrix_offsets": np.asarray(offsets),
        "matrix_bases": np.asarray(bases),
        "matrix_g_mins": np.asarray(g_mins),
        "matrix_g_maxs": np.asarray(g_maxs),
        "matrix_columns": np.asarray(matrix_columns, dtype=int),
        "future_config": np.array(json.dumps(asdict(future_config)), dtype=object),
    }


def save_future_output_atlas(path: Path, artifact: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **artifact)


def load_future_output_atlas(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def matrix_cache_from_artifact(artifact: dict) -> dict[tuple[int, ...], tuple[dict, int]]:
    cache = {}
    for index, (source, destination) in enumerate(artifact["matrix_keys"]):
        key = (int(source),) if source == destination else (int(source), int(destination))
        cache[key] = (
            {
                "offset": artifact["matrix_offsets"][index],
                "basis": artifact["matrix_bases"][index],
                "g_min": artifact["matrix_g_mins"][index],
                "g_max": artifact["matrix_g_maxs"][index],
            },
            int(artifact["matrix_columns"][index]),
        )
    return cache


def reference_future_query(reference: np.ndarray, artifact: dict) -> np.ndarray:
    normalized = (
        np.asarray(reference, dtype=float).reshape(pipe.T_F, pipe.P)
        - artifact["output_mean"][None, :]
    ) / artifact["output_scale"][None, :]
    return normalized.reshape(-1) * artifact["sqrt_future_weights"]


def select_reference_conditioned_neighbor(
    current_chart: int,
    reference: np.ndarray,
    atlas: dict,
    artifact: dict,
    config: FutureOutputAtlasConfig,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Score only current/outgoing-neighbor Y_f prototypes."""
    counts = np.asarray(atlas["transition_counts"], dtype=int)
    outgoing = np.flatnonzero(
        (counts[current_chart] >= config.edge_count_threshold)
        & (np.arange(counts.shape[0]) != current_chart)
    )
    candidates = np.r_[current_chart, outgoing].astype(int)
    query = reference_future_query(reference, artifact)
    chart_scores = np.full(counts.shape[0], np.inf, dtype=float)
    for chart in candidates:
        delta = artifact["prototypes"][chart] - query[None, :]
        chart_scores[chart] = float(np.min(np.mean(delta * delta, axis=1)))
    selectable = (
        outgoing
        if config.require_alternate_after_stall and outgoing.size
        else candidates
    )
    best = int(selectable[np.argmin(chart_scores[selectable])])
    current_score = chart_scores[current_chart]
    if (
        not config.require_alternate_after_stall
        and best != current_chart
        and chart_scores[best] + config.score_margin >= current_score
    ):
        best = current_chart
    return best, candidates, chart_scores


def run_future_output_atlas(
    H: np.ndarray,
    atlas: dict,
    artifact: dict,
    control_config: routed.RoutedDeePCConfig,
    future_config: FutureOutputAtlasConfig,
    save_path: Path | None = None,
    activation_mode: str = "triggered",
) -> dict:
    if activation_mode not in {"triggered", "always_on"}:
        raise ValueError(f"Unknown activation_mode: {activation_mode}")
    always_on = activation_mode == "always_on"
    Q = output_cost(control_config)
    controller = build_controller(control_config)
    matrices = matrix_cache_from_artifact(artifact)
    labels = np.asarray(atlas["labels"], dtype=int)

    env = gym.make(
        pipe.REACHER_ENV_ID,
        max_episode_steps=control_config.steps + pipe.T_INI + 5,
    )
    obs, _ = env.reset(seed=control_config.seed)
    target_xy = np.asarray(obs[4:6], dtype=float).copy()
    initial_y = pipe.reacher_output_from_obs(obs)
    goal, goal_metadata = chart_compatible_goal(
        H, target_xy, initial_y, control_config
    )
    goal_chart = int(labels[int(goal_metadata["data_goal_index"])])

    u_history = []
    y_history = []
    for _ in range(pipe.T_INI):
        action = np.zeros(pipe.M)
        obs, *_ = env.step(action)
        u_history.append(action.copy())
        y_history.append(pipe.reacher_output_from_obs(obs))

    target_reference_latched = False
    terminal_distance_history = []
    future_mode_active = always_on
    trigger_step = -1
    records = {key: [] for key in (
        "y", "u", "pred", "ref", "distance", "prediction_error_xy",
        "status", "chart", "goal_chart", "next_chart", "path_length",
        "future_mode_active", "future_mode_triggered", "future_chart",
        "future_current_score", "future_selected_score", "candidate_count",
        "matrix_columns", "localization_time_s", "routing_time_s",
        "future_scoring_time_s", "matrix_preparation_time_s", "solve_time_s",
        "control_compute_time_s",
    )}

    for step in range(control_config.steps):
        control_start = time.perf_counter()
        u_ini = np.asarray(u_history[-pipe.T_INI:]).reshape(pipe.M * pipe.T_INI, 1)
        y_ini = np.asarray(y_history[-pipe.T_INI:]).reshape(pipe.P * pipe.T_INI, 1)

        stage = time.perf_counter()
        feature = routed.online_behavioral_feature(
            u_ini,
            y_ini,
            atlas["feature_mean"],
            atlas["feature_scale"],
            control_config.clustering_input_weight,
            control_config.clustering_feature_mode,
            control_config.clustering_configuration_weight,
            control_config.clustering_velocity_weight,
            control_config.clustering_fingertip_weight,
        )
        current_chart = int(routed.assign_atlas_charts(atlas, feature)[0])
        localization_time = time.perf_counter() - stage

        stage = time.perf_counter()
        path = routed.shortest_chart_path(
            atlas,
            current_chart,
            goal_chart,
            control_config.edge_count_threshold,
        ) or [current_chart]
        next_chart = path[1] if len(path) > 1 else current_chart
        distance_before = float(np.linalg.norm(y_history[-1][6:8] - target_xy))
        if distance_before < control_config.target_switch_distance:
            target_reference_latched = True
        use_target = (
            target_reference_latched
            if control_config.latch_target_reference
            else distance_before < control_config.target_switch_distance
        )
        terminal_reference_active = use_target or current_chart == goal_chart
        active_reference = (
            goal if terminal_reference_active else atlas["medoid_states"][next_chart]
        )
        reference = np.tile(active_reference.reshape(pipe.P, 1), (pipe.T_F, 1))
        routing_time = time.perf_counter() - stage

        triggered = False
        if not always_on and terminal_reference_active:
            terminal_distance_history.append(distance_before)
            if len(terminal_distance_history) > future_config.stall_window:
                progress = (
                    terminal_distance_history[-future_config.stall_window - 1]
                    - terminal_distance_history[-1]
                )
                if (
                    not future_mode_active
                    and distance_before > future_config.error_threshold
                    and progress < future_config.minimum_progress
                ):
                    future_mode_active = True
                    triggered = True
                    trigger_step = step
        elif not always_on and not future_mode_active:
            terminal_distance_history.clear()

        stage = time.perf_counter()
        selected_chart = current_chart
        candidates = np.asarray([current_chart])
        scores = np.full(control_config.clusters, np.nan)
        if future_mode_active:
            selected_chart, candidates, scores = select_reference_conditioned_neighbor(
                current_chart,
                reference,
                atlas,
                artifact,
                future_config,
            )
        scoring_time = time.perf_counter() - stage

        stage = time.perf_counter()
        key = (
            (current_chart,)
            if selected_chart == current_chart
            else tuple(sorted((current_chart, selected_chart)))
        )
        matrix, matrix_columns = matrices[key]
        matrix_time = time.perf_counter() - stage

        stage = time.perf_counter()
        u_solution, y_solution, _g, status, _ = controller.solve(
            matrix, u_ini, y_ini, reference
        )
        solve_time = time.perf_counter() - stage
        if u_solution is None:
            action = np.zeros(pipe.M)
            prediction = np.full(pipe.P, np.nan)
        else:
            action = np.clip(
                u_solution[:pipe.M, 0],
                -control_config.u_limit,
                control_config.u_limit,
            )
            prediction = np.asarray(y_solution[:pipe.P, 0]).reshape(pipe.P)
        compute_time = time.perf_counter() - control_start

        obs, *_ = env.step(action)
        y_now = pipe.reacher_output_from_obs(obs)
        u_history.append(action.copy())
        y_history.append(y_now.copy())

        records["y"].append(y_now)
        records["u"].append(action)
        records["pred"].append(prediction)
        records["ref"].append(active_reference)
        records["distance"].append(float(np.linalg.norm(y_now[6:8] - target_xy)))
        records["prediction_error_xy"].append(
            float(np.linalg.norm(prediction[6:8] - y_now[6:8]))
        )
        records["status"].append(str(status))
        records["chart"].append(current_chart)
        records["goal_chart"].append(goal_chart)
        records["next_chart"].append(next_chart)
        records["path_length"].append(len(path))
        records["future_mode_active"].append(future_mode_active)
        records["future_mode_triggered"].append(triggered)
        records["future_chart"].append(selected_chart)
        records["future_current_score"].append(scores[current_chart])
        records["future_selected_score"].append(scores[selected_chart])
        records["candidate_count"].append(candidates.size)
        records["matrix_columns"].append(matrix_columns)
        records["localization_time_s"].append(localization_time)
        records["routing_time_s"].append(routing_time)
        records["future_scoring_time_s"].append(scoring_time)
        records["matrix_preparation_time_s"].append(matrix_time)
        records["solve_time_s"].append(solve_time)
        records["control_compute_time_s"].append(compute_time)
    env.close()

    arrays = {
        key: np.asarray(value, dtype=object if key == "status" else None)
        for key, value in records.items()
    }
    distance = arrays["distance"].astype(float)
    tail_30 = distance[-30:]
    tail_100 = distance[-min(100, distance.size):]
    reference_cost = pipe.closed_loop_control_cost(
        arrays["y"], arrays["u"], arrays["ref"], Q, control_config.r * np.eye(pipe.M)
    )
    final_goal_cost = pipe.closed_loop_control_cost(
        arrays["y"],
        arrays["u"],
        goal,
        Q,
        control_config.r * np.eye(pipe.M),
        reference_description="common final graph-compatible goal",
    )
    method = (
        "always_on_reference_conditioned_chart_union_atlas_deepc"
        if always_on
        else "reference_conditioned_future_output_atlas_deepc"
    )
    controller_description = (
        "Always-on Atlas-DeePC with a Q-weighted Y_f-selected chart union"
        if always_on
        else "Atlas-DeePC with triggered chart-local Q-weighted Y_f modes"
    )
    summary = {
        **asdict(control_config),
        "method": method,
        "contextual_activation_mode": activation_mode,
        "future_output_atlas": asdict(future_config),
        "future_mode_triggered": bool(not always_on and trigger_step >= 0),
        "future_mode_trigger_step": int(trigger_step),
        "future_mode_activation_step": int(0 if always_on else trigger_step),
        "future_mode_usage_fraction": float(np.mean(arrays["future_mode_active"])),
        "future_chart_pair_usage_fraction": float(
            np.mean(arrays["future_chart"] != arrays["chart"])
        ),
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
        "statuses": dict(Counter(arrays["status"].tolist())),
        "controller": controller_description,
        "graph_routing_used": True,
        "qp_count": control_config.steps,
        "optimized_coefficient_dimension": control_config.svd_rank,
        "online_hankel_columns_scored": 0,
        "online_nearest_trajectory_retrieval": False,
        "online_svd_rebuild": False,
        "fitted_dynamics_model": False,
        "use_ik_ref": False,
        "feedback_blend": 0.0,
        "feedback_fallback": False,
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
        **timing_summary(arrays["future_scoring_time_s"], "future_scoring"),
        **timing_summary(arrays["matrix_preparation_time_s"], "matrix_preparation"),
        **timing_summary(arrays["solve_time_s"], "solve"),
        **timing_summary(arrays["control_compute_time_s"], "control_compute"),
        **goal_metadata,
    }
    if save_path is not None:
        np.savez(
            save_path,
            **arrays,
            final_goal=goal,
            summary=np.array(json.dumps(summary), dtype=object),
        )
    return summary


_H = None
_ATLAS = None
_ARTIFACT = None
_CONTROL_CONFIG = None
_FUTURE_CONFIG = None
_OUTPUT_DIR = None
_ACTIVATION_MODE = None


def initialize_worker(
    data_path,
    atlas_path,
    artifact_path,
    control_config,
    future_config,
    output_dir,
    activation_mode,
):
    global _H, _ATLAS, _ARTIFACT, _CONTROL_CONFIG, _FUTURE_CONFIG, _OUTPUT_DIR
    global _ACTIVATION_MODE
    _H = np.load(data_path)
    _ATLAS = routed.load_atlas(Path(atlas_path))
    _ARTIFACT = load_future_output_atlas(Path(artifact_path))
    _CONTROL_CONFIG = routed.RoutedDeePCConfig(**control_config)
    _FUTURE_CONFIG = FutureOutputAtlasConfig(**future_config)
    _OUTPUT_DIR = Path(output_dir)
    _ACTIVATION_MODE = activation_mode


def run_worker(seed: int) -> dict:
    config = routed.RoutedDeePCConfig(
        **{**asdict(_CONTROL_CONFIG), "seed": int(seed)}
    )
    return run_future_output_atlas(
        _H,
        _ATLAS,
        _ARTIFACT,
        config,
        _FUTURE_CONFIG,
        save_path=_OUTPUT_DIR / f"seed{seed}_control_results.npz",
        activation_mode=_ACTIVATION_MODE,
    )


def _load_summary(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return json.loads(data["summary"].item())


def run_panel(
    seeds,
    baseline_dir: Path,
    output_dir: Path,
    future_config: FutureOutputAtlasConfig,
    data_path: Path = DEFAULT_DATA_PATH,
    workers: int = 1,
    overwrite: bool = False,
    activation_mode: str = "triggered",
) -> dict:
    if activation_mode not in {"triggered", "always_on"}:
        raise ValueError(f"Unknown activation_mode: {activation_mode}")
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_payload = json.loads((baseline_dir / "summary.json").read_text())
    control_config = baseline_config(baseline_payload)
    H = np.load(data_path, mmap_mode="r")
    atlas_path = baseline_dir / "behavioral_atlas.npz"
    atlas = routed.load_atlas(atlas_path)
    artifact_path = output_dir / "future_output_atlas.npz"
    if overwrite or not artifact_path.exists():
        started = time.perf_counter()
        artifact = build_future_output_atlas(H, atlas, control_config, future_config)
        save_future_output_atlas(artifact_path, artifact)
        offline_seconds = time.perf_counter() - started
    else:
        artifact = load_future_output_atlas(artifact_path)
        saved = json.loads(artifact["future_config"].item())
        if saved != asdict(future_config):
            raise RuntimeError(f"Incompatible future-output atlas: {artifact_path}")
        offline_seconds = 0.0

    results = {}
    if not overwrite:
        for seed in seeds:
            path = output_dir / f"seed{seed}_control_results.npz"
            if path.exists():
                summary = _load_summary(path)
                if summary.get("future_output_atlas") != asdict(future_config):
                    raise RuntimeError(f"Incompatible saved rollout: {path}")
                saved_mode = summary.get("contextual_activation_mode", "triggered")
                if saved_mode != activation_mode:
                    raise RuntimeError(f"Incompatible activation mode: {path}")
                results[int(seed)] = summary
    pending = [int(seed) for seed in seeds if int(seed) not in results]
    initargs = (
        str(data_path),
        str(atlas_path),
        str(artifact_path),
        asdict(control_config),
        asdict(future_config),
        str(output_dir),
        activation_mode,
    )
    if workers == 1:
        initialize_worker(*initargs)
        for seed in pending:
            results[seed] = run_worker(seed)
            print(
                f"seed={seed} success="
                f"{results[seed]['success_30'] and results[seed]['success_100']} "
                f"final100={1000 * results[seed]['mean_last_100']:.3f} mm",
                flush=True,
            )
    elif pending:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=initialize_worker,
            initargs=initargs,
        ) as executor:
            futures = {executor.submit(run_worker, seed): seed for seed in pending}
            for future in as_completed(futures):
                seed = futures[future]
                results[seed] = future.result()
                print(
                    f"seed={seed} success="
                    f"{results[seed]['success_30'] and results[seed]['success_100']} "
                    f"final100={1000 * results[seed]['mean_last_100']:.3f} mm",
                    flush=True,
                )

    method = (
        "always_on_reference_conditioned_chart_union_atlas_deepc"
        if activation_mode == "always_on"
        else "reference_conditioned_future_output_atlas_deepc"
    )
    payload = {
        "method": method,
        "contextual_activation_mode": activation_mode,
        "baseline_source": str(baseline_dir),
        "config": asdict(control_config),
        "future_output_atlas": asdict(future_config),
        "offline_build_seconds_this_run": offline_seconds,
        "fairness_audit": audit_iid_library(H, eval_seeds=seeds),
        "results": [results[int(seed)] for seed in seeds],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-stop", type=int, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--modes-per-chart", type=int, default=8)
    parser.add_argument("--score-margin", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_panel(
        list(range(args.seed_start, args.seed_stop + 1)),
        args.baseline_dir,
        args.output_dir,
        FutureOutputAtlasConfig(
            modes_per_chart=args.modes_per_chart,
            score_margin=args.score_margin,
        ),
        workers=args.workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
