
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import heapq
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import cvxpy as cp
import gymnasium as gym
import numpy as np

from atlas_deepc.control import expert_pipeline_search as pipe
from atlas_deepc.atlas.kmeans import fit_kmeans, nearest_centroid
from atlas_deepc.common.reacher_data import audit_iid_library
from atlas_deepc.common.reacher_config import normalize_saved_config
from atlas_deepc.atlas.goal_reference import chart_compatible_goal


WINDOWS_PER_EPISODE = 200 - pipe.L + 1


@dataclass
class RoutedDeePCConfig:
    """Current settings plus legacy fields needed to read saved protocol dictionaries."""

    clusters: int = 48
    cluster_seed: int = 11
    clustering_method: str = "minibatch_kmeans"
    clustering_feature_mode: str = "plant_joint_configuration"
    # Serialized compatibility only; alternate clustering implementations are retired.
    clustering_subspace_rank: int = 6
    clustering_max_iter: int = 20
    clustering_configuration_weight: float = 1.0
    clustering_velocity_weight: float = 1.0
    clustering_fingertip_weight: float = 1.0
    formulation: str = "svd_current"
    overlap_charts: int = 2
    svd_rank: int = 60
    svd_bound_scale: float = 3.0
    negative_mass: float = 0.2
    edge_count_threshold: int = 3
    target_switch_distance: float = 0.07
    latch_target_reference: bool = True
    target_matrix_mode: str = "current"
    terminal_overlap_distance: float = 0.01
    terminal_overlap_dwell: int = 5
    q_xy: float = 100000.0
    q_dq: float = 20.0
    q_data_goal_joint: float = 5000.0
    r: float = 0.0001
    lam_g: float = 0.0
    lam_ini: float = 100000.0
    lam_du: float = 0.0
    u_limit: float = 1.0
    goal_chart_candidates: int = 400
    goal_charts: int = 2
    goal_source_target_min_distance: float = 0.06
    goal_branch_selection: str = "compatible"
    goal_chart_override: int | None = None
    steps: int = 300
    seed: int = 1

    @classmethod
    def from_dict(cls, values):
        return cls(**normalize_saved_config(values))


def behavioral_features(
    H,
    feature_mode="plant_joint_configuration",
    *,
    standardize=True,
):
    _, _, Y_p, _ = pipe.partition_hankel_rows(H)
    if feature_mode != "plant_joint_configuration":
        raise ValueError(f"Unsupported clustering feature mode: {feature_mode}")
    history = Y_p.reshape(pipe.T_INI, pipe.P, H.shape[1])
    configuration_rows = pipe.reacher_joint_configuration_rows()
    raw = history[-1, configuration_rows]
    mean = raw.mean(axis=1) if standardize else np.zeros(raw.shape[0])
    scale = np.maximum(raw.std(axis=1), 1e-8) if standardize else np.ones(raw.shape[0])
    normalized = (raw - mean[:, None]) / scale[:, None]
    return normalized.T, mean, scale


def online_behavioral_feature(
    y_ini, mean, scale,
    feature_mode="plant_joint_configuration",
):
    if feature_mode != "plant_joint_configuration":
        raise ValueError(f"Unsupported clustering feature mode: {feature_mode}")
    history = y_ini[:, 0].reshape(pipe.T_INI, pipe.P)
    configuration_rows = pipe.reacher_joint_configuration_rows()
    raw = history[-1, configuration_rows]
    mean, scale = np.asarray(mean), np.asarray(scale)
    if mean.shape == scale.shape == raw.shape:
        return (raw - mean) / scale
    # Existing atlases stored four all-zero input coordinates. Keep their layout
    # when localizing against them, without reading or weighting the input history.
    padding = pipe.M * pipe.T_INI
    if mean.shape == scale.shape == (padding + raw.size,):
        return np.r_[np.zeros(padding), (raw - mean[padding:]) / scale[padding:]]
    raise ValueError("Atlas feature statistics do not match joint-configuration features")


def assign_charts(centers, features):
    return nearest_centroid(centers, features)


def assign_atlas_charts(atlas, features):
    features = np.asarray(features, dtype=float)
    if features.ndim == 1:
        features = features[None, :]
    method = str(np.asarray(atlas.get("clustering_method", "minibatch_kmeans")).item())
    delta = features[:, None, :] - atlas["centers"][None, :, :]
    if method != "minibatch_kmeans":
        raise ValueError(f"Unsupported clustering method: {method}")
    scores = np.mean(delta * delta, axis=2)
    return np.argmin(scores, axis=1)


def fit_behavioral_atlas(H, config, *, standardize=True):
    features, feature_mean, feature_scale = behavioral_features(
        H,
        config.clustering_feature_mode,
        standardize=standardize,
    )
    if config.clustering_method != "minibatch_kmeans":
        raise ValueError(f"Unsupported clustering method: {config.clustering_method}")
    labels, centers = fit_kmeans(features, config.clusters, config.cluster_seed)
    counts = np.zeros((config.clusters, config.clusters), dtype=int)
    for episode in range(H.shape[1] // WINDOWS_PER_EPISODE):
        start = episode * WINDOWS_PER_EPISODE
        stop = start + WINDOWS_PER_EPISODE
        for source, destination in zip(labels[start:stop - 1], labels[start + 1:stop]):
            counts[source, destination] += 1

    Y = H[pipe.M * pipe.L:].reshape(pipe.L, pipe.P, H.shape[1])
    medoid_states = []
    for cluster_id in range(config.clusters):
        idx = np.where(labels == cluster_id)[0]
        assigned_error = features[idx] - centers[cluster_id]
        error = np.mean(assigned_error * assigned_error, axis=1)
        medoid_index = int(idx[np.argmin(error)])
        state = Y[pipe.T_INI - 1, :, medoid_index].copy()
        state[4:6] = 0.0
        medoid_states.append(state)
    center_delta = centers[:, None, :] - centers[None, :, :]
    center_distance = np.sqrt(np.mean(center_delta * center_delta, axis=2))
    return {
        "labels": labels,
        "centers": centers,
        "clustering_method": np.array(config.clustering_method),
        "clustering_feature_mode": np.array(config.clustering_feature_mode),
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "transition_counts": counts,
        "center_distance": center_distance,
        "medoid_states": np.asarray(medoid_states),
    }


from atlas_deepc.common.atlas_io import save_atlas, load_atlas


def shortest_chart_path(atlas, source, destination, count_threshold):
    if source == destination:
        return [source]
    counts = atlas["transition_counts"]
    distances = [np.inf] * counts.shape[0]
    previous = [-1] * counts.shape[0]
    routing_edge_costs = atlas.get("routing_edge_costs")
    distances[source] = 0.0
    queue = [(0.0, source)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        if node == destination:
            break
        for neighbor in np.where(counts[node] >= count_threshold)[0]:
            neighbor = int(neighbor)
            if neighbor == node:
                continue
            if routing_edge_costs is None:
                reliability = np.sqrt(float(counts[node, neighbor]))
                edge_cost = atlas["center_distance"][node, neighbor] / reliability
            else:
                edge_cost = float(routing_edge_costs[node, neighbor])
                if not np.isfinite(edge_cost):
                    continue
            candidate = distance + edge_cost
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))
    if previous[destination] < 0:
        return []
    path = [destination]
    while path[-1] != source:
        path.append(previous[path[-1]])
    return path[::-1]


def chart_matrix(H, atlas, chart_ids, config, geometry="affine"):
    if geometry not in ("affine", "linear"):
        raise ValueError(f"Unknown chart geometry: {geometry}")
    selected = np.where(np.isin(atlas["labels"], np.asarray(chart_ids, dtype=int)))[0]
    X = H[:, selected]
    if config.formulation.startswith("raw"):
        coefficient_floor = -config.negative_mass / max(selected.size, 1)
        return {
            "offset": np.zeros((H.shape[0], 1)),
            "basis": X,
            "g_sum": 1.0,
            "g_min": coefficient_floor,
        }, selected

    center = X.mean(axis=1, keepdims=True) if geometry == "affine" else np.zeros((X.shape[0], 1))
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
    }, selected


def precompute_chart_matrix_cache(H, atlas, config, geometry="affine"):
    """Build the reduced basis for every single chart outside online timing."""
    chart_count = int(atlas["centers"].shape[0])
    return {
        (chart,): chart_matrix(H, atlas, (chart,), config, geometry=geometry)
        for chart in range(chart_count)
    }


def run_routed_deepc(
    H,
    atlas,
    config,
    save_path=None,
    summary_metadata=None,
    matrix_cache=None,
    goal_policy=None,
    reference_mode="graph",
    chart_selector=None,
    localization_history_length=None,
):
    if reference_mode not in ("graph", "terminal"):
        raise ValueError(f"Unknown reference mode: {reference_mode}")
    history_length = pipe.T_INI if localization_history_length is None else int(localization_history_length)
    if history_length < pipe.T_INI:
        raise ValueError("Localization history cannot be shorter than the DeePC initial window")
    # Atlas reduced bases are offline artifacts. Build them before creating the
    # controller/environment; only cached lookup is part of the online loop.
    matrix_cache = (
        precompute_chart_matrix_cache(H, atlas, config)
        if matrix_cache is None
        else dict(matrix_cache)
    )
    q_diagonal = [
        config.q_data_goal_joint,
        config.q_data_goal_joint,
        config.q_data_goal_joint,
        config.q_data_goal_joint,
        config.q_dq,
        config.q_dq,
        config.q_xy,
        config.q_xy,
    ]
    if pipe.P == 10:
        q_diagonal.extend([config.q_data_goal_joint, config.q_data_goal_joint])
    elif pipe.P != 8:
        raise ValueError(f"Unsupported Reacher output dimension P={pipe.P}.")
    Q = np.diag(q_diagonal)
    R = config.r * np.eye(pipe.M)
    pipe.validate_objective_options(config.lam_g, config.lam_du)
    controller = pipe.DeePCController(
        Q,
        R,
        config.lam_ini,
        config.u_limit,
        cp.CLARABEL,
    )
    env = gym.make(pipe.REACHER_ENV_ID, max_episode_steps=config.steps + history_length + 5)
    obs, _ = env.reset(seed=config.seed)
    target_xy = np.asarray(obs[4:6], dtype=float).copy()
    initial_y = pipe.reacher_output_from_obs(obs)
    goal_start = time.perf_counter()
    if goal_policy is None:
        goal, goal_metadata = chart_compatible_goal(H, target_xy, initial_y, config)
        goal_index = int(goal_metadata["data_goal_index"])
        goal_chart = int(atlas["labels"][goal_index])
    else:
        goal_policy.prepare(target_xy)
        goal, goal_metadata, goal_chart = None, {}, -1
    goal_setup_time = time.perf_counter() - goal_start

    u_history = []
    y_history = []
    for _ in range(history_length):
        action = np.zeros(pipe.M)
        obs, *_ = env.step(action)
        u_history.append(action)
        y_history.append(pipe.reacher_output_from_obs(obs))

    target_reference_latched = False
    terminal_overlap_count = 0
    terminal_overlap_latched = False
    terminal_overlap_trigger_step = -1
    records = {
        "y": [], "u": [], "pred": [], "ref": [], "distance": [],
        "prediction_error_xy": [], "status": [], "chart": [], "goal_chart": [],
        "next_chart": [], "path_length": [], "matrix_columns": [], "g_min": [],
        "localization_time_s": [], "routing_time_s": [],
        "matrix_preparation_time_s": [], "solve_time_s": [],
        "control_compute_time_s": [],
        "terminal_overlap_active": [],
        "goal_selection_time_s": [], "goal_index": [], "goal_route_cost": [],
    }
    for step in range(config.steps):
        control_start = time.perf_counter()
        u_ini = np.asarray(u_history[-pipe.T_INI:], dtype=float).reshape(pipe.M * pipe.T_INI, 1)
        y_ini = np.asarray(y_history[-pipe.T_INI:], dtype=float).reshape(pipe.P * pipe.T_INI, 1)
        localization_start = time.perf_counter()
        if chart_selector is None:
            feature = online_behavioral_feature(
                y_ini,
                atlas["feature_mean"],
                atlas["feature_scale"],
                config.clustering_feature_mode,
            )
            current_chart = int(assign_atlas_charts(atlas, feature)[0])
        else:
            u_window = np.asarray(u_history[-history_length:], dtype=float).reshape(-1)
            y_window = np.asarray(y_history[-history_length:], dtype=float).reshape(-1)
            current_chart = int(chart_selector.select(np.r_[u_window, y_window]))
        localization_time = time.perf_counter() - localization_start
        goal_selection_time = 0.0
        goal_route_cost = np.nan
        if goal_policy is not None:
            selection_start = time.perf_counter()
            goal, goal_metadata, goal_chart, goal_route_cost = goal_policy.select(current_chart)
            goal_selection_time = time.perf_counter() - selection_start
        routing_start = time.perf_counter()
        if reference_mode == "terminal":
            path = [current_chart]
        elif goal_policy is None:
            path = shortest_chart_path(
                atlas, current_chart, goal_chart, config.edge_count_threshold
            )
        else:
            path = goal_policy.path(current_chart, goal_chart)
        if not path:
            path = [current_chart]
        next_chart = path[1] if len(path) > 1 else current_chart
        distance = float(np.linalg.norm(y_history[-1][6:8] - target_xy))
        if distance < config.target_switch_distance:
            target_reference_latched = True
        if config.target_matrix_mode == "dwell_current_goal_overlap":
            if distance < config.terminal_overlap_distance:
                terminal_overlap_count += 1
            else:
                terminal_overlap_count = 0
            if (
                not terminal_overlap_latched
                and terminal_overlap_count >= config.terminal_overlap_dwell
            ):
                terminal_overlap_latched = True
                terminal_overlap_trigger_step = step
        routing_time = time.perf_counter() - routing_start
        use_target = reference_mode == "terminal" or (
            target_reference_latched if config.latch_target_reference else distance < config.target_switch_distance
        )
        active_reference = goal if use_target or current_chart == goal_chart else atlas["medoid_states"][next_chart]
        reference = np.tile(active_reference.reshape(pipe.P, 1), (pipe.T_F, 1))

        if use_target and config.target_matrix_mode == "goal":
            active_charts = (goal_chart,)
        elif use_target and config.target_matrix_mode == "current_goal_overlap":
            active_charts = tuple(sorted(set((current_chart, goal_chart))))
        elif use_target and terminal_overlap_latched:
            active_charts = tuple(sorted(set((current_chart, goal_chart))))
        elif config.formulation.endswith("overlap") and current_chart != next_chart:
            active_charts = tuple(sorted((current_chart, next_chart)))
        else:
            active_charts = (current_chart,)
        matrix_start = time.perf_counter()
        if active_charts not in matrix_cache:
            matrix_cache[active_charts] = chart_matrix(H, atlas, active_charts, config)
        matrix, selected = matrix_cache[active_charts]
        matrix_preparation_time = time.perf_counter() - matrix_start
        solve_start = time.perf_counter()
        u_solution, y_solution, g_solution, status, _ = controller.solve(
            matrix, u_ini, y_ini, reference
        )
        solve_time = time.perf_counter() - solve_start
        if u_solution is None:
            action = np.zeros(pipe.M)
            prediction = np.full(pipe.P, np.nan)
            minimum_g = np.nan
        else:
            action = np.clip(u_solution[:pipe.M, 0], -config.u_limit, config.u_limit)
            prediction = np.asarray(y_solution[:pipe.P, 0]).reshape(pipe.P)
            minimum_g = float(np.min(g_solution))
        control_compute_time = time.perf_counter() - control_start
        obs, *_ = env.step(action)
        y_now = pipe.reacher_output_from_obs(obs)
        u_history.append(action.copy())
        y_history.append(y_now.copy())
        records["y"].append(y_now)
        records["u"].append(action)
        records["pred"].append(prediction)
        records["ref"].append(active_reference)
        records["distance"].append(float(np.linalg.norm(y_now[6:8] - target_xy)))
        records["prediction_error_xy"].append(float(np.linalg.norm(prediction[6:8] - y_now[6:8])))
        records["status"].append(status)
        records["chart"].append(current_chart)
        records["goal_chart"].append(goal_chart)
        records["next_chart"].append(next_chart)
        records["path_length"].append(len(path))
        records["matrix_columns"].append(selected.size)
        records["g_min"].append(minimum_g)
        records["localization_time_s"].append(localization_time)
        records["routing_time_s"].append(routing_time)
        records["matrix_preparation_time_s"].append(matrix_preparation_time)
        records["solve_time_s"].append(solve_time)
        records["control_compute_time_s"].append(control_compute_time)
        records["terminal_overlap_active"].append(terminal_overlap_latched)
        records["goal_selection_time_s"].append(goal_selection_time)
        records["goal_index"].append(int(goal_metadata["data_goal_index"]))
        records["goal_route_cost"].append(goal_route_cost)
    env.close()

    for key in records:
        records[key] = np.asarray(records[key], dtype=object if key == "status" else None)
    distance = records["distance"]
    tail_30 = distance[-30:]
    tail_100 = distance[-min(100, distance.size):]
    reference_cost = pipe.closed_loop_control_cost(
        records["y"], records["u"], records["ref"], Q, R
    )
    final_goal_cost = pipe.closed_loop_control_cost(
        records["y"],
        records["u"],
        goal,
        Q,
        R,
        reference_description="common final graph-compatible goal",
    )
    summary = {
        **asdict(config),
        "reference_mode": reference_mode,
        "localization_history_length": history_length,
        "goal_setup_time_ms": float(1000 * goal_setup_time),
        "median_goal_selection_time_ms": float(1000 * np.median(records["goal_selection_time_s"])),
        "goal_sample_switches": int(np.count_nonzero(np.diff(records["goal_index"]))),
        "goal_chart_switches": int(np.count_nonzero(np.diff(records["goal_chart"]))),
        "dynamic_terminal_reference": goal_policy is not None,
        "minimum": float(distance.min()),
        "minimum_step": int(distance.argmin()),
        "last": float(distance[-1]),
        "mean_last_30": float(tail_30.mean()),
        "max_last_30": float(tail_30.max()),
        "mean_last_100": float(tail_100.mean()),
        "max_last_100": float(tail_100.max()),
        "success_30": bool(tail_30.mean() < 0.01 and tail_30.max() < 0.02),
        "success_100": bool(tail_100.mean() < 0.01 and tail_100.max() < 0.02),
        "mean_prediction_error_xy": float(np.nanmean(records["prediction_error_xy"])),
        "median_localization_time_ms": float(1000 * np.median(records["localization_time_s"])),
        "p95_localization_time_ms": float(1000 * np.percentile(records["localization_time_s"], 95)),
        "median_routing_time_ms": float(1000 * np.median(records["routing_time_s"])),
        "p95_routing_time_ms": float(1000 * np.percentile(records["routing_time_s"], 95)),
        "median_matrix_preparation_time_ms": float(1000 * np.median(records["matrix_preparation_time_s"])),
        "p95_matrix_preparation_time_ms": float(1000 * np.percentile(records["matrix_preparation_time_s"], 95)),
        "median_solve_time_ms": float(1000 * np.median(records["solve_time_s"])),
        "p95_solve_time_ms": float(1000 * np.percentile(records["solve_time_s"], 95)),
        "median_control_compute_time_ms": float(1000 * np.median(records["control_compute_time_s"])),
        "p95_control_compute_time_ms": float(1000 * np.percentile(records["control_compute_time_s"], 95)),
        "p99_control_compute_time_ms": float(1000 * np.percentile(records["control_compute_time_s"], 99)),
        "max_control_compute_time_ms": float(1000 * np.max(records["control_compute_time_s"])),
        "charts_visited": int(np.unique(records["chart"]).size),
        "statuses": {str(s): int(np.sum(records["status"] == s)) for s in np.unique(records["status"])},
        "controller": "DeePC behavioral column combination",
        "fitted_dynamics_model": False,
        "online_nearest_trajectory_retrieval": False,
        "use_ik_ref": False,
        "feedback_blend": 0.0,
        "feedback_fallback": False,
        "chart_basis_precomputed_offline": True,
        "control_compute_excludes_offline_chart_svd": True,
        "terminal_overlap_triggered": bool(terminal_overlap_latched),
        "terminal_overlap_trigger_step": int(terminal_overlap_trigger_step),
        "terminal_overlap_usage_fraction": float(
            np.mean(records["terminal_overlap_active"].astype(bool))
        ),
        **reference_cost,
        "closed_loop_reference_cost": reference_cost["closed_loop_control_cost"],
        "closed_loop_reference_tracking_cost": reference_cost["closed_loop_tracking_cost"],
        "mean_closed_loop_reference_stage_cost": reference_cost["mean_closed_loop_stage_cost"],
        "closed_loop_final_goal_cost": final_goal_cost["closed_loop_control_cost"],
        "closed_loop_final_goal_tracking_cost": final_goal_cost["closed_loop_tracking_cost"],
        "mean_closed_loop_final_goal_stage_cost": final_goal_cost["mean_closed_loop_stage_cost"],
        "closed_loop_final_goal_cost_definition": final_goal_cost["closed_loop_cost_definition"],
        **goal_metadata,
        **({} if summary_metadata is None else summary_metadata),
    }
    if save_path is not None:
        np.savez(
            save_path,
            **records,
            final_goal=goal,
            summary=np.array(json.dumps(summary), dtype=object),
        )
    return summary
