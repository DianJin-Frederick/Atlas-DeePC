"""Offline contextual-cluster Atlas-DeePC.

Contextual-aware DPC ranks every Hankel column from the measured past and the
future reference, then rebuilds a local SVD online.  This module vector-quantizes
that joint Y_p/Y_f query space offline.  Each contextual prototype owns a fixed
top-N neighborhood of complete input-output Hankel columns and a cached
fixed-rank affine basis. Online control scores only prototypes and retrieves
one basis.

Two offline atlases are supported:

* global_vq: contextual prototypes cover the complete offline library;
* hierarchical: each physical Y_p chart owns several contextual subcharts.

Graph routing, references, goal construction, DeePC costs, constraints, solver,
and success criteria are unchanged.
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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

import gymnasium as gym
import numpy as np
from sklearn.cluster import MiniBatchKMeans

import expert_pipeline_search as pipe
import routed_deepc_experiment as routed
from contextual_chart_expansion_experiment import baseline_config, output_cost
from local_affine_atlas_experiment import audit_iid_library
from online_selected_deepc_comparison import (
    build_controller,
    prepare_contextual_library,
    selected_affine_basis,
    timing_summary,
)
from stitched_atlas_mpc_experiment import chart_compatible_goal


DEFAULT_DATA_PATH = Path("data/select_iid_offline_hankel.npy")


@dataclass(frozen=True)
class ContextualClusterConfig:
    architecture: str = "global_vq"
    contextual_clusters: int = 128
    modes_per_chart: int = 4
    support_columns: int = 200
    cluster_seed: int = 73
    activation_mode: str = "always_on"
    error_threshold: float = 0.01
    stall_window: int = 15
    minimum_progress: float = 0.001
    past_weight: float = 1.0
    future_weight: float = 1.0
    distance_mode: str = "standardized_contextual"
    candidate_solve_count: int = 1
    score_margin: float = 0.0
    candidate_scope: str = "global"
    edge_count_threshold: int = 3
    reference_mode: str = "graph_routed"


_ARTIFACT_BUILD_FIELDS = (
    "architecture",
    "contextual_clusters",
    "modes_per_chart",
    "support_columns",
    "cluster_seed",
)


def _same_offline_atlas(left: dict, right: dict) -> bool:
    """Return whether two configs induce the same cached offline artifact.

    Activation, scoring weights, and hysteresis only affect online selection.
    Keeping them out of the compatibility check lets controlled sweeps reuse
    exactly the same prototypes, supports, and bases.
    """
    return all(left.get(key) == right.get(key) for key in _ARTIFACT_BUILD_FIELDS)


def _config_with_defaults(values: dict) -> dict:
    """Normalize artifacts written before selection-only fields were added."""
    return {**asdict(ContextualClusterConfig()), **values}


def _contextual_embedding(
    library: dict[str, np.ndarray], Q: np.ndarray
) -> np.ndarray:
    """Give past and Q-weighted future blocks equal average squared scale."""
    past = library["Y_p_normalized"].T
    future = library["Y_f_normalized"].T
    q_diagonal = np.diag(Q)
    sqrt_q = np.sqrt(np.tile(q_diagonal, pipe.T_F))
    past = past / np.sqrt(past.shape[1])
    future = future * sqrt_q[None, :] / np.sqrt(np.sum(sqrt_q**2))
    return np.hstack([past, future])


def _fit_model(features: np.ndarray, clusters: int, seed: int) -> MiniBatchKMeans:
    model = MiniBatchKMeans(
        n_clusters=clusters,
        random_state=seed,
        n_init=5,
        batch_size=2048,
        max_iter=400,
        reassignment_ratio=0.001,
    )
    model.fit(features)
    return model


def _nearest_support(
    distances: np.ndarray, candidates: np.ndarray, count: int
) -> np.ndarray:
    count = min(int(count), int(candidates.size))
    local = np.argpartition(distances, count - 1)[:count]
    return np.sort(candidates[local])


def build_contextual_cluster_atlas(
    H: np.ndarray,
    physical_atlas: dict,
    control_config: routed.RoutedDeePCConfig,
    cluster_config: ContextualClusterConfig,
) -> dict[str, np.ndarray]:
    if cluster_config.architecture not in {"global_vq", "hierarchical"}:
        raise ValueError(f"Unknown architecture: {cluster_config.architecture}")
    if cluster_config.activation_mode not in {"always_on", "triggered"}:
        raise ValueError(f"Unknown activation mode: {cluster_config.activation_mode}")

    library = prepare_contextual_library(H)
    Q = output_cost(control_config)
    embedding = _contextual_embedding(library, Q)
    physical_labels = np.asarray(physical_atlas["labels"], dtype=int)
    chart_count = int(physical_atlas["centers"].shape[0])

    supports = []
    prototype_past = []
    prototype_future = []
    prototype_chart = []
    assignment_counts = []

    if cluster_config.architecture == "global_vq":
        model = _fit_model(
            embedding,
            cluster_config.contextual_clusters,
            cluster_config.cluster_seed,
        )
        assignment = model.predict(embedding)
        distances = model.transform(embedding)
        all_indices = np.arange(H.shape[1])
        for cluster in range(cluster_config.contextual_clusters):
            support = _nearest_support(
                distances[:, cluster], all_indices, cluster_config.support_columns
            )
            supports.append(support)
            prototype_past.append(library["Y_p_normalized"][:, support].mean(axis=1))
            prototype_future.append(library["Y_f_normalized"][:, support].mean(axis=1))
            chart_histogram = np.bincount(physical_labels[support], minlength=chart_count)
            prototype_chart.append(int(np.argmax(chart_histogram)))
            assignment_counts.append(int(np.sum(assignment == cluster)))
    else:
        for chart in range(chart_count):
            members = np.flatnonzero(physical_labels == chart)
            model = _fit_model(
                embedding[members],
                cluster_config.modes_per_chart,
                cluster_config.cluster_seed + chart,
            )
            assignment = model.predict(embedding[members])
            distances = model.transform(embedding[members])
            for mode in range(cluster_config.modes_per_chart):
                support = _nearest_support(
                    distances[:, mode], members, cluster_config.support_columns
                )
                supports.append(support)
                prototype_past.append(library["Y_p_normalized"][:, support].mean(axis=1))
                prototype_future.append(library["Y_f_normalized"][:, support].mean(axis=1))
                prototype_chart.append(chart)
                assignment_counts.append(int(np.sum(assignment == mode)))

    offsets, bases, g_mins, g_maxs = [], [], [], []
    for support in supports:
        matrix = selected_affine_basis(H, support, control_config)
        offsets.append(matrix["offset"])
        bases.append(matrix["basis"])
        g_mins.append(matrix["g_min"])
        g_maxs.append(matrix["g_max"])

    physical_cache = routed.precompute_chart_matrix_cache(
        H, physical_atlas, control_config
    )
    physical_offsets, physical_bases, physical_g_mins, physical_g_maxs = [], [], [], []
    physical_columns = []
    for chart in range(chart_count):
        matrix, selected = physical_cache[(chart,)]
        physical_offsets.append(matrix["offset"])
        physical_bases.append(matrix["basis"])
        physical_g_mins.append(matrix["g_min"])
        physical_g_maxs.append(matrix["g_max"])
        physical_columns.append(selected.size)

    return {
        "prototype_past": np.asarray(prototype_past),
        "prototype_future": np.asarray(prototype_future),
        "prototype_chart": np.asarray(prototype_chart, dtype=int),
        "assignment_counts": np.asarray(assignment_counts, dtype=int),
        "support_indices": np.asarray(supports, dtype=int),
        "matrix_offsets": np.asarray(offsets),
        "matrix_bases": np.asarray(bases),
        "matrix_g_mins": np.asarray(g_mins),
        "matrix_g_maxs": np.asarray(g_maxs),
        "physical_matrix_offsets": np.asarray(physical_offsets),
        "physical_matrix_bases": np.asarray(physical_bases),
        "physical_matrix_g_mins": np.asarray(physical_g_mins),
        "physical_matrix_g_maxs": np.asarray(physical_g_maxs),
        "physical_matrix_columns": np.asarray(physical_columns, dtype=int),
        "output_mean": library["output_mean"],
        "output_scale": library["output_scale"],
        "q_diagonal": np.diag(Q),
        "basis_rank": np.array(control_config.svd_rank, dtype=int),
        "cluster_config": np.array(json.dumps(asdict(cluster_config)), dtype=object),
    }


def truncate_contextual_cluster_atlas(artifact: dict, rank: int) -> dict:
    """Reuse an atlas partition while reducing every cached affine basis rank."""
    basis_keys = ["matrix_bases"]
    if "physical_matrix_bases" in artifact:
        basis_keys.append("physical_matrix_bases")
    available = min(int(artifact[key].shape[2]) for key in basis_keys)
    if rank < 1 or rank > available:
        raise ValueError(
            f"Requested rank {rank}; cached bases provide only "
            f"{available} components"
        )
    reduced = {key: np.asarray(value).copy() for key, value in artifact.items()}
    for key in basis_keys:
        reduced[key] = reduced[key][..., :rank]
    for key in (
        "matrix_g_mins", "matrix_g_maxs",
        "physical_matrix_g_mins", "physical_matrix_g_maxs",
    ):
        if key in reduced:
            reduced[key] = reduced[key][..., :rank, :]
    reduced["basis_rank"] = np.array(rank, dtype=int)
    return reduced


def rebuild_contextual_cluster_atlas_bases(
    H: np.ndarray,
    physical_atlas: dict,
    source_artifact: dict,
    control_config: routed.RoutedDeePCConfig,
    cluster_config: ContextualClusterConfig,
) -> dict:
    """Reuse frozen prototypes/supports while rebuilding cached bases at a rank.

    A rank-60 source artifact cannot be truncated upward. This controlled
    rebuild preserves its contextual prototypes and 200-column supports, then
    recomputes only the contextual and physical affine SVD matrices.
    """
    saved = _config_with_defaults(
        json.loads(source_artifact["cluster_config"].item())
    )
    if not _same_offline_atlas(saved, asdict(cluster_config)):
        raise RuntimeError("Source artifact does not match the requested atlas")

    rebuilt = {
        key: np.asarray(value).copy()
        for key, value in source_artifact.items()
    }
    offsets, bases, g_mins, g_maxs = [], [], [], []
    for support in np.asarray(source_artifact["support_indices"], dtype=int):
        matrix = selected_affine_basis(H, support, control_config)
        offsets.append(matrix["offset"])
        bases.append(matrix["basis"])
        g_mins.append(matrix["g_min"])
        g_maxs.append(matrix["g_max"])
    rebuilt["matrix_offsets"] = np.asarray(offsets)
    rebuilt["matrix_bases"] = np.asarray(bases)
    rebuilt["matrix_g_mins"] = np.asarray(g_mins)
    rebuilt["matrix_g_maxs"] = np.asarray(g_maxs)

    physical_cache = routed.precompute_chart_matrix_cache(
        H, physical_atlas, control_config
    )
    physical_offsets, physical_bases = [], []
    physical_g_mins, physical_g_maxs, physical_columns = [], [], []
    chart_count = int(physical_atlas["centers"].shape[0])
    for chart in range(chart_count):
        matrix, selected = physical_cache[(chart,)]
        physical_offsets.append(matrix["offset"])
        physical_bases.append(matrix["basis"])
        physical_g_mins.append(matrix["g_min"])
        physical_g_maxs.append(matrix["g_max"])
        physical_columns.append(selected.size)
    rebuilt["physical_matrix_offsets"] = np.asarray(physical_offsets)
    rebuilt["physical_matrix_bases"] = np.asarray(physical_bases)
    rebuilt["physical_matrix_g_mins"] = np.asarray(physical_g_mins)
    rebuilt["physical_matrix_g_maxs"] = np.asarray(physical_g_maxs)
    rebuilt["physical_matrix_columns"] = np.asarray(
        physical_columns, dtype=int
    )
    rebuilt["basis_rank"] = np.array(control_config.svd_rank, dtype=int)
    rebuilt["cluster_config"] = np.array(
        json.dumps(asdict(cluster_config)), dtype=object
    )
    return rebuilt


def save_contextual_cluster_atlas(path: Path, artifact: dict) -> None:
    np.savez_compressed(path, **artifact)


def load_contextual_cluster_atlas(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def matrix_cache(artifact: dict) -> list[dict[str, np.ndarray]]:
    return [
        {
            "offset": artifact["matrix_offsets"][index],
            "basis": artifact["matrix_bases"][index],
            "g_min": artifact["matrix_g_mins"][index],
            "g_max": artifact["matrix_g_maxs"][index],
        }
        for index in range(artifact["prototype_past"].shape[0])
    ]


def physical_matrix_cache_from_artifact(artifact: dict) -> dict:
    return {
        (chart,): (
            {
                "offset": artifact["physical_matrix_offsets"][chart],
                "basis": artifact["physical_matrix_bases"][chart],
                "g_min": artifact["physical_matrix_g_mins"][chart],
                "g_max": artifact["physical_matrix_g_maxs"][chart],
            },
            np.arange(int(artifact["physical_matrix_columns"][chart])),
        )
        for chart in range(artifact["physical_matrix_offsets"].shape[0])
    }


def _normalized_query(
    y_ini: np.ndarray, reference: np.ndarray, artifact: dict
) -> tuple[np.ndarray, np.ndarray]:
    mean = artifact["output_mean"]
    scale = artifact["output_scale"]
    past = (
        (np.asarray(y_ini).reshape(pipe.T_INI, pipe.P) - mean[None, :])
        / scale[None, :]
    ).reshape(-1)
    future = (
        (np.asarray(reference).reshape(pipe.T_F, pipe.P) - mean[None, :])
        / scale[None, :]
    ).reshape(-1)
    return past, future


def select_contextual_cluster(
    y_ini: np.ndarray,
    reference: np.ndarray,
    current_chart: int,
    physical_atlas: dict,
    artifact: dict,
    config: ContextualClusterConfig,
    previous_cluster: int | None = None,
    next_chart: int | None = None,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    past_query, future_query = _normalized_query(y_ini, reference, artifact)
    prototype_chart = artifact["prototype_chart"]
    if config.candidate_scope not in {
        "global", "current_chart", "current_outgoing", "route_pair"
    }:
        raise ValueError(f"Unknown candidate scope: {config.candidate_scope}")
    if config.candidate_scope == "current_chart":
        candidates = np.flatnonzero(prototype_chart == current_chart)
    elif (
        config.architecture == "hierarchical"
        or config.candidate_scope == "current_outgoing"
    ):
        counts = np.asarray(physical_atlas["transition_counts"], dtype=int)
        outgoing = np.flatnonzero(
            counts[current_chart] >= config.edge_count_threshold
        )
        admissible_charts = np.unique(np.r_[current_chart, outgoing])
        candidates = np.flatnonzero(np.isin(prototype_chart, admissible_charts))
    elif config.candidate_scope == "route_pair":
        admissible_charts = np.unique(
            np.r_[current_chart, current_chart if next_chart is None else next_chart]
        )
        candidates = np.flatnonzero(np.isin(prototype_chart, admissible_charts))
    else:
        candidates = np.arange(prototype_chart.size)
    if candidates.size == 0:
        candidates = np.arange(prototype_chart.size)

    past_distance = np.linalg.norm(
        artifact["prototype_past"][candidates] - past_query[None, :], axis=1
    )
    delta_future = artifact["prototype_future"][candidates] - future_query[None, :]
    future_weights = np.tile(artifact["q_diagonal"], pipe.T_F)
    future_distance = np.sqrt(np.sum(future_weights[None, :] * delta_future**2, axis=1))

    def standardized(values: np.ndarray) -> np.ndarray:
        return (values - values.mean()) / max(float(values.std()), 1e-12)

    if config.distance_mode == "balanced_embedding":
        scores = (
            config.past_weight * past_distance**2 / past_query.size
            + config.future_weight * future_distance**2 / np.sum(future_weights)
        )
    elif config.distance_mode == "standardized_contextual":
        scores = (
            config.past_weight * standardized(past_distance)
            + config.future_weight * standardized(future_distance)
        )
    else:
        raise ValueError(f"Unknown contextual distance mode: {config.distance_mode}")
    best_local = int(np.argmin(scores))
    selected = int(candidates[best_local])
    if previous_cluster is not None:
        prior_location = np.flatnonzero(candidates == previous_cluster)
        if prior_location.size and scores[int(prior_location[0])] <= scores[best_local] + config.score_margin:
            selected = int(previous_cluster)
    return selected, candidates, past_distance, future_distance, scores


def _solution_objective(
    controller, matrix: dict, y_ini: np.ndarray, reference: np.ndarray,
    u_solution: np.ndarray, y_solution: np.ndarray, g_solution: np.ndarray,
) -> float:
    """Evaluate the same convex objective used inside ``DeePCController.solve``."""
    offset, basis, _g_min, _g_max = pipe.unpack_basis(matrix)
    reconstruction = offset + basis @ g_solution
    y_p_start = pipe.M * (pipe.T_INI + pipe.T_F)
    sigma = reconstruction[y_p_start:y_p_start + pipe.P * pipe.T_INI] - y_ini
    value = (
        controller.lam_ini * float(np.sum(sigma**2))
        + controller.lam_g * float(np.sum(g_solution**2))
    )
    for step in range(pipe.T_F):
        y_step = y_solution[step * pipe.P:(step + 1) * pipe.P]
        r_step = reference[step * pipe.P:(step + 1) * pipe.P]
        u_step = u_solution[step * pipe.M:(step + 1) * pipe.M]
        error = y_step - r_step
        value += float(error.T @ controller.Q @ error)
        value += float(u_step.T @ controller.R @ u_step)
        if step == pipe.T_F - 1:
            value += float(error.T @ controller.Q_terminal @ error)
    for step in range(pipe.T_F - 1):
        delta = (
            u_solution[(step + 1) * pipe.M:(step + 2) * pipe.M]
            - u_solution[step * pipe.M:(step + 1) * pipe.M]
        )
        value += controller.lam_du * float(np.sum(delta**2))
    return value


def run_contextual_cluster_atlas(
    H: np.ndarray,
    physical_atlas: dict,
    artifact: dict,
    control_config: routed.RoutedDeePCConfig,
    cluster_config: ContextualClusterConfig,
    save_path: Path | None = None,
    physical_matrix_cache: dict | None = None,
) -> dict:
    if cluster_config.reference_mode not in {"graph_routed", "final_only"}:
        raise ValueError(f"Unknown reference mode: {cluster_config.reference_mode}")
    controller = build_controller(control_config)
    matrices = matrix_cache(artifact)
    labels = np.asarray(physical_atlas["labels"], dtype=int)
    always_on = cluster_config.activation_mode == "always_on"
    if not always_on and physical_matrix_cache is None:
        physical_matrix_cache = physical_matrix_cache_from_artifact(artifact)

    env = gym.make(
        pipe.REACHER_ENV_ID,
        max_episode_steps=control_config.steps + pipe.T_INI + 5,
    )
    obs, _ = env.reset(seed=control_config.seed)
    target_xy = np.asarray(obs[4:6], dtype=float).copy()
    initial_y = pipe.reacher_output_from_obs(obs)
    goal, goal_metadata = chart_compatible_goal(H, target_xy, initial_y, control_config)
    goal_chart = int(labels[int(goal_metadata["data_goal_index"])])

    u_history, y_history = [], []
    for _ in range(pipe.T_INI):
        action = np.zeros(pipe.M)
        obs, *_ = env.step(action)
        u_history.append(action.copy())
        y_history.append(pipe.reacher_output_from_obs(obs))

    target_reference_latched = False
    terminal_distance_history = []
    contextual_active = always_on
    trigger_step = -1
    previous_cluster = None
    records = {key: [] for key in (
        "y", "u", "pred", "ref", "distance", "prediction_error_xy", "status",
        "chart", "goal_chart", "next_chart", "path_length", "contextual_active",
        "contextual_triggered", "contextual_cluster", "contextual_cluster_chart",
        "candidate_count", "candidate_solve_count", "chosen_candidate_rank",
        "matrix_columns", "localization_time_s", "routing_time_s",
        "cluster_scoring_time_s", "matrix_preparation_time_s", "solve_time_s",
        "control_compute_time_s",
    )}

    for step in range(control_config.steps):
        control_start = time.perf_counter()
        u_ini = np.asarray(u_history[-pipe.T_INI:]).reshape(pipe.M * pipe.T_INI, 1)
        y_ini = np.asarray(y_history[-pipe.T_INI:]).reshape(pipe.P * pipe.T_INI, 1)

        stage = time.perf_counter()
        feature = routed.online_behavioral_feature(
            u_ini, y_ini, physical_atlas["feature_mean"], physical_atlas["feature_scale"],
            control_config.clustering_input_weight, control_config.clustering_feature_mode,
            control_config.clustering_configuration_weight,
            control_config.clustering_velocity_weight,
            control_config.clustering_fingertip_weight,
        )
        current_chart = int(routed.assign_atlas_charts(physical_atlas, feature)[0])
        localization_time = time.perf_counter() - stage

        stage = time.perf_counter()
        path = routed.shortest_chart_path(
            physical_atlas, current_chart, goal_chart, control_config.edge_count_threshold
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
        terminal_reference = use_target or current_chart == goal_chart
        active_reference = (
            goal
            if cluster_config.reference_mode == "final_only" or terminal_reference
            else physical_atlas["medoid_states"][next_chart]
        )
        reference = np.tile(active_reference.reshape(pipe.P, 1), (pipe.T_F, 1))
        routing_time = time.perf_counter() - stage

        triggered = False
        if not always_on and terminal_reference:
            terminal_distance_history.append(distance_before)
            if len(terminal_distance_history) > cluster_config.stall_window:
                progress = (
                    terminal_distance_history[-cluster_config.stall_window - 1]
                    - terminal_distance_history[-1]
                )
                if (
                    not contextual_active
                    and distance_before > cluster_config.error_threshold
                    and progress < cluster_config.minimum_progress
                ):
                    contextual_active = True
                    triggered = True
                    trigger_step = step
        elif not always_on and not contextual_active:
            terminal_distance_history.clear()

        stage = time.perf_counter()
        selected_cluster = -1
        candidates = np.empty(0, dtype=int)
        solve_clusters = []
        if contextual_active:
            selected_cluster, candidates, _past, _future, candidate_scores = select_contextual_cluster(
                y_ini, reference, current_chart, physical_atlas, artifact,
                cluster_config, previous_cluster, next_chart,
            )
            ranked_clusters = candidates[np.argsort(candidate_scores, kind="stable")]
            solve_clusters = [selected_cluster] + [
                int(value) for value in ranked_clusters if int(value) != selected_cluster
            ]
            solve_clusters = solve_clusters[:max(1, cluster_config.candidate_solve_count)]
        scoring_time = time.perf_counter() - stage

        stage = time.perf_counter()
        if contextual_active:
            matrix = matrices[selected_cluster]
            matrix_columns = cluster_config.support_columns
        else:
            matrix, selected = physical_matrix_cache[(current_chart,)]
            matrix_columns = int(selected.size)
        matrix_time = time.perf_counter() - stage

        stage = time.perf_counter()
        chosen_rank = 0
        if contextual_active and len(solve_clusters) > 1:
            solutions = []
            for rank_index, cluster_index in enumerate(solve_clusters):
                candidate_matrix = matrices[cluster_index]
                u_candidate, y_candidate, g_candidate, candidate_status, _ = controller.solve(
                    candidate_matrix, u_ini, y_ini, reference
                )
                objective = (
                    np.inf if u_candidate is None else _solution_objective(
                        controller, candidate_matrix, y_ini, reference,
                        u_candidate, y_candidate, g_candidate,
                    )
                )
                solutions.append((
                    objective, rank_index, cluster_index, candidate_matrix,
                    u_candidate, y_candidate, g_candidate, candidate_status,
                ))
            (_objective, chosen_rank, selected_cluster, matrix,
             u_solution, y_solution, _g, status) = min(
                solutions, key=lambda row: row[0]
            )
        else:
            u_solution, y_solution, _g, status, _ = controller.solve(
                matrix, u_ini, y_ini, reference
            )
        previous_cluster = selected_cluster if contextual_active else previous_cluster
        solve_time = time.perf_counter() - stage
        if u_solution is None:
            action = np.zeros(pipe.M)
            prediction = np.full(pipe.P, np.nan)
        else:
            action = np.clip(
                u_solution[:pipe.M, 0], -control_config.u_limit, control_config.u_limit
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
        records["prediction_error_xy"].append(float(np.linalg.norm(prediction[6:8] - y_now[6:8])))
        records["status"].append(str(status))
        records["chart"].append(current_chart)
        records["goal_chart"].append(goal_chart)
        records["next_chart"].append(next_chart)
        records["path_length"].append(len(path))
        records["contextual_active"].append(contextual_active)
        records["contextual_triggered"].append(triggered)
        records["contextual_cluster"].append(selected_cluster)
        records["contextual_cluster_chart"].append(
            -1 if selected_cluster < 0 else int(artifact["prototype_chart"][selected_cluster])
        )
        records["candidate_count"].append(candidates.size)
        records["candidate_solve_count"].append(
            len(solve_clusters) if contextual_active else 1
        )
        records["chosen_candidate_rank"].append(chosen_rank)
        records["matrix_columns"].append(matrix_columns)
        records["localization_time_s"].append(localization_time)
        records["routing_time_s"].append(routing_time)
        records["cluster_scoring_time_s"].append(scoring_time)
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
    Q = output_cost(control_config)
    R = control_config.r * np.eye(pipe.M)
    reference_cost = pipe.closed_loop_control_cost(arrays["y"], arrays["u"], arrays["ref"], Q, R)
    final_goal_cost = pipe.closed_loop_control_cost(
        arrays["y"], arrays["u"], goal, Q, R,
        reference_description="common final graph-compatible goal",
    )
    summary = {
        **asdict(control_config),
        "method": "contextual_cluster_atlas_deepc",
        "contextual_cluster": asdict(cluster_config),
        "contextual_cluster_count": int(artifact["prototype_past"].shape[0]),
        "contextual_triggered": bool(not always_on and trigger_step >= 0),
        "contextual_trigger_step": int(trigger_step),
        "contextual_usage_fraction": float(np.mean(arrays["contextual_active"])),
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
        "controller": (
            "DeePC with offline contextual-cluster "
            f"rank-{control_config.svd_rank} basis"
        ),
        "graph_routing_used": True,
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
        "closed_loop_final_goal_cost": final_goal_cost["closed_loop_control_cost"],
        **timing_summary(arrays["localization_time_s"], "localization"),
        **timing_summary(arrays["routing_time_s"], "routing"),
        **timing_summary(arrays["cluster_scoring_time_s"], "cluster_scoring"),
        **timing_summary(arrays["matrix_preparation_time_s"], "matrix_preparation"),
        **timing_summary(arrays["solve_time_s"], "solve"),
        **timing_summary(arrays["control_compute_time_s"], "control_compute"),
        **goal_metadata,
    }
    if save_path is not None:
        np.savez(
            save_path, **arrays, final_goal=goal,
            summary=np.array(json.dumps(summary), dtype=object),
        )
    return summary


_H = None
_PHYSICAL_ATLAS = None
_ARTIFACT = None
_CONTROL_CONFIG = None
_CLUSTER_CONFIG = None
_OUTPUT_DIR = None
_PHYSICAL_CACHE = None
_SAVE_ROLLOUTS = True


COMPACT_RESULT_FIELDS = (
    "seed",
    "svd_rank",
    "success_30",
    "success_100",
    "mean_control_compute_ms",
    "median_control_compute_ms",
    "p95_control_compute_ms",
)


def compact_result_summary(result: dict) -> dict:
    """Retain only success-rate and end-to-end compute-time fields."""
    missing = [key for key in COMPACT_RESULT_FIELDS if key not in result]
    if missing:
        raise KeyError(f"Compact result is missing fields: {missing}")
    return {key: result[key] for key in COMPACT_RESULT_FIELDS}


def initialize_worker(
    data_path, atlas_path, artifact_path, control, cluster, output_dir,
    save_rollouts=True,
):
    global _H, _PHYSICAL_ATLAS, _ARTIFACT, _CONTROL_CONFIG, _CLUSTER_CONFIG
    global _OUTPUT_DIR, _PHYSICAL_CACHE, _SAVE_ROLLOUTS
    _H = np.load(data_path)
    _PHYSICAL_ATLAS = routed.load_atlas(Path(atlas_path))
    _CONTROL_CONFIG = routed.RoutedDeePCConfig(**control)
    _CLUSTER_CONFIG = ContextualClusterConfig(**cluster)
    loaded_artifact = load_contextual_cluster_atlas(Path(artifact_path))
    available_rank = int(
        loaded_artifact.get(
            "basis_rank", loaded_artifact["matrix_bases"].shape[2]
        )
    )
    if _CONTROL_CONFIG.svd_rank > available_rank:
        raise RuntimeError(
            f"Worker requested rank {_CONTROL_CONFIG.svd_rank}, but shared "
            f"artifact provides only rank {available_rank}"
        )
    _ARTIFACT = (
        loaded_artifact
        if _CONTROL_CONFIG.svd_rank == available_rank
        else truncate_contextual_cluster_atlas(
            loaded_artifact, _CONTROL_CONFIG.svd_rank
        )
    )
    _OUTPUT_DIR = Path(output_dir)
    _SAVE_ROLLOUTS = bool(save_rollouts)
    _PHYSICAL_CACHE = (
        None if _CLUSTER_CONFIG.activation_mode == "always_on"
        else physical_matrix_cache_from_artifact(_ARTIFACT)
    )


def run_worker(seed: int) -> dict:
    config = routed.RoutedDeePCConfig(**{**asdict(_CONTROL_CONFIG), "seed": int(seed)})
    save_path = (
        _OUTPUT_DIR / f"seed{seed}_control_results.npz"
        if _SAVE_ROLLOUTS
        else None
    )
    return run_contextual_cluster_atlas(
        _H, _PHYSICAL_ATLAS, _ARTIFACT, config, _CLUSTER_CONFIG,
        save_path=save_path,
        physical_matrix_cache=_PHYSICAL_CACHE,
    )


def _load_rollout_summary(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return json.loads(data["summary"].item())


def run_panel(
    seeds,
    baseline_dir: Path,
    output_dir: Path,
    cluster_config: ContextualClusterConfig,
    data_path: Path = DEFAULT_DATA_PATH,
    workers: int = 1,
    overwrite: bool = False,
    control_overrides: dict | None = None,
    source_artifact_path: Path | None = None,
    shared_artifact_path: Path | None = None,
    bootstrap_dirs=(),
    save_rollouts: bool = True,
    compact_results: bool = False,
    delete_local_rollouts_after_compaction: bool = False,
    progress_every: int = 1,
) -> dict:
    seeds = [int(seed) for seed in seeds]
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_payload = json.loads((baseline_dir / "summary.json").read_text())
    control_config = baseline_config(baseline_payload)
    if control_overrides:
        unknown = set(control_overrides) - set(asdict(control_config))
        if unknown:
            raise ValueError(f"Unknown control overrides: {sorted(unknown)}")
        control_config = routed.RoutedDeePCConfig(
            **{**asdict(control_config), **control_overrides}
        )
    if compact_results and save_rollouts:
        raise ValueError("compact_results requires save_rollouts=False")
    if delete_local_rollouts_after_compaction and not compact_results:
        raise ValueError(
            "delete_local_rollouts_after_compaction requires compact_results=True"
        )
    if progress_every < 1:
        raise ValueError("progress_every must be at least 1")
    H = np.load(data_path, mmap_mode="r")
    atlas_path = baseline_dir / "behavioral_atlas.npz"
    physical_atlas = routed.load_atlas(atlas_path)

    if shared_artifact_path is not None:
        artifact_path = Path(shared_artifact_path)
        artifact_source = load_contextual_cluster_atlas(artifact_path)
        saved = _config_with_defaults(
            json.loads(artifact_source["cluster_config"].item())
        )
        if saved != asdict(cluster_config):
            raise RuntimeError(
                f"Incompatible shared contextual atlas: {artifact_path}"
            )
        available_rank = int(
            artifact_source.get(
                "basis_rank", artifact_source["matrix_bases"].shape[2]
            )
        )
        if control_config.svd_rank > available_rank:
            raise RuntimeError(
                f"Shared atlas rank {available_rank} is below requested rank "
                f"{control_config.svd_rank}: {artifact_path}"
            )
        artifact = (
            artifact_source
            if control_config.svd_rank == available_rank
            else truncate_contextual_cluster_atlas(
                artifact_source, control_config.svd_rank
            )
        )
        offline_seconds = 0.0
    else:
        artifact_path = output_dir / "contextual_cluster_atlas.npz"

    if shared_artifact_path is None and (overwrite or not artifact_path.exists()):
        started = time.perf_counter()
        if source_artifact_path is None:
            artifact = build_contextual_cluster_atlas(
                H, physical_atlas, control_config, cluster_config
            )
        else:
            source = load_contextual_cluster_atlas(source_artifact_path)
            saved = _config_with_defaults(
                json.loads(source["cluster_config"].item())
            )
            if not _same_offline_atlas(saved, asdict(cluster_config)):
                raise RuntimeError(
                    f"Incompatible source contextual atlas: {source_artifact_path}"
                )
            available_rank = int(
                source.get("basis_rank", source["matrix_bases"].shape[2])
            )
            if control_config.svd_rank <= available_rank:
                artifact = truncate_contextual_cluster_atlas(
                    source, control_config.svd_rank
                )
                artifact["cluster_config"] = np.array(
                    json.dumps(asdict(cluster_config)), dtype=object
                )
            else:
                artifact = rebuild_contextual_cluster_atlas_bases(
                    H, physical_atlas, source, control_config, cluster_config
                )
        save_contextual_cluster_atlas(artifact_path, artifact)
        offline_seconds = time.perf_counter() - started
    elif shared_artifact_path is None:
        artifact = load_contextual_cluster_atlas(artifact_path)
        saved = _config_with_defaults(
            json.loads(artifact["cluster_config"].item())
        )
        if saved != asdict(cluster_config):
            raise RuntimeError(f"Incompatible contextual-cluster atlas: {artifact_path}")
        artifact_rank = int(
            artifact.get("basis_rank", artifact["matrix_bases"].shape[2])
        )
        if artifact_rank != control_config.svd_rank:
            raise RuntimeError(
                f"Cached rank {artifact_rank} does not match requested "
                f"rank {control_config.svd_rank}: {artifact_path}"
            )
        offline_seconds = 0.0

    results = {}
    checkpoint_path = output_dir / "compact_results.jsonl"
    checkpoint_metadata_path = output_dir / "compact_results_metadata.json"
    if compact_results:
        checkpoint_metadata = {
            "config": asdict(control_config),
            "contextual_cluster": asdict(cluster_config),
            "result_fields": list(COMPACT_RESULT_FIELDS),
        }
        if checkpoint_metadata_path.exists() and not overwrite:
            saved_metadata = json.loads(checkpoint_metadata_path.read_text())
            if saved_metadata != checkpoint_metadata:
                raise RuntimeError(
                    f"Incompatible compact checkpoint: {checkpoint_metadata_path}"
                )
        else:
            checkpoint_metadata_path.write_text(
                json.dumps(checkpoint_metadata, indent=2)
            )
        if overwrite:
            checkpoint_path.write_text("")
        elif (output_dir / "summary.json").exists():
            saved_payload = json.loads((output_dir / "summary.json").read_text())
            if (
                saved_payload.get("config") == asdict(control_config)
                and saved_payload.get("contextual_cluster")
                == asdict(cluster_config)
            ):
                for row in saved_payload.get("results", []):
                    results[int(row["seed"])] = compact_result_summary(row)
        if checkpoint_path.exists() and not overwrite:
            for line_number, line in enumerate(
                checkpoint_path.read_text().splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"Invalid compact checkpoint line {line_number}: "
                        f"{checkpoint_path}"
                    ) from error
                results[int(row["seed"])] = compact_result_summary(row)

    def retain_result(row):
        retained = compact_result_summary(row) if compact_results else row
        if compact_results:
            with checkpoint_path.open("a") as checkpoint:
                checkpoint.write(json.dumps(retained) + "\n")
        return retained

    def compatible(row):
        return (
            _config_with_defaults(row.get("contextual_cluster", {}))
            == asdict(cluster_config)
            and row.get("svd_rank") == control_config.svd_rank
        )

    if not overwrite:
        for bootstrap_dir in tuple(Path(path) for path in bootstrap_dirs):
            summary_path = bootstrap_dir / "summary.json"
            if not summary_path.exists():
                continue
            bootstrap_payload = json.loads(summary_path.read_text())
            if (
                bootstrap_payload.get("config", {}).get("svd_rank")
                != control_config.svd_rank
                or _config_with_defaults(
                    bootstrap_payload.get("contextual_cluster", {})
                ) != asdict(cluster_config)
            ):
                raise RuntimeError(f"Incompatible bootstrap panel: {summary_path}")
            for row in bootstrap_payload.get("results", []):
                seed = int(row["seed"])
                if seed in seeds and seed not in results:
                    results[seed] = retain_result(row)

        source_dirs = (output_dir, *tuple(Path(path) for path in bootstrap_dirs))
        for source_dir in source_dirs:
            for seed in seeds:
                if seed in results:
                    continue
                path = source_dir / f"seed{seed}_control_results.npz"
                if not path.exists():
                    continue
                row = _load_rollout_summary(path)
                if not compatible(row):
                    raise RuntimeError(f"Incompatible existing result: {path}")
                results[seed] = retain_result(row)
                if (
                    delete_local_rollouts_after_compaction
                    and source_dir.resolve() == output_dir.resolve()
                ):
                    path.unlink()

    pending = [seed for seed in seeds if seed not in results]
    initargs = (
        str(data_path), str(atlas_path), str(artifact_path), asdict(control_config),
        asdict(cluster_config), str(output_dir), save_rollouts,
    )
    if workers == 1:
        initialize_worker(*initargs)
        for completed_count, seed in enumerate(pending, start=1):
            full_result = run_worker(seed)
            results[seed] = retain_result(full_result)
            if completed_count % progress_every == 0 or completed_count == len(pending):
                print(
                    f"completed={completed_count}/{len(pending)} seed={seed} "
                    f"strict={full_result['success_30'] and full_result['success_100']} "
                    f"median_control={full_result['median_control_compute_ms']:.2f} ms",
                    flush=True,
                )
    elif pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker, initargs=initargs) as pool:
            futures = {pool.submit(run_worker, seed): seed for seed in pending}
            for completed_count, future in enumerate(as_completed(futures), start=1):
                seed = futures[future]
                full_result = future.result()
                results[seed] = retain_result(full_result)
                if completed_count % progress_every == 0 or completed_count == len(pending):
                    print(
                        f"completed={completed_count}/{len(pending)} seed={seed} "
                        f"strict={full_result['success_30'] and full_result['success_100']} "
                        f"median_control={full_result['median_control_compute_ms']:.2f} ms",
                        flush=True,
                    )

    payload = {
        "method": "contextual_cluster_atlas_deepc",
        "baseline_source": str(baseline_dir),
        "config": asdict(control_config),
        "contextual_cluster": asdict(cluster_config),
        "offline_build_seconds_this_run": offline_seconds,
        "shared_artifact_source": (
            None if shared_artifact_path is None else str(shared_artifact_path)
        ),
        "bootstrap_dirs": [str(path) for path in bootstrap_dirs],
        "save_rollouts": bool(save_rollouts),
        "compact_results": bool(compact_results),
        "artifact": {
            "prototype_count": int(artifact["prototype_past"].shape[0]),
            "support_columns": int(artifact["support_indices"].shape[1]),
            "basis_rank": int(control_config.svd_rank),
            "assignment_count_min": int(artifact["assignment_counts"].min()),
            "assignment_count_median": float(np.median(artifact["assignment_counts"])),
            "assignment_count_max": int(artifact["assignment_counts"].max()),
        },
        "fairness_audit": audit_iid_library(H, eval_seeds=seeds),
        "results": [results[seed] for seed in seeds],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    if compact_results and checkpoint_path.exists():
        checkpoint_path.unlink()
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-stop", type=int)
    parser.add_argument("--seeds", help="Comma-separated seed list for paired screens.")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--architecture", choices=("global_vq", "hierarchical"), default="global_vq")
    parser.add_argument("--clusters", type=int, default=128)
    parser.add_argument("--modes-per-chart", type=int, default=4)
    parser.add_argument("--support-columns", type=int, default=200)
    parser.add_argument("--cluster-seed", type=int, default=73)
    parser.add_argument("--activation-mode", choices=("always_on", "triggered"), default="always_on")
    parser.add_argument("--past-weight", type=float, default=1.0)
    parser.add_argument("--future-weight", type=float, default=1.0)
    parser.add_argument(
        "--distance-mode",
        choices=("standardized_contextual", "balanced_embedding"),
        default="standardized_contextual",
    )
    parser.add_argument("--score-margin", type=float, default=0.0)
    parser.add_argument("--candidate-solve-count", type=int, default=1)
    parser.add_argument(
        "--candidate-scope",
        choices=("global", "current_chart", "current_outgoing", "route_pair"),
        default="global",
    )
    parser.add_argument(
        "--reference-mode",
        choices=("graph_routed", "final_only"),
        default="graph_routed",
    )
    parser.add_argument("--svd-rank", type=int, default=60)
    parser.add_argument("--source-artifact", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = ContextualClusterConfig(
        architecture=args.architecture,
        contextual_clusters=args.clusters,
        modes_per_chart=args.modes_per_chart,
        support_columns=args.support_columns,
        cluster_seed=args.cluster_seed,
        activation_mode=args.activation_mode,
        past_weight=args.past_weight,
        future_weight=args.future_weight,
        distance_mode=args.distance_mode,
        score_margin=args.score_margin,
        candidate_solve_count=args.candidate_solve_count,
        candidate_scope=args.candidate_scope,
        reference_mode=args.reference_mode,
    )
    if args.seeds:
        seeds = [int(value) for value in args.seeds.split(",")]
    elif args.seed_start is not None and args.seed_stop is not None:
        seeds = range(args.seed_start, args.seed_stop + 1)
    else:
        raise SystemExit("provide --seeds or both --seed-start and --seed-stop")
    payload = run_panel(
        seeds, args.baseline_dir,
        args.output_dir, config, workers=args.workers, overwrite=args.overwrite,
        control_overrides={"svd_rank": args.svd_rank},
        source_artifact_path=args.source_artifact,
    )
    successes = sum(row["success_30"] and row["success_100"] for row in payload["results"])
    print(json.dumps({"successes": successes, "trials": len(payload["results"]), "artifact": payload["artifact"]}, indent=2))


if __name__ == "__main__":
    main()
