"""Pure, always-on chart-local contextual Atlas-DeePC.

Each physical localization chart owns a small offline atlas of Q-weighted
future-output modes.  At every control step exactly those modes are scored
against the active reference and the final graph goal.  The selected cached
rank-reduced DeePC basis is used immediately; there is no trigger, physical
basis fallback, online Hankel scan, or online SVD.
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
from sklearn.cluster import KMeans

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
class ChartLocalContextualConfig:
    modes_per_chart: int = 8
    columns_per_mode: int = 350
    mode_cluster_seed: int = 37
    graph_weight: float = 0.5
    switch_penalty: float = 0.1
    edge_count_threshold: int = 3
    reference_mode: str = "graph_routed"


_ARTIFACT_BUILD_FIELDS = (
    "modes_per_chart",
    "columns_per_mode",
    "mode_cluster_seed",
    "edge_count_threshold",
)


def _config_with_defaults(values: dict) -> dict:
    return {**asdict(ChartLocalContextualConfig()), **values}


def _same_offline_atlas(left: dict, right: dict) -> bool:
    return all(left.get(key) == right.get(key) for key in _ARTIFACT_BUILD_FIELDS)


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / max(float(values.std()), 1e-12)


def _all_pairs_graph_distance(atlas: dict, threshold: int) -> np.ndarray:
    counts = np.asarray(atlas["transition_counts"], dtype=int)
    distance = np.full(counts.shape, np.inf, dtype=float)
    np.fill_diagonal(distance, 0.0)
    for source, destination in zip(*np.where(counts >= threshold)):
        if source == destination:
            continue
        distance[source, destination] = (
            atlas["center_distance"][source, destination]
            / np.sqrt(float(counts[source, destination]))
        )
    for intermediate in range(distance.shape[0]):
        distance = np.minimum(
            distance,
            distance[:, intermediate, None] + distance[intermediate, None, :],
        )
    return distance


def build_chart_local_contextual_atlas(
    H: np.ndarray,
    atlas: dict,
    control: routed.RoutedDeePCConfig,
    config: ChartLocalContextualConfig,
) -> dict[str, np.ndarray]:
    """Build chart-local future modes and one cached DeePC basis per mode."""
    labels = np.asarray(atlas["labels"], dtype=int)
    chart_count = int(atlas["centers"].shape[0])
    library = prepare_contextual_library(H)
    sqrt_weights = np.sqrt(np.tile(np.diag(output_cost(control)), pipe.T_F))
    weighted_future = library["Y_f_normalized"].T * sqrt_weights[None, :]

    prototypes = np.empty(
        (chart_count, config.modes_per_chart, weighted_future.shape[1]), float
    )
    mode_counts = np.empty((chart_count, config.modes_per_chart), int)
    assignments: list[tuple[np.ndarray, np.ndarray]] = []
    for chart in range(chart_count):
        members = np.flatnonzero(labels == chart)
        model = KMeans(
            n_clusters=config.modes_per_chart,
            random_state=config.mode_cluster_seed + chart,
            n_init=5,
            max_iter=300,
            algorithm="lloyd",
        )
        local_labels = model.fit_predict(weighted_future[members])
        prototypes[chart] = model.cluster_centers_
        mode_counts[chart] = np.bincount(
            local_labels, minlength=config.modes_per_chart
        )
        assignments.append((members, local_labels))

    # Attribute each future mode to the chart occupied at the end of Y_f.
    # Episode boundaries are excluded because adjacent Hankel columns there do
    # not belong to the same physical trajectory.
    mode_destinations = np.empty((chart_count, config.modes_per_chart), int)
    for chart, (members, local_labels) in enumerate(assignments):
        for mode in range(config.modes_per_chart):
            owned = members[local_labels == mode]
            valid = owned[(owned % routed.WINDOWS_PER_EPISODE) + pipe.T_F < routed.WINDOWS_PER_EPISODE]
            destinations = labels[valid + pipe.T_F]
            mode_destinations[chart, mode] = int(
                np.argmax(np.bincount(destinations, minlength=chart_count))
            )

    supports = np.empty(
        (chart_count, config.modes_per_chart, config.columns_per_mode), int
    )
    offsets, bases, g_mins, g_maxs = [], [], [], []
    for chart in range(chart_count):
        members = np.flatnonzero(labels == chart)
        chart_offsets, chart_bases, chart_mins, chart_maxs = [], [], [], []
        for mode in range(config.modes_per_chart):
            distance = np.mean(
                (weighted_future[members] - prototypes[chart, mode]) ** 2,
                axis=1,
            )
            count = min(config.columns_per_mode, members.size)
            selected = np.sort(members[np.argpartition(distance, count - 1)[:count]])
            if count != config.columns_per_mode:
                raise RuntimeError(
                    f"Chart {chart} has only {count} columns; "
                    f"{config.columns_per_mode} are required."
                )
            supports[chart, mode] = selected
            matrix = selected_affine_basis(H, selected, control)
            chart_offsets.append(matrix["offset"])
            chart_bases.append(matrix["basis"])
            chart_mins.append(matrix["g_min"])
            chart_maxs.append(matrix["g_max"])
        offsets.append(chart_offsets)
        bases.append(chart_bases)
        g_mins.append(chart_mins)
        g_maxs.append(chart_maxs)

    return {
        "prototypes": prototypes,
        "mode_counts": mode_counts,
        "mode_destinations": mode_destinations,
        "selected_indices": supports,
        "matrix_offsets": np.asarray(offsets),
        "matrix_bases": np.asarray(bases),
        "matrix_g_mins": np.asarray(g_mins),
        "matrix_g_maxs": np.asarray(g_maxs),
        "output_mean": library["output_mean"],
        "output_scale": library["output_scale"],
        "sqrt_future_weights": sqrt_weights,
        "graph_distance": _all_pairs_graph_distance(
            atlas, config.edge_count_threshold
        ),
        "contextual_config": np.array(json.dumps(asdict(config)), dtype=object),
    }


def load_chart_local_contextual_atlas(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def save_chart_local_contextual_atlas(path: Path, artifact: dict) -> None:
    np.savez_compressed(path, **artifact)


def select_contextual_mode(
    current_chart: int,
    goal_chart: int,
    reference: np.ndarray,
    artifact: dict,
    config: ChartLocalContextualConfig,
    previous_mode: int | None,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    normalized = (
        np.asarray(reference).reshape(pipe.T_F, pipe.P)
        - artifact["output_mean"][None, :]
    ) / artifact["output_scale"][None, :]
    query = normalized.reshape(-1) * artifact["sqrt_future_weights"]
    delta = artifact["prototypes"][current_chart] - query[None, :]
    future_distance = np.mean(delta * delta, axis=1)
    destinations = artifact["mode_destinations"][current_chart]
    route_distance = artifact["graph_distance"][destinations, goal_chart]
    score = _standardize(future_distance) + config.graph_weight * _standardize(
        route_distance
    )
    if previous_mode is not None:
        score[np.arange(score.size) != previous_mode] += config.switch_penalty
    selected = int(np.argmin(score))
    return selected, score, future_distance, route_distance


def _matrix(artifact: dict, chart: int, mode: int) -> dict[str, np.ndarray]:
    return {
        "offset": artifact["matrix_offsets"][chart, mode],
        "basis": artifact["matrix_bases"][chart, mode],
        "g_min": artifact["matrix_g_mins"][chart, mode],
        "g_max": artifact["matrix_g_maxs"][chart, mode],
    }


def run_chart_local_contextual_atlas(
    H: np.ndarray,
    atlas: dict,
    artifact: dict,
    control: routed.RoutedDeePCConfig,
    config: ChartLocalContextualConfig,
    save_path: Path | None = None,
) -> dict:
    if config.reference_mode not in {"graph_routed", "final_only"}:
        raise ValueError(f"Unknown reference mode: {config.reference_mode}")
    controller = build_controller(control)
    labels = np.asarray(atlas["labels"], dtype=int)
    env = gym.make(
        pipe.REACHER_ENV_ID,
        max_episode_steps=control.steps + pipe.T_INI + 5,
    )
    obs, _ = env.reset(seed=control.seed)
    target_xy = np.asarray(obs[4:6], dtype=float).copy()
    initial_y = pipe.reacher_output_from_obs(obs)
    goal, goal_metadata = chart_compatible_goal(H, target_xy, initial_y, control)
    goal_chart = int(labels[int(goal_metadata["data_goal_index"])])

    u_history, y_history = [], []
    for _ in range(pipe.T_INI):
        action = np.zeros(pipe.M)
        obs, *_ = env.step(action)
        u_history.append(action.copy())
        y_history.append(pipe.reacher_output_from_obs(obs))

    previous_mode = None
    previous_chart = None
    target_reference_latched = False
    keys = (
        "y", "u", "pred", "ref", "distance", "prediction_error_xy",
        "status", "chart", "goal_chart", "next_chart", "path_length",
        "contextual_mode", "mode_destination", "mode_score",
        "mode_future_distance", "mode_route_distance", "mode_switched",
        "matrix_columns", "localization_time_s", "routing_time_s",
        "mode_scoring_time_s", "matrix_lookup_time_s", "solve_time_s",
        "control_compute_time_s",
    )
    records = {key: [] for key in keys}
    for _step in range(control.steps):
        control_start = time.perf_counter()
        u_ini = np.asarray(u_history[-pipe.T_INI:]).reshape(pipe.M * pipe.T_INI, 1)
        y_ini = np.asarray(y_history[-pipe.T_INI:]).reshape(pipe.P * pipe.T_INI, 1)

        stage = time.perf_counter()
        feature = routed.online_behavioral_feature(
            u_ini, y_ini, atlas["feature_mean"], atlas["feature_scale"],
            control.clustering_input_weight, control.clustering_feature_mode,
            control.clustering_configuration_weight,
            control.clustering_velocity_weight,
            control.clustering_fingertip_weight,
        )
        current_chart = int(routed.assign_atlas_charts(atlas, feature)[0])
        localization_time = time.perf_counter() - stage

        stage = time.perf_counter()
        path = routed.shortest_chart_path(
            atlas, current_chart, goal_chart, control.edge_count_threshold
        ) or [current_chart]
        next_chart = path[1] if len(path) > 1 else current_chart
        distance_before = float(np.linalg.norm(y_history[-1][6:8] - target_xy))
        if distance_before < control.target_switch_distance:
            target_reference_latched = True
        use_target = (
            target_reference_latched
            if control.latch_target_reference
            else distance_before < control.target_switch_distance
        )
        active_reference = (
            goal
            if config.reference_mode == "final_only" or use_target or current_chart == goal_chart
            else atlas["medoid_states"][next_chart]
        )
        reference = np.tile(active_reference.reshape(pipe.P, 1), (pipe.T_F, 1))
        routing_time = time.perf_counter() - stage

        stage = time.perf_counter()
        persistent_mode = previous_mode if previous_chart == current_chart else None
        mode, scores, future_distances, route_distances = select_contextual_mode(
            current_chart, goal_chart, reference, artifact, config, persistent_mode
        )
        switched = persistent_mode is not None and mode != persistent_mode
        previous_mode = mode
        previous_chart = current_chart
        scoring_time = time.perf_counter() - stage

        stage = time.perf_counter()
        matrix = _matrix(artifact, current_chart, mode)
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
                u_solution[:pipe.M, 0], -control.u_limit, control.u_limit
            )
            prediction = np.asarray(y_solution[:pipe.P, 0]).reshape(pipe.P)
        compute_time = time.perf_counter() - control_start

        obs, *_ = env.step(action)
        y_now = pipe.reacher_output_from_obs(obs)
        u_history.append(action.copy())
        y_history.append(y_now.copy())
        values = {
            "y": y_now, "u": action, "pred": prediction,
            "ref": active_reference,
            "distance": float(np.linalg.norm(y_now[6:8] - target_xy)),
            "prediction_error_xy": float(np.linalg.norm(prediction[6:8] - y_now[6:8])),
            "status": str(status), "chart": current_chart,
            "goal_chart": goal_chart, "next_chart": next_chart,
            "path_length": len(path), "contextual_mode": mode,
            "mode_destination": int(artifact["mode_destinations"][current_chart, mode]),
            "mode_score": float(scores[mode]),
            "mode_future_distance": float(future_distances[mode]),
            "mode_route_distance": float(route_distances[mode]),
            "mode_switched": switched,
            "matrix_columns": config.columns_per_mode,
            "localization_time_s": localization_time,
            "routing_time_s": routing_time,
            "mode_scoring_time_s": scoring_time,
            "matrix_lookup_time_s": matrix_time,
            "solve_time_s": solve_time,
            "control_compute_time_s": compute_time,
        }
        for key, value in values.items():
            records[key].append(value)
    env.close()

    arrays = {
        key: np.asarray(value, dtype=object if key == "status" else None)
        for key, value in records.items()
    }
    distance = arrays["distance"].astype(float)
    tail_30, tail_100 = distance[-30:], distance[-min(100, distance.size):]
    Q = output_cost(control)
    R = control.r * np.eye(pipe.M)
    reference_cost = pipe.closed_loop_control_cost(
        arrays["y"], arrays["u"], arrays["ref"], Q, R
    )
    final_cost = pipe.closed_loop_control_cost(
        arrays["y"], arrays["u"], goal, Q, R,
        reference_description="common final graph-compatible goal",
    )
    summary = {
        **asdict(control),
        "method": "always_on_chart_local_contextual_atlas_deepc",
        "contextual_activation_mode": "always_on",
        "always_on_contextual": asdict(config),
        "minimum": float(distance.min()), "minimum_step": int(distance.argmin()),
        "last": float(distance[-1]), "mean_last_30": float(tail_30.mean()),
        "max_last_30": float(tail_30.max()),
        "mean_last_100": float(tail_100.mean()),
        "max_last_100": float(tail_100.max()),
        "success_30": bool(tail_30.mean() < 0.01 and tail_30.max() < 0.02),
        "success_100": bool(tail_100.mean() < 0.01 and tail_100.max() < 0.02),
        "mean_prediction_error_xy": float(np.nanmean(arrays["prediction_error_xy"])),
        "statuses": dict(Counter(arrays["status"].tolist())),
        "mode_switch_fraction": float(np.mean(arrays["mode_switched"])),
        "controller": "always-on chart-local contextual Atlas-DeePC",
        "graph_routing_used": True,
        "online_modes_scored_per_step": config.modes_per_chart,
        "online_hankel_columns_scored": 0,
        "online_nearest_trajectory_retrieval": False,
        "online_svd_rebuild": False,
        "fitted_dynamics_model": False, "use_ik_ref": False,
        "feedback_blend": 0.0, "feedback_fallback": False,
        **reference_cost,
        "closed_loop_reference_cost": reference_cost["closed_loop_control_cost"],
        "closed_loop_final_goal_cost": final_cost["closed_loop_control_cost"],
        **timing_summary(arrays["localization_time_s"], "localization"),
        **timing_summary(arrays["routing_time_s"], "routing"),
        **timing_summary(arrays["mode_scoring_time_s"], "mode_scoring"),
        **timing_summary(arrays["matrix_lookup_time_s"], "matrix_lookup"),
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


_H = _ATLAS = _ARTIFACT = _CONTROL = _CONFIG = _OUTPUT_DIR = None


def _initialize(data_path, atlas_path, artifact_path, control, config, output_dir):
    global _H, _ATLAS, _ARTIFACT, _CONTROL, _CONFIG, _OUTPUT_DIR
    _H = np.load(data_path)
    _ATLAS = routed.load_atlas(Path(atlas_path))
    _ARTIFACT = load_chart_local_contextual_atlas(Path(artifact_path))
    _CONTROL = routed.RoutedDeePCConfig(**control)
    _CONFIG = ChartLocalContextualConfig(**config)
    _OUTPUT_DIR = Path(output_dir)


def _worker(seed: int) -> dict:
    control = routed.RoutedDeePCConfig(
        **{**asdict(_CONTROL), "seed": int(seed)}
    )
    return run_chart_local_contextual_atlas(
        _H, _ATLAS, _ARTIFACT, control, _CONFIG,
        _OUTPUT_DIR / f"seed{seed}_control_results.npz",
    )


def run_panel(
    seeds,
    baseline_dir: Path,
    output_dir: Path,
    config: ChartLocalContextualConfig,
    data_path: Path = DEFAULT_DATA_PATH,
    workers: int = 1,
    overwrite: bool = False,
    source_artifact: Path | None = None,
) -> dict:
    seeds = [int(seed) for seed in seeds]
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_payload = json.loads((baseline_dir / "summary.json").read_text())
    control = baseline_config(baseline_payload)
    H = np.load(data_path, mmap_mode="r")
    atlas_path = baseline_dir / "behavioral_atlas.npz"
    atlas = routed.load_atlas(atlas_path)
    artifact_path = output_dir / "chart_local_contextual_atlas.npz"
    if overwrite or not artifact_path.exists():
        started = time.perf_counter()
        artifact = (
            load_chart_local_contextual_atlas(source_artifact)
            if source_artifact is not None
            else build_chart_local_contextual_atlas(H, atlas, control, config)
        )
        saved_config = _config_with_defaults(
            json.loads(artifact["contextual_config"].item())
        )
        if not _same_offline_atlas(saved_config, asdict(config)):
            raise RuntimeError("Source artifact configuration does not match.")
        artifact["contextual_config"] = np.array(
            json.dumps(asdict(config)), dtype=object
        )
        save_chart_local_contextual_atlas(artifact_path, artifact)
        build_seconds = time.perf_counter() - started
    else:
        artifact = load_chart_local_contextual_atlas(artifact_path)
        if _config_with_defaults(
            json.loads(artifact["contextual_config"].item())
        ) != asdict(config):
            raise RuntimeError(f"Incompatible contextual atlas: {artifact_path}")
        build_seconds = 0.0

    results = {}
    if not overwrite:
        for seed in seeds:
            path = output_dir / f"seed{seed}_control_results.npz"
            if path.exists():
                with np.load(path, allow_pickle=True) as data:
                    row = json.loads(data["summary"].item())
                if _config_with_defaults(row.get("always_on_contextual", {})) == asdict(config):
                    results[seed] = row
    pending = [seed for seed in seeds if seed not in results]
    initargs = (
        str(data_path), str(atlas_path), str(artifact_path), asdict(control),
        asdict(config), str(output_dir),
    )
    if workers == 1:
        _initialize(*initargs)
        for seed in pending:
            results[seed] = _worker(seed)
    elif pending:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_initialize, initargs=initargs
        ) as pool:
            futures = {pool.submit(_worker, seed): seed for seed in pending}
            for future in as_completed(futures):
                seed = futures[future]
                results[seed] = future.result()
                print(
                    f"seed={seed} strict="
                    f"{results[seed]['success_30'] and results[seed]['success_100']}",
                    flush=True,
                )
    payload = {
        "method": "always_on_chart_local_contextual_atlas_deepc",
        "baseline_source": str(baseline_dir),
        "config": asdict(control),
        "always_on_contextual": asdict(config),
        "offline_build_seconds_this_run": build_seconds,
        "fairness_audit": audit_iid_library(H, eval_seeds=seeds),
        "results": [results[seed] for seed in seeds],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-stop", type=int, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-artifact", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--modes-per-chart", type=int, default=8)
    parser.add_argument("--columns-per-mode", type=int, default=350)
    parser.add_argument("--mode-cluster-seed", type=int, default=37)
    parser.add_argument(
        "--reference-mode",
        choices=("graph_routed", "final_only"),
        default="graph_routed",
    )
    parser.add_argument("--graph-weight", type=float, default=0.5)
    parser.add_argument("--switch-penalty", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    payload = run_panel(
        range(args.seed_start, args.seed_stop + 1), args.baseline_dir,
        args.output_dir, ChartLocalContextualConfig(
            modes_per_chart=args.modes_per_chart,
            columns_per_mode=args.columns_per_mode,
            mode_cluster_seed=args.mode_cluster_seed,
            reference_mode=args.reference_mode,
            graph_weight=args.graph_weight,
            switch_penalty=args.switch_penalty,
        ), workers=args.workers,
        overwrite=args.overwrite, source_artifact=args.source_artifact,
    )
    successes = sum(
        row["success_30"] and row["success_100"] for row in payload["results"]
    )
    print(json.dumps({"successes": successes, "trials": len(payload["results"])}, indent=2))


if __name__ == "__main__":
    main()
