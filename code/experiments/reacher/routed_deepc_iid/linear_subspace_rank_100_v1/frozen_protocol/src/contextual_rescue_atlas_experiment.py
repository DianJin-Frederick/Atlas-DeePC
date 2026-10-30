"""Graph-routed Atlas-DeePC with causal contextual stagnation refinement.

The frozen Atlas controller is used normally. Once the exact terminal reference
is active, a rescue is latched only if fingertip error remains above a threshold
and fails to improve sufficiently over a fixed observation window. The rescue
uses Contextual Sampling's top-N score and the same rank-matched DeePC matrix.
The candidate pool may be global, current-chart only, or the current chart plus
one contextually identified alternate chart. No outcome, future online state,
fitted model, IK, or feedback controller is used by the trigger.
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

import expert_pipeline_search as pipe
import routed_deepc_experiment as routed
from contextual_chart_expansion_experiment import baseline_config, output_cost
from local_affine_atlas_experiment import audit_iid_library
from online_selected_deepc_comparison import (
    build_controller,
    contextual_distance_components,
    prepare_contextual_library,
    selected_affine_basis,
    timing_summary,
)
from stitched_atlas_mpc_experiment import chart_compatible_goal


DEFAULT_DATA_PATH = Path("data/select_iid_offline_hankel.npy")


@dataclass(frozen=True)
class RescueConfig:
    selected_columns: int = 200
    error_threshold: float = 0.01
    stall_window: int = 15
    minimum_progress: float = 0.001
    candidate_scope: str = "global"


def select_atlas_constrained_columns(
    library: dict,
    y_ini: np.ndarray,
    reference: np.ndarray,
    Q: np.ndarray,
    labels: np.ndarray,
    current_chart: int,
    rescue: RescueConfig,
) -> np.ndarray:
    past_score, future_score = contextual_distance_components(
        library, y_ini, reference, Q
    )
    score = past_score + future_score
    if rescue.candidate_scope == "global":
        pool = np.arange(labels.size)
    elif rescue.candidate_scope == "current_chart":
        pool = np.flatnonzero(labels == current_chart)
    elif rescue.candidate_scope == "current_plus_contextual_chart":
        global_top = np.argsort(score, kind="stable")[: rescue.selected_columns]
        counts = np.bincount(labels[global_top], minlength=int(labels.max()) + 1)
        alternate = next(
            (
                int(chart)
                for chart in np.argsort(-counts, kind="stable")
                if int(chart) != current_chart
            ),
            current_chart,
        )
        pool = np.flatnonzero(
            (labels == current_chart) | (labels == alternate)
        )
    else:
        raise ValueError(f"Unknown candidate_scope: {rescue.candidate_scope}")
    count = min(rescue.selected_columns, pool.size)
    return pool[np.argsort(score[pool], kind="stable")[:count]]


def run_contextual_rescue(
    H: np.ndarray,
    atlas: dict,
    config: routed.RoutedDeePCConfig,
    rescue: RescueConfig,
    contextual_library: dict | None = None,
    chart_matrix_cache: dict | None = None,
    save_path: Path | None = None,
) -> dict:
    contextual_library = (
        prepare_contextual_library(H)
        if contextual_library is None
        else contextual_library
    )
    # Atlas bases are offline artifacts. Contextual rescue bases are rebuilt
    # online only after the causal stagnation trigger fires.
    chart_matrix_cache = (
        routed.precompute_chart_matrix_cache(H, atlas, config)
        if chart_matrix_cache is None
        else chart_matrix_cache
    )
    Q = output_cost(config)
    controller = build_controller(config)
    labels = np.asarray(atlas["labels"], dtype=int)

    env = gym.make(
        pipe.REACHER_ENV_ID,
        max_episode_steps=config.steps + pipe.T_INI + 5,
    )
    obs, _ = env.reset(seed=config.seed)
    target_xy = np.asarray(obs[4:6], dtype=float).copy()
    initial_y = pipe.reacher_output_from_obs(obs)
    goal, goal_metadata = chart_compatible_goal(H, target_xy, initial_y, config)
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
    rescue_active = False
    rescue_trigger_step = -1
    records = {
        key: []
        for key in (
            "y", "u", "pred", "ref", "distance", "prediction_error_xy",
            "status", "chart", "goal_chart", "next_chart", "path_length",
            "rescue_active", "rescue_triggered", "stall_progress",
            "selected_indices", "matrix_columns", "localization_time_s",
            "routing_time_s", "selection_time_s", "svd_rebuild_time_s",
            "matrix_preparation_time_s", "solve_time_s", "control_compute_time_s",
        )
    }

    for step in range(config.steps):
        control_start = time.perf_counter()
        u_ini = np.asarray(u_history[-pipe.T_INI:]).reshape(
            pipe.M * pipe.T_INI, 1
        )
        y_ini = np.asarray(y_history[-pipe.T_INI:]).reshape(
            pipe.P * pipe.T_INI, 1
        )

        stage = time.perf_counter()
        feature = routed.online_behavioral_feature(
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
        current_chart = int(routed.assign_atlas_charts(atlas, feature)[0])
        localization_time = time.perf_counter() - stage

        stage = time.perf_counter()
        path = routed.shortest_chart_path(
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
        terminal_reference_active = use_target or current_chart == goal_chart
        active_reference = (
            goal
            if terminal_reference_active
            else atlas["medoid_states"][next_chart]
        )
        reference = np.tile(active_reference.reshape(pipe.P, 1), (pipe.T_F, 1))
        routing_time = time.perf_counter() - stage

        stall_progress = np.nan
        rescue_triggered = False
        if terminal_reference_active:
            terminal_distance_history.append(distance_before)
            if len(terminal_distance_history) > rescue.stall_window:
                stall_progress = (
                    terminal_distance_history[-rescue.stall_window - 1]
                    - terminal_distance_history[-1]
                )
                if (
                    not rescue_active
                    and distance_before > rescue.error_threshold
                    and stall_progress < rescue.minimum_progress
                ):
                    rescue_active = True
                    rescue_triggered = True
                    rescue_trigger_step = step
        elif not rescue_active:
            terminal_distance_history.clear()

        selection_start = time.perf_counter()
        selected = np.empty(0, dtype=int)
        if rescue_active:
            selected = select_atlas_constrained_columns(
                contextual_library,
                y_ini,
                reference,
                Q,
                labels,
                current_chart,
                rescue,
            )
        selection_time = time.perf_counter() - selection_start

        matrix_start = time.perf_counter()
        if rescue_active:
            matrix = selected_affine_basis(H, selected, config)
            matrix_columns = selected.size
        else:
            key = (current_chart,)
            if key not in chart_matrix_cache:
                chart_matrix_cache[key] = routed.chart_matrix(H, atlas, key, config)
            matrix, matrix_indices = chart_matrix_cache[key]
            matrix_columns = matrix_indices.size
        matrix_time = time.perf_counter() - matrix_start
        svd_rebuild_time = matrix_time if rescue_active else 0.0

        solve_start = time.perf_counter()
        u_solution, y_solution, _g, status, _ = controller.solve(
            matrix, u_ini, y_ini, reference
        )
        solve_time = time.perf_counter() - solve_start
        if u_solution is None:
            action = np.zeros(pipe.M)
            prediction = np.full(pipe.P, np.nan)
        else:
            action = np.clip(
                u_solution[:pipe.M, 0], -config.u_limit, config.u_limit
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
        records["rescue_active"].append(rescue_active)
        records["rescue_triggered"].append(rescue_triggered)
        records["stall_progress"].append(stall_progress)
        records["selected_indices"].append(selected)
        records["matrix_columns"].append(matrix_columns)
        records["localization_time_s"].append(localization_time)
        records["routing_time_s"].append(routing_time)
        records["selection_time_s"].append(selection_time)
        records["svd_rebuild_time_s"].append(svd_rebuild_time)
        records["matrix_preparation_time_s"].append(matrix_time)
        records["solve_time_s"].append(solve_time)
        records["control_compute_time_s"].append(compute_time)
    env.close()

    arrays = {
        key: np.asarray(
            value,
            dtype=object if key in ("status", "selected_indices") else None,
        )
        for key, value in records.items()
    }
    distance = arrays["distance"].astype(float)
    tail_30 = distance[-30:]
    tail_100 = distance[-min(100, distance.size):]
    reference_cost = pipe.closed_loop_control_cost(
        arrays["y"], arrays["u"], arrays["ref"], Q, config.r * np.eye(pipe.M)
    )
    final_goal_cost = pipe.closed_loop_control_cost(
        arrays["y"],
        arrays["u"],
        goal,
        Q,
        config.r * np.eye(pipe.M),
        reference_description="common final graph-compatible goal",
    )
    summary = {
        **asdict(config),
        "method": (
            "graph_routed_atlas_with_contextual_stagnation_rescue"
            if rescue.candidate_scope == "global"
            else "graph_routed_atlas_with_hierarchical_contextual_refinement"
        ),
        "rescue": asdict(rescue),
        "rescue_triggered": bool(rescue_trigger_step >= 0),
        "rescue_trigger_step": int(rescue_trigger_step),
        "rescue_usage_fraction": float(np.mean(arrays["rescue_active"].astype(bool))),
        "graph_routing_used": True,
        "qp_count": config.steps,
        "optimized_coefficient_dimension": config.svd_rank,
        "minimum": float(distance.min()),
        "minimum_step": int(distance.argmin()),
        "last": float(distance[-1]),
        "mean_last_30": float(tail_30.mean()),
        "max_last_30": float(tail_30.max()),
        "mean_last_100": float(tail_100.mean()),
        "max_last_100": float(tail_100.max()),
        "success_30": bool(tail_30.mean() < 0.01 and tail_30.max() < 0.02),
        "success_100": bool(tail_100.mean() < 0.01 and tail_100.max() < 0.02),
        "mean_prediction_error_xy": float(
            np.nanmean(arrays["prediction_error_xy"].astype(float))
        ),
        "statuses": dict(Counter(arrays["status"].tolist())),
        "controller": (
            "Atlas-DeePC with causally triggered contextual refinement "
            f"({rescue.candidate_scope})"
        ),
        "contextual_candidate_scope": rescue.candidate_scope,
        "fitted_dynamics_model": False,
        "online_nearest_trajectory_retrieval": bool(rescue_trigger_step >= 0),
        "use_ik_ref": False,
        "feedback_blend": 0.0,
        "feedback_fallback": False,
        "chart_basis_precomputed_offline": True,
        "control_compute_excludes_offline_chart_svd": True,
        "control_compute_includes_svd_rebuild": True,
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
_CONFIG = None
_RESCUE = None
_LIBRARY = None
_MATRIX_CACHE = None
_OUTPUT_DIR = None
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


def compact_result_summary(result):
    """Keep only the fields needed for success-rate and compute-time analysis."""
    missing = [key for key in COMPACT_RESULT_FIELDS if key not in result]
    if missing:
        raise KeyError(f"Compact result is missing fields: {missing}")
    return {key: result[key] for key in COMPACT_RESULT_FIELDS}


def initialize_worker(
    data_path, atlas_path, config, rescue, output_dir, save_rollouts=True
):
    global _H, _ATLAS, _CONFIG, _RESCUE, _LIBRARY, _MATRIX_CACHE, _OUTPUT_DIR
    global _SAVE_ROLLOUTS
    _H = np.load(data_path)
    _ATLAS = routed.load_atlas(Path(atlas_path))
    _CONFIG = routed.RoutedDeePCConfig(**config)
    _RESCUE = RescueConfig(**rescue)
    _LIBRARY = prepare_contextual_library(_H)
    _MATRIX_CACHE = routed.precompute_chart_matrix_cache(_H, _ATLAS, _CONFIG)
    _OUTPUT_DIR = Path(output_dir)
    _SAVE_ROLLOUTS = bool(save_rollouts)


def run_worker(seed):
    config = routed.RoutedDeePCConfig(**{**asdict(_CONFIG), "seed": int(seed)})
    save_path = (
        _OUTPUT_DIR / f"seed{seed}_control_results.npz"
        if _SAVE_ROLLOUTS
        else None
    )
    return run_contextual_rescue(
        _H,
        _ATLAS,
        config,
        _RESCUE,
        contextual_library=_LIBRARY,
        chart_matrix_cache=_MATRIX_CACHE,
        save_path=save_path,
    )


def load_summary(path):
    data = np.load(path, allow_pickle=True)
    return json.loads(data["summary"].item())


_BASELINE_H = None
_BASELINE_ATLAS = None
_BASELINE_CONFIG = None
_BASELINE_OUTPUT_DIR = None


def initialize_baseline_worker(data_path, atlas_path, config, output_dir):
    global _BASELINE_H, _BASELINE_ATLAS, _BASELINE_CONFIG, _BASELINE_OUTPUT_DIR
    _BASELINE_H = np.load(data_path)
    _BASELINE_ATLAS = routed.load_atlas(Path(atlas_path))
    _BASELINE_CONFIG = routed.RoutedDeePCConfig(**config)
    _BASELINE_OUTPUT_DIR = Path(output_dir)


def run_baseline_worker(seed):
    config = routed.RoutedDeePCConfig(
        **{**asdict(_BASELINE_CONFIG), "seed": int(seed)}
    )
    path = _BASELINE_OUTPUT_DIR / f"seed{seed}_control_results.npz"
    result = routed.run_routed_deepc(
        _BASELINE_H,
        _BASELINE_ATLAS,
        config,
        save_path=path,
    )
    result["result_path"] = str(path)
    return result


def _summary_matches_config(summary, config):
    """Reject bootstrapped rows produced by a materially different controller."""
    ignored = {"seed"}
    for key, expected in asdict(config).items():
        if key not in ignored and key in summary and summary[key] != expected:
            return False
    return True


def run_resumable_baseline_panel(
    seeds,
    template_dir,
    output_dir,
    data_path=DEFAULT_DATA_PATH,
    bootstrap_dirs=(),
    workers=1,
    overwrite=False,
):
    """Build a paired Atlas baseline while reusing compatible saved rollouts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    template_payload = json.loads((template_dir / "summary.json").read_text())
    config = baseline_config(template_payload)
    H = np.load(data_path, mmap_mode="r")
    audit = audit_iid_library(H, eval_seeds=seeds)
    atlas_path = template_dir / "behavioral_atlas.npz"
    atlas = routed.load_atlas(atlas_path)
    routed.save_atlas(output_dir / "behavioral_atlas.npz", atlas, config)

    results = {}
    source_dirs = (output_dir, *tuple(Path(path) for path in bootstrap_dirs))
    if not overwrite:
        for source_dir in source_dirs:
            for seed in seeds:
                if seed in results:
                    continue
                path = source_dir / f"seed{seed}_control_results.npz"
                if not path.exists():
                    continue
                summary = load_summary(path)
                if not _summary_matches_config(summary, config):
                    raise RuntimeError(f"Incompatible baseline result: {path}")
                results[seed] = summary

    pending = [seed for seed in seeds if seed not in results]
    initargs = (
        str(data_path),
        str(atlas_path),
        asdict(config),
        str(output_dir),
    )
    if workers == 1:
        initialize_baseline_worker(*initargs)
        for seed in pending:
            results[seed] = run_baseline_worker(seed)
            print(
                f"baseline seed={seed} "
                f"success={results[seed]['success_30'] and results[seed]['success_100']} "
                f"final100={1000 * results[seed]['mean_last_100']:.3f} mm",
                flush=True,
            )
    elif pending:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=initialize_baseline_worker,
            initargs=initargs,
        ) as executor:
            futures = {
                executor.submit(run_baseline_worker, seed): seed for seed in pending
            }
            for future in as_completed(futures):
                seed = futures[future]
                results[seed] = future.result()
                print(
                    f"baseline seed={seed} "
                    f"success={results[seed]['success_30'] and results[seed]['success_100']} "
                    f"final100={1000 * results[seed]['mean_last_100']:.3f} mm",
                    flush=True,
                )

    sizes = np.bincount(atlas["labels"].astype(int), minlength=config.clusters)
    payload = {
        "method": "graph_routed_fixed_chart_deepc",
        "config": asdict(config),
        "fairness_audit": audit,
        "chart_sizes": sizes.tolist(),
        "bootstrap_dirs": [str(path) for path in bootstrap_dirs],
        "results": [results[seed] for seed in seeds],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def run_panel(
    seeds,
    baseline_dir,
    output_dir,
    rescue,
    data_path=DEFAULT_DATA_PATH,
    workers=1,
    overwrite=False,
    config_override=None,
    bootstrap_dirs=(),
    save_rollouts=True,
    compact_results=False,
    delete_local_rollouts_after_compaction=False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_payload = json.loads((baseline_dir / "summary.json").read_text())
    template_config = baseline_config(baseline_payload)
    config = template_config if config_override is None else config_override
    if not isinstance(config, routed.RoutedDeePCConfig):
        raise TypeError("config_override must be a RoutedDeePCConfig")
    if compact_results and save_rollouts:
        raise ValueError("compact_results requires save_rollouts=False")
    if delete_local_rollouts_after_compaction and not compact_results:
        raise ValueError(
            "delete_local_rollouts_after_compaction requires compact_results=True"
        )
    H = np.load(data_path, mmap_mode="r")
    audit = audit_iid_library(H, eval_seeds=seeds)
    results = {}
    checkpoint_path = output_dir / "compact_results.jsonl"
    checkpoint_metadata_path = output_dir / "compact_results_metadata.json"
    if compact_results:
        checkpoint_metadata = {
            "config": asdict(config),
            "rescue": asdict(rescue),
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
                saved_payload.get("config") == asdict(config)
                and saved_payload.get("rescue") == asdict(rescue)
            ):
                for result in saved_payload.get("results", []):
                    results[int(result["seed"])] = compact_result_summary(result)
        if checkpoint_path.exists() and not overwrite:
            for line_number, line in enumerate(
                checkpoint_path.read_text().splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    result = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"Invalid compact checkpoint line {line_number}: "
                        f"{checkpoint_path}"
                    ) from error
                results[int(result["seed"])] = compact_result_summary(result)

    def retain_result(result):
        retained = compact_result_summary(result) if compact_results else result
        if compact_results:
            with checkpoint_path.open("a") as checkpoint:
                checkpoint.write(json.dumps(retained) + "\n")
        return retained

    source_dirs = (output_dir, *tuple(Path(path) for path in bootstrap_dirs))
    if not overwrite:
        for source_dir in source_dirs:
            for seed in seeds:
                if seed in results:
                    continue
                path = source_dir / f"seed{seed}_control_results.npz"
                if not path.exists():
                    continue
                result = load_summary(path)
                if (
                    result.get("rescue") != asdict(rescue)
                    or not _summary_matches_config(result, config)
                ):
                    raise RuntimeError(f"Incompatible existing result: {path}")
                results[seed] = retain_result(result)
                if (
                    delete_local_rollouts_after_compaction
                    and source_dir.resolve() == output_dir.resolve()
                ):
                    path.unlink()
    pending = [seed for seed in seeds if seed not in results]

    initargs = (
        str(data_path),
        str(baseline_dir / "behavioral_atlas.npz"),
        asdict(config),
        asdict(rescue),
        str(output_dir),
        save_rollouts,
    )
    if workers == 1:
        initialize_worker(*initargs)
        for seed in pending:
            full_result = run_worker(seed)
            results[seed] = retain_result(full_result)
            print(
                f"seed={seed} "
                f"success={full_result['success_30'] and full_result['success_100']} "
                f"median_control={full_result['median_control_compute_ms']:.2f} ms",
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
                full_result = future.result()
                results[seed] = retain_result(full_result)
                print(
                    f"seed={seed} "
                    f"success={full_result['success_30'] and full_result['success_100']} "
                    f"median_control={full_result['median_control_compute_ms']:.2f} ms",
                    flush=True,
                )

    payload = {
        "method": (
            "graph_routed_atlas_with_contextual_stagnation_rescue"
            if rescue.candidate_scope == "global"
            else "graph_routed_atlas_with_hierarchical_contextual_refinement"
        ),
        "baseline_source": str(baseline_dir),
        "config": asdict(config),
        "rescue": asdict(rescue),
        "fairness_audit": audit,
        "bootstrap_dirs": [str(path) for path in bootstrap_dirs],
        "save_rollouts": bool(save_rollouts),
        "compact_results": bool(compact_results),
        "results": [results[seed] for seed in seeds],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-stop", type=int, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--error-threshold", type=float, default=0.01)
    parser.add_argument("--stall-window", type=int, default=15)
    parser.add_argument("--minimum-progress", type=float, default=0.001)
    parser.add_argument(
        "--candidate-scope",
        choices=("global", "current_chart", "current_plus_contextual_chart"),
        default="global",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_panel(
        list(range(args.seed_start, args.seed_stop + 1)),
        args.baseline_dir,
        args.output_dir,
        RescueConfig(
            error_threshold=args.error_threshold,
            stall_window=args.stall_window,
            minimum_progress=args.minimum_progress,
            candidate_scope=args.candidate_scope,
        ),
        workers=args.workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
