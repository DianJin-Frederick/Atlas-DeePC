"""Contextually gated expansion of cached Atlas-DeePC chart matrices.

Graph routing, references, and the DeePC QP are identical to the frozen Atlas
baseline. While the exact terminal reference is active, Contextual Sampling's
past/future score is evaluated only to diagnose whether the current chart is
too narrow. If the selected top-N columns are insufficiently concentrated in
the current chart, one strongly represented alternate chart is added and the
cached rank-60 basis of that chart pair is used. No online SVD is performed.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

import gymnasium as gym
import numpy as np

from atlas_deepc.control import expert_pipeline_search as pipe
from atlas_deepc.atlas import routed_deepc_experiment as routed
from atlas_deepc.common.reacher_data import audit_iid_library
from atlas_deepc.benchmarks.online_selected_deepc_comparison import (
    build_controller,
    contextual_distance_components,
    prepare_contextual_library,
    select_contextual_columns,
    timing_summary,
)
from atlas_deepc.atlas.goal_reference import chart_compatible_goal


RESULT_ROOT = Path("experiments/routed_deepc_iid")
DEFAULT_DATA_PATH = Path("data/select_iid_offline_hankel.npy")
DEFAULT_DEVELOPMENT_BASELINE = (
    RESULT_ROOT / "clustering_jointconfig48_benchmark111_210"
)
DEFAULT_EVALUATION_BASELINE = (
    RESULT_ROOT / "clustering_jointconfig48_fresh211_310"
)


@dataclass(frozen=True)
class ExpansionConfig:
    selected_columns: int = 200
    current_share_threshold: float = 0.70
    minimum_alternate_share: float = 0.15
    terminal_only: bool = True
    selection_mode: str = "top_column_histogram"
    chart_score_top: int = 20
    chart_score_margin: float = 0.25
    candidate_scope: str = "graph_neighbors"


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


def baseline_config(payload: dict) -> routed.RoutedDeePCConfig:
    allowed = {field.name for field in fields(routed.RoutedDeePCConfig)}
    values = {
        key: value for key, value in payload["config"].items() if key in allowed
    }
    return routed.RoutedDeePCConfig(**values)


def contextual_chart_choice(
    selected: np.ndarray,
    labels: np.ndarray,
    current_chart: int,
    terminal_reference_active: bool,
    config: ExpansionConfig,
) -> tuple[tuple[int, ...], dict]:
    counts = np.bincount(labels[selected], minlength=int(labels.max()) + 1)
    selected_count = max(int(selected.size), 1)
    current_share = float(counts[current_chart] / selected_count)
    alternate_order = np.argsort(-counts, kind="stable")
    alternate = next(
        (int(chart) for chart in alternate_order if int(chart) != current_chart),
        current_chart,
    )
    alternate_share = float(counts[alternate] / selected_count)
    eligible_time = terminal_reference_active or not config.terminal_only
    expand = (
        eligible_time
        and current_share < config.current_share_threshold
        and alternate != current_chart
        and alternate_share >= config.minimum_alternate_share
    )
    active_charts = (
        tuple(sorted((current_chart, alternate))) if expand else (current_chart,)
    )
    return active_charts, {
        "current_contextual_share": current_share,
        "alternate_contextual_chart": alternate,
        "alternate_contextual_share": alternate_share,
        "chart_expanded": expand,
        "current_contextual_chart_score": np.nan,
        "alternate_contextual_chart_score": np.nan,
    }


def contextual_score_chart_choice(
    chart_column_score: np.ndarray,
    labels: np.ndarray,
    atlas: dict,
    current_chart: int,
    terminal_reference_active: bool,
    edge_count_threshold: int,
    config: ExpansionConfig,
) -> tuple[tuple[int, ...], dict]:
    chart_count = int(atlas["centers"].shape[0])
    if config.candidate_scope == "graph_neighbors":
        transitions = np.asarray(atlas["transition_counts"], dtype=int)
        adjacent = (
            (transitions[current_chart] >= edge_count_threshold)
            | (transitions[:, current_chart] >= edge_count_threshold)
        )
        candidates = np.flatnonzero(adjacent).astype(int).tolist()
        if current_chart not in candidates:
            candidates.append(current_chart)
    elif config.candidate_scope == "all":
        candidates = list(range(chart_count))
    else:
        raise ValueError(f"Unknown candidate_scope: {config.candidate_scope}")

    chart_scores = np.full(chart_count, np.inf, dtype=float)
    for chart in candidates:
        values = chart_column_score[labels == chart]
        count = min(config.chart_score_top, values.size)
        if count:
            chart_scores[chart] = float(
                np.mean(np.partition(values, count - 1)[:count])
            )
    alternate = int(min(candidates, key=lambda chart: chart_scores[chart]))
    current_score = float(chart_scores[current_chart])
    alternate_score = float(chart_scores[alternate])
    expand = (
        terminal_reference_active
        and alternate != current_chart
        and alternate_score + config.chart_score_margin < current_score
    )
    active_charts = (
        tuple(sorted((current_chart, alternate))) if expand else (current_chart,)
    )
    return active_charts, {
        "current_contextual_share": np.nan,
        "alternate_contextual_chart": alternate,
        "alternate_contextual_share": np.nan,
        "chart_expanded": expand,
        "current_contextual_chart_score": current_score,
        "alternate_contextual_chart_score": alternate_score,
    }


def run_contextual_chart_expansion(
    H: np.ndarray,
    atlas: dict,
    config: routed.RoutedDeePCConfig,
    expansion: ExpansionConfig,
    contextual_library: dict | None = None,
    matrix_cache: dict | None = None,
    save_path: Path | None = None,
) -> dict:
    if expansion.selected_columns > H.shape[1]:
        raise ValueError("selected_columns exceeds the offline library size.")
    contextual_library = (
        prepare_contextual_library(H)
        if contextual_library is None
        else contextual_library
    )
    matrix_cache = {} if matrix_cache is None else matrix_cache
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
    records = {
        key: []
        for key in (
            "y", "u", "pred", "ref", "distance", "prediction_error_xy",
            "status", "chart", "goal_chart", "next_chart", "path_length",
            "active_charts", "matrix_columns", "chart_expanded",
            "current_contextual_share", "alternate_contextual_chart",
            "alternate_contextual_share", "current_contextual_chart_score",
            "alternate_contextual_chart_score", "localization_time_s",
            "routing_time_s", "selection_time_s", "matrix_preparation_time_s",
            "solve_time_s", "control_compute_time_s",
        )
    }

    for _ in range(config.steps):
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

        selection_start = time.perf_counter()
        if terminal_reference_active or not expansion.terminal_only:
            if expansion.selection_mode == "future_chart_score":
                past_score, future_score = contextual_distance_components(
                    contextual_library, y_ini, reference, Q
                )
                contextual_score = past_score + future_score
                selected = np.argsort(
                    contextual_score, kind="stable"
                )[: expansion.selected_columns]
            else:
                selected, contextual_score = select_contextual_columns(
                    contextual_library,
                    y_ini,
                    reference,
                    Q,
                    expansion.selected_columns,
                )
            if expansion.selection_mode == "top_column_histogram":
                active_charts, selection = contextual_chart_choice(
                    selected,
                    labels,
                    current_chart,
                    terminal_reference_active,
                    expansion,
                )
            elif expansion.selection_mode in ("chart_score", "future_chart_score"):
                active_charts, selection = contextual_score_chart_choice(
                    (
                        future_score
                        if expansion.selection_mode == "future_chart_score"
                        else contextual_score
                    ),
                    labels,
                    atlas,
                    current_chart,
                    terminal_reference_active,
                    config.edge_count_threshold,
                    expansion,
                )
            else:
                raise ValueError(
                    f"Unknown selection_mode: {expansion.selection_mode}"
                )
        else:
            active_charts = (current_chart,)
            selection = {
                "current_contextual_share": np.nan,
                "alternate_contextual_chart": -1,
                "alternate_contextual_share": np.nan,
                "chart_expanded": False,
                "current_contextual_chart_score": np.nan,
                "alternate_contextual_chart_score": np.nan,
            }
        selection_time = time.perf_counter() - selection_start

        matrix_start = time.perf_counter()
        if active_charts not in matrix_cache:
            matrix_cache[active_charts] = routed.chart_matrix(
                H, atlas, active_charts, config
            )
        matrix, selected_matrix_columns = matrix_cache[active_charts]
        matrix_time = time.perf_counter() - matrix_start

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
        records["active_charts"].append(active_charts)
        records["matrix_columns"].append(selected_matrix_columns.size)
        for key, value in selection.items():
            records[key].append(value)
        records["localization_time_s"].append(localization_time)
        records["routing_time_s"].append(routing_time)
        records["selection_time_s"].append(selection_time)
        records["matrix_preparation_time_s"].append(matrix_time)
        records["solve_time_s"].append(solve_time)
        records["control_compute_time_s"].append(compute_time)
    env.close()

    arrays = {
        key: np.asarray(
            value,
            dtype=object if key in ("status", "active_charts") else None,
        )
        for key, value in records.items()
    }
    distance = arrays["distance"].astype(float)
    tail_30 = distance[-30:]
    tail_100 = distance[-min(100, distance.size):]
    summary = {
        **asdict(config),
        "method": "contextually_gated_cached_chart_expansion",
        "expansion": asdict(expansion),
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
        "chart_expansion_fraction": float(
            np.mean(arrays["chart_expanded"].astype(bool))
        ),
        "terminal_chart_expansion_fraction": float(
            np.mean(arrays["chart_expanded"][-100:].astype(bool))
        ),
        "statuses": dict(Counter(arrays["status"].tolist())),
        "charts_visited": int(np.unique(arrays["chart"].astype(int)).size),
        "controller": "DeePC behavioral column combination",
        "selector_role": "cached Atlas chart expansion only; no online SVD",
        "fitted_dynamics_model": False,
        "online_nearest_trajectory_retrieval": False,
        "online_contextual_column_scoring": True,
        "use_ik_ref": False,
        "feedback_blend": 0.0,
        "feedback_fallback": False,
        **timing_summary(arrays["localization_time_s"], "localization"),
        **timing_summary(arrays["routing_time_s"], "routing"),
        **timing_summary(arrays["selection_time_s"], "selection"),
        **timing_summary(arrays["matrix_preparation_time_s"], "matrix_preparation"),
        **timing_summary(arrays["solve_time_s"], "solve"),
        **timing_summary(arrays["control_compute_time_s"], "control_compute"),
        **goal_metadata,
    }
    if save_path is not None:
        np.savez(
            save_path,
            **arrays,
            summary=np.array(json.dumps(summary), dtype=object),
        )
    return summary


_WORKER_H = None
_WORKER_ATLAS = None
_WORKER_CONFIG = None
_WORKER_EXPANSION = None
_WORKER_LIBRARY = None
_WORKER_MATRIX_CACHE = None
_WORKER_OUTPUT_DIR = None


def initialize_worker(
    data_path: str,
    atlas_path: str,
    config_values: dict,
    expansion_values: dict,
    output_dir: str,
) -> None:
    global _WORKER_H, _WORKER_ATLAS, _WORKER_CONFIG, _WORKER_EXPANSION
    global _WORKER_LIBRARY, _WORKER_MATRIX_CACHE, _WORKER_OUTPUT_DIR
    _WORKER_H = np.load(data_path)
    _WORKER_ATLAS = routed.load_atlas(Path(atlas_path))
    _WORKER_CONFIG = routed.RoutedDeePCConfig(**config_values)
    _WORKER_EXPANSION = ExpansionConfig(**expansion_values)
    _WORKER_LIBRARY = prepare_contextual_library(_WORKER_H)
    _WORKER_MATRIX_CACHE = {}
    _WORKER_OUTPUT_DIR = Path(output_dir)


def run_worker(seed: int) -> dict:
    run_config = routed.RoutedDeePCConfig(
        **{**asdict(_WORKER_CONFIG), "seed": int(seed)}
    )
    return run_contextual_chart_expansion(
        _WORKER_H,
        _WORKER_ATLAS,
        run_config,
        _WORKER_EXPANSION,
        contextual_library=_WORKER_LIBRARY,
        matrix_cache=_WORKER_MATRIX_CACHE,
        save_path=_WORKER_OUTPUT_DIR / f"seed{seed}_control_results.npz",
    )


def load_saved_summary(path: Path) -> dict:
    result = np.load(path, allow_pickle=True)
    return json.loads(result["summary"].item())


def run_panel(
    seeds: list[int],
    baseline_dir: Path,
    output_dir: Path,
    expansion: ExpansionConfig,
    data_path: Path = DEFAULT_DATA_PATH,
    workers: int = 1,
    overwrite: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_payload = json.loads((baseline_dir / "summary.json").read_text())
    config = baseline_config(baseline_payload)
    H = np.load(data_path, mmap_mode="r")
    audit = audit_iid_library(H, eval_seeds=seeds)
    results_by_seed = {}
    pending = []
    for seed in seeds:
        path = output_dir / f"seed{seed}_control_results.npz"
        if path.exists() and not overwrite:
            result = load_saved_summary(path)
            if result.get("expansion") != asdict(expansion):
                raise RuntimeError(
                    f"{path} uses different expansion settings; use a new output "
                    "directory or overwrite=True."
                )
            results_by_seed[seed] = result
        else:
            pending.append(seed)

    if workers == 1:
        initialize_worker(
            str(data_path),
            str(baseline_dir / "behavioral_atlas.npz"),
            asdict(config),
            asdict(expansion),
            str(output_dir),
        )
        for seed in pending:
            results_by_seed[seed] = run_worker(seed)
            result = results_by_seed[seed]
            print(
                f"seed={seed} success={result['success_30'] and result['success_100']} "
                f"final100={1000 * result['mean_last_100']:.3f} mm",
                flush=True,
            )
    elif pending:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=initialize_worker,
            initargs=(
                str(data_path),
                str(baseline_dir / "behavioral_atlas.npz"),
                asdict(config),
                asdict(expansion),
                str(output_dir),
            ),
        ) as executor:
            futures = {executor.submit(run_worker, seed): seed for seed in pending}
            for future in as_completed(futures):
                seed = futures[future]
                results_by_seed[seed] = future.result()
                result = results_by_seed[seed]
                print(
                    f"seed={seed} success={result['success_30'] and result['success_100']} "
                    f"final100={1000 * result['mean_last_100']:.3f} mm",
                    flush=True,
                )

    results = [results_by_seed[seed] for seed in seeds]
    payload = {
        "method": "contextually_gated_cached_chart_expansion",
        "baseline_source": str(baseline_dir),
        "config": asdict(config),
        "expansion": asdict(expansion),
        "fairness_audit": audit,
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-stop", type=int, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--selected-columns", type=int, default=200)
    parser.add_argument("--current-share-threshold", type=float, default=0.70)
    parser.add_argument("--minimum-alternate-share", type=float, default=0.15)
    parser.add_argument(
        "--selection-mode",
        choices=("top_column_histogram", "chart_score", "future_chart_score"),
        default="top_column_histogram",
    )
    parser.add_argument("--chart-score-top", type=int, default=20)
    parser.add_argument("--chart-score-margin", type=float, default=0.25)
    parser.add_argument(
        "--candidate-scope",
        choices=("graph_neighbors", "all"),
        default="graph_neighbors",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expansion = ExpansionConfig(
        selected_columns=args.selected_columns,
        current_share_threshold=args.current_share_threshold,
        minimum_alternate_share=args.minimum_alternate_share,
        selection_mode=args.selection_mode,
        chart_score_top=args.chart_score_top,
        chart_score_margin=args.chart_score_margin,
        candidate_scope=args.candidate_scope,
    )
    run_panel(
        seeds=list(range(args.seed_start, args.seed_stop + 1)),
        baseline_dir=args.baseline_dir,
        output_dir=args.output_dir,
        expansion=expansion,
        data_path=args.data_path,
        workers=args.workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
