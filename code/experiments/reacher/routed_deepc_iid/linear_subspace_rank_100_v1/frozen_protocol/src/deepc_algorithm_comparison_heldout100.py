"""Paired multi-seed six-way DeePC benchmark.

The online selectors use the exact frozen settings from
``deepc_algorithm_comparison.py``.  Existing Atlas results from the authoritative
48-chart seeds-111--210 benchmark are reused because their data, atlas, router,
reference policy, controller, rank, solver, horizon, and seed panel are identical.
Select-DPC and both contextual variants rebuild the same centered rank-60 SVD
representation from their selected columns at every step. Their online timing
includes that SVD; Atlas chart bases are precomputed offline.
Missing results are saved per seed so the long experiment is safely resumable.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from itertools import combinations
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np

import expert_pipeline_search as pipe
from contextual_rescue_atlas_experiment import RescueConfig, run_contextual_rescue
from deepc_algorithm_comparison import DEFAULT_ATLAS_PATH
from future_output_atlas_experiment import (
    FutureOutputAtlasConfig,
    run_panel as run_future_output_panel,
)
from local_affine_atlas_experiment import environment_target
from online_selected_deepc_comparison import (
    prepare_contextual_library,
    run_online_selected_deepc,
    timing_summary,
)
from routed_deepc_experiment import (
    DEFAULT_DATA_PATH,
    RoutedDeePCConfig,
    load_atlas,
    precompute_chart_matrix_cache,
    run_routed_deepc,
)


DEFAULT_BASELINE_SUMMARY = Path(
    "experiments/routed_deepc_iid/"
    "clustering_jointconfig48_benchmark111_210/summary.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/deepc_algorithm_comparison/five_way_svd_heldout111_610_n200_rank60"
)
DEFAULT_BOOTSTRAP_DIRS = (
    Path("experiments/deepc_algorithm_comparison/five_way_svd_heldout111_1210_n200_rank60"),
    Path("experiments/deepc_algorithm_comparison/four_way_svd_heldout111_210_n200_rank60"),
    Path("experiments/deepc_algorithm_comparison/heldout111_210_n200_rank60"),
    Path("experiments/deepc_algorithm_comparison/heldout111_610_n200_rank60"),
)
DEFAULT_RESCUE_BOOTSTRAP_DIRS = (
    Path("experiments/routed_deepc_iid/contextual_rescue_probe_w15"),
    Path("experiments/routed_deepc_iid/contextual_rescue_eval211_310"),
)
DEFAULT_TRIGGERED_CHART_BOOTSTRAP_DIRS = (
    Path("experiments/routed_deepc_iid/future_output_atlas_dev111_210"),
    Path("experiments/routed_deepc_iid/future_output_atlas_eval211_310"),
)
METHODS = (
    "select_dpc_svd",
    "contextual_graph_svd",
    "contextual_final_svd",
    "atlas_contextual_rescue",
    "triggered_contextual_chart",
)
ALL_METHODS = ("atlas", *METHODS)
LABELS = {
    "atlas": "Atlas-DeePC",
    "select_dpc_svd": "Select-DPC selector",
    "contextual_graph_svd": "Contextual + graph ref",
    "contextual_final_svd": "Contextual + final ref",
    "atlas_contextual_rescue": "Atlas + contextual rescue",
    "triggered_contextual_chart": "Triggered contextual chart union",
}


_H = None
_ATLAS = None
_CONFIG = None
_SELECTED_COLUMNS = None
_OUTPUT_DIR = None
_ATLAS_MATRIX_CACHE = None
_CONTEXTUAL_LIBRARY = None
_RESCUE = None


def initialize_worker(
    data_path: str,
    atlas_path: str,
    config_payload: dict,
    selected_columns: int,
    rescue_payload: dict,
    output_dir: str,
) -> None:
    global _H, _ATLAS, _CONFIG, _SELECTED_COLUMNS, _OUTPUT_DIR
    global _ATLAS_MATRIX_CACHE, _CONTEXTUAL_LIBRARY, _RESCUE
    _H = np.load(data_path)
    _ATLAS = load_atlas(Path(atlas_path))
    _CONFIG = RoutedDeePCConfig(**config_payload)
    _SELECTED_COLUMNS = int(selected_columns)
    _RESCUE = RescueConfig(**rescue_payload)
    _OUTPUT_DIR = Path(output_dir)
    _ATLAS_MATRIX_CACHE = precompute_chart_matrix_cache(_H, _ATLAS, _CONFIG)
    _CONTEXTUAL_LIBRARY = prepare_contextual_library(_H)


def result_path(output_dir: Path, method: str, seed: int) -> Path:
    return output_dir / method / f"seed{seed}_control_results.npz"


def bootstrap_method_names(method: str) -> tuple[str, ...]:
    """Map current explicit SVD names to compatible legacy rank-matched paths."""
    aliases = {
        "select_dpc_svd": ("select_dpc_svd", "select_dpc"),
        "contextual_graph_svd": ("contextual_graph_svd", "contextual"),
    }
    return aliases.get(method, (method,))


def final_goal_from_summary(H: np.ndarray, summary: dict) -> np.ndarray:
    outputs = H[pipe.M * pipe.L :].reshape(pipe.L, pipe.P, H.shape[1])[0].T
    goal = outputs[int(summary["data_goal_index"])].copy()
    goal[4:6] = 0.0
    goal[6:8] = environment_target(int(summary["seed"]))
    return goal


def update_control_compute_timing(summary: dict, seconds: np.ndarray) -> None:
    """Store uniform control-compute statistics for old and new summary schemas."""
    statistics = timing_summary(np.asarray(seconds, dtype=float), "control_compute")
    summary.update(statistics)
    for name in ("mean", "median", "p95", "p99", "max"):
        summary[f"{name}_control_compute_time_ms"] = statistics[
            f"{name}_control_compute_ms"
        ]


def add_saved_rollout_metrics(
    summary: dict,
    data,
    H: np.ndarray,
) -> dict:
    """Recompute costs from saved trajectories so rollouts never need rerunning."""
    goal = (
        np.asarray(data["final_goal"], dtype=float)
        if "final_goal" in data.files
        else final_goal_from_summary(H, summary)
    )
    Q = np.diag(
        [
            summary["q_data_goal_joint"],
            summary["q_data_goal_joint"],
            summary["q_data_goal_joint"],
            summary["q_data_goal_joint"],
            summary["q_dq"],
            summary["q_dq"],
            summary["q_xy"],
            summary["q_xy"],
            summary["q_data_goal_joint"],
            summary["q_data_goal_joint"],
        ]
    )
    R = summary["r"] * np.eye(pipe.M)
    active_reference = (
        np.asarray(data["ref"], dtype=float)
        if "ref" in data.files
        else goal
    )
    reference_cost = pipe.closed_loop_control_cost(
        data["y"], data["u"], active_reference, Q, R
    )
    final_goal_cost = pipe.closed_loop_control_cost(
        data["y"],
        data["u"],
        goal,
        Q,
        R,
        reference_description="common final graph-compatible goal",
    )
    summary.update(
        reference_cost,
        closed_loop_reference_cost=reference_cost["closed_loop_control_cost"],
        closed_loop_reference_tracking_cost=reference_cost["closed_loop_tracking_cost"],
        mean_closed_loop_reference_stage_cost=reference_cost["mean_closed_loop_stage_cost"],
        closed_loop_final_goal_cost=final_goal_cost["closed_loop_control_cost"],
        closed_loop_final_goal_tracking_cost=final_goal_cost["closed_loop_tracking_cost"],
        mean_closed_loop_final_goal_stage_cost=final_goal_cost["mean_closed_loop_stage_cost"],
        closed_loop_final_goal_cost_definition=final_goal_cost["closed_loop_cost_definition"],
    )
    if "control_compute_time_s" in data.files:
        update_control_compute_timing(summary, data["control_compute_time_s"])

    if (
        summary.get("selected_model") == "rank_matched"
        and "matrix_preparation_time_s" in data.files
    ):
        # Compatible legacy rank-matched rollouts stored the SVD construction
        # under matrix preparation before it received an explicit timing name.
        summary.update(
            timing_summary(
                np.asarray(data["matrix_preparation_time_s"], dtype=float),
                "svd_rebuild",
            )
        )
        summary["control_compute_includes_svd_rebuild"] = True

    if "rescue_active" in data.files:
        rescue_active = np.asarray(data["rescue_active"], dtype=bool)
        preparation = np.asarray(data["matrix_preparation_time_s"], dtype=float)
        svd_rebuild = np.where(rescue_active, preparation, 0.0)
        summary.update(timing_summary(svd_rebuild, "svd_rebuild"))
        summary["rescue_usage_fraction"] = float(np.mean(rescue_active))
        summary["control_compute_includes_svd_rebuild"] = True
        if not summary.get("control_compute_excludes_offline_chart_svd", False):
            total = np.asarray(data["control_compute_time_s"], dtype=float)
            cached_samples = preparation[~rescue_active]
            cached_lookup = (
                float(np.median(cached_samples)) if cached_samples.size else 0.0
            )
            offline_svd = np.where(
                rescue_active,
                0.0,
                np.maximum(preparation - cached_lookup, 0.0),
            )
            corrected = np.maximum(total - offline_svd, 0.0)
            update_control_compute_timing(summary, corrected)
            summary.update(
                control_compute_excludes_offline_chart_svd=True,
                rescue_legacy_timing_correction=(
                    "removed non-rescue matrix-preparation time above the median "
                    "cached Atlas-basis lookup cost"
                ),
            )

    # The authoritative Atlas baseline was recorded before chart bases were
    # explicitly precomputed. Correct only those legacy timings by removing
    # matrix-preparation spikes above the normal cached-lookup cost. New Atlas
    # rollouts precompute all bases and already need no correction.
    if (
        summary.get("online_nearest_trajectory_retrieval") is False
        and not summary.get("control_compute_excludes_offline_chart_svd", False)
        and "control_compute_time_s" in data.files
        and "matrix_preparation_time_s" in data.files
    ):
        total = np.asarray(data["control_compute_time_s"], dtype=float)
        preparation = np.asarray(data["matrix_preparation_time_s"], dtype=float)
        cached_lookup = float(np.median(preparation))
        offline_svd = np.maximum(preparation - cached_lookup, 0.0)
        corrected = np.maximum(total - offline_svd, 0.0)
        update_control_compute_timing(summary, corrected)
        summary.update(
            control_compute_excludes_offline_chart_svd=True,
            atlas_legacy_timing_correction=(
                "removed matrix-preparation time above the median cached-basis "
                "lookup cost"
            ),
        )
    return summary


def load_saved_summary(path: Path, H: np.ndarray) -> dict:
    with np.load(path, allow_pickle=True) as data:
        summary = json.loads(str(data["summary"].item()))
        return add_saved_rollout_metrics(summary, data, H)


def run_worker(task: tuple[int, str]) -> tuple[int, str, dict]:
    seed, method = task
    config = replace(_CONFIG, seed=int(seed))
    path = result_path(_OUTPUT_DIR, method, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Suppress the per-20-step trace inside workers; the parent reports episodes.
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        if method == "atlas":
            summary = run_routed_deepc(
                _H,
                _ATLAS,
                config,
                save_path=path,
                matrix_cache=_ATLAS_MATRIX_CACHE,
            )
        elif method == "select_dpc_svd":
            summary = run_online_selected_deepc(
                _H,
                _ATLAS,
                config,
                _SELECTED_COLUMNS,
                selected_model="rank_matched",
                selection_method="select_dpc",
                reference_mode="graph_routed",
                save_path=path,
            )
        elif method == "atlas_contextual_rescue":
            summary = run_contextual_rescue(
                _H,
                _ATLAS,
                config,
                _RESCUE,
                contextual_library=_CONTEXTUAL_LIBRARY,
                chart_matrix_cache=_ATLAS_MATRIX_CACHE,
                save_path=path,
            )
        else:
            summary = run_online_selected_deepc(
                _H,
                _ATLAS,
                config,
                _SELECTED_COLUMNS,
                selected_model="rank_matched",
                selection_method="contextual",
                reference_mode=(
                    "graph_routed"
                    if method == "contextual_graph_svd"
                    else "final_only"
                ),
                save_path=path,
            )
    return seed, method, summary


def strict_success(result: dict) -> bool:
    return bool(result["success_30"] and result["success_100"])


def wilson_interval(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    ) / denominator
    # At the 0/N and N/N boundaries, roundoff can put an endpoint a few ulps
    # outside [0, 1]. Matplotlib rejects the resulting negative yerr.
    return [
        float(np.clip(center - radius, 0.0, 1.0)),
        float(np.clip(center + radius, 0.0, 1.0)),
    ]


def exact_mcnemar_p(repaired: int, regressed: int) -> float:
    discordant = repaired + regressed
    if discordant == 0:
        return 1.0
    lower = min(repaired, regressed)
    probability = sum(
        math.comb(discordant, k) for k in range(lower + 1)
    ) / (2**discordant)
    return float(min(1.0, 2.0 * probability))


def control_timing_value(row: dict, statistic: str) -> float:
    for key in (
        f"{statistic}_control_compute_ms",
        f"{statistic}_control_compute_time_ms",
    ):
        if key in row:
            return float(row[key])
    raise KeyError(f"Missing {statistic} control timing in {sorted(row)}")


def method_summary(name: str, rows: dict[int, dict], seeds: list[int]) -> dict:
    successes = sum(strict_success(rows[seed]) for seed in seeds)
    result = {
        "method": name,
        "label": LABELS[name],
        "successes": int(successes),
        "total": len(seeds),
        "success_rate": float(successes / len(seeds)),
        "wilson_95": wilson_interval(successes, len(seeds)),
        "median_final100_mm": float(
            1000.0 * np.median([rows[s]["mean_last_100"] for s in seeds])
        ),
        "median_prediction_error_mm": float(
            1000.0
            * np.median([rows[s]["mean_prediction_error_xy"] for s in seeds])
        ),
        "median_closed_loop_control_cost": float(
            np.median([rows[s]["closed_loop_control_cost"] for s in seeds])
        ),
        "median_closed_loop_reference_cost": float(
            np.median([rows[s]["closed_loop_reference_cost"] for s in seeds])
        ),
        "median_closed_loop_final_goal_cost": float(
            np.median([rows[s]["closed_loop_final_goal_cost"] for s in seeds])
        ),
        "median_svd_rebuild_ms": float(
            0.0
            if name in {"atlas", "triggered_contextual_chart"}
            else np.median([rows[s]["median_svd_rebuild_ms"] for s in seeds])
        ),
        "median_mean_closed_loop_stage_cost": float(
            np.median([rows[s]["mean_closed_loop_stage_cost"] for s in seeds])
        ),
        "median_control_compute_ms": float(
            np.median([control_timing_value(rows[s], "median") for s in seeds])
        ),
        "median_p99_control_compute_ms": float(
            np.median([control_timing_value(rows[s], "p99") for s in seeds])
        ),
        "max_control_compute_ms": float(
            np.max([control_timing_value(rows[s], "max") for s in seeds])
        ),
        "control_compute_includes_online_svd_rebuild": name not in {
            "atlas", "triggered_contextual_chart"
        },
        "all_qps_optimal": bool(
            all(set(rows[s]["statuses"]) <= {"optimal", "optimal_inaccurate"} for s in seeds)
        ),
    }
    if name == "atlas_contextual_rescue":
        triggered = [s for s in seeds if rows[s].get("rescue_triggered", False)]
        result.update(
            rescue_triggered_seeds=triggered,
            rescue_trigger_count=len(triggered),
            rescue_trigger_rate=float(len(triggered) / len(seeds)),
            median_rescue_usage_fraction=float(
                np.median([rows[s].get("rescue_usage_fraction", 0.0) for s in seeds])
            ),
        )
    if name == "triggered_contextual_chart":
        triggered = [s for s in seeds if rows[s].get("future_mode_triggered", False)]
        result.update(
            contextual_chart_triggered_seeds=triggered,
            contextual_chart_trigger_count=len(triggered),
            contextual_chart_trigger_rate=float(len(triggered) / len(seeds)),
            median_contextual_chart_usage_fraction=float(
                np.median([rows[s].get("future_mode_usage_fraction", 0.0) for s in seeds])
            ),
            online_hankel_columns_scored=0,
            online_svd_rebuild=False,
        )
    return result


def paired_summary(
    first_name: str,
    first: dict[int, dict],
    second_name: str,
    second: dict[int, dict],
    seeds: list[int],
) -> dict:
    repaired = [
        s for s in seeds if not strict_success(first[s]) and strict_success(second[s])
    ]
    regressed = [
        s for s in seeds if strict_success(first[s]) and not strict_success(second[s])
    ]
    shared_failures = [
        s for s in seeds if not strict_success(first[s]) and not strict_success(second[s])
    ]
    return {
        "first": first_name,
        "second": second_name,
        "first_fail_second_pass": repaired,
        "first_pass_second_fail": regressed,
        "shared_failures": shared_failures,
        "discordant_count": len(repaired) + len(regressed),
        "exact_mcnemar_p": exact_mcnemar_p(len(repaired), len(regressed)),
    }


def make_figure(payload: dict, output_dir: Path) -> None:
    methods = payload["methods"]
    short_labels = {
        "atlas": "Atlas",
        "select_dpc_svd": "Select-DPC",
        "contextual_graph_svd": "Contextual\n(graph ref)",
        "contextual_final_svd": "Contextual\n(final ref)",
        "atlas_contextual_rescue": "Contextual\nrescue",
        "triggered_contextual_chart": "Triggered\nchart union",
    }
    labels = [short_labels[row["method"]] for row in methods]
    rates = np.array([row["success_rate"] for row in methods])
    lower = np.maximum(
        0.0,
        rates - np.array([row["wilson_95"][0] for row in methods]),
    )
    upper = np.maximum(
        0.0,
        np.array([row["wilson_95"][1] for row in methods]) - rates,
    )
    tracking = [row["median_final100_mm"] for row in methods]
    prediction = [row["median_prediction_error_mm"] for row in methods]
    closed_loop_cost = [row["median_closed_loop_control_cost"] for row in methods]
    compute = [row["median_control_compute_ms"] for row in methods]
    colors = [
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"
    ]

    fig, axes = plt.subplots(1, 5, figsize=(24.0, 4.6))
    axes[0].bar(labels, 100 * rates, color=colors)
    axes[0].errorbar(
        np.arange(len(methods)), 100 * rates, yerr=100 * np.vstack([lower, upper]),
        fmt="none", ecolor="black", capsize=4,
    )
    axes[0].set(title="Strict success", ylabel="Success rate [%]", ylim=(0, 105))
    for index, row in enumerate(methods):
        axes[0].text(
            index,
            max(8.0, 100 * row["success_rate"] - 8.0),
            f"{row['successes']}\n({100 * row['success_rate']:.1f}%)",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
        )
    axes[1].bar(labels, closed_loop_cost, color=colors)
    axes[1].set(
        title="Median active-reference cost",
        ylabel="Total realized stage cost",
        ylim=(0.0, 1.08 * max(closed_loop_cost)),
    )
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[2].bar(labels, tracking, color=colors)
    axes[2].set(title="Median final-100 error", ylabel="Error [mm]")
    axes[3].bar(labels, prediction, color=colors)
    axes[3].set(title="Median prediction error", ylabel="Error [mm]")
    axes[4].bar(labels, compute, color=colors)
    axes[4].set(
        title="Complete online compute\n(batch-recorded)",
        ylabel="Median time [ms/step]",
    )
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.tick_params(axis="x", rotation=0, labelsize=8)
    fig.suptitle(
        f"Paired evaluation seeds {payload['seeds'][0]}–{payload['seeds'][-1]}: "
        "six-way DeePC selector/reference comparison"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "heldout_six_way_comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / "heldout_six_way_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_stop + 1))
    H_parent = np.load(args.data_path)
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
    rescue = RescueConfig(
        selected_columns=args.selected_columns,
        error_threshold=args.rescue_error_threshold,
        stall_window=args.rescue_stall_window,
        minimum_progress=args.rescue_minimum_progress,
    )
    future_config = FutureOutputAtlasConfig(
        modes_per_chart=8,
        error_threshold=args.rescue_error_threshold,
        stall_window=args.rescue_stall_window,
        minimum_progress=args.rescue_minimum_progress,
    )

    baseline = json.loads(args.baseline_summary.read_text())
    atlas = {}
    for row in baseline["results"]:
        seed = int(row["seed"])
        if seed not in seeds:
            continue
        result_file = Path(row.get("result_path", ""))
        atlas[seed] = (
            load_saved_summary(result_file, H_parent)
            if result_file.exists()
            else row
        )
    frozen = baseline["config"]
    for key in (
        "clusters", "clustering_input_weight", "clustering_method",
        "clustering_feature_mode", "formulation", "svd_rank", "steps",
    ):
        if frozen[key] != getattr(config, key):
            raise RuntimeError(
                f"Atlas baseline mismatch for {key}: {frozen[key]} != {getattr(config, key)}"
            )

    method_rows: dict[str, dict[int, dict]] = {
        method: (atlas if method == "atlas" else {}) for method in ALL_METHODS
    }
    if args.resume:
        for bootstrap_dir in args.bootstrap_output_dirs:
            summary_path = bootstrap_dir / "summary.json"
            if not summary_path.exists():
                continue
            try:
                bootstrap_payload = json.loads(summary_path.read_text())
            except Exception:
                continue
            for method, saved_rows in bootstrap_payload.get(
                "method_summaries", {}
            ).items():
                if method not in method_rows:
                    continue
                for saved in saved_rows:
                    seed = int(saved["seed"])
                    if seed in seeds and seed not in method_rows[method]:
                        method_rows[method][seed] = saved
    pending: list[tuple[int, str]] = []
    triggered_chart_pending: list[int] = []
    for method in ALL_METHODS:
        for seed in seeds:
            if seed in method_rows[method]:
                continue
            path = result_path(args.output_dir, method, seed)
            if args.resume and path.exists():
                try:
                    saved = load_saved_summary(path, H_parent)
                    if (
                        method != "atlas_contextual_rescue"
                        or saved.get("rescue") == asdict(rescue)
                    ) and (
                        method != "triggered_contextual_chart"
                        or saved.get("future_output_atlas") == asdict(future_config)
                    ):
                        method_rows[method][seed] = saved
                        continue
                except Exception:
                    pass
            if args.resume:
                for bootstrap_dir in args.bootstrap_output_dirs:
                    for bootstrap_method in bootstrap_method_names(method):
                        bootstrap_path = result_path(
                            bootstrap_dir, bootstrap_method, seed
                        )
                        if not bootstrap_path.exists():
                            continue
                        try:
                            method_rows[method][seed] = load_saved_summary(
                                bootstrap_path, H_parent
                            )
                            break
                        except Exception:
                            pass
                    if seed in method_rows[method]:
                        break
                if seed in method_rows[method]:
                    continue
                if method == "atlas_contextual_rescue":
                    for bootstrap_dir in args.rescue_bootstrap_dirs:
                        bootstrap_path = bootstrap_dir / f"seed{seed}_control_results.npz"
                        if not bootstrap_path.exists():
                            continue
                        try:
                            saved = load_saved_summary(bootstrap_path, H_parent)
                            if saved.get("rescue") == asdict(rescue):
                                method_rows[method][seed] = saved
                                break
                        except Exception:
                            pass
                    if seed in method_rows[method]:
                        continue
                if method == "triggered_contextual_chart":
                    for bootstrap_dir in args.triggered_chart_bootstrap_dirs:
                        bootstrap_path = bootstrap_dir / f"seed{seed}_control_results.npz"
                        if not bootstrap_path.exists():
                            continue
                        try:
                            saved = load_saved_summary(bootstrap_path, H_parent)
                            if saved.get("future_output_atlas") == asdict(future_config):
                                method_rows[method][seed] = saved
                                break
                        except Exception:
                            pass
                    if seed in method_rows[method]:
                        continue
            if method == "triggered_contextual_chart":
                triggered_chart_pending.append(seed)
            else:
                pending.append((seed, method))

    if triggered_chart_pending:
        triggered_output_dir = args.output_dir / "triggered_contextual_chart"
        print(
            f"running {len(triggered_chart_pending)} missing triggered contextual "
            "chart-union rollouts",
            flush=True,
        )
        triggered_payload = run_future_output_panel(
            triggered_chart_pending,
            args.atlas_path.parent,
            triggered_output_dir,
            future_config,
            data_path=args.data_path,
            workers=args.workers,
            overwrite=False,
        )
        method_rows["triggered_contextual_chart"].update(
            {int(row["seed"]): row for row in triggered_payload["results"]}
        )

    if pending:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize_worker,
            initargs=(
                str(args.data_path), str(args.atlas_path), asdict(config),
                args.selected_columns, asdict(rescue), str(args.output_dir),
            ),
        ) as executor:
            futures = {executor.submit(run_worker, task): task for task in pending}
            completed = len(ALL_METHODS) * len(seeds) - len(pending)
            for future in as_completed(futures):
                seed, method, row = future.result()
                method_rows[method][seed] = row
                completed += 1
                print(
                    f"completed {completed}/{len(ALL_METHODS) * len(seeds)} "
                    f"method={method} seed={seed} "
                    f"success={strict_success(row)} tail={1000 * row['mean_last_100']:.2f} mm",
                    flush=True,
                )

    atlas = method_rows["atlas"]
    online = {method: method_rows[method] for method in METHODS}
    payload = {
        "benchmark": "controlled six-way DeePC selector/reference comparison",
        "seeds": seeds,
        "protocol": {
            "same_seed_panel": True,
            "same_offline_hankel_library": True,
            "same_48_chart_atlas_and_final_graph_compatible_goal": True,
            "same_horizons_costs_constraints_solver": True,
            "steps": args.steps,
            "selected_columns": args.selected_columns,
            "rank": args.rank,
            "strict_success": (
                "both final-30 and final-100 mean error < 1 cm and max error < 2 cm"
            ),
            "experimental_factors": [
                "local Hankel-column selection",
                "graph-routed waypoint reference versus final reference only",
                "always-online contextual selection versus causally triggered rescue",
                "online trajectory rescue versus cached contextual chart-union rescue",
            ],
            "shared_local_representation": (
                "centered rank-60 SVD with the same coefficient bounds; Atlas "
                "precomputed offline, online selectors rebuilt every step"
            ),
            "online_timing_includes": (
                "localization/routing where used + selection + SVD rebuild + QP solve"
            ),
            "closed_loop_cost": (
                "sum_t (y_t-r_t)^T Q (y_t-r_t) + u_t^T R u_t, where r_t "
                "is the waypoint/final reference active during control interval t"
            ),
            "secondary_final_goal_cost": (
                "the previous common-final-goal metric is retained separately as "
                "median_closed_loop_final_goal_cost"
            ),
            "atlas_results_reused": str(args.baseline_summary.resolve()),
            "results_bootstrapped_from": [
                str(path.resolve()) for path in args.bootstrap_output_dirs
            ],
            "rescue_results_bootstrapped_from": [
                str(path.resolve()) for path in args.rescue_bootstrap_dirs
            ],
            "contextual_rescue": asdict(rescue),
            "triggered_contextual_chart": asdict(future_config),
            "triggered_chart_results_bootstrapped_from": [
                str(path.resolve()) for path in args.triggered_chart_bootstrap_dirs
            ],
            "timing_note": (
                "This parallel batch is used for outcome statistics. Refer to the "
                "single-seed controlled notebook cells for fair serial timing."
            ),
        },
        "methods": [
            method_summary(name, method_rows[name], seeds)
            for name in ALL_METHODS
        ],
        "paired_comparisons": [
            paired_summary(first, method_rows[first], second, method_rows[second], seeds)
            for first, second in combinations(ALL_METHODS, 2)
        ],
        "config": asdict(config),
        "method_summaries": {
            name: [method_rows[name][s] for s in seeds] for name in ALL_METHODS
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    make_figure(payload, args.output_dir)
    print(json.dumps({"methods": payload["methods"], "paired": payload["paired_comparisons"]}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--atlas-path", type=Path, default=DEFAULT_ATLAS_PATH)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument(
        "--bootstrap-output-dir",
        dest="bootstrap_output_dirs",
        type=Path,
        action="append",
        default=list(DEFAULT_BOOTSTRAP_DIRS),
        help="Earlier per-seed result directory to reuse; may be repeated.",
    )
    parser.add_argument(
        "--rescue-bootstrap-dir",
        dest="rescue_bootstrap_dirs",
        type=Path,
        action="append",
        default=list(DEFAULT_RESCUE_BOOTSTRAP_DIRS),
        help="Earlier direct rescue-result directory to reuse; may be repeated.",
    )
    parser.add_argument(
        "--triggered-chart-bootstrap-dir",
        dest="triggered_chart_bootstrap_dirs",
        type=Path,
        action="append",
        default=list(DEFAULT_TRIGGERED_CHART_BOOTSTRAP_DIRS),
        help="Earlier direct triggered-chart result directory to reuse; may be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed-start", type=int, default=111)
    parser.add_argument("--seed-stop", type=int, default=610)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--selected-columns", type=int, default=200)
    parser.add_argument("--rank", type=int, default=60)
    parser.add_argument("--rescue-error-threshold", type=float, default=0.01)
    parser.add_argument("--rescue-stall-window", type=int, default=15)
    parser.add_argument("--rescue-minimum-progress", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
