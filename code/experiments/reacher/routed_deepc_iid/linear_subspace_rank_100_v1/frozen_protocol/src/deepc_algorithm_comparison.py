"""Controlled six-way comparison of local-data/reference mechanisms for DeePC.

The graph router, causal waypoint policy, offline Hankel library, output
representation, horizons, costs, constraints, solver, and one-QP
receding-horizon loop are identical. Every method uses the same centered
rank-60 affine representation after selecting columns. Atlas bases are built
offline; online selectors rebuild their SVD basis at every control step.

* Atlas-DeePC: use the current offline K-means chart and its cached basis.
* Select-DPC selector: top-N full-trajectory L1 columns, rebuilt to rank 60.
* Contextual Sampling + graph reference: top-N past-output/future-reference
  columns, rebuilt to rank 60 after paper-style scoring.
* Contextual Sampling + final reference: the same contextual selection, but use only
  the final goal from step zero and remove online graph routing/waypoints.
* Atlas + contextual rescue: use cached Atlas bases normally, then latch the
  contextual top-N/rank-60 selector after a causal terminal-tracking stall.
* Triggered contextual chart union: use the current cached Atlas basis normally,
  then select an outgoing chart from cached Q-weighted Y_f prototypes and use
  the cached current/selected-chart union basis after the same causal stall.

The reported end-to-end online time for every method starts before chart
localization and graph routing.  Contextual Sampling therefore includes graph
routing, contextual selection, matrix reconstruction, and the QP solve.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import expert_pipeline_search as pipe
from contextual_rescue_atlas_experiment import RescueConfig, run_contextual_rescue
from future_output_atlas_experiment import (
    FutureOutputAtlasConfig,
    build_future_output_atlas,
    run_future_output_atlas,
)
from online_selected_deepc_comparison import run_online_selected_deepc
from routed_deepc_experiment import (
    DEFAULT_DATA_PATH,
    RoutedDeePCConfig,
    load_atlas,
    precompute_chart_matrix_cache,
    run_routed_deepc,
)


DEFAULT_ATLAS_PATH = Path(
    "experiments/routed_deepc_iid/"
    "clustering_jointconfig48_benchmark111_210/behavioral_atlas.npz"
)
def timing_value(result: dict, statistic: str, component: str) -> float:
    for key in (
        f"{statistic}_{component}_ms",
        f"{statistic}_{component}_time_ms",
    ):
        if key in result:
            return float(result[key])
    if component in ("selection", "svd_rebuild"):
        return 0.0
    raise KeyError(f"No {statistic} timing for {component}: {sorted(result)}")


def canonical_result(
    result: dict,
    method: str,
    selector: str,
    selected_columns,
    coefficient_dimension: int,
) -> dict:
    timing_components = (
        "localization",
        "routing",
        "selection",
        "svd_rebuild",
        "matrix_preparation",
        "solve",
        "control_compute",
    )
    timings = {
        component: {
            statistic: timing_value(result, statistic, component)
            for statistic in ("mean", "median", "p95", "p99", "max")
        }
        for component in timing_components
    }
    canonical = {
        "method": method,
        "selector": selector,
        "selected_columns": selected_columns,
        "coefficient_dimension": coefficient_dimension,
        "qp_count": int(result.get("qp_count", result["steps"])),
        "final_error_mm": 1000.0 * float(result["last"]),
        "mean_final_30_mm": 1000.0 * float(result["mean_last_30"]),
        "mean_final_100_mm": 1000.0 * float(result["mean_last_100"]),
        "prediction_error_mm": 1000.0 * float(result["mean_prediction_error_xy"]),
        "closed_loop_control_cost": float(result["closed_loop_control_cost"]),
        "closed_loop_reference_cost": float(result["closed_loop_reference_cost"]),
        "closed_loop_final_goal_cost": float(result["closed_loop_final_goal_cost"]),
        "mean_closed_loop_stage_cost": float(result["mean_closed_loop_stage_cost"]),
        "strict_success": bool(result["success_30"] and result["success_100"]),
        "timing_ms": timings,
        "total_includes_graph_routing": bool(result.get("graph_routing_used", True)),
        "total_includes_online_svd_rebuild": bool(
            result.get("control_compute_includes_svd_rebuild", False)
        ),
    }
    if "rescue_triggered" in result:
        canonical.update(
            rescue_triggered=bool(result["rescue_triggered"]),
            rescue_trigger_step=int(result["rescue_trigger_step"]),
            rescue_usage_fraction=float(result["rescue_usage_fraction"]),
        )
    if "future_mode_triggered" in result:
        canonical.update(
            contextual_chart_triggered=bool(result["future_mode_triggered"]),
            contextual_chart_trigger_step=int(result["future_mode_trigger_step"]),
            contextual_chart_usage_fraction=float(result["future_mode_usage_fraction"]),
            contextual_chart_pair_usage_fraction=float(
                result["future_chart_pair_usage_fraction"]
            ),
        )
    return canonical


def make_figure(output_dir: Path, result_paths: list[tuple[str, Path, str]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 3.8))
    for label, path, color in result_paths:
        data = np.load(path, allow_pickle=True)
        axes[0].plot(1000.0 * data["distance"], label=label, color=color)
        axes[1].plot(
            1000.0 * data["prediction_error_xy"],
            label=label,
            color=color,
        )
        axes[2].plot(
            1000.0 * data["control_compute_time_s"],
            label=label,
            color=color,
            alpha=0.85,
        )
    axes[0].set(title="Target tracking", xlabel="Control step", ylabel="Error [mm]")
    axes[1].set(
        title="One-step prediction",
        xlabel="Control step",
        ylabel="XY error [mm]",
    )
    axes[2].set(
        title="Complete online compute",
        xlabel="Control step",
        ylabel="Time [ms]",
        yscale="log",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle(
        "Controlled six-way DeePC selector/reference comparison"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "six_way_comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / "six_way_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    H = np.load(args.data_path)
    atlas = load_atlas(args.atlas_path)
    config = RoutedDeePCConfig(
        clusters=48,
        clustering_input_weight=0.0,
        clustering_method="minibatch_kmeans",
        clustering_feature_mode="plant_joint_configuration",
        formulation="svd_current",
        svd_rank=args.rank,
        steps=args.steps,
        seed=args.seed,
    )

    atlas_path = args.output_dir / "atlas_control_results.npz"
    select_path = args.output_dir / "select_dpc_selector_control_results.npz"
    contextual_graph_path = args.output_dir / "contextual_graph_svd_control_results.npz"
    contextual_final_path = args.output_dir / "contextual_final_svd_control_results.npz"
    rescue_path = args.output_dir / "atlas_contextual_rescue_control_results.npz"
    triggered_chart_path = (
        args.output_dir / "triggered_contextual_chart_control_results.npz"
    )

    print("precomputing Atlas chart bases offline", flush=True)
    atlas_matrix_cache = precompute_chart_matrix_cache(H, atlas, config)
    print("running Atlas-DeePC chart selector", flush=True)
    atlas_summary = run_routed_deepc(
        H,
        atlas,
        config,
        save_path=atlas_path,
        matrix_cache=atlas_matrix_cache,
    )
    atlas_arrays = np.load(atlas_path, allow_pickle=True)
    atlas_selected = int(np.median(atlas_arrays["matrix_columns"]))
    for component in (
        "localization",
        "routing",
        "matrix_preparation",
        "solve",
        "control_compute",
    ):
        milliseconds = 1000.0 * np.asarray(
            atlas_arrays[f"{component}_time_s"], dtype=float
        )
        atlas_summary.update(
            {
                f"mean_{component}_time_ms": float(np.mean(milliseconds)),
                f"p99_{component}_time_ms": float(np.percentile(milliseconds, 99)),
                f"max_{component}_time_ms": float(np.max(milliseconds)),
            }
        )
    atlas_summary.update(
        method="atlas_deepc",
        qp_count=args.steps,
        optimized_coefficient_dimension=args.rank,
        selected_columns=atlas_selected,
    )

    print("running Select-DPC trajectory selector ablation", flush=True)
    select_summary = run_online_selected_deepc(
        H,
        atlas,
        config,
        args.selected_columns,
        selected_model="rank_matched",
        selection_method="select_dpc",
        save_path=select_path,
    )

    print("running Contextual Sampling with graph-routed reference", flush=True)
    contextual_graph_summary = run_online_selected_deepc(
        H,
        atlas,
        config,
        args.selected_columns,
        selected_model="rank_matched",
        selection_method="contextual",
        reference_mode="graph_routed",
        save_path=contextual_graph_path,
    )

    print("running Contextual Sampling with final reference only", flush=True)
    contextual_final_summary = run_online_selected_deepc(
        H,
        atlas,
        config,
        args.selected_columns,
        selected_model="rank_matched",
        selection_method="contextual",
        reference_mode="final_only",
        save_path=contextual_final_path,
    )

    rescue = RescueConfig(
        selected_columns=args.selected_columns,
        error_threshold=args.rescue_error_threshold,
        stall_window=args.rescue_stall_window,
        minimum_progress=args.rescue_minimum_progress,
    )
    print("running Atlas-DeePC with causal contextual stagnation rescue", flush=True)
    rescue_summary = run_contextual_rescue(
        H,
        atlas,
        config,
        rescue,
        chart_matrix_cache=atlas_matrix_cache,
        save_path=rescue_path,
    )

    future_config = FutureOutputAtlasConfig(
        modes_per_chart=8,
        error_threshold=args.rescue_error_threshold,
        stall_window=args.rescue_stall_window,
        minimum_progress=args.rescue_minimum_progress,
    )
    print("building cached contextual chart-union bases offline", flush=True)
    future_artifact = build_future_output_atlas(H, atlas, config, future_config)
    print("running Atlas-DeePC with triggered contextual chart union", flush=True)
    triggered_chart_summary = run_future_output_atlas(
        H,
        atlas,
        future_artifact,
        config,
        future_config,
        save_path=triggered_chart_path,
    )

    comparison_rows = [
        canonical_result(
            atlas_summary,
            "Atlas-DeePC",
            "offline 48-chart joint-configuration K-means; cached current-chart basis",
            atlas_selected,
            args.rank,
        ),
        canonical_result(
            select_summary,
            "Select-DPC selector",
            f"online full-trajectory L1 ranking; selected columns rebuilt to rank {args.rank}",
            args.selected_columns,
            args.rank,
        ),
        canonical_result(
            contextual_graph_summary,
            "Contextual Sampling + graph reference",
            f"online normalized past-output plus Q-weighted graph-reference ranking; rebuilt to rank {args.rank}",
            args.selected_columns,
            args.rank,
        ),
        canonical_result(
            contextual_final_summary,
            "Contextual Sampling + final reference",
            f"online normalized past-output plus Q-weighted final-reference ranking; rebuilt to rank {args.rank}",
            args.selected_columns,
            args.rank,
        ),
        canonical_result(
            rescue_summary,
            "Atlas + contextual rescue",
            (
                "offline Atlas chart basis until a causal terminal stall; then "
                f"online contextual top-{args.selected_columns} rebuilt to rank {args.rank}"
            ),
            f"Atlas chart / {args.selected_columns} after rescue",
            args.rank,
        ),
        canonical_result(
            triggered_chart_summary,
            "Triggered contextual chart union",
            (
                "cached current-chart basis until a causal terminal stall; then "
                "Q-weighted Y_f prototype selection over graph neighbors and a "
                "cached current/selected-chart union basis"
            ),
            "Atlas chart / cached two-chart union after trigger",
            args.rank,
        ),
    ]

    payload = {
        "benchmark": "controlled six-way DeePC selector/reference comparison",
        "protocol": {
            "data_path": str(args.data_path.resolve()),
            "atlas_path": str(args.atlas_path.resolve()),
            "environment": pipe.REACHER_ENV_ID,
            "seed": args.seed,
            "steps": args.steps,
            "T_ini": pipe.T_INI,
            "T_f": pipe.T_F,
            "input_dimension": pipe.M,
            "output_dimension": pipe.P,
            "selected_columns_for_online_methods": args.selected_columns,
            "affine_rank": args.rank,
            "solver": "CLARABEL",
            "qp_per_step": 1,
            "shared_components": [
                "offline Hankel library",
                "48-chart localization used by graph routing",
                "goal chart and shortest-path router",
                "output representation and horizons",
                "costs, constraints, regularization, and solver",
            ],
            "experimental_factors": [
                "local Hankel-column selection",
                "graph-routed waypoint reference versus final reference only",
                "always-online contextual selection versus causally triggered rescue",
                "online trajectory rescue versus offline cached chart-union rescue",
            ],
        },
        "timing_protocol": {
            "boundary": (
                "observation availability through returned action; environment stepping "
                "and plotting excluded"
            ),
            "atlas_total": (
                "chart localization + graph routing + cached matrix retrieval + QP solve"
            ),
            "select_dpc_total": (
                "chart localization + graph routing + L1 selection + online SVD rebuild + QP solve"
            ),
            "contextual_graph_total": (
                "chart localization + graph routing + contextual selection + online SVD rebuild + QP solve"
            ),
            "contextual_final_total": (
                "contextual selection + online SVD rebuild + QP solve; no online graph routing"
            ),
            "atlas_contextual_rescue_total": (
                "Atlas localization + graph routing + cached basis + QP normally; "
                "after a causal stall, contextual selection + online SVD rebuild + QP"
            ),
            "triggered_contextual_chart_total": (
                "Atlas localization + graph routing + cached current-chart basis + QP "
                "normally; after a causal stall, cached Y_f-prototype scoring + cached "
                "current/selected-chart union basis + QP"
            ),
            "contextual_graph_routing_included": True,
            "contextual_final_graph_routing_included": False,
        },
        "closed_loop_cost_protocol": {
            "definition": "sum_t (y_t-r_t)^T Q (y_t-r_t) + u_t^T R u_t",
            "reference": (
                "r_t is the graph waypoint or final reference active during "
                "control interval t"
            ),
            "secondary_metric": (
                "closed_loop_final_goal_cost retains evaluation against the same "
                "final graph-compatible goal"
            ),
            "reported_statistic_in_multi_seed_panel": "median total cost across paired seeds",
        },
        "contextual_sampling_definition": {
            "paper": "https://arxiv.org/pdf/2503.23890v2",
            "past_score": "||Y_p,i - y_ini||_2 after per-output training-data normalization",
            "future_score": "||Y_f,i - r_graph||_Q after the same normalization",
            "combination": "z-score both distance vectors, add, and retain the smallest N",
            "reference_sources": {
                "contextual_graph_svd": "Atlas graph router at the current control step",
                "contextual_final_svd": "same final graph-compatible goal from step zero",
            },
            "selected_representation": (
                f"top-{args.selected_columns} Hankel columns rebuilt online as the "
                f"same centered rank-{args.rank} SVD basis and coefficient bounds "
                "used by Atlas"
            ),
            "timing": "online selector totals include per-step SVD reconstruction",
        },
        "contextual_rescue_definition": {
            "policy": "Atlas normally; contextual selection is latched after a causal terminal stall",
            "selected_columns": rescue.selected_columns,
            "error_threshold_m": rescue.error_threshold,
            "stall_window_steps": rescue.stall_window,
            "minimum_progress_m": rescue.minimum_progress,
            "trigger": (
                "terminal reference active, error above threshold, and progress over "
                "the trailing window below minimum_progress"
            ),
        },
        "triggered_contextual_chart_definition": {
            "policy": (
                "current-chart Atlas normally; after the same causal terminal stall, "
                "select an observed outgoing chart from cached Q-weighted Y_f prototypes"
            ),
            "future_modes_per_chart": future_config.modes_per_chart,
            "error_threshold_m": future_config.error_threshold,
            "stall_window_steps": future_config.stall_window,
            "minimum_progress_m": future_config.minimum_progress,
            "online_hankel_columns_scored": 0,
            "online_svd_rebuild": False,
            "active_basis": "cached basis of current chart union selected chart",
        },
        "results": comparison_rows,
        "method_summaries": {
            "atlas_deepc": atlas_summary,
            "select_dpc_selector": select_summary,
            "contextual_graph_svd": contextual_graph_summary,
            "contextual_final_svd": contextual_final_summary,
            "atlas_contextual_rescue": rescue_summary,
            "triggered_contextual_chart": triggered_chart_summary,
        },
        "config": asdict(config),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    make_figure(
        args.output_dir,
        [
            ("Atlas-DeePC", atlas_path, "tab:blue"),
            ("Select-DPC selector", select_path, "tab:orange"),
            ("Contextual + graph ref", contextual_graph_path, "tab:green"),
            ("Contextual + final ref", contextual_final_path, "tab:red"),
            ("Atlas + contextual rescue", rescue_path, "tab:purple"),
            ("Triggered chart union", triggered_chart_path, "tab:brown"),
        ],
    )

    print(json.dumps(comparison_rows, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--atlas-path", type=Path, default=DEFAULT_ATLAS_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path(
        "experiments/deepc_algorithm_comparison/six_way_seed1_svd_selector_n200_rank60"
    ))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--selected-columns", type=int, default=200)
    parser.add_argument("--rank", type=int, default=60)
    parser.add_argument("--rescue-error-threshold", type=float, default=0.01)
    parser.add_argument("--rescue-stall-window", type=int, default=15)
    parser.add_argument("--rescue-minimum-progress", type=float, default=0.001)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
