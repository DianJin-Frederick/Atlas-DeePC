"""Run task-specific Atlas-DeePC profiles on a common seed panel."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from simple_robot_atlas_benchmark import PilotConfig, TASKS, run_task


PROFILES = {
    "point_mass_easy": PilotConfig(
        offline_episodes=200,
        offline_steps=200,
        offline_input_momentum=0.9,
        offline_input_scale=0.3,
        clusters=4,
        contextual_clusters=64,
        contextual_support=300,
        svd_rank=29,
        physical_feature_mode="state",
        physical_goal_weight=4.0,
        reference_mode="final",
        control_steps=400,
        lam_du=1.0,
        trigger_window=40,
    ),
    "cartpole_balance": PilotConfig(
        offline_episodes=60,
        offline_steps=400,
        offline_input_momentum=0.9,
        offline_input_scale=0.3,
        clusters=16,
        contextual_clusters=128,
        contextual_support=200,
        svd_rank=16,
        physical_feature_mode="state",
        physical_goal_weight=4.0,
        reference_mode="final",
        control_steps=320,
        contextual_activation="always",
    ),
    "finger_turn_easy": PilotConfig(
        offline_episodes=200,
        offline_steps=200,
        clusters=48,
        contextual_clusters=256,
        contextual_support=200,
        contextual_frame="goal_relative",
        svd_rank=60,
        edge_count_threshold=1,
        physical_feature_mode="configuration",
        physical_goal_weight=4.0,
        reference_mode="graph_routed",
        waypoint_mode="chart_medoid",
        control_steps=320,
        trigger_window=15,
    ),
    # Cartpole uses disjoint global swing and local upright random-walk
    # partitions. No retained episode stabilizes at the goal.
    "cartpole_swingup": PilotConfig(
        t_f=30,
        offline_episodes=200,
        offline_steps=480,
        offline_reset_mode="blocked_near_goal",
        offline_input_mode="mixed",
        offline_input_momentum=0.9,
        offline_input_scale=1.0,
        offline_local_input_scale=0.3,
        offline_input_period=150.0,
        clusters=48,
        contextual_clusters=512,
        contextual_support=100,
        contextual_training_partition="local",
        contextual_t_f=10,
        contextual_lam_g=0.05,
        contextual_lam_du=1.0,
        contextual_r=1.0e-4,
        contextual_penalize_initial_move=True,
        contextual_velocity_weight_scale=100.0,
        contextual_output_bias_gain=0.1,
        svd_rank=100,
        contextual_svd_rank=16,
        svd_bound_scale=5.0,
        edge_count_threshold=1,
        waypoint_mode="edge_transition",
        physical_feature_mode="state",
        reference_mode="template_path",
        control_steps=800,
        lam_g=1.0e-4,
        lam_du=0.0,
        r=1.0e-6,
        contextual_activation="successful",
        contextual_success_streak=20,
        latched_matrix_mode="goal_focused",
        terminal_t_f=10,
        terminal_svd_rank=16,
        terminal_lam_g=0.05,
        terminal_lam_du=5.0,
        terminal_r=1.0e-4,
    ),
    "pendulum_swingup": PilotConfig(
        t_f=20,
        offline_episodes=100,
        offline_steps=500,
        offline_input_mode="square",
        offline_input_scale=1.0,
        offline_input_period=100.0,
        clusters=48,
        contextual_clusters=128,
        contextual_support=200,
        svd_rank=60,
        edge_count_threshold=1,
        graph_direction="directed",
        waypoint_mode="edge_transition",
        physical_feature_mode="state",
        reference_mode="goal_path",
        control_steps=500,
        lam_g=1.0e-3,
        lam_du=0.2,
        r=1.0e-6,
        contextual_activation="latched",
    ),
    "ball_in_cup": PilotConfig(
        t_f=20,
        offline_episodes=100,
        offline_steps=400,
        offline_input_hold=10,
        clusters=48,
        contextual_clusters=128,
        contextual_architecture="goal_enriched",
        contextual_support=200,
        svd_rank=100,
        svd_bound_scale=5.0,
        edge_count_threshold=1,
        waypoint_mode="edge_transition",
        physical_feature_mode="state",
        physical_goal_weight=4.0,
        reference_mode="edge_sequence",
        control_steps=500,
        lam_du=1.0,
        r=1.0e-5,
        contextual_activation="triggered",
        trigger_window=100,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="point_mass_easy,cartpole_balance")
    parser.add_argument("--methods", default="atlas,contextual_atlas")
    parser.add_argument("--evaluation-seed-start", type=int, default=601)
    parser.add_argument("--evaluation-seeds", type=int, default=20)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/simple_robot_atlas_profiles"),
    )
    return parser


def main(args: argparse.Namespace) -> None:
    task_names = [item.strip() for item in args.tasks.split(",") if item.strip()]
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    unknown = sorted(set(task_names) - set(PROFILES))
    if unknown:
        raise SystemExit(f"Unknown task profiles: {unknown}")
    unknown_methods = sorted(set(methods) - {"atlas", "contextual_atlas"})
    if unknown_methods:
        raise SystemExit(f"Unknown methods: {unknown_methods}")

    summaries = []
    used_configs = {}
    for task_name in task_names:
        config = replace(
            PROFILES[task_name],
            evaluation_seed_start=args.evaluation_seed_start,
            evaluation_seeds=args.evaluation_seeds,
        )
        used_configs[task_name] = config
        summaries.append(run_task(TASKS[task_name], config, args.output_root, methods))

    aggregate = {
        "evaluation_seed_range": [
            args.evaluation_seed_start,
            args.evaluation_seed_start + args.evaluation_seeds - 1,
        ],
        "methods": list(methods),
        "profiles": {name: asdict(used_configs[name]) for name in task_names},
        "tasks": [
            {
                "task": summary["task"],
                "offline_audit": summary["offline_audit"],
                "offline_atlas": summary["offline_atlas"],
                "preprocessing_timing_ms": summary["preprocessing_timing_ms"],
                "atlas": summary.get("atlas"),
                "contextual_atlas": summary.get("contextual_atlas"),
            }
            for summary in summaries
        ],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "profile_summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        previous_tasks = {row["task"]: row for row in previous.get("tasks", [])}
        previous_tasks.update({row["task"]: row for row in aggregate["tasks"]})
        aggregate["tasks"] = [previous_tasks[name] for name in sorted(previous_tasks)]
        aggregate["profiles"] = {
            **previous.get("profiles", {}),
            **aggregate["profiles"],
        }
    for task_name in aggregate["profiles"]:
        task_summary_path = args.output_root / task_name / "summary.json"
        if task_summary_path.exists():
            aggregate["profiles"][task_name] = json.loads(
                task_summary_path.read_text()
            )["config"]
    summary_path.write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main(build_parser().parse_args())
