"""End-to-end Pusher Atlas-DeePC experiment orchestration."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
from time import perf_counter

import numpy as np

from .atlas import (
    PastScaler,
    assign_chart,
    build_behavior_library,
    fit_atlas,
    nearest_behavior,
    select_behavior_with_policy,
    validate_atlas,
)
from .config import PusherAtlasConfig
from .controller import LocalAffineDeePC, geometric_references, update_phase
from .coverage import (
    plot_coverage,
    save_coverage_report,
    serializable_coverage,
    validate_offline_coverage,
)
from .data import collect_teacher_data, load_data
from .environment import (
    ACTION_DIM,
    OBJECT_XY,
    PHASE_APPROACH,
    PHASE_HOLD,
    TIP,
    behavior_output,
    goal_xy_from_env,
    make_env,
    object_goal_distance,
    push_direction,
    state_from_env,
)
from .plotting import plot_dataset_summary, plot_rollout
from .policy import (
    load_behavior_policy,
    save_behavior_policy,
    train_behavior_policy,
)


def save_atlas(path: Path, atlas: dict, cfg: PusherAtlasConfig):
    """Save only behavioral-chart quantities; there are deliberately no A/B matrices."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "labels": atlas["labels"],
        "transition_counts": atlas["transition_counts"],
        "past_mean": atlas["past_scaler"].mean,
        "past_scale": atlas["past_scaler"].scale,
        "direction_feature_weight": np.array(atlas["direction_feature_weight"]),
        "direction_candidate_margin": np.array(atlas["direction_candidate_margin"]),
        "action_scale": atlas["action_scale"],
        "config": np.array(json.dumps(asdict(cfg), sort_keys=True)),
    }
    for chart in atlas["charts"]:
        prefix = f"chart_{chart['id']}"
        arrays[f"{prefix}_h_bar"] = chart["h_bar"]
        arrays[f"{prefix}_basis"] = chart["basis"]
        arrays[f"{prefix}_behavior_scale"] = chart["behavior_scale"]
        arrays[f"{prefix}_coefficient_bound"] = chart["coefficient_bound"]
        arrays[f"{prefix}_member_gate_features"] = chart["member_gate_features"]
        arrays[f"{prefix}_member_coordinates"] = chart["member_coordinates"]
        arrays[f"{prefix}_member_future_y"] = chart["member_future_y"]
        arrays[f"{prefix}_member_first_action"] = chart["member_first_action"]
        arrays[f"{prefix}_gate_center"] = chart["gate_center"]
        arrays[f"{prefix}_gate_radius"] = np.array(chart["gate_radius"])
        arrays[f"{prefix}_representative_y"] = chart["representative_y"]
        arrays[f"{prefix}_direction_center"] = chart["direction_center"]
        arrays[f"{prefix}_phase"] = np.array(chart["phase"])
        arrays[f"{prefix}_rank"] = np.array(chart["rank"])
        arrays[f"{prefix}_sample_count"] = np.array(chart["sample_count"])
        arrays[f"{prefix}_relative_residual"] = np.array(
            chart["relative_reconstruction_residual"]
        )
    np.savez_compressed(path, **arrays)


def load_atlas(path: Path):
    """Load a saved behavioral atlas for evaluation without refitting its charts."""
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    charts = []
    chart_id = 0
    while f"chart_{chart_id}_phase" in arrays:
        prefix = f"chart_{chart_id}"
        required = (
            "h_bar",
            "basis",
            "behavior_scale",
            "coefficient_bound",
            "member_gate_features",
            "member_coordinates",
            "member_future_y",
            "member_first_action",
            "gate_center",
            "gate_radius",
            "representative_y",
            "direction_center",
            "phase",
            "rank",
            "sample_count",
            "relative_residual",
        )
        missing = [name for name in required if f"{prefix}_{name}" not in arrays]
        if missing:
            raise ValueError(
                f"Saved atlas {path} is missing chart {chart_id} fields: {missing}"
            )
        charts.append(
            {
                "id": chart_id,
                "h_bar": arrays[f"{prefix}_h_bar"],
                "basis": arrays[f"{prefix}_basis"],
                "behavior_scale": arrays[f"{prefix}_behavior_scale"],
                "coefficient_bound": arrays[f"{prefix}_coefficient_bound"],
                "member_gate_features": arrays[f"{prefix}_member_gate_features"],
                "member_coordinates": arrays[f"{prefix}_member_coordinates"],
                "member_future_y": arrays[f"{prefix}_member_future_y"],
                "member_first_action": arrays[f"{prefix}_member_first_action"],
                "gate_center": arrays[f"{prefix}_gate_center"],
                "gate_radius": float(arrays[f"{prefix}_gate_radius"]),
                "representative_y": arrays[f"{prefix}_representative_y"],
                "direction_center": arrays[f"{prefix}_direction_center"],
                "phase": int(arrays[f"{prefix}_phase"]),
                "rank": int(arrays[f"{prefix}_rank"]),
                "sample_count": int(arrays[f"{prefix}_sample_count"]),
                "relative_reconstruction_residual": float(
                    arrays[f"{prefix}_relative_residual"]
                ),
            }
        )
        chart_id += 1
    if not charts:
        raise ValueError(f"Saved atlas {path} contains no charts")
    phase_chart_ids = {}
    for chart in charts:
        phase_chart_ids.setdefault(chart["phase"], []).append(chart["id"])
    return {
        "charts": charts,
        "labels": arrays["labels"],
        "transition_counts": arrays["transition_counts"],
        "phase_chart_ids": phase_chart_ids,
        "past_scaler": PastScaler(arrays["past_mean"], arrays["past_scale"]),
        "direction_feature_weight": float(arrays["direction_feature_weight"]),
        "direction_candidate_margin": float(arrays["direction_candidate_margin"]),
        "action_scale": arrays["action_scale"],
    }


def run_rollout(atlas, cfg: PusherAtlasConfig, seed: int, behavior_policy=None):
    """Run receding-horizon local DeePC; only the first optimized input is applied."""
    env, _ = make_env(seed, cfg.rollout_steps + cfg.t_ini + 2)
    controllers = [LocalAffineDeePC(chart, cfg) for chart in atlas["charts"]]
    goal_xy = goal_xy_from_env(env)
    previous_u = np.zeros(ACTION_DIM)
    phase = PHASE_APPROACH
    hold_counter = 0
    previous_chart = None
    chart_age = 0
    initial_direction = push_direction(state_from_env(env), goal_xy)
    orientation_turn = float(
        initial_direction[0] < cfg.collection_orientation_turn_threshold
    )

    # DeePC needs measured past input/output pairs.  These zero-input samples are
    # part of the experiment, not an identified initial state for an A/B model.
    u_history, y_history, warmup_actions = [], [], []
    for _ in range(cfg.t_ini):
        action = np.zeros(ACTION_DIM)
        warmup_actions.append(action.copy())
        env.step(action.astype(np.float32))
        u_history.append(action.copy())
        y_history.append(behavior_output(state_from_env(env), goal_xy))

    records = {
        "state": [],
        "action": [],
        "distance": [],
        "phase": [],
        "chart": [],
        "gate_ratio": [],
        "solve_time": [],
        "state_reference_time_s": [],
        "policy_inference_time_s": [],
        "behavior_selection_time_s": [],
        "qp_call_time_s": [],
        "postprocess_time_s": [],
        "control_compute_time_s": [],
        "environment_step_time_s": [],
        "feedback_time_s": [],
        "online_step_wall_time_s": [],
        "status": [],
        "prediction_error": [],
        "slack_norm": [],
        "coefficient_norm": [],
        "policy_prior_deviation": [],
    }

    for step in range(cfg.rollout_steps):
        online_step_start = perf_counter()
        state_reference_start = perf_counter()
        state = state_from_env(env)
        phase = update_phase(state, goal_xy, phase, step, cfg)
        tip_reference, object_reference, direction = geometric_references(
            state, goal_xy, phase, cfg
        )
        u_ini = np.asarray(u_history[-cfg.t_ini :])
        y_ini = np.asarray(y_history[-cfg.t_ini :])
        state_reference_time = perf_counter() - state_reference_start

        policy_start = perf_counter()
        u_prior = (
            behavior_policy.predict(
                state, goal_xy, direction, phase, orientation_turn
            )
            if behavior_policy is not None
            else np.zeros(ACTION_DIM)
        )
        policy_inference_time = perf_counter() - policy_start

        selection_start = perf_counter()
        if behavior_policy is not None:
            chart_id, gate_ratio, g_anchor, demonstrated_future = (
                select_behavior_with_policy(
                    atlas,
                    u_ini,
                    y_ini,
                    direction,
                    phase,
                    u_prior,
                    action_weight=cfg.policy_chart_selection_weight,
                    previous_chart=previous_chart,
                )
            )
        else:
            chart_id, gate_ratio = assign_chart(
                atlas,
                u_ini,
                y_ini,
                direction,
                phase,
                previous_chart=previous_chart,
                chart_exit_ratio=cfg.chart_exit_ratio,
                chart_switch_advantage=cfg.chart_switch_advantage,
                force_previous=(
                    previous_chart is not None
                    and atlas["charts"][int(previous_chart)]["phase"] == int(phase)
                    and chart_age < cfg.chart_minimum_dwell_steps
                ),
            )
            if cfg.lam_behavior_anchor > 0.0 or cfg.behavior_reference_weight > 0.0:
                g_anchor, demonstrated_future = nearest_behavior(
                    atlas, chart_id, u_ini, y_ini, direction
                )
            else:
                # Pure fixed-atlas mode: localization selects an offline chart,
                # but no individual demonstration is retrieved online.
                g_anchor = np.zeros(atlas["charts"][chart_id]["rank"])
                demonstrated_future = None
        reference_weight = (
            0.0 if phase == PHASE_HOLD else cfg.behavior_reference_weight
        )
        if reference_weight > 0.0:
            tip_reference = (
                reference_weight * demonstrated_future[:, TIP]
                + (1.0 - reference_weight) * tip_reference
            )
            object_reference = (
                reference_weight * demonstrated_future[:, OBJECT_XY]
                + (1.0 - reference_weight) * object_reference
            )
        behavior_selection_time = perf_counter() - selection_start

        qp_call_start = perf_counter()
        action, solve = controllers[chart_id].solve(
            u_ini,
            y_ini,
            previous_u,
            tip_reference,
            object_reference,
            g_anchor,
            u_prior,
        )
        qp_call_time = perf_counter() - qp_call_start

        postprocess_start = perf_counter()
        if action is None:
            action = np.zeros(ACTION_DIM)
        action = np.clip(action, -cfg.torque_limit, cfg.torque_limit)
        postprocess_time = perf_counter() - postprocess_start
        control_compute_time = perf_counter() - online_step_start

        environment_step_start = perf_counter()
        env.step(action.astype(np.float32))
        environment_step_time = perf_counter() - environment_step_start

        feedback_start = perf_counter()
        next_state = state_from_env(env)
        next_output = behavior_output(next_state, goal_xy)
        # Offline outputs are post-input states, so each new (u, y) pair is
        # appended after the MuJoCo step.
        u_history.append(action.copy())
        y_history.append(next_output)

        prediction_error = np.nan
        if "predicted_y" in solve:
            prediction_error = float(
                np.linalg.norm(
                    solve["predicted_y"][0, OBJECT_XY]
                    - next_output[OBJECT_XY]
                )
            )
        distance = object_goal_distance(next_state, goal_xy)
        hold_counter = hold_counter + 1 if distance <= cfg.success_radius else 0
        feedback_time = perf_counter() - feedback_start
        online_step_wall_time = perf_counter() - online_step_start
        records["state"].append(next_state)
        records["action"].append(action)
        records["distance"].append(distance)
        records["phase"].append(phase)
        records["chart"].append(chart_id)
        records["gate_ratio"].append(gate_ratio)
        records["solve_time"].append(solve["solve_time"])
        records["state_reference_time_s"].append(state_reference_time)
        records["policy_inference_time_s"].append(policy_inference_time)
        records["behavior_selection_time_s"].append(behavior_selection_time)
        records["qp_call_time_s"].append(qp_call_time)
        records["postprocess_time_s"].append(postprocess_time)
        records["control_compute_time_s"].append(control_compute_time)
        records["environment_step_time_s"].append(environment_step_time)
        records["feedback_time_s"].append(feedback_time)
        records["online_step_wall_time_s"].append(online_step_wall_time)
        records["status"].append(solve["status"])
        records["prediction_error"].append(prediction_error)
        records["slack_norm"].append(solve.get("slack_norm", np.nan))
        records["coefficient_norm"].append(solve.get("coefficient_norm", np.nan))
        records["policy_prior_deviation"].append(
            float(np.linalg.norm(action - u_prior))
        )
        previous_u = action
        chart_age = chart_age + 1 if previous_chart == chart_id else 0
        previous_chart = chart_id

    env.close()
    result = {
        key: np.asarray(value, dtype=object if key == "status" else None)
        for key, value in records.items()
    }
    finite_prediction = np.asarray(result["prediction_error"], dtype=float)
    result.update(
        seed=np.array(seed),
        goal_xy=goal_xy,
        dt=np.array(0.05),
        warmup_action=np.asarray(warmup_actions),
        success_radius=np.array(cfg.success_radius),
        success=np.array(hold_counter >= cfg.success_hold_steps),
        reached_goal=np.array(np.min(result["distance"]) <= cfg.success_radius),
        reached_10cm=np.array(np.min(result["distance"]) <= 0.10),
        minimum_distance=np.array(np.min(result["distance"])),
        final_distance=np.array(result["distance"][-1]),
        solve_rate=np.array(
            np.mean(np.char.startswith(result["status"].astype(str), "optimal"))
        ),
        median_solve_ms=np.array(1000.0 * np.median(result["solve_time"])),
        median_control_compute_ms=np.array(
            1000.0 * np.median(result["control_compute_time_s"])
        ),
        p95_control_compute_ms=np.array(
            1000.0 * np.quantile(result["control_compute_time_s"], 0.95)
        ),
        median_online_step_wall_ms=np.array(
            1000.0 * np.median(result["online_step_wall_time_s"])
        ),
        p95_online_step_wall_ms=np.array(
            1000.0 * np.quantile(result["online_step_wall_time_s"], 0.95)
        ),
        mean_prediction_error_xy=np.array(
            np.nanmean(finite_prediction)
            if np.any(np.isfinite(finite_prediction))
            else np.nan
        ),
    )
    return result


def _summary_from_rollouts(results):
    if not results:
        return {
            "success_rate": np.nan,
            "goal_reach_rate": np.nan,
            "reach_10cm_rate": np.nan,
            "mean_minimum_distance": np.nan,
            "mean_final_distance": np.nan,
            "mean_solve_rate": np.nan,
            "median_solve_ms": np.nan,
            "median_control_compute_ms": np.nan,
            "mean_control_compute_ms": np.nan,
            "p95_control_compute_ms": np.nan,
            "p99_control_compute_ms": np.nan,
            "median_online_step_wall_ms": np.nan,
            "p95_online_step_wall_ms": np.nan,
            "control_compute_within_period_rate": np.nan,
            "median_online_stage_ms": {},
            "mean_prediction_error_xy": np.nan,
            "on_policy_inside_chart_radius_rate": np.nan,
            "on_policy_gate_ratio_p95": np.nan,
            "median_policy_prior_deviation": np.nan,
        }
    gate_ratios = np.concatenate(
        [np.asarray(result["gate_ratio"], dtype=float) for result in results]
    )
    prior_deviations = np.concatenate(
        [np.asarray(result["policy_prior_deviation"], dtype=float) for result in results]
    )
    timing = {
        key: np.concatenate(
            [np.asarray(result[key], dtype=float) for result in results]
        )
        for key in (
            "state_reference_time_s",
            "policy_inference_time_s",
            "behavior_selection_time_s",
            "qp_call_time_s",
            "postprocess_time_s",
            "control_compute_time_s",
            "environment_step_time_s",
            "feedback_time_s",
            "online_step_wall_time_s",
        )
    }
    control_periods = np.concatenate(
        [
            np.full(len(result["control_compute_time_s"]), float(result["dt"]))
            for result in results
        ]
    )
    return {
        "success_rate": float(np.mean([bool(result["success"]) for result in results])),
        "goal_reach_rate": float(
            np.mean([bool(result["reached_goal"]) for result in results])
        ),
        "reach_10cm_rate": float(
            np.mean([bool(result["reached_10cm"]) for result in results])
        ),
        "mean_minimum_distance": float(
            np.mean([float(result["minimum_distance"]) for result in results])
        ),
        "mean_final_distance": float(
            np.mean([float(result["final_distance"]) for result in results])
        ),
        "mean_solve_rate": float(
            np.mean([float(result["solve_rate"]) for result in results])
        ),
        "median_solve_ms": float(
            np.median([float(result["median_solve_ms"]) for result in results])
        ),
        "mean_control_compute_ms": float(
            1000.0 * np.mean(timing["control_compute_time_s"])
        ),
        "median_control_compute_ms": float(
            1000.0 * np.median(timing["control_compute_time_s"])
        ),
        "p95_control_compute_ms": float(
            1000.0 * np.quantile(timing["control_compute_time_s"], 0.95)
        ),
        "p99_control_compute_ms": float(
            1000.0 * np.quantile(timing["control_compute_time_s"], 0.99)
        ),
        "median_online_step_wall_ms": float(
            1000.0 * np.median(timing["online_step_wall_time_s"])
        ),
        "p95_online_step_wall_ms": float(
            1000.0 * np.quantile(timing["online_step_wall_time_s"], 0.95)
        ),
        "control_compute_within_period_rate": float(
            np.mean(timing["control_compute_time_s"] <= control_periods)
        ),
        "median_online_stage_ms": {
            key.removesuffix("_time_s"): float(1000.0 * np.median(values))
            for key, values in timing.items()
        },
        "mean_prediction_error_xy": float(
            np.nanmean([float(result["mean_prediction_error_xy"]) for result in results])
        ),
        "on_policy_inside_chart_radius_rate": float(np.mean(gate_ratios <= 1.0)),
        "on_policy_gate_ratio_p95": float(np.quantile(gate_ratios, 0.95)),
        "median_policy_prior_deviation": float(np.median(prior_deviations)),
    }


def run_experiment(
    cfg: PusherAtlasConfig | None = None,
    output_dir: str | Path = "experiments/pusher_atlas/default",
    reuse_data: bool = True,
    probe_data: dict | None = None,
):
    cfg = PusherAtlasConfig() if cfg is None else cfg
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "offline_teacher_data.npz"
    if reuse_data and data_path.exists():
        data = load_data(data_path)
    else:
        data = collect_teacher_data(cfg, data_path)

    policy_path = output_dir / "offline_behavior_policy.joblib"
    regional_policy_path = output_dir / "offline_behavior_policy_regional.joblib"
    if reuse_data and regional_policy_path.exists():
        behavior_policy = load_behavior_policy(regional_policy_path)
    elif reuse_data and policy_path.exists():
        behavior_policy = load_behavior_policy(policy_path)
    else:
        behavior_policy = train_behavior_policy(data, cfg)
        save_behavior_policy(behavior_policy, policy_path)

    library = build_behavior_library(data, cfg)
    atlas = fit_atlas(library, cfg)
    validation = validate_atlas(library, atlas)
    coverage = None
    if "episode_metadata" in data:
        coverage = validate_offline_coverage(
            data, library, atlas, cfg, probe_data=probe_data
        )
        save_coverage_report(coverage, output_dir / "coverage_report.json")
        plot_coverage(data, coverage, output_dir / "offline_coverage_validation.png")
    save_atlas(output_dir / "behavioral_atlas.npz", atlas, cfg)
    plot_dataset_summary(data, output_dir / "offline_coverage.png")

    results = []
    for seed in cfg.test_seeds:
        result = run_rollout(atlas, cfg, int(seed), behavior_policy=behavior_policy)
        results.append(result)
        np.savez_compressed(output_dir / f"rollout_seed{seed}.npz", **result)
    if results:
        plot_rollout(results[0], output_dir / f"rollout_seed{cfg.test_seeds[0]}.png")

    summary = _summary_from_rollouts(results)
    summary.update(
        config=asdict(cfg),
        chart_count=len(atlas["charts"]),
        chart_sizes=[int(chart["sample_count"]) for chart in atlas["charts"]],
        chart_phases=[int(chart["phase"]) for chart in atlas["charts"]],
        chart_ranks=[int(chart["rank"]) for chart in atlas["charts"]],
        atlas_validation={
            key: value.tolist() if isinstance(value, np.ndarray) else float(value)
            for key, value in validation.items()
        },
        behavior_matrix_shape=list(map(int, library["H"].shape)),
        research_invariant=(
            "At every step one fixed, data-derived affine behavior chart enforces "
            "[u_ini + sigma_u; u; y_ini + sigma_y; y] = h_bar + V g. "
            "No A/B state-space model is identified or used. The phase manager and "
            "observed chart graph and offline regional intent prior select the local "
            "behavior and coefficient anchor; the applied action is always the first "
            "optimized u."
        ),
        controller="local affine DeePC QP",
        timing_definition={
            "clock": "time.perf_counter",
            "solve_time": "solver call inside LocalAffineDeePC.solve",
            "control_compute_time": (
                "state read, phase/reference construction, policy inference, "
                "behavior selection, QP call, and action clipping"
            ),
            "online_step_wall_time": (
                "control compute plus MuJoCo env.step and post-step feedback processing"
            ),
            "logging_excluded": True,
        },
        offline_behavior_policy=(
            f"{type(behavior_policy).__name__} first-input QP regularizer"
        ),
        policy_action_applied_directly=False,
        identified_state_space_model=False,
        online_teacher_actions_used=False,
        offline_ik_role="optional offline contact-rich demonstration generator only",
        windows_crossing_resets=0,
        offline_data_config=str(data.get("config", "unknown")),
    )
    if coverage is not None:
        summary["coverage_validation"] = serializable_coverage(coverage)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return {
        "config": cfg,
        "data": data,
        "library": library,
        "atlas": atlas,
        "validation": validation,
        "coverage": coverage,
        "behavior_policy": behavior_policy,
        "results": results,
        "summary": summary,
        "output_dir": output_dir,
    }
