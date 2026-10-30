"""Bounded, serial, paired study of the retained Pusher policy/DeePC artifacts.

No fitting or controller tuning is performed here. See the study README and
protocol.json for estimands, exclusions, and the benchmark's physics limitations.
"""

from __future__ import annotations

import os

# Set before importing numerical libraries. Also enforced by threadpoolctl below.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[_name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pusher_contribution_mpl")

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter
import traceback

import mujoco
import numpy as np
from scipy.stats import binomtest
from threadpoolctl import threadpool_info, threadpool_limits

from pusher_atlas.atlas import select_behavior_with_policy
from pusher_atlas.config import PusherAtlasConfig
from pusher_atlas.controller import LocalAffineDeePC, geometric_references, update_phase
from pusher_atlas.environment import (
    ACTION_DIM, OBJECT_XY, PHASE_APPROACH, behavior_output,
    goal_xy_from_env, make_env, object_goal_distance, push_direction, state_from_env,
)
from pusher_atlas.experiment import load_atlas, run_rollout
from pusher_atlas.policy import load_behavior_policy
from pusher_heldout_evaluation import wilson_interval


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/pusher_atlas/contribution_study_20260903"
SOURCE = ROOT / "experiments/pusher_atlas/rebuild_300_routed"
STANDARD = ROOT / "experiments/pusher_atlas/rebuild_300_standard"
METHODS = {
    "A_policy": {"selection": False, "regularization": False, "qp": False},
    "B_predictive": {"selection": False, "regularization": False, "qp": True},
    "C_combined": {"selection": True, "regularization": True, "qp": True},
    "D_selection": {"selection": True, "regularization": False, "qp": True},
    "E_regularization": {"selection": False, "regularization": True, "qp": True},
}
CONDITIONS = {
    "nominal": {},
    "friction": {"object_joint_frictionloss_N": 0.02},
    "mass": {"object_mass_kg": 0.05},
    "disturbance": {"step": 120, "displacement_m": 0.10},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def config():
    with np.load(SOURCE / "offline_teacher_data.npz", allow_pickle=False) as data:
        return PusherAtlasConfig(**json.loads(str(data["config"].item())))


def artifacts():
    return [SOURCE / name for name in (
        "offline_teacher_data.npz", "behavioral_atlas.npz",
        "offline_behavior_policy_regional.joblib", "heldout_coverage_probe_data.npz",
    )] + [STANDARD / name for name in (
        "offline_teacher_data.npz", "heldout_coverage_probe_data.npz",
    )]


def source_files():
    return [Path(__file__)] + sorted((ROOT / "src/pusher_atlas").glob("*.py")) + [
        ROOT / "src/pusher_contribution_recovery_pilot.py",
        ROOT / "src/render_pusher_study_trial.py",
        ROOT / "src/test_pusher_contribution_study.py",
    ]


def protocol_path(study, stage):
    return study / ("final_protocol.json" if stage == "final" else "protocol.json")


def provenance():
    import gymnasium

    xml = Path(gymnasium.__file__).resolve().parent / "envs/mujoco/assets/pusher_v5.xml"
    return {
        "artifacts_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in artifacts()},
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_files()},
        "packages": {name: importlib.metadata.version(name) for name in (
            "numpy", "scipy", "cvxpy", "osqp", "clarabel", "gymnasium", "mujoco",
            "scikit-learn", "joblib", "imageio", "threadpoolctl",
        )},
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pusher_xml_sha256": sha256(xml),
        "numerical_threads": threadpool_info(),
    }


def physics(env):
    m = env.unwrapped.model
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    joints = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
              for name in ("obj_slidex", "obj_slidey")]
    dofs = m.jnt_dofadr[joints]
    return {
        "object_mass_kg": float(m.body_mass[bid]),
        "object_inertia": m.body_inertia[bid].tolist(),
        "object_dof_armature": m.dof_armature[dofs].tolist(),
        "object_dof_damping": m.dof_damping[dofs].tolist(),
        "object_dof_frictionloss": m.dof_frictionloss[dofs].tolist(),
        "all_contact_dimensions": np.unique(m.geom_condim).tolist(),
        "gravity": m.opt.gravity.tolist(),
        "dt": float(env.unwrapped.dt),
        "xml_path": str(env.unwrapped.fullpath),
        "xml_sha256": sha256(env.unwrapped.fullpath),
    }


def apply_condition(env, condition, specification):
    plant = env.unwrapped
    m, d = plant.model, plant.data
    if condition == "mass":
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
        factor = specification["object_mass_kg"] / m.body_mass[bid]
        m.body_mass[bid] *= factor
        m.body_inertia[bid] *= factor
        # Recompute derived inertial constants without losing the reset state.
        qpos, qvel = d.qpos.copy(), d.qvel.copy()
        mujoco.mj_setConst(m, d)
        plant.set_state(qpos, qvel)
    elif condition == "friction":
        for name in ("obj_slidex", "obj_slidey"):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            m.dof_frictionloss[m.jnt_dofadr[jid]] = specification["object_joint_frictionloss_N"]
        mujoco.mj_forward(m, d)


def displace_object(env, delta_xy):
    plant = env.unwrapped
    for name, delta in zip(("obj_slidex", "obj_slidey"), delta_xy):
        jid = mujoco.mj_name2id(plant.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        plant.data.qpos[plant.model.jnt_qposadr[jid]] += delta
    # A kinematic relocation, not an impulse. Velocities are preserved.
    mujoco.mj_forward(plant.model, plant.data)


def audit(study):
    study.mkdir(parents=True, exist_ok=True)
    report = {"provenance": provenance(), "historical": {}, "data": {}}
    forbidden = set(range(101, 117)) | set(range(131, 161)) | set(range(201, 301))
    for folder in (SOURCE, STANDARD):
        for name in ("offline_teacher_data.npz", "heldout_coverage_probe_data.npz"):
            path = folder / name
            with np.load(path, allow_pickle=False) as data:
                cfg = json.loads(str(data["config"].item()))
                selected = np.asarray(data["selected_seeds"], int)
                pool = list(range(cfg["seed"], cfg["seed"] +
                                  cfg["offline_episodes"] * cfg["stratify_reset_pool_multiplier"]))
                forbidden.update(pool)
                forbidden.update(selected.tolist())
                report["data"][str(path.relative_to(ROOT))] = {
                    "transitions": len(data["u"]), "selected_seeds": selected,
                    "entire_candidate_pool": [pool[0], pool[-1]], "config": cfg,
                }
    for dirname in ("heldout201_300", "heldout201_300_pure_atlas"):
        folder = SOURCE.parent / dirname
        summary = json.loads((folder / "summary.json").read_text())
        rows, statuses = [], Counter()
        for seed in range(201, 301):
            path = folder / f"rollout_seed{seed}.npz"
            result = load_npz(path)
            distances = np.linalg.norm(result["state"][:, OBJECT_XY] - result["goal_xy"], axis=1)
            recomputed = bool(np.all(distances[-8:] <= 0.08))
            assert len(distances) == 240 and int(result["seed"]) == seed
            assert np.allclose(distances, result["distance"], atol=1e-12, rtol=0)
            assert recomputed == bool(result["success"])
            statuses.update(result["status"].astype(str).tolist())
            rows.append({"seed": seed, "success": recomputed,
                         "final_distance_m": float(distances[-1]), "sha256": sha256(path)})
        count = sum(row["success"] for row in rows)
        assert count == summary["successes"]
        report["historical"][dirname] = {
            "successes": count, "trials": len(rows),
            "wilson95": wilson_interval(count, len(rows)), "statuses": statuses,
            "per_trial": rows,
            "verification": "Recomputed from saved states; not a fresh 100-trial execution.",
        }
    candidate_dev = [51001, 51002]
    candidate_final = list(range(62001, 62101))
    assert not forbidden.intersection(candidate_dev + candidate_final)
    report["seed_audit"] = {
        "development_seeds": candidate_dev, "proposed_final_seeds": candidate_final,
        "known_forbidden_seeds": sorted(forbidden), "overlap": [],
        "limitation": "Checked retained corpora, full stratification candidate pools, scripts, "
                      "notebook and saved Pusher evaluations. Missing external history cannot be certified.",
    }
    policy = load_behavior_policy(SOURCE / "offline_behavior_policy_regional.joblib")
    decoder_rows = []
    for expert_name in ("standard_policy", "turned_policy"):
        expert = getattr(policy, expert_name)
        errors = []
        for folder in (SOURCE, STANDARD):
            with np.load(folder / "offline_teacher_data.npz", allow_pickle=False) as data:
                for index in np.linspace(0, len(data["x"]) - 1, 50, dtype=int):
                    state, goal, direction = data["x"][index], data["goal"][index], data["direction"][index]
                    phase, turn = int(data["phase"][index]), float(data["orientation_turn"][index])
                    f = expert.feature(state, goal, direction, phase, turn)[None, :]
                    models = expert.model if isinstance(expert.model, (list, tuple)) else [expert.model]
                    q = expert.action_scaler.inverse_transform(np.mean(
                        [model.predict(expert.input_scaler.transform(f)) for model in models], axis=0))[0]
                    expected = np.clip(expert.teacher_kp * (q - state[:7]) -
                                       expert.teacher_kd * state[7:14], -expert.torque_limit, expert.torque_limit)
                    errors.append(np.max(np.abs(expected - expert.predict(state, goal, direction, phase, turn))))
        assert max(errors) == 0 and expert.output_kind == "joint_target"
        decoder_rows.append({"expert": expert_name, "output_kind": expert.output_kind,
                             "samples": len(errors), "max_absolute_decoder_error": max(errors),
                             "kp": expert.teacher_kp, "kd": expert.teacher_kd,
                             "torque_limit": expert.torque_limit,
                             "hidden_layers": [m.hidden_layer_sizes for m in models]})
    report["decoder_validation"] = decoder_rows
    env, _ = make_env(candidate_dev[0], 250)
    report["nominal_physics"] = physics(env)
    env.close()
    # Deterministic saved-action replay is independent of the policy/QP implementation.
    replays = []
    for seed in (201, 202):
        saved = load_npz(SOURCE.parent / "heldout201_300" / f"rollout_seed{seed}.npz")
        env, _ = make_env(seed, 250)
        for action in saved["warmup_action"]:
            env.step(action.astype(np.float32))
        observed = []
        for action in saved["action"]:
            env.step(action.astype(np.float32))
            observed.append(state_from_env(env))
        env.close()
        replays.append({"seed": seed, "max_state_absolute_error": float(np.max(np.abs(
            np.asarray(observed) - saved["state"]))),
            "current_engine_success": bool(np.all(np.linalg.norm(
                np.asarray(observed)[-8:, OBJECT_XY] - saved["goal_xy"], axis=1) <= .08))})
    report["historical_action_replay"] = replays
    write_json(study / "audit.json", report)
    print(json.dumps(jsonable({"historical": {k: v["successes"] for k, v in report["historical"].items()},
                              "decoder": decoder_rows, "replay": replays,
                              "physics": report["nominal_physics"]})), flush=True)
    return report


def create_protocol(study):
    report = audit(study)
    path = study / "protocol.json"
    if path.exists():
        raise FileExistsError(f"Refusing to replace protocol: {path}")
    p = {
        "status": "development_pilot_predeclared",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_config": asdict(config()), "methods": METHODS, "conditions": CONDITIONS,
        "development_seeds": report["seed_audit"]["development_seeds"],
        "final_seeds": report["seed_audit"]["proposed_final_seeds"],
        "timing_warmup_control_steps": 20,
        "timing_exclude_first_call_each_chart": True,
        "bootstrap_replicates": 10000, "bootstrap_seed": 932607,
        "primary": "All final 8 post-control distances <=0.08 m after 240 steps at dt=.05 s.",
        "practical_success_difference_pp": 5,
        "family": "C vs A and C vs B in four conditions: 8 exact paired McNemar tests, Holm corrected.",
        "secondary": "Exploratory paired seed-bootstrap mean differences, percentile 95% CIs; "
                     "no equivalence claims. Timing has both warm-up-excluded and all-step values.",
        "resource_cap_seconds": 14400,
        "provenance": report["provenance"],
    }
    write_json(path, p)
    return p


def trial(atlas, base_cfg, policy, method, condition, specification, seed, timing_warmup=20):
    started = perf_counter()
    mode = METHODS[method]
    cfg = replace(base_cfg, policy_prior_weight=(base_cfg.policy_prior_weight if mode["regularization"] else 0.0))
    env, _ = make_env(seed, cfg.rollout_steps + cfg.t_ini + 2)
    apply_condition(env, condition, specification)
    actual_physics = physics(env)
    goal = goal_xy_from_env(env)
    initial_direction = push_direction(state_from_env(env), goal)
    turn = float(initial_direction[0] < cfg.collection_orientation_turn_threshold)
    displacement = specification.get("displacement_m", 0.0) * (1 if seed % 2 else -1) * np.array(
        [-initial_direction[1], initial_direction[0]])
    init_start = perf_counter()
    controllers = [LocalAffineDeePC(chart, cfg) for chart in atlas["charts"]] if mode["qp"] else []
    setup_time = perf_counter() - init_start
    past_u, past_y = [], []
    for _ in range(cfg.t_ini):
        action = np.zeros(ACTION_DIM)
        env.step(action.astype(np.float32))
        past_u.append(action)
        past_y.append(behavior_output(state_from_env(env), goal))
    phase, previous_chart = PHASE_APPROACH, None
    previous_u = np.zeros(ACTION_DIM)
    seen = set()
    records = {key: [] for key in (
        "state", "action", "raw_action", "prior", "distance", "phase", "chart", "status",
        "fallback", "control_compute_time_s", "environment_step_time_s", "feedback_time_s",
        "steady_timing_mask", "joint_violation_rad", "actuator_violation", "gate_ratio",
        "prediction_error", "unexpected_termination", "qpos", "qvel",
    )}
    joint_low, joint_high = env.unwrapped.model.jnt_range[:7].T
    error = None
    disturbance_pre_distance = disturbance_post_distance = np.nan
    try:
        for step in range(cfg.rollout_steps):
            if condition == "disturbance" and step == specification["step"]:
                disturbance_pre_distance = object_goal_distance(state_from_env(env), goal)
                displace_object(env, displacement)
                disturbance_post_distance = object_goal_distance(state_from_env(env), goal)
            control_start = perf_counter()
            state = state_from_env(env)
            phase = update_phase(state, goal, phase, step, cfg)
            tip_ref, obj_ref, direction = geometric_references(state, goal, phase, cfg)
            u_ini, y_ini = np.asarray(past_u[-cfg.t_ini:]), np.asarray(past_y[-cfg.t_ini:])
            use_policy = not mode["qp"] or mode["selection"] or mode["regularization"]
            prior = policy.predict(state, goal, direction, phase, turn) if use_policy else np.zeros(ACTION_DIM)
            chart, gate = -1, np.nan
            solve = {"status": "policy_only"}
            if mode["qp"]:
                # A common retrieval operator makes B/C/D/E a two-factor ablation.
                # With weight=0 the prior cannot affect chart/member selection.
                chart, gate, anchor, _ = select_behavior_with_policy(
                    atlas, u_ini, y_ini, direction, phase, prior,
                    action_weight=cfg.policy_chart_selection_weight if mode["selection"] else 0.0,
                    previous_chart=previous_chart)
                raw, solve = controllers[chart].solve(u_ini, y_ini, previous_u,
                                                      tip_ref, obj_ref, anchor, prior)
            else:
                raw = prior.copy()  # Exactly the existing joint-target -> PD -> clip decoder.
            fallback = raw is None or not np.all(np.isfinite(raw))
            if fallback:
                raw = np.zeros(ACTION_DIM)
            action = np.clip(raw, -cfg.torque_limit, cfg.torque_limit)
            compute_time = perf_counter() - control_start
            cold = mode["qp"] and chart not in seen
            seen.add(chart)
            sim_start = perf_counter()
            _, _, terminated, truncated, _ = env.step(action.astype(np.float32))
            sim_time = perf_counter() - sim_start
            feedback_start = perf_counter()
            next_state = state_from_env(env)
            output = behavior_output(next_state, goal)
            past_u.append(action.copy())
            past_y.append(output)
            distance = object_goal_distance(next_state, goal)
            feedback_time = perf_counter() - feedback_start
            prediction_error = (float(np.linalg.norm(solve["predicted_y"][0, OBJECT_XY] - output[OBJECT_XY]))
                                if "predicted_y" in solve else np.nan)
            values = {
                "state": next_state, "action": action, "raw_action": raw, "prior": prior,
                "distance": distance, "phase": phase, "chart": chart, "status": solve["status"],
                "fallback": fallback, "control_compute_time_s": compute_time,
                "environment_step_time_s": sim_time, "feedback_time_s": feedback_time,
                "steady_timing_mask": step >= timing_warmup and not cold,
                "joint_violation_rad": float(np.maximum.reduce([
                    joint_low - next_state[:7], next_state[:7] - joint_high, np.zeros(7)]).max()),
                "actuator_violation": float(max(0.0, np.max(np.abs(env.unwrapped.data.ctrl)) - cfg.torque_limit)),
                "gate_ratio": gate, "prediction_error": prediction_error,
                "unexpected_termination": bool(terminated or truncated),
                "qpos": env.unwrapped.data.qpos.copy(), "qvel": env.unwrapped.data.qvel.copy(),
            }
            for key, value in values.items():
                records[key].append(value)
            previous_u, previous_chart = action, chart
            if not np.all(np.isfinite(next_state)):
                raise RuntimeError("nonfinite simulator state")
            if terminated or truncated:
                raise RuntimeError("unexpected early environment termination")
    except Exception:
        error = traceback.format_exc()
    finally:
        env.close()
    result = {key: np.asarray(value) for key, value in records.items()}
    result.update(seed=np.array(seed), method=np.array(method), condition=np.array(condition),
                  goal_xy=goal, dt=np.array(actual_physics["dt"]),
                  warmup_action=np.zeros((cfg.t_ini, ACTION_DIM)),
                  error=np.array(error or ""), physics_json=np.array(json.dumps(actual_physics)),
                  disturbance_delta_xy=displacement,
                  disturbance_pre_distance=np.array(disturbance_pre_distance),
                  disturbance_post_distance=np.array(disturbance_post_distance),
                  setup_time_s=np.array(setup_time), trial_wall_time_s=np.array(perf_counter() - started))
    return result


def metrics(result, cfg, spec):
    distance = np.asarray(result["distance"])
    n = len(distance)
    complete = n == cfg.rollout_steps and not str(result["error"])
    inside = distance <= cfg.success_radius
    success = bool(complete and np.all(inside[-cfg.success_hold_steps:]))
    dt = float(result["dt"])
    # Settling time is entry into the FINAL uninterrupted in-goal suffix.
    last_outside = np.flatnonzero(~inside)
    settle_index = int(last_outside[-1] + 1) if len(last_outside) else 0
    settling = (settle_index + 1) * dt if success else None
    statuses = result["status"].astype(str)
    qp = METHODS[str(result["method"])]["qp"]
    steady = result["control_compute_time_s"][result["steady_timing_mask"].astype(bool)]
    action = result["action"]
    differences = np.diff(np.vstack([np.zeros((1, 7)), action]), axis=0) if n else np.empty((0, 7))
    disturbance = str(result["condition"]) == "disturbance"
    onset = int(spec.get("step", 0))
    prior_success = bool(disturbance and n >= onset and np.all(inside[onset-cfg.success_hold_steps:onset]))
    eligible = bool(prior_success and float(result["disturbance_post_distance"]) > cfg.success_radius)
    recovery = max(0.0, ((max(settle_index, onset) + 1) - onset) * dt) if disturbance and success else None
    row = {
        "seed": int(result["seed"]), "method": str(result["method"]), "condition": str(result["condition"]),
        "complete": complete, "error": str(result["error"]), "completed_steps": n,
        "success": success, "reached_goal": bool(np.any(inside)),
        "final_distance_m": float(distance[-1]) if n else None,
        "settling_time_s": settling,
        "capped_settling_time_s": settling if success else cfg.rollout_steps * dt,
        "recovery_eligible": eligible, "recovery_success": bool(eligible and success),
        "pre_disturbance_sustained_success": prior_success,
        "disturbance_pre_distance_m": float(result["disturbance_pre_distance"]) if disturbance else None,
        "disturbance_post_distance_m": float(result["disturbance_post_distance"]) if disturbance else None,
        "post_disturbance_settling_time_s": recovery,
        "capped_post_disturbance_settling_time_s": (
            recovery if success else (cfg.rollout_steps - onset) * dt
        ) if disturbance else None,
        "fallback_steps": int(np.sum(result["fallback"])),
        "solver_nonoptimal_steps": int(np.sum(~np.char.startswith(statuses, "optimal"))) if qp else 0,
        "solver_retry_steps": int(np.sum(np.char.find(statuses, "clarabel_retry") >= 0)),
        "solver_inaccurate_steps": int(np.sum(np.char.find(statuses, "inaccurate") >= 0)),
        "torque_violation_steps": int(np.sum(result["actuator_violation"] > 1e-6)),
        "max_torque_violation": float(np.max(result["actuator_violation"])) if n else None,
        "raw_qp_torque_exceedance_steps": int(np.sum(np.max(np.abs(result["raw_action"]), axis=1) > cfg.torque_limit + 1e-6)) if n and qp else 0,
        "joint_limit_violation_steps": int(np.sum(result["joint_violation_rad"] > 1e-4)),
        "max_joint_limit_excess_rad": float(np.max(result["joint_violation_rad"])) if n else None,
        "control_effort_integral": float(dt * np.sum(action ** 2)),
        "rms_action_increment": float(np.sqrt(np.mean(differences ** 2))) if n else None,
        "median_compute_ms": float(1000 * np.median(steady)) if len(steady) else None,
        "p95_compute_ms": float(1000 * np.quantile(steady, .95)) if len(steady) else None,
        "p99_compute_ms": float(1000 * np.quantile(steady, .99)) if len(steady) else None,
        "all_step_median_compute_ms": float(1000 * np.median(result["control_compute_time_s"])) if n else None,
        "all_step_p95_compute_ms": float(1000 * np.quantile(result["control_compute_time_s"], .95)) if n else None,
        "excluded_timing_steps": int(n - len(steady)),
        "median_simulation_ms": float(1000 * np.median(result["environment_step_time_s"])) if n else None,
        "trial_wall_time_s": float(result["trial_wall_time_s"]),
        "setup_time_s": float(result["setup_time_s"]),
    }
    return row


def validate_engine(study):
    cfg = config()
    atlas = load_atlas(SOURCE / "behavioral_atlas.npz")
    policy = load_behavior_policy(SOURCE / "offline_behavior_policy_regional.joblib")
    seed = 51001
    before = run_rollout(atlas, cfg, seed, behavior_policy=policy)
    after = trial(atlas, cfg, policy, "C_combined", "nominal", {}, seed)
    action_error = float(np.max(np.abs(before["action"] - after["action"])))
    state_error = float(np.max(np.abs(before["state"] - after["state"])))
    assert action_error < 1e-9 and state_error < 1e-9, (action_error, state_error)
    checks = {"seed": seed, "max_action_error_original_vs_study": action_error,
              "max_state_error_original_vs_study": state_error}
    # Neither the zero action weight nor the zero first-action penalty may carry
    # policy information. Selection outputs must be invariant to supplied prior.
    env, _ = make_env(seed, 250)
    goal = goal_xy_from_env(env)
    u, y = [], []
    for _ in range(cfg.t_ini):
        env.step(np.zeros(7, np.float32)); u.append(np.zeros(7))
        y.append(behavior_output(state_from_env(env), goal))
    direction = push_direction(state_from_env(env), goal)
    s0 = select_behavior_with_policy(atlas, u, y, direction, 0, np.zeros(7), action_weight=0)
    s1 = select_behavior_with_policy(atlas, u, y, direction, 0, np.full(7, 2.), action_weight=0)
    assert s0[0] == s1[0] and np.array_equal(s0[2], s1[2])
    checks["zero_weight_selection_invariant"] = True
    env.close()
    # Check parameter changes alter real dynamics, not just metadata.
    probe = []
    for condition in ("nominal", "friction", "mass"):
        env, _ = make_env(seed, 250)
        apply_condition(env, condition, CONDITIONS[condition])
        m, d = env.unwrapped.model, env.unwrapped.data
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "obj_slidex")
        d.qvel[m.jnt_dofadr[jid]] = .16
        mujoco.mj_forward(m, d)
        x0 = state_from_env(env)[OBJECT_XY].copy()
        for _ in range(10):
            env.step(np.zeros(7, np.float32))
        probe.append({"condition": condition, "displacement_0_5s_m": float(np.linalg.norm(
            state_from_env(env)[OBJECT_XY] - x0)), "physics": physics(env)})
        env.close()
    assert abs(probe[1]["displacement_0_5s_m"] - probe[0]["displacement_0_5s_m"]) > 1e-5
    assert abs(probe[2]["displacement_0_5s_m"] - probe[0]["displacement_0_5s_m"]) > 1e-5
    checks["dynamics_probe"] = probe
    write_json(study / "validation.json", checks)
    print(json.dumps(checks), flush=True)


def run_panel(study, stage):
    selected_protocol = protocol_path(study, stage)
    p = json.loads(selected_protocol.read_text())
    if stage == "final":
        lock = json.loads((study / "frozen_protocol.json").read_text())
        assert lock["protocol_sha256"] == sha256(selected_protocol)
        assert lock["provenance"] == provenance(), "Environment/source changed after freeze"
    cfg = PusherAtlasConfig(**p["base_config"])
    seeds = p["development_seeds"] if stage == "pilot" else p["final_seeds"]
    output = study / stage
    output.mkdir(exist_ok=True)
    manifest = {"protocol_sha256": sha256(selected_protocol), "provenance": provenance()}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, "Cannot mix configurations, code, or dependencies on resume"
    else:
        write_json(manifest_path, manifest)
    atlas = load_atlas(SOURCE / "behavioral_atlas.npz")
    policy = load_behavior_policy(SOURCE / "offline_behavior_policy_regional.joblib")
    started = perf_counter()
    methods = list(p["methods"])
    conditions = list(p["conditions"])
    completed = 0
    for seed_index, seed in enumerate(seeds):
        for offset in range(len(conditions)):
            condition_index = (offset + seed_index) % len(conditions)
            condition = conditions[condition_index]
            rotation = (seed_index + condition_index) % len(methods)
            for method in methods[rotation:] + methods[:rotation]:
                path = output / condition / method / f"seed{seed}.npz"
                row_path = path.with_suffix(".json")
                if not path.exists():
                    if perf_counter() - started > p["resource_cap_seconds"]:
                        raise RuntimeError("Predeclared per-invocation wall-clock cap reached; completed trials preserved")
                    result = trial(atlas, cfg, policy, method, condition, p["conditions"][condition],
                                   seed, p["timing_warmup_control_steps"])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(".partial.npz")
                    np.savez_compressed(temporary, **result)
                    temporary.replace(path)
                    row = metrics(result, cfg, p["conditions"][condition])
                    write_json(row_path, row)
                else:
                    result = load_npz(path)
                    row = metrics(result, cfg, p["conditions"][condition])
                    if not row_path.exists():
                        write_json(row_path, row)
                completed += 1
                print(json.dumps({"stage": stage, "trial": completed,
                                  "total": len(seeds) * len(methods) * len(conditions),
                                  **{k: row[k] for k in ("seed", "condition", "method", "success", "final_distance_m", "trial_wall_time_s")}}), flush=True)
    write_json(output / "run_environment.json", {
        "provenance": provenance(), "wall_time_s_this_invocation": perf_counter() - started,
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "load_average_at_end": os.getloadavg(),
    })
    summarize(study, stage)


def paired_comparison(left, right, p, rng):
    assert [r["seed"] for r in left] == [r["seed"] for r in right]
    n = len(left)
    a = np.array([r["success"] for r in left], int)
    b = np.array([r["success"] for r in right], int)
    win, lose = int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 0) & (b == 1)))
    draws = rng.integers(0, n, (p["bootstrap_replicates"], n))
    answer = {"n_pairs": n, "combined_only": win, "comparator_only": lose,
              "both_success": int(np.sum((a == 1) & (b == 1))),
              "both_fail": int(np.sum((a == 0) & (b == 0))),
              "mcnemar_exact_p": float(binomtest(win, win + lose, .5).pvalue) if win + lose else 1.0}
    for key in ("success", "final_distance_m", "capped_settling_time_s",
                "control_effort_integral", "rms_action_increment", "median_compute_ms",
                "max_joint_limit_excess_rad"):
        x = np.array([r[key] if r[key] is not None else np.nan for r in left], float)
        y = np.array([r[key] if r[key] is not None else np.nan for r in right], float)
        if not np.all(np.isfinite(x - y)):
            answer[key] = {"not_estimated": "Missing outcome; retained as failed trial, not silently dropped."}
            continue
        scale = 100 if key == "success" else 1
        delta = (x - y) * scale
        answer[key] = {"mean_difference": float(delta.mean()),
                       "paired_bootstrap95": np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist(),
                       "units": "percentage_points" if key == "success" else key}
    if left[0]["condition"] == "disturbance":
        eligible = np.array([x["recovery_eligible"] and y["recovery_eligible"] for x, y in zip(left, right)])
        answer["common_pre_disturbance_successful_pairs"] = int(eligible.sum())
        answer["recovery_common_eligible"] = {
            "combined_successes": int(a[eligible].sum()), "comparator_successes": int(b[eligible].sum()),
            "note": "Both methods were sustained in-goal before, and outside immediately after, disturbance. "
                    "Descriptive selected subgroup; overall post-disturbance success is primary.",
        }
        if np.any(eligible):
            count = int(eligible.sum())
            resamples = rng.integers(0, count, (p["bootstrap_replicates"], count))
            delta = (a[eligible] - b[eligible]) * 100
            recovery_summary = answer["recovery_common_eligible"]
            recovery_summary["success_difference_pp"] = float(delta.mean())
            recovery_summary["success_difference_paired_bootstrap95_pp"] = np.quantile(
                delta[resamples].mean(axis=1), [.025, .975]).tolist()
            x = np.array([r["capped_post_disturbance_settling_time_s"] for r in left], float)[eligible]
            y = np.array([r["capped_post_disturbance_settling_time_s"] for r in right], float)[eligible]
            recovery_summary["capped_time_difference_s"] = float((x - y).mean())
            recovery_summary["capped_time_difference_paired_bootstrap95_s"] = np.quantile(
                (x - y)[resamples].mean(axis=1), [.025, .975]).tolist()
            jointly_recovered = eligible & (a == 1) & (b == 1)
            recovery_summary["both_recovered_pairs"] = int(jointly_recovered.sum())
            if np.any(jointly_recovered):
                delta_time = np.array([
                    x["post_disturbance_settling_time_s"] - y["post_disturbance_settling_time_s"]
                    for x, y, keep in zip(left, right, jointly_recovered) if keep
                ])
                time_draws = rng.integers(0, len(delta_time), (p["bootstrap_replicates"], len(delta_time)))
                recovery_summary["both_recovered_time_difference_s"] = float(delta_time.mean())
                recovery_summary["both_recovered_time_difference_paired_bootstrap95_s"] = np.quantile(
                    delta_time[time_draws].mean(axis=1), [.025, .975]).tolist()
    return answer


def summarize(study, stage):
    import csv
    p = json.loads(protocol_path(study, stage).read_text())
    cfg = PusherAtlasConfig(**p["base_config"])
    output = study / stage
    rows, aggregates, comparisons = [], [], {}
    grouped = {}
    for condition in p["conditions"]:
        for method in p["methods"]:
            results = [load_npz(path) for path in sorted((output / condition / method).glob("seed*.npz"))]
            current = [metrics(result, cfg, p["conditions"][condition]) for result in results]
            rows.extend(current)
            grouped[(condition, method)] = current
            if not current:
                continue
            count = sum(r["success"] for r in current)
            timing = np.concatenate([r["control_compute_time_s"][r["steady_timing_mask"].astype(bool)] for r in results])
            all_timing = np.concatenate([r["control_compute_time_s"] for r in results])
            agg = {"condition": condition, "method": method, "n": len(current),
                   "successes": count, "success_percent": 100 * count / len(current),
                   "success_wilson95_percent": np.array(wilson_interval(count, len(current))) * 100,
                   "failed_or_incomplete_trials": sum(not r["complete"] for r in current),
                   "mean_final_distance_m": float(np.mean([r["final_distance_m"] for r in current]))
                       if all(r["final_distance_m"] is not None for r in current) else None,
                   "median_compute_ms": float(1000 * np.median(timing)),
                   "p95_compute_ms": float(1000 * np.quantile(timing, .95)),
                   "p99_compute_ms": float(1000 * np.quantile(timing, .99)),
                   "all_step_p95_compute_ms": float(1000 * np.quantile(all_timing, .95)),
                   "compute_over_50ms_percent": float(100 * np.mean(timing > .05)),
                   "sum_trial_wall_time_s": sum(r["trial_wall_time_s"] for r in current),
                   "mean_capped_settling_time_s": float(np.mean([r["capped_settling_time_s"] for r in current])),
                   "mean_control_effort_integral": float(np.mean([r["control_effort_integral"] for r in current])),
                   "mean_rms_action_increment": float(np.mean([r["rms_action_increment"] for r in current])),
                   "recovery_eligible_trials": sum(r["recovery_eligible"] for r in current),
                   "recovery_successes": sum(r["recovery_success"] for r in current)}
            for key in ("fallback_steps", "solver_nonoptimal_steps", "solver_retry_steps", "solver_inaccurate_steps",
                        "torque_violation_steps", "joint_limit_violation_steps", "raw_qp_torque_exceedance_steps"):
                agg[key] = sum(r[key] for r in current)
            aggregates.append(agg)
    rng = np.random.default_rng(p["bootstrap_seed"])
    for condition in p["conditions"]:
        left = grouped[(condition, "C_combined")]
        for method in ("A_policy", "B_predictive", "D_selection", "E_regularization"):
            right = grouped[(condition, method)]
            if left and len(left) == len(right):
                comparisons[f"{condition}:C_minus_{method}"] = paired_comparison(left, right, p, rng)
    primary = [k for k in comparisons if k.endswith(("A_policy", "B_predictive"))]
    ordered = sorted(primary, key=lambda k: comparisons[k]["mcnemar_exact_p"])
    previous = 0.0
    for i, key in enumerate(ordered):
        adjusted = min(1.0, (len(ordered) - i) * comparisons[key]["mcnemar_exact_p"])
        previous = max(previous, adjusted)
        comparisons[key]["holm_adjusted_p"] = previous
    summary = {"stage": stage, "completed_trials": len(rows), "aggregates": aggregates,
               "paired_comparisons": comparisons,
               "pilot_warning": "Development only; not held-out evidence." if stage == "pilot" else None}
    if stage == "pilot" and rows:
        projected = sum(r["trial_wall_time_s"] for r in rows) * len(p["final_seeds"]) / len(p["development_seeds"])
        summary["projected_final_serial_seconds"] = projected
        summary["projected_final_with_30percent_margin_seconds"] = projected * 1.3
        size = sum(path.stat().st_size for path in output.rglob("seed*.npz"))
        summary["projected_final_rollout_bytes"] = size * len(p["final_seeds"]) / len(p["development_seeds"])
    write_json(output / "summary.json", summary)
    if rows:
        with (output / "per_trial.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    lines = [f"# {stage.capitalize()} results", "", "Development results only; not a held-out claim." if stage == "pilot" else "Frozen paired evaluation.", "",
             "| Condition | Method | Success | Mean final distance (cm) | Compute median / p95 (ms) |", "|---|---|---:|---:|---:|"]
    for a in aggregates:
        distance = "missing" if a["mean_final_distance_m"] is None else f"{100*a['mean_final_distance_m']:.2f}"
        lines.append(f"| {a['condition']} | {a['method']} | {a['successes']}/{a['n']} | {distance} | {a['median_compute_ms']:.2f} / {a['p95_compute_ms']:.2f} |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(jsonable({k: v for k, v in summary.items() if k not in ("paired_comparisons", "aggregates")})), flush=True)


def freeze(study):
    p = json.loads((study / "protocol.json").read_text())
    summary = json.loads((study / "pilot/summary.json").read_text())
    assert summary["completed_trials"] == len(p["development_seeds"]) * len(p["methods"]) * len(p["conditions"])
    assert (study / "validation.json").exists()
    path = study / "frozen_protocol.json"
    if path.exists():
        raise FileExistsError(path)
    recovery = json.loads((study / "recovery_pilot/summary.json").read_text())
    assert recovery["completed_trials"] == 10
    assert any(row["recovery_eligible"] for row in recovery["results"])
    p["status"] = "frozen_before_final_evaluation"
    p["conditions"]["disturbance"] = recovery["condition"]
    p["development_decision"] = (
        "Only disturbance time changed from 6 to 8 seconds: original pilot had zero "
        "pre-goal recovery-eligible trials; a predeclared 10-trial check established eligibility. "
        "The difficult mass/sliding-resistance conditions and all controller settings were retained."
    )
    p["provenance"] = provenance()
    write_json(study / "final_protocol.json", p)
    write_json(path, {"frozen_utc": datetime.now(timezone.utc).isoformat(),
                      "protocol_sha256": sha256(study / "final_protocol.json"),
                      "pilot_summary_sha256": sha256(study / "pilot/summary.json"),
                      "recovery_pilot_summary_sha256": sha256(study / "recovery_pilot/summary.json"),
                      "provenance": provenance(),
                      "note": p["development_decision"]})


def render_selected(study, stage):
    import imageio.v2 as imageio
    p = json.loads(protocol_path(study, stage).read_text())
    output = study / stage
    rows = [json.loads(path.read_text()) for path in output.glob("*/*/seed*.json")]
    # Predeclared representative selection: lowest-seed success/failure for A/C
    # nominal and disturbance, at most eight. No aesthetic/outcome cherry picking.
    selected = []
    for condition in ("nominal", "disturbance"):
        for method in ("A_policy", "C_combined"):
            for success in (True, False):
                eligible = sorted([r for r in rows if r["condition"] == condition and r["method"] == method and r["success"] == success], key=lambda r: r["seed"])
                if eligible:
                    selected.append(eligible[0])
    index = []
    for row in selected:
        path = output / row["condition"] / row["method"] / f"seed{row['seed']}.npz"
        result = load_npz(path)
        env, _ = make_env(row["seed"], 250, "rgb_array")
        apply_condition(env, row["condition"], p["conditions"][row["condition"]])
        video = output / "videos" / f"{row['condition']}_{row['method']}_seed{row['seed']}_{'success' if row['success'] else 'failure'}.mp4"
        video.parent.mkdir(exist_ok=True)
        errors = []
        with imageio.get_writer(video, fps=20, codec="libx264", quality=7) as writer:
            writer.append_data(env.render())
            for action in result["warmup_action"]:
                env.step(action.astype(np.float32)); writer.append_data(env.render())
            for step, action in enumerate(result["action"]):
                if row["condition"] == "disturbance" and step == p["conditions"]["disturbance"]["step"]:
                    displace_object(env, result["disturbance_delta_xy"])
                env.step(action.astype(np.float32))
                errors.append(float(np.max(np.abs(state_from_env(env) - result["state"][step]))))
                writer.append_data(env.render())
        env.close()
        assert max(errors) < 1e-9, "Replay must match the actual perturbed dynamics"
        index.append({"video": str(video.relative_to(study)), "source": str(path.relative_to(study)),
                      "seed": row["seed"], "success": row["success"], "max_replay_error": max(errors)})
        print(json.dumps(index[-1]), flush=True)
    write_json(output / "videos/index.json", index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "init", "validate", "pilot", "freeze", "final", "summarize", "videos"))
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--stage", choices=("pilot", "final"), default="final")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.command == "audit": audit(args.study)
        elif args.command == "init": create_protocol(args.study)
        elif args.command == "validate": validate_engine(args.study)
        elif args.command in ("pilot", "final"): run_panel(args.study, args.command)
        elif args.command == "freeze": freeze(args.study)
        elif args.command == "summarize": summarize(args.study, args.stage)
        elif args.command == "videos": render_selected(args.study, args.stage)


if __name__ == "__main__":
    main()
