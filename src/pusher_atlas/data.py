"""Contact-rich offline data collection for Pusher Atlas-DeePC."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np

from .config import PusherAtlasConfig
from .environment import (
    ACTION_DIM,
    OBJECT_XY,
    PHASE_APPROACH,
    PHASE_HOLD,
    PHASE_PUSH,
    TIP,
    goal_xy_from_env,
    make_env,
    object_goal_distance,
    push_direction,
    state_from_env,
)


def _tip_body_id(env) -> int:
    import mujoco

    return int(
        mujoco.mj_name2id(
            env.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, "tips_arm"
        )
    )


def solve_tip_pose(env, target, direction, q_seed, restarts=1, random_seed=0):
    """Offline-only damped IK for the U-shaped pusher body pose."""
    import mujoco
    from scipy.optimize import least_squares

    plant = env.unwrapped
    body_id = _tip_body_id(env)
    q_saved = plant.data.qpos.copy()
    v_saved = plant.data.qvel.copy()
    low = plant.model.jnt_range[:7, 0] + 2e-4
    high = plant.model.jnt_range[:7, 1] - 2e-4
    direction = np.asarray(direction, dtype=float)
    desired_rotation = np.array(
        [
            [direction[0], -direction[1], 0.0],
            [direction[1], direction[0], 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    def kinematics(q):
        plant.data.qpos[:7] = q
        mujoco.mj_forward(plant.model, plant.data)
        position = plant.get_body_com("tips_arm").copy()
        rotation = plant.data.xmat[body_id].reshape(3, 3).copy()
        return position, rotation

    def residual(q):
        position, rotation = kinematics(q)
        return np.concatenate(
            [
                25.0 * (position - target),
                1.5 * (rotation - desired_rotation).ravel(),
                0.02 * (q - q_seed),
            ]
        )

    rng = np.random.default_rng(random_seed)
    starts = [np.clip(np.asarray(q_seed, dtype=float), low, high)]
    starts.extend(rng.uniform(low, high) for _ in range(max(0, restarts - 1)))
    best = None
    for initial in starts:
        result = least_squares(
            residual,
            initial,
            bounds=(low, high),
            max_nfev=180,
        )
        value = float(np.linalg.norm(residual(result.x)))
        if best is None or value < best[0]:
            best = (value, result.x.copy())
    plant.set_state(q_saved, v_saved)
    return best[1]


def reset_feature(state, goal_xy):
    direction = push_direction(state, goal_xy)
    return np.r_[
        direction,
        4.0 * object_goal_distance(state, goal_xy),
        3.0 * state[OBJECT_XY],
        2.0 * (state[TIP.start : TIP.start + 2] - state[OBJECT_XY]),
    ]


def select_stratified_reset_seeds(
    cfg: PusherAtlasConfig,
    start_seed: int | None = None,
    episodes: int | None = None,
):
    """Greedy k-center sampling over candidate Pusher reset geometries."""
    count = cfg.offline_episodes if episodes is None else int(episodes)
    first_seed = cfg.seed if start_seed is None else int(start_seed)
    pool_count = max(count, cfg.stratify_reset_pool_multiplier * count)
    env, _ = make_env(first_seed, 2)
    candidate_seeds = np.arange(first_seed, first_seed + pool_count, dtype=int)
    features = []
    for seed in candidate_seeds:
        env.reset(seed=int(seed))
        state = state_from_env(env)
        features.append(reset_feature(state, goal_xy_from_env(env)))
    env.close()
    features = np.asarray(features, dtype=float)
    normalized = (features - features.mean(axis=0)) / np.maximum(
        features.std(axis=0), 1e-8
    )

    # Include a representative central reset, then greedily cover the extrema.
    selected = [int(np.argmin(np.linalg.norm(normalized, axis=1)))]
    nearest = np.linalg.norm(normalized - normalized[selected[0]], axis=1)
    nearest[selected[0]] = -np.inf
    while len(selected) < count:
        index = int(np.argmax(nearest))
        selected.append(index)
        nearest = np.minimum(nearest, np.linalg.norm(normalized - normalized[index], axis=1))
        nearest[selected] = -np.inf
    return candidate_seeds[selected], features[selected]


def _variant(cfg: PusherAtlasConfig, episode: int):
    lengths = {
        len(cfg.collection_precontact_offsets),
        len(cfg.collection_push_offsets),
        len(cfg.collection_dither_scales),
        len(cfg.collection_recovery_variants),
        len(cfg.collection_warmup_steps),
    }
    if len(lengths) != 1 or not lengths or 0 in lengths:
        raise ValueError("All collection variant tuples must have the same nonzero length")
    index = int(episode % next(iter(lengths)))
    return {
        "id": index,
        "precontact_offset": float(cfg.collection_precontact_offsets[index]),
        "push_offset": float(cfg.collection_push_offsets[index]),
        "dither": float(cfg.collection_dither_scales[index]),
        "recovery": bool(cfg.collection_recovery_variants[index]),
        "warmup_steps": int(cfg.collection_warmup_steps[index]),
    }


def _next_teacher_phase(step, state, goal_xy, precontact, phase, cfg):
    distance = object_goal_distance(state, goal_xy)
    if distance <= cfg.success_radius:
        return PHASE_HOLD
    if phase == PHASE_HOLD:
        return PHASE_PUSH
    if phase == PHASE_APPROACH:
        if step >= cfg.approach_steps or np.linalg.norm(state[TIP] - precontact) <= cfg.approach_tolerance:
            return PHASE_PUSH
        return PHASE_APPROACH
    if np.linalg.norm(state[TIP.start : TIP.start + 2] - state[OBJECT_XY]) > cfg.contact_lost_distance:
        return PHASE_APPROACH
    return PHASE_PUSH


def collect_teacher_data(
    cfg: PusherAtlasConfig,
    save_path: Path | None = None,
    selected_seeds: np.ndarray | None = None,
):
    """Collect stratified, excited contact trajectories for offline DeePC only.

    Five deterministic policy variants change the approach/contact geometry,
    correlated excitation, and occasional recovery disturbance.  The selected
    reset seeds cover the candidate reset distribution by greedy k-center.
    """
    rng = np.random.default_rng(cfg.seed + 17)
    records = {
        key: []
        for key in [
            "x",
            "u",
            "x_next",
            "goal",
            "direction",
            "phase",
            "episode",
            "step",
            "variant",
            "contact",
            "q_target",
            "orientation_turn",
        ]
    }
    episode_summaries = []
    episode_metadata = []
    if selected_seeds is None:
        selected_seeds, reset_features = select_stratified_reset_seeds(cfg)
    else:
        selected_seeds = np.asarray(selected_seeds, dtype=int)
        reset_features = np.full((len(selected_seeds), 7), np.nan)
    if len(selected_seeds) != cfg.offline_episodes:
        raise ValueError("selected_seeds must contain cfg.offline_episodes entries")

    for episode, env_seed in enumerate(selected_seeds):
        variant = _variant(cfg, episode)
        env, _ = make_env(int(env_seed), cfg.episode_steps + 2)
        state = state_from_env(env)
        goal_xy = goal_xy_from_env(env)
        direction = push_direction(state, goal_xy)
        initial_direction = direction.copy()
        orientation_turn = bool(
            initial_direction[0] < cfg.collection_orientation_turn_threshold
        )
        angle = (
            float(cfg.collection_orientation_turn_radians)
            if orientation_turn
            else 0.0
        )
        cosine, sine = np.cos(angle), np.sin(angle)
        orientation_direction = np.array(
            [
                cosine * direction[0] - sine * direction[1],
                sine * direction[0] + cosine * direction[1],
            ]
        )
        object_z = float(env.unwrapped.get_body_com("object")[2])
        precontact = np.r_[
            state[OBJECT_XY] - variant["precontact_offset"] * direction,
            object_z,
        ]
        q_target = solve_tip_pose(
            env,
            precontact,
            orientation_direction,
            state[:7],
            restarts=cfg.ik_restarts,
            random_seed=cfg.seed + 1000 + episode,
        )
        phase = PHASE_APPROACH
        previous_phase = phase
        excitation = np.zeros(ACTION_DIM)
        initial_distance = object_goal_distance(state, goal_xy)
        min_distance = initial_distance
        max_object_motion = 0.0
        contact_steps = 0
        recovery_count = 0
        initial_object = state[OBJECT_XY].copy()

        for step in range(cfg.episode_steps):
            state = state_from_env(env)
            direction = push_direction(state, goal_xy)
            orientation_direction = np.array(
                [
                    cosine * direction[0] - sine * direction[1],
                    sine * direction[0] + cosine * direction[1],
                ]
            )
            precontact = np.r_[
                state[OBJECT_XY] - variant["precontact_offset"] * direction,
                object_z,
            ]
            teacher_step = max(0, step - variant["warmup_steps"])
            if step < variant["warmup_steps"]:
                phase = PHASE_APPROACH
            else:
                phase = _next_teacher_phase(
                    teacher_step, state, goal_xy, precontact, phase, cfg
                )
            if phase == PHASE_APPROACH:
                target = precontact
            elif phase == PHASE_PUSH:
                target = np.r_[
                    state[OBJECT_XY] - variant["push_offset"] * direction,
                    object_z,
                ]
            else:
                target = np.r_[
                    goal_xy - variant["precontact_offset"] * direction,
                    object_z,
                ]

            if (step > 0 and step % cfg.ik_period == 0) or phase != previous_phase:
                q_target = solve_tip_pose(
                    env,
                    target,
                    orientation_direction,
                    state[:7],
                    restarts=1,
                    random_seed=cfg.seed + 10000 + episode * cfg.episode_steps + step,
                )
            rho = 0.72
            excitation = rho * excitation + rng.normal(
                0.0, variant["dither"] * np.sqrt(1.0 - rho * rho), ACTION_DIM
            )
            if step < variant["warmup_steps"]:
                action = rng.normal(
                    0.0, cfg.collection_warmup_torque_scale, ACTION_DIM
                )
            else:
                action = (
                    cfg.teacher_kp * (q_target - state[:7])
                    - cfg.teacher_kd * state[7:14]
                    + excitation
                )
            recovery_active = (
                variant["recovery"]
                and cfg.recovery_start_step <= step < cfg.recovery_start_step + cfg.recovery_steps
            )
            if recovery_active:
                action += rng.normal(0.0, cfg.recovery_torque_scale, ACTION_DIM)
                recovery_count += 1
            action = np.clip(action, -cfg.torque_limit, cfg.torque_limit)
            env.step(action.astype(np.float32))
            next_state = state_from_env(env)
            contact = (
                np.linalg.norm(
                    next_state[TIP.start : TIP.start + 2] - next_state[OBJECT_XY]
                )
                <= 0.09
                and next_state[TIP.stop - 1] <= -0.20
            )

            records["x"].append(state)
            records["u"].append(action)
            records["x_next"].append(next_state)
            records["goal"].append(goal_xy.copy())
            records["direction"].append(direction)
            records["phase"].append(phase)
            records["episode"].append(episode)
            records["step"].append(step)
            records["variant"].append(variant["id"])
            records["contact"].append(contact)
            # Preserve the smooth intent behind the saturated PD torque.  These
            # targets are optional labels for the offline action-prior learner;
            # the DeePC behavioral matrix continues to use only measured u/y.
            records["q_target"].append(q_target.copy())
            records["orientation_turn"].append(orientation_turn)
            min_distance = min(min_distance, object_goal_distance(next_state, goal_xy))
            max_object_motion = max(
                max_object_motion,
                float(np.linalg.norm(next_state[OBJECT_XY] - initial_object)),
            )
            contact_steps += int(contact)
            previous_phase = phase
        env.close()
        episode_summaries.append((initial_distance, min_distance, max_object_motion))
        episode_metadata.append(
            (
                int(env_seed),
                variant["id"],
                contact_steps / cfg.episode_steps,
                float(min_distance <= cfg.success_radius),
                recovery_count,
                *initial_direction,
                *initial_object,
                *goal_xy,
            )
        )

    data = {key: np.asarray(value) for key, value in records.items()}
    data["episode_summary"] = np.asarray(episode_summaries)
    data["episode_metadata"] = np.asarray(episode_metadata)
    data["selected_seeds"] = np.asarray(selected_seeds, dtype=int)
    data["selected_reset_features"] = np.asarray(reset_features, dtype=float)
    data["episode_metadata_columns"] = np.asarray(
        [
            "seed",
            "variant",
            "contact_fraction",
            "success",
            "recovery_steps",
            "direction_x",
            "direction_y",
            "object_x",
            "object_y",
            "goal_x",
            "goal_y",
        ]
    )
    data["config"] = np.array(json.dumps(asdict(cfg), sort_keys=True))
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, **data)
    return data


def load_data(path: Path):
    with np.load(path, allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def select_context_episodes(data: dict, state, goal_xy, count: int):
    """Select offline episodes using only their initial measured reset geometry."""
    episode_ids = np.unique(np.asarray(data["episode"], dtype=int))
    features = []
    for episode in episode_ids:
        first = int(np.flatnonzero(np.asarray(data["episode"], dtype=int) == episode)[0])
        features.append(reset_feature(data["x"][first], data["goal"][first]))
    features = np.asarray(features, dtype=float)
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-8)
    target = (reset_feature(state, goal_xy) - mean) / scale
    distance = np.linalg.norm((features - mean) / scale - target, axis=1)
    selected = np.argsort(distance)[: min(int(count), len(episode_ids))]
    return episode_ids[selected], distance[selected]


def subset_episodes(data: dict, selected_episodes):
    """Return a transition-level dataset restricted to complete episodes."""
    selected_episodes = np.asarray(selected_episodes, dtype=int)
    mask = np.isin(np.asarray(data["episode"], dtype=int), selected_episodes)
    transition_count = len(mask)
    subset = {}
    for key, value in data.items():
        array = np.asarray(value)
        if array.ndim >= 1 and len(array) == transition_count:
            subset[key] = array[mask]
        else:
            subset[key] = array
    return subset
