"""Matched Atlas-DeePC pilots on compact dm-control robotic tasks.

The benchmark keeps the central Reacher protocol fixed: Select-DPC IID-only
offline data, output-configuration K-means charts, an observed transition
graph, cached rank-reduced affine behavioral bases, graph-routed waypoint
references, and a causally triggered contextual atlas that scores prototypes
rather than all Hankel columns online.
"""

from __future__ import annotations

import argparse
import contextlib
import heapq
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")

import cvxpy as cp
import numpy as np
from dm_control import suite
from sklearn.cluster import MiniBatchKMeans


T_INI = 2
T_F = 10
L = T_INI + T_F


@dataclass(frozen=True)
class TaskSpec:
    name: str
    domain: str
    task: str
    output_keys: tuple[str, ...]
    physical_indices: tuple[int, ...]
    goal_indices: tuple[int, ...]
    reference_zero_indices: tuple[int, ...]
    q_diagonal: tuple[float, ...]
    success_threshold: float
    latch_threshold: float

    @property
    def output_dim(self) -> int:
        return len(self.q_diagonal)


TASKS = {
    "point_mass_easy": TaskSpec(
        name="point_mass_easy",
        domain="point_mass",
        task="easy",
        output_keys=("position", "velocity"),
        physical_indices=(0, 1),
        goal_indices=(0, 1),
        reference_zero_indices=(2, 3),
        q_diagonal=(1.0e5, 1.0e5, 20.0, 20.0),
        success_threshold=0.03,
        latch_threshold=0.07,
    ),
    "cartpole_swingup": TaskSpec(
        name="cartpole_swingup",
        domain="cartpole",
        task="swingup",
        output_keys=("position", "velocity"),
        physical_indices=(0, 1, 2),
        goal_indices=(0, 1, 2),
        reference_zero_indices=(),
        q_diagonal=(5.0e3, 1.0e5, 1.0e5, 20.0, 20.0),
        success_threshold=0.15,
        latch_threshold=0.30,
    ),
    "cartpole_balance": TaskSpec(
        name="cartpole_balance",
        domain="cartpole",
        task="balance",
        output_keys=("position", "velocity"),
        physical_indices=(0, 1, 2),
        goal_indices=(0, 1, 2),
        reference_zero_indices=(),
        q_diagonal=(5.0e3, 1.0e5, 1.0e5, 20.0, 20.0),
        success_threshold=0.15,
        latch_threshold=0.30,
    ),
    "pendulum_swingup": TaskSpec(
        name="pendulum_swingup",
        domain="pendulum",
        task="swingup",
        output_keys=("orientation", "velocity"),
        physical_indices=(0, 1),
        goal_indices=(0, 1),
        reference_zero_indices=(),
        q_diagonal=(1.0e5, 1.0e5, 50.0),
        success_threshold=0.15,
        latch_threshold=0.15,
    ),
    "finger_turn_easy": TaskSpec(
        name="finger_turn_easy",
        domain="finger",
        task="turn_easy",
        output_keys=("position", "velocity", "touch"),
        physical_indices=(0, 1, 2, 3),
        goal_indices=(2, 3),
        reference_zero_indices=(),
        q_diagonal=(5.0e3, 5.0e3, 1.0e5, 1.0e5, 20.0, 20.0, 20.0, 2.0, 2.0),
        success_threshold=0.07,
        latch_threshold=0.12,
    ),
    "ball_in_cup": TaskSpec(
        name="ball_in_cup",
        domain="ball_in_cup",
        task="catch",
        output_keys=("position", "velocity"),
        physical_indices=(0, 1, 2, 3),
        goal_indices=(2, 3),
        reference_zero_indices=(),
        q_diagonal=(100.0, 100.0, 1.0e5, 1.0e5, 5.0, 5.0, 20.0, 20.0),
        success_threshold=0.025,
        latch_threshold=0.06,
    ),
}


@dataclass(frozen=True)
class PilotConfig:
    t_ini: int = 2
    t_f: int = 10
    offline_episodes: int = 200
    offline_steps: int = 200
    offline_seed_start: int = 2026
    offline_reset_mode: str = "task_default"
    offline_input_mode: str = "uniform_ar"
    offline_input_momentum: float = 0.0
    offline_input_scale: float = 1.0
    offline_local_input_scale: float = 0.3
    offline_input_hold: int = 1
    offline_input_period: float = 100.0
    clusters: int = 48
    contextual_clusters: int = 128
    contextual_architecture: str = "global"
    contextual_modes_per_chart: int = 4
    contextual_support: int = 200
    contextual_goal_pool: int = 4000
    contextual_frame: str = "absolute"
    contextual_future_weight: float = 1.0
    contextual_training_partition: str = "all"
    contextual_t_f: int = 0
    contextual_lam_g: float | None = None
    contextual_lam_du: float | None = None
    contextual_r: float | None = None
    contextual_penalize_initial_move: bool = False
    contextual_velocity_weight_scale: float | None = None
    contextual_cart_position_weight_scale: float | None = None
    contextual_selection_cart_position_weight_scale: float | None = None
    contextual_output_bias_gain: float = 0.0
    transition_support: int = 200
    cluster_seed: int = 11
    contextual_seed: int = 73
    svd_rank: int = 60
    contextual_svd_rank: int = 0
    svd_bound_scale: float = 3.0
    edge_count_threshold: int = 2
    graph_direction: str = "directed"
    waypoint_mode: str = "chart_medoid"
    physical_feature_mode: str = "configuration"
    physical_goal_weight: float = 1.0
    physical_matrix_mode: str = "current"
    reference_mode: str = "graph_routed"
    route_dwell_steps: int = 1
    template_selection_dwell_steps: int = 1
    template_velocity_weight_scale: float = 1.0
    goal_construction_mode: str = "exact"
    control_steps: int = 320
    evaluation_seed_start: int = 501
    evaluation_seeds: int = 10
    lam_g: float = 0.05
    lam_ini: float = 1.0e5
    lam_du: float = 5.0
    penalize_initial_move: bool = False
    r: float = 1.0e-4
    velocity_weight_scale: float = 1.0
    cart_position_weight_scale: float = 1.0
    latch_threshold_override: float | None = None
    trigger_window: int = 15
    trigger_minimum_progress: float = 1.0e-3
    contextual_activation: str = "triggered"
    contextual_success_streak: int = 1
    contextual_entry_threshold: float | None = None
    contextual_exit_threshold: float | None = None
    latched_matrix_mode: str = "normal"
    terminal_support: int = 400
    terminal_svd_rank: int = 0
    terminal_t_f: int = 0
    terminal_lock_steps: int = 0
    terminal_matrix_mode: str = "selected"
    terminal_lam_g: float | None = None
    terminal_lam_du: float | None = None
    terminal_r: float | None = None
    terminal_velocity_weight_scale: float | None = None


def configure_horizons(config: PilotConfig) -> None:
    global T_INI, T_F, L
    T_INI = int(config.t_ini)
    T_F = int(config.t_f)
    L = T_INI + T_F


def controller_q_diagonal(spec: TaskSpec, config: PilotConfig) -> np.ndarray:
    q = np.asarray(spec.q_diagonal, dtype=float).copy()
    if spec.name in {"cartpole_swingup", "cartpole_balance"}:
        q[0] *= config.cart_position_weight_scale
    velocity_start = {
        "point_mass_easy": 2,
        "cartpole_swingup": 3,
        "cartpole_balance": 3,
        "pendulum_swingup": 2,
        "finger_turn_easy": 4,
        "ball_in_cup": 4,
    }[spec.name]
    q[velocity_start:] *= config.velocity_weight_scale
    return q


def make_env(spec: TaskSpec, seed: int):
    return suite.load(spec.domain, spec.task, task_kwargs={"random": int(seed)})


def flatten_output(spec: TaskSpec, observation) -> np.ndarray:
    values = [np.asarray(observation[key], dtype=float).reshape(-1) for key in spec.output_keys]
    output = np.concatenate(values)
    if spec.name == "ball_in_cup":
        position = output[:4]
        velocity = output[4:8]
        output = np.concatenate(
            [
                position[:2],
                position[2:4] - position[:2],
                velocity[:2],
                velocity[2:4] - velocity[:2],
            ]
        )
    if output.size != spec.output_dim:
        raise ValueError(f"{spec.name}: output has {output.size} entries, expected {spec.output_dim}")
    return output


def target_context(spec: TaskSpec, observation) -> np.ndarray:
    if spec.name == "finger_turn_easy":
        return np.asarray(observation["target_position"], dtype=float).reshape(2)
    return np.empty(0)


def goal_error(spec: TaskSpec, output: np.ndarray, context: np.ndarray) -> float:
    y = np.asarray(output, dtype=float)
    if spec.name == "point_mass_easy":
        return float(np.linalg.norm(y[:2]))
    if spec.name in {"cartpole_swingup", "cartpole_balance"}:
        return float(
            np.sqrt(
                0.01 * y[0] ** 2
                + (1.0 - y[1]) ** 2
                + y[2] ** 2
                + 0.01 * y[3] ** 2
                + 0.01 * y[4] ** 2
            )
        )
    if spec.name == "pendulum_swingup":
        return float(
            np.sqrt((1.0 - y[0]) ** 2 + y[1] ** 2 + 0.01 * y[2] ** 2)
        )
    if spec.name == "finger_turn_easy":
        return float(np.linalg.norm(y[2:4] - context))
    if spec.name == "ball_in_cup":
        return float(np.max(np.abs(y[2:4] - np.array([0.0, 0.35]))))
    raise ValueError(spec.name)


def construct_goal(
    spec: TaskSpec,
    initial_output: np.ndarray,
    context: np.ndarray,
    offline_states: np.ndarray,
    offline_contexts: np.ndarray,
    feature_scale: np.ndarray,
    config_goal_mode: str,
) -> tuple[np.ndarray, int]:
    if spec.name == "point_mass_easy":
        return np.zeros(4), -1
    if spec.name in {"cartpole_swingup", "cartpole_balance"}:
        return np.array([0.0, 1.0, 0.0, 0.0, 0.0]), -1
    if spec.name == "pendulum_swingup":
        return np.array([1.0, 0.0, 0.0]), -1
    if spec.name == "finger_turn_easy":
        target_distance = np.linalg.norm(offline_states[:, 2:4] - context[None, :], axis=1)
        source_target_distance = np.linalg.norm(
            np.asarray(offline_contexts.tolist(), dtype=float) - context[None, :], axis=1
        )
        fair = np.flatnonzero(source_target_distance >= 0.06)
        candidates = fair[np.argsort(target_distance[fair], kind="stable")[:400]]
        posture_rows = np.array([0, 1, 2, 3])
        posture_error = (
            offline_states[candidates][:, posture_rows] - initial_output[posture_rows]
        ) / np.maximum(feature_scale[posture_rows], 1e-8)
        selected = int(candidates[np.argmin(np.sum(posture_error**2, axis=1))])
        goal = offline_states[selected].copy()
        if config_goal_mode == "exact":
            goal[2:4] = context
        elif config_goal_mode != "feasible_sample":
            raise ValueError(f"Unknown goal construction mode: {config_goal_mode}")
        goal[4:7] = 0.0
        goal[7:9] = 0.0
        return goal, selected
    if spec.name == "ball_in_cup":
        catch_error = np.max(
            np.abs(offline_states[:, 2:4] - np.array([0.0, 0.35])),
            axis=1,
        )
        candidates = np.argsort(catch_error, kind="stable")[:400]
        configuration_error = (
            offline_states[candidates, :4] - initial_output[None, :4]
        ) / np.maximum(feature_scale[:4], 1e-8)
        selected = int(candidates[np.argmin(np.sum(configuration_error**2, axis=1))])
        goal = offline_states[selected].copy()
        if config_goal_mode == "exact":
            goal[2:4] = np.array([0.0, 0.35])
        elif config_goal_mode != "feasible_sample":
            raise ValueError(f"Unknown goal construction mode: {config_goal_mode}")
        goal[4:] = 0.0
        return goal, selected
    raise ValueError(spec.name)


def library_signature(spec: TaskSpec, config: PilotConfig) -> dict:
    signature = {
        "collector_version": 4,
        "task": spec.name,
        "offline_episodes": config.offline_episodes,
        "offline_steps": config.offline_steps,
        "offline_seed_start": config.offline_seed_start,
        "offline_reset_mode": config.offline_reset_mode,
        "offline_input_mode": config.offline_input_mode,
        "offline_input_momentum": config.offline_input_momentum,
        "offline_input_scale": config.offline_input_scale,
        "offline_local_input_scale": config.offline_local_input_scale,
        "offline_input_hold": config.offline_input_hold,
        "offline_input_period": config.offline_input_period,
        "stabilization_threshold": spec.success_threshold,
    }
    if spec.name in {"cartpole_swingup", "cartpole_balance"}:
        signature["goal_error_version"] = 2
    if spec.name == "ball_in_cup":
        signature["goal_error_version"] = 2
    return signature


def collect_exploration_library(spec: TaskSpec, config: PilotConfig, path: Path) -> dict:
    accepted_u = []
    accepted_y = []
    accepted_context = []
    accepted_roles = []
    accepted_seeds = []
    rejected_stabilized = []
    candidate_seed = config.offline_seed_start
    while len(accepted_u) < config.offline_episodes:
        env = make_env(spec, candidate_seed)
        step = env.reset()
        mixed_local_episode = False
        if config.offline_reset_mode in {"mixed", "mixed_near_goal"}:
            mixed_local_episode = (
                (candidate_seed - config.offline_seed_start) % 2 == 1
            )
        elif config.offline_reset_mode == "blocked_near_goal":
            mixed_local_episode = len(accepted_u) >= config.offline_episodes // 2
        near_goal_reset = (
            config.offline_reset_mode in {"mixed_near_goal", "blocked_near_goal"}
            and mixed_local_episode
        )
        use_uniform_reset = (
            config.offline_reset_mode == "state_uniform"
            or (mixed_local_episode and not near_goal_reset)
        )
        if near_goal_reset:
            if spec.name != "cartpole_swingup":
                raise ValueError("mixed_near_goal reset is defined for Cartpole swing-up")
            reset_rng = np.random.default_rng(candidate_seed + 200_000)
            env.physics.named.data.qpos["slider"] = reset_rng.uniform(-0.1, 0.1)
            env.physics.named.data.qpos["hinge_1"] = reset_rng.uniform(-0.034, 0.034)
            env.physics.data.qvel[:] = 0.01 * reset_rng.standard_normal(
                env.physics.model.nv
            )
            env.physics.after_reset()
        elif use_uniform_reset:
            reset_rng = np.random.default_rng(candidate_seed + 200_000)
            if spec.name == "cartpole_swingup":
                env.physics.data.qpos[0] = reset_rng.uniform(-0.5, 0.5)
                env.physics.data.qpos[1:] = reset_rng.uniform(
                    -np.pi, np.pi, size=env.physics.data.qpos[1:].shape
                )
                env.physics.data.qvel[:] = reset_rng.uniform(
                    -1.0, 1.0, size=env.physics.data.qvel.shape
                )
            elif spec.name == "pendulum_swingup":
                env.physics.data.qpos[:] = reset_rng.uniform(-np.pi, np.pi)
                env.physics.data.qvel[:] = reset_rng.uniform(-2.0, 2.0)
            else:
                raise ValueError(
                    f"state_uniform reset is not defined for {spec.name}"
                )
            env.physics.after_reset()
        elif config.offline_reset_mode not in {
            "task_default",
            "mixed",
            "mixed_near_goal",
            "blocked_near_goal",
        }:
            raise ValueError(f"Unknown offline reset mode: {config.offline_reset_mode}")
        context = target_context(spec, step.observation)
        action_spec = env.action_spec()
        rng = np.random.default_rng(candidate_seed + 100_000)
        action = np.zeros(action_spec.shape, dtype=float)
        sine_amplitude = rng.uniform(0.7, 1.0, size=action_spec.shape)
        sine_phase = rng.uniform(0.0, 2.0 * np.pi, size=action_spec.shape)
        sine_period = config.offline_input_period * rng.uniform(
            0.85, 1.15, size=action_spec.shape
        )
        actions, outputs, errors = [], [], []
        for time_index in range(config.offline_steps):
            input_mode = config.offline_input_mode
            if input_mode == "mixed":
                input_mode = "uniform_ar" if mixed_local_episode else "square"
            if input_mode == "uniform_ar":
                if time_index % config.offline_input_hold == 0:
                    input_scale = (
                        config.offline_local_input_scale
                        if config.offline_input_mode == "mixed"
                        else config.offline_input_scale
                    )
                    action = np.clip(
                        config.offline_input_momentum * action
                        + input_scale
                        * rng.uniform(action_spec.minimum, action_spec.maximum),
                        action_spec.minimum,
                        action_spec.maximum,
                    )
            elif input_mode in {"sine", "square"}:
                wave = np.sin(2.0 * np.pi * time_index / sine_period + sine_phase)
                if input_mode == "square":
                    wave = np.sign(wave)
                action = np.clip(
                    config.offline_input_scale * sine_amplitude * wave,
                    action_spec.minimum,
                    action_spec.maximum,
                )
            else:
                raise ValueError(f"Unknown offline input mode: {config.offline_input_mode}")
            step = env.step(action)
            output = flatten_output(spec, step.observation)
            actions.append(action.copy())
            outputs.append(output)
            errors.append(goal_error(spec, output, context))
        env.close()
        errors = np.asarray(errors)
        stabilized = bool(
            np.mean(errors[-30:]) < spec.success_threshold
            and np.mean(errors[-100:]) < spec.success_threshold
        )
        if stabilized:
            rejected_stabilized.append(candidate_seed)
        else:
            accepted_u.append(np.asarray(actions))
            accepted_y.append(np.asarray(outputs))
            accepted_context.append(context)
            accepted_roles.append("local" if mixed_local_episode else "global")
            accepted_seeds.append(candidate_seed)
        candidate_seed += 1

    U = np.asarray(accepted_u)
    Y = np.asarray(accepted_y)
    context_array = np.asarray(accepted_context, dtype=object)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        U=U,
        Y=Y,
        contexts=context_array,
        roles=np.asarray(accepted_roles),
        accepted_seeds=np.asarray(accepted_seeds),
        rejected_stabilized=np.asarray(rejected_stabilized),
        task=np.array(spec.name),
        library_signature=np.array(json.dumps(library_signature(spec, config))),
    )
    return {
        "U": U,
        "Y": Y,
        "contexts": context_array,
        "roles": np.asarray(accepted_roles),
        "accepted_seeds": accepted_seeds,
        "rejected_stabilized": rejected_stabilized,
    }


def load_or_collect_library(spec: TaskSpec, config: PilotConfig, path: Path) -> dict:
    if not path.exists():
        return collect_exploration_library(spec, config, path)
    with np.load(path, allow_pickle=True) as data:
        saved_signature = (
            json.loads(str(data["library_signature"].item()))
            if "library_signature" in data.files
            else None
        )
        if saved_signature != library_signature(spec, config):
            return collect_exploration_library(spec, config, path)
        return {
            "U": np.asarray(data["U"]),
            "Y": np.asarray(data["Y"]),
            "contexts": np.asarray(data["contexts"], dtype=object),
            "roles": (
                np.asarray(data["roles"]).astype(str)
                if "roles" in data.files
                else np.full(data["U"].shape[0], "global")
            ),
            "accepted_seeds": np.asarray(data["accepted_seeds"]).astype(int).tolist(),
            "rejected_stabilized": np.asarray(data["rejected_stabilized"]).astype(int).tolist(),
        }


def build_hankel(
    library: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    U, Y = library["U"], library["Y"]
    columns, current_states, future_states, episode_ids = [], [], [], []
    windows_per_episode = U.shape[1] - L + 1
    for episode in range(U.shape[0]):
        for start in range(windows_per_episode):
            u_window = U[episode, start : start + L]
            y_window = Y[episode, start : start + L]
            columns.append(
                np.concatenate(
                    [
                        u_window[:T_INI].reshape(-1),
                        u_window[T_INI:].reshape(-1),
                        y_window[:T_INI].reshape(-1),
                        y_window[T_INI:].reshape(-1),
                    ]
                )
            )
            current_states.append(y_window[T_INI - 1])
            future_states.append(y_window[-1])
            episode_ids.append(episode)
    return (
        np.asarray(columns).T,
        np.asarray(current_states),
        np.asarray(future_states),
        np.asarray(episode_ids),
        windows_per_episode,
    )


def offline_goal_errors(spec: TaskSpec, library: dict) -> np.ndarray:
    Y = np.asarray(library["Y"])
    if spec.name == "point_mass_easy":
        return np.linalg.norm(Y[..., :2], axis=-1)
    if spec.name in {"cartpole_swingup", "cartpole_balance"}:
        return np.sqrt(
            0.01 * Y[..., 0] ** 2
            + (1.0 - Y[..., 1]) ** 2
            + Y[..., 2] ** 2
            + 0.01 * Y[..., 3] ** 2
            + 0.01 * Y[..., 4] ** 2
        )
    if spec.name == "pendulum_swingup":
        return np.sqrt(
            (1.0 - Y[..., 0]) ** 2
            + Y[..., 1] ** 2
            + 0.01 * Y[..., 2] ** 2
        )
    if spec.name == "finger_turn_easy":
        contexts = np.asarray(library["contexts"].tolist(), dtype=float)
        return np.linalg.norm(Y[..., 2:4] - contexts[:, None, :], axis=-1)
    if spec.name == "ball_in_cup":
        return np.max(
            np.abs(Y[..., 2:4] - np.array([0.0, 0.35])), axis=-1
        )
    raise ValueError(spec.name)


def fixed_goal_errors(spec: TaskSpec, outputs: np.ndarray) -> np.ndarray:
    """Vectorized goal error for tasks whose target is fixed by the environment."""
    y = np.asarray(outputs, dtype=float)
    if spec.name == "point_mass_easy":
        return np.linalg.norm(y[..., :2], axis=-1)
    if spec.name in {"cartpole_swingup", "cartpole_balance"}:
        return np.sqrt(
            0.01 * y[..., 0] ** 2
            + (1.0 - y[..., 1]) ** 2
            + y[..., 2] ** 2
            + 0.01 * y[..., 3] ** 2
            + 0.01 * y[..., 4] ** 2
        )
    if spec.name == "pendulum_swingup":
        return np.sqrt(
            (1.0 - y[..., 0]) ** 2
            + y[..., 1] ** 2
            + 0.01 * y[..., 2] ** 2
        )
    if spec.name == "ball_in_cup":
        return np.max(
            np.abs(y[..., 2:4] - np.array([0.0, 0.35])), axis=-1
        )
    raise ValueError(f"{spec.name} has no fixed goal")


def fixed_goal_output(spec: TaskSpec) -> np.ndarray | None:
    if spec.name == "point_mass_easy":
        return np.zeros(4)
    if spec.name in {"cartpole_swingup", "cartpole_balance"}:
        return np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    if spec.name == "pendulum_swingup":
        return np.array([1.0, 0.0, 0.0])
    if spec.name == "ball_in_cup":
        # [cup position, ball-minus-cup position, cup velocity,
        #  ball-minus-cup velocity] in flatten_output coordinates.
        return np.array([0.0, 0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 0.0])
    return None


def fixed_rank_basis(X: np.ndarray, rank: int, bound_scale: float) -> dict:
    center = X.mean(axis=1, keepdims=True)
    actual_rank = min(rank, X.shape[0] - 1, X.shape[1] - 1)
    basis = np.linalg.svd(X - center, full_matrices=False)[0][:, :actual_rank]
    coordinates = basis.T @ (X - center)
    mean = coordinates.mean(axis=1, keepdims=True)
    scale = np.maximum(coordinates.std(axis=1, keepdims=True), 1e-8)
    return {
        "offset": center,
        "basis": basis,
        "rank": int(actual_rank),
        "g_min": mean - bound_scale * scale,
        "g_max": mean + bound_scale * scale,
    }


def truncate_hankel_future(
    H: np.ndarray,
    action_dim: int,
    output_dim: int,
    prediction_horizon: int,
) -> np.ndarray:
    """Keep the common past and the requested prefix of both future blocks."""
    if prediction_horizon == T_F:
        return H
    if not 0 < prediction_horizon < T_F:
        raise ValueError("prediction_horizon must be in [1, t_f]")
    u_p_end = action_dim * T_INI
    u_f_end = action_dim * L
    y_p_end = u_f_end + output_dim * T_INI
    rows = np.concatenate(
        [
            np.arange(u_p_end),
            np.arange(u_p_end, u_p_end + action_dim * prediction_horizon),
            np.arange(u_f_end, y_p_end),
            np.arange(y_p_end, y_p_end + output_dim * prediction_horizon),
        ]
    )
    return H[rows]


def physical_feature_array(
    spec: TaskSpec,
    mode: str,
    outputs: np.ndarray,
    contexts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical-chart features and the ordinary output rows they contain."""
    states = np.atleast_2d(np.asarray(outputs, dtype=float))
    if mode == "configuration":
        rows = np.asarray(spec.physical_indices, dtype=int)
        return states[:, rows], rows
    if mode == "state":
        rows = np.arange(spec.output_dim, dtype=int)
        return states, rows
    if mode in {"target_relative", "target_conditioned"}:
        if spec.name != "finger_turn_easy":
            raise ValueError(f"{mode} physical features are defined for Finger")
        target = (
            np.asarray(contexts.tolist(), dtype=float)
            if contexts.dtype == object
            else np.asarray(contexts, dtype=float)
        )
        target = np.atleast_2d(target)
        # Keep the physical finger/spinner posture and append the task-relevant
        # tip-to-target displacement. The latter makes graph progress invariant
        # to the target angle without discarding contact geometry.
        relative_tip = states[:, 2:4] - target
        rows = np.asarray(spec.physical_indices, dtype=int)
        if mode == "target_relative":
            return np.concatenate([states[:, rows], relative_tip], axis=1), rows
        return np.concatenate(
            [states[:, rows], target, relative_tip], axis=1
        ), rows
    raise ValueError(f"Unknown physical feature mode: {mode}")


def build_physical_atlas(
    spec: TaskSpec,
    config: PilotConfig,
    H: np.ndarray,
    current_states: np.ndarray,
    future_states: np.ndarray,
    episode_ids: np.ndarray,
    column_contexts: np.ndarray,
    physical_training_mask: np.ndarray,
) -> dict:
    raw, rows = physical_feature_array(
        spec, config.physical_feature_mode, current_states, column_contexts
    )
    physical_training_mask = np.asarray(physical_training_mask, dtype=bool)
    feature_mean = raw[physical_training_mask].mean(axis=0)
    feature_scale = np.maximum(raw[physical_training_mask].std(axis=0), 1e-8)
    feature_weights = np.ones(rows.size, dtype=float)
    if config.physical_feature_mode in {"target_relative", "target_conditioned"}:
        feature_weights = np.ones(raw.shape[1], dtype=float)
        feature_weights[-2:] = np.sqrt(config.physical_goal_weight)
    else:
        feature_weights[np.isin(rows, np.asarray(spec.goal_indices, dtype=int))] = np.sqrt(
            config.physical_goal_weight
        )
    features = (raw - feature_mean) / feature_scale * feature_weights[None, :]
    model = MiniBatchKMeans(
        n_clusters=config.clusters,
        random_state=config.cluster_seed,
        n_init=10,
        batch_size=2048,
        max_iter=300,
    ).fit(features[physical_training_mask])
    labels = model.predict(features).astype(int)
    centers = model.cluster_centers_
    medoids = []
    matrices = {}
    counts = np.zeros((config.clusters, config.clusters), dtype=int)
    transition_destinations = {
        (source, destination): []
        for source in range(config.clusters)
        for destination in range(config.clusters)
    }
    for chart in range(config.clusters):
        members = np.flatnonzero((labels == chart) & physical_training_mask)
        distance = np.sum((features[members] - centers[chart]) ** 2, axis=1)
        medoid = current_states[int(members[np.argmin(distance)])].copy()
        medoid[np.asarray(spec.reference_zero_indices, dtype=int)] = 0.0
        medoids.append(medoid)
        matrices[chart] = fixed_rank_basis(
            H[:, members], config.svd_rank, config.svd_bound_scale
        )
    for index in range(labels.size - 1):
        if (
            physical_training_mask[index]
            and physical_training_mask[index + 1]
            and episode_ids[index] == episode_ids[index + 1]
        ):
            counts[labels[index], labels[index + 1]] += 1
            transition_destinations[(labels[index], labels[index + 1])].append(index + 1)
    edge_waypoints = np.empty(
        (config.clusters, config.clusters, spec.output_dim), dtype=float
    )
    edge_reference_sequences = np.empty(
        (config.clusters, config.clusters, T_F, spec.output_dim), dtype=float
    )
    action_dim = (H.shape[0] - spec.output_dim * L) // L
    Y_p, Y_f = partition_output_rows(H, action_dim, spec.output_dim)
    future_sequences = Y_f.T.reshape(-1, T_F, spec.output_dim)
    medoid_array = np.asarray(medoids)
    state_scale = np.maximum(
        current_states[physical_training_mask].std(axis=0), 1e-8
    )
    for source in range(config.clusters):
        for destination in range(config.clusters):
            indices = transition_destinations[(source, destination)]
            if not indices:
                edge_waypoints[source, destination] = medoid_array[destination]
                edge_reference_sequences[source, destination] = np.tile(
                    medoid_array[destination], (T_F, 1)
                )
                continue
            candidates = current_states[np.asarray(indices, dtype=int)]
            center = candidates.mean(axis=0)
            normalized = (candidates - center[None, :]) / state_scale[None, :]
            waypoint = candidates[np.argmin(np.sum(normalized**2, axis=1))].copy()
            waypoint[np.asarray(spec.reference_zero_indices, dtype=int)] = 0.0
            edge_waypoints[source, destination] = waypoint
            source_indices = np.asarray(indices, dtype=int) - 1
            if spec.name == "finger_turn_easy":
                endpoint, _ = physical_feature_array(
                    spec,
                    config.physical_feature_mode,
                    future_states[source_indices],
                    column_contexts[source_indices],
                )
                score = np.sum(
                    (
                        (endpoint - feature_mean[None, :])
                        / feature_scale[None, :]
                        * feature_weights[None, :]
                        - centers[destination][None, :]
                    )
                    ** 2,
                    axis=1,
                )
            else:
                sequence_error = fixed_goal_errors(
                    spec, future_sequences[source_indices]
                )
                score = np.mean(sequence_error, axis=1) + np.min(
                    sequence_error, axis=1
                )
            representative = int(source_indices[np.argmin(score)])
            edge_reference_sequences[source, destination] = future_sequences[
                representative
            ]
    goal_path_sequences = np.tile(medoid_array[:, None, :], (1, T_F, 1))
    goal_path_hit_steps = np.full(config.clusters, -1, dtype=int)
    goal_path_source_indices = np.full(config.clusters, -1, dtype=int)
    if spec.name != "finger_turn_easy":
        state_goal_error = fixed_goal_errors(spec, current_states)
        for episode in np.unique(episode_ids):
            indices = np.flatnonzero(
                (episode_ids == episode) & physical_training_mask
            )
            if indices.size == 0:
                continue
            next_hit = -1
            for index in indices[::-1]:
                if state_goal_error[index] <= spec.latch_threshold:
                    next_hit = int(index)
                if next_hit < 0:
                    continue
                chart = int(labels[index])
                hit_steps = next_hit - int(index)
                previous = goal_path_hit_steps[chart]
                if previous >= 0 and hit_steps >= previous:
                    continue
                goal_path_sequences[chart] = future_sequences[index]
                goal_path_hit_steps[chart] = hit_steps
                goal_path_source_indices[chart] = int(index)
    goal_path_matrices = {}
    goal_path_matrix_support_sizes = {}
    if spec.name != "finger_turn_easy":
        path_q = controller_q_diagonal(spec, config)
        path_q = path_q / np.mean(path_q)
        for chart in range(config.clusters):
            if goal_path_hit_steps[chart] < 0:
                continue
            members = np.flatnonzero((labels == chart) & physical_training_mask)
            path = goal_path_sequences[chart]
            path_distance = np.mean(
                np.sum(
                    (future_sequences[members] - path[None, :, :]) ** 2
                    * path_q[None, None, :],
                    axis=2,
                ),
                axis=1,
            )
            support_count = min(config.transition_support, members.size)
            support = members[np.argsort(path_distance, kind="stable")[:support_count]]
            goal_path_matrices[chart] = fixed_rank_basis(
                H[:, support], config.svd_rank, config.svd_bound_scale
            )
            goal_path_matrix_support_sizes[chart] = int(support.size)
    template_path_sequences = np.tile(medoid_array[:, None, :], (1, T_F, 1))
    template_path_available = np.zeros(config.clusters, dtype=bool)
    template_path_source_indices = np.full(config.clusters, -1, dtype=int)
    template_goal_episode = -1
    if spec.name != "finger_turn_easy":
        state_goal_error = fixed_goal_errors(spec, current_states)
        template_stage_score = state_goal_error**2
        if (
            config.template_selection_dwell_steps > 1
            or config.template_velocity_weight_scale != 1.0
        ):
            fixed_goal = fixed_goal_output(spec)
            if fixed_goal is None:
                raise ValueError("Weighted template selection needs a fixed task goal")
            template_score_config = replace(
                config,
                velocity_weight_scale=config.template_velocity_weight_scale,
            )
            template_q = controller_q_diagonal(spec, template_score_config)
            template_q = template_q / np.mean(template_q)
            template_stage_score = np.sum(
                (current_states - fixed_goal[None, :]) ** 2 * template_q[None, :],
                axis=1,
            )

        def episode_template_score(episode: int) -> float:
            values = template_stage_score[episode_ids == episode]
            dwell = min(config.template_selection_dwell_steps, values.size)
            if dwell <= 1:
                return float(np.min(values))
            rolling = np.convolve(values, np.ones(dwell) / dwell, mode="valid")
            return float(np.min(rolling))

        episode_best = {
            int(episode): episode_template_score(int(episode))
            for episode in np.unique(episode_ids[physical_training_mask])
        }
        template_candidates = list(episode_best)
        if spec.name == "cartpole_swingup":
            downward_starts = []
            for episode in template_candidates:
                first = np.flatnonzero(episode_ids == episode)[0]
                if state_goal_error[first] > 1.5:
                    downward_starts.append(episode)
            if downward_starts:
                template_candidates = downward_starts
        template_goal_episode = min(
            template_candidates, key=lambda episode: episode_best[episode]
        )
        episode_indices = np.flatnonzero(
            (episode_ids == template_goal_episode) & physical_training_mask
        )
        hit_index = int(
            episode_indices[np.argmin(state_goal_error[episode_indices])]
        )
        for index in episode_indices[episode_indices <= hit_index]:
            chart = int(labels[index])
            # Repeated visits to one chart use the latest phase before the
            # successful transient, yielding a monotone time-to-go policy.
            template_path_sequences[chart] = future_sequences[index]
            template_path_available[chart] = True
            template_path_source_indices[chart] = int(index)
    goal_matrix = None
    goal_matrix_support = np.empty(0, dtype=int)
    fixed_goal = fixed_goal_output(spec)
    if fixed_goal is not None:
        past_outputs = Y_p.T.reshape(-1, T_INI, spec.output_dim)
        future_outputs = Y_f.T.reshape(-1, T_F, spec.output_dim)
        terminal_score_config = replace(
            config,
            velocity_weight_scale=(
                config.velocity_weight_scale
                if config.terminal_velocity_weight_scale is None
                else config.terminal_velocity_weight_scale
            ),
        )
        q = controller_q_diagonal(spec, terminal_score_config)
        q = q / np.mean(q)
        past_cost = np.mean(
            np.sum((past_outputs - fixed_goal[None, None, :]) ** 2 * q, axis=2),
            axis=1,
        )
        future_stage_cost = np.sum(
            (future_outputs - fixed_goal[None, None, :]) ** 2 * q,
            axis=2,
        )
        # A terminal controller must remain near the goal for the complete
        # prediction horizon. A minimum-only score admitted fast fly-bys and
        # produced the observed post-arrival drift on swing-up tasks.
        terminal_score = past_cost + np.mean(future_stage_cost, axis=1)
        terminal_score += 0.25 * np.max(future_stage_cost, axis=1)
        support_count = min(config.terminal_support, H.shape[1])
        goal_matrix_support = np.argsort(terminal_score, kind="stable")[:support_count]
        terminal_rank = (
            config.svd_rank
            if config.terminal_svd_rank <= 0
            else config.terminal_svd_rank
        )
        terminal_horizon = T_F if config.terminal_t_f <= 0 else config.terminal_t_f
        if terminal_horizon > T_F:
            raise ValueError("terminal_t_f cannot exceed the main prediction horizon")
        action_dim = (H.shape[0] - spec.output_dim * L) // L
        terminal_hankel = truncate_hankel_future(
            H, action_dim, spec.output_dim, terminal_horizon
        )
        goal_matrix = fixed_rank_basis(
            terminal_hankel[:, goal_matrix_support],
            terminal_rank,
            config.svd_bound_scale,
        )
        goal_matrix["prediction_horizon"] = int(terminal_horizon)
    edge_matrices = {}
    edge_support_sizes = {}
    if config.physical_matrix_mode in {"edge_conditioned", "edge_union"}:
        successor_features = np.full_like(features, np.nan)
        successor_labels = np.full(labels.shape, -1, dtype=int)
        valid_successor = episode_ids[:-1] == episode_ids[1:]
        valid_indices = np.flatnonzero(valid_successor)
        successor_features[valid_indices] = features[valid_indices + 1]
        successor_labels[valid_indices] = labels[valid_indices + 1]
        current_center_error = np.sum((features - centers[labels]) ** 2, axis=1)
        past_input_norm = np.sum(H[: action_dim * T_INI] ** 2, axis=0)
        for source in range(config.clusters):
            members = np.flatnonzero((labels == source) & physical_training_mask)
            destinations = np.flatnonzero(
                counts[source] >= config.edge_count_threshold
            )
            for destination in destinations:
                destination = int(destination)
                if destination == source:
                    continue
                if config.physical_matrix_mode == "edge_union":
                    destination_members = np.flatnonzero(
                        (labels == destination) & physical_training_mask
                    )
                    per_chart = max(1, config.transition_support // 2)
                    source_order = members[
                        np.argsort(current_center_error[members], kind="stable")
                    ]
                    destination_order = destination_members[
                        np.argsort(
                            current_center_error[destination_members], kind="stable"
                        )
                    ]
                    support = np.unique(
                        np.concatenate(
                            [source_order[:per_chart], destination_order[:per_chart]]
                        )
                    )
                    edge_matrices[(source, destination)] = fixed_rank_basis(
                        H[:, support], config.svd_rank, config.svd_bound_scale
                    )
                    edge_support_sizes[(source, destination)] = int(support.size)
                    continue
                candidates = members[successor_labels[members] >= 0]
                successor_error = np.sum(
                    (successor_features[candidates] - centers[destination]) ** 2,
                    axis=1,
                )
                score = successor_error + 0.1 * current_center_error[candidates]
                support_count = min(config.transition_support, members.size)
                directional_order = candidates[np.argsort(score, kind="stable")]
                exact = directional_order[
                    successor_labels[directional_order] == destination
                ]
                history_order = members[
                    np.argsort(past_input_norm[members], kind="stable")
                ]
                selected = []
                seen = set()
                for index in exact:
                    selected.append(int(index))
                    seen.add(int(index))
                    if len(selected) == support_count:
                        break
                directional_count = max(
                    len(selected), (support_count + 1) // 2
                )
                for index in directional_order:
                    if int(index) not in seen:
                        selected.append(int(index))
                        seen.add(int(index))
                    if len(selected) == directional_count:
                        break
                for index in history_order:
                    if int(index) not in seen:
                        selected.append(int(index))
                        seen.add(int(index))
                    if len(selected) == support_count:
                        break
                support = np.asarray(selected, dtype=int)
                edge_matrices[(source, destination)] = fixed_rank_basis(
                    H[:, support], config.svd_rank, config.svd_bound_scale
                )
                edge_support_sizes[(source, destination)] = int(support.size)
    return {
        "labels": labels,
        "physical_training_mask": physical_training_mask,
        "centers": centers,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "feature_weights": feature_weights,
        "physical_rows": rows,
        "physical_feature_mode": config.physical_feature_mode,
        "medoid_states": medoid_array,
        "edge_waypoints": edge_waypoints,
        "edge_reference_sequences": edge_reference_sequences,
        "goal_path_sequences": goal_path_sequences,
        "goal_path_hit_steps": goal_path_hit_steps,
        "goal_path_source_indices": goal_path_source_indices,
        "goal_path_matrices": goal_path_matrices,
        "goal_path_matrix_support_sizes": goal_path_matrix_support_sizes,
        "template_path_sequences": template_path_sequences,
        "template_path_available": template_path_available,
        "template_path_source_indices": template_path_source_indices,
        "template_goal_episode": int(template_goal_episode),
        "goal_matrix": goal_matrix,
        "goal_matrix_support": goal_matrix_support,
        "transition_counts": counts,
        "matrices": matrices,
        "edge_matrices": edge_matrices,
        "edge_support_sizes": edge_support_sizes,
    }


def partition_output_rows(H: np.ndarray, action_dim: int, output_dim: int):
    start = action_dim * L
    Y_p = H[start : start + output_dim * T_INI]
    Y_f = H[start + output_dim * T_INI :]
    return Y_p, Y_f


def fixed_goal_future_score(spec: TaskSpec, future: np.ndarray) -> np.ndarray:
    if spec.name == "point_mass_easy":
        error = np.linalg.norm(future[:, :2], axis=1)
    elif spec.name in {"cartpole_swingup", "cartpole_balance"}:
        error = np.sqrt(
            0.01 * future[:, 0] ** 2
            + (1.0 - future[:, 1]) ** 2
            + future[:, 2] ** 2
            + 0.01 * future[:, 3] ** 2
            + 0.01 * future[:, 4] ** 2
        )
    elif spec.name == "pendulum_swingup":
        error = np.sqrt(
            (1.0 - future[:, 0]) ** 2
            + future[:, 1] ** 2
            + 0.01 * future[:, 2] ** 2
        )
    elif spec.name == "ball_in_cup":
        error = np.max(
            np.abs(
                np.moveaxis(future[:, 2:4], 1, 2)
                - np.array([0.0, 0.35])[None, None, :]
            ),
            axis=2,
        )
    else:
        raise ValueError(
            f"Goal-enriched contextual prototypes need a fixed task goal: {spec.name}"
        )
    return np.min(error, axis=0)


def build_contextual_atlas(
    spec: TaskSpec,
    config: PilotConfig,
    H: np.ndarray,
    action_dim: int,
    physical_atlas: dict,
    column_contexts: np.ndarray,
    column_roles: np.ndarray,
) -> dict:
    Y_p, Y_f = partition_output_rows(H, action_dim, spec.output_dim)
    past_raw = Y_p.reshape(T_INI, spec.output_dim, H.shape[1]).copy()
    future_raw = Y_f.reshape(T_F, spec.output_dim, H.shape[1]).copy()
    if config.contextual_frame == "goal_relative":
        if spec.name != "finger_turn_easy":
            raise ValueError("goal_relative contextual frame is currently defined for Finger")
        contexts = np.asarray(column_contexts.tolist(), dtype=float)
        past_raw[:, 2:4] -= contexts.T[None, :, :]
        future_raw[:, 2:4] -= contexts.T[None, :, :]
    elif config.contextual_frame != "absolute":
        raise ValueError(f"Unknown contextual frame: {config.contextual_frame}")
    all_y = np.vstack([past_raw, future_raw])
    output_mean = all_y.mean(axis=(0, 2))
    output_scale = np.maximum(all_y.std(axis=(0, 2)), 1e-8)
    past = (
        past_raw - output_mean[None, :, None]
    ) / output_scale[None, :, None]
    future = (
        future_raw - output_mean[None, :, None]
    ) / output_scale[None, :, None]
    contextual_score_config = replace(
        config,
        velocity_weight_scale=(
            config.velocity_weight_scale
            if config.contextual_velocity_weight_scale is None
            else config.contextual_velocity_weight_scale
        ),
        cart_position_weight_scale=(
            (
                config.cart_position_weight_scale
                if config.contextual_cart_position_weight_scale is None
                else config.contextual_cart_position_weight_scale
            )
            if config.contextual_selection_cart_position_weight_scale is None
            else config.contextual_selection_cart_position_weight_scale
        ),
    )
    q = controller_q_diagonal(spec, contextual_score_config)
    q = q / np.mean(q)
    sqrt_q = np.sqrt(q)
    sqrt_future_weight = np.sqrt(config.contextual_future_weight)
    embedding = np.vstack(
        [
            past.reshape(T_INI * spec.output_dim, H.shape[1]),
            (
                future * sqrt_q[None, :, None] * sqrt_future_weight
            ).reshape(T_F * spec.output_dim, H.shape[1]),
        ]
    ).T
    if config.contextual_training_partition == "all":
        training_indices = np.arange(H.shape[1])
    elif config.contextual_training_partition == "local":
        training_indices = np.flatnonzero(np.asarray(column_roles) == "local")
        if training_indices.size == 0:
            raise ValueError("No local columns are available for contextual training")
    else:
        raise ValueError(
            "contextual_training_partition must be 'all' or 'local'"
        )
    prototype_centers = []
    prototype_charts = []
    prototype_candidates = []
    if config.contextual_architecture == "global":
        model = MiniBatchKMeans(
            n_clusters=config.contextual_clusters,
            random_state=config.contextual_seed,
            n_init=5,
            batch_size=2048,
            max_iter=300,
        ).fit(embedding[training_indices])
        for center in model.cluster_centers_:
            prototype_centers.append(center)
            prototype_charts.append(-1)
            prototype_candidates.append(training_indices)
    elif config.contextual_architecture == "goal_enriched":
        global_count = config.contextual_clusters // 2
        goal_count = config.contextual_clusters - global_count
        global_model = MiniBatchKMeans(
            n_clusters=global_count,
            random_state=config.contextual_seed,
            n_init=5,
            batch_size=2048,
            max_iter=300,
        ).fit(embedding[training_indices])
        future_raw = Y_f.reshape(T_F, spec.output_dim, H.shape[1])
        goal_score = fixed_goal_future_score(spec, future_raw)
        pool_count = min(config.contextual_goal_pool, training_indices.size)
        goal_pool = training_indices[
            np.argsort(goal_score[training_indices], kind="stable")[:pool_count]
        ]
        goal_model = MiniBatchKMeans(
            n_clusters=min(goal_count, goal_pool.size),
            random_state=config.contextual_seed + 1,
            n_init=5,
            batch_size=min(2048, goal_pool.size),
            max_iter=300,
        ).fit(embedding[goal_pool])
        for center in global_model.cluster_centers_:
            prototype_centers.append(center)
            prototype_charts.append(-1)
            prototype_candidates.append(training_indices)
        for center in goal_model.cluster_centers_:
            prototype_centers.append(center)
            prototype_charts.append(-1)
            prototype_candidates.append(training_indices)
    elif config.contextual_architecture == "hierarchical":
        labels = np.asarray(physical_atlas["labels"], dtype=int)
        for chart in range(config.clusters):
            members = np.intersect1d(
                np.flatnonzero(labels == chart), training_indices, assume_unique=True
            )
            if members.size == 0:
                continue
            modes = min(config.contextual_modes_per_chart, members.size)
            model = MiniBatchKMeans(
                n_clusters=modes,
                random_state=config.contextual_seed + chart,
                n_init=5,
                batch_size=min(2048, members.size),
                max_iter=300,
            ).fit(embedding[members])
            for center in model.cluster_centers_:
                prototype_centers.append(center)
                prototype_charts.append(chart)
                prototype_candidates.append(members)
    else:
        raise ValueError(
            f"Unknown contextual architecture: {config.contextual_architecture}"
        )
    centers = np.asarray(prototype_centers)
    matrices = {}
    support_indices = []
    contextual_rank = (
        config.svd_rank
        if config.contextual_svd_rank <= 0
        else config.contextual_svd_rank
    )
    contextual_horizon = T_F if config.contextual_t_f <= 0 else config.contextual_t_f
    contextual_hankel = truncate_hankel_future(
        H, action_dim, spec.output_dim, contextual_horizon
    )
    for prototype, (center, candidates) in enumerate(
        zip(centers, prototype_candidates)
    ):
        distance = np.sum((embedding[candidates] - center) ** 2, axis=1)
        support_count = min(config.contextual_support, candidates.size)
        local = np.argpartition(distance, support_count - 1)[:support_count]
        local = local[np.argsort(distance[local], kind="stable")]
        support = candidates[local]
        support_indices.append(support)
        matrices[prototype] = fixed_rank_basis(
            contextual_hankel[:, support], contextual_rank, config.svd_bound_scale
        )
        matrices[prototype]["prediction_horizon"] = int(contextual_horizon)
    return {
        "centers": centers,
        "prototype_charts": np.asarray(prototype_charts, dtype=int),
        "matrices": matrices,
        "support_indices": np.asarray(support_indices, dtype=object),
        "output_mean": output_mean,
        "output_scale": output_scale,
        "sqrt_q": sqrt_q,
        "sqrt_future_weight": float(sqrt_future_weight),
        "contextual_frame": config.contextual_frame,
        "training_partition": config.contextual_training_partition,
        "training_columns": int(training_indices.size),
        "prediction_horizon": int(contextual_horizon),
    }


def online_feature(
    atlas: dict, output: np.ndarray, context: np.ndarray
) -> np.ndarray:
    rows = atlas["physical_rows"]
    if atlas["physical_feature_mode"] in {"target_relative", "target_conditioned"}:
        pieces = [np.asarray(output, dtype=float)[rows]]
        if atlas["physical_feature_mode"] == "target_conditioned":
            pieces.append(context)
        pieces.append(output[2:4] - context)
        raw = np.concatenate(pieces)
    else:
        raw = np.asarray(output, dtype=float)[rows]
    return (
        (raw - atlas["feature_mean"])
        / atlas["feature_scale"]
        * atlas["feature_weights"]
    )


def localize(atlas: dict, output: np.ndarray, context: np.ndarray) -> int:
    feature = online_feature(atlas, output, context)
    return int(np.argmin(np.sum((atlas["centers"] - feature[None, :]) ** 2, axis=1)))


def shortest_route(
    atlas: dict,
    start: int,
    goal: int,
    threshold: int,
    graph_direction: str,
) -> list[int]:
    if start == goal:
        return [start]
    counts = atlas["transition_counts"]
    if graph_direction == "undirected":
        counts = np.maximum(counts, counts.T)
    elif graph_direction != "directed":
        raise ValueError(f"Unknown graph direction: {graph_direction}")
    queue = [(0.0, start)]
    cost = {start: 0.0}
    previous = {}
    while queue:
        distance, node = heapq.heappop(queue)
        if node == goal:
            break
        if distance != cost[node]:
            continue
        destinations = np.flatnonzero(counts[node] >= threshold)
        for destination in destinations:
            if destination == node:
                continue
            center_distance = float(
                np.sqrt(np.mean((atlas["centers"][node] - atlas["centers"][destination]) ** 2))
            )
            edge = center_distance / np.sqrt(float(counts[node, destination]))
            candidate = distance + edge
            if candidate < cost.get(int(destination), np.inf):
                cost[int(destination)] = candidate
                previous[int(destination)] = node
                heapq.heappush(queue, (candidate, int(destination)))
    if goal not in previous:
        return [start]
    route = [goal]
    while route[-1] != start:
        route.append(previous[route[-1]])
    return route[::-1]


def route_reference(
    spec: TaskSpec,
    config: PilotConfig,
    atlas: dict,
    route: list[int],
    goal_chart: int,
    goal: np.ndarray,
    latched: bool,
) -> np.ndarray:
    """Build the output reference seen by the finite-horizon DeePC problem."""
    current_chart = route[0]
    if config.reference_mode == "final":
        return np.tile(goal, T_F).reshape(-1, 1)
    if latched or current_chart == goal_chart:
        return np.tile(goal, T_F).reshape(-1, 1)
    if config.reference_mode == "graph_routed":
        next_chart = route[1] if len(route) > 1 else current_chart
        if config.waypoint_mode == "chart_medoid":
            active_reference = atlas["medoid_states"][next_chart]
        elif config.waypoint_mode == "edge_transition":
            active_reference = atlas["edge_waypoints"][current_chart, next_chart]
        else:
            raise ValueError(f"Unknown waypoint mode: {config.waypoint_mode}")
        return np.tile(active_reference, T_F).reshape(-1, 1)
    if config.reference_mode == "edge_sequence":
        next_chart = route[1] if len(route) > 1 else current_chart
        return atlas["edge_reference_sequences"][current_chart, next_chart].reshape(
            -1, 1
        )
    if config.reference_mode == "template_path":
        if atlas["template_path_available"][current_chart]:
            return atlas["template_path_sequences"][current_chart].reshape(-1, 1)
        # Charts not visited by the template return to the ordinary graph route.
    elif config.reference_mode == "goal_path":
        if atlas["goal_path_hit_steps"][current_chart] >= 0:
            return atlas["goal_path_sequences"][current_chart].reshape(-1, 1)
        # A chart without a recorded goal-reaching suffix falls back to routing.
    elif config.reference_mode != "route_sequence":
        raise ValueError(f"Unknown reference mode: {config.reference_mode}")
    if config.route_dwell_steps < 1:
        raise ValueError("route_dwell_steps must be positive")

    references = []
    for horizon_step in range(T_F):
        route_step = horizon_step // config.route_dwell_steps
        source_position = min(route_step, len(route) - 1)
        destination_position = min(route_step + 1, len(route) - 1)
        source = route[source_position]
        destination = route[destination_position]
        if source == destination:
            waypoint = (
                goal
                if destination == goal_chart
                else atlas["medoid_states"][destination]
            )
        elif config.waypoint_mode == "chart_medoid":
            waypoint = atlas["medoid_states"][destination]
        elif config.waypoint_mode == "edge_transition":
            waypoint = atlas["edge_waypoints"][source, destination]
        else:
            raise ValueError(f"Unknown waypoint mode: {config.waypoint_mode}")
        references.append(waypoint)
    return np.asarray(references).reshape(-1, 1)


def select_contextual_matrix(
    contextual: dict,
    y_ini: np.ndarray,
    reference: np.ndarray,
    output_dim: int,
    context: np.ndarray,
    candidate_charts: tuple[int, ...] | None = None,
) -> tuple[int, dict]:
    mean, scale = contextual["output_mean"], contextual["output_scale"]
    past_raw = y_ini.reshape(T_INI, output_dim).copy()
    future_raw = reference.reshape(T_F, output_dim).copy()
    if contextual["contextual_frame"] == "goal_relative":
        past_raw[:, 2:4] -= context[None, :]
        future_raw[:, 2:4] -= context[None, :]
    past = (past_raw - mean[None, :]) / scale[None, :]
    future = (future_raw - mean[None, :]) / scale[None, :]
    query = np.concatenate(
        [
            past.reshape(-1),
            (
                future
                * contextual["sqrt_q"][None, :]
                * contextual["sqrt_future_weight"]
            ).reshape(-1),
        ]
    )
    candidates = np.arange(contextual["centers"].shape[0])
    if candidate_charts is not None and np.any(contextual["prototype_charts"] >= 0):
        candidates = np.flatnonzero(
            np.isin(contextual["prototype_charts"], np.asarray(candidate_charts))
        )
    distance = np.sum((contextual["centers"][candidates] - query[None, :]) ** 2, axis=1)
    prototype = int(candidates[np.argmin(distance)])
    return prototype, contextual["matrices"][prototype]


class DeePCController:
    def __init__(
        self,
        spec: TaskSpec,
        action_dim: int,
        config: PilotConfig,
        prediction_horizon: int | None = None,
    ):
        self.spec = spec
        self.action_dim = action_dim
        self.config = config
        self.prediction_horizon = T_F if prediction_horizon is None else prediction_horizon

    def solve(self, matrix: dict, u_ini: np.ndarray, y_ini: np.ndarray, reference: np.ndarray):
        m, p = self.action_dim, self.spec.output_dim
        horizon = self.prediction_horizon
        u = cp.Variable((m * horizon, 1))
        y = cp.Variable((p * horizon, 1))
        g = cp.Variable((matrix["basis"].shape[1], 1))
        sigma = cp.Variable((p * T_INI, 1))
        behavior = cp.vstack([u_ini, u, y_ini + sigma, y])
        Q = np.diag(controller_q_diagonal(self.spec, self.config))
        cost = self.config.lam_ini * cp.sum_squares(sigma)
        cost += self.config.lam_g * cp.sum_squares(g)
        for step in range(horizon):
            yi = y[step * p : (step + 1) * p]
            ri = reference[step * p : (step + 1) * p]
            ui = u[step * m : (step + 1) * m]
            cost += cp.quad_form(yi - ri, Q)
            cost += self.config.r * cp.sum_squares(ui)
        for step in range(horizon - 1):
            cost += self.config.lam_du * cp.sum_squares(
                u[(step + 1) * m : (step + 2) * m] - u[step * m : (step + 1) * m]
            )
        if self.config.penalize_initial_move:
            cost += self.config.lam_du * cp.sum_squares(
                u[:m] - u_ini[-m:]
            )
        constraints = [
            behavior == matrix["offset"] + matrix["basis"] @ g,
            g >= matrix["g_min"],
            g <= matrix["g_max"],
            u >= -1.0,
            u <= 1.0,
        ]
        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(solver=cp.CLARABEL, verbose=False, max_iter=10_000)
        except Exception:
            return None, None, "crashed"
        if problem.status not in ("optimal", "optimal_inaccurate"):
            return None, None, str(problem.status)
        return np.asarray(u.value), np.asarray(y.value), str(problem.status)


def run_rollout(
    spec: TaskSpec,
    config: PilotConfig,
    atlas: dict,
    contextual: dict,
    offline_states: np.ndarray,
    offline_contexts: np.ndarray,
    seed: int,
    method: str,
) -> dict:
    env = make_env(spec, seed)
    step = env.reset()
    initial = flatten_output(spec, step.observation)
    context = target_context(spec, step.observation)
    goal, goal_source = construct_goal(
        spec,
        initial,
        context,
        offline_states,
        offline_contexts,
        atlas["feature_scale"],
        config.goal_construction_mode,
    )
    goal_source_target_distance = (
        None
        if goal_source < 0 or spec.name != "finger_turn_easy"
        else float(
            np.linalg.norm(
                np.asarray(offline_contexts[goal_source], dtype=float) - context
            )
        )
    )
    goal_chart = localize(atlas, goal, context)
    action_dim = int(np.prod(env.action_spec().shape))
    controller = DeePCController(spec, action_dim, config)
    terminal_config = replace(
        config,
        lam_g=config.lam_g if config.terminal_lam_g is None else config.terminal_lam_g,
        lam_du=(
            config.lam_du if config.terminal_lam_du is None else config.terminal_lam_du
        ),
        r=config.r if config.terminal_r is None else config.terminal_r,
        velocity_weight_scale=(
            config.velocity_weight_scale
            if config.terminal_velocity_weight_scale is None
            else config.terminal_velocity_weight_scale
        ),
    )
    terminal_horizon = T_F if config.terminal_t_f <= 0 else config.terminal_t_f
    terminal_controller = DeePCController(
        spec, action_dim, terminal_config, prediction_horizon=terminal_horizon
    )
    contextual_config = replace(
        config,
        lam_g=(
            config.lam_g
            if config.contextual_lam_g is None
            else config.contextual_lam_g
        ),
        lam_du=(
            config.lam_du
            if config.contextual_lam_du is None
            else config.contextual_lam_du
        ),
        r=config.r if config.contextual_r is None else config.contextual_r,
        penalize_initial_move=config.contextual_penalize_initial_move,
        velocity_weight_scale=(
            config.velocity_weight_scale
            if config.contextual_velocity_weight_scale is None
            else config.contextual_velocity_weight_scale
        ),
        cart_position_weight_scale=(
            config.cart_position_weight_scale
            if config.contextual_cart_position_weight_scale is None
            else config.contextual_cart_position_weight_scale
        ),
    )
    contextual_horizon = (
        T_F if contextual is None else int(contextual["prediction_horizon"])
    )
    contextual_controller = DeePCController(
        spec, action_dim, contextual_config, prediction_horizon=contextual_horizon
    )
    u_history, y_history = [], []
    for _ in range(T_INI):
        action = np.zeros(action_dim)
        step = env.step(action)
        u_history.append(action.copy())
        y_history.append(flatten_output(spec, step.observation))

    errors, actions, charts, requested, statuses = [], [], [], [], []
    contextual_prototypes = []
    rollout_outputs = []
    solve_times, selection_times = [], []
    routing_times, environment_times, control_step_times = [], [], []
    contextual_active = (
        method == "contextual_atlas" and config.contextual_activation == "always"
    )
    trigger_step = None
    contextual_steps = 0
    contextual_output_bias_by_prototype: dict[int, np.ndarray] = {}
    contextual_entry_streak = 0
    contextual_activation_count = int(contextual_active)
    contextual_deactivation_count = 0
    latched = False
    error_history = []
    success_streak = 0
    locked_matrix = None
    locked_matrix_is_contextual = False
    locked_contextual_prototype = None
    terminal_lock_step = None
    latch_threshold = (
        spec.latch_threshold
        if config.latch_threshold_override is None
        else config.latch_threshold_override
    )
    for control_step in range(config.control_steps):
        control_step_start = time.perf_counter()
        current_output = y_history[-1]
        error = goal_error(spec, current_output, context)
        error_history.append(error)
        if error <= latch_threshold:
            latched = True
        success_streak = success_streak + 1 if error <= spec.success_threshold else 0
        entry_threshold = (
            spec.success_threshold
            if config.contextual_entry_threshold is None
            else config.contextual_entry_threshold
        )
        exit_threshold = (
            latch_threshold
            if config.contextual_exit_threshold is None
            else config.contextual_exit_threshold
        )
        contextual_entry_streak = (
            contextual_entry_streak + 1 if error <= entry_threshold else 0
        )
        if (
            method == "contextual_atlas"
            and config.contextual_activation == "hysteretic"
        ):
            if contextual_active and error > exit_threshold:
                contextual_active = False
                contextual_deactivation_count += 1
            elif (
                not contextual_active
                and contextual_entry_streak >= config.contextual_success_streak
            ):
                contextual_active = True
                contextual_activation_count += 1
                if trigger_step is None:
                    trigger_step = control_step
        if method == "contextual_atlas" and not contextual_active:
            if config.contextual_activation == "latched" and latched:
                contextual_active = True
                contextual_activation_count += 1
                trigger_step = control_step
            elif (
                config.contextual_activation == "successful"
                and success_streak >= config.contextual_success_streak
            ):
                contextual_active = True
                contextual_activation_count += 1
                trigger_step = control_step
            elif (
                config.contextual_activation == "triggered"
                and len(error_history) >= config.trigger_window
            ):
                progress = error_history[-config.trigger_window] - error_history[-1]
                if error > spec.success_threshold and progress < config.trigger_minimum_progress:
                    contextual_active = True
                    contextual_activation_count += 1
                    trigger_step = control_step

        routing_start = time.perf_counter()
        current_chart = localize(atlas, current_output, context)
        route = shortest_route(
            atlas,
            current_chart,
            goal_chart,
            config.edge_count_threshold,
            config.graph_direction,
        )
        next_chart = route[1] if len(route) > 1 else route[0]
        reference = route_reference(
            spec, config, atlas, route, goal_chart, goal, latched
        )
        routing_times.append(time.perf_counter() - routing_start)
        u_ini = np.asarray(u_history[-T_INI:]).reshape(action_dim * T_INI, 1)
        y_ini = np.asarray(y_history[-T_INI:]).reshape(spec.output_dim * T_INI, 1)

        selection_start = time.perf_counter()
        using_terminal_matrix = False
        matrix_is_contextual = False
        selected_contextual_prototype = None
        if contextual_active:
            selected_contextual_prototype, matrix = select_contextual_matrix(
                contextual,
                y_ini,
                reference,
                spec.output_dim,
                context,
                candidate_charts=(current_chart,),
            )
            contextual_steps += 1
            matrix_is_contextual = True
        elif latched and config.latched_matrix_mode in {"goal_chart", "goal_focused"}:
            if config.latched_matrix_mode == "goal_focused" and atlas["goal_matrix"] is not None:
                matrix = atlas["goal_matrix"]
                using_terminal_matrix = True
            else:
                matrix = atlas["matrices"][goal_chart]
        elif locked_matrix is not None:
            matrix = locked_matrix
            if locked_matrix_is_contextual:
                contextual_steps += 1
                matrix_is_contextual = True
                selected_contextual_prototype = locked_contextual_prototype
        else:
            if config.physical_matrix_mode == "current":
                matrix = atlas["matrices"][current_chart]
            elif config.physical_matrix_mode == "next":
                matrix = atlas["matrices"][next_chart]
            elif config.physical_matrix_mode in {"edge_conditioned", "edge_union"}:
                matrix = atlas["edge_matrices"].get(
                    (current_chart, next_chart), atlas["matrices"][current_chart]
                )
            elif config.physical_matrix_mode == "goal_path":
                matrix = atlas["goal_path_matrices"].get(
                    current_chart, atlas["matrices"][current_chart]
                )
            else:
                raise ValueError(
                    f"Unknown physical matrix mode: {config.physical_matrix_mode}"
                )
        if (
            locked_matrix is None
            and config.terminal_lock_steps > 0
            and success_streak >= config.terminal_lock_steps
        ):
            if config.terminal_matrix_mode == "selected":
                locked_matrix = matrix
                locked_matrix_is_contextual = contextual_active
                locked_contextual_prototype = selected_contextual_prototype
            elif config.terminal_matrix_mode == "goal_chart":
                locked_matrix = atlas["matrices"][goal_chart]
                locked_matrix_is_contextual = False
                matrix = locked_matrix
            else:
                raise ValueError(
                    f"Unknown terminal matrix mode: {config.terminal_matrix_mode}"
                )
            terminal_lock_step = control_step
        selection_times.append(time.perf_counter() - selection_start)
        solve_start = time.perf_counter()
        matrix_horizon = int(matrix.get("prediction_horizon", T_F))
        if matrix_horizon not in {T_F, terminal_horizon, contextual_horizon}:
            raise ValueError(
                "Selected matrix horizon must match the main, terminal, or "
                "contextual horizon"
            )
        use_terminal_controller = latched and matrix_horizon == terminal_horizon
        if matrix_is_contextual:
            active_controller = contextual_controller
        else:
            active_controller = terminal_controller if use_terminal_controller else controller
        solve_reference = reference[: matrix_horizon * spec.output_dim]
        if matrix_is_contextual and config.contextual_output_bias_gain > 0.0:
            contextual_output_bias = contextual_output_bias_by_prototype.setdefault(
                int(selected_contextual_prototype), np.zeros(spec.output_dim)
            )
            solve_reference = solve_reference - np.tile(
                contextual_output_bias, matrix_horizon
            ).reshape(-1, 1)
        solution, prediction, status = active_controller.solve(
            matrix, u_ini, y_ini, solve_reference
        )
        solve_times.append(time.perf_counter() - solve_start)
        if solution is None:
            action = np.zeros(action_dim)
        else:
            action = np.clip(solution[:action_dim, 0], -1.0, 1.0)
        environment_start = time.perf_counter()
        step = env.step(action)
        environment_times.append(time.perf_counter() - environment_start)
        next_output = flatten_output(spec, step.observation)
        if (
            matrix_is_contextual
            and prediction is not None
            and config.contextual_output_bias_gain > 0.0
        ):
            residual = next_output - prediction[: spec.output_dim, 0]
            gain = config.contextual_output_bias_gain
            contextual_output_bias = (1.0 - gain) * contextual_output_bias + gain * residual
            contextual_output_bias_by_prototype[
                int(selected_contextual_prototype)
            ] = contextual_output_bias
        u_history.append(action.copy())
        y_history.append(next_output)
        errors.append(goal_error(spec, next_output, context))
        actions.append(action.copy())
        rollout_outputs.append(next_output.copy())
        charts.append(current_chart)
        requested.append(next_chart)
        contextual_prototypes.append(
            -1
            if selected_contextual_prototype is None
            else int(selected_contextual_prototype)
        )
        statuses.append(status)
        control_step_times.append(time.perf_counter() - control_step_start)
    env.close()

    errors = np.asarray(errors)
    strict = bool(
        np.mean(errors[-30:]) < spec.success_threshold
        and np.mean(errors[-100:]) < spec.success_threshold
    )
    return {
        "task": spec.name,
        "seed": int(seed),
        "method": method,
        "strict_success": strict,
        "mean_last_30": float(np.mean(errors[-30:])),
        "mean_last_100": float(np.mean(errors[-100:])),
        "minimum_error": float(np.min(errors)),
        "final_error": float(errors[-1]),
        "goal_chart": int(goal_chart),
        "goal_source_column": int(goal_source),
        "goal_source_target_distance": goal_source_target_distance,
        "contextual_triggered": bool(contextual_activation_count),
        "contextual_trigger_step": trigger_step,
        "contextual_activation_count": int(contextual_activation_count),
        "contextual_deactivation_count": int(contextual_deactivation_count),
        "contextual_usage_fraction": float(contextual_steps / config.control_steps),
        "terminal_lock_step": terminal_lock_step,
        "terminal_matrix_locked": terminal_lock_step is not None,
        "all_qps_optimal": bool(set(statuses) <= {"optimal", "optimal_inaccurate"}),
        "qp_status_counts": {
            status: int(statuses.count(status)) for status in sorted(set(statuses))
        },
        "median_solve_ms": float(1000 * np.median(solve_times)),
        "median_selection_ms": float(1000 * np.median(selection_times)),
        "median_routing_ms": float(1000 * np.median(routing_times)),
        "median_environment_ms": float(1000 * np.median(environment_times)),
        "median_control_step_ms": float(1000 * np.median(control_step_times)),
        "errors": errors.tolist(),
        "actions": np.asarray(actions).tolist(),
        "outputs": np.asarray(rollout_outputs).tolist(),
        "charts": charts,
        "requested_charts": requested,
        "contextual_prototypes": contextual_prototypes,
    }


def run_task(
    spec: TaskSpec,
    config: PilotConfig,
    output_root: Path,
    methods: tuple[str, ...] = ("atlas", "contextual_atlas"),
) -> dict:
    configure_horizons(config)
    task_dir = output_root / spec.name
    task_dir.mkdir(parents=True, exist_ok=True)
    library_path = task_dir / "offline_library.npz"
    library_start = time.perf_counter()
    library = load_or_collect_library(spec, config, library_path)
    library_seconds = time.perf_counter() - library_start
    hankel_start = time.perf_counter()
    H, current_states, future_states, episode_ids, windows_per_episode = build_hankel(
        library
    )
    hankel_seconds = time.perf_counter() - hankel_start
    action_dim = library["U"].shape[-1]
    column_roles = np.asarray(library["roles"])[episode_ids]
    physical_training_mask = column_roles != "local"
    physical_start = time.perf_counter()
    atlas = build_physical_atlas(
        spec,
        config,
        H,
        current_states,
        future_states,
        episode_ids,
        library["contexts"][episode_ids],
        physical_training_mask,
    )
    physical_seconds = time.perf_counter() - physical_start
    contextual = None
    contextual_seconds = 0.0
    if "contextual_atlas" in methods:
        contextual_start = time.perf_counter()
        contextual = build_contextual_atlas(
            spec,
            config,
            H,
            action_dim,
            atlas,
            library["contexts"][episode_ids],
            column_roles,
        )
        contextual_seconds = time.perf_counter() - contextual_start
    coverage_errors = offline_goal_errors(spec, library)
    seeds = list(
        range(config.evaluation_seed_start, config.evaluation_seed_start + config.evaluation_seeds)
    )
    results = []
    for seed in seeds:
        for method in methods:
            result = run_rollout(
                spec,
                config,
                atlas,
                contextual,
                current_states,
                library["contexts"][episode_ids],
                seed,
                method,
            )
            results.append(result)
            print(
                f"task={spec.name} seed={seed} method={method} "
                f"strict={result['strict_success']} "
                f"tail100={result['mean_last_100']:.4f}",
                flush=True,
            )
    summary = {
        "task": spec.name,
        "config": asdict(config),
        "task_spec": asdict(spec),
        "offline_audit": {
            "collection_policy": (
                f"reset={config.offline_reset_mode}, {config.offline_input_mode}: "
                f"a={config.offline_input_momentum:g}, "
                f"b={config.offline_input_scale:g}, "
                f"b_local={config.offline_local_input_scale:g}, "
                f"hold={config.offline_input_hold}, "
                f"period={config.offline_input_period:g}"
            ),
            "accepted_episode_count": len(library["accepted_seeds"]),
            "accepted_seed_range": [
                min(library["accepted_seeds"]), max(library["accepted_seeds"])
            ],
            "rejected_stabilized_episode_count": len(library["rejected_stabilized"]),
            "rejected_stabilized_seeds": library["rejected_stabilized"],
            "stabilized_episodes_in_offline_data": 0,
            "online_eval_seed_range": [min(seeds), max(seeds)],
            "online_offline_seed_disjoint": not bool(
                set(seeds) & set(library["accepted_seeds"])
            ),
            "best_instantaneous_offline_error": float(np.min(coverage_errors)),
            "offline_samples_below_success_threshold": int(
                np.sum(coverage_errors < spec.success_threshold)
            ),
            "offline_error_first_percentile": float(np.quantile(coverage_errors, 0.01)),
        },
        "offline_atlas": {
            "hankel_shape": list(H.shape),
            "windows_per_episode": windows_per_episode,
            "chart_count": config.clusters,
            "chart_size_min": int(
                np.bincount(
                    atlas["labels"][atlas["physical_training_mask"]],
                    minlength=config.clusters,
                ).min()
            ),
            "chart_size_median": float(
                np.median(
                    np.bincount(
                        atlas["labels"][atlas["physical_training_mask"]],
                        minlength=config.clusters,
                    )
                )
            ),
            "physical_training_columns": int(np.sum(physical_training_mask)),
            "terminal_only_columns": int(np.sum(~physical_training_mask)),
            "retained_edges": int(
                np.sum(atlas["transition_counts"] >= config.edge_count_threshold)
            ),
            "edge_conditioned_matrices": len(atlas["edge_matrices"]),
            "goal_path_conditioned_matrices": len(atlas["goal_path_matrices"]),
            "charts_with_coherent_goal_paths": int(
                np.sum(atlas["goal_path_hit_steps"] >= 0)
            ),
            "charts_on_single_episode_template": int(
                np.sum(atlas["template_path_available"])
            ),
            "template_goal_episode": int(atlas["template_goal_episode"]),
            "goal_focused_support": int(atlas["goal_matrix_support"].size),
            "goal_focused_rank": (
                None
                if atlas["goal_matrix"] is None
                else int(atlas["goal_matrix"]["rank"])
            ),
            "goal_focused_horizon": (
                None
                if atlas["goal_matrix"] is None
                else int(atlas["goal_matrix"]["prediction_horizon"])
            ),
            "transition_support": config.transition_support,
            "contextual_prototypes": (
                int(contextual["centers"].shape[0]) if contextual is not None else 0
            ),
            "contextual_architecture": (
                config.contextual_architecture if contextual is not None else None
            ),
            "contextual_support": (
                config.contextual_support if contextual is not None else None
            ),
            "contextual_training_partition": (
                contextual["training_partition"] if contextual is not None else None
            ),
            "contextual_training_columns": (
                contextual["training_columns"] if contextual is not None else 0
            ),
            "contextual_horizon": (
                contextual["prediction_horizon"] if contextual is not None else None
            ),
            "rank_cap": config.svd_rank,
            "physical_rank_min": min(
                matrix["rank"] for matrix in atlas["matrices"].values()
            ),
            "physical_rank_median": float(
                np.median([matrix["rank"] for matrix in atlas["matrices"].values()])
            ),
            "physical_rank_max": max(
                matrix["rank"] for matrix in atlas["matrices"].values()
            ),
        },
        "preprocessing_timing_ms": {
            "library_load_or_collection": 1000.0 * library_seconds,
            "hankel_construction": 1000.0 * hankel_seconds,
            "physical_clustering_and_svd": 1000.0 * physical_seconds,
            "contextual_clustering_and_svd": 1000.0 * contextual_seconds,
        },
        "results": results,
    }
    for method in methods:
        rows = [row for row in results if row["method"] == method]
        summary[method] = {
            "successes": sum(row["strict_success"] for row in rows),
            "trials": len(rows),
            "median_last_100": float(np.median([row["mean_last_100"] for row in rows])),
            "median_solve_ms": float(np.median([row["median_solve_ms"] for row in rows])),
            "median_routing_ms": float(
                np.median([row["median_routing_ms"] for row in rows])
            ),
            "median_selection_ms": float(
                np.median([row["median_selection_ms"] for row in rows])
            ),
            "median_environment_ms": float(
                np.median([row["median_environment_ms"] for row in rows])
            ),
            "median_control_step_ms": float(
                np.median([row["median_control_step_ms"] for row in rows])
            ),
            "triggered_rollouts": sum(row["contextual_triggered"] for row in rows),
        }
    (task_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        default=",".join(TASKS),
        help=f"Comma-separated subset of: {','.join(TASKS)}",
    )
    parser.add_argument(
        "--methods",
        default="atlas,contextual_atlas",
        help="Comma-separated subset of: atlas,contextual_atlas",
    )
    parser.add_argument("--t-ini", type=int, default=2)
    parser.add_argument("--t-f", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=Path("experiments/simple_robot_atlas"))
    parser.add_argument("--offline-episodes", type=int, default=200)
    parser.add_argument("--offline-steps", type=int, default=200)
    parser.add_argument(
        "--offline-reset-mode",
        choices=(
            "task_default",
            "state_uniform",
            "mixed",
            "mixed_near_goal",
            "blocked_near_goal",
        ),
        default="task_default",
    )
    parser.add_argument(
        "--offline-input-mode",
        choices=("uniform_ar", "sine", "square", "mixed"),
        default="uniform_ar",
    )
    parser.add_argument("--offline-input-momentum", type=float, default=0.0)
    parser.add_argument("--offline-input-scale", type=float, default=1.0)
    parser.add_argument("--offline-local-input-scale", type=float, default=0.3)
    parser.add_argument("--offline-input-hold", type=int, default=1)
    parser.add_argument("--offline-input-period", type=float, default=100.0)
    parser.add_argument("--control-steps", type=int, default=320)
    parser.add_argument("--evaluation-seeds", type=int, default=10)
    parser.add_argument("--evaluation-seed-start", type=int, default=501)
    parser.add_argument("--clusters", type=int, default=48)
    parser.add_argument("--contextual-clusters", type=int, default=128)
    parser.add_argument(
        "--contextual-architecture",
        choices=("global", "hierarchical", "goal_enriched"),
        default="global",
    )
    parser.add_argument("--contextual-modes-per-chart", type=int, default=4)
    parser.add_argument("--contextual-support", type=int, default=200)
    parser.add_argument("--contextual-goal-pool", type=int, default=4000)
    parser.add_argument(
        "--contextual-frame", choices=("absolute", "goal_relative"), default="absolute"
    )
    parser.add_argument("--contextual-future-weight", type=float, default=1.0)
    parser.add_argument(
        "--contextual-training-partition",
        choices=("all", "local"),
        default="all",
    )
    parser.add_argument("--contextual-t-f", type=int, default=0)
    parser.add_argument("--contextual-lam-g", type=float, default=None)
    parser.add_argument("--contextual-lam-du", type=float, default=None)
    parser.add_argument("--contextual-r", type=float, default=None)
    parser.add_argument("--contextual-penalize-initial-move", action="store_true")
    parser.add_argument("--contextual-velocity-weight-scale", type=float, default=None)
    parser.add_argument(
        "--contextual-cart-position-weight-scale", type=float, default=None
    )
    parser.add_argument(
        "--contextual-selection-cart-position-weight-scale",
        type=float,
        default=None,
    )
    parser.add_argument("--contextual-output-bias-gain", type=float, default=0.0)
    parser.add_argument("--transition-support", type=int, default=200)
    parser.add_argument(
        "--contextual-activation",
        choices=("triggered", "always", "latched", "successful", "hysteretic"),
        default="triggered",
    )
    parser.add_argument("--contextual-entry-threshold", type=float, default=None)
    parser.add_argument("--contextual-exit-threshold", type=float, default=None)
    parser.add_argument("--svd-rank", type=int, default=60)
    parser.add_argument("--contextual-svd-rank", type=int, default=0)
    parser.add_argument("--svd-bound-scale", type=float, default=3.0)
    parser.add_argument("--edge-count-threshold", type=int, default=2)
    parser.add_argument(
        "--graph-direction", choices=("directed", "undirected"), default="directed"
    )
    parser.add_argument("--waypoint-mode", choices=("chart_medoid", "edge_transition"), default="chart_medoid")
    parser.add_argument(
        "--physical-feature-mode",
        choices=("configuration", "state", "target_relative", "target_conditioned"),
        default="configuration",
    )
    parser.add_argument("--physical-goal-weight", type=float, default=1.0)
    parser.add_argument(
        "--physical-matrix-mode",
        choices=("current", "next", "edge_conditioned", "edge_union", "goal_path"),
        default="current",
    )
    parser.add_argument(
        "--reference-mode",
        choices=(
            "graph_routed",
            "route_sequence",
            "edge_sequence",
            "goal_path",
            "template_path",
            "final",
        ),
        default="graph_routed",
    )
    parser.add_argument("--route-dwell-steps", type=int, default=1)
    parser.add_argument("--template-selection-dwell-steps", type=int, default=1)
    parser.add_argument("--template-velocity-weight-scale", type=float, default=1.0)
    parser.add_argument(
        "--goal-construction-mode",
        choices=("exact", "feasible_sample"),
        default="exact",
    )
    parser.add_argument("--lam-g", type=float, default=0.05)
    parser.add_argument("--lam-ini", type=float, default=1.0e5)
    parser.add_argument("--lam-du", type=float, default=5.0)
    parser.add_argument("--penalize-initial-move", action="store_true")
    parser.add_argument("--r", type=float, default=1.0e-4)
    parser.add_argument("--velocity-weight-scale", type=float, default=1.0)
    parser.add_argument("--cart-position-weight-scale", type=float, default=1.0)
    parser.add_argument("--latch-threshold", type=float, default=None)
    parser.add_argument("--trigger-window", type=int, default=15)
    parser.add_argument("--trigger-minimum-progress", type=float, default=1.0e-3)
    parser.add_argument("--contextual-success-streak", type=int, default=1)
    parser.add_argument(
        "--latched-matrix-mode",
        choices=("normal", "goal_chart", "goal_focused"),
        default="normal",
    )
    parser.add_argument("--terminal-support", type=int, default=400)
    parser.add_argument("--terminal-svd-rank", type=int, default=0)
    parser.add_argument("--terminal-t-f", type=int, default=0)
    parser.add_argument("--terminal-lock-steps", type=int, default=0)
    parser.add_argument(
        "--terminal-matrix-mode",
        choices=("selected", "goal_chart"),
        default="selected",
    )
    parser.add_argument("--terminal-lam-g", type=float, default=None)
    parser.add_argument("--terminal-lam-du", type=float, default=None)
    parser.add_argument("--terminal-r", type=float, default=None)
    parser.add_argument("--terminal-velocity-weight-scale", type=float, default=None)
    return parser


def main(args: argparse.Namespace) -> None:
    config = PilotConfig(
        t_ini=args.t_ini,
        t_f=args.t_f,
        offline_episodes=args.offline_episodes,
        offline_steps=args.offline_steps,
        offline_reset_mode=args.offline_reset_mode,
        offline_input_mode=args.offline_input_mode,
        offline_input_momentum=args.offline_input_momentum,
        offline_input_scale=args.offline_input_scale,
        offline_local_input_scale=args.offline_local_input_scale,
        offline_input_hold=args.offline_input_hold,
        offline_input_period=args.offline_input_period,
        control_steps=args.control_steps,
        evaluation_seeds=args.evaluation_seeds,
        evaluation_seed_start=args.evaluation_seed_start,
        clusters=args.clusters,
        contextual_clusters=args.contextual_clusters,
        contextual_architecture=args.contextual_architecture,
        contextual_modes_per_chart=args.contextual_modes_per_chart,
        contextual_support=args.contextual_support,
        contextual_goal_pool=args.contextual_goal_pool,
        contextual_frame=args.contextual_frame,
        contextual_future_weight=args.contextual_future_weight,
        contextual_training_partition=args.contextual_training_partition,
        contextual_t_f=args.contextual_t_f,
        contextual_lam_g=args.contextual_lam_g,
        contextual_lam_du=args.contextual_lam_du,
        contextual_r=args.contextual_r,
        contextual_penalize_initial_move=args.contextual_penalize_initial_move,
        contextual_velocity_weight_scale=args.contextual_velocity_weight_scale,
        contextual_cart_position_weight_scale=(
            args.contextual_cart_position_weight_scale
        ),
        contextual_selection_cart_position_weight_scale=(
            args.contextual_selection_cart_position_weight_scale
        ),
        contextual_output_bias_gain=args.contextual_output_bias_gain,
        transition_support=args.transition_support,
        contextual_activation=args.contextual_activation,
        svd_rank=args.svd_rank,
        contextual_svd_rank=args.contextual_svd_rank,
        svd_bound_scale=args.svd_bound_scale,
        edge_count_threshold=args.edge_count_threshold,
        graph_direction=args.graph_direction,
        waypoint_mode=args.waypoint_mode,
        physical_feature_mode=args.physical_feature_mode,
        physical_goal_weight=args.physical_goal_weight,
        physical_matrix_mode=args.physical_matrix_mode,
        reference_mode=args.reference_mode,
        route_dwell_steps=args.route_dwell_steps,
        template_selection_dwell_steps=args.template_selection_dwell_steps,
        template_velocity_weight_scale=args.template_velocity_weight_scale,
        goal_construction_mode=args.goal_construction_mode,
        lam_g=args.lam_g,
        lam_ini=args.lam_ini,
        lam_du=args.lam_du,
        penalize_initial_move=args.penalize_initial_move,
        r=args.r,
        velocity_weight_scale=args.velocity_weight_scale,
        cart_position_weight_scale=args.cart_position_weight_scale,
        latch_threshold_override=args.latch_threshold,
        trigger_window=args.trigger_window,
        trigger_minimum_progress=args.trigger_minimum_progress,
        contextual_success_streak=args.contextual_success_streak,
        contextual_entry_threshold=args.contextual_entry_threshold,
        contextual_exit_threshold=args.contextual_exit_threshold,
        latched_matrix_mode=args.latched_matrix_mode,
        terminal_support=args.terminal_support,
        terminal_svd_rank=args.terminal_svd_rank,
        terminal_t_f=args.terminal_t_f,
        terminal_lock_steps=args.terminal_lock_steps,
        terminal_matrix_mode=args.terminal_matrix_mode,
        terminal_lam_g=args.terminal_lam_g,
        terminal_lam_du=args.terminal_lam_du,
        terminal_r=args.terminal_r,
        terminal_velocity_weight_scale=args.terminal_velocity_weight_scale,
    )
    task_names = [name.strip() for name in args.tasks.split(",") if name.strip()]
    unknown = sorted(set(task_names) - set(TASKS))
    if unknown:
        raise SystemExit(f"Unknown tasks: {unknown}")
    methods = tuple(name.strip() for name in args.methods.split(",") if name.strip())
    unknown_methods = sorted(set(methods) - {"atlas", "contextual_atlas"})
    if unknown_methods:
        raise SystemExit(f"Unknown methods: {unknown_methods}")
    summaries = [
        run_task(TASKS[name], config, args.output_root, methods) for name in task_names
    ]
    aggregate = {
        "config": asdict(config),
        "tasks": [
            {
                "task": summary["task"],
                "offline_atlas": summary["offline_atlas"],
                "preprocessing_timing_ms": summary["preprocessing_timing_ms"],
                "atlas": summary.get("atlas"),
                "contextual_atlas": summary.get("contextual_atlas"),
                "offline_audit": summary["offline_audit"],
            }
            for summary in summaries
        ],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    aggregate_path = args.output_root / "summary.json"
    if aggregate_path.exists():
        previous = json.loads(aggregate_path.read_text())
        merged = {row["task"]: row for row in previous.get("tasks", [])}
        merged.update({row["task"]: row for row in aggregate["tasks"]})
        aggregate["tasks"] = [merged[name] for name in sorted(merged)]
    aggregate_path.write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main(build_parser().parse_args())
