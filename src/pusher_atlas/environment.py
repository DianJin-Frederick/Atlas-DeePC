"""Pusher-v5 environment and reset-invariant output helpers."""

from __future__ import annotations

import numpy as np


STATE_NAMES = np.array(
    [
        *[f"q{i}" for i in range(7)],
        *[f"qvel{i}" for i in range(7)],
        "tip_x",
        "tip_y",
        "tip_z",
        "object_x",
        "object_y",
        "object_vx",
        "object_vy",
    ]
)

QPOS = slice(0, 7)
QVEL = slice(7, 14)
TIP = slice(14, 17)
OBJECT_XY = slice(17, 19)
OBJECT_VXY = slice(19, 21)
STATE_DIM = 21
ACTION_DIM = 7

PHASE_APPROACH = 0
PHASE_PUSH = 1
PHASE_HOLD = 2
PHASE_NAMES = np.array(["approach", "push", "hold"])


def make_env(seed: int, max_episode_steps: int = 100, render_mode: str | None = None):
    import gymnasium as gym

    env = gym.make(
        "Pusher-v5",
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
    )
    obs, _ = env.reset(seed=int(seed))
    return env, np.asarray(obs, dtype=float)


def state_from_env(env) -> np.ndarray:
    """Return a Markov task state; convert object joint order to world x/y."""
    unwrapped = env.unwrapped
    qpos = np.asarray(unwrapped.data.qpos[:7], dtype=float)
    qvel = np.asarray(unwrapped.data.qvel[:7], dtype=float)
    tip = np.asarray(unwrapped.get_body_com("tips_arm"), dtype=float)
    obj = np.asarray(unwrapped.get_body_com("object")[:2], dtype=float)
    # XML order is obj_slidey, obj_slidex.
    obj_vel = np.asarray(
        [unwrapped.data.qvel[8], unwrapped.data.qvel[7]], dtype=float
    )
    state = np.concatenate([qpos, qvel, tip, obj, obj_vel])
    if state.shape != (STATE_DIM,):
        raise RuntimeError(f"Unexpected Pusher state shape {state.shape}")
    return state


def goal_xy_from_env(env) -> np.ndarray:
    return np.asarray(env.unwrapped.get_body_com("goal")[:2], dtype=float)


def behavior_output(state: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
    """Express task positions in reset-invariant coordinates for DeePC charts.

    Joint positions/velocities and object velocity are unchanged.  The fingertip
    xy position becomes tip-minus-object, and object xy becomes object-minus-goal.
    This is a measured output transformation, not a dynamics model.
    """
    output = np.asarray(state, dtype=float).copy()
    object_xy = output[OBJECT_XY].copy()
    output[TIP.start : TIP.start + 2] -= object_xy
    output[OBJECT_XY] = object_xy - np.asarray(goal_xy, dtype=float)
    return output


def push_direction(state: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
    delta = np.asarray(goal_xy, dtype=float) - np.asarray(state[OBJECT_XY], dtype=float)
    norm = np.linalg.norm(delta)
    return delta / max(norm, 1e-9)


def object_goal_distance(state: np.ndarray, goal_xy: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(state[OBJECT_XY]) - np.asarray(goal_xy)))
