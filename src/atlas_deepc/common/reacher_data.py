"""Reacher offline-data metadata, target lookup, and independence checks.

These helpers are independent of the chart model (linear or affine).
"""

import gymnasium as gym
import numpy as np

from atlas_deepc.control import expert_pipeline_search as pipe


DEFAULT_EVAL_SEEDS = [1, 2, 3, 4, 5, 7, 8, 42]


DEFAULT_DATA_SEED = 2026


DEFAULT_EPISODE_LENGTH = 200


def environment_target(seed):
    env = gym.make(pipe.REACHER_ENV_ID)
    obs, _ = env.reset(seed=seed)
    target = np.asarray(obs[4:6], dtype=float).copy()
    env.close()
    return target


def audit_iid_library(H, data_seed=2026, episodes=200, episode_len=200, eval_seeds=DEFAULT_EVAL_SEEDS):
    windows_per_episode = episode_len - pipe.L + 1
    expected_columns = episodes * windows_per_episode
    if H.shape[1] != expected_columns:
        raise ValueError(f"Expected {expected_columns} columns for the audit, got {H.shape[1]}.")
    Y = H[pipe.M * pipe.L:].reshape(pipe.L, pipe.P, H.shape[1])
    episode_metrics = []
    offline_targets = []
    for episode in range(episodes):
        start = episode * windows_per_episode
        stop = (episode + 1) * windows_per_episode
        trajectory_y = np.vstack([Y[:, :, start], Y[-1, :, start + 1:stop].T])
        target = environment_target(data_seed + episode)
        offline_targets.append(target)
        distance = np.linalg.norm(trajectory_y[:, 6:8] - target, axis=1)
        episode_metrics.append({
            "seed": data_seed + episode,
            "minimum": float(distance.min()),
            "mean_last_30": float(distance[-30:].mean()),
            "max_last_30": float(distance[-30:].max()),
            "success": bool(distance[-30:].mean() < 0.01 and distance[-30:].max() < 0.02),
        })
    stabilized = [row for row in episode_metrics if row["success"]]
    if stabilized:
        raise RuntimeError(f"Fairness audit failed: {len(stabilized)} offline episodes stabilize at their targets.")
    offline_targets = np.asarray(offline_targets)
    held_out = {}
    for seed in eval_seeds:
        if data_seed <= seed < data_seed + episodes:
            raise RuntimeError(f"Evaluation seed {seed} appears in the offline collection seed range.")
        target = environment_target(seed)
        distances = np.linalg.norm(offline_targets - target, axis=1)
        held_out[str(seed)] = {
            "target": target.tolist(),
            "nearest_offline_target_distance": float(distances.min()),
            "nearest_offline_seed": int(data_seed + distances.argmin()),
        }
    return {
        "data_seed_range": [data_seed, data_seed + episodes - 1],
        "offline_stabilized_episode_count": 0,
        "offline_transient_within_1cm_count": int(sum(row["minimum"] < 0.01 for row in episode_metrics)),
        "offline_best_tail_mean": float(min(row["mean_last_30"] for row in episode_metrics)),
        "held_out_targets": held_out,
    }
