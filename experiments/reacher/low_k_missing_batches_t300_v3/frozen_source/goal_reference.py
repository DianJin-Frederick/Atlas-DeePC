"""Shared offline target lookup and Reacher terminal-reference selection."""

from functools import lru_cache

import numpy as np
from sklearn.cluster import KMeans

from atlas_deepc.control import expert_pipeline_search as pipe
from atlas_deepc.common.reacher_data import (
    DEFAULT_DATA_SEED,
    DEFAULT_EPISODE_LENGTH,
    environment_target,
)


@lru_cache(maxsize=None)
def offline_environment_targets(episodes, data_seed=DEFAULT_DATA_SEED):
    return np.asarray([environment_target(data_seed + episode) for episode in range(episodes)])


def chart_compatible_goal(H, target_xy, current_y, config):
    """Choose an elbow branch from fixed random-state goal charts, then its best sample."""
    outputs = H[pipe.M * pipe.L:].reshape(pipe.L, pipe.P, H.shape[1])[0].T
    xy_error = np.linalg.norm(outputs[:, 6:8] - target_xy, axis=1)
    windows_per_episode = DEFAULT_EPISODE_LENGTH - pipe.L + 1
    if outputs.shape[0] % windows_per_episode:
        raise ValueError("Offline Hankel columns do not contain complete episodes.")
    episodes = outputs.shape[0] // windows_per_episode
    source_target_distance = np.linalg.norm(
        offline_environment_targets(episodes) - target_xy,
        axis=1,
    )
    minimum_separation = getattr(config, "goal_source_target_min_distance", 0.0)
    eligible_episodes = source_target_distance >= minimum_separation
    eligible_indices = np.flatnonzero(
        np.repeat(eligible_episodes, windows_per_episode)
    )
    count = min(config.goal_chart_candidates, eligible_indices.size)
    if count < config.goal_charts:
        raise RuntimeError(
            f"Only {count} fair goal candidates remain after enforcing "
            f"{minimum_separation:.3f} m source-target separation."
        )
    local_candidates = np.argpartition(xy_error[eligible_indices], count - 1)[:count]
    candidates = eligible_indices[local_candidates]
    joint_rows = pipe.reacher_joint_configuration_rows()
    joint_features = outputs[candidates][:, joint_rows]
    n_charts = min(config.goal_charts, count)
    labels = KMeans(n_clusters=n_charts, n_init=10, random_state=0).fit_predict(joint_features)
    centers = np.asarray([joint_features[labels == k].mean(axis=0) for k in range(n_charts)])
    compatible_chart = int(np.argmin(np.mean((centers - current_y[joint_rows]) ** 2, axis=1)))
    branch_xy_error = np.asarray([
        np.min(xy_error[candidates[labels == k]]) for k in range(n_charts)
    ])
    selection_rule = getattr(config, "goal_branch_selection", "compatible")
    if selection_rule == "compatible":
        selected_chart = compatible_chart
    elif selection_rule == "closest_xy":
        selected_chart = int(np.argmin(branch_xy_error))
    else:
        raise ValueError(f"Unknown goal_branch_selection {selection_rule!r}.")
    requested_chart = getattr(config, "goal_chart_override", None)
    active_chart = selected_chart if requested_chart is None else int(requested_chart)
    if not 0 <= active_chart < n_charts:
        raise ValueError(f"goal_chart_override must be in [0, {n_charts}), got {active_chart}.")
    branch_candidates = candidates[labels == active_chart]
    index = int(branch_candidates[np.argmin(xy_error[branch_candidates])])
    reference = outputs[index].copy()
    reference[4:6] = 0.0
    reference[6:8] = target_xy
    source_seed = DEFAULT_DATA_SEED + index // windows_per_episode
    return reference, {
        "data_goal_index": index,
        "data_goal_xy_error": float(xy_error[index]),
        "data_goal_source_seed": source_seed,
        "data_goal_source_target_distance": float(
            np.linalg.norm(environment_target(source_seed) - target_xy)
        ),
        "data_goal_chart": active_chart,
        "data_goal_chart_count": n_charts,
    }
