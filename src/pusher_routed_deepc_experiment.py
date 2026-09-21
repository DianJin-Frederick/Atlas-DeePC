"""Graph-routed, fixed-chart Atlas-DeePC for Gymnasium's Pusher-v5.

The routing and chart machinery is reused from ``routed_deepc_experiment``.
This module supplies the Pusher-specific dimensions, observation semantics,
offline data collection, goal construction, and dimensioned DeePC controller.
No dynamics model is fitted: every prediction is an affine combination of
columns from a fixed local behavioral chart.
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import cvxpy as cp
import gymnasium as gym
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

import expert_pipeline_search as pipe
import routed_deepc_experiment as reacher_routed


PUSHER_ENV_ID = "Pusher-v5"
M = 7
P = 23
T_INI = 2
T_F = 10
L = T_INI + T_F

QPOS = slice(0, 7)
QVEL = slice(7, 14)
FINGERTIP = slice(14, 17)
OBJECT = slice(17, 20)
GOAL = slice(20, 23)
OBJECT_XY = slice(17, 19)
GOAL_XY = slice(20, 22)

EXPERIMENT_DIR = Path("experiments/routed_deepc_pusher")
DEFAULT_DATA_PATH = Path("data/pusher_iid_offline_hankel.npy")


@dataclass
class PusherRoutedDeePCConfig:
    clusters: int = 12
    cluster_seed: int = 11
    formulation: str = "svd_current"
    svd_rank: int = 40
    svd_bound_scale: float = 4.0
    negative_mass: float = 0.2
    edge_count_threshold: int = 2
    target_switch_distance: float = 0.10
    latch_target_reference: bool = True
    target_matrix_mode: str = "current"
    q_joint: float = 5.0
    q_velocity: float = 10.0
    q_fingertip: float = 2000.0
    q_object_xy: float = 100000.0
    q_object_z: float = 100.0
    r: float = 0.001
    lam_g: float = 0.05
    lam_ini: float = 100000.0
    lam_du: float = 2.0
    u_limit: float = 2.0
    goal_chart_candidates: int = 256
    goal_charts: int = 2
    steps: int = 100
    seed: int = 1
    data_seed: int = 3026
    offline_episodes: int = 48
    offline_episode_length: int = 100
    input_scale: float = 2.0


def pusher_output_from_obs(obs):
    """Use the documented Pusher-v5 observation directly as DeePC output."""
    output = np.asarray(obs, dtype=float).reshape(-1)
    if output.size != P:
        raise ValueError(f"Expected a {P}-element Pusher-v5 observation, got {output.size}.")
    return output.copy()


def partition_hankel_rows(H):
    """Pusher counterpart of ``expert_pipeline_search.partition_hankel_rows``."""
    expected_rows = (M + P) * L
    if H.shape[0] != expected_rows:
        raise ValueError(
            f"Hankel row mismatch: got {H.shape[0]} rows, expected {expected_rows} "
            f"for M={M}, P={P}, T_INI={T_INI}, T_F={T_F}, L={L}."
        )
    U = H[: M * L].reshape(L, M, H.shape[1])
    Y = H[M * L :].reshape(L, P, H.shape[1])
    return (
        U[:T_INI].reshape(M * T_INI, H.shape[1]),
        U[T_INI:].reshape(M * T_F, H.shape[1]),
        Y[:T_INI].reshape(P * T_INI, H.shape[1]),
        Y[T_INI:].reshape(P * T_F, H.shape[1]),
    )


def collect_iid_hankel(config, save_path=DEFAULT_DATA_PATH):
    """Collect fixed Pusher behavioral columns with Select-DPC-style IID inputs."""
    rng = np.random.default_rng(config.data_seed)
    columns = []
    action_low = None
    action_high = None
    for episode in range(config.offline_episodes):
        env_seed = config.data_seed + episode
        env = gym.make(PUSHER_ENV_ID, max_episode_steps=config.offline_episode_length)
        obs, _ = env.reset(seed=env_seed)
        action_low = np.maximum(env.action_space.low, -config.input_scale)
        action_high = np.minimum(env.action_space.high, config.input_scale)
        actions = []
        outputs = []
        for _ in range(config.offline_episode_length):
            action = rng.uniform(action_low, action_high)
            obs, _, terminated, truncated, _ = env.step(action)
            actions.append(action.copy())
            outputs.append(pusher_output_from_obs(obs))
            if terminated or truncated:
                break
        env.close()
        actions = np.asarray(actions, dtype=float)
        outputs = np.asarray(outputs, dtype=float)
        for start in range(actions.shape[0] - L + 1):
            columns.append(
                np.concatenate(
                    [
                        actions[start : start + L].reshape(L * M),
                        outputs[start : start + L].reshape(L * P),
                    ]
                )
            )
    H = np.asarray(columns, dtype=float).T
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, H)
    return H


def behavioral_features(H):
    U_p, _, Y_p, _ = partition_hankel_rows(H)
    raw = np.vstack([U_p, Y_p])
    mean = raw.mean(axis=1)
    scale = np.maximum(raw.std(axis=1), 1e-8)
    normalized = (raw - mean[:, None]) / scale[:, None]
    normalized[: M * T_INI] *= np.sqrt(0.1)
    return normalized.T, mean, scale


def online_behavioral_feature(u_ini, y_ini, mean, scale):
    raw = np.concatenate([u_ini[:, 0], y_ini[:, 0]])
    feature = (raw - mean) / scale
    feature[: M * T_INI] *= np.sqrt(0.1)
    return feature


def fit_behavioral_atlas(H, config):
    """Fit fixed behavioral charts and their observed directed transition graph."""
    features, feature_mean, feature_scale = behavioral_features(H)
    gate = MiniBatchKMeans(
        n_clusters=config.clusters,
        random_state=config.cluster_seed,
        n_init=5,
        batch_size=2048,
        max_iter=300,
        reassignment_ratio=0.002,
    )
    labels = gate.fit_predict(features)
    windows_per_episode = config.offline_episode_length - L + 1
    expected_columns = config.offline_episodes * windows_per_episode
    if H.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} ordered offline windows, got {H.shape[1]}."
        )
    counts = np.zeros((config.clusters, config.clusters), dtype=int)
    for episode in range(config.offline_episodes):
        start = episode * windows_per_episode
        stop = start + windows_per_episode
        for source, destination in zip(labels[start : stop - 1], labels[start + 1 : stop]):
            counts[source, destination] += 1

    outputs = H[M * L :].reshape(L, P, H.shape[1])
    medoid_states = []
    for chart_id in range(config.clusters):
        indices = np.flatnonzero(labels == chart_id)
        if not indices.size:
            raise RuntimeError(f"Behavioral atlas produced empty chart {chart_id}.")
        error = np.mean((features[indices] - gate.cluster_centers_[chart_id]) ** 2, axis=1)
        state = outputs[T_INI - 1, :, int(indices[np.argmin(error)])].copy()
        state[QVEL] = 0.0
        medoid_states.append(state)
    center_delta = gate.cluster_centers_[:, None, :] - gate.cluster_centers_[None, :, :]
    center_distance = np.sqrt(np.mean(center_delta * center_delta, axis=2))
    return {
        "labels": labels,
        "centers": gate.cluster_centers_,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "transition_counts": counts,
        "center_distance": center_distance,
        "medoid_states": np.asarray(medoid_states),
    }


def chart_compatible_goal(H, target_xy, current_y, config):
    """Choose a fixed offline goal posture and translate its contact geometry."""
    outputs = H[M * L :].reshape(L, P, H.shape[1])[0].T
    object_error = np.linalg.norm(outputs[:, OBJECT_XY] - target_xy, axis=1)
    count = min(config.goal_chart_candidates, outputs.shape[0])
    candidates = np.argpartition(object_error, count - 1)[:count]
    # Separate common approach/contact postures, then choose the one compatible
    # with the current arm configuration, mirroring the Reacher goal selector.
    contact_features = np.column_stack(
        [outputs[candidates, QPOS], outputs[candidates, FINGERTIP] - outputs[candidates, OBJECT]]
    )
    n_charts = min(config.goal_charts, count)
    labels = KMeans(n_clusters=n_charts, n_init=10, random_state=0).fit_predict(contact_features)
    candidate_contact_distance = np.linalg.norm(
        outputs[candidates, FINGERTIP][:, :2] - outputs[candidates, OBJECT][:, :2],
        axis=1,
    )
    candidate_goal_score = object_error[candidates] + 0.25 * candidate_contact_distance
    branch_indices = np.asarray(
        [
            int(candidates[labels == k][np.argmin(candidate_goal_score[labels == k])])
            for k in range(n_charts)
        ]
    )
    branch_contact_distance = np.linalg.norm(
        outputs[branch_indices, FINGERTIP][:, :2] - outputs[branch_indices, OBJECT][:, :2],
        axis=1,
    )
    # Unlike Reacher, Pusher needs a contact configuration rather than the
    # posture most similar to its reset state. Choose the fixed branch that
    # actually places the fingertip beside the cylinder.
    contact_chart = int(np.argmin(branch_contact_distance))
    index = int(branch_indices[contact_chart])
    reference = outputs[index].copy()
    shift = target_xy - reference[OBJECT_XY]
    reference[QVEL] = 0.0
    reference[OBJECT_XY] = target_xy
    reference[14:16] += shift
    return reference, {
        "data_goal_index": index,
        "data_goal_object_xy_error": float(object_error[index]),
        "data_goal_chart": contact_chart,
        "data_goal_chart_count": n_charts,
        "data_goal_contact_distance": float(branch_contact_distance[contact_chart]),
        "reference_contact_offset_xy": (
            reference[FINGERTIP][:2] - reference[OBJECT][:2]
        ).tolist(),
    }


class DimensionedDeePCController:
    """The Reacher DeePC objective with explicit Pusher dimensions."""

    def __init__(self, Q, R, lam_g, lam_ini, lam_du, u_limit, solver=cp.CLARABEL):
        self.Q = Q
        self.R = R
        self.lam_g = lam_g
        self.lam_ini = lam_ini
        self.lam_du = lam_du
        self.u_limit = u_limit
        self.solver = solver

    def solve(self, chart, u_ini, y_ini, y_ref):
        offset, matrix, g_min, g_max = pipe.unpack_basis(chart)
        g_sum = chart.get("g_sum") if isinstance(chart, dict) else None
        u = cp.Variable((M * T_F, 1))
        y = cp.Variable((P * T_F, 1))
        g = cp.Variable((matrix.shape[1], 1))
        sigma = cp.Variable((P * T_INI, 1))
        behavior = cp.vstack([u_ini, u, y_ini + sigma, y])
        cost = self.lam_ini * cp.sum_squares(sigma) + self.lam_g * cp.sum_squares(g)
        for step in range(T_F):
            y_step = y[step * P : (step + 1) * P]
            ref_step = y_ref[step * P : (step + 1) * P]
            u_step = u[step * M : (step + 1) * M]
            cost += cp.quad_form(y_step - ref_step, self.Q)
            cost += cp.quad_form(u_step, self.R)
        for step in range(T_F - 1):
            cost += self.lam_du * cp.sum_squares(
                u[(step + 1) * M : (step + 2) * M] - u[step * M : (step + 1) * M]
            )
        constraints = [
            behavior == offset + matrix @ g,
            u <= self.u_limit,
            u >= -self.u_limit,
        ]
        if g_min is not None:
            constraints.append(g >= g_min)
        if g_max is not None:
            constraints.append(g <= g_max)
        if g_sum is not None:
            constraints.append(cp.sum(g) == float(g_sum))
        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(solver=self.solver, verbose=False, max_iter=10000)
        except Exception as exc:
            return None, None, None, f"crashed: {exc}"
        if problem.status not in ("optimal", "optimal_inaccurate"):
            return None, None, None, problem.status
        return u.value, y.value, g.value, problem.status


def pusher_cost_matrix(config):
    weights = np.zeros(P)
    weights[QPOS] = config.q_joint
    weights[QVEL] = config.q_velocity
    weights[FINGERTIP] = config.q_fingertip
    weights[OBJECT_XY] = config.q_object_xy
    weights[19] = config.q_object_z
    # Goal coordinates are constants supplied by the environment.
    weights[GOAL] = 0.0
    return np.diag(weights)


def run_routed_deepc(H, atlas, config, save_path=None):
    rollout_start = time.perf_counter()
    controller = DimensionedDeePCController(
        pusher_cost_matrix(config),
        config.r * np.eye(M),
        config.lam_g,
        config.lam_ini,
        config.lam_du,
        config.u_limit,
    )
    env = gym.make(PUSHER_ENV_ID, max_episode_steps=config.steps + T_INI + 5)
    obs, _ = env.reset(seed=config.seed)
    target_xy = np.asarray(obs[GOAL_XY], dtype=float).copy()
    initial_y = pusher_output_from_obs(obs)
    goal, goal_metadata = chart_compatible_goal(H, target_xy, initial_y, config)
    goal_index = int(goal_metadata["data_goal_index"])
    goal_chart = int(atlas["labels"][goal_index])

    u_history = []
    y_history = []
    for _ in range(T_INI):
        action = np.zeros(M)
        obs, *_ = env.step(action)
        u_history.append(action)
        y_history.append(pusher_output_from_obs(obs))

    target_reference_latched = False
    records = {
        "y": [],
        "u": [],
        "pred": [],
        "ref": [],
        "distance": [],
        "prediction_error_xy": [],
        "status": [],
        "chart": [],
        "goal_chart": [],
        "next_chart": [],
        "path_length": [],
        "matrix_columns": [],
        "g_min": [],
        "solve_time_s": [],
    }
    matrix_cache = {}
    for _ in range(config.steps):
        u_ini = np.asarray(u_history[-T_INI:], dtype=float).reshape(M * T_INI, 1)
        y_ini = np.asarray(y_history[-T_INI:], dtype=float).reshape(P * T_INI, 1)
        feature = online_behavioral_feature(
            u_ini, y_ini, atlas["feature_mean"], atlas["feature_scale"]
        )
        current_chart = int(reacher_routed.assign_charts(atlas["centers"], feature)[0])
        path = reacher_routed.shortest_chart_path(
            atlas,
            current_chart,
            goal_chart,
            config.edge_count_threshold,
        )
        if not path:
            path = [current_chart]
        next_chart = path[1] if len(path) > 1 else current_chart
        distance = float(np.linalg.norm(y_history[-1][OBJECT_XY] - target_xy))
        if distance < config.target_switch_distance:
            target_reference_latched = True
        use_target = (
            target_reference_latched
            if config.latch_target_reference
            else distance < config.target_switch_distance
        )
        active_reference = (
            goal if use_target or current_chart == goal_chart else atlas["medoid_states"][next_chart]
        )
        reference = np.tile(active_reference.reshape(P, 1), (T_F, 1))

        if use_target and config.target_matrix_mode == "goal":
            active_charts = (goal_chart,)
        elif use_target and config.target_matrix_mode == "current_goal_overlap":
            active_charts = tuple(sorted(set((current_chart, goal_chart))))
        else:
            active_charts = (current_chart,)
        if active_charts not in matrix_cache:
            matrix_cache[active_charts] = reacher_routed.chart_matrix(
                H, atlas, active_charts, config
            )
        matrix, selected = matrix_cache[active_charts]
        solve_start = time.perf_counter()
        u_solution, y_solution, g_solution, status = controller.solve(
            matrix, u_ini, y_ini, reference
        )
        solve_time = time.perf_counter() - solve_start
        if u_solution is None:
            action = np.zeros(M)
            prediction = np.full(P, np.nan)
            minimum_g = np.nan
        else:
            action = np.clip(u_solution[:M, 0], -config.u_limit, config.u_limit)
            prediction = np.asarray(y_solution[:P, 0]).reshape(P)
            minimum_g = float(np.min(g_solution))
        obs, *_ = env.step(action)
        y_now = pusher_output_from_obs(obs)
        u_history.append(action.copy())
        y_history.append(y_now.copy())
        records["y"].append(y_now)
        records["u"].append(action)
        records["pred"].append(prediction)
        records["ref"].append(active_reference)
        records["distance"].append(float(np.linalg.norm(y_now[OBJECT_XY] - target_xy)))
        records["prediction_error_xy"].append(
            float(np.linalg.norm(prediction[OBJECT_XY] - y_now[OBJECT_XY]))
        )
        records["status"].append(status)
        records["chart"].append(current_chart)
        records["goal_chart"].append(goal_chart)
        records["next_chart"].append(next_chart)
        records["path_length"].append(len(path))
        records["matrix_columns"].append(selected.size)
        records["g_min"].append(minimum_g)
        records["solve_time_s"].append(solve_time)
    env.close()

    for key in records:
        records[key] = np.asarray(records[key], dtype=object if key == "status" else None)
    distance = records["distance"]
    tail = distance[-min(20, distance.size) :]
    summary = {
        **asdict(config),
        "minimum": float(distance.min()),
        "minimum_step": int(distance.argmin()),
        "last": float(distance[-1]),
        "mean_last_20": float(tail.mean()),
        "max_last_20": float(tail.max()),
        "success_10cm": bool(tail.mean() < 0.10 and tail.max() < 0.12),
        "success_5cm": bool(tail.mean() < 0.05 and tail.max() < 0.07),
        "mean_prediction_error_xy": float(np.nanmean(records["prediction_error_xy"])),
        "mean_solve_time_ms": float(1000 * records["solve_time_s"].mean()),
        "rollout_wall_time_s": float(time.perf_counter() - rollout_start),
        "charts_visited": int(np.unique(records["chart"]).size),
        "statuses": {
            str(status): int(np.sum(records["status"] == status))
            for status in np.unique(records["status"])
        },
        "controller": "DeePC behavioral column combination",
        "fitted_dynamics_model": False,
        "online_nearest_trajectory_retrieval": False,
        **goal_metadata,
    }
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_path, **records, summary=np.array(json.dumps(summary), dtype=object))
    return summary


def run_experiment(config, data_path=DEFAULT_DATA_PATH, output_dir=None, force_collect=False):
    output_dir = Path(output_dir or EXPERIMENT_DIR / "notebook_run")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_path)
    if force_collect or not data_path.exists():
        H = collect_iid_hankel(config, save_path=data_path)
    else:
        H = np.load(data_path)
    atlas = fit_behavioral_atlas(H, config)
    atlas_path = output_dir / "behavioral_atlas.npz"
    reacher_routed.save_atlas(atlas_path, atlas, config)
    result_path = output_dir / f"seed{config.seed}_control_results.npz"
    summary = run_routed_deepc(H, atlas, config, save_path=result_path)
    payload = {
        "method": "graph_routed_fixed_chart_deepc",
        "environment": PUSHER_ENV_ID,
        "config": asdict(config),
        "hankel_shape": list(H.shape),
        "chart_sizes": np.bincount(atlas["labels"].astype(int), minlength=config.clusters).tolist(),
        "result": summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Graph-routed fixed-chart Atlas-DeePC for Pusher-v5")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "cli_run")
    parser.add_argument("--force-collect", action="store_true")
    parser.add_argument("--clusters", type=int, default=12)
    parser.add_argument("--svd-rank", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main(args):
    config = PusherRoutedDeePCConfig(
        clusters=args.clusters,
        svd_rank=args.svd_rank,
        offline_episodes=args.episodes,
        offline_episode_length=args.episode_length,
        steps=args.steps,
        seed=args.seed,
    )
    payload = run_experiment(
        config,
        data_path=args.data_path,
        output_dir=args.output_dir,
        force_collect=args.force_collect,
    )
    print(json.dumps(payload["result"], indent=2))


if __name__ == "__main__":
    main(parse_args())
