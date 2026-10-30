"""Graph-routed, chart-local DeePC for Gymnasium's Swimmer-v5.

The transition graph selects a data-backed output reference.  Control actions
are produced only by a DeePC behavioral trajectory optimization; no dynamics
matrices are identified.
"""

from __future__ import annotations

import heapq
import json
import os
import time
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import cvxpy as cp
import gymnasium as gym
import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.cluster import MiniBatchKMeans


ENV_ID = "Swimmer-v5"
M = 2
P = 8


@dataclass
class SwimmerDeePCConfig:
    seed: int = 7
    data_seed: int = 23
    reset_noise_scale: float = 0.0
    episodes: int = 18
    episode_steps: int = 140
    t_ini: int = 2
    t_f: int = 10
    clusters: int = 12
    cluster_seed: int = 11
    edge_count_threshold: int = 2
    svd_rank: int = 48
    svd_bound_scale: float = 4.0
    control_steps: int = 60
    q_angle: float = 0.25
    q_x_velocity: float = 30.0
    q_y_velocity: float = 2.0
    q_angular_velocity: float = 0.15
    r: float = 0.01
    lam_g: float = 0.03
    lam_ini: float = 30000.0
    lam_du: float = 0.25
    u_limit: float = 1.0

    @property
    def horizon(self) -> int:
        return self.t_ini + self.t_f


def make_env(
    max_episode_steps: int,
    reset_noise_scale: float = 0.0,
    render_mode: str | None = None,
    width: int = 640,
    height: int = 480,
):
    """Use the documented default 8-vector observation (positions excluded)."""
    return gym.make(
        ENV_ID,
        exclude_current_positions_from_observation=True,
        reset_noise_scale=reset_noise_scale,
        render_mode=render_mode,
        width=width,
        height=height,
        max_episode_steps=max_episode_steps,
    )


def _behavior_action(rng, episode: int, step: int, previous: np.ndarray) -> np.ndarray:
    """Mixture of random and oscillatory excitation, independent of reward."""
    if episode % 4 == 0:
        target = rng.uniform(-1.0, 1.0, size=M)
        action = 0.82 * previous + 0.18 * target
    else:
        amplitude = 0.45 + 0.12 * (episode % 4)
        omega = 0.16 + 0.025 * (episode % 5)
        phase = 0.71 * episode
        offset = (np.pi / 2.0) * (1 if episode % 2 else -1)
        action = amplitude * np.array([
            np.sin(omega * step + phase),
            np.sin(omega * step + phase + offset),
        ])
        action += rng.normal(0.0, 0.08, size=M)
    return np.clip(action, -1.0, 1.0)


def collect_behavior_library(config: SwimmerDeePCConfig):
    """Collect contiguous trajectories and return a Hankel-column library."""
    rng = np.random.default_rng(config.data_seed)
    columns = []
    episode_slices = []
    episode_stats = []
    start_column = 0
    for episode in range(config.episodes):
        env = make_env(config.episode_steps + 2, config.reset_noise_scale)
        obs, info = env.reset(seed=config.data_seed + episode)
        previous = np.zeros(M)
        actions, outputs, x_positions, rewards = [], [], [], []
        for step in range(config.episode_steps):
            action = _behavior_action(rng, episode, step, previous)
            obs, reward, terminated, truncated, info = env.step(action.astype(np.float32))
            actions.append(action.copy())
            outputs.append(np.asarray(obs, dtype=float).copy())
            x_positions.append(float(info["x_position"]))
            rewards.append(float(reward))
            previous = action
            if terminated or truncated:
                break
        env.close()
        U = np.asarray(actions, dtype=float)
        Y = np.asarray(outputs, dtype=float)
        count = max(0, U.shape[0] - config.horizon + 1)
        for start in range(count):
            columns.append(np.concatenate([
                U[start:start + config.horizon].reshape(-1),
                Y[start:start + config.horizon].reshape(-1),
            ]))
        episode_slices.append((start_column, start_column + count))
        start_column += count
        episode_stats.append({
            "episode": episode,
            "columns": count,
            "x_progress": float(x_positions[-1] - x_positions[0]),
            "mean_reward": float(np.mean(rewards)),
        })
    H = np.asarray(columns, dtype=float).T
    expected_rows = (M + P) * config.horizon
    if H.shape[0] != expected_rows:
        raise RuntimeError(f"Hankel library has {H.shape[0]} rows, expected {expected_rows}.")
    return H, episode_slices, episode_stats


def partition_hankel(H: np.ndarray, config: SwimmerDeePCConfig):
    U = H[: M * config.horizon].reshape(config.horizon, M, H.shape[1])
    Y = H[M * config.horizon :].reshape(config.horizon, P, H.shape[1])
    U_p = U[: config.t_ini].reshape(M * config.t_ini, H.shape[1])
    U_f = U[config.t_ini :].reshape(M * config.t_f, H.shape[1])
    Y_p = Y[: config.t_ini].reshape(P * config.t_ini, H.shape[1])
    Y_f = Y[config.t_ini :].reshape(P * config.t_f, H.shape[1])
    return U_p, U_f, Y_p, Y_f


def behavioral_features(H: np.ndarray, config: SwimmerDeePCConfig):
    U_p, _, Y_p, _ = partition_hankel(H, config)
    raw = np.vstack([U_p, Y_p])
    mean = raw.mean(axis=1)
    scale = np.maximum(raw.std(axis=1), 1e-8)
    return ((raw - mean[:, None]) / scale[:, None]).T, mean, scale


def online_feature(u_ini, y_ini, mean, scale):
    raw = np.concatenate([np.asarray(u_ini).reshape(-1), np.asarray(y_ini).reshape(-1)])
    return (raw - mean) / scale


def assign_chart(centers: np.ndarray, feature: np.ndarray) -> int:
    delta = centers - np.asarray(feature).reshape(1, -1)
    return int(np.argmin(np.mean(delta * delta, axis=1)))


def fit_behavioral_atlas(H, episode_slices, config: SwimmerDeePCConfig):
    features, feature_mean, feature_scale = behavioral_features(H, config)
    gate = MiniBatchKMeans(
        n_clusters=config.clusters,
        random_state=config.cluster_seed,
        n_init=10,
        batch_size=1024,
        max_iter=300,
        reassignment_ratio=0.002,
    )
    labels = gate.fit_predict(features)
    counts = np.zeros((config.clusters, config.clusters), dtype=int)
    for start, stop in episode_slices:
        for source, destination in zip(labels[start : stop - 1], labels[start + 1 : stop]):
            counts[int(source), int(destination)] += 1

    Y = H[M * config.horizon :].reshape(config.horizon, P, H.shape[1])
    current_outputs = Y[config.t_ini - 1].T
    medoid_outputs = []
    chart_vx = []
    for chart in range(config.clusters):
        members = np.flatnonzero(labels == chart)
        error = np.mean((features[members] - gate.cluster_centers_[chart]) ** 2, axis=1)
        medoid_outputs.append(current_outputs[members[np.argmin(error)]])
        chart_vx.append(float(np.median(current_outputs[members, 3])))

    center_delta = gate.cluster_centers_[:, None, :] - gate.cluster_centers_[None, :, :]
    center_distance = np.sqrt(np.mean(center_delta * center_delta, axis=2))
    chart_vx = np.asarray(chart_vx)
    support = np.bincount(labels, minlength=config.clusters)
    supported = support >= max(20, int(0.25 * np.median(support)))
    goal_chart = int(np.argmax(np.where(supported, chart_vx, -np.inf)))
    outgoing = counts[goal_chart].copy()
    outgoing[goal_chart] = 0
    eligible = outgoing >= config.edge_count_threshold
    if np.any(eligible):
        goal_successor = int(np.argmax(np.where(eligible, outgoing, -1)))
    else:
        goal_successor = goal_chart
    return {
        "labels": labels,
        "centers": gate.cluster_centers_,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "transition_counts": counts,
        "center_distance": center_distance,
        "medoid_outputs": np.asarray(medoid_outputs),
        "chart_vx": chart_vx,
        "goal_chart": goal_chart,
        "goal_successor": goal_successor,
    }


def shortest_chart_path(atlas, source: int, destination: int, threshold: int):
    if source == destination:
        return [source]
    counts = atlas["transition_counts"]
    distances = np.full(counts.shape[0], np.inf)
    previous = np.full(counts.shape[0], -1, dtype=int)
    distances[source] = 0.0
    queue = [(0.0, source)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        if node == destination:
            break
        for neighbor in np.flatnonzero(counts[node] >= threshold):
            neighbor = int(neighbor)
            if neighbor == node:
                continue
            edge_cost = atlas["center_distance"][node, neighbor] / np.sqrt(counts[node, neighbor])
            candidate = distance + edge_cost
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))
    if previous[destination] < 0:
        return []
    path = [destination]
    while path[-1] != source:
        path.append(int(previous[path[-1]]))
    return path[::-1]


def fixed_rank_left_svd_basis(X: np.ndarray, rank: int):
    covariance = X @ X.T
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][: min(rank, X.shape[0], X.shape[1] - 1)]
    return eigenvectors[:, order]


def chart_matrix(H, atlas, chart_ids, config: SwimmerDeePCConfig):
    selected = np.flatnonzero(np.isin(atlas["labels"], np.asarray(chart_ids, dtype=int)))
    X = H[:, selected]
    center = X.mean(axis=1, keepdims=True)
    basis = fixed_rank_left_svd_basis(X - center, config.svd_rank)
    coordinates = basis.T @ (X - center)
    coordinate_scale = np.maximum(coordinates.std(axis=1, keepdims=True), 1e-8)
    return {
        "offset": center,
        "basis": basis,
        "g_min": -config.svd_bound_scale * coordinate_scale,
        "g_max": config.svd_bound_scale * coordinate_scale,
    }, selected


class DeePCController:
    """Dimension-local DeePC optimizer for the Swimmer behavioral library."""

    def __init__(self, config: SwimmerDeePCConfig):
        self.config = config
        self.Q = np.diag([
            config.q_angle,
            config.q_angle,
            config.q_angle,
            config.q_x_velocity,
            config.q_y_velocity,
            config.q_angular_velocity,
            config.q_angular_velocity,
            config.q_angular_velocity,
        ])
        self.R = config.r * np.eye(M)

    def solve(self, matrix, u_ini, y_ini, reference):
        c = self.config
        u = cp.Variable((M * c.t_f, 1))
        y = cp.Variable((P * c.t_f, 1))
        g = cp.Variable((matrix["basis"].shape[1], 1))
        sigma = cp.Variable((P * c.t_ini, 1))
        behavior = cp.vstack([u_ini, u, y_ini + sigma, y])
        cost = c.lam_ini * cp.sum_squares(sigma) + c.lam_g * cp.sum_squares(g)
        for step in range(c.t_f):
            y_step = y[step * P : (step + 1) * P]
            u_step = u[step * M : (step + 1) * M]
            r_step = reference[step * P : (step + 1) * P]
            cost += cp.quad_form(y_step - r_step, self.Q)
            cost += cp.quad_form(u_step, self.R)
        for step in range(c.t_f - 1):
            delta = u[(step + 1) * M : (step + 2) * M] - u[step * M : (step + 1) * M]
            cost += c.lam_du * cp.sum_squares(delta)
        constraints = [
            behavior == matrix["offset"] + matrix["basis"] @ g,
            u <= c.u_limit,
            u >= -c.u_limit,
            g <= matrix["g_max"],
            g >= matrix["g_min"],
        ]
        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(solver=cp.CLARABEL, verbose=False, max_iter=10000)
        except Exception as exc:
            return None, None, f"crashed: {exc}"
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return None, None, problem.status
        return np.asarray(u.value), np.asarray(y.value), problem.status


def run_graph_routed_deepc(H, atlas, config: SwimmerDeePCConfig):
    controller = DeePCController(config)
    env = make_env(config.control_steps + config.t_ini + 2, config.reset_noise_scale)
    obs, info = env.reset(seed=config.seed)
    u_history, y_history = [], []
    for _ in range(config.t_ini):
        obs, _, _, _, info = env.step(np.zeros(M, dtype=np.float32))
        u_history.append(np.zeros(M))
        y_history.append(np.asarray(obs, dtype=float))

    goal_chart = int(atlas["goal_chart"])
    cache = {}
    records = {
        "x": [], "y_position": [], "x_velocity": [], "reward": [], "u": [],
        "output": [], "prediction": [], "reference": [], "chart": [],
        "next_chart": [], "path_length": [], "status": [], "solve_time_s": [],
        "matrix_columns": [],
    }
    for _ in range(config.control_steps):
        u_ini_array = np.asarray(u_history[-config.t_ini :]).reshape(-1, 1)
        y_ini_array = np.asarray(y_history[-config.t_ini :]).reshape(-1, 1)
        feature = online_feature(
            u_ini_array,
            y_ini_array,
            atlas["feature_mean"],
            atlas["feature_scale"],
        )
        current_chart = assign_chart(atlas["centers"], feature)
        if current_chart == goal_chart and int(atlas["goal_successor"]) != goal_chart:
            successor = int(atlas["goal_successor"])
            return_path = shortest_chart_path(
                atlas, successor, goal_chart, config.edge_count_threshold
            )
            path = [goal_chart] + (return_path if return_path else [successor])
        else:
            path = shortest_chart_path(
                atlas, current_chart, goal_chart, config.edge_count_threshold
            )
        if not path:
            path = [current_chart]
        next_chart = path[1] if len(path) > 1 else current_chart
        reference_output = atlas["medoid_outputs"][next_chart].copy()
        reference = np.tile(reference_output, config.t_f).reshape(-1, 1)

        active_charts = tuple(sorted({current_chart, next_chart}))
        if active_charts not in cache:
            cache[active_charts] = chart_matrix(H, atlas, active_charts, config)
        matrix, selected = cache[active_charts]
        solve_start = time.perf_counter()
        u_solution, y_solution, status = controller.solve(
            matrix, u_ini_array, y_ini_array, reference
        )
        solve_time = time.perf_counter() - solve_start
        if u_solution is None:
            action = np.zeros(M)
            prediction = np.full(P, np.nan)
        else:
            action = np.clip(u_solution[:M, 0], -config.u_limit, config.u_limit)
            prediction = y_solution[:P, 0]
        obs, reward, terminated, truncated, info = env.step(action.astype(np.float32))
        output = np.asarray(obs, dtype=float)
        u_history.append(action.copy())
        y_history.append(output.copy())
        records["x"].append(float(info["x_position"]))
        records["y_position"].append(float(info["y_position"]))
        records["x_velocity"].append(float(info["x_velocity"]))
        records["reward"].append(float(reward))
        records["u"].append(action)
        records["output"].append(output)
        records["prediction"].append(prediction)
        records["reference"].append(reference_output)
        records["chart"].append(current_chart)
        records["next_chart"].append(next_chart)
        records["path_length"].append(len(path))
        records["status"].append(status)
        records["solve_time_s"].append(solve_time)
        records["matrix_columns"].append(selected.size)
        if terminated or truncated:
            break
    env.close()
    for key, values in records.items():
        records[key] = np.asarray(values, dtype=object if key == "status" else None)
    records["goal_chart"] = np.array(goal_chart)
    return records


def run_zero_action_baseline(config: SwimmerDeePCConfig):
    env = make_env(config.control_steps + config.t_ini + 2, config.reset_noise_scale)
    obs, info = env.reset(seed=config.seed)
    x, vx, rewards = [], [], []
    for _ in range(config.t_ini + config.control_steps):
        obs, reward, terminated, truncated, info = env.step(np.zeros(M, dtype=np.float32))
        if len(x) >= config.t_ini:
            vx.append(float(info["x_velocity"]))
            rewards.append(float(reward))
        x.append(float(info["x_position"]))
        if terminated or truncated:
            break
    env.close()
    x = np.asarray(x[config.t_ini :], dtype=float)
    return {"x": x, "x_velocity": np.asarray(vx), "reward": np.asarray(rewards)}


def summarize(records, baseline, atlas, config: SwimmerDeePCConfig):
    statuses = records["status"].astype(str)
    progress = float(records["x"][-1] - records["x"][0])
    baseline_progress = float(baseline["x"][-1] - baseline["x"][0])
    return {
        "method": "graph_routed_chart_local_deepc",
        "environment": ENV_ID,
        "config": asdict(config),
        "offline_columns": int(atlas["labels"].size),
        "goal_chart": int(atlas["goal_chart"]),
        "goal_successor": int(atlas["goal_successor"]),
        "goal_chart_median_x_velocity": float(atlas["chart_vx"][atlas["goal_chart"]]),
        "forward_progress": progress,
        "zero_action_forward_progress": baseline_progress,
        "mean_x_velocity": float(np.mean(records["x_velocity"])),
        "mean_reward": float(np.mean(records["reward"])),
        "charts_visited": int(np.unique(records["chart"]).size),
        "route_transitions_requested": int(np.sum(records["chart"] != records["next_chart"])),
        "optimal_fraction": float(np.mean(np.isin(statuses, ["optimal", "optimal_inaccurate"]))),
        "median_solve_time_ms": float(1000 * np.median(records["solve_time_s"])),
        "fitted_dynamics_model": False,
        "graph_generates_actions": False,
    }


def save_experiment(output_dir: Path, H, atlas, records, baseline, summary, episode_stats):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "offline_hankel.npy", H)
    np.savez(output_dir / "behavioral_atlas.npz", **atlas)
    np.savez(output_dir / "rollout.npz", **records)
    np.savez(output_dir / "zero_action_baseline.npz", **baseline)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "offline_episode_stats.json").write_text(json.dumps(episode_stats, indent=2))


def render_saved_rollout(
    result_path: Path,
    output_path: Path,
    config: SwimmerDeePCConfig,
    verification_path: Path | None = None,
    fps: int = 25,
):
    """Render a headless mechanism dashboard directly from a saved rollout."""
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    result = np.load(result_path, allow_pickle=True)
    actions = np.asarray(result["u"], dtype=float)
    outputs = np.asarray(result["output"], dtype=float)
    x_position = np.asarray(result["x"], dtype=float)
    y_position = np.asarray(result["y_position"], dtype=float)
    x_velocity = np.asarray(result["x_velocity"], dtype=float)
    charts = np.asarray(result["chart"], dtype=int)
    next_charts = np.asarray(result["next_chart"], dtype=int)
    goal_chart = int(np.asarray(result["goal_chart"]).item())
    steps = np.arange(actions.shape[0])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#f8fafc")
    grid = fig.add_gridspec(2, 3, width_ratios=(1.45, 1.0, 1.0))
    body_ax = fig.add_subplot(grid[:, 0])
    progress_ax = fig.add_subplot(grid[0, 1:])
    route_ax = fig.add_subplot(grid[1, 1])
    input_ax = fig.add_subplot(grid[1, 2])
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    link_length = 0.1
    try:
        for step in steps:
            for axis in (body_ax, progress_ax, route_ax, input_ax):
                axis.clear()

            q0, q1, q2 = outputs[step, :3]
            directions = np.column_stack([
                np.cos([q0, q0 + q1, q0 + q1 + q2]),
                np.sin([q0, q0 + q1, q0 + q1 + q2]),
            ])
            root = np.array([x_position[step], y_position[step]])
            nose = root + 0.5 * link_length * directions[0]
            joint_1 = root - 0.5 * link_length * directions[0]
            joint_2 = joint_1 - link_length * directions[1]
            tail = joint_2 - link_length * directions[2]
            points = np.vstack([nose, joint_1, joint_2, tail])

            body_ax.set_facecolor("#e0f2fe")
            body_ax.plot(
                points[:, 0], points[:, 1], color="#0f172a", linewidth=15,
                solid_capstyle="round", zorder=2,
            )
            body_ax.plot(
                points[:, 0], points[:, 1], color="#38bdf8", linewidth=9,
                solid_capstyle="round", zorder=3,
            )
            body_ax.scatter(
                points[1:3, 0], points[1:3, 1], s=65, color="#f97316",
                edgecolor="#7c2d12", linewidth=1.0, zorder=4,
            )
            body_ax.scatter(
                nose[0], nose[1], s=90, color="#f8fafc",
                edgecolor="#0f172a", linewidth=1.2, zorder=5,
            )
            camera_x = x_position[step]
            camera_y = y_position[step]
            body_ax.set_xlim(camera_x - 0.32, camera_x + 0.32)
            body_ax.set_ylim(camera_y - 0.25, camera_y + 0.25)
            body_ax.set_aspect("equal")
            body_ax.grid(color="#7dd3fc", alpha=0.25)
            body_ax.set_xlabel("world x (m)")
            body_ax.set_ylabel("world y (m)")
            body_ax.set_title(
                f"Swimmer mechanism | t={step * 0.04:4.2f} s\n"
                f"x={x_position[step]:+.3f} m, vx={x_velocity[step]:+.3f} m/s",
                fontsize=12,
            )

            progress = x_position - x_position[0]
            progress_ax.plot(steps, progress, color="#94a3b8", linewidth=1.5)
            progress_ax.plot(steps[: step + 1], progress[: step + 1], color="#1565c0", linewidth=3)
            progress_ax.scatter(step, progress[step], color="#dc2626", s=45, zorder=3)
            progress_ax.axhline(0.0, color="#64748b", linestyle="--", linewidth=0.8)
            progress_ax.set_xlim(0, max(1, actions.shape[0] - 1))
            pad = max(0.01, 0.08 * np.ptp(progress))
            progress_ax.set_ylim(progress.min() - pad, progress.max() + pad)
            progress_ax.set_title("Forward progress")
            progress_ax.set_xlabel("control step")
            progress_ax.set_ylabel("x displacement (m)")
            progress_ax.grid(alpha=0.2)

            route_ax.step(steps, charts, where="post", color="#1565c0", alpha=0.25)
            route_ax.step(steps, next_charts, where="post", color="#f57c00", alpha=0.25)
            route_ax.step(steps[: step + 1], charts[: step + 1], where="post", color="#1565c0", label="localized")
            route_ax.step(steps[: step + 1], next_charts[: step + 1], where="post", color="#f57c00", label="requested")
            route_ax.axhline(goal_chart, color="#eab308", linestyle="--", label="fast anchor")
            route_ax.set_xlim(0, max(1, actions.shape[0] - 1))
            route_ax.set_ylim(-0.5, max(goal_chart, charts.max(), next_charts.max()) + 0.5)
            route_ax.set_title("Graph routing")
            route_ax.set_xlabel("control step")
            route_ax.set_ylabel("chart ID")
            route_ax.legend(loc="upper right", fontsize=7, frameon=False)
            route_ax.grid(alpha=0.2)

            input_ax.plot(steps, actions[:, 0], color="#94a3b8", alpha=0.35)
            input_ax.plot(steps, actions[:, 1], color="#cbd5e1", alpha=0.55)
            input_ax.plot(steps[: step + 1], actions[: step + 1, 0], color="#1565c0", label="rotor 1")
            input_ax.plot(steps[: step + 1], actions[: step + 1, 1], color="#f57c00", label="rotor 2")
            input_ax.scatter([step, step], actions[step], color=["#1565c0", "#f57c00"], s=22)
            input_ax.set_xlim(0, max(1, actions.shape[0] - 1))
            input_ax.set_ylim(-1.05, 1.05)
            input_ax.set_title("Applied DeePC torques")
            input_ax.set_xlabel("control step")
            input_ax.set_ylabel("torque")
            input_ax.legend(loc="lower right", fontsize=7, frameon=False)
            input_ax.grid(alpha=0.2)

            fig.suptitle("Graph-routed Atlas-DeePC | Swimmer-v5", fontsize=16, weight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
            writer.append_data(frame)
    finally:
        writer.close()
        plt.close(fig)

    verification = {
        "result_path": str(result_path),
        "video_path": str(output_path),
        "environment": ENV_ID,
        "seed": config.seed,
        "actions": int(actions.shape[0]),
        "frames": int(actions.shape[0]),
        "fps": fps,
        "rendered_from_saved_rollout": True,
        "renderer": "headless_kinematic_dashboard",
    }
    if verification_path is not None:
        Path(verification_path).write_text(json.dumps(verification, indent=2))
    return verification


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-result", type=Path)
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--verification-output", type=Path)
    parser.add_argument("--config-json", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.render_result is None or args.video_output is None or args.config_json is None:
        raise SystemExit(
            "Rendering requires --render-result, --video-output, and --config-json."
        )
    payload = json.loads(args.config_json.read_text())
    render_config = SwimmerDeePCConfig(**payload["config"])
    audit = render_saved_rollout(
        args.render_result,
        args.video_output,
        render_config,
        verification_path=args.verification_output,
    )
    print(json.dumps(audit, indent=2), flush=True)
