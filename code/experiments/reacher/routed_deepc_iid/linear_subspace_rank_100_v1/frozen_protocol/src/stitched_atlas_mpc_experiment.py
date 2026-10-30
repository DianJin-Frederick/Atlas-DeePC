import argparse
import json
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import cvxpy as cp
import gymnasium as gym
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

import expert_pipeline_search as pipe
from local_affine_atlas_experiment import DEFAULT_EVAL_SEEDS, audit_iid_library
from local_affine_atlas_experiment import DEFAULT_DATA_SEED, DEFAULT_EPISODE_LENGTH, environment_target


EXPERIMENT_DIR = Path("experiments/stitched_atlas_iid")
DEFAULT_DATA_PATH = Path("data/select_iid_offline_hankel.npy")


@dataclass
class StitchedAtlasConfig:
    clusters: int = 64
    cluster_seed: int = 7
    ridge: float = 0.001
    schedule_iterations: int = 3
    horizon: int = 50
    near_horizon: int = 3
    near_switch_distance: float = 0.03
    goal_chart_candidates: int = 400
    goal_charts: int = 2
    q_xy: float = 100000.0
    q_dq: float = 20.0
    q_data_goal_joint: float = 5000.0
    r: float = 0.0001
    lam_du: float = 5.0
    u_limit: float = 1.0
    steps: int = 320
    seed: int = 1


def transition_data(H):
    U = H[:pipe.M * pipe.L].reshape(pipe.L, pipe.M, H.shape[1])
    Y = H[pipe.M * pipe.L:].reshape(pipe.L, pipe.P, H.shape[1])
    return Y[0].T, U[1].T, Y[1].T


def fit_stitched_atlas(H, config):
    """Fit fixed local affine one-step charts from the random offline library."""
    y, u, y_next = transition_data(H)
    y_mean = y.mean(axis=0)
    y_scale = np.maximum(y.std(axis=0), 1e-8)
    z = (y - y_mean) / y_scale
    z_next = (y_next - y_mean) / y_scale
    gate = MiniBatchKMeans(
        n_clusters=config.clusters,
        random_state=config.cluster_seed,
        n_init=5,
        batch_size=2048,
        max_iter=300,
        reassignment_ratio=0.005,
    )
    labels = gate.fit_predict(z)
    A = []
    B = []
    c = []
    fit_rmse = []
    for cluster_id in range(config.clusters):
        idx = np.where(labels == cluster_id)[0]
        X = np.column_stack([z[idx], u[idx], np.ones(idx.size)])
        gram = X.T @ X + config.ridge * np.eye(X.shape[1])
        coefficients = np.linalg.solve(gram, X.T @ z_next[idx])
        prediction = X @ coefficients
        A.append(coefficients[:pipe.P].T)
        B.append(coefficients[pipe.P:pipe.P + pipe.M].T)
        c.append(coefficients[-1])
        physical_error = (prediction - z_next[idx]) * y_scale
        fit_rmse.append(float(np.sqrt(np.mean(physical_error * physical_error))))
    return {
        "labels": labels,
        "gate_centers": gate.cluster_centers_,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "A": np.asarray(A),
        "B": np.asarray(B),
        "c": np.asarray(c),
        "fit_rmse": np.asarray(fit_rmse),
    }


def save_atlas(path, atlas, config):
    np.savez(path, **atlas, config=np.array(json.dumps(asdict(config)), dtype=object))


def load_atlas(path):
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files if key != "config"}


def nearest_charts(atlas, z):
    z = np.asarray(z)
    if z.ndim == 1:
        z = z[None, :]
    differences = z[:, None, :] - atlas["gate_centers"][None, :, :]
    return np.argmin(np.mean(differences * differences, axis=2), axis=1)


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


class StitchedAtlasController:
    """MPC that stitches a scheduled sequence of fixed affine dynamics charts."""

    def __init__(self, atlas, config):
        self.atlas = atlas
        self.config = config
        self.z_initial = cp.Parameter(pipe.P)
        horizon = config.horizon
        self.reference = cp.Parameter((horizon, pipe.P))
        self.A = [cp.Parameter((pipe.P, pipe.P)) for _ in range(horizon)]
        self.B = [cp.Parameter((pipe.P, pipe.M)) for _ in range(horizon)]
        self.c = [cp.Parameter(pipe.P) for _ in range(horizon)]
        self.z = cp.Variable((horizon + 1, pipe.P))
        self.u = cp.Variable((horizon, pipe.M))
        y_mean = atlas["y_mean"]
        y_scale = atlas["y_scale"]
        Q = np.diag([
            config.q_data_goal_joint,
            config.q_data_goal_joint,
            config.q_data_goal_joint,
            config.q_data_goal_joint,
            config.q_dq,
            config.q_dq,
            config.q_xy,
            config.q_xy,
            config.q_data_goal_joint,
            config.q_data_goal_joint,
        ])
        constraints = [
            self.z[0] == self.z_initial,
            self.u <= config.u_limit,
            self.u >= -config.u_limit,
        ]
        cost = 0
        for j in range(horizon):
            constraints.append(self.z[j + 1] == self.A[j] @ self.z[j] + self.B[j] @ self.u[j] + self.c[j])
            y_next = y_mean + cp.multiply(y_scale, self.z[j + 1])
            cost += cp.quad_form(y_next - self.reference[j], Q)
            cost += config.r * cp.sum_squares(self.u[j])
        for j in range(horizon - 1):
            cost += config.lam_du * cp.sum_squares(self.u[j + 1] - self.u[j])
        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, current_y, reference):
        z0 = (current_y - self.atlas["y_mean"]) / self.atlas["y_scale"]
        schedule = np.repeat(nearest_charts(self.atlas, z0)[0], self.config.horizon)
        best = None
        iterations = 0
        for iteration in range(self.config.schedule_iterations):
            iterations = iteration + 1
            self.z_initial.value = z0
            self.reference.value = reference
            for j, chart in enumerate(schedule):
                self.A[j].value = self.atlas["A"][chart]
                self.B[j].value = self.atlas["B"][chart]
                self.c[j].value = self.atlas["c"][chart]
            try:
                self.problem.solve(solver=cp.CLARABEL, warm_start=True, verbose=False, max_iter=10000)
            except Exception as exc:
                return None, None, schedule, iterations, f"crashed: {exc}"
            if self.problem.status not in ("optimal", "optimal_inaccurate"):
                return None, None, schedule, iterations, self.problem.status
            best = (self.u.value.copy(), self.z.value.copy(), float(self.problem.value))
            new_schedule = nearest_charts(self.atlas, self.z.value[:-1])
            if np.array_equal(new_schedule, schedule):
                break
            schedule = new_schedule
        return best[0], best[1], schedule, iterations, self.problem.status


class SoftStitchedAtlasController(StitchedAtlasController):
    """Stitch smooth convex blends of fixed one-step dynamics charts."""

    def __init__(self, atlas, config, temperature=0.1, top_charts=4):
        super().__init__(atlas, config)
        self.temperature = float(temperature)
        self.top_charts = int(top_charts)
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive.")

    def model_weights(self, z):
        delta = self.atlas["gate_centers"] - z[None, :]
        scores = np.mean(delta * delta, axis=1)
        count = min(max(self.top_charts, 1), scores.size)
        selected = np.argpartition(scores, count - 1)[:count]
        logits = -(scores[selected] - scores[selected].min()) / self.temperature
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        return selected, weights

    def solve(self, current_y, reference):
        z0 = (current_y - self.atlas["y_mean"]) / self.atlas["y_scale"]
        nominal = np.tile(z0, (self.config.horizon, 1))
        best = None
        dominant = np.zeros(self.config.horizon, dtype=int)
        iterations = 0
        for iteration in range(self.config.schedule_iterations):
            iterations = iteration + 1
            self.z_initial.value = z0
            self.reference.value = reference
            for j, state in enumerate(nominal):
                selected, weights = self.model_weights(state)
                dominant[j] = int(selected[np.argmax(weights)])
                self.A[j].value = np.tensordot(weights, self.atlas["A"][selected], axes=1)
                self.B[j].value = np.tensordot(weights, self.atlas["B"][selected], axes=1)
                self.c[j].value = np.tensordot(weights, self.atlas["c"][selected], axes=1)
            try:
                self.problem.solve(solver=cp.CLARABEL, warm_start=True, verbose=False, max_iter=10000)
            except Exception as exc:
                return None, None, dominant, iterations, f"crashed: {exc}"
            if self.problem.status not in ("optimal", "optimal_inaccurate"):
                return None, None, dominant, iterations, self.problem.status
            best = (self.u.value.copy(), self.z.value.copy())
            new_nominal = self.z.value[:-1].copy()
            if np.max(np.abs(new_nominal - nominal)) < 1e-5:
                break
            nominal = new_nominal
        return best[0], best[1], dominant, iterations, self.problem.status


def run_stitched_atlas(H, atlas, config, save_path=None):
    far_controller = StitchedAtlasController(atlas, config)
    near_config = StitchedAtlasConfig(**{**asdict(config), "horizon": config.near_horizon})
    near_controller = StitchedAtlasController(atlas, near_config)
    env = gym.make(pipe.REACHER_ENV_ID, max_episode_steps=config.steps + 5)
    obs, _ = env.reset(seed=config.seed)
    target_xy = np.asarray(obs[4:6], dtype=float).copy()
    current_y = pipe.reacher_output_from_obs(obs)
    if config.q_data_goal_joint > 0.0:
        goal, goal_metadata = chart_compatible_goal(H, target_xy, current_y, config)
    else:
        goal, goal_metadata = None, {}
    reference_single = goal if goal is not None else np.r_[np.zeros(6), target_xy, np.zeros(2)]
    reference = np.tile(reference_single, (config.horizon, 1))
    records = {
        "y": [], "u": [], "pred": [], "ref": [], "distance": [],
        "prediction_error_xy": [], "status": [], "schedule": [], "schedule_iterations": [],
        "active_horizon": [],
    }
    for _ in range(config.steps):
        current_distance = float(np.linalg.norm(current_y[6:8] - target_xy))
        active_config = near_config if current_distance < config.near_switch_distance else config
        active_controller = near_controller if current_distance < config.near_switch_distance else far_controller
        active_reference = np.tile(reference_single, (active_config.horizon, 1))
        u_solution, z_prediction, schedule, iterations, status = active_controller.solve(
            current_y, active_reference
        )
        if u_solution is None:
            action = np.zeros(pipe.M)
            prediction = np.full(pipe.P, np.nan)
        else:
            action = np.clip(u_solution[0], -config.u_limit, config.u_limit)
            prediction = atlas["y_mean"] + atlas["y_scale"] * z_prediction[1]
        obs, *_ = env.step(action)
        y_now = pipe.reacher_output_from_obs(obs)
        records["y"].append(y_now)
        records["u"].append(action)
        records["pred"].append(prediction)
        records["ref"].append(reference_single)
        records["distance"].append(float(np.linalg.norm(y_now[6:8] - target_xy)))
        records["prediction_error_xy"].append(float(np.linalg.norm(prediction[6:8] - y_now[6:8])))
        records["status"].append(status)
        records["schedule"].append(schedule)
        records["schedule_iterations"].append(iterations)
        records["active_horizon"].append(active_config.horizon)
        current_y = y_now
    env.close()

    charts_used = int(np.unique(np.concatenate(records["schedule"])).size)
    for key in records:
        records[key] = np.asarray(records[key], dtype=object if key in ("status", "schedule") else None)
    distance = records["distance"]
    tail_30 = distance[-30:]
    tail_100 = distance[-min(100, distance.size):]
    summary = {
        **asdict(config),
        "minimum": float(distance.min()),
        "minimum_step": int(distance.argmin()),
        "last": float(distance[-1]),
        "mean_last_30": float(tail_30.mean()),
        "max_last_30": float(tail_30.max()),
        "mean_last_100": float(tail_100.mean()),
        "max_last_100": float(tail_100.max()),
        "success_30": bool(tail_30.mean() < 0.01 and tail_30.max() < 0.02),
        "success_100": bool(tail_100.mean() < 0.01 and tail_100.max() < 0.02),
        "mean_prediction_error_xy": float(np.nanmean(records["prediction_error_xy"])),
        "mean_schedule_iterations": float(records["schedule_iterations"].mean()),
        "charts_used": charts_used,
        "statuses": {str(s): int(np.sum(records["status"] == s)) for s in np.unique(records["status"])},
        "online_trajectory_retrieval": False,
        "use_ik_ref": False,
        "feedback_blend": 0.0,
        "feedback_fallback": False,
        **goal_metadata,
    }
    if save_path is not None:
        np.savez(save_path, **records, summary=np.array(json.dumps(summary), dtype=object))
    return summary


def main(args):
    output_dir = EXPERIMENT_DIR / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    H = np.load(args.data_path)
    audit = audit_iid_library(H, eval_seeds=args.seeds)
    (output_dir / "fairness_audit.json").write_text(json.dumps(audit, indent=2))
    config = StitchedAtlasConfig(
        clusters=args.clusters,
        cluster_seed=args.cluster_seed,
        ridge=args.ridge,
        schedule_iterations=args.schedule_iterations,
        horizon=args.horizon,
        near_horizon=args.near_horizon,
        near_switch_distance=args.near_switch_distance,
        goal_chart_candidates=args.goal_chart_candidates,
        goal_charts=args.goal_charts,
        q_data_goal_joint=args.q_data_goal_joint,
        steps=args.steps,
    )
    atlas_path = output_dir / "stitched_atlas.npz"
    if args.reuse_atlas and args.reuse_atlas.exists():
        atlas = load_atlas(args.reuse_atlas)
        save_atlas(atlas_path, atlas, config)
    else:
        atlas = fit_stitched_atlas(H, config)
        save_atlas(atlas_path, atlas, config)
    sizes = np.bincount(atlas["labels"].astype(int), minlength=config.clusters)
    print(
        "chart sizes", int(sizes.min()), float(np.median(sizes)), int(sizes.max()),
        "fit rmse", float(atlas["fit_rmse"].mean()),
        flush=True,
    )
    summaries = []
    for seed in args.seeds:
        run_config = StitchedAtlasConfig(**{**asdict(config), "seed": seed})
        result_path = output_dir / f"seed{seed}_control_results.npz"
        summary = run_stitched_atlas(H, atlas, run_config, save_path=result_path)
        summary["result_path"] = str(result_path)
        summaries.append(summary)
        print(
            "seed", seed,
            "mean30", f"{summary['mean_last_30']:.6f}",
            "max30", f"{summary['max_last_30']:.6f}",
            "mean100", f"{summary['mean_last_100']:.6f}",
            "pred", f"{summary['mean_prediction_error_xy']:.6f}",
            "success", summary["success_30"] and summary["success_100"],
            flush=True,
        )
    payload = {
        "method": "stitched_fixed_local_affine_dynamics_atlas_mpc",
        "essential_difference_from_select_dpc": "The controller composes fixed learned one-step charts across its horizon and performs no trajectory retrieval.",
        "config": asdict(config),
        "fairness_audit": audit,
        "chart_sizes": sizes.tolist(),
        "fit_rmse": atlas["fit_rmse"].tolist(),
        "results": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Stitched local-dynamics Atlas predictive control")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--cluster-seed", type=int, default=7)
    parser.add_argument("--ridge", type=float, default=0.001)
    parser.add_argument("--schedule-iterations", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--near-horizon", type=int, default=3)
    parser.add_argument("--near-switch-distance", type=float, default=0.03)
    parser.add_argument("--goal-chart-candidates", type=int, default=400)
    parser.add_argument("--goal-charts", type=int, default=2)
    parser.add_argument("--q-data-goal-joint", type=float, default=5000.0)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_EVAL_SEEDS)
    parser.add_argument("--output-name", default="candidate")
    parser.add_argument("--reuse-atlas", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
