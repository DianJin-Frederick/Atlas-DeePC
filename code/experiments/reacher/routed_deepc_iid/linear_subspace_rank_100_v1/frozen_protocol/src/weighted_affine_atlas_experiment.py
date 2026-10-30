import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import cvxpy as cp
import gymnasium as gym
import numpy as np

import expert_pipeline_search as pipe
from local_affine_atlas_experiment import (
    DEFAULT_DATA_SEED,
    DEFAULT_EPISODE_LENGTH,
    DEFAULT_EVAL_SEEDS,
    audit_iid_library,
    environment_target,
)


EXPERIMENT_DIR = Path("experiments/weighted_affine_atlas_iid")
DEFAULT_DATA_PATH = Path("data/select_iid_offline_hankel.npy")


@dataclass
class WeightedAtlasConfig:
    clusters: int = 8
    chart_rank: int = 28
    cluster_seed: int = 7
    cluster_max_iter: int = 20
    coordinate_quantile: float = 0.005
    coordinate_expansion: float = 1.25
    gating_weight: float = 100.0
    coordinate_weight: float = 0.05
    q_xy: float = 100000.0
    q_dq: float = 20.0
    q_data_goal_joint: float = 5000.0
    r: float = 0.0001
    lam_ini: float = 100000.0
    lam_du: float = 5.0
    u_limit: float = 1.0
    steps: int = 320
    seed: int = 1


def fit_weighted_atlas(H, config):
    """Fit fixed affine K-subspaces and their bounded local coordinates."""
    fit = pipe.fit_affine_k_subspaces(
        H,
        n_clusters=config.clusters,
        rank=config.chart_rank,
        seed=config.cluster_seed,
        max_iter=config.cluster_max_iter,
        standardize_rows=True,
    )
    row_mean = fit["row_mean"]
    row_scale = fit["row_scale"]
    Z = (H - row_mean) / row_scale
    centers_z = []
    bases_z = []
    lower = []
    upper = []
    past_scale = []
    past_rows = np.r_[
        np.arange(pipe.M * pipe.T_INI),
        pipe.M * pipe.L + np.arange(pipe.P * pipe.T_INI),
    ]
    q = float(config.coordinate_quantile)
    if not 0.0 <= q < 0.5:
        raise ValueError("coordinate_quantile must be in [0, 0.5).")

    for cluster_id in range(config.clusters):
        idx = np.where(fit["labels"] == cluster_id)[0]
        X = Z[:, idx]
        center = X.mean(axis=1, keepdims=True)
        basis = pipe.fixed_rank_left_svd_basis(X - center, config.chart_rank)
        coords = basis.T @ (X - center)
        lo = np.quantile(coords, q, axis=1)
        hi = np.quantile(coords, 1.0 - q, axis=1)
        midpoint = 0.5 * (lo + hi)
        half_width = np.maximum(0.5 * (hi - lo) * config.coordinate_expansion, 1e-5)
        centers_z.append(center[:, 0])
        bases_z.append(basis)
        lower.append(midpoint - half_width)
        upper.append(midpoint + half_width)
        past_scale.append(np.maximum(X[past_rows].std(axis=1), 0.15))

    centers_z = np.asarray(centers_z)
    bases_z = np.asarray(bases_z)
    return {
        "labels": fit["labels"],
        "row_mean": row_mean[:, 0],
        "row_scale": row_scale[:, 0],
        "centers_z": centers_z,
        "bases_z": bases_z,
        "centers": row_mean[:, 0][None, :] + row_scale[:, 0][None, :] * centers_z,
        "bases": row_scale[:, 0][None, :, None] * bases_z,
        "lower": np.asarray(lower),
        "upper": np.asarray(upper),
        "past_scale": np.asarray(past_scale),
        "past_rows": past_rows,
        "objective_history": fit["objective_history"],
        "change_history": fit["change_history"],
    }


def save_atlas(path, atlas, config):
    np.savez(
        path,
        **atlas,
        config=np.array(json.dumps(asdict(config)), dtype=object),
    )


def load_atlas(path):
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files if key != "config"}


def data_derived_goal(H, target_xy, weight):
    if weight <= 0.0:
        return None, {}
    outputs = H[pipe.M * pipe.L:].reshape(pipe.L, pipe.P, H.shape[1])[0].T
    errors = np.linalg.norm(outputs[:, 6:8] - target_xy, axis=1)
    index = int(np.argmin(errors))
    windows_per_episode = DEFAULT_EPISODE_LENGTH - pipe.L + 1
    source_seed = DEFAULT_DATA_SEED + index // windows_per_episode
    reference = outputs[index].copy()
    reference[4:6] = 0.0
    reference[6:8] = target_xy
    metadata = {
        "data_goal_index": index,
        "data_goal_xy_error": float(errors[index]),
        "data_goal_source_seed": source_seed,
        "data_goal_source_target_distance": float(
            np.linalg.norm(environment_target(source_seed) - target_xy)
        ),
    }
    return reference, metadata


def normalized_gating_scores(atlas, u_ini, y_ini):
    past = np.concatenate([u_ini[:, 0], y_ini[:, 0]])
    rows = atlas["past_rows"].astype(int)
    past_z = (past - atlas["row_mean"][rows]) / atlas["row_scale"][rows]
    delta = (past_z[None, :] - atlas["centers_z"][:, rows]) / atlas["past_scale"]
    scores = np.mean(delta * delta, axis=1)
    scores -= scores.min()
    positive = scores[scores > 1e-12]
    scale = np.median(positive) if positive.size else 1.0
    return scores / max(float(scale), 1e-8)


class WeightedAtlasController:
    """Convex DeePC over a fixed family of bounded affine trajectory charts."""

    def __init__(self, atlas, config):
        self.atlas = atlas
        self.config = config
        K, ambient, rank = atlas["bases"].shape
        if ambient != (pipe.M + pipe.P) * pipe.L:
            raise ValueError(f"Atlas ambient dimension {ambient} does not match the DeePC horizon.")

        self.u_ini = cp.Parameter((pipe.M * pipe.T_INI, 1))
        self.y_ini = cp.Parameter((pipe.P * pipe.T_INI, 1))
        self.y_ref = cp.Parameter((pipe.P * pipe.T_F, 1))
        self.gating = cp.Parameter(K, nonneg=True)
        self.u = cp.Variable((pipe.M * pipe.T_F, 1))
        self.y = cp.Variable((pipe.P * pipe.T_F, 1))
        self.sigma = cp.Variable((pipe.P * pipe.T_INI, 1))
        self.weights = cp.Variable(K)
        self.coordinates = cp.Variable((K, rank))

        trajectory = 0
        constraints = [self.weights >= 0.0, cp.sum(self.weights) == 1.0]
        coordinate_cost = 0
        for k in range(K):
            ak = cp.reshape(self.coordinates[k], (rank, 1), order="F")
            trajectory += atlas["centers"][k, :, None] * self.weights[k]
            trajectory += atlas["bases"][k] @ ak
            constraints += [
                self.coordinates[k] >= atlas["lower"][k] * self.weights[k],
                self.coordinates[k] <= atlas["upper"][k] * self.weights[k],
            ]
            coordinate_cost += cp.quad_over_lin(self.coordinates[k], self.weights[k] + 1e-6)

        behavioral = cp.vstack([self.u_ini, self.u, self.y_ini + self.sigma, self.y])
        constraints += [
            behavioral == trajectory,
            self.u <= config.u_limit,
            self.u >= -config.u_limit,
        ]
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
        R = config.r * np.eye(pipe.M)
        cost = config.lam_ini * cp.sum_squares(self.sigma)
        cost += config.coordinate_weight * coordinate_cost
        cost += config.gating_weight * (self.gating @ self.weights)
        for j in range(pipe.T_F):
            yj = self.y[j * pipe.P:(j + 1) * pipe.P]
            rj = self.y_ref[j * pipe.P:(j + 1) * pipe.P]
            uj = self.u[j * pipe.M:(j + 1) * pipe.M]
            cost += cp.quad_form(yj - rj, Q) + cp.quad_form(uj, R)
        for j in range(pipe.T_F - 1):
            du = self.u[(j + 1) * pipe.M:(j + 2) * pipe.M] - self.u[j * pipe.M:(j + 1) * pipe.M]
            cost += config.lam_du * cp.sum_squares(du)
        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, u_ini, y_ini, y_ref, gating):
        self.u_ini.value = u_ini
        self.y_ini.value = y_ini
        self.y_ref.value = y_ref
        self.gating.value = gating
        try:
            self.problem.solve(solver=cp.CLARABEL, warm_start=True, verbose=False, max_iter=10000)
        except Exception as exc:
            return None, None, None, f"crashed: {exc}"
        if self.problem.status not in ("optimal", "optimal_inaccurate"):
            return None, None, None, self.problem.status
        return self.u.value, self.y.value, self.weights.value, self.problem.status


def run_weighted_atlas_deepc(H, atlas, config, save_path=None):
    controller = WeightedAtlasController(atlas, config)
    env = gym.make(pipe.REACHER_ENV_ID, max_episode_steps=config.steps + pipe.T_INI + 5)
    obs, _ = env.reset(seed=config.seed)
    target_xy = np.asarray(obs[4:6], dtype=float).copy()
    goal, goal_metadata = data_derived_goal(H, target_xy, config.q_data_goal_joint)
    reference_single = goal if goal is not None else np.r_[np.zeros(6), target_xy, np.zeros(2)]
    y_ref = np.tile(reference_single.reshape(pipe.P, 1), (pipe.T_F, 1))

    u_history = []
    y_history = []
    for _ in range(pipe.T_INI):
        action = np.zeros(pipe.M)
        obs, *_ = env.step(action)
        u_history.append(action)
        y_history.append(pipe.reacher_output_from_obs(obs))

    records = {
        "y": [],
        "u": [],
        "pred": [],
        "ref": [],
        "distance": [],
        "prediction_error_xy": [],
        "status": [],
        "chart_weights": [],
        "gating_scores": [],
    }
    for _ in range(config.steps):
        u_ini = np.asarray(u_history[-pipe.T_INI:], dtype=float).reshape(pipe.M * pipe.T_INI, 1)
        y_ini = np.asarray(y_history[-pipe.T_INI:], dtype=float).reshape(pipe.P * pipe.T_INI, 1)
        gating = normalized_gating_scores(atlas, u_ini, y_ini)
        u_solution, y_solution, weights, status = controller.solve(u_ini, y_ini, y_ref, gating)
        if u_solution is None:
            action = np.zeros(pipe.M)
            prediction = np.full(pipe.P, np.nan)
            weights = np.full(config.clusters, np.nan)
        else:
            action = np.clip(u_solution[:pipe.M, 0], -config.u_limit, config.u_limit)
            prediction = np.asarray(y_solution[:pipe.P, 0]).reshape(pipe.P)
        obs, *_ = env.step(action)
        y_now = pipe.reacher_output_from_obs(obs)
        u_history.append(action.copy())
        y_history.append(y_now.copy())
        records["y"].append(y_now)
        records["u"].append(action)
        records["pred"].append(prediction)
        records["ref"].append(reference_single)
        records["distance"].append(float(np.linalg.norm(y_now[6:8] - target_xy)))
        records["prediction_error_xy"].append(float(np.linalg.norm(prediction[6:8] - y_now[6:8])))
        records["status"].append(status)
        records["chart_weights"].append(weights)
        records["gating_scores"].append(gating)
    env.close()

    for key in records:
        records[key] = np.asarray(records[key], dtype=object if key == "status" else None)
    distance = records["distance"]
    tail_30 = distance[-30:]
    tail_100 = distance[-min(100, distance.size):]
    finite_weights = np.nan_to_num(records["chart_weights"], nan=0.0)
    active = np.sum(finite_weights > 0.01, axis=1)
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
        "mean_active_charts": float(active.mean()),
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
    config = WeightedAtlasConfig(
        clusters=args.clusters,
        chart_rank=args.chart_rank,
        cluster_seed=args.cluster_seed,
        cluster_max_iter=args.cluster_max_iter,
        coordinate_quantile=args.coordinate_quantile,
        coordinate_expansion=args.coordinate_expansion,
        gating_weight=args.gating_weight,
        coordinate_weight=args.coordinate_weight,
        q_data_goal_joint=args.q_data_goal_joint,
        steps=args.steps,
    )
    atlas_path = output_dir / "weighted_affine_atlas.npz"
    if args.reuse_atlas and args.reuse_atlas.exists():
        atlas = load_atlas(args.reuse_atlas)
    else:
        atlas = fit_weighted_atlas(H, config)
        save_atlas(atlas_path, atlas, config)
    print("cluster sizes", np.bincount(atlas["labels"].astype(int), minlength=config.clusters).tolist(), flush=True)
    print("cluster change", atlas["change_history"].tolist(), flush=True)

    summaries = []
    for seed in args.seeds:
        run_config = WeightedAtlasConfig(**{**asdict(config), "seed": seed})
        result_path = output_dir / f"seed{seed}_control_results.npz"
        summary = run_weighted_atlas_deepc(H, atlas, run_config, save_path=result_path)
        summary["result_path"] = str(result_path)
        summaries.append(summary)
        print(
            "seed", seed,
            "mean30", f"{summary['mean_last_30']:.6f}",
            "max30", f"{summary['max_last_30']:.6f}",
            "mean100", f"{summary['mean_last_100']:.6f}",
            "active", f"{summary['mean_active_charts']:.2f}",
            "success", summary["success_30"] and summary["success_100"],
            flush=True,
        )
    payload = {
        "method": "fixed_affine_k_subspaces_with_convex_perspective_chart_blending",
        "essential_difference_from_select_dpc": "No online trajectory or nearest-neighbor retrieval; a fixed offline atlas is optimized through simplex chart weights.",
        "config": asdict(config),
        "fairness_audit": audit,
        "cluster_sizes": np.bincount(atlas["labels"].astype(int), minlength=config.clusters).tolist(),
        "results": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Convex chart-weighted affine Atlas-DeePC experiment")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--chart-rank", type=int, default=28)
    parser.add_argument("--cluster-seed", type=int, default=7)
    parser.add_argument("--cluster-max-iter", type=int, default=20)
    parser.add_argument("--coordinate-quantile", type=float, default=0.005)
    parser.add_argument("--coordinate-expansion", type=float, default=1.25)
    parser.add_argument("--gating-weight", type=float, default=100.0)
    parser.add_argument("--coordinate-weight", type=float, default=0.05)
    parser.add_argument("--q-data-goal-joint", type=float, default=5000.0)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_EVAL_SEEDS)
    parser.add_argument("--output-name", default="candidate")
    parser.add_argument("--reuse-atlas", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
