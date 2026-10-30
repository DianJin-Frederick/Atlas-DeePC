import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import cvxpy as cp
import gymnasium as gym
import numpy as np
from sklearn.cluster import MiniBatchKMeans

import expert_pipeline_search as pipe
from local_affine_atlas_experiment import DEFAULT_EVAL_SEEDS, audit_iid_library
from weighted_affine_atlas_experiment import data_derived_goal


EXPERIMENT_DIR = Path("experiments/gated_chart_atlas_iid")
DEFAULT_DATA_PATH = Path("data/select_iid_offline_hankel.npy")


@dataclass
class GatedAtlasConfig:
    clusters: int = 64
    chart_rank: int = 80
    cluster_seed: int = 7
    coordinate_quantile: float = 0.002
    coordinate_expansion: float = 1.5
    candidate_charts: int = 4
    selection_gate_weight: float = 100000.0
    q_xy: float = 100000.0
    q_dq: float = 20.0
    q_data_goal_joint: float = 5000.0
    r: float = 0.0001
    lam_coordinate: float = 0.05
    lam_ini: float = 100000.0
    lam_du: float = 5.0
    u_limit: float = 1.0
    steps: int = 320
    seed: int = 1


def fit_gated_atlas(H, config):
    """Learn fixed state-conditioned cells, then an affine SVD model in each cell."""
    U_p, _, Y_p, _ = pipe.partition_hankel_rows(H)
    past = np.vstack([U_p, Y_p])
    past_mean = past.mean(axis=1)
    past_scale = np.maximum(past.std(axis=1), 1e-8)
    past_z = ((past - past_mean[:, None]) / past_scale[:, None]).T
    gate = MiniBatchKMeans(
        n_clusters=config.clusters,
        random_state=config.cluster_seed,
        n_init=5,
        batch_size=2048,
        max_iter=300,
        reassignment_ratio=0.005,
    )
    labels = gate.fit_predict(past_z)

    row_mean = H.mean(axis=1)
    row_scale = np.maximum(H.std(axis=1), 1e-8)
    Z = (H - row_mean[:, None]) / row_scale[:, None]
    centers = []
    bases = []
    lower = []
    upper = []
    q = config.coordinate_quantile
    for cluster_id in range(config.clusters):
        idx = np.where(labels == cluster_id)[0]
        if idx.size <= config.chart_rank:
            raise RuntimeError(
                f"Chart {cluster_id} has {idx.size} samples, not enough for rank {config.chart_rank}."
            )
        X = Z[:, idx]
        center_z = X.mean(axis=1, keepdims=True)
        basis_z = pipe.fixed_rank_left_svd_basis(X - center_z, config.chart_rank)
        coordinates = basis_z.T @ (X - center_z)
        lo = np.quantile(coordinates, q, axis=1)
        hi = np.quantile(coordinates, 1.0 - q, axis=1)
        midpoint = 0.5 * (lo + hi)
        half_width = np.maximum(0.5 * (hi - lo) * config.coordinate_expansion, 1e-5)
        centers.append(row_mean + row_scale * center_z[:, 0])
        bases.append(row_scale[:, None] * basis_z)
        lower.append(midpoint - half_width)
        upper.append(midpoint + half_width)
    return {
        "labels": labels,
        "gate_centers": gate.cluster_centers_,
        "past_mean": past_mean,
        "past_scale": past_scale,
        "centers": np.asarray(centers),
        "bases": np.asarray(bases),
        "lower": np.asarray(lower),
        "upper": np.asarray(upper),
    }


def save_atlas(path, atlas, config):
    np.savez(path, **atlas, config=np.array(json.dumps(asdict(config)), dtype=object))


def load_atlas(path):
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files if key != "config"}


def chart_scores(atlas, u_ini, y_ini):
    past = np.concatenate([u_ini[:, 0], y_ini[:, 0]])
    z = (past - atlas["past_mean"]) / atlas["past_scale"]
    delta = atlas["gate_centers"] - z[None, :]
    return np.mean(delta * delta, axis=1)


class GatedChartController:
    """One parameterized QP reused for every fixed atlas chart."""

    def __init__(self, config):
        ambient = (pipe.M + pipe.P) * pipe.L
        rank = config.chart_rank
        self.u_ini = cp.Parameter((pipe.M * pipe.T_INI, 1))
        self.y_ini = cp.Parameter((pipe.P * pipe.T_INI, 1))
        self.y_ref = cp.Parameter((pipe.P * pipe.T_F, 1))
        self.center = cp.Parameter((ambient, 1))
        self.basis = cp.Parameter((ambient, rank))
        self.lower = cp.Parameter(rank)
        self.upper = cp.Parameter(rank)
        self.u = cp.Variable((pipe.M * pipe.T_F, 1))
        self.y = cp.Variable((pipe.P * pipe.T_F, 1))
        self.sigma = cp.Variable((pipe.P * pipe.T_INI, 1))
        self.coordinates = cp.Variable((rank, 1))

        behavioral = cp.vstack([self.u_ini, self.u, self.y_ini + self.sigma, self.y])
        constraints = [
            behavioral == self.center + self.basis @ self.coordinates,
            self.coordinates[:, 0] >= self.lower,
            self.coordinates[:, 0] <= self.upper,
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
        cost += config.lam_coordinate * cp.sum_squares(self.coordinates)
        for j in range(pipe.T_F):
            yj = self.y[j * pipe.P:(j + 1) * pipe.P]
            rj = self.y_ref[j * pipe.P:(j + 1) * pipe.P]
            uj = self.u[j * pipe.M:(j + 1) * pipe.M]
            cost += cp.quad_form(yj - rj, Q) + cp.quad_form(uj, R)
        for j in range(pipe.T_F - 1):
            du = self.u[(j + 1) * pipe.M:(j + 2) * pipe.M] - self.u[j * pipe.M:(j + 1) * pipe.M]
            cost += config.lam_du * cp.sum_squares(du)
        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, atlas, chart, u_ini, y_ini, y_ref):
        self.u_ini.value = u_ini
        self.y_ini.value = y_ini
        self.y_ref.value = y_ref
        self.center.value = atlas["centers"][chart, :, None]
        self.basis.value = atlas["bases"][chart]
        self.lower.value = atlas["lower"][chart]
        self.upper.value = atlas["upper"][chart]
        try:
            self.problem.solve(solver=cp.CLARABEL, warm_start=True, verbose=False, max_iter=10000)
        except Exception as exc:
            return None, None, None, f"crashed: {exc}"
        if self.problem.status not in ("optimal", "optimal_inaccurate"):
            return None, None, None, self.problem.status
        return self.u.value, self.y.value, float(self.problem.value), self.problem.status


def run_gated_atlas_deepc(H, atlas, config, save_path=None):
    controller = GatedChartController(config)
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
        "y": [], "u": [], "pred": [], "ref": [], "distance": [],
        "prediction_error_xy": [], "status": [], "chart": [], "chart_score": [],
    }
    for _ in range(config.steps):
        u_ini = np.asarray(u_history[-pipe.T_INI:], dtype=float).reshape(pipe.M * pipe.T_INI, 1)
        y_ini = np.asarray(y_history[-pipe.T_INI:], dtype=float).reshape(pipe.P * pipe.T_INI, 1)
        scores = chart_scores(atlas, u_ini, y_ini)
        ranked = np.argsort(scores)
        u_solution = None
        y_solution = None
        status = "not_solved"
        chart = -1
        best = None
        score_scale = max(float(np.median(scores)), 1e-8)
        for candidate in ranked[:max(1, config.candidate_charts)]:
            candidate_u, candidate_y, objective, candidate_status = controller.solve(
                atlas, int(candidate), u_ini, y_ini, y_ref
            )
            if candidate_u is None:
                status = candidate_status
                continue
            selection_cost = objective + config.selection_gate_weight * scores[candidate] / score_scale
            if best is None or selection_cost < best[0]:
                best = (
                    selection_cost,
                    int(candidate),
                    candidate_u.copy(),
                    candidate_y.copy(),
                    candidate_status,
                )
        if best is not None:
            _, chart, u_solution, y_solution, status = best
        if u_solution is None:
            action = np.zeros(pipe.M)
            prediction = np.full(pipe.P, np.nan)
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
        records["chart"].append(chart)
        records["chart_score"].append(float(scores[chart]) if chart >= 0 else np.nan)
    env.close()

    for key in records:
        records[key] = np.asarray(records[key], dtype=object if key == "status" else None)
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
        "chart_switches": int(np.sum(records["chart"][1:] != records["chart"][:-1])),
        "charts_used": int(np.unique(records["chart"][records["chart"] >= 0]).size),
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
    config = GatedAtlasConfig(
        clusters=args.clusters,
        chart_rank=args.chart_rank,
        cluster_seed=args.cluster_seed,
        coordinate_quantile=args.coordinate_quantile,
        coordinate_expansion=args.coordinate_expansion,
        candidate_charts=args.candidate_charts,
        selection_gate_weight=args.selection_gate_weight,
        q_data_goal_joint=args.q_data_goal_joint,
        steps=args.steps,
    )
    atlas_path = output_dir / "gated_chart_atlas.npz"
    if args.reuse_atlas and args.reuse_atlas.exists():
        atlas = load_atlas(args.reuse_atlas)
    else:
        atlas = fit_gated_atlas(H, config)
        save_atlas(atlas_path, atlas, config)
    sizes = np.bincount(atlas["labels"].astype(int), minlength=config.clusters)
    print("chart sizes", int(sizes.min()), float(np.median(sizes)), int(sizes.max()), flush=True)

    summaries = []
    for seed in args.seeds:
        run_config = GatedAtlasConfig(**{**asdict(config), "seed": seed})
        result_path = output_dir / f"seed{seed}_control_results.npz"
        summary = run_gated_atlas_deepc(H, atlas, run_config, save_path=result_path)
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
        "method": "offline_learned_voronoi_atlas_with_hard_gating_and_local_svd_deepc",
        "essential_difference_from_select_dpc": "Online control selects one fixed learned chart; it never retrieves or rebuilds a neighborhood of trajectories.",
        "config": asdict(config),
        "fairness_audit": audit,
        "chart_sizes": sizes.tolist(),
        "results": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Fixed-gate local-SVD Atlas-DeePC experiment")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--chart-rank", type=int, default=80)
    parser.add_argument("--cluster-seed", type=int, default=7)
    parser.add_argument("--coordinate-quantile", type=float, default=0.002)
    parser.add_argument("--coordinate-expansion", type=float, default=1.5)
    parser.add_argument("--candidate-charts", type=int, default=4)
    parser.add_argument("--selection-gate-weight", type=float, default=100000.0)
    parser.add_argument("--q-data-goal-joint", type=float, default=5000.0)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_EVAL_SEEDS)
    parser.add_argument("--output-name", default="candidate")
    parser.add_argument("--reuse-atlas", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
