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


EXPERIMENT_DIR = Path("experiments/local_affine_iid")
DEFAULT_DATA_PATH = Path("data/select_iid_offline_hankel.npy")
DEFAULT_EVAL_SEEDS = [1, 2, 3, 4, 5, 7, 8, 42]
DEFAULT_DATA_SEED = 2026
DEFAULT_EPISODE_LENGTH = 200


@dataclass
class LocalAffineConfig:
    neighbors: int = 100
    g_lower_bound: float = -0.002
    g_negative_weight: float = 0.0
    g_lower_bound_far: float | None = None
    g_switch_distance: float = 0.05
    q_xy: float = 100000.0
    q_dq: float = 20.0
    q_data_goal_joint: float = 5000.0
    r: float = 0.0001
    lam_g: float = 0.05
    lam_ini: float = 100000.0
    lam_du: float = 5.0
    u_limit: float = 1.0
    steps: int = 320
    seed: int = 1


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


def run_local_affine_deepc(H, config, save_path=None):
    pipe.T_INI = 2
    pipe.T_F = 10
    pipe.L = 12
    U_p, U_f, Y_p, Y_f = pipe.partition_hankel_rows(H)
    u_scale = np.maximum(U_p.std(axis=1), 1e-6)
    y_scale = np.maximum(Y_p.std(axis=1), 1e-6)
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
    controller = pipe.DeePCController(
        Q,
        config.r * np.eye(pipe.M),
        np.zeros((pipe.P, pipe.P)),
        config.lam_g,
        config.lam_ini,
        config.lam_du,
        config.u_limit,
        cp.CLARABEL,
    )
    env = gym.make(pipe.REACHER_ENV_ID, max_episode_steps=config.steps + pipe.T_INI + 5)
    obs, _ = env.reset(seed=config.seed)
    target_xy = np.asarray(obs[4:6], dtype=float).copy()
    data_goal = None
    data_goal_index = None
    data_goal_xy_error = None
    data_goal_source_seed = None
    data_goal_source_target_distance = None
    if config.q_data_goal_joint > 0.0:
        output_samples = H[pipe.M * pipe.L:].reshape(pipe.L, pipe.P, H.shape[1])[0].T
        xy_error = np.linalg.norm(output_samples[:, 6:8] - target_xy, axis=1)
        data_goal_index = int(np.argmin(xy_error))
        data_goal_xy_error = float(xy_error[data_goal_index])
        windows_per_episode = DEFAULT_EPISODE_LENGTH - pipe.L + 1
        data_goal_source_seed = DEFAULT_DATA_SEED + data_goal_index // windows_per_episode
        data_goal_source_target_distance = float(
            np.linalg.norm(environment_target(data_goal_source_seed) - target_xy)
        )
        data_goal = output_samples[data_goal_index].copy()
        data_goal[4:6] = 0.0
        data_goal[6:8] = target_xy
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
        "ref": [],
        "pred": [],
        "distance": [],
        "prediction_error_xy": [],
        "status": [],
        "g_min": [],
        "g_lower_bound": [],
        "selected_past_score_mean": [],
        "selected_indices": [],
    }
    for _ in range(config.steps):
        u_ini = np.asarray(u_history[-pipe.T_INI:], dtype=float).reshape(pipe.M * pipe.T_INI, 1)
        y_ini = np.asarray(y_history[-pipe.T_INI:], dtype=float).reshape(pipe.P * pipe.T_INI, 1)
        past_score = np.mean(((Y_p.T - y_ini[:, 0]) / y_scale) ** 2, axis=1)
        past_score += 0.1 * np.mean(((U_p.T - u_ini[:, 0]) / u_scale) ** 2, axis=1)
        selected = np.argpartition(past_score, config.neighbors)[:config.neighbors]
        current_distance = float(np.linalg.norm(y_history[-1][6:8] - obs[4:6]))
        active_g_lower_bound = config.g_lower_bound
        if config.g_lower_bound_far is not None and current_distance > config.g_switch_distance:
            active_g_lower_bound = config.g_lower_bound_far
        local_chart = {
            "offset": np.zeros((H.shape[0], 1)),
            "basis": H[:, selected],
            "g_sum": 1.0,
            "g_min": active_g_lower_bound,
            "g_negative_weight": config.g_negative_weight,
        }
        reference = (
            np.tile(data_goal.reshape(pipe.P, 1), (pipe.T_F, 1))
            if data_goal is not None
            else pipe.make_ref(obs, use_ik_ref=False)
        )
        u_solution, y_solution, g_solution, status, residual = controller.solve(
            local_chart, u_ini, y_ini, reference
        )
        if u_solution is None:
            action = np.zeros(pipe.M)
            prediction = np.full(pipe.P, np.nan)
            minimum_g = np.nan
        else:
            action = np.clip(u_solution[:pipe.M, 0], -config.u_limit, config.u_limit)
            prediction = np.asarray(y_solution[:pipe.P, 0]).reshape(pipe.P)
            minimum_g = float(np.min(g_solution))
        obs, *_ = env.step(action)
        y_now = pipe.reacher_output_from_obs(obs)
        u_history.append(action.copy())
        y_history.append(y_now.copy())
        distance = float(np.linalg.norm(y_now[6:8] - obs[4:6]))
        prediction_error = float(np.linalg.norm(prediction[6:8] - y_now[6:8]))
        records["y"].append(y_now)
        records["u"].append(action)
        records["ref"].append(reference[:pipe.P, 0])
        records["pred"].append(prediction)
        records["distance"].append(distance)
        records["prediction_error_xy"].append(prediction_error)
        records["status"].append(status)
        records["g_min"].append(minimum_g)
        records["g_lower_bound"].append(active_g_lower_bound)
        records["selected_past_score_mean"].append(float(past_score[selected].mean()))
        records["selected_indices"].append(selected)
    env.close()

    for key in records:
        dtype = object if key == "status" else None
        records[key] = np.asarray(records[key], dtype=dtype)
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
        "statuses": {str(s): int(np.sum(records["status"] == s)) for s in np.unique(records["status"])},
        "feedback_blend": 0.0,
        "feedback_fallback": False,
        "use_ik_ref": False,
        "data_goal_index": data_goal_index,
        "data_goal_xy_error": data_goal_xy_error,
        "data_goal_source_seed": data_goal_source_seed,
        "data_goal_source_target_distance": data_goal_source_target_distance,
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
    summaries = []
    for seed in args.seeds:
        config = LocalAffineConfig(
            neighbors=args.neighbors,
            g_lower_bound=args.g_lower_bound,
            g_negative_weight=args.g_negative_weight,
            g_lower_bound_far=args.g_lower_bound_far,
            g_switch_distance=args.g_switch_distance,
            q_data_goal_joint=args.q_data_goal_joint,
            steps=args.steps,
            seed=seed,
        )
        result_path = output_dir / f"seed{seed}_control_results.npz"
        summary = run_local_affine_deepc(H, config, save_path=result_path)
        summary["result_path"] = str(result_path)
        summaries.append(summary)
        print(
            "seed", seed,
            "mean30", f"{summary['mean_last_30']:.6f}",
            "max30", f"{summary['max_last_30']:.6f}",
            "mean100", f"{summary['mean_last_100']:.6f}",
            "success30", summary["success_30"],
            "success100", summary["success_100"],
            flush=True,
        )
    payload = {"fairness_audit": audit, "results": summaries}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive local-affine Atlas-DeePC experiment")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--neighbors", type=int, default=100)
    parser.add_argument("--g-lower-bound", type=float, default=-0.002)
    parser.add_argument("--g-negative-weight", type=float, default=0.0)
    parser.add_argument("--g-lower-bound-far", type=float)
    parser.add_argument("--g-switch-distance", type=float, default=0.05)
    parser.add_argument("--q-data-goal-joint", type=float, default=5000.0)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_EVAL_SEEDS)
    parser.add_argument("--output-name", default="candidate")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
