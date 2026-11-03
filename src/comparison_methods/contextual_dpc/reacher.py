"""Equation-based contextual DPC (arXiv:2503.23890v2), adapted to Reacher.

No authors' implementation is available. See the companion notebook for the
norm/softmin ambiguities and the explicit incremental-input interpretation.
No SVD, projection regularization, Atlas control, routing, or refinement loop.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import cvxpy as cp
import gymnasium as gym
import numpy as np
from threadpoolctl import threadpool_limits

from comparison_methods.common.reacher import digest, write_json, output_from_obs, recover_episodes, windows, timing_stats
from atlas_deepc.common.metrics import strict_success, true_error_metrics

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/reacher/select_iid_offline_hankel.npy"
OUTPUT = ROOT / "experiments/reacher/contextual_dpc_raw_final_211_310_v1"
VERSION = "contextual_v2_reacher_incremental_raw_v1"


@dataclass(frozen=True)
class Config:
    tini: int = 2
    horizon: int = 10
    columns: int = 200
    warmup: int = 12
    q_xy: float = 1.0
    r: float = 0.1
    lambda_g_bar: float = 0.1
    lambda_sigma: float = 100.0
    max_iter: int = 1000
    tolerance: float = 1e-8
    accept_residual: float = 1e-4


class Library:
    def __init__(self, u, y, config):
        self.config = config
        if (config.tini, config.horizon) != (2, 10):
            raise ValueError("This protocol is fixed at T_ini=2, T_f=10")
        if config.columns < 1 or config.warmup < config.tini + 1:
            raise ValueError("Invalid column count or warmup")
        if min(config.q_xy, config.r, config.lambda_g_bar, config.lambda_sigma) < 0:
            raise ValueError("Weights must be nonnegative")
        if config.q_xy == 0:
            raise ValueError("A positive tracking weight is required")
        # Do not invent the pre-episode input. Drop sample zero in each episode.
        du = np.diff(u, axis=1)
        z = np.concatenate([y[:, 1:], u[:, 1:]], axis=-1)
        self.du_mean = du.mean(axis=(0, 1))
        self.du_scale = np.maximum(du.std(axis=(0, 1)), 1e-8)
        self.z_mean = z.mean(axis=(0, 1))
        self.z_scale = np.maximum(z.std(axis=(0, 1)), 1e-8)
        self.u_mean, self.u_scale = self.z_mean[8:], self.z_scale[8:]
        d = windows((du - self.du_mean) / self.du_scale, 12)
        zmat = windows((z - self.z_mean) / self.z_scale, 12)
        self.dp, self.df = d[:4], d[4:]
        self.zp, self.zf = zmat[:20], zmat[20:]
        self.q = np.r_[np.zeros(6), [config.q_xy] * 2, np.zeros(2)]
        if config.columns > d.shape[1]:
            raise ValueError("Not enough offline columns")
        self.future_weight = np.tile(self.q, config.horizon)
        self.past_norm2 = np.sum(self.zp**2, axis=0)
        self.future_norm2 = np.sum(self.zf**2 * self.future_weight[:, None], axis=0)
        self.meta = {"episodes": len(u), "samples_per_episode": u.shape[1],
                     "source_columns": len(u) * (u.shape[1] - 11),
                     "incremental_columns": d.shape[1], "dropped_first_sample_per_episode": True}

    @classmethod
    def from_path(cls, path, config):
        return cls(*recover_episodes(np.load(path, mmap_mode="r")), config)

    def select(self, past_z, ref):
        # Euclidean distances, NOT squared distances; separate population z-scores.
        p = np.sqrt(np.maximum(self.past_norm2 - 2 * past_z @ self.zp + past_z @ past_z, 0))
        wf = ref * self.future_weight
        f = np.sqrt(np.maximum(self.future_norm2 - 2 * wf @ self.zf + wf @ ref, 0))
        score = (p - p.mean()) / max(p.std(), 1e-12) + (f - f.mean()) / max(f.std(), 1e-12)
        # Stable column-index tie breaking; no softmin or random sampling.
        chosen = np.argsort(score, kind="stable")[:self.config.columns]
        return chosen, score[chosen]


def compute_action(lib, u_history, y_history, target, low, high):
    started = time.perf_counter()
    c = lib.config
    past_du = (np.diff(u_history[-3:], axis=0) - lib.du_mean) / lib.du_scale
    past_z = (np.c_[y_history[-2:], u_history[-2:]] - lib.z_mean) / lib.z_scale
    reference = np.zeros((c.horizon, 10))
    reference[:, 6:8] = target
    reference = (reference - lib.z_mean) / lib.z_scale
    selection_start = time.perf_counter()
    indices, scores = lib.select(past_z.ravel(), reference.ravel())
    selection_s = time.perf_counter() - selection_start
    setup_start = time.perf_counter()
    g = cp.Variable(c.columns)
    zp = lib.zp[:, indices] @ g
    zf = cp.reshape(lib.zf[:, indices] @ g, (c.horizon, 10), order="C")
    df = cp.reshape(lib.df[:, indices] @ g, (c.horizon, 2), order="C")
    increments = cp.multiply(df, lib.du_scale) + lib.du_mean
    actions = u_history[-1] + np.tril(np.ones((c.horizon, c.horizon))) @ increments
    predicted = cp.multiply(zf, lib.z_scale) + lib.z_mean
    slack = zp - past_z.ravel()
    # Explicit coordinate identity for the incremental-input adaptation:
    # the appended output is the actual applied actuator input, not a free signal.
    constraints = [lib.dp[:, indices] @ g == past_du.ravel(), cp.sum(g) == 1,
                   predicted[:, 8:] == actions, actions >= low, actions <= high]
    input_normalized = cp.multiply(actions - lib.u_mean, 1 / lib.u_scale)
    change_normalized = cp.multiply(increments, 1 / lib.u_scale)
    objective = (cp.sum_squares(cp.multiply(zf - reference, np.sqrt(lib.q)))
                 + c.r * (cp.sum_squares(input_normalized) + cp.sum_squares(change_normalized))
                 + c.lambda_g_bar * c.columns * cp.norm(g, 2)
                 + c.lambda_sigma * cp.norm(slack, 1))
    problem = cp.Problem(cp.Minimize(objective), constraints)
    construction_s = time.perf_counter() - setup_start
    solve_start = time.perf_counter()
    error = ""
    try:
        problem.solve(solver="CLARABEL", max_iter=c.max_iter, max_threads=1,
                      tol_gap_abs=c.tolerance, tol_gap_rel=c.tolerance,
                      tol_feas=c.tolerance, warm_start=False)
    except cp.error.SolverError as exc:
        error = str(exc)
    solve_s = time.perf_counter() - solve_start
    status = str(problem.status or "solver_error")
    residual = float("nan")
    accepted = not error and status in ("optimal", "optimal_inaccurate")
    if accepted:
        residual = float(max(np.max(item.violation()) for item in constraints))
        accepted = bool(np.isfinite(actions.value).all() and np.isfinite(g.value).all()
                        and residual <= c.accept_residual)
    if accepted:
        raw_action = np.asarray(actions.value[0])
        action = np.clip(raw_action, low, high)  # Numerical bound tolerance only.
        clip = float(np.max(np.abs(raw_action - action)))
        prediction, plan = predicted.value[:, :8].copy(), actions.value.copy()
        coefficients, slack_value = g.value.copy(), slack.value.copy()
    else:
        # Documented safety fallback, not claimed to be from the paper.
        action, clip = np.zeros(2), 0.0
        prediction, plan = np.full((c.horizon, 8), np.nan), np.full((c.horizon, 2), np.nan)
        coefficients, slack_value = np.full(c.columns, np.nan), np.full(20, np.nan)
    return action, {"selected_indices": indices, "selected_scores": scores,
        "g_coefficients": coefficients, "slack": slack_value, "planned_y": prediction,
        "pred": prediction[0], "planned_u": plan, "status": status, "solver_error": error,
        "fallback": not accepted, "max_constraint_violation": residual, "action_clip": clip,
        "objective": float(problem.value) if accepted else np.nan,
        "selection_time_s": selection_s, "matrix_preparation_time_s": construction_s,
        "solve_time_s": solve_s, "control_compute_time_s": time.perf_counter() - started}


def run_seed(lib, seed, steps, directory):
    started = time.perf_counter()
    c = lib.config
    env = gym.make("Reacher-v5", max_episode_steps=c.warmup + steps)
    records, uh, yh = {}, [], []
    try:
        obs, _ = env.reset(seed=seed)
        initial = obs.copy()
        target = obs[4:6].copy()
        for _ in range(c.warmup):
            action = np.zeros(2)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                raise RuntimeError("Environment ended during warmup")
            uh.append(action)
            yh.append(output_from_obs(obs))
        warmup_u, warmup_y = np.array(uh), np.array(yh)
        for step in range(steps):
            before = obs.copy()
            previous = uh[-1].copy()
            action, row = compute_action(lib, np.array(uh[-3:]), np.array(yh[-2:]), target,
                                         env.action_space.low, env.action_space.high)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or (truncated and step != steps - 1):
                raise RuntimeError("Unexpected environment termination")
            y = output_from_obs(obs)
            tracking = c.q_xy * np.sum(((y[6:8] - target) / lib.z_scale[6:8])**2)
            effort = c.r * np.sum(((action - lib.u_mean) / lib.u_scale)**2)
            smoothing = c.r * np.sum(((action - previous) / lib.u_scale)**2)
            row.update(y=y, u=action, ref=np.r_[np.zeros(6), target],
                y_before=output_from_obs(before), observation_before=before, observation_after=obs,
                distance=np.linalg.norm(y[6:8] - target),
                prediction_error_xy=np.linalg.norm(row["pred"][6:8] - y[6:8]),
                stage_tracking_cost=tracking, stage_input_cost=effort, stage_smoothing_cost=smoothing,
                stage_control_cost=tracking + effort + smoothing)
            for key, value in row.items():
                records.setdefault(key, []).append(value)
            uh.append(action.copy())
            yh.append(y)
            if (step + 1) % 20 == 0 or step == 0 or step == steps - 1:
                write_json(directory / "progress.json", {"state": "running", "seed": seed,
                    "step": step + 1, "steps": steps, "updated_unix": time.time()})
                print(f"seed {seed}, step {step+1}/{steps}: {row['distance']*1000:.2f} mm, {row['status']}, fallback={row['fallback']}", flush=True)
    finally:
        env.close()
    a = {key: np.array(value) for key, value in records.items()}
    metrics = true_error_metrics(a["distance"])
    prediction = a["prediction_error_xy"][np.isfinite(a["prediction_error_xy"])]
    summary = {"seed": seed, "steps": steps, **metrics,
        "strict_success": bool(steps >= 100 and strict_success(metrics)),
        "success_evaluable": steps >= 100, "fallback_steps": int(a["fallback"].sum()),
        "closed_loop_control_cost": float(a["stage_control_cost"].sum()),
        "closed_loop_reference_cost": float(a["stage_tracking_cost"].sum()),
        "closed_loop_input_cost": float(a["stage_input_cost"].sum()),
        "closed_loop_smoothing_cost": float(a["stage_smoothing_cost"].sum()),
        "xy_squared_error_sum_m2": float(np.sum(a["distance"]**2)),
        "input_squared_sum": float(np.sum(a["u"]**2)),
        "delta_input_squared_sum": float(np.sum(np.diff(np.vstack([warmup_u[-1], a["u"]]), axis=0)**2)),
        "mean_prediction_error_xy": float(prediction.mean()) if prediction.size else None,
        "timing": timing_stats(a["control_compute_time_s"]), "wall_seconds": time.perf_counter()-started}
    path = directory / f"seed{seed}_control_results.npz"
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **a, initial_observation=initial, target_xy=target,
            warmup_u=warmup_u, warmup_y=warmup_y, summary=np.array(json.dumps(summary)))
    temp.replace(path)
    return summary


def load_results(directory):
    rows, times = [], []
    for path in sorted(Path(directory).glob("seed*_control_results.npz")):
        with np.load(path) as a:
            rows.append(json.loads(a["summary"].item()))
            times.append(a["control_compute_time_s"].copy())
    return sorted(rows, key=lambda r: r["seed"]), times


def summarize(directory, seeds):
    rows, times = load_results(directory)
    if not rows:
        return {"completed": 0, "requested": len(seeds), "results": []}
    successes = sum(row["strict_success"] for row in rows)
    result = {"completed": len(rows), "requested": len(seeds), "complete": len(rows) == len(seeds),
        "successes": successes, "success_rate": successes / len(rows), "results": rows,
        "median_final100_error_mm": float(np.median([r["mean_last_100"] for r in rows])*1000),
        "median_closed_loop_control_cost": float(np.median([r["closed_loop_control_cost"] for r in rows])),
        "timing_all_steps": timing_stats(np.concatenate(times)),
        "timing_discard_first5": timing_stats(np.concatenate([t[5:] for t in times])) if all(len(t)>5 for t in times) else None}
    write_json(Path(directory) / "summary.json", result)
    return result


def run(config=Config(), seed_start=211, seed_stop=310, steps=300, output=OUTPUT, data=DATA):
    import fcntl
    output, data = Path(output).resolve(), Path(data).resolve()
    seeds = list(range(seed_start, seed_stop + 1))
    if not seeds or steps < 1:
        raise ValueError("Invalid seed range or steps")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+") as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signature = {"version": VERSION, "source_sha256": digest(__file__), "data_sha256": digest(data),
            "utilities_sha256": digest(ROOT / "src/comparison_methods/common/reacher.py"),
            "metrics_sha256": digest(ROOT / "src/atlas_deepc/common/metrics.py"),
            "config": asdict(config), "seeds": seeds, "steps": steps,
            "environment": "Reacher-v5", "solver": "CLARABEL", "workers": 1, "threads": 1,
            "reference": "fixed final Cartesian target; no graph or terminal posture",
            "svd": False, "selection": "deterministic standardized past/future distances",
            "norm_g": "unsquared L2", "norm_slack": "L1", "weights_status": "untuned Reacher starting values",
            "preprocessing": "unique training samples z-score; increments; y_post,u_current augmentation; no spatial alignment",
            "increment_identity": "u_predicted = u_last + cumulative_sum(delta_u); augmented future inputs equal u_predicted",
            "fallback": "zero physical input after a failed/rejected solve; no retries",
            "timing": "online normalization + selection + construction + solve + diagnostics; excludes simulation/logging/offline setup",
            "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "cvxpy", "clarabel", "gymnasium", "mujoco")}}
        manifest = output / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != signature:
                raise ValueError("Configuration/source/data changed. Use a new output directory; existing results were not modified.")
        else:
            if list(output.glob("seed*_control_results.npz")):
                raise ValueError("Unversioned results exist")
            write_json(manifest, signature)
        write_json(output / "session.json", {"pid": os.getpid(), "started_unix": time.time()})
        try:
            prep_start = time.perf_counter()
            lib = Library.from_path(data, config)
            np.savez_compressed(output / "normalization.npz", du_mean=lib.du_mean, du_scale=lib.du_scale,
                z_mean=lib.z_mean, z_scale=lib.z_scale, Q_diagonal=lib.q)
            write_json(output / "data_audit.json", {**lib.meta, "offline_setup_s": time.perf_counter()-prep_start})
            for seed in seeds:
                path = output / f"seed{seed}_control_results.npz"
                if path.exists():
                    with np.load(path) as a:
                        row = json.loads(a["summary"].item())
                    if row["seed"] != seed or row["steps"] != steps:
                        raise ValueError("Incompatible cached seed")
                else:
                    row = run_seed(lib, seed, steps, output)
                panel = summarize(output, seeds)
                print(f"{panel['completed']}/{len(seeds)} saved; {panel['successes']} successes; seed {seed} final100={1000*row['mean_last_100']:.2f} mm", flush=True)
            write_json(output / "progress.json", {"state": "complete", "completed": panel["completed"], "updated_unix": time.time()})
            return panel
        except BaseException as exc:
            write_json(output / "progress.json", {"state": "interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                "message": str(exc), "updated_unix": time.time()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=211)
    parser.add_argument("--seed-stop", type=int, default=310)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--q-xy", type=float, default=1.)
    parser.add_argument("--r", type=float, default=.1)
    parser.add_argument("--lambda-g-bar", type=float, default=.1)
    parser.add_argument("--lambda-sigma", type=float, default=100.)
    args = parser.parse_args()
    run(Config(q_xy=args.q_xy, r=args.r, lambda_g_bar=args.lambda_g_bar, lambda_sigma=args.lambda_sigma),
        args.seed_start, args.seed_stop, args.steps, args.output_dir)


if __name__ == "__main__":
    main()
