"""Contextual DPC with supplied eight-output references.

The current evaluation launcher is contextual_dpc_terminal_panel. The retained
development directory provides its frozen weight selection and comparison data.
"""
from dataclasses import dataclass, asdict
import json
import time

import numpy as np
from comparison_methods.contextual_dpc import reacher as base

cp, gym = base.cp, base.gym
output_from_obs, write_json = base.output_from_obs, base.write_json
true_error_metrics, strict_success, timing_stats = base.true_error_metrics, base.strict_success, base.timing_stats
ROOT = base.ROOT / "experiments/reacher/contextual_dpc_atlas_terminal8_v1"


@dataclass(frozen=True)
class Config(base.Config):
    q_xy: float = 100.
    q_joint: float = 10.
    q_dq: float = .1


class Library(base.Library):
    def __init__(self, u, y, config):
        super().__init__(u, y, config)
        if not np.isfinite([config.q_joint, config.q_dq]).all() or min(config.q_joint, config.q_dq) < 0:
            raise ValueError("Joint/velocity weights must be finite and nonnegative")
        self.q = np.r_[[config.q_joint]*4, [config.q_dq]*2, [config.q_xy]*2, np.zeros(2)]
        self.future_weight = np.tile(self.q, config.horizon)
        self.future_norm2 = np.sum(self.zf**2 * self.future_weight[:, None], axis=0)


def compute_action(lib, u_history, y_history, reference8, low, high):
    started = time.perf_counter()
    c = lib.config
    past_du = (np.diff(u_history[-3:], axis=0) - lib.du_mean) / lib.du_scale
    past_z = (np.c_[y_history[-2:], u_history[-2:]] - lib.z_mean) / lib.z_scale
    reference = np.zeros((c.horizon, 10))
    reference[:, :8] = reference8
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


def run_seed(lib, seed, steps, directory, reference8, reference_setup_s=0.):
    started = time.perf_counter()
    c = lib.config
    env = gym.make("Reacher-v5", max_episode_steps=c.warmup + steps)
    records, uh, yh = {}, [], []
    try:
        obs, _ = env.reset(seed=seed)
        initial = obs.copy()
        target = obs[4:6].copy()
        np.testing.assert_allclose(reference8[6:8], target, rtol=0, atol=1e-12)
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
            action, row = compute_action(lib, np.array(uh[-3:]), np.array(yh[-2:]), reference8,
                                         env.action_space.low, env.action_space.high)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or (truncated and step != steps - 1):
                raise RuntimeError("Unexpected environment termination")
            y = output_from_obs(obs)
            tracking = np.sum(lib.q[:8] * ((y - reference8) / lib.z_scale[:8])**2)
            effort = c.r * np.sum(((action - lib.u_mean) / lib.u_scale)**2)
            smoothing = c.r * np.sum(((action - previous) / lib.u_scale)**2)
            row.update(y=y, u=action, ref=reference8.copy(),
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
        "reference_setup_s": reference_setup_s,
        "reference_mode": "supplied_unguided_atlas_fixed_eight_output",
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
        np.savez_compressed(f, **a, initial_observation=initial, target_xy=target, final_goal=reference8,
            warmup_u=warmup_u, warmup_y=warmup_y, summary=np.array(json.dumps(summary)))
    temp.replace(path)
    return summary


def summarize(directory,seeds):
    rows, timing = [],{k:[] for k in ('control_compute_time_s','selection_time_s','matrix_preparation_time_s','solve_time_s')}
    for seed in seeds:
        path=directory/f'seed{seed}_control_results.npz'
        if not path.exists():continue
        with np.load(path) as a:
            row=json.loads(a['summary'].item())
            assert row['seed']==seed and len(a['distance'])==row['steps']
            np.testing.assert_array_equal(a['ref'],np.tile(a['final_goal'],(row['steps'],1)))
            rows.append(row)
            for key in timing:timing[key].extend(a[key][5:].tolist())
    report=dict(completed=len(rows),requested=len(seeds),complete=len(rows)==len(seeds),results=rows,
        successes=sum(r['strict_success'] for r in rows),
        success_rate=float(np.mean([r['strict_success'] for r in rows])) if rows else None,
        mean_final100_mm=float(1000*np.mean([r['mean_last_100'] for r in rows])) if rows else None,
        median_final100_mm=float(1000*np.median([r['mean_last_100'] for r in rows])) if rows else None,
        fallback_steps=sum(r['fallback_steps'] for r in rows),
        modular_timing_steps6_onwards={k:timing_stats(v) if v else None for k,v in timing.items()},
        note='Development seeds; reference loading excluded from per-step latency, separately recorded. No SVD. Native costs are not comparable across different output weights.')
    write_json(directory/'summary.json',report)
    return report
