"""Unmodified upstream Select-DPC, with a final-target evaluation/logging adapter.

No Atlas controller is run. Upstream source and notebook definitions are loaded
verbatim. Only the requested solver substitution changes controller execution.
Run with python scripts/run_select_dpc.py after the optional external setup.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from unittest.mock import patch

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/select_dpc_upstream_mpl")

import cvxpy as cp
import gymnasium as gym
import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.common.metrics import strict_success, true_error_metrics
from atlas_deepc.control.expert_pipeline_search import closed_loop_control_cost
from comparison_methods.select_dpc import dependency as select_dpc_dependency

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = select_dpc_dependency.source_dir()
COMMIT = select_dpc_dependency.COMMIT
OUTPUT = ROOT / "experiments/reacher/select_dpc_upstream_final_clarabel_211_310_t300_v1"
# Upstream: cosines, sines, fingertip xy, velocities. Saved baseline: velocities before xy.
TO_BASELINE = np.array([0, 1, 2, 3, 6, 7, 4, 5])
VERSION = "upstream_final_reference_v1"
CLARABEL_OPTIONS = {"max_threads": 1, "static_regularization_constant": 1e-6}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def load_upstream():
    if not UPSTREAM.is_dir():
        raise RuntimeError(select_dpc_dependency.setup_hint())
    revision = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if revision != COMMIT:
        raise ValueError(f"Unexpected upstream revision {revision}")
    dirty = subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty:
        raise ValueError("Upstream tracked files must remain unchanged")
    sys.path.insert(0, str(UPSTREAM))
    from select_deepc import deepc_controller as dc
    from select_deepc import deepc_utils as utils
    # Extract just these definitions, without executing any notebook experiment cells.
    names = {"setup_DeePC", "fix_dataset", "DefaultReacherSetpoint"}
    namespace = {**vars(dc), **vars(utils), "np": np}
    found = set()
    notebook = json.loads((UPSTREAM / "reacher_simulation.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if not any(f"def {name}(" in source or f"class {name}(" in source for name in names):
            continue
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(UPSTREAM / "reacher_simulation.ipynb"), "exec"), namespace)
                found.add(node.name)
    if found != names:
        raise ValueError(f"Missing original definitions: {names - found}")
    dataset = utils.load_data_from_folder(str(UPSTREAM / "data/reacher/dataset_iid"), "iid")
    namespace["fix_dataset"](dataset)
    args = namespace["setup_DeePC"](dataset, enable_measurement_constraint=True)
    assert (args.deepc_dims.p, args.deepc_dims.m, args.deepc_dims.T_past, args.deepc_dims.T_fut) == (8, 2, 2, 15)
    return dc, args, namespace["DefaultReacherSetpoint"]


def audit_baseline_data(dataset):
    """Verify original episodes against existing Atlas data, allowing episode permutation.

    Rebuild L=12 solely for this read-only audit. Actual Select-DPC retains L=17.
    """
    baseline_path = ROOT / "data/reacher/select_iid_offline_hankel.npy"
    h = np.load(baseline_path, mmap_mode="r")
    n_windows = 189
    if h.shape != (144, len(dataset.dataset) * n_windows):
        return {"matches_baseline": False, "reason": "Unexpected baseline shape", "shape": list(h.shape)}
    first_inputs = h[:24, ::n_windows]
    used, maximum = [], 0.
    for trajectory in dataset.dataset:
        u = trajectory.input_trajectory
        y = trajectory.state_trajectory[:, TO_BASELINE]
        y = np.column_stack([y, y[:, 0]*y[:, 1]-y[:, 2]*y[:, 3], y[:, 2]*y[:, 1]+y[:, 0]*y[:, 3]])
        episode = int(np.argmin(np.max(np.abs(first_inputs - u[:12].reshape(-1, 1)), axis=0)))
        used.append(episode)
        hu = np.stack([u[t:t+12].ravel() for t in range(n_windows)], axis=1)
        hy = np.stack([y[t:t+12].ravel() for t in range(n_windows)], axis=1)
        maximum = max(maximum, float(np.max(np.abs(h[:, episode*n_windows:(episode+1)*n_windows] - np.vstack([hu, hy])))))
    return {"matches_baseline": maximum < 1e-12 and len(set(used)) == len(used),
            "max_absolute_difference": maximum, "baseline_sha256": digest(baseline_path),
            "nearest_episode_indices_in_baseline": used,
            "note": "Comparison at L=12 only; a false match is expected when using the authors' original dataset. Select-DPC retains L=17."}


def native_output(obs):
    obs = np.asarray(obs).copy()
    obs[4:6] += obs[8:10]
    return obs[:8]


class Instrumentation:
    """Observe original calls; never change selections, objectives, or fallback actions."""
    def __init__(self, dc):
        self.dc = dc
        self.reset()

    def reset(self):
        self.selection = []
        self.construction = []
        self.solve = []
        self.violations = []
        self.inner = []

    def selector(self, original):
        def observed(*args, **kwargs):
            started = time.perf_counter()
            indices, scores = original(*args, **kwargs)
            duration = time.perf_counter() - started
            self.selection.append((duration, indices[:200].copy(), scores[indices[:200]].copy()))
            return indices, scores
        return observed

    def install(self, stack):
        setup = self.dc.DataDrivenPredictiveController._setup_solver
        compute = self.dc.DataDrivenPredictiveController.compute_action
        solve = cp.Problem.solve

        def observed_setup(instance, *args, **kwargs):
            assert kwargs.get("decompose_hankel_matrix", args[-1] if args else False) is False
            started = time.perf_counter()
            result = setup(instance, *args, **kwargs)
            self.construction.append(time.perf_counter() - started)
            assert instance._g.shape == (200,)
            return result

        def clarabel_solve(problem, *args, **kwargs):
            assert kwargs["solver"] == "OSQP"
            kwargs["solver"] = "CLARABEL"
            # Stabilize Clarabel's KKT factorization, not the control objective.
            # Its default 1e-8 produces NumericalError on the first seed's QP.
            kwargs.update(CLARABEL_OPTIONS)
            # OSQP-specific tolerances have no exact cross-solver equivalence.
            # Use Clarabel defaults; retain original max_iter=6000/warm_start=False.
            kwargs.pop("eps_abs")
            kwargs.pop("eps_rel")
            started = time.perf_counter()
            try:
                return solve(problem, *args, **kwargs)
            finally:
                self.solve.append(time.perf_counter() - started)
                # Observe before upstream shifts its past-input parameter for
                # the next solve, which otherwise makes residuals meaningless.
                self.violations.append(float(max(np.max(c.violation()) for c in problem.constraints))
                                       if problem.status in ("optimal", "optimal_inaccurate") else np.nan)

        def observed_compute(instance, *args, **kwargs):
            action = compute(instance, *args, **kwargs)
            status = str(instance._problem.status or "failed")
            good = status in ("optimal", "optimal_inaccurate")
            self.inner.append({
                "status": status,
                "g": instance._g.value.copy() if good else np.full(200, np.nan),
                "pred": instance._y.value.reshape(15, 8).copy() if good else np.full((15, 8), np.nan),
                "planned_u": instance._u.value.reshape(15, 2).copy() if good else np.full((15, 2), np.nan),
                "objective": float(instance._problem.value) if good else np.nan,
                "max_constraint_violation": self.violations[-1],
            })
            return action
        stack.enter_context(patch.object(self.dc.DataDrivenPredictiveController, "_setup_solver", observed_setup))
        stack.enter_context(patch.object(self.dc.DataDrivenPredictiveController, "compute_action", observed_compute))
        stack.enter_context(patch.object(cp.Problem, "solve", clarabel_solve))


def timing_stats(seconds):
    ms = 1000*np.asarray(seconds)
    return {"mean_ms": float(ms.mean()), "median_ms": float(np.median(ms)),
            "p95_ms": float(np.percentile(ms, 95)), "p99_ms": float(np.percentile(ms, 99)),
            "max_ms": float(ms.max())}


def run_seed(dc, args, scheduler_class, seed, steps, output, progress):
    watch = Instrumentation(dc)
    started = time.perf_counter()
    controller = dc.SelectDeePC(args, watch.selector(dc.LkSelector()), 200, 5)
    setup_seconds = time.perf_counter() - started
    scheduler = scheduler_class()
    env = gym.make("Reacher-v4", max_episode_steps=steps)
    obs, _ = env.reset(seed=seed)
    initial_obs = obs.copy()
    target = obs[4:6].copy()
    records = {k: [] for k in (
        "y", "u", "pred", "ref", "distance", "prediction_error_xy", "status", "g_coefficients", "g_min",
        "observation_before", "observation_after", "y_before", "planned_y", "planned_u",
        "selected_indices", "selected_scores", "inner_status", "inner_g", "inner_objective", "inner_max_constraint_violation",
        "inner_selection_time_s", "inner_construction_time_s", "inner_solve_time_s",
        "selection_time_s", "localization_time_s", "matrix_preparation_time_s", "solve_time_s", "control_compute_time_s",
        "stage_tracking_cost", "stage_input_cost", "stage_control_cost", "qp_objective",
    )}
    start = time.perf_counter()
    try:
        with ExitStack() as stack:
            watch.install(stack)
            for step in range(steps):
                watch.reset()
                control_start = time.perf_counter()
                reference = np.asarray(scheduler(obs), dtype=float)
                state = native_output(obs)
                action = controller.compute_action(state, reference)
                total = time.perf_counter() - control_start
                assert len(watch.inner) == len(watch.selection) == len(watch.construction) == len(watch.solve) == 5
                assert np.asarray(action).shape == (2,) and np.all(np.isfinite(action))
                last = watch.inner[-1]
                before = obs.copy()
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or (truncated and step != steps-1):
                    raise RuntimeError("Unexpected early environment termination")
                realized = native_output(obs)
                delta = realized-reference
                tracking = float(delta @ args.controller_costs.Q @ delta)
                effort = float(action @ args.controller_costs.R @ action)
                selection = sum(x[0] for x in watch.selection)
                construction, solving = sum(watch.construction), sum(watch.solve)
                assert total >= selection + construction + solving
                values = {
                    "y": realized[TO_BASELINE], "u": action.copy(), "pred": last["pred"][0, TO_BASELINE],
                    "ref": reference[TO_BASELINE], "distance": np.linalg.norm(realized[4:6]-target),
                    "prediction_error_xy": np.linalg.norm(last["pred"][0, 4:6]-realized[4:6]),
                    "status": last["status"], "g_coefficients": last["g"], "g_min": np.min(last["g"]),
                    "observation_before": before, "observation_after": obs.copy(), "y_before": state[TO_BASELINE],
                    "planned_y": last["pred"][:, TO_BASELINE], "planned_u": last["planned_u"],
                    "selected_indices": [x[1] for x in watch.selection], "selected_scores": [x[2] for x in watch.selection],
                    "inner_status": [x["status"] for x in watch.inner], "inner_g": [x["g"] for x in watch.inner],
                    "inner_objective": [x["objective"] for x in watch.inner],
                    "inner_max_constraint_violation": [x["max_constraint_violation"] for x in watch.inner],
                    "inner_selection_time_s": [x[0] for x in watch.selection], "inner_construction_time_s": watch.construction.copy(),
                    "inner_solve_time_s": watch.solve.copy(), "selection_time_s": selection, "localization_time_s": selection,
                    "matrix_preparation_time_s": construction, "solve_time_s": solving, "control_compute_time_s": total,
                    "stage_tracking_cost": tracking, "stage_input_cost": effort, "stage_control_cost": tracking+effort,
                    "qp_objective": last["objective"],
                }
                for key, value in values.items():
                    records[key].append(value)
                if (step+1) % 20 == 0 or step == 0 or step == steps-1:
                    write_json(progress, {"state": "running", "seed": seed, "step": step+1, "steps": steps,
                                         "seed_elapsed_seconds": time.perf_counter()-start,
                                         "last_error_mm": 1000*values["distance"], "updated_unix": time.time()})
                    print(f"seed {seed} step {step+1}/{steps}: error={1000*values['distance']:.2f} mm", flush=True)
    finally:
        env.close()
    arrays = {k: np.asarray(v) for k, v in records.items()}
    for key in ("chart", "goal_chart", "next_chart", "goal_index"):
        arrays[key] = np.full(steps, -1, dtype=int)  # Not applicable, not fabricated Atlas choices.
    for key in ("routing_time_s", "goal_selection_time_s"):
        arrays[key] = np.zeros(steps)
    arrays.update(path_length=np.zeros(steps, dtype=int), matrix_columns=np.full(steps, 200),
                  goal_route_cost=np.full(steps, np.nan), terminal_overlap_active=np.zeros(steps, dtype=bool),
                  initial_observation=initial_obs, target_xy=target, final_goal=arrays["ref"][0])
    q = args.controller_costs.Q[np.ix_(TO_BASELINE, TO_BASELINE)]
    cost = closed_loop_control_cost(arrays["y"], arrays["u"], arrays["ref"], q, args.controller_costs.R,
                                    "seed-dependent final Cartesian target (original Select-DPC weights)")
    valid_prediction = arrays["prediction_error_xy"][np.isfinite(arrays["prediction_error_xy"])]
    row = {"experiment_version": VERSION, "seed": seed, "steps": steps, "solver": "CLARABEL",
           "output_dimension": 8, "reference_mode": "terminal", "graph_waypoints_enabled": False,
           "terminal_reference_policy": "upstream_DefaultReacherSetpoint", "target_xy": target.tolist(),
           "target_above_original_height_constraint": bool(target[1] > .1),
           "target_height_violation_m": float(max(0., target[1]-.1)),
           "goal_setup_time_ms": 0., "localization_history_length": 0,
           "controller_setup_time_s": setup_seconds, "wall_seconds": time.perf_counter()-start,
           "selected_columns": 200, "refinement_iterations": 5, "affine_sum_constraint": True,
           "svd_reduction": False, "unsolved_steps": int(np.count_nonzero(~np.isin(arrays["status"], ["optimal", "optimal_inaccurate"]))),
           "unsolved_refinements": int(np.count_nonzero(~np.isin(arrays["inner_status"], ["optimal", "optimal_inaccurate"]))),
           "statuses": {str(s): int(np.sum(arrays["status"] == s)) for s in np.unique(arrays["status"])},
           "mean_prediction_error_xy": float(valid_prediction.mean()) if valid_prediction.size else None,
           "result_path": str(output.resolve()), **true_error_metrics(arrays["distance"]), **cost,
           "closed_loop_reference_cost": cost["closed_loop_control_cost"],
           "closed_loop_reference_tracking_cost": cost["closed_loop_tracking_cost"],
           "mean_closed_loop_reference_stage_cost": cost["mean_closed_loop_stage_cost"],
           "closed_loop_final_goal_cost": cost["closed_loop_control_cost"],
           "closed_loop_final_goal_tracking_cost": cost["closed_loop_tracking_cost"],
           "mean_closed_loop_final_goal_stage_cost": cost["mean_closed_loop_stage_cost"],
           "xy_squared_error_sum_m2": float(np.sum(arrays["distance"]**2)),
           "input_squared_sum": float(np.sum(arrays["u"]**2)),
           "upstream_scaled_pre_action_cost": float(1e-6*np.sum(
               np.einsum("ti,ij,tj->t", arrays["y_before"]-arrays["ref"], q, arrays["y_before"]-arrays["ref"])
               + arrays["stage_input_cost"])),
           "post_action_height_violations": int(np.count_nonzero(arrays["y"][:, 7] > .1)),
           "max_affine_sum_residual": float(np.nanmax(np.abs(arrays["inner_g"].sum(axis=-1)-1)))
               if np.isfinite(arrays["inner_g"]).any() else None}
    row["strict_success"] = strict_success(row)
    for name in ("localization", "routing", "matrix_preparation", "solve", "control_compute"):
        stats = timing_stats(arrays[f"{name}_time_s"])
        for statistic in ("mean", "median", "p95", "p99", "max"):
            row[f"{statistic}_{name}_time_ms"] = stats[f"{statistic}_ms"]
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays, summary=np.array(json.dumps(row)))
    temporary.replace(output)
    return row


def summarize(rows, seeds, output, signature):
    total, after5 = [], []
    for row in rows:
        with np.load(row["result_path"]) as trace:
            total.append(trace["control_compute_time_s"])
            after5.append(trace["control_compute_time_s"][5:])
    feasible = [r for r in rows if not r["target_above_original_height_constraint"]]
    result = {"method": "Select-DPC (upstream, final target, Clarabel)", "signature": signature,
              "requested_seeds": seeds, "completed_seeds": [r["seed"] for r in rows],
              "complete": len(rows) == len(seeds), "results": rows,
              "successes": sum(strict_success(r) for r in rows), "total": len(rows),
              "success_rate": float(np.mean([strict_success(r) for r in rows])),
              "targets_above_height_constraint": len(rows)-len(feasible),
              "height_compatible_targets": {"total": len(feasible), "successes": sum(strict_success(r) for r in feasible)},
              "median_closed_loop_reference_cost": float(np.median([r["closed_loop_reference_cost"] for r in rows])),
              "median_final100_error_mm": float(1000*np.median([r["mean_last_100"] for r in rows])),
              "timing_all_steps": timing_stats(np.concatenate(total)),
              "timing_discard_first5": timing_stats(np.concatenate(after5)) if all(len(v) for v in after5) else None,
              "cost_caveat": "Own Q/R, not comparable numerically with Atlas own-Q/R costs; raw traces and unweighted XY/input metrics retained.",
              "comparison_caveats": signature["comparison_caveats"]}
    write_json(output / "summary.json", result)
    return result


def requested_block(cli):
    """Choose this invocation's work without changing the full-panel manifest."""
    start, stop = getattr(cli, 'block_start', None), getattr(cli, 'block_stop', None)
    limit = getattr(cli, 'max_new_seeds', None)
    if (start is None) != (stop is None):
        raise ValueError('Supply both --block-start and --block-stop, or neither')
    if start is None:
        start, stop = cli.seed_start, cli.seed_stop
    if not cli.seed_start <= start <= stop <= cli.seed_stop:
        raise ValueError('Block must be inside the full evaluation seed interval')
    if limit is not None and limit < 1:
        raise ValueError('--max-new-seeds must be positive')
    return list(range(start, stop+1)), limit


def run_block(dc, args, scheduler, cli, output, signature):
    seeds = list(range(cli.seed_start, cli.seed_stop+1))
    block, limit = requested_block(cli)
    rows = {}
    # Read all completed seeds, including earlier/noncontiguous blocks, so the
    # cumulative summary always describes the same full evaluation panel.
    for seed in seeds:
        path = output / f'seed{seed}_control_results.npz'
        if path.exists():
            with np.load(path) as trace:
                row = json.loads(trace['summary'].item())
                if (row['seed'], row['steps'], row['experiment_version']) != (seed, cli.steps, VERSION):
                    raise ValueError(f'Incompatible saved rollout: seed {seed}')
                for key in ('distance', 'y', 'u', 'ref', 'control_compute_time_s'):
                    if len(trace[key]) != cli.steps:
                        raise ValueError(f'Incomplete saved rollout: seed {seed}, {key}')
            row['result_path'] = str(path.resolve())
            rows[seed] = row
    pending = [seed for seed in block if seed not in rows]
    scheduled = pending if limit is None else pending[:limit]
    print(f'Panel: {len(rows)}/{len(seeds)} already complete. This invocation: '
          f'{len(scheduled)} new seeds, {cli.steps} steps each.', flush=True)
    if rows:
        summarize([rows[s] for s in sorted(rows)], seeds, output, signature)
    active_seed = None
    completed_now = 0
    try:
        for seed in scheduled:
            active_seed = seed
            path = output / f'seed{seed}_control_results.npz'
            row = run_seed(dc, args, scheduler, seed, cli.steps, path, output / 'progress.json')
            rows[seed] = row
            completed_now += 1
            payload = summarize([rows[s] for s in sorted(rows)], seeds, output, signature)
            print(f"{len(rows)}/{len(seeds)} complete: seed {seed}, success={row['strict_success']}, "
                  f"final100={1000*row['mean_last_100']:.3f} mm, successes={payload['successes']}/{len(rows)}, "
                  f"median compute={row['median_control_compute_time_ms']:.1f} ms; "
                  f"this batch {completed_now}/{len(scheduled)}", flush=True)
    except BaseException:
        write_json(output / 'progress.json', dict(state='interrupted', seed=active_seed,
            completed=len(rows), requested=len(seeds), completed_this_batch=completed_now,
            updated_unix=time.time(), note='Completed seeds preserved; rerun the same batch to resume.'))
        raise
    complete = len(rows) == len(seeds)
    write_json(output / 'progress.json', dict(state='complete' if complete else 'batch_complete',
        completed=len(rows), requested=len(seeds), successes=sum(r['strict_success'] for r in rows.values()),
        block_start=block[0], block_stop=block[-1], block_complete=all(s in rows for s in block),
        completed_this_batch=completed_now, next_missing_seed=next((s for s in seeds if s not in rows), None),
        updated_unix=time.time(), note='Full panel complete.' if complete else
        'Batch finished and worker exited. Run again to continue; completed seeds will be skipped.'))


def main(cli):
    seeds = list(range(cli.seed_start, cli.seed_stop+1))
    if not seeds or cli.steps < 1:
        raise ValueError("Nonempty seed interval and positive steps required")
    requested_block(cli)  # Reject invalid blocks before creating output files.
    output = cli.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "run.lock"
    # Never launch two writers into the same experiment directory.
    with lock.open("a+") as lockfile:
        import fcntl
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dc, args, scheduler = load_upstream()
        with threadpool_limits(limits=1):
            audit = audit_baseline_data(args.trajectory_data)
            signature = {"version": VERSION, "upstream_commit": COMMIT, "runner_sha256": digest(__file__),
                "upstream_files": {str(p.relative_to(UPSTREAM)): digest(p) for p in
                                   [*sorted((UPSTREAM / "select_deepc").glob("*.py")), UPSTREAM / "reacher_simulation.ipynb"]},
                "data_source": "authors' released data/reacher/dataset_iid (not Atlas offline data)",
                "data_files": {p.name: digest(p) for p in sorted((UPSTREAM / "data/reacher/dataset_iid").glob("*.csv"))},
                "data_audit": audit, "steps": cli.steps, "seed_start": cli.seed_start, "seed_stop": cli.seed_stop,
                "selected_columns": 200, "refinement_iterations": 5, "p": 8, "T_ini": 2, "T_f": 15,
                "solver": "CLARABEL", "clarabel_options": CLARABEL_OPTIONS, "max_iter": 6000, "solver_tolerances": "Clarabel defaults", "warm_start": False,
                "environment": "Reacher-v4", "warmup_steps": 0, "workers": 1, "numerical_threads": 1,
                "reference": "upstream DefaultReacherSetpoint; no waypoints or Atlas terminal posture",
                "Q_native": args.controller_costs.Q.tolist(), "R": args.controller_costs.R.tolist(),
                "lambda_g_l1": args.controller_costs.regularizer_cost_g_1,
                "lambda_projection": args.controller_costs.regularizer_cost_g_pi, "lambda_slack_l1": args.controller_costs.slack_cost,
                "input_limit": .5, "output_height_limit": .1,
                "saved_output_order": ["cos_q1", "cos_q2", "sin_q1", "sin_q2", "dq1", "dq2", "fingertip_x", "fingertip_y"],
                "timing_definition": "End-to-end scheduler/output-map/compute_action, all five refinements, construction/pseudoinverse/solve; excludes simulation and file logging. Instrumentation overhead included.",
                "comparison_caveats": ["Authors' original offline data (exploration +/-0.1), not Atlas offline data (+/-1)", "Original Reacher-v4 versus Atlas Reacher-v5", "No physical warmup (upstream repeated-state initialization), versus Atlas 2/12 zero-input steps", "Original horizon 15 versus Atlas 10", "Original input limit 0.5 and output y<=0.1 constraint retained", "Own cost weights; compare unweighted metrics or rescore both methods with common weights"],
                "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "cvxpy", "clarabel", "gymnasium", "mujoco", "torch")}}
            manifest = output / "manifest.json"
            if manifest.exists():
                if json.loads(manifest.read_text()) != signature:
                    raise ValueError("Incompatible run manifest; use a fresh output directory")
            else:
                if list(output.glob("seed*_control_results.npz")):
                    raise ValueError("Unversioned rollout cache")
                write_json(manifest, signature)
            write_json(output / "session.json", {"pid": os.getpid(), "started_unix": time.time(),
                "python": sys.executable, "platform": platform.platform(), "load_average": os.getloadavg(),
                "block_seeds": requested_block(cli)[0], "max_new_seeds": requested_block(cli)[1]})
            print(f"Using authors' original offline data (matches Atlas: {audit['matches_baseline']}). Original 200-column/five-refinement Select-DPC.", flush=True)
            run_block(dc, args, scheduler, cli, output, signature)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=211)
    parser.add_argument("--seed-stop", type=int, default=310)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument('--block-start', type=int, help='First seed to run this time; panel --seed-start stays fixed')
    parser.add_argument('--block-stop', type=int, help='Last seed to run this time; panel --seed-stop stays fixed')
    parser.add_argument('--max-new-seeds', type=int, help='Stop after this many previously unfinished seeds')
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    cli = parser.parse_args()
    try:
        main(cli)
    except BaseException as exc:
        if cli.output_dir.exists():
            write_json(cli.output_dir / "failure.json", {"error": repr(exc), "pid": os.getpid(), "time": time.time()})
        raise
