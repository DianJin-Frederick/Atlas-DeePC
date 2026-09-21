"""One explicit Select-DPC trial at a time, with inspectable tracking diagnostics.

No grid, automatic candidate expansion, or automatic 100-seed launch.
The controller formulation and solver are imported without modification.
"""
from dataclasses import asdict, replace
import argparse
import json
from pathlib import Path
import shutil
import time

import select_dpc_development as study  # Sets numerical thread limits first.
import numpy as np
from threadpoolctl import threadpool_limits


def scaled_config(config, scale):
    """Uniform positive objective scaling; no change to relative penalties."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Objective scale must be finite and positive")
    weights = ("q_xy", "q_velocity", "r", "lambda_g1", "lambda_projection", "lambda_slack")
    return replace(config, **{key: getattr(config, key) * scale for key in weights})


class WaypointSchedule:
    """External reference aid, not a change to Select-DPC's selector or QP."""
    def __init__(self, waypoints, target, tolerance=.005, max_steps=0):
        points = np.asarray(waypoints, dtype=float).reshape(-1, 2)
        self.points = np.vstack([points, np.asarray(target, dtype=float).reshape(1, 2)])
        if not np.all(np.isfinite(self.points)) or tolerance <= 0 or max_steps < 0:
            raise ValueError("Invalid waypoint schedule")
        self.tolerance, self.max_steps = tolerance, max_steps
        self.index, self.entered = 0, 0
        self.events = []

    def reference(self, xy, step):
        while self.index < len(self.points) - 1:
            distance = float(np.linalg.norm(np.asarray(xy) - self.points[self.index]))
            reached = distance < self.tolerance
            timeout = self.max_steps > 0 and step - self.entered >= self.max_steps
            if not (reached or timeout):
                break
            self.events.append({"step": step, "waypoint": self.index,
                                "reason": "reached" if reached else "timeout",
                                "distance_mm": 1000 * distance})
            self.index += 1
            self.entered = step
        return study.native_reference(self.points[self.index]), self.index


def load_waypoints(path, seed, target):
    if path is None:
        return [], None
    spec = json.loads(path.read_text())
    if spec["seed"] != seed or not np.allclose(spec["final_target_xy"], target, rtol=0, atol=1e-8):
        raise ValueError("Manual waypoint file does not match this seed and final target")
    points = np.asarray(spec["waypoints_xy"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.all(np.isfinite(points)):
        raise ValueError("Expected a nonempty list of finite xy waypoints")
    return points.tolist(), spec


def diagnostic(controller, measured, prediction, reference):
    model = controller.subproblem
    c = controller.config
    row = {"prediction_error_mm": None, "objective_terms": None,
           "past_xy_slack_mm": None, "past_posture_slack": None}
    if np.all(np.isfinite(prediction)):
        row["prediction_error_mm"] = float(1000 * np.linalg.norm(prediction[4:6] - measured[6:8]))
    if model.g.value is not None and model.problem.status in ("optimal", "optimal_inaccurate"):
        g, u, y, slack = model.g.value, model.u.value, model.y.value, model.slack.value
        residual = y - np.tile(reference, c.horizon)
        row["objective_terms"] = {
            "tracking": float(np.sum(np.tile(np.diag(model.Q), c.horizon) * residual**2)),
            "effort": float(c.r * np.sum(u**2)),
            "slack": float(c.lambda_slack * np.abs(slack).sum()),
            "coefficient_l1": float(c.lambda_g1 * np.abs(g).sum()),
            "projection": float(c.lambda_projection * np.sum((g - model.projection @ g)**2)),
        }
        s = slack.reshape(c.past, 8)
        row["past_xy_slack_mm"] = float(1000 * np.linalg.norm(s[:, 4:6], axis=1).mean())
        row["past_posture_slack"] = float(np.abs(s[:, :4]).mean())
    return row


def snapshot(trace, diagnostics, target, plan, requested_steps):
    packed = {**{k: np.asarray(v) for k, v in trace.items()}, "target": target}
    metrics = study.summarize(packed, plan)
    if "ref" in packed:
        # Use the reference active when each action was computed, not the next waypoint.
        active_error = packed["y"][:, 6:8] - packed["ref"][:, 4:6]
        final_error = packed["y"][:, 6:8] - target
        metrics["active_reference_task_cost"] = float(metrics["common_task_cost"] +
            plan["evaluation_cost"]["q_xy"] * (np.sum(active_error**2) - np.sum(final_error**2)))
    if len(trace["u"]) != requested_steps:
        # A partial tail meeting the tolerance is not a completed rollout success.
        metrics["success"] = None
    errors = np.linalg.norm(packed["y"][:, 6:8] - target, axis=1) * 1000
    return packed, {"step": len(errors), "requested_steps": requested_steps,
                    "current_error_mm": float(errors[-1]),
                    "last30_mean_error_mm": float(errors[-30:].mean()),
                    "last30_max_error_mm": float(errors[-30:].max()),
                    "last_diagnostic": diagnostics[-1], **metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--steps", type=int, choices=(200, 320), default=200)
    parser.add_argument("--q-xy", type=float, default=4e4)
    parser.add_argument("--lambda-projection", type=float, default=1e6)
    parser.add_argument("--lambda-slack", type=float, default=5e4)
    parser.add_argument("--objective-scale", type=float, default=1.,
                        help="Uniform multiplier for all six objective weights; recorded numerical adaptation")
    parser.add_argument("--waypoints-file", type=Path,
                        help="Explicit seed-specific xy waypoints; final target is appended automatically")
    parser.add_argument("--waypoint-tolerance", type=float, default=.005)
    parser.add_argument("--waypoint-max-steps", type=int, default=0,
                        help="Optional intermediate-reference timeout; 0 uses proximity switching only")
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    if args.seed not in range(211, 311):
        parser.error("Use one development seed 211..310")
    if args.waypoint_tolerance <= 0 or args.waypoint_max_steps < 0:
        parser.error("Waypoint tolerance must be positive and timeout nonnegative")
    if min(args.q_xy, args.lambda_projection, args.lambda_slack) <= 0:
        parser.error("All three weights must be positive")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "plan.json").exists():
        raise FileExistsError("Trial directory already used; preserve it and choose a new directory")
    config = replace(study.SelectConfig(), q_xy=args.q_xy,
                     lambda_projection=args.lambda_projection, lambda_slack=args.lambda_slack)
    effective_config = scaled_config(config, args.objective_scale)
    base = study.default_plan()
    reference_mode = "manual_waypoints" if args.waypoints_file else "final_target"
    plan = {"seed": args.seed, "steps": args.steps, "parameters": asdict(config),
            "reference_mode": reference_mode,
            "waypoint_file": str(args.waypoints_file.resolve()) if args.waypoints_file else None,
            "waypoint_tolerance_m": args.waypoint_tolerance,
            "waypoint_max_steps": args.waypoint_max_steps,
            "reference_note": "External, seed-specific development reference aid; not claimed as the paper's path planner or an unbiased benchmark. All success metrics use the unchanged final target.",
            "timing_note": "Online waypoint switching included; manual offline path design not timed.",
            "objective_scale": args.objective_scale, "effective_parameters": asdict(effective_config),
            "numerical_note": "Uniform positive objective scaling preserves the exact optimizer, but finite solver tolerances can change the numerical solution. Diagnostic objective terms use effective weights.",
            "reason": args.reason, "data_path": base["data_path"],
            "evaluation_cost": base["evaluation_cost"], "success": base["success"],
            "adaptations": [*base["adaptations"][:3],
                            f"{args.steps}-step pilot; success always assessed against the final environment target.",
                            "No Atlas posture reference or graph routing; optional manually supplied xy waypoints."],
            "solver": base["solver"],
            "development_only": True, "created": study.stamp(),
            "next_action": "Stop after this seed; inspect results before selecting the next trial."}
    files = [Path(__file__), *[Path(study.__file__).parent / n for n in study.SOURCES]]
    frozen = root / "source_snapshot"
    frozen.mkdir()
    for p in files:
        shutil.copy2(p, frozen / p.name)
    study.write_json(root / "plan.json", plan)
    study.write_json(root / "manifest.json", {
        "source_sha256": {p.name: study.sha(p) for p in files},
        "data_sha256": study.sha(plan["data_path"]), "runtime": study.runtime()})
    with study.single_writer(root), threadpool_limits(limits=1):
        study.init_worker(plan["data_path"])
        controller = study.SelectDPC(*study._LIBRARY, effective_config)
        env = study.gym.make(study.pipe.REACHER_ENV_ID, max_episode_steps=args.steps + 7)
        trace = {k: [] for k in ("y", "u", "pred", "ref", "waypoint_index", "reference_time_s",
                                 "control_compute_time_s", "selection_time_s",
                                 "matrix_preparation_time_s", "solve_time_s", "statuses")}
        diagnostics = []
        try:
            obs, _ = env.reset(seed=args.seed)
            target = np.asarray(obs[4:6]).copy()
            for _ in range(2):
                obs, *_ = env.step(np.zeros(2))
            initial_qpos, initial_qvel = env.unwrapped.data.qpos.copy(), env.unwrapped.data.qvel.copy()
            points, spec = load_waypoints(args.waypoints_file, args.seed, target)
            if spec is not None:
                study.write_json(root / "waypoints.json", spec)
            schedule = WaypointSchedule(points, target, args.waypoint_tolerance, args.waypoint_max_steps)
            for step in range(args.steps):
                start = time.perf_counter()
                state = study.pipe.reacher_output_from_obs(obs)
                reference, waypoint_index = schedule.reference(state[6:8], step)
                reference_elapsed = time.perf_counter() - start
                action, prediction, info = controller.step(state[study.OUTPUT_INDICES], reference)
                elapsed = time.perf_counter() - start
                if np.asarray(action).shape != (2,) or not np.all(np.isfinite(action)):
                    raise RuntimeError("Invalid controller action")
                obs, *_ = env.step(action)
                measured = study.pipe.reacher_output_from_obs(obs)
                diagnostics.append(diagnostic(controller, measured, prediction, reference))
                for key, value in {"y": measured, "u": action, "pred": prediction,
                                   "ref": reference, "waypoint_index": waypoint_index,
                                   "reference_time_s": reference_elapsed,
                                   "control_compute_time_s": elapsed, "selection_time_s": info["selection_s"],
                                   "matrix_preparation_time_s": info["matrix_s"], "solve_time_s": info["solve_s"],
                                   "statuses": info["statuses"]}.items():
                    trace[key].append(value)
                if (step + 1) % 10 == 0 or step == 0 or step + 1 == args.steps:
                    packed, metrics = snapshot(trace, diagnostics, target, plan, args.steps)
                    payload = {"updated": study.stamp(), "seed": args.seed,
                               "reference_mode": reference_mode, "active_waypoint_index": waypoint_index,
                               "waypoint_events": schedule.events, **metrics}
                    study.write_json(root / "progress.json", payload)
                    study.write_json(root / "diagnostics.json", diagnostics)
                    temporary = root / "trace.partial.npz"
                    np.savez_compressed(temporary, **packed, initial_qpos=initial_qpos, initial_qvel=initial_qvel)
                    temporary.replace(root / "trace.npz")
                    print(study.stamp(), json.dumps(payload), flush=True)
            study.write_json(root / "result.json", {**payload, "parameters": asdict(config),
                             "objective_scale": args.objective_scale,
                             "effective_parameters": asdict(effective_config),
                             "complete": True, "development_only": True})
        except BaseException as exc:
            study.write_json(root / "interrupted.json", {"updated": study.stamp(), "error": repr(exc),
                                                        "completed_steps": len(trace["u"])})
            raise
        finally:
            env.close()


if __name__ == "__main__":
    main()
