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
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--q-xy", type=float, default=4e4)
    parser.add_argument("--lambda-projection", type=float, default=1e6)
    parser.add_argument("--lambda-slack", type=float, default=5e4)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    if args.seed not in range(211, 311) or args.steps != 320:
        parser.error("Use one development seed 211..310 and the full 320-step task")
    if min(args.q_xy, args.lambda_projection, args.lambda_slack) <= 0:
        parser.error("All three weights must be positive")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "plan.json").exists():
        raise FileExistsError("Trial directory already used; preserve it and choose a new directory")
    config = replace(study.SelectConfig(), q_xy=args.q_xy,
                     lambda_projection=args.lambda_projection, lambda_slack=args.lambda_slack)
    base = study.default_plan()
    plan = {"seed": args.seed, "steps": args.steps, "parameters": asdict(config),
            "reason": args.reason, "data_path": base["data_path"],
            "evaluation_cost": base["evaluation_cost"], "success": base["success"],
            "adaptations": base["adaptations"], "solver": base["solver"],
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
        controller = study.SelectDPC(*study._LIBRARY, config)
        env = study.gym.make(study.pipe.REACHER_ENV_ID, max_episode_steps=args.steps + 7)
        trace = {k: [] for k in ("y", "u", "pred", "control_compute_time_s", "selection_time_s",
                                 "matrix_preparation_time_s", "solve_time_s", "statuses")}
        diagnostics = []
        try:
            obs, _ = env.reset(seed=args.seed)
            target = np.asarray(obs[4:6]).copy()
            for _ in range(2):
                obs, *_ = env.step(np.zeros(2))
            initial_qpos, initial_qvel = env.unwrapped.data.qpos.copy(), env.unwrapped.data.qvel.copy()
            reference = study.native_reference(target)
            for step in range(args.steps):
                start = time.perf_counter()
                action, prediction, info = controller.step(study.pipe.reacher_output_from_obs(obs)[study.OUTPUT_INDICES], reference)
                elapsed = time.perf_counter() - start
                if np.asarray(action).shape != (2,) or not np.all(np.isfinite(action)):
                    raise RuntimeError("Invalid controller action")
                obs, *_ = env.step(action)
                measured = study.pipe.reacher_output_from_obs(obs)
                diagnostics.append(diagnostic(controller, measured, prediction, reference))
                for key, value in {"y": measured, "u": action, "pred": prediction,
                                   "control_compute_time_s": elapsed, "selection_time_s": info["selection_s"],
                                   "matrix_preparation_time_s": info["matrix_s"], "solve_time_s": info["solve_s"],
                                   "statuses": info["statuses"]}.items():
                    trace[key].append(value)
                if (step + 1) % 10 == 0 or step == 0 or step + 1 == args.steps:
                    packed, metrics = snapshot(trace, diagnostics, target, plan, args.steps)
                    payload = {"updated": study.stamp(), "seed": args.seed, **metrics}
                    study.write_json(root / "progress.json", payload)
                    study.write_json(root / "diagnostics.json", diagnostics)
                    temporary = root / "trace.partial.npz"
                    np.savez_compressed(temporary, **packed, initial_qpos=initial_qpos, initial_qvel=initial_qvel)
                    temporary.replace(root / "trace.npz")
                    print(study.stamp(), json.dumps(payload), flush=True)
            study.write_json(root / "result.json", {**payload, "parameters": asdict(config),
                             "complete": True, "development_only": True})
        except BaseException as exc:
            study.write_json(root / "interrupted.json", {"updated": study.stamp(), "error": repr(exc),
                                                        "completed_steps": len(trace["u"])})
            raise
        finally:
            env.close()


if __name__ == "__main__":
    main()
