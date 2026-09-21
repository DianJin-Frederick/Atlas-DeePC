"""One fixed Select-DPC configuration, 100 development seeds, 200 steps.

Two straight-line intermediate references; no parameter search or Atlas rerun.
Initialize once and execute the frozen entry point printed by init.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import json
import multiprocessing
from pathlib import Path
import shutil
import time

import select_dpc_development as study  # Numerical thread limits before NumPy.
import select_dpc_focused as focused
import numpy as np
from threadpoolctl import threadpool_limits

SOURCES = (Path(__file__).name, "select_dpc_focused.py", *study.SOURCES)


def intermediate_points(start, target):
    start, target = np.asarray(start, dtype=float), np.asarray(target, dtype=float)
    if start.shape != (2,) or target.shape != (2,) or not np.all(np.isfinite([start, target])):
        raise ValueError("Expected finite start and target xy vectors")
    return start + np.array([[1 / 3], [2 / 3]]) * (target - start)


def default_plan():
    base = study.default_plan()
    parameters = asdict(replace(study.SelectConfig(), lambda_projection=1e6, lambda_slack=5e4))
    return {
        "version": "select_dpc_waypoint_dev100_v1", "seeds": list(range(211, 311)),
        "steps": 200, "workers": 2, "numerical_threads": 1,
        "parameters": parameters, "objective_scale": .01,
        "data_path": base["data_path"], "baseline_notebook": base["baseline_notebook"],
        "reference_mode": "two_straight_line_intermediate_waypoints",
        "waypoint_fractions": [1 / 3, 2 / 3], "waypoint_tolerance_m": .005,
        "waypoint_max_steps": 40, "evaluation_cost": base["evaluation_cost"],
        "success": base["success"], "solver": base["solver"], "development_only": True,
        "notes": [
            "One fixed configuration, exactly 100 seeds; no adaptive parameter search.",
            "Same Atlas offline recordings, 200 raw selected columns, no trajectory SVD.",
            "All objective weights multiplied by .01; exact optimizer unchanged, finite solver behavior may differ.",
            "Intermediate points are computed from initial fingertip and final target only, never rollout outcomes.",
            "Straight-line waypoint references are a heuristic, not a guarantee of physical or closed-loop feasibility.",
            "Switch within 5mm or after 40 steps per intermediate point; hold final target from step80 at latest.",
            "Success and common cost use the unchanged final target; active-reference cost is also saved.",
            "Online reference generation and switching are charged to computation time.",
            "Atlas keeps its graph routing; no Atlas reruns or mutations. Comparison is end-to-end, not selector-only.",
            "Existing Atlas notebook reports different seeds and 320 steps; do not claim a paired comparison with this pilot panel.",
            "Two-worker timing is descriptive, not a controlled latency comparison to prior single-worker Atlas runs.",
        ],
    }


def initialize(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "plan.json").exists():
        raise FileExistsError("Study already initialized; resume with run")
    plan = default_plan()
    import baseline_atlas_deepc as baseline
    if Path(plan["data_path"]).resolve() != baseline.DATA_PATH.resolve():
        raise ValueError("Recheck current Atlas data path")
    source = "\n".join("".join(c.get("source", [])) for c in json.loads(Path(plan["baseline_notebook"]).read_text())["cells"])
    if "DATA_PATH, ATLAS_PATH = baseline.DATA_PATH, baseline.ATLAS_PATH" not in source:
        raise ValueError("Recheck current baseline notebook data source")
    frozen = root / "frozen_protocol/src"
    frozen.mkdir(parents=True)
    for name in SOURCES:
        shutil.copy2(Path(__file__).parent / name, frozen / name)
    manifest = {"plan_sha256": study.digest(plan), "data_sha256": study.sha(plan["data_path"]),
                "baseline_notebook_sha256": study.sha(plan["baseline_notebook"]),
                "source_sha256": {n: study.sha(frozen / n) for n in SOURCES},
                "runtime": study.runtime(), "created": study.stamp()}
    study.write_json(root / "plan.json", plan)
    study.write_json(root / "manifest.json", manifest)
    report(root, plan)
    print("Frozen entry:", frozen / Path(__file__).name, flush=True)


def check_manifest(root, plan):
    manifest = json.loads((root / "manifest.json").read_text())
    if study.digest(plan) != manifest["plan_sha256"] or study.runtime() != manifest["runtime"]:
        raise ValueError("Frozen plan or runtime changed")
    if study.sha(plan["data_path"]) != manifest["data_sha256"]:
        raise ValueError("Offline recordings changed")
    if {n: study.sha(Path(__file__).parent / n) for n in SOURCES} != manifest["source_sha256"]:
        raise ValueError("Use the matching frozen source entry point")


def run_seed(root_string, plan, seed):
    root = Path(root_string) / f"seed{seed}"
    root.mkdir(exist_ok=True)
    signature = study.digest(plan)
    result_path = root / "result.json"
    if result_path.exists():
        row = json.loads(result_path.read_text())
        with np.load(root / "trace.npz") as z:
            if row["plan_sha256"] != signature or row["seed"] != seed or len(z["u"]) != 200:
                raise ValueError("Incompatible cached seed result")
        return row
    with study.single_writer(root), threadpool_limits(limits=1):
        config = focused.scaled_config(study.SelectConfig(**plan["parameters"]), plan["objective_scale"])
        controller = study.SelectDPC(*study._LIBRARY, config)
        env = study.gym.make(study.pipe.REACHER_ENV_ID, max_episode_steps=207)
        trace = {k: [] for k in ("y", "u", "pred", "ref", "waypoint_index", "reference_time_s",
                                 "control_compute_time_s", "selection_time_s", "matrix_preparation_time_s",
                                 "solve_time_s", "statuses")}
        diagnostics = []
        try:
            obs, _ = env.reset(seed=seed)
            target = np.asarray(obs[4:6]).copy()
            for _ in range(2):
                obs, *_ = env.step(np.zeros(2))
            initial_qpos, initial_qvel = env.unwrapped.data.qpos.copy(), env.unwrapped.data.qvel.copy()
            start_xy = study.pipe.reacher_output_from_obs(obs)[6:8].copy()
            schedule = None
            for step in range(plan["steps"]):
                start = time.perf_counter()
                if schedule is None:
                    points = intermediate_points(start_xy, target)
                    schedule = focused.WaypointSchedule(points, target, plan["waypoint_tolerance_m"],
                                                        plan["waypoint_max_steps"])
                state = study.pipe.reacher_output_from_obs(obs)
                reference, index = schedule.reference(state[6:8], step)
                reference_time = time.perf_counter() - start
                action, prediction, info = controller.step(state[study.OUTPUT_INDICES], reference)
                elapsed = time.perf_counter() - start
                if np.asarray(action).shape != (2,) or not np.all(np.isfinite(action)):
                    raise RuntimeError("Invalid action")
                obs, *_ = env.step(action)
                measured = study.pipe.reacher_output_from_obs(obs)
                diagnostics.append(focused.diagnostic(controller, measured, prediction, reference))
                for key, value in {"y": measured, "u": action, "pred": prediction, "ref": reference,
                                   "waypoint_index": index, "reference_time_s": reference_time,
                                   "control_compute_time_s": elapsed, "selection_time_s": info["selection_s"],
                                   "matrix_preparation_time_s": info["matrix_s"], "solve_time_s": info["solve_s"],
                                   "statuses": info["statuses"]}.items():
                    trace[key].append(value)
                if step == 0 or (step + 1) % 10 == 0:
                    packed, metrics = focused.snapshot(trace, diagnostics, target, plan, plan["steps"])
                    payload = {"seed": seed, "updated": study.stamp(), "plan_sha256": signature,
                               "waypoints_xy": points.tolist(), "final_target_xy": target.tolist(),
                               "waypoint_events": schedule.events, **metrics}
                    study.write_json(root / "progress.json", payload)
                    temporary = root / "trace.partial.npz"
                    np.savez_compressed(temporary, **packed, initial_qpos=initial_qpos, initial_qvel=initial_qvel)
                    temporary.replace(root / "trace.npz")
                    study.write_json(root / "diagnostics.json", diagnostics)
            row = {**payload, "complete": True, "development_only": True}
            study.write_json(result_path, row)
            print(study.stamp(), f"seed={seed} success={row['success']} last100={row['final100_mm']:.3f}mm "
                  f"failed_QPs={row['unsolved_subproblems']}/{row['subproblems']}", flush=True)
            return row
        except BaseException as exc:
            study.write_json(root / "interrupted.json", {"time": study.stamp(), "error": repr(exc),
                                                        "completed_steps": len(trace["u"])})
            raise
        finally:
            env.close()


def aggregate(rows, requested):
    done = len(rows)
    successes = sum(r["success"] for r in rows)
    return {"completed_seeds": done, "requested_seeds": requested, "strict_successes": successes,
            "strict_success_rate_completed": successes / done if done else None,
            "last30_successes": sum(r["success_30"] for r in rows),
            "last100_successes": sum(r["success_100"] for r in rows),
            "complete": done == requested,
            "median_final100_mm": float(np.median([r["final100_mm"] for r in rows])) if done else None,
            "median_common_task_cost": float(np.median([r["common_task_cost"] for r in rows])) if done else None,
            "median_active_reference_task_cost": float(np.median([r["active_reference_task_cost"] for r in rows])) if done else None,
            "unsolved_subproblems": sum(r["unsolved_subproblems"] for r in rows),
            "subproblems": sum(r["subproblems"] for r in rows)}


def report(root, plan):
    rows, times, maxima = [], [], []
    signature = study.digest(plan)
    for seed in plan["seeds"]:
        path = root / f"seed{seed}/result.json"
        if path.exists():
            row = json.loads(path.read_text())
            if row["plan_sha256"] != signature or row["seed"] != seed or row["step"] != plan["steps"]:
                raise ValueError("Incompatible summary row")
            rows.append(row)
            with np.load(path.with_name("trace.npz")) as z:
                t = z["control_compute_time_s"] * 1000
                times.extend(t[5:].tolist())
                maxima.append(float(t.max()))
    payload = {"updated": study.stamp(), "development_only": True, "plan_sha256": signature,
               **aggregate(rows, len(plan["seeds"])), "completed_seed_ids": [r["seed"] for r in rows],
               "median_compute_ms": float(np.median(times)) if times else None,
               "p99_compute_ms": float(np.percentile(times, 99)) if times else None,
               "max_compute_ms_including_startup": max(maxima) if maxima else None,
               "timing_note": "Pooled steps, first five per seed excluded for median/p99. Two concurrent workers.",
               "comparison_note": "Atlas retains graph routing; not a matched Atlas comparison until seeds, targets, duration and timing conditions are aligned."}
    study.write_json(root / "summary.json", payload)
    return payload


def run(root, plan):
    with study.single_writer(root):
        check_manifest(root, plan)
        print(study.stamp(), "Starting fixed 100-seed, 200-step Select-DPC waypoint panel; two workers.", flush=True)
        with ProcessPoolExecutor(max_workers=plan["workers"], mp_context=multiprocessing.get_context("spawn"),
                                 initializer=study.init_worker, initargs=(plan["data_path"],)) as pool:
            futures = [pool.submit(run_seed, str(root), plan, seed) for seed in plan["seeds"]]
            for future in as_completed(futures):
                future.result()
                payload = report(root, plan)
                print(study.stamp(), f"Completed {payload['completed_seeds']}/100; "
                      f"strict successes {payload['strict_successes']}", flush=True)
        print(study.stamp(), "COMPLETE", json.dumps(report(root, plan)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("init", "run", "report"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if args.phase == "init":
        initialize(root)
        return
    plan = json.loads((root / "plan.json").read_text())
    if args.phase == "report":
        print(json.dumps(report(root, plan), indent=2))
    else:
        try:
            run(root, plan)
        except Exception as exc:
            study.write_json(root / "failure.json", {"time": study.stamp(), "error": repr(exc)})
            raise


if __name__ == "__main__":
    main()
