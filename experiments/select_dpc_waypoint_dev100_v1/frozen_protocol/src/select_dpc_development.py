"""Bounded, resumable Select-DPC-only development study. No Atlas reruns.

Initialize once, then run the frozen entry point printed by `init`.
Raw 200-column formulation lives in faithful_dpc_controllers.py unchanged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

import cvxpy as cp
import gymnasium as gym
import mujoco
import numpy as np
import osqp
from threadpoolctl import threadpool_limits

import expert_pipeline_search as pipe
from faithful_dpc_controllers import (
    OUTPUT_INDICES, SelectConfig, SelectDPC, native_library,
    native_reference, reconstruct_offline_episodes,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments/select_dpc_200_dev211_310_v1"
SOURCES = ("select_dpc_development.py", "faithful_dpc_controllers.py", "expert_pipeline_search.py")
_LIBRARY = None


def stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def candidate_configs():
    paper = SelectConfig()
    anchor = replace(paper, lambda_projection=1e6, lambda_slack=5e4)
    return {
        "paper_weights": asdict(paper),
        "relaxed_slack": asdict(replace(paper, lambda_slack=5e4)),
        "prior_promising": asdict(anchor),
        "middle_projection": asdict(replace(anchor, lambda_projection=1e5)),
        "lower_slack": asdict(replace(anchor, lambda_slack=5e3)),
        "stronger_tracking": asdict(replace(anchor, q_xy=4e5)),
    }


def default_plan(all_candidates=False):
    # Evenly spread, fixed before observing any result on this development panel.
    screen = list(range(211, 311, 10))
    return {
        "version": 1, "method": "Select-DPC (L1 selector, authors-code raw QP)",
        "data_path": str(ROOT / "data/select_iid_offline_hankel.npy"),
        "baseline_notebook": str(ROOT / "notebooks/baselines/baseline_graph_routed_atlas_deepc.ipynb"),
        "seeds": list(range(211, 311)), "screening_seeds": screen,
        "steps": 320, "workers": 2, "numerical_threads": 1,
        "full_panel_policy": "all" if all_candidates else "paper_plus_top_two_alternatives",
        "maximum_development_rollouts": 600 if all_candidates else 330,
        "candidates": candidate_configs(),
        "ranking": "success count descending, mean fixed evaluation cost ascending, candidate name",
        "evaluation_cost": {"q_xy": 1e5, "r": 1e-4, "r_delta": 5.},
        "success": "both last-30 and last-100 mean xy error <0.01 m and max <0.02 m",
        "solver": {"name": "OSQP", "max_iter": 6000, "eps_abs": 1e-5,
                   "eps_rel": 1e-4, "warm_start": False},
        "adaptations": [
            "Shared Atlas offline recordings; native eight outputs and 2+15-step windows reconstructed exactly.",
            "Shared Atlas task torque bound +/-1 (authors Reacher example uses +/-0.5).",
            "Two zero-action environment initialization steps match Atlas; original repeated-state internal history retained.",
            "Requested final fingertip target, no graph routing, no Atlas posture reference.",
            "320-step shared task and strict Atlas success criterion; not the paper's exact experimental protocol.",
        ],
        "timing_caveat": "Two independent workers on a shared computer; tuning timings are descriptive, not final latency benchmarks.",
        "development_only": True,
        "selection_caveat": "Best of fully evaluated candidates, not a global optimum or unbiased held-out success rate.",
    }


def validate_plan(plan):
    if plan["seeds"] != list(range(211, 311)) or plan["steps"] != 320:
        raise ValueError("This study is fixed to seeds 211..310 and 320 control steps")
    if not set(plan["screening_seeds"]).issubset(plan["seeds"]):
        raise ValueError("Screening must use only development seeds")
    if plan["workers"] != 2 or plan["numerical_threads"] != 1:
        raise ValueError("Fixed two-worker / one-numerical-thread execution")
    if plan["full_panel_policy"] not in ("all", "paper_plus_top_two_alternatives"):
        raise ValueError("Unknown evaluation policy")
    if plan["candidates"] != candidate_configs():
        raise ValueError("Candidate shortlist changed; create a separately versioned study")
    for params in plan["candidates"].values():
        c = SelectConfig(**params)
        if (c.columns, c.past, c.horizon, c.iterations, c.input_limit) != (200, 2, 15, 5, 1.):
            raise ValueError("Raw architecture and shared input bounds must remain fixed")


def runtime():
    return {"python": platform.python_version(), "numpy": np.__version__, "cvxpy": cp.__version__,
            "osqp": osqp.__version__, "gymnasium": gym.__version__, "mujoco": mujoco.__version__}


def initialize(root, all_candidates=False):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "plan.json").exists():
        raise FileExistsError("Study already initialized; use run/report, not init")
    plan = default_plan(all_candidates)
    validate_plan(plan)
    notebook = Path(plan["baseline_notebook"])
    source = "\n".join("".join(c.get("source", [])) for c in json.loads(notebook.read_text())["cells"])
    if "DATA_PATH, ATLAS_PATH = baseline.DATA_PATH, baseline.ATLAS_PATH" not in source:
        raise ValueError("Recheck baseline notebook data source before initializing")
    import baseline_atlas_deepc as baseline
    if Path(plan["data_path"]).resolve() != baseline.DATA_PATH.resolve():
        raise ValueError("Data does not match current baseline")
    frozen = root / "frozen_protocol/src"
    frozen.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        shutil.copy2(Path(__file__).parent / name, frozen / name)
    reference = ROOT.parent / "choose-wisely-paper-main"
    refs = [reference / "select_deepc/deepc_controller.py", reference / "select_deepc/data_selectors.py",
            reference / "reacher_simulation.ipynb"]
    manifest = {"plan_sha256": digest(plan), "data_sha256": sha(plan["data_path"]),
                "baseline_notebook_sha256": sha(notebook), "runtime": runtime(),
                "source_sha256": {name: sha(frozen / name) for name in SOURCES},
                "reference_sha256": {str(p): sha(p) for p in refs}, "created": stamp()}
    write_json(root / "plan.json", plan)
    write_json(root / "manifest.json", manifest)
    print("Initialized", root, "\nFrozen entry:", frozen / Path(__file__).name, flush=True)


def check_manifest(root, plan):
    manifest = json.loads((root / "manifest.json").read_text())
    if digest(plan) != manifest["plan_sha256"] or runtime() != manifest["runtime"]:
        raise ValueError("Plan/runtime differs from frozen study")
    if sha(plan["data_path"]) != manifest["data_sha256"]:
        raise ValueError("Offline data changed")
    if {n: sha(Path(__file__).parent / n) for n in SOURCES} != manifest["source_sha256"]:
        raise ValueError("Code changed; run the frozen entry point")


@contextmanager
def single_writer(root):
    with (root / ".run.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def init_worker(data_path):
    global _LIBRARY
    with threadpool_limits(limits=1):
        H = np.load(data_path, mmap_mode="r")
        inputs, outputs = reconstruct_offline_episodes(H)
        _LIBRARY = native_library(inputs, outputs, 2, 15)


def summarize(trace, plan):
    error = np.linalg.norm(trace["y"][:, 6:8] - trace["target"], axis=1)
    success = lambda n: bool(len(error) >= n and error[-n:].mean() < .01 and error[-n:].max() < .02)
    u = trace["u"]
    w = plan["evaluation_cost"]
    cost = w["q_xy"] * np.sum(error**2) + w["r"] * np.sum(u**2)
    cost += w["r_delta"] * np.sum(np.diff(np.vstack([np.zeros(2), u]), axis=0)**2)
    all_times = trace["control_compute_time_s"] * 1000
    times = all_times[5:] if len(all_times) > 5 else all_times
    statuses = trace["statuses"]
    return {"success": success(30) and success(100), "success_30": success(30), "success_100": success(100),
            "final100_mm": float(1000 * error[-100:].mean()), "common_task_cost": float(cost),
            "unsolved_subproblems": int(np.sum(~np.isin(statuses, ["optimal", "optimal_inaccurate"]))),
            "subproblems": int(statuses.size), "median_compute_ms": float(np.median(times)),
            "p99_compute_ms": float(np.percentile(times, 99)), "max_compute_ms": float(all_times.max()),
            "input_bound_violations": int(np.sum(np.any(np.abs(u) > 1.0001, axis=1)))}


def rollout(config, seed, steps, progress_path):
    controller = SelectDPC(*_LIBRARY, config)
    env = gym.make(pipe.REACHER_ENV_ID, max_episode_steps=steps + 7)
    trace = {k: [] for k in ("y", "u", "pred", "control_compute_time_s", "selection_time_s",
                             "matrix_preparation_time_s", "solve_time_s", "statuses")}
    try:
        obs, _ = env.reset(seed=seed)
        target = np.asarray(obs[4:6]).copy()
        for _ in range(2):
            obs, *_ = env.step(np.zeros(2))
        initial_qpos, initial_qvel = env.unwrapped.data.qpos.copy(), env.unwrapped.data.qvel.copy()
        reference = native_reference(target)
        for step in range(steps):
            state = pipe.reacher_output_from_obs(obs)[OUTPUT_INDICES]
            start = time.perf_counter()
            action, pred, info = controller.step(state, reference)
            elapsed = time.perf_counter() - start
            if np.asarray(action).shape != (2,) or not np.all(np.isfinite(action)):
                raise RuntimeError("Invalid controller action")
            obs, *_ = env.step(action)
            for key, value in {"y": pipe.reacher_output_from_obs(obs), "u": action, "pred": pred,
                               "control_compute_time_s": elapsed, "selection_time_s": info["selection_s"],
                               "matrix_preparation_time_s": info["matrix_s"], "solve_time_s": info["solve_s"],
                               "statuses": info["statuses"]}.items():
                trace[key].append(value)
            if (step + 1) % 10 == 0 or step == 0 or step + 1 == steps:
                write_json(progress_path, {"seed": seed, "step": step + 1, "steps": steps,
                                          "pid": os.getpid(), "updated": stamp(),
                                          "last_step_compute_s": elapsed,
                                          "last_statuses": info["statuses"]})
    finally:
        env.close()
    return {**{k: np.asarray(v) for k, v in trace.items()}, "target": target,
            "initial_qpos": initial_qpos, "initial_qvel": initial_qvel}


def run_one(root_str, plan, name, seed, steps=None, phase="development"):
    root = Path(root_str)
    steps = plan["steps"] if steps is None else steps
    folder = root / phase / name
    folder.mkdir(parents=True, exist_ok=True)
    record = folder / f"seed{seed}.json"
    trace_path = record.with_suffix(".npz")
    config = plan["candidates"][name]
    if record.exists() and trace_path.exists():
        row = json.loads(record.read_text())
        if (row["parameters"], row["seed"], row["steps"]) != (config, seed, steps):
            raise ValueError("Incompatible cached rollout")
        return row
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        trace = rollout(SelectConfig(**config), seed, steps, root / "live" / f"{name}_seed{seed}.json")
    row = {"candidate": name, "seed": seed, "steps": steps, "parameters": config,
           "phase": phase, "completed": stamp(), "wall_time_s": time.perf_counter() - start,
           "trace_path": str(trace_path), **summarize(trace, plan)}
    tmp = trace_path.with_suffix(".partial.npz")
    np.savez_compressed(tmp, **trace, summary=np.array(json.dumps(row)))
    tmp.replace(trace_path)
    write_json(record, row)
    print(f"{stamp()} {phase} {name} seed={seed} success={row['success']} "
          f"tail={row['final100_mm']:.2f}mm failed_QPs={row['unsolved_subproblems']}/{row['subproblems']}", flush=True)
    return row


def score(rows, name):
    return (-sum(r["success"] for r in rows), float(np.mean([r["common_task_cost"] for r in rows])), name)


def choose_finalists(rows_by_name, plan):
    if plan["full_panel_policy"] == "all":
        return list(plan["candidates"])
    alternatives = [n for n in plan["candidates"] if n != "paper_weights"]
    return ["paper_weights", *sorted(alternatives, key=lambda n: score(rows_by_name[n], n))[:2]]


def report(root, plan):
    rows_by_name = {}
    entries = []
    for name, params in plan["candidates"].items():
        rows = [json.loads(p.read_text()) for p in sorted((root / "development" / name).glob("seed*.json"))]
        rows_by_name[name] = rows
        successes = sum(r["success"] for r in rows)
        entries.append({"candidate": name, "completed": len(rows), "requested": 100, "successes": successes,
                        "success_rate_completed": successes / len(rows) if rows else None,
                        "complete_100": {r["seed"] for r in rows} == set(plan["seeds"]),
                        "mean_common_cost": float(np.mean([r["common_task_cost"] for r in rows])) if rows else None,
                        "unsolved_subproblems": sum(r["unsolved_subproblems"] for r in rows),
                        "subproblems": sum(r["subproblems"] for r in rows), "parameters": params})
    full = [e for e in entries if e["complete_100"]]
    best = min(full, key=lambda e: score(rows_by_name[e["candidate"]], e["candidate"])) if full else None
    finalists_path = root / "finalists.json"
    finalists = json.loads(finalists_path.read_text())["candidates"] if finalists_path.exists() else []
    complete = bool(finalists and all(next(e for e in entries if e["candidate"] == n)["complete_100"] for n in finalists))
    payload = {"updated": stamp(), "complete": complete, "development_only": True,
               "phase": "complete" if complete else ("full_panel" if finalists else "screening"),
               "completed_rollouts": sum(e["completed"] for e in entries), "candidates": entries,
               "finalists": finalists, "best_completed_100_seed_candidate": best,
               "selection_caveat": plan["selection_caveat"], "timing_caveat": plan["timing_caveat"]}
    write_json(root / "summary.json", payload)
    if complete:
        write_json(root / "best_parameters.json", {"parameters": best["parameters"],
                   "candidate": best["candidate"], "successes": best["successes"], "trials": 100,
                   "seeds": plan["seeds"], "development_only": True, "selection_caveat": plan["selection_caveat"]})
    return payload


def run(root, plan):
    with single_writer(root):
        check_manifest(root, plan)
        with ProcessPoolExecutor(max_workers=plan["workers"], mp_context=multiprocessing.get_context("spawn"),
                                 initializer=init_worker, initargs=(plan["data_path"],)) as pool:
            def batch(tasks):
                futures = [pool.submit(run_one, str(root), plan, name, seed) for name, seed in tasks]
                for future in as_completed(futures):
                    future.result()
                    report(root, plan)
            print(stamp(), "Screening six fixed candidates; no adaptive expansion.", flush=True)
            batch([(name, seed) for seed in plan["screening_seeds"] for name in plan["candidates"]])
            rows = {n: [json.loads((root / "development" / n / f"seed{s}.json").read_text())
                        for s in plan["screening_seeds"]] for n in plan["candidates"]}
            finalists = choose_finalists(rows, plan)
            write_json(root / "finalists.json", {"candidates": finalists, "screening_seeds": plan["screening_seeds"],
                       "ranking": plan["ranking"], "selected": stamp()})
            print(stamp(), "Full 100-seed candidates:", finalists, flush=True)
            batch([(n, s) for s in plan["seeds"] if s not in plan["screening_seeds"] for n in finalists])
        payload = report(root, plan)
        print(stamp(), "COMPLETE", json.dumps(payload["best_completed_100_seed_candidate"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("init", "smoke", "run", "report"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--all-candidates", action="store_true", help="init only: all six candidates on 100 seeds")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if args.phase == "init":
        initialize(root, args.all_candidates)
        return
    plan = json.loads((root / "plan.json").read_text())
    validate_plan(plan)
    if args.phase == "report":
        print(json.dumps(report(root, plan), indent=2))
    elif args.phase == "smoke":
        with single_writer(root):
            check_manifest(root, plan)
            init_worker(plan["data_path"])
            for name in ("paper_weights", "prior_promising"):
                run_one(str(root), plan, name, 210, steps=10, phase="smoke")
    else:
        try:
            run(root, plan)
        except Exception as exc:
            write_json(root / "failure.json", {"time": stamp(), "error": repr(exc)})
            raise


if __name__ == "__main__":
    main()
