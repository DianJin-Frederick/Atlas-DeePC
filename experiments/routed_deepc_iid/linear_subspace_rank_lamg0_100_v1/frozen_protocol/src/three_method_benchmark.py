"""Resumable three-method tuning, freezing, testing, and reporting.

Run --help or open comparison_three_methods.ipynb. All files live in a new
experiment directory; no legacy notebook or rollout is changed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import fcntl
import hashlib
from itertools import combinations, product
import json
import math
import os
from pathlib import Path
import platform
import re
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

import cvxpy as cp
import gymnasium as gym
import numpy as np
from threadpoolctl import threadpool_limits

import baseline_atlas_deepc as baseline
import expert_pipeline_search as pipe
import routed_deepc_experiment as routed
from faithful_dpc_controllers import (
    ContextualConfig, ContextualDPC, OUTPUT_INDICES, SelectConfig, SelectDPC,
    native_library, native_reference, reconstruct_offline_episodes,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments/three_method_faithful_v1"
METHODS = ("select_dpc", "contextual_dpc", "atlas_deepc")
LABELS = {"select_dpc": "Select-DPC", "contextual_dpc": "Contextual DPC", "atlas_deepc": "Atlas-DeePC"}
VERSION = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def single_writer(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A benchmark is already running in this directory") from exc
        yield


def default_plan():
    return {
        "version": VERSION,
        "data_path": str(baseline.DATA_PATH), "atlas_path": str(baseline.ATLAS_PATH),
        "steps": 320, "input_limit": 1.0,
        "smoke_seed": 90001, "warmup_seed": 90000,
        "development_seeds": list(range(91001, 91033)),
        "test_seeds": list(range(100001, 101001)),
        "screening_seeds": 8, "finalists": 5,
        "evaluation_cost": {"q_xy": 100000.0, "r": 0.0001, "r_delta": 5.0},
        "selection_rule": "maximize development sustained successes; then minimize mean common task cost; then config hash",
        "numerical_threads": 1,
        "methods": {
            "select_dpc": {"base": asdict(SelectConfig()), "stages": [
                {"lambda_g1": [0.1, 10., 1000.], "lambda_projection": [100., 10000., 1000000.],
                 "lambda_slack": [50000., 5000000., 500000000.]},
                {"columns": [100, 200, 400], "q_xy": [4000., 40000., 400000.], "r": [1., 10., 100.]},
            ]},
            "contextual_dpc": {"base": asdict(ContextualConfig()), "stages": [
                {"lambda_g_bar": [0.001, 1., 1000.], "lambda_slack": [0.001, 1., 1000.],
                 "columns": [100, 200, 400]},
                {"q_xy": [0.1, 1., 10.], "r": [0.01, 0.1, 1.], "horizon": [10, 15, 20]},
            ]},
            "atlas_deepc": {"base": {k: v for k, v in asdict(baseline.default_config()).items()
                                         if k not in ("seed", "steps")}, "stages": [
                {"lam_g": [0.0005, 0.05, 5.], "lam_ini": [1000., 100000., 10000000.],
                 "lam_du": [0.05, 5., 500.]},
                {"q_xy": [10000., 100000., 1000000.], "q_data_goal_joint": [500., 5000., 50000.],
                 "r": [0.000001, 0.0001, 0.01]},
            ]},
        },
    }


def validate_plan(plan):
    if plan["version"] != VERSION or set(plan["methods"]) != set(METHODS):
        raise ValueError("Expected exactly the three benchmark methods")
    dev, test = plan["development_seeds"], plan["test_seeds"]
    if not dev or not test or len(set(dev)) != len(dev) or len(set(test)) != len(test):
        raise ValueError("Nonempty unique development/test seeds are required")
    if set(dev) & set(test):
        raise ValueError("Development and test seeds overlap")
    reserved = {plan["smoke_seed"], plan["warmup_seed"]}
    if len(reserved) != 2 or reserved & (set(dev) | set(test)):
        raise ValueError("Smoke/warm-up seeds must be separate from development and test")
    if (set(dev) | set(test) | reserved) & set(range(2026, 2226)):
        raise ValueError("A rollout seed overlaps the offline collection seeds")
    if not 1 <= plan["screening_seeds"] <= len(dev) or plan["finalists"] < 1:
        raise ValueError("Invalid screening or finalist budget")
    if plan["steps"] < 100 or plan["input_limit"] != 1.0:
        raise ValueError("The fixed Reacher task uses >=100 steps and |torque|<=1")
    if plan["numerical_threads"] != 1:
        raise ValueError("The controlled timing protocol requires one numerical thread")
    expected_atlas = {"clusters": 48, "svd_rank": 60, "formulation": "svd_current", "target_matrix_mode": "current"}
    for method, spec in plan["methods"].items():
        for config in [spec["base"], *[c for grid in spec["stages"] for c in grid_configs(spec["base"], grid)]]:
            if config.get("u_limit", config.get("input_limit")) != plan["input_limit"]:
                raise ValueError("Input bounds cannot be tuned between methods")
            if method == "atlas_deepc":
                if any(config[k] != v for k, v in expected_atlas.items()):
                    raise ValueError("Preserve the current 48-chart/rank-60 Atlas architecture")
                routed.RoutedDeePCConfig(**config)
            elif method == "select_dpc":
                c = SelectConfig(**config)
                if not 1 <= c.columns <= 400 or c.iterations < 1 or c.past != 2:
                    raise ValueError("Select requires raw <=400 columns, positive refinements, and two-step initialization")
            else:
                c = ContextualConfig(**config)
                if c.formulation not in ("paper_eq4", "public_code_qp") or c.columns < 1 or c.past != 2:
                    raise ValueError("Invalid contextual formulation or dimensions")


def grid_configs(base, grid):
    if any(not isinstance(values, list) or not values for values in grid.values()):
        raise ValueError("Every grid axis must have a nonempty list")
    keys = list(grid)
    candidates = [dict(base, **dict(zip(keys, values))) for values in product(*(grid[k] for k in keys))]
    # The incumbent is always eligible and never lost by a subsequent grid.
    return list({digest(c): c for c in [base, *candidates]}.values())


def provenance(plan):
    files = ["three_method_benchmark.py", "faithful_dpc_controllers.py", "baseline_atlas_deepc.py",
             "routed_deepc_experiment.py", "expert_pipeline_search.py", "candidate_goal_routing_experiment.py",
             "stitched_atlas_mpc_experiment.py", "local_affine_atlas_experiment.py"]
    return {"version": VERSION, "plan_sha256": digest(plan),
            "source_sha256": {name: file_digest(ROOT / "src" / name) for name in files},
            "data_sha256": file_digest(plan["data_path"]), "atlas_sha256": file_digest(plan["atlas_path"]),
            "python": platform.python_version(), "numpy": np.__version__, "cvxpy": cp.__version__,
            "gymnasium": gym.__version__, "platform": platform.platform(),
            "baseline_version": baseline.BASELINE_VERSION}


def check_manifest(root, plan):
    signature = provenance(plan)
    path = root / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != signature:
        raise ValueError("Plan, source, data, or runtime changed. Use a new output directory; cached results cannot be mixed.")
    if not path.exists():
        write_json(path, signature)
    return signature


def measured_metrics(y, u, target, times, prediction=None, previous_action=None):
    error = np.linalg.norm(np.asarray(y)[:, 6:8] - target, axis=1)
    previous = np.zeros(2) if previous_action is None else previous_action
    return {"error": error, "du": np.diff(np.vstack([previous, u]), axis=0)}


def summarize_trace(trace, plan):
    y, u, target = trace["y"], trace["u"], trace["target"]
    physical = measured_metrics(y, u, target, trace["control_compute_time_s"])
    error, du = physical["error"], physical["du"]
    weights = plan["evaluation_cost"]
    tracking = float(weights["q_xy"] * np.sum(error**2))
    effort = float(weights["r"] * np.sum(u**2))
    smoothness = float(weights["r_delta"] * np.sum(du**2))
    full_times = 1000 * np.asarray(trace["control_compute_time_s"])
    times = full_times[5:] if len(full_times) > 5 else full_times
    good = ("optimal", "optimal_inaccurate")
    statuses = np.asarray(trace["statuses"], dtype=str)
    tail = lambda n: error[-min(n, len(error)):]
    success30 = len(error) >= 30 and np.mean(tail(30)) < .01 and np.max(tail(30)) < .02
    success100 = len(error) >= 100 and np.mean(tail(100)) < .01 and np.max(tail(100)) < .02
    prediction_errors = np.asarray(trace.get("prediction_error_xy", []))
    finite = prediction_errors[np.isfinite(prediction_errors)]
    return {
        "steps": len(error), "success": bool(success30 and success100),
        "success_30": bool(success30), "success_100": bool(success100),
        "final100_mm": float(1000 * np.mean(tail(100))),
        "common_task_cost": tracking + effort + smoothness,
        "tracking_cost": tracking, "input_cost": effort, "input_difference_cost": smoothness,
        "prediction_mm": float(1000 * finite.mean()) if finite.size else None,
        "prediction_valid_fraction": float(finite.size / len(error)),
        "unsolved_subproblems": int(np.sum(~np.isin(statuses, good))),
        "subproblems": int(statuses.size),
        "input_bound_violations": int(np.sum(np.any(np.abs(u) > plan["input_limit"] + 1e-4, axis=1))),
        "median_compute_ms": float(np.median(times)), "p99_compute_ms": float(np.percentile(times, 99)),
        "max_compute_ms": float(np.max(full_times)),
    }


class Runner:
    def __init__(self, plan):
        self.plan = plan
        self.H = np.load(plan["data_path"], mmap_mode="r")
        self.atlas = routed.load_atlas(Path(plan["atlas_path"]))
        self.inputs, self.outputs = reconstruct_offline_episodes(self.H)
        self.input_samples = np.vstack(self.inputs)
        self.output_samples = np.vstack(self.outputs)[:, OUTPUT_INDICES]
        self.libraries = {}
        self.atlas_cache = None

    def run(self, method, parameters, seed, path, steps=None):
        steps = self.plan["steps"] if steps is None else steps
        stage = time.perf_counter()
        if method == "atlas_deepc":
            config = routed.RoutedDeePCConfig(**parameters, seed=seed, steps=steps)
            # Call the current baseline itself: no duplicate chart/reference code.
            cache_key = (config.svd_rank, config.svd_bound_scale, config.edge_count_threshold)
            if self.atlas_cache is None or self.atlas_cache[0] != cache_key:
                self.atlas_cache = (cache_key, baseline.BaselineAtlasDeePC(self.H, self.atlas, config))
            controller = self.atlas_cache[1]
            controller.config = config
            setup_time = time.perf_counter() - stage
            row = controller.run(seed, path)
            with np.load(path, allow_pickle=True) as saved:
                trace = {k: saved[k] for k in saved.files if k != "summary"}
            # Observation target is fixed even when Atlas terminal posture changes.
            env = gym.make(pipe.REACHER_ENV_ID)
            try:
                obs, _ = env.reset(seed=seed)
                target = np.asarray(obs[4:6]).copy()
            finally:
                env.close()
            trace["target"] = target
            trace["statuses"] = trace["status"].astype(str)
            metadata = {"matrix_geometry": row["matrix_geometry"], "baseline_version": row["baseline_version"],
                        "candidate_count": row["candidate_count"], "dynamic_terminal_reference": row["dynamic_terminal_reference"],
                        "goal_setup_ms": row.get("goal_setup_time_ms", 0.), "offline_or_cached_setup_ms": 1000 * setup_time}
        else:
            c = SelectConfig(**parameters) if method == "select_dpc" else ContextualConfig(**parameters)
            key = (c.past, c.horizon)
            if key not in self.libraries:
                self.libraries[key] = native_library(self.inputs, self.outputs, *key)
            Hu, Hy = self.libraries[key]
            controller = (SelectDPC(Hu, Hy, c) if method == "select_dpc" else
                          ContextualDPC(Hu, Hy, self.input_samples, self.output_samples, c))
            metadata = {"representation": "raw_trajectory_coefficients", "coefficient_count": c.columns,
                        "online_svd_reduction": False, "graph_routing": False,
                        "reference": "requested fingertip target, zero velocity, no posture objective",
                        "formulation": "choose_wisely_raw" if method == "select_dpc" else c.formulation,
                        "offline_or_cached_setup_ms": 1000 * (time.perf_counter() - stage)}
            trace = self.raw_rollout(controller, seed, steps)
        metrics = summarize_trace(trace, self.plan)
        row = {"method": method, "seed": seed, "parameters": parameters, **metadata, **metrics}
        # Store all step timings for pooled median/p99/max, and post-action y/u.
        np.savez_compressed(path, **trace, summary=np.array(json.dumps(row, allow_nan=False)))
        return row

    @staticmethod
    def raw_rollout(controller, seed, steps):
        env = gym.make(pipe.REACHER_ENV_ID, max_episode_steps=steps + 7)
        trace = {k: [] for k in ("y", "u", "pred", "prediction_error_xy", "control_compute_time_s",
                                  "selection_time_s", "matrix_preparation_time_s", "solve_time_s", "statuses")}
        try:
            obs, _ = env.reset(seed=seed)
            target = np.asarray(obs[4:6]).copy()
            uh, yh = [], []
            # Same two zero-input physical steps as the unchanged Atlas runner.
            for _ in range(2):
                obs, *_ = env.step(np.zeros(2))
                uh.append(np.zeros(2))
                yh.append(pipe.reacher_output_from_obs(obs))
            reference = native_reference(target)
            for _ in range(steps):
                start = time.perf_counter()
                past = controller.config.past
                upast = np.asarray(uh[-past:])
                ypast = np.asarray(yh[-past:])[:, OUTPUT_INDICES]
                action, prediction, info = controller.step(ypast[-1], reference, upast=upast, ypast=ypast)
                compute = time.perf_counter() - start
                if np.asarray(action).shape != (2,) or not np.all(np.isfinite(action)):
                    raise RuntimeError("Controller returned invalid action")
                obs, *_ = env.step(action)
                y = pipe.reacher_output_from_obs(obs)
                uh.append(action.copy())
                yh.append(y)
                trace["y"].append(y)
                trace["u"].append(action)
                trace["pred"].append(prediction)
                trace["prediction_error_xy"].append(np.linalg.norm(prediction[4:6] - y[6:8]))
                trace["control_compute_time_s"].append(compute)
                trace["selection_time_s"].append(info["selection_s"])
                trace["matrix_preparation_time_s"].append(info["matrix_s"])
                trace["solve_time_s"].append(info["solve_s"])
                trace["statuses"].append(info["statuses"])
        finally:
            env.close()
        return {**{k: np.asarray(v) for k, v in trace.items()}, "target": target}


def run_saved(runner, root, phase, method, config, seed, steps=None):
    config_id = digest(config)[:16]
    folder = root / phase / method / config_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"seed{seed}.npz"
    record = path.with_suffix(".json")
    if record.exists() and path.exists():
        row = json.loads(record.read_text())
        if row["parameters"] != config or row["seed"] != seed or row["steps"] != (steps or runner.plan["steps"]):
            raise ValueError("Incompatible cached rollout")
        return row
    row = runner.run(method, config, seed, path, steps)
    row["trace_path"] = str(path.resolve())
    write_json(record, row)
    print(f"{phase} {method} {config_id} seed={seed}: success={row['success']}, "
          f"tail={row['final100_mm']:.2f} mm, failed QPs={row['unsolved_subproblems']}", flush=True)
    return row


def selection_score(rows, config):
    return (-sum(row["success"] for row in rows),
            float(np.mean([row["common_task_cost"] for row in rows])), digest(config))


def tune(runner, root):
    plan = runner.plan
    if (root / "frozen.json").exists():
        raise ValueError("Tuning is frozen. Use a new directory for another study.")
    development = plan["development_seeds"]
    screening = development[:plan["screening_seeds"]]
    selected = {}
    history = {}
    for method in METHODS:
        incumbent = plan["methods"][method]["base"]
        history[method] = []
        for stage_index, grid in enumerate(plan["methods"][method]["stages"]):
            candidates = grid_configs(incumbent, grid)
            screened = []
            for config in candidates:
                rows = [run_saved(runner, root, "development", method, config, seed) for seed in screening]
                screened.append((selection_score(rows, config), config))
            screened.sort(key=lambda item: item[0])
            finalists = [config for _, config in screened[:plan["finalists"]]]
            # Reserve one finalist slot for the incumbent, keeping the budget fixed.
            if incumbent not in finalists:
                finalists[-1] = incumbent
            evaluated = []
            for config in finalists:
                rows = [run_saved(runner, root, "development", method, config, seed) for seed in development]
                evaluated.append((selection_score(rows, config), config))
            evaluated.sort(key=lambda item: item[0])
            incumbent = evaluated[0][1]
            stage = {"stage": stage_index + 1, "candidates": len(candidates),
                     "screening_seeds": screening, "finalist_seeds": development,
                     "screened": [{"parameters": c, "score": score} for score, c in screened],
                     "finalists": [{"parameters": c, "score": score} for score, c in evaluated],
                     "winner": incumbent}
            history[method].append(stage)
            write_json(root / "tuning_progress.json", history)
        selected[method] = incumbent
    write_json(root / "tuning.json", {"plan_sha256": digest(plan), "selected": selected, "history": history})


def freeze(root, plan):
    tuning = json.loads((root / "tuning.json").read_text())
    if tuning["plan_sha256"] != digest(plan) or set(tuning["selected"]) != set(METHODS):
        raise ValueError("Incomplete or incompatible tuning")
    frozen = {"plan_sha256": digest(plan), "tuning_sha256": file_digest(root / "tuning.json"),
              "selected": tuning["selected"], "test_seeds": plan["test_seeds"],
              "manifest_sha256": file_digest(root / "manifest.json")}
    if (root / "frozen.json").exists() and json.loads((root / "frozen.json").read_text()) != frozen:
        raise ValueError("Cannot change a frozen test protocol")
    write_json(root / "frozen.json", frozen)
    print("Frozen one development-selected configuration per method. Test data have not been evaluated.")


def evaluate(runner, root):
    frozen = json.loads((root / "frozen.json").read_text())
    if (frozen["plan_sha256"] != digest(runner.plan)
            or frozen["manifest_sha256"] != file_digest(root / "manifest.json")
            or frozen["tuning_sha256"] != file_digest(root / "tuning.json")):
        raise ValueError("Frozen settings changed")
    # Rotate method order by seed to reduce ordering/load effects in serial timing.
    for index, seed in enumerate(frozen["test_seeds"]):
        order = METHODS[index % 3:] + METHODS[:index % 3]
        for method in order:
            run_saved(runner, root, "test", method, frozen["selected"][method], seed)


def wilson(successes, total):
    if total == 0:
        return [0., 1.]
    z = 1.959963984540054
    rate, denominator = successes / total, 1 + z*z/total
    center = (rate + z*z/(2*total)) / denominator
    half = z * math.sqrt(rate*(1-rate)/total + z*z/(4*total*total)) / denominator
    return [max(0., center-half), min(1., center+half)]


def report(root, plan):
    frozen_path = root / "frozen.json"
    if not frozen_path.exists():
        print("No frozen test results yet. Review tuning_progress.json for development results.")
        return None
    frozen = json.loads(frozen_path.read_text())
    if frozen["plan_sha256"] != digest(plan):
        raise ValueError("Report plan differs from the frozen experiment")
    by_method = {}
    for method in METHODS:
        folder = root / "test" / method / digest(frozen["selected"][method])[:16]
        by_method[method] = {seed: json.loads((folder / f"seed{seed}.json").read_text())
                             for seed in frozen["test_seeds"] if (folder / f"seed{seed}.json").exists()}
    paired = sorted(set.intersection(*(set(rows) for rows in by_method.values())))
    if not paired:
        print("No complete three-method seed triplets yet.")
        return None
    summaries = []
    for method in METHODS:
        rows = [by_method[method][seed] for seed in paired]
        successes = sum(r["success"] for r in rows)
        pooled, all_times = [], []
        for row in rows:
            with np.load(row["trace_path"]) as trace:
                t = 1000 * trace["control_compute_time_s"]
            pooled.extend(t[5:])
            all_times.extend(t)
        summaries.append({"method": method, "label": LABELS[method], "successes": successes,
                          "total": len(paired), "success_rate": successes / len(paired),
                          "wilson_95": wilson(successes, len(paired)),
                          "median_final100_mm": float(np.median([r["final100_mm"] for r in rows])),
                          "median_common_cost": float(np.median([r["common_task_cost"] for r in rows])),
                          "mean_common_cost": float(np.mean([r["common_task_cost"] for r in rows])),
                          "median_compute_ms": float(np.median(pooled)),
                          "p99_compute_ms": float(np.percentile(pooled, 99)),
                          "max_compute_ms": float(np.max(all_times)),
                          "unsolved_subproblems": sum(r["unsolved_subproblems"] for r in rows),
                          "input_bound_violations": sum(r["input_bound_violations"] for r in rows)})
    pairs = []
    for first, second in combinations(METHODS, 2):
        a = sum(by_method[first][s]["success"] and not by_method[second][s]["success"] for s in paired)
        b = sum(by_method[second][s]["success"] and not by_method[first][s]["success"] for s in paired)
        n = a+b
        p = min(1., 2*sum(math.comb(n, k) for k in range(min(a, b)+1)) / 2**n) if n else 1.
        pairs.append({"first": first, "second": second, "first_only": a, "second_only": b,
                      "exact_mcnemar_p": p})
    payload = {"complete": len(paired) == len(plan["test_seeds"]), "paired_seeds": paired,
               "methods": summaries, "paired_comparisons": pairs,
               "evaluation_cost": plan["evaluation_cost"],
               "timing": "pooled serial step times; median/p99 exclude first 5 steps per rollout; max includes all steps"}
    write_json(root / "summary.json", payload)
    make_figure(payload, root)
    print(json.dumps(payload["methods"], indent=2))
    return payload


def make_figure(payload, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = payload["methods"]
    labels = [r["label"] for r in rows]
    colors = ["#D55E00", "#CC79A7", "#0072B2"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), layout="constrained")
    rates = np.array([100*r["success_rate"] for r in rows])
    intervals = 100*np.array([r["wilson_95"] for r in rows])
    axes[0].bar(labels, rates, color=colors)
    axes[0].errorbar(range(3), rates, yerr=np.maximum(0, np.vstack([rates-intervals[:, 0], intervals[:, 1]-rates])),
                     fmt="none", color="black", capsize=4)
    axes[0].set(title="Sustained success", ylabel="%", ylim=(0, 105))
    for ax, field, title, units in zip(axes[1:], ["median_common_cost", "median_final100_mm", "median_compute_ms"],
                                     ["Median common task cost", "Median final-100 error", "Median online computation"],
                                     ["Fixed evaluation weights", "mm", "ms/step"]):
        values = [r[field] for r in rows]
        ax.bar(labels, values, color=colors)
        ax.set(title=title, ylabel=units, ylim=(0, max(1e-9, 1.15*max(values))))
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(axis="y", alpha=.2)
    status = "Final" if payload["complete"] else "PARTIAL paired-seed report"
    fig.suptitle(f"{status}: {len(payload['paired_seeds'])} seeds, independently tuned methods")
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"three_method_comparison.{suffix}", dpi=170)
    plt.close(fig)


def initialize(root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "plan.json"
    if path.exists():
        raise ValueError("A plan already exists; edit it before the first run or choose another directory")
    plan = default_plan()
    # File-name audit of prior project rollouts; this does not open test outcomes.
    prior = set()
    for parent, dirs, files in os.walk(ROOT / "experiments"):
        dirs[:] = [d for d in dirs if d != ".git"]
        for name in files:
            match = re.search(r"seed_?(\d+)", name)
            if match:
                prior.add(int(match.group(1)))
    collision = prior & set(plan["test_seeds"])
    if collision:
        raise ValueError(f"Proposed test seeds already occur in project artifacts: {sorted(collision)[:10]}")
    validate_plan(plan)
    write_json(path, plan)
    write_json(root / "seed_audit.json", {"prior_seed_filename_count": len(prior),
                "test_seed_collisions": [], "scope": "filename scan of this project's experiments directory"})
    print(f"Created {path}. Review the grids before the first run.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["init", "smoke", "tune", "freeze", "test", "report"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-steps", type=int, default=12)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if args.phase == "init":
        with single_writer(root):
            initialize(root)
        return
    plan = json.loads((root / "plan.json").read_text())
    validate_plan(plan)
    with single_writer(root), threadpool_limits(limits=1):
        if args.phase == "report":
            # Figure-only edits must never force another controller rollout.
            report(root, plan)
            return
        check_manifest(root, plan)
        if args.phase == "freeze":
            freeze(root, plan)
            return
        runner = Runner(plan)
        if args.phase == "smoke":
            rows = [run_saved(runner, root, "smoke", m, plan["methods"][m]["base"],
                              plan["smoke_seed"], args.smoke_steps) for m in METHODS]
            write_json(root / "smoke_summary.json", {"validation_only": True, "rows": rows})
        else:
            for method in METHODS:
                parameters = (plan["methods"][method]["base"] if args.phase == "tune" else
                              json.loads((root / "frozen.json").read_text())["selected"][method])
                runner.run(method, parameters, plan["warmup_seed"], root / f"warmup_{method}.npz", steps=10)
            if args.phase == "tune":
                tune(runner, root)
            else:
                evaluate(runner, root)
                report(root, plan)


if __name__ == "__main__":
    main()
