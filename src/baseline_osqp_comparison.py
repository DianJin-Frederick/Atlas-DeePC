"""OSQP-only solver ablation of the two zero-regularization baseline variants."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import cvxpy as cp
import numpy as np
import osqp
from threadpoolctl import threadpool_limits

import baseline_atlas_deepc as baseline
import baseline_waypoint_comparison as paired
import expert_pipeline_search as pipe
import routed_deepc_experiment as routed
from baseline_1000_benchmark import summarize_timing
from candidate_goal_routing_experiment import file_digest, strict_success


VERSION = "baseline_osqp_default_v1"
RESULT_ROOT = baseline.RESULT_ROOT / "osqp_waypoint_comparison"
_controller = _folder = _limits = None


@contextmanager
def use_osqp(records):
    """Change only the solver, scoped to an isolated experiment worker."""
    controller_class = pipe.DeePCController
    problem_solve = cp.Problem.solve

    def factory(*args, **kwargs):
        controller = controller_class(*args, **kwargs)
        controller.solver = cp.OSQP
        return controller

    def tracked_solve(problem, *args, **kwargs):
        assert kwargs["solver"] == cp.OSQP
        assert kwargs["max_iter"] == 10000
        started = time.perf_counter()
        error = None
        try:
            return problem_solve(problem, *args, **kwargs)
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            stats = problem.solver_stats
            info = getattr(getattr(stats, "extra_stats", None), "info", None)
            records.append({
                "solver": "OSQP", "status": problem.status, "error": error,
                "problem_solve_wall_s": time.perf_counter() - started,
                "solver_reported_time_s": getattr(stats, "solve_time", None),
                "iterations": getattr(stats, "num_iters", None),
                "primal_residual": getattr(info, "prim_res", None),
                "dual_residual": getattr(info, "dual_res", None),
            })

    with patch.object(pipe, "DeePCController", factory), patch.object(cp.Problem, "solve", tracked_solve):
        yield


def prepare(output_dir=RESULT_ROOT):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = {
        "version": VERSION, "baseline_version": baseline.BASELINE_VERSION,
        "config": asdict(baseline.default_config()),
        "solver": "OSQP", "solver_options": {"max_iter": 10000},
        "other_solver_options": "CVXPY OSQP adapter defaults; fresh Problem per step",
        "cvxpy_version": cp.__version__, "osqp_version": osqp.__version__,
        "data_sha256": file_digest(baseline.DATA_PATH), "atlas_sha256": file_digest(baseline.ATLAS_PATH),
        "source_sha256": {name: file_digest(baseline.ROOT / "src" / name) for name in (
            "baseline_osqp_comparison.py", "baseline_waypoint_comparison.py", "baseline_atlas_deepc.py",
            "routed_deepc_experiment.py", "expert_pipeline_search.py",
            "candidate_goal_routing_experiment.py", "stitched_atlas_mpc_experiment.py")},
    }
    assert "OSQP" in cp.installed_solvers() and signature["config"]["lam_g"] == 0
    path = output_dir / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != signature:
            raise ValueError("Incompatible OSQP cache; choose a new output directory")
    elif any(output_dir.glob("*/seed*_control_results.npz")):
        raise ValueError("Refusing unversioned OSQP traces")
    else:
        path.write_text(json.dumps(signature, indent=2))
    return signature


def initialize_worker(output_dir):
    global _controller, _folder, _limits
    _limits = threadpool_limits(limits=1)
    pipe.P = 10
    _controller = baseline.BaselineAtlasDeePC(
        np.load(baseline.DATA_PATH, mmap_mode="r"), routed.load_atlas(baseline.ATLAS_PATH))
    _folder = Path(output_dir)


def run_pair(seed):
    rows = []
    for mode in (("terminal", "graph") if seed % 2 else ("graph", "terminal")):
        folder = _folder / mode
        folder.mkdir(exist_ok=True)
        path = folder / f"seed{seed}_control_results.npz"
        if not path.exists():
            records = []
            with use_osqp(records):
                row = _controller.run(seed, path, reference_mode=mode)
            assert len(records) == _controller.config.steps
            with np.load(path, allow_pickle=True) as trace:
                arrays = {key: trace[key] for key in trace.files if key != "summary"}
            row.update(solver="OSQP", solver_ablation_version=VERSION,
                       solver_options={"max_iter": 10000},
                       solver_exception_steps=sum(r["error"] is not None for r in records),
                       optimal_inaccurate_steps=sum(r["status"] == "optimal_inaccurate" for r in records))
            arrays["solver_diagnostics"] = np.array(json.dumps(records))
            np.savez_compressed(path, **arrays, summary=np.array(json.dumps(row)))
        row = paired.validate_trace(path, mode, seed, _controller)
        assert row["solver"] == "OSQP" and row["solver_ablation_version"] == VERSION
        with np.load(path, allow_pickle=True) as trace:
            assert len(json.loads(trace["solver_diagnostics"].item())) == row["steps"]
        rows.append(row)
    np.testing.assert_array_equal(rows[0]["exact_target_xy"], rows[1]["exact_target_xy"])
    return rows


def run_panel(seeds=range(211, 1211), output_dir=RESULT_ROOT, workers=4):
    seeds = list(map(int, seeds))
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and unique")
    output_dir = Path(output_dir)
    signature = prepare(output_dir)
    rows = []
    print(f"OSQP: {len(seeds)} paired seeds, {workers} workers, lambda_g=0", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker, initargs=(output_dir,)) as pool:
        futures = [pool.submit(run_pair, seed) for seed in seeds]
        try:
            for index, future in enumerate(as_completed(futures), 1):
                rows.extend(future.result())
                if index % 25 == 0 or index == len(seeds):
                    print(f"{index}/{len(seeds)} pairs: " + str({mode: {
                        "successes": sum(strict_success(r) for r in rows if r["reference_mode"] == mode),
                        "unsolved_qps": sum(r["unsolved_steps"] for r in rows if r["reference_mode"] == mode)
                    } for mode in paired.METHODS}), flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    payload = {"version": VERSION, "signature": signature, "seeds": seeds, "workers": workers,
               "numerical_threads": 1, "panels": {},
               "fairness_audit": routed.audit_iid_library(np.load(baseline.DATA_PATH), eval_seeds=seeds)}
    for mode in paired.METHODS:
        selected = sorted([r for r in rows if r["reference_mode"] == mode], key=lambda r: r["seed"])
        payload["panels"][mode] = {
            "method": paired.METHODS[mode], "solver": "OSQP", "results": selected,
            "successes": sum(strict_success(r) for r in selected),
            "unsolved_qps": sum(r["unsolved_steps"] for r in selected),
            "timing": summarize_timing([r["result_path"] for r in selected]),
        }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=211)
    parser.add_argument("--seed-stop", type=int, default=1210)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()
    run_panel(args.seeds or range(args.seed_start, args.seed_stop+1), args.output_dir, args.workers)
