"""GS-DeePC composite-region Reacher port of Zieglmeier et al., 2512.02797v2.

Independent Python implementation checked against public MATLAB commit below.
Keeps its raw mosaic matrices, quadratic regularizers, terminal equality and
zero-input solve-failure fallback. Reacher-specific settings are in Config.
No Atlas controller, graph, normalization, SVD reduction or online column search.
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

# Reuse data decoding, observation mapping and reporting utilities ONLY.
from atlas_deepc.benchmarks import contextual_dpc_reacher as utils
from atlas_deepc.common.metrics import strict_success, true_error_metrics

ROOT, DATA = utils.ROOT, utils.DATA
OUTPUT = ROOT / "experiments/reacher/gs_deepc_composite_xy_211_310_tini2_tf10_v1"
PAPER = "https://arxiv.org/abs/2512.02797v2"
UPSTREAM_COMMIT = "8738264c101c6e2942735ada3dc6bfb24ee5a101"
VERSION = "gs_deepc_composite_xy_v1"


@dataclass(frozen=True)
class Config:
    tini: int = 2
    horizon: int = 10
    regions: int = 8
    columns_per_region: int = 200
    warmup: int = 12
    q_xy: float = 100.0
    r: float = 0.05
    lambda_ini: float = 1e6
    lambda_g: float = 1e3
    xy_bound: float = 0.22
    tolerance: float = 1e-8
    max_iter: int = 1000
    accept_residual: float = 1e-4

    def validate(self):
        if min(self.tini, self.horizon, self.columns_per_region, self.max_iter) < 1:
            raise ValueError("Horizons, column count and solver iterations must be positive")
        if self.regions < 2 or self.warmup < self.tini:
            raise ValueError("Need >=2 elementary regions and warmup >= T_ini")
        if min(self.q_xy, self.r, self.lambda_g, self.lambda_ini,
               self.xy_bound, self.tolerance, self.accept_residual) <= 0:
            raise ValueError("Weights, output bound and tolerances must be positive")


def scheduling_variable(y8):
    """Wrapped second joint angle from measured cos(q2), sin(q2), radians."""
    a = np.asarray(y8)
    return np.arctan2(a[..., 3], a[..., 1])


def region_labels(rho, edges):
    rho = np.asarray(rho)
    if not np.isfinite(rho).all() or np.any(rho < edges[0]) or np.any(rho > edges[-1]):
        raise ValueError("Scheduler outside declared domain")
    # Half-open intervals; include the outermost upper endpoint.
    return np.minimum(np.searchsorted(edges, rho, side="right") - 1, len(edges) - 2)


def select_composite(rho, edges, current=None):
    """Retain current composite while inside; otherwise move toward rho.

    Extends upstream's one-neighbor move to handle multi-region jumps safely.
    Includes the final endpoint. No circular adjacency across -pi/pi.
    """
    region_labels(rho, edges)  # Validate even while retaining a composite.
    last = len(edges) - 3
    if current is None:
        for i in range(last + 1):
            if edges[i] <= rho < edges[i + 2] or (i == last and rho == edges[-1]):
                return i
    if not 0 <= current <= last:
        raise ValueError("Invalid composite index")
    while current < last and rho >= edges[current + 2]:
        current += 1
    while current > 0 and rho < edges[current]:
        current -= 1
    return current


class Library:
    def __init__(self, u, y8, config=Config()):
        config.validate()
        self.config = config
        self.edges = np.linspace(-np.pi, np.pi, config.regions + 1)
        u, y8 = np.asarray(u), np.asarray(y8)
        if u.ndim != 3 or y8.shape != u.shape[:2] + (8,) or u.shape[-1] != 2:
            raise ValueError("Expected episodic u[...,2], y[...,8]")
        if not np.isfinite(u).all() or not np.isfinite(y8).all():
            raise ValueError("Nonfinite offline samples")
        length = config.tini + config.horizon
        labels = region_labels(scheduling_variable(y8), self.edges)
        candidates = [[] for _ in range(config.regions)]
        # Enumerating all within-region windows is exactly the mosaic of the
        # Hankel matrices of maximal constant-label segments; never join gaps.
        for episode, lab in enumerate(labels):
            for start in range(len(lab) - length + 1):
                region = lab[start]
                if np.all(lab[start:start + length] == region):
                    candidates[region].append((episode, start))
        counts = [len(items) for items in candidates]
        if min(counts) < config.columns_per_region:
            raise ValueError(f"Insufficient regional data: {counts}; need {config.columns_per_region} each. "
                             "Use fewer regions/a shorter horizon, or collect suitable data; no global fallback.")
        # Upstream truncates the chronological mosaic to its first c columns.
        self.provenance = np.array([items[:config.columns_per_region] for items in candidates])
        self.hu = np.stack([np.column_stack([u[e, t:t+length].ravel() for e, t in ids])
                            for ids in self.provenance])
        self.hy = np.stack([np.column_stack([y8[e, t:t+length, 6:8].ravel() for e, t in ids])
                            for ids in self.provenance])
        self.composite_u = np.stack([np.concatenate(self.hu[i:i+2], axis=1)
                                     for i in range(config.regions - 1)])
        self.composite_y = np.stack([np.concatenate(self.hy[i:i+2], axis=1)
                                     for i in range(config.regions - 1)])
        # Singular values below are DIAGNOSTICS ONLY, never a model reduction.
        self.audit = {"episodes": len(u), "samples_per_episode": u.shape[1],
            "valid_columns_per_region": counts, "retained_per_region": config.columns_per_region,
            "active_columns": 2 * config.columns_per_region, "composite_regions": config.regions - 1,
            "edges_rad": self.edges.tolist(), "no_svd_reduction": True,
            "input_rank_required_at_L": 2 * length, "regions": []}
        for i, hu in enumerate(self.hu):
            s = np.linalg.svd(hu, compute_uv=False)
            self.audit["regions"].append({"region": i, "input_rank_at_L": int(np.linalg.matrix_rank(hu)),
                "input_min_singular_value": float(s[-1]),
                "unique_source_episodes": int(len(np.unique(self.provenance[i, :, 0])))})
        if any(row["input_rank_at_L"] < 2 * length for row in self.audit["regions"]):
            raise ValueError("A retained regional input matrix is row-rank deficient; inspect offline data")
        self.audit["excitation_caveat"] = "Full input rank at L is only a necessary diagnostic, not PE at L+n or proof of local LTI validity."

    @classmethod
    def from_path(cls, path=DATA, config=Config()):
        return cls(*utils.recover_episodes(np.load(path, mmap_mode="r")), config)

    def save(self, path):
        np.savez_compressed(path, hu=self.hu, hy=self.hy, composite_u=self.composite_u,
                            composite_y=self.composite_y, provenance=self.provenance, edges=self.edges)


class Controller:
    def __init__(self, library):
        start = time.perf_counter()
        self.library, self.config = library, library.config
        c = self.config
        columns, length = 2 * c.columns_per_region, c.tini + c.horizon
        self.hu, self.hy = cp.Parameter((2*length, columns)), cp.Parameter((2*length, columns))
        self.up, self.yp = cp.Parameter((c.tini, 2)), cp.Parameter((c.tini, 2))
        self.ref = cp.Parameter((c.horizon, 2))
        self.g, self.sigma = cp.Variable(columns), cp.Variable((c.tini, 2))
        self.u, self.y = cp.Variable((c.horizon, 2)), cp.Variable((c.horizon, 2))
        past = 2*c.tini
        self.constraints = [
            self.hu[:past] @ self.g == cp.vec(self.up, order="C"),
            self.hy[:past] @ self.g == cp.vec(self.yp + self.sigma, order="C"),
            self.hu[past:] @ self.g == cp.vec(self.u, order="C"),
            self.hy[past:] @ self.g == cp.vec(self.y, order="C"),
            self.y[-1] == self.ref[-1],  # Authors' MATLAB code, absent from printed Eq. (3).
            self.u >= -1, self.u <= 1,
            self.y >= -c.xy_bound, self.y <= c.xy_bound]
        cost = (c.q_xy*cp.sum_squares(self.y-self.ref) + c.r*cp.sum_squares(self.u)
                + c.lambda_g*cp.sum_squares(self.g) + c.lambda_ini*cp.sum_squares(self.sigma))
        self.problem = cp.Problem(cp.Minimize(cost), self.constraints)
        assert self.problem.is_qp() and self.problem.is_dpp()
        self.construction_s = time.perf_counter() - start
        self.current = None

    def action(self, u_history, y8_history, target):
        started = time.perf_counter()
        c = self.config
        select_start = time.perf_counter()
        rho = float(scheduling_variable(y8_history[-1]))
        previous = self.current
        self.current = select_composite(rho, self.library.edges, previous)
        selection_s = time.perf_counter() - select_start
        prepare_start = time.perf_counter()
        self.hu.value = self.library.composite_u[self.current]
        self.hy.value = self.library.composite_y[self.current]
        self.up.value = u_history[-c.tini:]
        self.yp.value = y8_history[-c.tini:, 6:8]
        self.ref.value = np.tile(target, (c.horizon, 1))
        preparation_s = time.perf_counter() - prepare_start
        solve_start = time.perf_counter()
        error = ""
        try:
            self.problem.solve(solver="CLARABEL", max_iter=c.max_iter, max_threads=1,
                tol_gap_abs=c.tolerance, tol_gap_rel=c.tolerance, tol_feas=c.tolerance,
                warm_start=False, enforce_dpp=True)
        except cp.error.SolverError as exc:
            error = str(exc)
        solve_s = time.perf_counter() - solve_start
        accepted = not error and self.problem.status in ("optimal", "optimal_inaccurate")
        residual = np.nan
        if accepted:
            residual = float(max(np.max(item.violation()) for item in self.constraints))
            accepted = residual <= c.accept_residual and all(
                np.isfinite(v.value).all() for v in (self.u, self.y, self.g, self.sigma))
        if accepted:
            commanded = self.u.value[0].copy()
            action = np.clip(commanded, -1, 1)  # Only numerical bound tolerance.
            plan_u, plan_y = self.u.value.copy(), self.y.value.copy()
            g, slack = self.g.value.copy(), self.sigma.value.copy()
        else:
            # Same physical fallback as upstream; always explicitly recorded.
            commanded, action = np.zeros(2), np.zeros(2)
            plan_u = np.full((c.horizon, 2), np.nan)
            plan_y = np.full((c.horizon, 2), np.nan)
            g, slack = np.full(2*c.columns_per_region, np.nan), np.full((c.tini, 2), np.nan)
        row = {"scheduler_q2": rho, "composite_index": self.current,
            "region_switched": previous is not None and self.current != previous,
            "commanded_u": commanded, "planned_u": plan_u, "planned_y": plan_y,
            "pred_xy": plan_y[0], "g_coefficients": g, "slack": slack,
            "status": "solver_error" if error else str(self.problem.status or "solver_error"), "solver_error": error,
            "fallback": not accepted, "max_constraint_violation": residual,
            "objective": float(self.problem.value) if accepted else np.nan,
            "selection_time_s": selection_s, "matrix_preparation_time_s": preparation_s,
            "solve_time_s": solve_s,
            "solver_internal_time_s": float(self.problem.solver_stats.solve_time or 0) if accepted else np.nan}
        row["control_compute_time_s"] = time.perf_counter() - started
        return action, row


def rollout_summary(a, seed, steps):
    distance = a["distance"]
    evaluable = len(distance) == steps and steps >= 100
    metrics = true_error_metrics(distance)
    for n in (30, 100):
        if len(distance) < n:
            for key in (f"mean_last_{n}", f"max_last_{n}", f"success_{n}"):
                metrics[key] = None
    return {"seed": seed, "steps": steps, **metrics, "success_evaluable": evaluable,
        "strict_success": bool(evaluable and strict_success(metrics)),
        "rollout_rmse_mm": float(1000*np.sqrt(np.mean(distance**2))),
        "fallback_steps": int(a["fallback"].sum()),
        "region_switches": int(a["region_switched"].sum()),
        "closed_loop_control_cost": float(a["stage_control_cost"].sum()),
        "xy_squared_error_sum_m2": float(np.sum(distance**2)),
        "input_squared_sum": float(np.sum(a["u"]**2))}


def run_seed(controller, seed, steps, directory):
    c = controller.config
    controller.current = None
    records, uh, yh = {}, [], []
    env = gym.make("Reacher-v5", max_episode_steps=c.warmup + steps)
    try:
        obs, _ = env.reset(seed=seed)
        initial, target = obs.copy(), obs[4:6].copy()
        for _ in range(c.warmup):
            obs, _, terminated, truncated, _ = env.step(np.zeros(2))
            if terminated or truncated:
                raise RuntimeError("Unexpected warmup termination")
            uh.append(np.zeros(2)); yh.append(utils.output_from_obs(obs))
        warmup_u, warmup_y = np.array(uh), np.array(yh)
        for step in range(steps):
            before = obs.copy()
            action, row = controller.action(np.array(uh[-c.tini:]), np.array(yh[-c.tini:]), target)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or (truncated and step != steps-1):
                raise RuntimeError("Unexpected rollout termination")
            y = utils.output_from_obs(obs)
            distance = np.linalg.norm(y[6:8]-target)
            tracking, effort = c.q_xy*distance**2, c.r*np.sum(action**2)
            row.update(u=action, y=y, ref_xy=target.copy(), observation_before=before,
                observation_after=obs.copy(), distance=distance,
                prediction_error_xy=np.linalg.norm(row["pred_xy"]-y[6:8]),
                stage_tracking_cost=tracking, stage_input_cost=effort, stage_control_cost=tracking+effort)
            for key, value in row.items():
                records.setdefault(key, []).append(value)
            uh.append(action.copy()); yh.append(y.copy())
            if step == 0 or (step+1) % 20 == 0 or step == steps-1:
                utils.write_json(directory / "progress.json", {"state": "running", "seed": seed,
                    "step": step+1, "steps": steps, "updated_unix": time.time()})
                print(f"seed {seed}: {step+1}/{steps}, {distance*1000:.2f} mm, "
                      f"composite {controller.current}, fallback={row['fallback']}", flush=True)
    finally:
        env.close()
    arrays = {key: np.asarray(value) for key, value in records.items()}
    summary = rollout_summary(arrays, seed, steps)
    path = directory / f"seed{seed}_control_results.npz"
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as f:
        np.savez_compressed(f, **arrays, summary=np.array(json.dumps(summary)),
            initial_observation=initial, target_xy=target, warmup_u=warmup_u, warmup_y=warmup_y)
    temporary.replace(path)
    return summary


def summarize(directory, seeds):
    rows, timing = [], {k: [] for k in ("control_compute_time_s", "selection_time_s",
                                      "matrix_preparation_time_s", "solve_time_s")}
    all_times = []
    for seed in seeds:
        path = Path(directory) / f"seed{seed}_control_results.npz"
        if not path.exists():
            continue
        with np.load(path) as a:
            rows.append(json.loads(a["summary"].item()))
            all_times.extend(a["control_compute_time_s"].tolist())
            for key in timing:
                timing[key].extend(a[key][5:].tolist())
    eligible = [r for r in rows if r["success_evaluable"]]
    successes = sum(r["strict_success"] for r in eligible)
    result = {"completed": len(rows), "requested": len(seeds), "complete": len(rows) == len(seeds),
        "success_evaluable_rollouts": len(eligible), "successes": successes,
        "success_rate": successes/len(eligible) if eligible else None, "results": rows,
        "error_quantiles_mm": dict(zip(["median", "p90", "p95", "p99"],
            np.percentile([1000*r["mean_last_100"] for r in eligible], [50,90,95,99]).tolist())) if eligible else None,
        "median_rollout_rmse_mm": float(np.median([r["rollout_rmse_mm"] for r in rows])) if rows else None,
        "median_closed_loop_control_cost": float(np.median([r["closed_loop_control_cost"] for r in rows])) if rows else None,
        "fallback_steps": sum(r["fallback_steps"] for r in rows),
        "timing_all_steps": utils.timing_stats(all_times) if all_times else None,
        "modular_timing_steps6_onwards": {k: utils.timing_stats(v) if v else None for k,v in timing.items()},
        "note": "All completed rollouts, including unsuccessful ones, included. Smoke runs <100 steps have no success-rate estimate."}
    utils.write_json(Path(directory)/"summary.json", result)
    return result


def run(config=Config(), seed_start=211, seed_stop=310, steps=300, output=OUTPUT, data=DATA, prepare_only=False):
    import fcntl
    config.validate()
    seeds = list(range(seed_start, seed_stop+1))
    if not seeds or steps < 1:
        raise ValueError("Invalid seed range or steps")
    output, data = Path(output).resolve(), Path(data).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output/"run.lock").open("a+") as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = {"version": VERSION, "paper": PAPER, "upstream_commit": UPSTREAM_COMMIT,
            "source_sha256": utils.digest(__file__), "data_sha256": utils.digest(data),
            "utilities_sha256": utils.digest(utils.__file__),
            "metrics_sha256": utils.digest(Path(__file__).parents[1]/"common/metrics.py"),
            "config": asdict(config), "seeds": seeds, "steps": steps,
            "packages": {p: importlib.metadata.version(p) for p in
                         ("numpy", "scipy", "cvxpy", "clarabel", "gymnasium", "mujoco")},
            "solver": "CLARABEL", "threads": 1, "environment": "Reacher-v5",
            "reference": "fixed final fingertip xy; no posture or waypoints",
            "scheduler": "atan2(sin(q2),cos(q2)), uniform [-pi,pi], 1D non-circular bins",
            "representation": "raw unnormalized physical u/xy; first c within-region windows; composite 2c columns",
            "terminal_equality": True, "affine_constraint": False, "svd_reduction": False,
            "fallback": "zero physical input on failed/rejected solve, as upstream; no retries",
            "weights_status": "paper's disc weights, NOT tuned for Reacher",
            "timing": "selection + parameter assignment + solve call (includes canonicalization/setup) + diagnostics; excludes simulation/logging/offline; first solve compilation INCLUDED",
            "adaptations": ["Reacher plant/offline episodes/XY output/q2 scheduler/8 bins/physical bounds/warmup/seeds",
                            "Default T_ini=2, T_f=10 to match contextual DPC; paper uses T_f=5",
                            "Python CVXPY Clarabel rather than MATLAB YALMIP MOSEK",
                            "safe multi-boundary switching and inclusive final endpoint"]}
        path = output/"manifest.json"
        if path.exists():
            if json.loads(path.read_text()) != manifest:
                raise ValueError("Source/data/settings changed. Use a fresh output directory; saved results are preserved.")
        else:
            if list(output.glob("seed*_control_results.npz")):
                raise ValueError("Unversioned results already exist")
            utils.write_json(path, manifest)
        try:
            start = time.perf_counter()
            library = Library.from_path(data, config)
            library.save(output/"offline_library.npz")
            controller = Controller(library)
            utils.write_json(output/"data_audit.json", {**library.audit,
                "offline_setup_s": time.perf_counter()-start,
                "problem_construction_s": controller.construction_s})
            if prepare_only:
                print(json.dumps(library.audit, indent=2))
                return library.audit
            utils.write_json(output/"session.json", {"pid": os.getpid(), "started_unix": time.time()})
            for seed in seeds:
                if not (output/f"seed{seed}_control_results.npz").exists():
                    run_seed(controller, seed, steps, output)
                panel = summarize(output, seeds)
                print(f"Saved {panel['completed']}/{len(seeds)} rollouts; success rate={panel['success_rate']}", flush=True)
            utils.write_json(output/"progress.json", {"state": "complete", "updated_unix": time.time()})
            return panel
        except BaseException as exc:
            utils.write_json(output/"progress.json", {"state": "interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                "message": str(exc), "updated_unix": time.time()})
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-start", type=int, default=211)
    p.add_argument("--seed-stop", type=int, default=310)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--output-dir", type=Path, default=OUTPUT)
    p.add_argument("--prepare-only", action="store_true")
    for field in ("tini", "horizon", "regions", "columns_per_region", "warmup"):
        p.add_argument("--"+field.replace("_", "-"), type=int, default=getattr(Config(), field))
    for field in ("q_xy", "r", "lambda_ini", "lambda_g", "xy_bound"):
        p.add_argument("--"+field.replace("_", "-"), type=float, default=getattr(Config(), field))
    args = vars(p.parse_args())
    config = Config(**{key: args.pop(key) for key in asdict(Config()) if key in args})
    args["output"] = args.pop("output_dir")
    run(config=config, **args)


if __name__ == "__main__":
    main()
