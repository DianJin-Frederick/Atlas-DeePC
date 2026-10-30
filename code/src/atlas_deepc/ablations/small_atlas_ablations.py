"""Small 100-seed ablations against the current baseline Atlas-DeePC."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from atlas_deepc.common.paths import source_hashes, source_path

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

import numpy as np

from atlas_deepc.baselines import baseline_atlas_deepc as baseline
from atlas_deepc.atlas import routed_deepc_experiment as routed
from atlas_deepc.benchmarks.baseline_1000_benchmark import summarize_timing
from atlas_deepc.atlas.candidate_goal_routing_experiment import CandidateGoalPolicy, file_digest, strict_success


ROOT = baseline.ROOT
RESULT_ROOT = ROOT / "experiments/routed_deepc_iid/small_current_atlas_100"
SUM_DIR = RESULT_ROOT / "svd_coefficient_sum_one"
BASIS_DIR = ROOT / "experiments/routed_deepc_iid/basis_routing_iterative_n10_100"
NEAREST_DIR = ROOT / "experiments/routed_deepc_iid/ablation_nearest_terminal_linear_v1"
SEEDS = list(range(211, 311))
VERSION = "linear_svd_g_sum_one_iterative_n10_v1"
METHOD = "Baseline + sum of SVD coefficients = 1"
_controller = None
_sum_cache = None


def sum_one_cache(cache):
    return {key: ({**matrix, "g_sum": 1.0}, indices) for key, (matrix, indices) in cache.items()}


def initialize():
    global _controller, _sum_cache
    H = np.load(baseline.DATA_PATH, mmap_mode="r")
    atlas = routed.load_atlas(baseline.ATLAS_PATH)
    _controller = baseline.BaselineAtlasDeePC(H, atlas)
    _sum_cache = sum_one_cache(_controller.matrix_cache)
    for key, (matrix, _) in _sum_cache.items():
        assert not np.any(matrix["offset"])
        assert matrix["basis"].shape[1] == 60 and matrix["g_sum"] == 1
        assert "g_sum" not in _controller.matrix_cache[key][0]


def run_seed(seed, output_dir):
    controller = _controller
    config = replace(controller.config, seed=int(seed))
    policy = CandidateGoalPolicy(controller.H, controller.atlas, config, 10, controller.route_tables)
    coefficient_sums = []
    original_class = routed.pipe.DeePCController

    class AuditedController(original_class):
        def solve(self, A, u_ini, y_ini, y_ref):
            assert A["g_sum"] == 1.0
            result = super().solve(A, u_ini, y_ini, y_ref)
            coefficient_sums.append(float(np.sum(result[2])) if result[2] is not None else np.nan)
            return result

    path = Path(output_dir) / f"seed{seed}_control_results.npz"
    # This worker-local wrapper records returned coefficients without changing the QP.
    with patch.object(routed.pipe, "DeePCController", AuditedController):
        row = routed.run_routed_deepc(
            controller.H, controller.atlas, config, save_path=path,
            matrix_cache=_sum_cache, goal_policy=policy, reference_mode="graph",
            summary_metadata={"ablation_version": VERSION, "method": METHOD, "matrix_geometry": "linear",
                              "candidate_count": 10, "terminal_reference_policy": "iterative_minimum_route_cost",
                              "g_sum_target": 1.0, "g_sum_coordinates": "uncentered_SVD_basis"},
        )
    sums = np.asarray(coefficient_sums)
    assert sums.shape == (config.steps,)
    with np.load(path, allow_pickle=True) as trace:
        solved = np.isin(trace["status"], ["optimal", "optimal_inaccurate"])
        assert np.array_equal(np.isfinite(sums), solved)
        indices = trace["goal_index"].astype(int)
        sources = policy.source_targets[indices // routed.WINDOWS_PER_EPISODE]
        row["minimum_selected_goal_source_separation"] = float(np.min(np.linalg.norm(sources-policy.target, axis=1)))
        row["unsolved_steps"] = int((~solved).sum())
        row["initial_goal_index"] = int(indices[0])
        row["max_g_sum_residual"] = float(np.max(np.abs(sums[solved]-1))) if solved.any() else None
        arrays = {k: trace[k] for k in trace.files if k != "summary"}
    assert row["minimum_selected_goal_source_separation"] >= config.goal_source_target_min_distance
    row["result_path"] = str(path.resolve())
    np.savez_compressed(path, **arrays, coefficient_sum=sums, coefficient_sum_residual=np.abs(sums-1),
                        summary=np.array(json.dumps(row)))
    return row


def run_sum_panel(seeds=SEEDS, output_dir=SUM_DIR, workers=4):
    seeds = list(map(int, seeds))
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("Seeds must be nonempty and unique")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = baseline.default_config()
    signature = {
        "version": VERSION, "g_sum": 1.0, "coordinates": "uncentered_SVD_basis", "candidate_count": 10,
        "config": {k: v for k, v in asdict(config).items() if k != "seed"},
        "data_sha256": file_digest(baseline.DATA_PATH), "atlas_sha256": file_digest(baseline.ATLAS_PATH),
        "source_sha256": source_hashes((
            "routed_deepc_experiment.py", "expert_pipeline_search.py", "baseline_atlas_deepc.py",
            "candidate_goal_routing_experiment.py", "small_atlas_ablations.py")),
        "workers": workers, "numerical_threads": 1,
    }
    original = json.loads((BASIS_DIR / "summary.json").read_text())
    for key in ("config", "data_sha256", "atlas_sha256"):
        assert signature[key] == original["signature"][key], f"Baseline mismatch: {key}"
    for name in ("routed_deepc_experiment.py", "expert_pipeline_search.py", "baseline_atlas_deepc.py", "candidate_goal_routing_experiment.py"):
        assert signature["source_sha256"][name] == original["signature"]["source_sha256"][name]
    baseline_rows = {r["seed"]: r for r in original["results"] if r["method"] == "Linear charts + graph waypoints (baseline)"}
    assert set(seeds) <= set(baseline_rows)
    manifest = output_dir / "manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == signature, "Incompatible sum-constraint cache"
    elif any(output_dir.glob("seed*_control_results.npz")):
        raise ValueError("Unversioned cache")
    manifest.write_text(json.dumps(signature, indent=2))
    rows, pending = [], []
    for seed in seeds:
        path = output_dir / f"seed{seed}_control_results.npz"
        if path.exists():
            with np.load(path, allow_pickle=True) as trace:
                row = json.loads(trace["summary"].item())
            assert row["ablation_version"] == VERSION and row["seed"] == seed
            rows.append(row)
        else:
            pending.append(seed)
    print(f"{len(pending)} new sum-to-one rollouts; {len(rows)} cached; baseline reused.", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize) as executor:
            futures = [executor.submit(run_seed, seed, output_dir) for seed in pending]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                print(f"{len(rows)}/{len(seeds)} seed {row['seed']}: success={strict_success(row)}, tail100={1000*row['mean_last_100']:.3f} mm, unsolved={row['unsolved_steps']}", flush=True)
    rows.sort(key=lambda r: r["seed"])
    assert [r["seed"] for r in rows] == sorted(seeds)
    assert all(row["initial_goal_index"] == baseline_rows[row["seed"]]["initial_goal_index"] for row in rows)
    timing = summarize_timing([row["result_path"] for row in rows])
    comparison = {
        "baseline_successes": sum(strict_success(baseline_rows[s]) for s in seeds),
        "sum_constraint_successes": sum(strict_success(r) for r in rows),
        "repaired_seeds": [r["seed"] for r in rows if strict_success(r) and not strict_success(baseline_rows[r["seed"]])],
        "regressed_seeds": [r["seed"] for r in rows if not strict_success(r) and strict_success(baseline_rows[r["seed"]])],
        "unsolved_steps": sum(r["unsolved_steps"] for r in rows),
        "max_g_sum_residual": max((r["max_g_sum_residual"] for r in rows if r["max_g_sum_residual"] is not None), default=None),
    }
    payload = {"version": VERSION, "signature": signature, "seeds": seeds, "results": rows,
               "timing": timing, "comparison": comparison,
               "timing_note": "Concurrent shared-computer run; separate batch from baseline; first five steps per rollout excluded, no dedicated warmup."}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(comparison, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run_sum_panel(workers=args.workers)
