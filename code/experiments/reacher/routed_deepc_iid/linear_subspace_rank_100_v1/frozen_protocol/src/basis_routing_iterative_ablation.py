"""Current N=10 baseline: paired chart-geometry and waypoint ablations."""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

import numpy as np

import baseline_atlas_deepc as baseline
import routed_deepc_experiment as routed
from baseline_1000_benchmark import summarize_timing
from baseline_success_rate import validate_panel
from candidate_goal_routing_experiment import CandidateGoalPolicy, file_digest, strict_success


VERSION = "iterative_n10_geometry_waypoints_v1"
OUTPUT_DIR = baseline.ROOT / "experiments/routed_deepc_iid/basis_routing_iterative_n10_100"
METHODS = {
    "Linear charts + graph waypoints (baseline)": ("linear", "graph"),
    "Affine charts + graph waypoints": ("affine", "graph"),
    "Linear charts + terminal reference only": ("linear", "terminal"),
    "Affine charts + terminal reference only": ("affine", "terminal"),
}
_controller = None
_caches = None


def initialize(output_dir):
    global _controller, _caches
    start = time.perf_counter()
    H = np.load(baseline.DATA_PATH, mmap_mode="r")
    atlas = routed.load_atlas(baseline.ATLAS_PATH)
    _controller = baseline.BaselineAtlasDeePC(H, atlas)
    _caches = {"linear": _controller.matrix_cache,
               "affine": routed.precompute_chart_matrix_cache(H, atlas, _controller.config, geometry="affine")}
    for geometry, cache in _caches.items():
        assert len(cache) == 48
        assert all(matrix["basis"].shape[1] == 60 for matrix, _ in cache.values())
        if geometry == "linear":
            assert all(not np.any(matrix["offset"]) for matrix, _ in cache.values())
    # Warm each representation before collecting timings; keep warmup out of results.
    setup_seconds = time.perf_counter()-start
    for geometry in _caches:
        cfg = replace(_controller.config, seed=210, steps=10)
        policy = CandidateGoalPolicy(H, atlas, cfg, 10, _controller.route_tables)
        routed.run_routed_deepc(H, atlas, cfg, matrix_cache=_caches[geometry], goal_policy=policy)
    (Path(output_dir) / f"worker_setup_{os.getpid()}.json").write_text(json.dumps({
        "pid": os.getpid(), "offline_setup_seconds": setup_seconds,
        "warmup_seed": 210, "warmup_steps_per_geometry": 10,
    }, indent=2))


def run_seed(method, seed, output_dir):
    geometry, mode = METHODS[method]
    controller = _controller
    config = replace(controller.config, seed=int(seed))
    policy = CandidateGoalPolicy(controller.H, controller.atlas, config, 10, controller.route_tables)
    folder = Path(output_dir) / f"{geometry}_{mode}"
    path = folder / f"seed{seed}_control_results.npz"
    row = routed.run_routed_deepc(
        controller.H, controller.atlas, config, matrix_cache=_caches[geometry],
        goal_policy=policy, reference_mode=mode, save_path=path,
        summary_metadata={"ablation_version": VERSION, "method": method, "matrix_geometry": geometry,
                          "candidate_count": 10, "terminal_reference_policy": "iterative_minimum_route_cost"},
    )
    with np.load(path, allow_pickle=True) as trace:
        indices = trace["goal_index"].astype(int)
        sources = policy.source_targets[indices // routed.WINDOWS_PER_EPISODE]
        row["minimum_selected_goal_source_separation"] = float(np.min(np.linalg.norm(sources-policy.target, axis=1)))
        row["unsolved_steps"] = sum(str(s) not in ("optimal", "optimal_inaccurate") for s in trace["status"])
        row["reference_change_steps"] = int(np.any(np.diff(trace["ref"], axis=0) != 0, axis=1).sum())
        if mode == "terminal":
            assert np.all(trace["path_length"] == 1)
            assert np.array_equal(trace["next_chart"], trace["chart"])
            for reference, index in zip(trace["ref"], indices):
                expected = policy.references[np.flatnonzero(policy.indices == index)[0]]
                np.testing.assert_array_equal(reference, expected)
        row["initial_goal_index"] = int(indices[0])
        arrays = {k: trace[k] for k in trace.files if k != "summary"}
    assert row["minimum_selected_goal_source_separation"] >= config.goal_source_target_min_distance
    assert row["dynamic_terminal_reference"] and not row["fitted_dynamics_model"]
    row["result_path"] = str(path.resolve())
    np.savez_compressed(path, **arrays, summary=np.array(json.dumps(row)))
    return row


def run_panel(seeds, output_dir=OUTPUT_DIR, workers=4):
    seeds = list(map(int, seeds))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = baseline.default_config()
    signature = {
        "version": VERSION, "candidate_count": 10,
        "config": {k: v for k, v in asdict(config).items() if k != "seed"},
        "data_sha256": file_digest(baseline.DATA_PATH), "atlas_sha256": file_digest(baseline.ATLAS_PATH),
        "source_sha256": {name: file_digest(baseline.ROOT / "src" / name) for name in (
            "routed_deepc_experiment.py", "candidate_goal_routing_experiment.py", "expert_pipeline_search.py",
            "baseline_atlas_deepc.py", "basis_routing_iterative_ablation.py")},
        "workers": workers, "numerical_threads": 1, "timing_discard": 5,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == signature, "Incompatible ablation cache"
    elif any(output_dir.glob("*/seed*_control_results.npz")):
        raise ValueError("Unversioned ablation cache")
    manifest_path.write_text(json.dumps(signature, indent=2))
    for geometry, mode in METHODS.values():
        (output_dir / f"{geometry}_{mode}").mkdir(exist_ok=True)
    rows, jobs = [], []
    methods = list(METHODS)
    for i, seed in enumerate(seeds):
        offset = i % len(methods)
        for method in methods[offset:]+methods[:offset]:
            geometry, mode = METHODS[method]
            path = output_dir / f"{geometry}_{mode}" / f"seed{seed}_control_results.npz"
            if path.exists():
                with np.load(path, allow_pickle=True) as trace:
                    row = json.loads(trace["summary"].item())
                assert row["seed"] == seed and row["method"] == method and row["ablation_version"] == VERSION
                rows.append(row)
            else:
                jobs.append((method, seed))
    print(f"{len(rows)} cached; {len(jobs)} new rollouts; {len(seeds)} paired seeds, {len(METHODS)} methods.", flush=True)
    if jobs:
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(output_dir,)) as executor:
            futures = [executor.submit(run_seed, method, seed, output_dir) for method, seed in jobs]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                if len(rows) % 10 == 0 or len(rows) == len(seeds)*len(METHODS):
                    counts = {m: [sum(strict_success(r) for r in rows if r["method"] == m),
                                   sum(r["method"] == m for r in rows)] for m in METHODS}
                    print(f"{len(rows)}/{len(seeds)*len(METHODS)} {counts}", flush=True)
    rows.sort(key=lambda r: (list(METHODS).index(r["method"]), r["seed"]))
    assert len(rows) == len(seeds)*len(METHODS)
    timings = {}
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        assert {r["seed"] for r in selected} == set(seeds)
        timings[method] = summarize_timing([r["result_path"] for r in selected])
    for seed in seeds:
        assert len({r["initial_goal_index"] for r in rows if r["seed"] == seed}) == 1
    baseline_dir = baseline.RESULT_ROOT / "evaluation_1000"
    original = json.loads((baseline_dir / "summary.json").read_text())
    old_rows, missing = validate_panel(original, seeds, config)
    assert not missing
    old_by_seed = {r["seed"]: r for r in old_rows}
    checks = []
    for row in rows[:len(seeds)]:
        seed = row["seed"]
        with np.load(row["result_path"], allow_pickle=True) as new, np.load(
                baseline_dir / f"seed{seed}_control_results.npz", allow_pickle=True) as old:
            checks.append({"seed": seed, "success_matches": strict_success(row) == strict_success(old_by_seed[seed]),
                           "max_output_difference": float(np.max(np.abs(new["y"]-old["y"]))),
                           "goal_indices_match": bool(np.array_equal(new["goal_index"], old["goal_index"]))})
    payload = {"version": VERSION, "signature": signature, "seeds": seeds, "results": rows,
               "timing": timings, "baseline_reproduction": checks,
               "worker_setup": [json.loads(p.read_text()) for p in output_dir.glob("worker_setup_*.json")],
               "timing_caveat": "Four concurrent worker processes on a shared computer; not isolated serial latency."}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({m: sum(strict_success(r) for r in rows if r["method"] == m) for m in METHODS}, indent=2), flush=True)
    print("Baseline reproduction max difference:", max(c["max_output_difference"] for c in checks), flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=211)
    parser.add_argument("--seed-stop", type=int, default=310)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run_panel(range(args.seed_start, args.seed_stop+1), args.output_dir, args.workers)
