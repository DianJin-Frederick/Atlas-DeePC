"""Matched lambda_g=0 baseline rollouts with and without graph waypoints."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from atlas_deepc.common.paths import source_hashes, source_path

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.baselines import baseline_atlas_deepc as baseline
from atlas_deepc.control import expert_pipeline_search as pipe
from atlas_deepc.atlas import routed_deepc_experiment as routed
from atlas_deepc.benchmarks.baseline_1000_benchmark import summarize_timing
from atlas_deepc.atlas.candidate_goal_routing_experiment import file_digest, strict_success

VERSION = "lamg0_graph_waypoint_comparison_v1"
RESULT_ROOT = baseline.RESULT_ROOT / "waypoint_comparison"
METHODS = {"terminal": "Atlas-DeePC without graph waypoints", "graph": "Atlas-DeePC with graph waypoints"}
_controller = _folder = _thread_limit = None


def prepare(output_dir=RESULT_ROOT):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = {
        "version": VERSION, "baseline_version": baseline.BASELINE_VERSION,
        "config": asdict(baseline.default_config()), "reference_modes": list(METHODS),
        "candidate_count": 10, "terminal_selection_uses_graph_in_both_methods": True,
        "data_sha256": file_digest(baseline.DATA_PATH), "atlas_sha256": file_digest(baseline.ATLAS_PATH),
        "source_sha256": source_hashes((
            "baseline_waypoint_comparison.py", "baseline_atlas_deepc.py", "routed_deepc_experiment.py",
            "expert_pipeline_search.py", "candidate_goal_routing_experiment.py", "goal_reference.py")),
    }
    assert signature["config"]["lam_g"] == 0
    path = output_dir / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != signature:
            raise ValueError("Incompatible comparison cache; use another output directory")
    elif any(output_dir.glob("*/seed*_control_results.npz")):
        raise ValueError("Refusing unversioned comparison traces")
    else:
        path.write_text(json.dumps(signature, indent=2))
    return signature


def initialize_worker(output_dir):
    global _controller, _folder, _thread_limit
    _thread_limit = threadpool_limits(limits=1)
    pipe.P = 10
    H = np.load(baseline.DATA_PATH, mmap_mode="r")
    atlas = routed.load_atlas(baseline.ATLAS_PATH)
    _controller = baseline.BaselineAtlasDeePC(H, atlas)
    _folder = Path(output_dir)


def validate_trace(path, mode, seed, controller):
    with np.load(path, allow_pickle=True) as trace:
        row = json.loads(trace["summary"].item())
        expected_config = {**asdict(controller.config), "seed": seed}
        assert all(row[key] == value for key, value in expected_config.items())
        assert row["baseline_version"] == baseline.BASELINE_VERSION
        assert row["reference_mode"] == mode and row["graph_waypoints_enabled"] == (mode == "graph")
        assert row["matrix_geometry"] == "linear" and row["candidate_count"] == 10
        assert row["minimum_selected_goal_source_separation"] >= .06
        assert row["dynamic_terminal_reference"] and not row["fitted_dynamics_model"]
        assert trace["y"].shape == (controller.config.steps, 10)
        target = trace["final_goal"].reshape(-1)[6:8]
        np.testing.assert_allclose(trace["distance"], np.linalg.norm(trace["y"][:, 6:8]-target, axis=1))
        if mode == "terminal":
            states = controller.H[pipe.M*pipe.L:].reshape(pipe.L, pipe.P, controller.H.shape[1])[pipe.T_INI-1].T
            expected_ref = states[trace["goal_index"].astype(int)].copy()
            expected_ref[:, 4:6] = 0
            expected_ref[:, 6:8] = target
            np.testing.assert_allclose(trace["ref"], expected_ref, rtol=0, atol=1e-12)
            assert np.all(trace["path_length"] == 1)
        row["result_path"] = str(path)
        row["exact_target_xy"] = target.tolist()
    return row


def run_pair(seed):
    rows = []
    for mode in (("terminal", "graph") if seed % 2 else ("graph", "terminal")):
        folder = _folder / mode
        folder.mkdir(exist_ok=True)
        path = folder / f"seed{seed}_control_results.npz"
        if not path.exists():
            _controller.run(seed, path, reference_mode=mode)
        rows.append(validate_trace(path, mode, seed, _controller))
    np.testing.assert_array_equal(rows[0]["exact_target_xy"], rows[1]["exact_target_xy"])
    return rows


def run_panel(seeds=range(211, 1211), output_dir=RESULT_ROOT, workers=4):
    seeds = list(map(int, seeds))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique")
    output_dir = Path(output_dir)
    signature = prepare(output_dir)
    rows = []
    print(f"Running {len(seeds)} paired seeds with lambda_g=0; {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker, initargs=(output_dir,)) as pool:
        futures = [pool.submit(run_pair, seed) for seed in seeds]
        try:
            for index, future in enumerate(as_completed(futures), 1):
                rows.extend(future.result())
                if index % 25 == 0 or index == len(seeds):
                    counts = {mode: sum(strict_success(r) for r in rows if r["reference_mode"] == mode) for mode in METHODS}
                    print(f"{index}/{len(seeds)} pairs: {counts}", flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    payload = {"version": VERSION, "signature": signature, "seeds": seeds,
               "workers": workers, "numerical_threads": 1, "panels": {},
               "fairness_audit": routed.audit_iid_library(np.load(baseline.DATA_PATH), eval_seeds=seeds)}
    for mode in METHODS:
        selected = sorted([row for row in rows if row["reference_mode"] == mode], key=lambda row: row["seed"])
        assert {row["seed"] for row in selected} == set(seeds)
        panel = {"method": METHODS[mode], "results": selected,
                 "successes": sum(strict_success(row) for row in selected),
                 "timing": summarize_timing([row["result_path"] for row in selected])}
        payload["panels"][mode] = panel
        (output_dir / mode / "summary.json").write_text(json.dumps(panel, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({mode: panel["successes"] for mode, panel in payload["panels"].items()}), flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=211)
    parser.add_argument("--seed-stop", type=int, default=1210)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()
    run_panel(range(args.seed_start, args.seed_stop+1), args.output_dir, args.workers)
