"""Single-worker Clarabel, lambda_g=0.05 waypoint ablation."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from atlas_deepc.common.paths import source_hashes, source_path

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import patch

import cvxpy as cp
import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.baselines import baseline_atlas_deepc as baseline
from atlas_deepc.atlas import routed_deepc_experiment as routed
from atlas_deepc.benchmarks.baseline_1000_benchmark import summarize_timing
from atlas_deepc.benchmarks.baseline_waypoint_comparison import METHODS, validate_trace
from atlas_deepc.atlas.candidate_goal_routing_experiment import file_digest


VERSION = "clarabel_lamg005_waypoints_v1"
RESULT_ROOT = baseline.ROOT / "experiments/routed_deepc_iid" / VERSION


def default_config():
    return replace(baseline.default_config(), lam_g=0.05)


def run_panel(seeds, output_dir=RESULT_ROOT, modes=("terminal",), config=None):
    """Resume compatible traces; optionally remeasure graph mode in the same run."""
    seeds, modes = list(map(int, seeds)), tuple(modes)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and unique")
    if not modes or len(modes) != len(set(modes)) or not set(modes) <= set(METHODS):
        raise ValueError("Modes must be terminal and/or graph")
    config = default_config() if config is None else config
    if config.lam_g != 0.05:
        raise ValueError("This experiment requires lambda_g=0.05")
    if "CLARABEL" not in cp.installed_solvers():
        raise RuntimeError("Clarabel is not installed")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = {
        "version": VERSION, "config": asdict(config), "solver": "CLARABEL",
        "workers": 1, "numerical_threads": 1,
        "data_sha256": file_digest(baseline.DATA_PATH),
        "atlas_sha256": file_digest(baseline.ATLAS_PATH),
        "source_sha256": source_hashes((
            "baseline_clarabel_waypoint_comparison.py", "baseline_atlas_deepc.py",
            "baseline_waypoint_comparison.py", "baseline_1000_benchmark.py",
            "routed_deepc_experiment.py", "candidate_goal_routing_experiment.py",
            "expert_pipeline_search.py", "goal_reference.py")),
    }
    manifest = output_dir / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != signature:
            raise ValueError("Incompatible cache; choose a new comparison output directory")
    elif any(output_dir.glob("*/seed*_control_results.npz")):
        raise ValueError("Unversioned traces; choose a new output directory")
    else:
        manifest.write_text(json.dumps(signature, indent=2))
    session = {"started_utc": datetime.now(timezone.utc).isoformat(),
               "requested_seeds": seeds, "modes": modes, "new_rollouts": 0,
               "reused_rollouts": 0, "workers": 1, "numerical_threads": 1,
               "warmup_seed": 210, "warmup_steps": 10,
               "discard_initial_steps_per_rollout": 5}
    session_path = output_dir / ("session_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    session_path.write_text(json.dumps(session, indent=2))
    solve = cp.Problem.solve

    def checked_solve(problem, *args, **kwargs):
        if kwargs.get("solver") != cp.CLARABEL:
            raise AssertionError("This experiment must use Clarabel for every QP")
        return solve(problem, *args, **kwargs)

    rows = {mode: [] for mode in modes}
    with threadpool_limits(limits=1), patch.object(cp.Problem, "solve", checked_solve):
        H = np.load(baseline.DATA_PATH, mmap_mode="r")
        atlas = routed.load_atlas(baseline.ATLAS_PATH)
        controller = baseline.BaselineAtlasDeePC(H, atlas, config)
        baseline.BaselineAtlasDeePC(H, atlas, replace(config, steps=10)).run(210)
        for index, seed in enumerate(seeds, 1):
            order = modes if seed % 2 else tuple(reversed(modes))
            for mode in order:
                path = output_dir / mode / f"seed{seed}_control_results.npz"
                path.parent.mkdir(exist_ok=True)
                if not path.exists():
                    controller.run(seed, path, reference_mode=mode)
                    session["new_rollouts"] += 1
                else:
                    session["reused_rollouts"] += 1
                row = validate_trace(path, mode, seed, controller)
                rows[mode].append(row)
            session_path.write_text(json.dumps(session, indent=2))
            if index % 25 == 0 or index == len(seeds):
                print(f"Clarabel lambda_g=0.05: {index}/{len(seeds)} seeds completed", flush=True)
    panels = {mode: {"method": METHODS[mode], "results": sorted(items, key=lambda r: r["seed"]),
                     "timing": summarize_timing([r["result_path"] for r in items])}
              for mode, items in rows.items()}
    session["completed_utc"] = datetime.now(timezone.utc).isoformat()
    session_path.write_text(json.dumps(session, indent=2))
    payload = {"version": VERSION, "signature": signature, "panels": panels,
               "session": session, "seeds": seeds}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def load_rows(output_dir, mode, seeds):
    """Read completed intermediate files without requiring the final summary."""
    output_dir = Path(output_dir)
    paths = [output_dir / mode / f"seed{s}_control_results.npz" for s in seeds]
    if not any(p.exists() for p in paths):
        return []
    signature = json.loads((output_dir / "manifest.json").read_text())
    if signature["version"] != VERSION or signature["solver"] != "CLARABEL" or signature["config"]["lam_g"] != .05:
        raise ValueError("Not the requested Clarabel lambda_g=0.05 experiment")
    rows = []
    for seed, path in zip(seeds, paths):
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as trace:
            row = json.loads(trace["summary"].item())
            assert row["seed"] == seed and row["reference_mode"] == mode
            assert all(row[k] == v for k, v in signature["config"].items() if k != "seed")
            assert len(trace["distance"]) == signature["config"]["steps"]
            assert row["matrix_geometry"] == "linear" and row["candidate_count"] == 10
            assert row["graph_waypoints_enabled"] == (mode == "graph")
        rows.append({**row, "result_path": str(path)})
    return rows
