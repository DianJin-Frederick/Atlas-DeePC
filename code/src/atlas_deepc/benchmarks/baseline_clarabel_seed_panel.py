"""Preserve old reports while evaluating all baseline-notebook methods on a new panel."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from atlas_deepc.common.paths import source_hashes, source_path

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from unittest.mock import patch

import cvxpy as cp
import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.baselines import baseline_atlas_deepc as baseline
from atlas_deepc.benchmarks import baseline_clarabel_waypoint_comparison as comparison
from atlas_deepc.atlas import routed_deepc_experiment as routed
from atlas_deepc.benchmarks.baseline_1000_benchmark import summarize_timing
from atlas_deepc.atlas.candidate_goal_routing_experiment import file_digest

VERSION = "clarabel_lamg005_panel_311_1310_v1"
RESULT_ROOT = baseline.ROOT / "experiments/routed_deepc_iid" / VERSION
OLD_GRAPH = baseline.ROOT / "experiments/routed_deepc_iid/baseline_linear_iterative_n10_v1/evaluation_1000"
OLD_TERMINAL = comparison.RESULT_ROOT
METHODS = {"graph": "Atlas-DeePC with graph waypoints",
           "terminal": "Atlas-DeePC without graph waypoints"}


def validate_panel(payload, seeds, config):
    expected = {k: v for k, v in asdict(config).items() if k != "seed"}
    assert payload["method"] == VERSION
    assert payload["signature"]["config"] == expected
    assert payload["signature"]["solver"] == "CLARABEL"
    mode = payload["variant"]
    rows = {int(r["seed"]): r for r in payload["results"]}
    assert len(rows) == len(payload["results"])
    seeds = list(seeds)
    assert len(seeds) == len(set(seeds))
    selected = [rows[s] for s in seeds if s in rows]
    for row in selected:
        assert all(row[k] == v for k, v in expected.items())
        assert row["matrix_geometry"] == "linear"
        assert row["candidate_count"] == 10
        assert row["reference_mode"] == ("terminal" if mode == "terminal" else "graph")
        assert row["minimum_selected_goal_source_separation"] >= .06
        assert not row["fitted_dynamics_model"] and not row["feedback_fallback"]
        for w in (30, 100):
            mean, maximum = row[f"mean_last_{w}"], row[f"max_last_{w}"]
            assert math.isfinite(mean) and math.isfinite(maximum) and min(mean, maximum) >= 0
            assert bool(row[f"success_{w}"]) == bool(mean < .01 and maximum < .02)
    return selected, [s for s in seeds if s not in rows]


def read_trace(path, mode, seed, config):
    with np.load(path, allow_pickle=True) as trace:
        row = json.loads(trace["summary"].item())
        assert row["seed"] == seed
        assert all(row[k] == v for k, v in asdict(config).items() if k != "seed")
        assert trace["y"].shape == (config.steps, 10)
        target = trace["final_goal"].reshape(-1)[6:8]
        distance = np.linalg.norm(trace["y"][:, 6:8] - target, axis=1)
        np.testing.assert_allclose(trace["distance"], distance)
        for w in (30, 100):
            np.testing.assert_allclose(row[f"mean_last_{w}"], distance[-w:].mean())
            np.testing.assert_allclose(row[f"max_last_{w}"], distance[-w:].max())
        assert row["baseline_version"] in ("linear_iterative_n10_v1", baseline.BASELINE_VERSION)
        if mode == "terminal":
            assert np.all(trace["path_length"] == 1)
        row["result_path"] = str(Path(path).resolve())
        row["exact_target_xy"] = target.tolist()
        row["trace_sha256"] = file_digest(path)
        return row


def run_panel(seeds=range(311, 1311), output_dir=RESULT_ROOT):
    seeds = list(map(int, seeds))
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and unique")
    config = comparison.default_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = {k: v for k, v in asdict(config).items() if k != "seed"}
    signature = {"config": cfg, "solver": "CLARABEL", "matrix_geometry": "linear",
                 "data_sha256": file_digest(baseline.DATA_PATH), "atlas_sha256": file_digest(baseline.ATLAS_PATH),
                 "source_sha256": source_hashes((
                     "baseline_clarabel_seed_panel.py", "baseline_atlas_deepc.py",
                     "baseline_clarabel_waypoint_comparison.py",
                     "baseline_1000_benchmark.py", "routed_deepc_experiment.py",
                     "candidate_goal_routing_experiment.py", "expert_pipeline_search.py",
                     "goal_reference.py"))}
    source_payloads = {"graph": json.loads((OLD_GRAPH / "summary.json").read_text()),
                       "terminal": json.loads((OLD_TERMINAL / "summary.json").read_text())}
    for mode, source in source_payloads.items():
        sig = source["signature"]
        assert {k: v for k, v in sig["config"].items() if k != "seed"} == cfg, mode
        assert all(sig[k] == signature[k] for k in ("data_sha256", "atlas_sha256")), mode
    assert source_payloads["graph"]["method"] == "linear_iterative_n10_v1"
    assert source_payloads["terminal"]["version"] == comparison.VERSION
    assert source_payloads["terminal"]["signature"]["solver"] == "CLARABEL"
    manifest_data = {"version": VERSION, "seeds": seeds, **signature,
                     "reused_source_summaries": {mode: file_digest(path) for mode, path in {
                         "graph": OLD_GRAPH / "summary.json", "terminal": OLD_TERMINAL / "summary.json"}.items()}}
    manifest = output_dir / "manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == manifest_data, "Incompatible shifted-panel cache"
    else:
        assert not any(output_dir.glob("*/seed*.npz")), "Unversioned traces"
        manifest.write_text(json.dumps(manifest_data, indent=2))
    old_rows = {
        "graph": {r["seed"]: OLD_GRAPH / f"seed{r['seed']}_control_results.npz" for r in source_payloads["graph"]["results"]},
        "terminal": {r["seed"]: OLD_TERMINAL / "terminal" / f"seed{r['seed']}_control_results.npz"
                     for r in source_payloads["terminal"]["panels"]["terminal"]["results"]}}
    protocol = {"started_utc": datetime.now(timezone.utc).isoformat(), "workers": 1, "numerical_threads": 1,
                "warmup_seed": 210, "warmup_steps": 10, "discard_initial_steps_per_rollout": 5,
                "new": {m: 0 for m in METHODS}, "reused": {m: 0 for m in METHODS},
                "timing_caveat": "Overlapping seeds reuse historical measurements; timings span different sessions."}
    session_file = output_dir / ("session_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    session_file.write_text(json.dumps(protocol, indent=2))
    rows = {m: [] for m in METHODS}
    solve = cp.Problem.solve

    def checked_solve(problem, *args, **kwargs):
        assert kwargs.get("solver") == cp.CLARABEL
        return solve(problem, *args, **kwargs)

    with threadpool_limits(limits=1), patch.object(cp.Problem, "solve", checked_solve):
        H = np.load(baseline.DATA_PATH, mmap_mode="r")
        atlas = routed.load_atlas(baseline.ATLAS_PATH)
        controller = baseline.BaselineAtlasDeePC(H, atlas, config)
        baseline.BaselineAtlasDeePC(H, atlas, replace(config, steps=10)).run(210)
        for index, seed in enumerate(seeds, 1):
            order = tuple(METHODS) if seed % 2 else tuple(reversed(METHODS))
            for mode in order:
                folder = output_dir / mode
                folder.mkdir(exist_ok=True)
                source = old_rows[mode].get(seed)
                if source is not None and not source.exists():
                    raise FileNotFoundError(f"Historical trace is missing; refusing to overwrite its source: {source}")
                path = source if source is not None else folder / f"seed{seed}_control_results.npz"
                reused = path.exists()
                if not reused:
                    pending = folder / f"pending_seed{seed}.npz"
                    controller.run(seed, pending, reference_mode=mode)
                    pending.replace(path)
                row = read_trace(path, mode, seed, config)
                row["provenance"] = "historical_reuse" if source is not None else "new_panel_rollout"
                rows[mode].append(row)
                protocol["reused" if reused else "new"][mode] += 1
            targets = [rows[m][-1]["exact_target_xy"] for m in METHODS]
            np.testing.assert_array_equal(targets[0], targets[1])
            session_file.write_text(json.dumps(protocol, indent=2))
            if index % 25 == 0 or index == len(seeds):
                print(f"{index}/{len(seeds)} seed pairs: " + str({m: sum(r["success_30"] and r["success_100"] for r in rr) for m, rr in rows.items()}), flush=True)
        audit = routed.audit_iid_library(H, eval_seeds=seeds)
    panels = {}
    for mode, items in rows.items():
        payload = {"method": VERSION, "variant": mode, "signature": signature,
                   "fairness_audit": audit, "results": sorted(items, key=lambda r: r["seed"]),
                   "timing_caveat": protocol["timing_caveat"]}
        validate_panel(payload, seeds, config)
        timing = summarize_timing([r["result_path"] for r in items])
        timing["sessions"] = json.loads((OLD_GRAPH / "timing_summary.json").read_text())["sessions"] if mode == "graph" else []
        timing["timing_caveat"] = protocol["timing_caveat"]
        folder = output_dir / mode
        (folder / "summary.json").write_text(json.dumps(payload, indent=2))
        (folder / "timing_summary.json").write_text(json.dumps(timing, indent=2))
        panels[mode] = {"successes": sum(r["success_30"] and r["success_100"] for r in items),
                        "timing": timing["modules"], "summary": str(folder / "summary.json")}
    protocol["completed_utc"] = datetime.now(timezone.utc).isoformat()
    session_file.write_text(json.dumps(protocol, indent=2))
    report = {"version": VERSION, "seeds": seeds, "panels": panels, "session": protocol}
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({m: p["successes"] for m, p in panels.items()}), flush=True)
    return report


if __name__ == "__main__":
    run_panel()
