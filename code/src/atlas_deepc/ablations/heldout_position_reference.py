"""Paired fingertip-only reference ablation of the frozen four-method baseline."""

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np

from atlas_deepc.benchmarks import reacher_heldout as baseline

RESULT_ROOT = baseline.ROOT / "experiments/reacher/fingertip_reference_100_k48_v1"
SEEDS = tuple(range(10001, 10101))
VERSION = "fingertip_reference_100_k48_v1"
CASES = baseline.CASES
_config_for = baseline.config_for
_case_signature = baseline.case_signature


def config_for(p, mode):
    return replace(_config_for(p, mode), q_data_goal_joint=0., q_dq=0.)


def signature(root, p, mode):
    result = _case_signature(root, p, mode)
    result.update(ablation_code_sha256=baseline.file_digest(__file__),
                  tracked_output_rows=[6, 7], reference_dimension=2,
                  retained_output_dimension=p, ablation="Only Q non-position weights set to zero",
                  paired_baseline_manifest_sha256=baseline.file_digest(
                      baseline.RESULT_ROOT / f"output{p}/{mode}/manifest.json"),
                  timing_caveat="Serial one-thread execution on a shared computer")
    return result


@contextmanager
def ablation_context():
    # Reuse the current runner without changing its frozen source or baseline caches.
    with patch.object(baseline, "SEEDS", SEEDS), patch.object(baseline, "VERSION", VERSION),\
            patch.object(baseline, "config_for", config_for),\
            patch.object(baseline, "case_signature", signature):
        yield


def run_case(root, p, mode):
    original_controller = baseline.experiment.routed.pipe.DeePCController

    class PositionController(original_controller):
        def __init__(self, Q, *args, **kwargs):
            expected = np.zeros(p)
            expected[6:8] = config_for(p, mode).q_xy
            np.testing.assert_array_equal(Q, np.diag(expected))
            super().__init__(Q, *args, **kwargs)

    with ablation_context(), patch.object(baseline.experiment.routed.pipe, "DeePCController", PositionController):
        return baseline.run_case(Path(root), p, mode)


def prepare(root, cases):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    signatures = {}
    with ablation_context():
        for p, mode, rank in cases:
            source = baseline.atlas_path(baseline.RESULT_ROOT, p)
            old = json.loads((baseline.RESULT_ROOT / f"output{p}/{mode}/manifest.json").read_text())
            assert old["atlas_sha256"] == baseline.file_digest(source)
            assert old["data_sha256"] == baseline.file_digest(baseline.DATA_PATH)
            assert rank == config_for(p, mode).svd_rank
            destination = baseline.atlas_path(root, p)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                temporary = destination.with_suffix(".pending.npz")
                shutil.copy2(source, temporary)
                temporary.replace(destination)
            assert baseline.file_digest(destination) == old["atlas_sha256"]
            signatures[f"{p}_{mode}"] = signature(root, p, mode)
    protocol = {"version": VERSION, "seeds": list(SEEDS), "cases": [list(c) for c in cases],
                "case_signatures": signatures, "workers": 1, "numerical_threads": 1,
                "changes": "Only q_data_goal_joint and q_dq changed to zero",
                "reference_dimension": 2, "retained_prediction_outputs": "Unchanged (p=8 or p=10)",
                "caveat": "Exploratory paired ablation on the first 100 already evaluated test seeds"}
    manifest = root / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != protocol:
            raise ValueError("Incompatible position-reference ablation cache")
    else:
        assert not list(root.glob("output*/*/seed*_control_results.npz"))
        shutil.copytree(baseline.RESULT_ROOT / "frozen_source", root / "frozen_source", dirs_exist_ok=True)
        shutil.copy2(__file__, root / "frozen_source" / Path(__file__).name)
        baseline.save_json(manifest, protocol)
    for key, value in signatures.items():
        p, mode = key.split("_", 1)
        path = root / f"output{p}/{mode}/manifest.json"
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f"Incompatible panel: {path}")
        baseline.save_json(path, value)
    return protocol


def paired_report(root=RESULT_ROOT):
    root = Path(root)
    comparisons = []
    for p, mode, rank in CASES:
        directory = root / f"output{p}/{mode}"
        if not (directory / "metrics.json").exists():
            continue
        position = json.loads((directory / "metrics.json").read_text())
        original_directory = baseline.RESULT_ROOT / f"output{p}/{mode}"
        original = json.loads((original_directory / "summary.json").read_text())
        original_rows = {r["seed"]: r for r in original["results"] if r["seed"] in SEEDS}
        assert set(original_rows) == set(SEEDS)
        original_failed = {s for s, r in original_rows.items() if not baseline.strict_success(r)}
        changed_failed = set(position["failed_seeds"])
        errors, rmses = [], []
        for seed in SEEDS:
            filename = f"seed{seed}_control_results.npz"
            with np.load(original_directory / filename) as old, np.load(directory / filename) as new:
                np.testing.assert_array_equal(old["selection_windows"][0], new["selection_windows"][0])
                np.testing.assert_array_equal(old["final_goal"].reshape(-1)[6:8],
                                              new["final_goal"].reshape(-1)[6:8])
                if mode == "terminal":
                    np.testing.assert_array_equal(old["ref"], new["ref"])
                distance = old["distance"]
                errors.append(1000*np.mean(distance[-100:]))
                rmses.append(1000*np.sqrt(np.mean(distance**2)))
        comparisons.append({"output_dimension": p, "reference_mode": mode, "rank": rank,
            "seeds": list(SEEDS), "baseline_successes": len(SEEDS)-len(original_failed),
            "position_only_successes": position["successes"],
            "baseline_success_rate": 100*(len(SEEDS)-len(original_failed))/len(SEEDS),
            "position_only_success_rate": position["success_rate"],
            "baseline_median_error_mm": float(np.median(errors)),
            "position_only_median_error_mm": position["median_error_mm"],
            "baseline_p95_error_mm": float(np.percentile(errors, 95)),
            "position_only_p95_error_mm": position["p95_error_mm"],
            "baseline_median_rollout_rmse_mm": float(np.median(rmses)),
            "position_only_median_rollout_rmse_mm": position["median_rollout_rmse_mm"],
            "rescued_seeds": sorted(original_failed-changed_failed),
            "regressed_seeds": sorted(changed_failed-original_failed),
            "failed_seeds": sorted(changed_failed), "unsolved_steps": position["unsolved_steps"]})
    result = {"version": VERSION, "comparisons": comparisons,
              "complete_four_cases": len(comparisons) == 4,
              "interpretation": "Exploratory paired ablation on previously evaluated seeds; not a new held-out test",
              "reference": "Only fingertip position is penalized; all 8/10 outputs remain in prediction and history matching",
              "graph": "Graph cases keep Cartesian intermediate waypoints and iterative terminal-candidate selection"}
    baseline.save_json(root / "paired_comparison.json", result)
    return result


def run_all(cases=(CASES[0],), root=RESULT_ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(root, cases)
        status = {"status": "running", "pid": os.getpid(), "workers": 1,
                  "started_utc": datetime.now(timezone.utc).isoformat(),
                  "requested_cases": [list(c) for c in cases], "requested_rollouts": len(SEEDS)*len(cases)}
        baseline.save_json(root / "run_status.json", status)
        try:
            for p, mode, _ in cases:
                run_case(root, p, mode)
                paired_report(root)
            status["status"] = "complete"
        except BaseException as exc:
            status.update(status="stopped", error=str(exc))
            raise
        finally:
            status["updated_utc"] = datetime.now(timezone.utc).isoformat()
            baseline.save_json(root / "run_status.json", status)
    return paired_report(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("8_terminal", "8_graph", "10_terminal", "10_graph"),
                        default="8_terminal")
    args = parser.parse_args()
    run_all(cases=tuple(c for c in CASES if f"{c[0]}_{c[1]}" == args.case))
