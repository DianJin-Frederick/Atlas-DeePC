"""Derive a shorter saved Select-DPC evaluation without running a controller.

Original traces remain unchanged. A derived manifest deliberately cannot be
resumed by the simulation launcher. Per-step measurements are retained verbatim;
all horizon-dependent summaries are recomputed on the retained prefix.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from atlas_deepc.benchmarks import select_dpc_upstream as original
from atlas_deepc.common.metrics import strict_success, true_error_metrics
from atlas_deepc.control.expert_pipeline_search import closed_loop_control_cost


STATIC_KEYS = {"initial_observation", "target_xy", "final_goal"}


def prefix_arrays(arrays, source_steps, steps):
    if not 100 <= steps < source_steps:
        raise ValueError("Require at least 100 retained steps and a shorter prefix")
    result = {}
    for key, value in arrays.items():
        if key in STATIC_KEYS:
            result[key] = value.copy()
        elif value.ndim and value.shape[0] == source_steps:
            result[key] = value[:steps].copy()
        else:
            raise ValueError(f"Unrecognized trace layout: {key}: {value.shape}")
    return result


def rescore(arrays, source_row, manifest, destination, steps):
    row = copy.deepcopy(source_row)
    q_native = np.asarray(manifest["Q_native"])
    q = q_native[np.ix_(original.TO_BASELINE, original.TO_BASELINE)]
    r = np.asarray(manifest["R"])
    cost = closed_loop_control_cost(arrays["y"], arrays["u"], arrays["ref"], q, r,
                                   "seed-dependent final Cartesian target (original Select-DPC weights)")
    # Independently check the original stored stage costs before summing them.
    delta = arrays["y"] - arrays["ref"]
    tracking = np.einsum("ti,ij,tj->t", delta, q, delta)
    effort = np.einsum("ti,ij,tj->t", arrays["u"], r, arrays["u"])
    np.testing.assert_allclose(tracking, arrays["stage_tracking_cost"], rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(effort, arrays["stage_input_cost"], rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(tracking + effort, arrays["stage_control_cost"], rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(arrays["distance"],
                               np.linalg.norm(arrays["y"][:, 6:8] - arrays["target_xy"], axis=1))
    valid = arrays["prediction_error_xy"][np.isfinite(arrays["prediction_error_xy"])]
    row.update(steps=steps, result_path=str(destination.resolve()),
               source_steps=source_row["steps"], source_result_path=source_row["result_path"],
               derived_from_saved_prefix=True, rerun=False,
               original_wall_seconds=source_row["wall_seconds"], wall_seconds=None,
               wall_seconds_note="300-step wall-clock runtime was not measured; original full-run runtime retained separately.",
               retained_control_compute_seconds=float(arrays["control_compute_time_s"].sum()),
               unsolved_steps=int(np.count_nonzero(~np.isin(arrays["status"], ["optimal", "optimal_inaccurate"]))),
               unsolved_refinements=int(np.count_nonzero(~np.isin(arrays["inner_status"], ["optimal", "optimal_inaccurate"]))),
               statuses={str(s): int(np.sum(arrays["status"] == s)) for s in np.unique(arrays["status"])},
               mean_prediction_error_xy=float(valid.mean()) if valid.size else None,
               **true_error_metrics(arrays["distance"]), **cost)
    for kind in ("reference", "final_goal"):
        row[f"closed_loop_{kind}_cost"] = cost["closed_loop_control_cost"]
        row[f"closed_loop_{kind}_tracking_cost"] = cost["closed_loop_tracking_cost"]
        row[f"mean_closed_loop_{kind}_stage_cost"] = cost["mean_closed_loop_stage_cost"]
    before = arrays["y_before"] - arrays["ref"]
    row.update(xy_squared_error_sum_m2=float(np.sum(arrays["distance"]**2)),
               input_squared_sum=float(np.sum(arrays["u"]**2)),
               upstream_scaled_pre_action_cost=float(1e-6 * np.sum(
                   np.einsum("ti,ij,tj->t", before, q, before) + arrays["stage_input_cost"])),
               post_action_height_violations=int(np.count_nonzero(arrays["y"][:, 7] > .1)),
               max_affine_sum_residual=float(np.nanmax(np.abs(arrays["inner_g"].sum(axis=-1)-1)))
               if np.isfinite(arrays["inner_g"]).any() else None)
    row["strict_success"] = strict_success(row)
    for name in ("localization", "routing", "matrix_preparation", "solve", "control_compute"):
        stats = original.timing_stats(arrays[f"{name}_time_s"])
        for statistic in ("mean", "median", "p95", "p99", "max"):
            row[f"{statistic}_{name}_time_ms"] = stats[f"{statistic}_ms"]
    return row


def derive(source, output, steps=300):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.exists():
        raise ValueError("Use a new output directory; never overwrite source or derived results")
    manifest = json.loads((source / "manifest.json").read_text())
    old_summary = json.loads((source / "summary.json").read_text())
    seeds = list(range(manifest["seed_start"], manifest["seed_stop"] + 1))
    if not old_summary["complete"] or old_summary["total"] != len(seeds):
        raise ValueError("Source panel must be complete")
    paths = [source / f"seed{s}_control_results.npz" for s in seeds]
    if not all(p.is_file() for p in paths):
        raise ValueError("Source traces missing")
    source_hashes = {p.name: original.digest(p) for p in paths}
    signature = copy.deepcopy(manifest)
    signature.update(steps=steps, derived_from_saved_prefix=True, rerun=False,
                     derivation={"source_directory": str(source), "source_steps": manifest["steps"],
                                 "source_manifest_sha256": original.digest(source / "manifest.json"),
                                 "trace_sha256": source_hashes, "retained_slice": [0, steps],
                                 "script_sha256": original.digest(__file__),
                                 "note": "Postprocessed prefix; not a new simulation. Original offline training data unchanged."})
    output.mkdir()
    original.write_json(output / "manifest.json", signature)
    rows = []
    for seed, path in zip(seeds, paths):
        with np.load(path, allow_pickle=False) as trace:
            old_row = json.loads(trace["summary"].item())
            if old_row["seed"] != seed or old_row["steps"] != manifest["steps"]:
                raise ValueError(f"Source metadata mismatch: {path}")
            arrays = prefix_arrays({k: trace[k] for k in trace.files if k != "summary"},
                                   old_row["steps"], steps)
        dest = output / path.name
        row = rescore(arrays, old_row, manifest, dest, steps)
        np.savez_compressed(dest, **arrays, summary=np.array(json.dumps(row)))
        with np.load(dest, allow_pickle=False) as saved:
            for key, value in arrays.items():
                np.testing.assert_array_equal(saved[key], value)
        if original.digest(path) != source_hashes[path.name]:
            raise RuntimeError(f"Source unexpectedly changed: {path}")
        rows.append(row)
        if len(rows) % 20 == 0:
            print(f"Converted and verified {len(rows)}/{len(seeds)} saved traces", flush=True)
    report = original.summarize(rows, seeds, output, signature)
    report.update(derived_from_saved_prefix=True, rerun=False, steps=steps,
                  note=f"First {steps} steps of original {manifest['steps']}-step saved runs; all window metrics recomputed. No rerun.")
    original.write_json(output / "summary.json", report)
    original.write_json(output / "progress.json", dict(state="complete", completed=len(rows),
                        steps=steps, successes=report["successes"], derived_from_saved_prefix=True,
                        rerun=False, updated_unix=time.time()))
    print(json.dumps({k: report[k] for k in ("complete", "total", "successes", "steps", "median_final100_error_mm", "timing_discard_first5")}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    derive(args.source, args.output, args.steps)
