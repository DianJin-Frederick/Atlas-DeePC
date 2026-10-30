"""Convert saved p=8 contextual DPC runs to a shorter prefix, without simulation."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import time

import numpy as np

from atlas_deepc.benchmarks import contextual_dpc_atlas_terminal as case
from atlas_deepc.common.metrics import strict_success, true_error_metrics


STATIC_KEYS = {"initial_observation", "target_xy", "final_goal", "warmup_u", "warmup_y"}


def prefix_arrays(arrays, source_steps, steps):
    if not 100 <= steps < source_steps:
        raise ValueError("Require a shorter prefix with at least 100 steps")
    out = {}
    for key, value in arrays.items():
        if key in STATIC_KEYS:
            out[key] = value.copy()
        elif value.ndim and value.shape[0] == source_steps:
            out[key] = value[:steps].copy()
        else:
            raise ValueError(f"Unrecognized trace layout: {key}: {value.shape}")
    return out


def rescore(a, old, manifest, norm, steps):
    config = manifest["config"]
    delta_u = np.diff(np.vstack([a["warmup_u"][-1], a["u"]]), axis=0)
    tracking = np.sum(norm["Q_diagonal"][:8] * ((a["y"] - a["ref"]) / norm["z_scale"][:8])**2, axis=1)
    effort = config["r"] * np.sum(((a["u"] - norm["z_mean"][-2:]) / norm["z_scale"][-2:])**2, axis=1)
    smoothing = config["r"] * np.sum((delta_u / norm["z_scale"][-2:])**2, axis=1)
    for key, value in (("stage_tracking_cost", tracking), ("stage_input_cost", effort),
                       ("stage_smoothing_cost", smoothing), ("stage_control_cost", tracking+effort+smoothing)):
        np.testing.assert_allclose(a[key], value, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(a["distance"], np.linalg.norm(a["y"][:, 6:8] - a["target_xy"], axis=1))
    np.testing.assert_array_equal(a["ref"], np.tile(a["final_goal"], (steps, 1)))
    np.testing.assert_array_equal(a["final_goal"][4:6], np.zeros(2))
    prediction = a["prediction_error_xy"][np.isfinite(a["prediction_error_xy"])]
    row = copy.deepcopy(old)
    row.update(steps=steps, source_steps=old["steps"], derived_from_saved_prefix=True, rerun=False,
               original_wall_seconds=old["wall_seconds"], wall_seconds=None,
               wall_seconds_note="Prefix wall-clock runtime was not measured; original full-run runtime retained separately.",
               retained_control_compute_seconds=float(a["control_compute_time_s"].sum()),
               **true_error_metrics(a["distance"]), success_evaluable=True,
               fallback_steps=int(a["fallback"].sum()),
               closed_loop_control_cost=float(a["stage_control_cost"].sum()),
               closed_loop_reference_cost=float(a["stage_tracking_cost"].sum()),
               closed_loop_input_cost=float(a["stage_input_cost"].sum()),
               closed_loop_smoothing_cost=float(a["stage_smoothing_cost"].sum()),
               xy_squared_error_sum_m2=float(np.sum(a["distance"]**2)),
               input_squared_sum=float(np.sum(a["u"]**2)),
               delta_input_squared_sum=float(np.sum(delta_u**2)),
               mean_prediction_error_xy=float(prediction.mean()) if prediction.size else None,
               timing=case.timing_stats(a["control_compute_time_s"]))
    row["strict_success"] = strict_success(row)
    return row


def derive(source, output, steps=300):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.exists():
        raise ValueError("Use a new output directory; source and existing results must not be overwritten")
    manifest = json.loads((source / "manifest.json").read_text())
    previous = json.loads((source / "summary.json").read_text())
    seeds = manifest["seeds"]
    if seeds != list(range(211, 311)) or not previous["complete"] or previous["completed"] != 100:
        raise ValueError("Expected complete original 100-seed p=8 panel")
    paths = [source / f"seed{s}_control_results.npz" for s in seeds]
    hashes = {p.name: case.base.digest(p) for p in paths}
    with np.load(source / "normalization.npz", allow_pickle=False) as saved:
        norm = {k: saved[k] for k in saved.files}
    reference_root = source.parent
    if case.base.digest(reference_root / "references.npz") != manifest["reference_bundle_sha256"]:
        raise ValueError("Reference bundle changed")
    if case.base.digest(reference_root / "reference_provenance.json") != manifest["reference_provenance_sha256"]:
        raise ValueError("Reference provenance changed")
    with np.load(reference_root / "references.npz", allow_pickle=False) as bundle:
        np.testing.assert_array_equal(bundle["seeds"], seeds)
        references = bundle["references"].copy()
    signature = copy.deepcopy(manifest)
    signature.update(steps=steps, derived_from_saved_prefix=True, rerun=False,
                     derivation=dict(source_directory=str(source), source_steps=manifest["steps"],
                                     source_manifest_sha256=case.base.digest(source / "manifest.json"),
                                     trace_sha256=hashes, retained_slice=[0, steps],
                                     script_sha256=case.base.digest(__file__),
                                     note="Saved-prefix conversion only; no controller, reference or offline-data changes."))
    output.mkdir()
    case.write_json(output / "manifest.json", signature)
    for name in ("normalization.npz", "data_audit.json"):
        shutil.copy2(source / name, output / name)
    for name in ("references.npz", "reference_provenance.json", "tuning_selection.json", "tuning_report.json"):
        shutil.copy2(reference_root / name, output / name)
    for index, (seed, path) in enumerate(zip(seeds, paths)):
        with np.load(path, allow_pickle=False) as saved:
            old = json.loads(saved["summary"].item())
            if old["seed"] != seed or old["steps"] != manifest["steps"] or old["reference_mode"] != "supplied_unguided_atlas_fixed_eight_output":
                raise ValueError(f"Unexpected source metadata: {path}")
            arrays = prefix_arrays({k: saved[k] for k in saved.files if k != "summary"}, old["steps"], steps)
        np.testing.assert_array_equal(arrays["final_goal"], references[index])
        row = rescore(arrays, old, manifest, norm, steps)
        row.update(source_result_path=str(path), result_path=str(output / path.name))
        np.savez_compressed(output / path.name, **arrays, summary=np.array(json.dumps(row)))
        with np.load(output / path.name, allow_pickle=False) as saved:
            for key, value in arrays.items():
                np.testing.assert_array_equal(saved[key], value)
        if case.base.digest(path) != hashes[path.name]:
            raise RuntimeError(f"Source changed: {path}")
        if (index+1) % 20 == 0:
            print(f"Converted and verified {index+1}/100 saved p=8 traces", flush=True)
    report = case.summarize(output, seeds)
    report.update(steps=steps, seeds=seeds, derived_from_saved_prefix=True, rerun=False,
                  note=f"Development seeds 211–310; first {steps} steps of original 320-step p=8 reference runs. All window metrics recomputed. No rerun. Timing uses retained original measurements; excludes reference loading.")
    case.write_json(output / "summary.json", report)
    case.write_json(output / "progress.json", dict(state="complete", completed=100, steps=steps,
                    successes=report["successes"], derived_from_saved_prefix=True, rerun=False, updated_unix=time.time()))
    print(json.dumps({k: report[k] for k in ("complete", "completed", "successes", "steps", "median_final100_mm", "fallback_steps", "modular_timing_steps6_onwards")}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    derive(args.source, args.output, args.steps)
