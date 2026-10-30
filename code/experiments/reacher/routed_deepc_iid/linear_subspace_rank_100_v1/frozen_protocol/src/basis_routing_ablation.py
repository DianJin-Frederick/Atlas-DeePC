"""Paired 2x2 ablation of chart centering and graph-routed references."""

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import routed_deepc_experiment as routed
from candidate_goal_routing_experiment import TIMING_KEYS, file_digest, strict_success
from stitched_atlas_mpc_experiment import offline_environment_targets


METHODS = {
    "Affine + graph (baseline)": ("affine", "graph"),
    "Linear + graph": ("linear", "graph"),
    "Affine + terminal": ("affine", "terminal"),
    "Linear + terminal": ("linear", "terminal"),
}


def run_panel(seeds, config, data_path, atlas_path, output_dir, overwrite=False,
              progress_every=10):
    seeds = [int(seed) for seed in seeds]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and unique")
    if config.formulation != "svd_current" or config.target_matrix_mode != "current":
        raise ValueError("This ablation requires the current-chart SVD baseline")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = {
        "version": 1,
        "config": {k: v for k, v in asdict(config).items() if k != "seed"},
        "data_sha256": file_digest(data_path),
        "atlas_sha256": file_digest(atlas_path),
        "runner_sha256": file_digest(Path(routed.__file__)),
        "protocol": "serial_interleaved_one_blas_thread",
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["signature"] != signature:
        raise RuntimeError("Settings or controller implementation changed; use a new output directory")
    with threadpool_limits(limits=1):
        H = np.load(data_path, mmap_mode="r")
        atlas = routed.load_atlas(atlas_path)
        started = time.perf_counter()
        offline_environment_targets(H.shape[1] // routed.WINDOWS_PER_EPISODE)
        offline_times = {"source_metadata_s": time.perf_counter() - started}
        caches = {}
        for geometry in ("affine", "linear"):
            started = time.perf_counter()
            caches[geometry] = routed.precompute_chart_matrix_cache(H, atlas, config, geometry=geometry)
            offline_times[f"{geometry}_chart_svd_s"] = time.perf_counter() - started
        manifest = {"signature": signature, "seeds": seeds, "methods": list(METHODS),
                    "offline_times": offline_times}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        rows = []
        total = len(seeds) * len(METHODS)
        jobs = list(METHODS.items())
        for position, seed in enumerate(seeds):
            offset = position % len(jobs)
            seed_rows = []
            for method, (geometry, reference_mode) in jobs[offset:] + jobs[:offset]:
                folder = output_dir / f"{geometry}_{reference_mode}"
                folder.mkdir(exist_ok=True)
                checkpoint = folder / f"seed{seed}.json"
                if checkpoint.exists() and not overwrite:
                    row = json.loads(checkpoint.read_text())
                    if row.get("signature") != signature or row["method"] != method or row["seed"] != seed:
                        raise RuntimeError(f"Incompatible checkpoint: {checkpoint}")
                else:
                    trace_path = folder / f"seed{seed}.npz"
                    row = routed.run_routed_deepc(
                        H, atlas, replace(config, seed=seed), save_path=trace_path,
                        matrix_cache=caches[geometry], reference_mode=reference_mode,
                        summary_metadata={"matrix_geometry": geometry, "method": method},
                    )
                    with np.load(trace_path, allow_pickle=True) as trace:
                        assert np.all(trace["goal_index"] == trace["goal_index"][0])
                        assert np.all(trace["goal_chart"] == trace["goal_chart"][0])
                        if reference_mode == "terminal":
                            np.testing.assert_array_equal(trace["ref"], np.tile(trace["final_goal"], (config.steps, 1)))
                        for key in TIMING_KEYS:
                            values = 1000 * np.asarray(trace[key], dtype=float)
                            row[f"mean_{key[:-1]}ms"] = float(values.mean())
                            row[f"p95_{key[:-1]}ms"] = float(np.percentile(values, 95))
                        row["mean_other_time_ms"] = float(1000 * np.mean(
                            trace["control_compute_time_s"] - sum(trace[k] for k in TIMING_KEYS[:-1])
                        ))
                        row["reference_change_steps"] = int(np.count_nonzero(np.any(np.diff(trace["ref"], axis=0) != 0, axis=1)))
                        row["unsolved_steps"] = int(sum(str(s) not in ("optimal", "optimal_inaccurate") for s in trace["status"]))
                        compact = {k: trace[k] for k in trace.files if k != "summary"}
                    np.savez_compressed(trace_path, **compact)
                    row["signature"] = signature
                    checkpoint.write_text(json.dumps(row, indent=2))
                rows.append(row)
                seed_rows.append(row)
                if len(rows) % progress_every == 0 or len(rows) == total:
                    print(f"{len(rows)}/{total}: seed {seed}, {method}, strict={strict_success(row)}, tail100={1000*row['mean_last_100']:.3f} mm", flush=True)
            assert len({r["data_goal_index"] for r in seed_rows}) == 1, "Methods constructed different terminal references"
        payload = {"manifest": manifest, "results": rows}
        (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
        return payload
