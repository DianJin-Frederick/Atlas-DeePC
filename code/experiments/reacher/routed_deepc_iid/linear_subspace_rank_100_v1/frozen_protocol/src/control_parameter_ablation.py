"""Resumable control-parameter sweeps for the baseline Atlas-DeePC controller.

The worker processes load the frozen Hankel library and behavioral atlas once,
then reuse the same offline chart-matrix cache for every seed and parameter
value.  Only compact per-rollout summaries are retained.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np

import routed_deepc_experiment as routed
from local_affine_atlas_experiment import audit_iid_library


COMPACT_RESULT_FIELDS = (
    "seed",
    "lam_g",
    "minimum",
    "minimum_step",
    "last",
    "mean_last_30",
    "max_last_30",
    "mean_last_100",
    "max_last_100",
    "success_30",
    "success_100",
    "mean_prediction_error_xy",
    "median_solve_time_ms",
    "p95_solve_time_ms",
    "median_control_compute_time_ms",
    "p95_control_compute_time_ms",
    "charts_visited",
    "statuses",
    "closed_loop_reference_cost",
    "closed_loop_reference_tracking_cost",
    "mean_closed_loop_reference_stage_cost",
    "closed_loop_final_goal_cost",
    "closed_loop_final_goal_tracking_cost",
    "mean_closed_loop_final_goal_stage_cost",
    "closed_loop_input_cost",
    "data_goal_index",
    "data_goal_chart",
)


def compact_result(summary: dict) -> dict:
    """Return the metrics needed by the ablation notebook."""
    missing = [field for field in COMPACT_RESULT_FIELDS if field not in summary]
    if missing:
        raise KeyError(f"Rollout summary is missing compact fields: {missing}")
    return {field: summary[field] for field in COMPACT_RESULT_FIELDS}


def lambda_directory_name(index: int, value: float) -> str:
    """Stable, sortable directory name for a lambda value."""
    token = f"{float(value):.12e}".replace("+", "p").replace("-", "m").replace(".", "d")
    return f"lam_g_{index:03d}_{token}"


def _read_jsonl(path: Path) -> dict[int, dict]:
    results = {}
    if not path.exists():
        return results
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            result = compact_result(json.loads(line))
        except (json.JSONDecodeError, KeyError) as error:
            raise RuntimeError(f"Invalid checkpoint line {line_number}: {path}") from error
        results[int(result["seed"])] = result
    return results


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


_H = None
_ATLAS = None
_BASE_CONFIG = None
_MATRIX_CACHE = None


def _initialize_worker(data_path: str, atlas_path: str, config_payload: dict) -> None:
    global _H, _ATLAS, _BASE_CONFIG, _MATRIX_CACHE
    _H = np.load(data_path, mmap_mode="r")
    _ATLAS = routed.load_atlas(Path(atlas_path))
    _BASE_CONFIG = routed.RoutedDeePCConfig(**config_payload)
    _MATRIX_CACHE = routed.precompute_chart_matrix_cache(_H, _ATLAS, _BASE_CONFIG)


def _run_worker(task: tuple[float, int]) -> dict:
    lam_g, seed = task
    config = replace(_BASE_CONFIG, lam_g=float(lam_g), seed=int(seed))
    summary = routed.run_routed_deepc(
        _H,
        _ATLAS,
        config,
        matrix_cache=_MATRIX_CACHE,
    )
    return compact_result(summary)


def _load_lambda_checkpoint(
    output_dir: Path,
    config: routed.RoutedDeePCConfig,
    seeds: list[int],
    overwrite: bool,
) -> tuple[dict[int, dict], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "compact_results.jsonl"
    metadata_path = output_dir / "checkpoint_metadata.json"
    metadata = {
        "config": asdict(config),
        "seeds": seeds,
        "result_fields": list(COMPACT_RESULT_FIELDS),
    }
    if overwrite:
        checkpoint_path.write_text("")
        _write_json(metadata_path, metadata)
        return {}, checkpoint_path
    if metadata_path.exists():
        saved_metadata = json.loads(metadata_path.read_text())
        saved_seeds = [int(seed) for seed in saved_metadata.get("seeds", [])]
        seeds_can_expand = (
            saved_metadata.get("config") == metadata["config"]
            and saved_metadata.get("result_fields") == metadata["result_fields"]
            and set(saved_seeds).issubset(seeds)
            and saved_seeds == [seed for seed in seeds if seed in set(saved_seeds)]
        )
        if saved_metadata != metadata and not seeds_can_expand:
            raise RuntimeError(f"Incompatible checkpoint metadata: {metadata_path}")
        if saved_metadata != metadata:
            _write_json(metadata_path, metadata)
    else:
        _write_json(metadata_path, metadata)
    results = _read_jsonl(checkpoint_path)
    extras = sorted(set(results) - set(seeds))
    if extras:
        raise RuntimeError(f"Checkpoint contains seeds outside the requested panel: {extras[:10]}")
    return results, checkpoint_path


def _save_lambda_summary(
    output_dir: Path,
    config: routed.RoutedDeePCConfig,
    seeds: list[int],
    results: dict[int, dict],
) -> dict:
    missing = [seed for seed in seeds if seed not in results]
    payload = {
        "method": "graph_routed_fixed_chart_deepc_lam_g_ablation",
        "parameter": "lam_g",
        "parameter_value": float(config.lam_g),
        "config": asdict(config),
        "requested_seeds": seeds,
        "complete": not missing,
        "missing_seeds": missing,
        "results": [results[seed] for seed in seeds if seed in results],
    }
    _write_json(output_dir / "summary.json", payload)
    return payload


def run_lam_g_sweep(
    lam_g_values,
    seeds,
    baseline_dir,
    output_root,
    data_path,
    workers=1,
    overwrite=False,
) -> dict:
    """Run or resume a paired ``lam_g`` sweep of the frozen baseline.

    One process pool is kept alive across parameter values, so each worker pays
    the data-loading and chart-basis construction costs only once. Parameter
    values are processed in order and each lambda summary is finalized before
    moving to the next value.
    """
    lam_g_values = [float(value) for value in lam_g_values]
    seeds = [int(seed) for seed in seeds]
    if len(lam_g_values) != len(set(lam_g_values)):
        raise ValueError("lam_g_values must be unique")
    if any(not np.isfinite(value) or value < 0 for value in lam_g_values):
        raise ValueError("Every lam_g value must be finite and nonnegative")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    if int(workers) < 1:
        raise ValueError("workers must be at least one")

    baseline_dir = Path(baseline_dir)
    output_root = Path(output_root)
    data_path = Path(data_path)
    atlas_path = baseline_dir / "behavioral_atlas.npz"
    baseline_summary_path = baseline_dir / "summary.json"
    for path in (data_path, atlas_path, baseline_summary_path):
        if not path.exists():
            raise FileNotFoundError(path)

    baseline_payload = json.loads(baseline_summary_path.read_text())
    base_config = routed.RoutedDeePCConfig(**baseline_payload["config"])
    output_root.mkdir(parents=True, exist_ok=True)
    H = np.load(data_path, mmap_mode="r")
    audit = audit_iid_library(H, eval_seeds=seeds)
    manifest = {
        "method": "graph_routed_fixed_chart_deepc_lam_g_ablation",
        "baseline_source": str(baseline_dir),
        "data_path": str(data_path),
        "atlas_path": str(atlas_path),
        "base_config": asdict(base_config),
        "parameter": "lam_g",
        "parameter_values": lam_g_values,
        "seeds": seeds,
        "workers": int(workers),
        "fairness_audit": audit,
        "lambda_directories": [
            lambda_directory_name(index, value)
            for index, value in enumerate(lam_g_values)
        ],
    }
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists() and not overwrite:
        saved_manifest = json.loads(manifest_path.read_text())
        comparable_keys = (
            "base_config", "parameter", "parameter_values",
        )
        path_keys = ("baseline_source", "data_path", "atlas_path")
        paths_compatible = all(
            Path(saved_manifest.get(key, "")).resolve()
            == Path(manifest.get(key, "")).resolve()
            for key in path_keys
        )
        saved_seeds = [int(seed) for seed in saved_manifest.get("seeds", [])]
        seeds_can_expand = (
            set(saved_seeds).issubset(seeds)
            and saved_seeds == [seed for seed in seeds if seed in set(saved_seeds)]
        )
        if (
            any(saved_manifest.get(key) != manifest.get(key) for key in comparable_keys)
            or not paths_compatible
            or not seeds_can_expand
        ):
            raise RuntimeError(f"Incompatible sweep manifest: {manifest_path}")
    _write_json(manifest_path, manifest)

    initargs = (str(data_path), str(atlas_path), asdict(base_config))
    executor = None
    if int(workers) > 1:
        executor = ProcessPoolExecutor(
            max_workers=int(workers),
            initializer=_initialize_worker,
            initargs=initargs,
        )
    else:
        _initialize_worker(*initargs)

    summaries = []
    try:
        for index, lam_g in enumerate(lam_g_values):
            config = replace(base_config, lam_g=lam_g)
            lambda_dir = output_root / lambda_directory_name(index, lam_g)
            results, checkpoint_path = _load_lambda_checkpoint(
                lambda_dir, config, seeds, bool(overwrite)
            )
            pending = [seed for seed in seeds if seed not in results]
            print(
                f"lam_g={lam_g:.8g} | saved={len(results)} | pending={len(pending)}",
                flush=True,
            )
            completed_this_call = 0

            def retain(result: dict) -> None:
                nonlocal completed_this_call
                seed = int(result["seed"])
                with checkpoint_path.open("a") as checkpoint:
                    checkpoint.write(json.dumps(result) + "\n")
                results[seed] = result
                completed_this_call += 1
                if (
                    completed_this_call == 1
                    or completed_this_call % 25 == 0
                    or len(results) == len(seeds)
                ):
                    success = bool(result["success_30"] and result["success_100"])
                    print(
                        f"  progress={len(results)}/{len(seeds)} "
                        f"latest_seed={seed} success={success} "
                        f"final100={1000 * result['mean_last_100']:.3f} mm "
                        f"cost={result['closed_loop_final_goal_cost']:.6g}",
                        flush=True,
                    )

            if executor is None:
                for seed in pending:
                    retain(_run_worker((lam_g, seed)))
            elif pending:
                futures = {
                    executor.submit(_run_worker, (lam_g, seed)): seed
                    for seed in pending
                }
                for future in as_completed(futures):
                    seed = futures[future]
                    try:
                        retain(future.result())
                    except Exception as error:
                        print(f"  seed={seed} failed and remains pending: {error!r}", flush=True)

            summary = _save_lambda_summary(lambda_dir, config, seeds, results)
            summaries.append(summary)
            print(
                f"completed lam_g={lam_g:.8g}: "
                f"{len(summary['results'])}/{len(seeds)} saved",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    manifest["complete"] = all(summary["complete"] for summary in summaries)
    manifest["completed_parameter_values"] = [
        summary["parameter_value"] for summary in summaries if summary["complete"]
    ]
    _write_json(manifest_path, manifest)
    return manifest


def load_lam_g_sweep(output_root) -> tuple[dict, dict[float, dict]]:
    """Load the manifest and every existing per-lambda summary."""
    output_root = Path(output_root)
    manifest = json.loads((output_root / "manifest.json").read_text())
    payloads = {}
    for index, value in enumerate(manifest["parameter_values"]):
        summary_path = output_root / lambda_directory_name(index, value) / "summary.json"
        if summary_path.exists():
            payload = json.loads(summary_path.read_text())
            payloads[float(value)] = payload
    return manifest, payloads
