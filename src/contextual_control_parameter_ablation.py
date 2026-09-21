"""Resumable ``lam_g`` ablation for triggered contextual Atlas-DeePC.

The sweep freezes the physical atlas, the 128-prototype contextual artifact,
the causal activation trigger, routing, costs, constraints, and seed panel from
``baseline_contextual_atlas_deepc.ipynb``.  Only ``lam_g`` changes.
"""

from __future__ import annotations

import argparse
import fcntl
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

import contextual_cluster_atlas_experiment as contextual
import routed_deepc_experiment as routed
from local_affine_atlas_experiment import audit_iid_library


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "experiments" / "routed_deepc_iid"
DEFAULT_DATA_PATH = ROOT / "data" / "select_iid_offline_hankel.npy"
DEFAULT_PHYSICAL_ATLAS_DIR = RESULT_ROOT / "clustering_jointconfig48_fresh211_310"
DEFAULT_CONTEXTUAL_BASELINE_DIR = (
    RESULT_ROOT / "contextual_cluster_eval211_710_global128_triggered"
)
DEFAULT_OUTPUT_ROOT = (
    RESULT_ROOT / "ablation_contextual_control_parameter_lam_g_500"
)


def _try_acquire_sweep_lock(output_root: Path):
    """Return an open single-writer lock, or ``None`` if a sweep is active."""
    output_root.mkdir(parents=True, exist_ok=True)
    lock = (output_root / ".sweep.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None
    lock.seek(0)
    lock.truncate()
    lock.write(f"{os.getpid()}\n")
    lock.flush()
    return lock


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
    "statuses",
    "closed_loop_reference_cost",
    "closed_loop_final_goal_cost",
    "closed_loop_input_cost",
    "mean_closed_loop_stage_cost",
    "mean_solve_ms",
    "median_solve_ms",
    "p95_solve_ms",
    "mean_control_compute_ms",
    "median_control_compute_ms",
    "p95_control_compute_ms",
    "contextual_triggered",
    "contextual_trigger_step",
    "contextual_usage_fraction",
    "data_goal_index",
    "data_goal_chart",
)


def compact_result(summary: dict) -> dict:
    missing = [field for field in COMPACT_RESULT_FIELDS if field not in summary]
    if missing:
        raise KeyError(f"Contextual rollout summary is missing fields: {missing}")
    return {field: summary[field] for field in COMPACT_RESULT_FIELDS}


def lam_g_values(log_grid_size: int = 50, baseline_value: float = 0.05) -> np.ndarray:
    positive = np.unique(
        np.r_[np.logspace(-2, 2, int(log_grid_size)), float(baseline_value)]
    )
    positive.sort()
    return np.r_[0.0, positive]


def lambda_directory_name(index: int, value: float) -> str:
    token = (
        f"{float(value):.12e}"
        .replace("+", "p")
        .replace("-", "m")
        .replace(".", "d")
    )
    return f"lam_g_{index:03d}_{token}"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _normalized_contextual_config(values: dict) -> dict:
    return {**asdict(contextual.ContextualClusterConfig()), **values}


def _read_jsonl(path: Path) -> dict[int, dict]:
    results = {}
    if not path.exists():
        return results
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = compact_result(json.loads(line))
        except (json.JSONDecodeError, KeyError) as error:
            raise RuntimeError(
                f"Invalid contextual checkpoint line {line_number}: {path}"
            ) from error
        results[int(row["seed"])] = row
    return results


_H = None
_PHYSICAL_ATLAS = None
_CONTEXTUAL_ARTIFACT = None
_BASE_CONTROL_CONFIG = None
_CONTEXTUAL_CONFIG = None
_PHYSICAL_MATRIX_CACHE = None


def _initialize_worker(
    data_path: str,
    physical_atlas_path: str,
    contextual_artifact_path: str,
    control_payload: dict,
    contextual_payload: dict,
) -> None:
    global _H, _PHYSICAL_ATLAS, _CONTEXTUAL_ARTIFACT
    global _BASE_CONTROL_CONFIG, _CONTEXTUAL_CONFIG, _PHYSICAL_MATRIX_CACHE
    _H = np.load(data_path, mmap_mode="r")
    _PHYSICAL_ATLAS = routed.load_atlas(Path(physical_atlas_path))
    _CONTEXTUAL_ARTIFACT = contextual.load_contextual_cluster_atlas(
        Path(contextual_artifact_path)
    )
    _BASE_CONTROL_CONFIG = routed.RoutedDeePCConfig(**control_payload)
    _CONTEXTUAL_CONFIG = contextual.ContextualClusterConfig(**contextual_payload)
    saved_contextual = _normalized_contextual_config(
        json.loads(_CONTEXTUAL_ARTIFACT["cluster_config"].item())
    )
    if saved_contextual != asdict(_CONTEXTUAL_CONFIG):
        raise RuntimeError("Worker contextual artifact/config mismatch")
    available_rank = int(
        _CONTEXTUAL_ARTIFACT.get(
            "basis_rank", _CONTEXTUAL_ARTIFACT["matrix_bases"].shape[2]
        )
    )
    if available_rank != _BASE_CONTROL_CONFIG.svd_rank:
        raise RuntimeError(
            f"Contextual artifact rank {available_rank} does not match "
            f"controller rank {_BASE_CONTROL_CONFIG.svd_rank}"
        )
    _PHYSICAL_MATRIX_CACHE = contextual.physical_matrix_cache_from_artifact(
        _CONTEXTUAL_ARTIFACT
    )


def _run_worker(task: tuple[float, int]) -> dict:
    value, seed = task
    config = replace(
        _BASE_CONTROL_CONFIG,
        lam_g=float(value),
        seed=int(seed),
    )
    result = contextual.run_contextual_cluster_atlas(
        _H,
        _PHYSICAL_ATLAS,
        _CONTEXTUAL_ARTIFACT,
        config,
        _CONTEXTUAL_CONFIG,
        physical_matrix_cache=_PHYSICAL_MATRIX_CACHE,
    )
    return compact_result(result)


def _load_lambda_checkpoint(
    output_dir: Path,
    control_config: routed.RoutedDeePCConfig,
    contextual_config: contextual.ContextualClusterConfig,
    seeds: list[int],
    overwrite: bool,
) -> tuple[dict[int, dict], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "compact_results.jsonl"
    metadata_path = output_dir / "checkpoint_metadata.json"
    metadata = {
        "config": asdict(control_config),
        "contextual_cluster": asdict(contextual_config),
        "seeds": seeds,
        "result_fields": list(COMPACT_RESULT_FIELDS),
    }
    if overwrite:
        checkpoint_path.write_text("")
        _write_json(metadata_path, metadata)
        return {}, checkpoint_path
    if metadata_path.exists():
        saved = json.loads(metadata_path.read_text())
        if saved != metadata:
            raise RuntimeError(f"Incompatible contextual checkpoint: {metadata_path}")
    else:
        _write_json(metadata_path, metadata)
    results = _read_jsonl(checkpoint_path)
    extras = sorted(set(results) - set(seeds))
    if extras:
        raise RuntimeError(
            f"Contextual checkpoint contains out-of-panel seeds: {extras[:10]}"
        )
    return results, checkpoint_path


def _same_control_config(
    payload: dict,
    expected: routed.RoutedDeePCConfig,
) -> bool:
    normalized = asdict(routed.RoutedDeePCConfig(**payload))
    expected_payload = asdict(expected)
    return all(
        normalized[key] == expected_payload[key]
        for key in expected_payload
        if key != "seed"
    )


def _bootstrap_baseline_rows(
    baseline_payload: dict,
    expected_control: routed.RoutedDeePCConfig,
    expected_contextual: contextual.ContextualClusterConfig,
    seeds: list[int],
    results: dict[int, dict],
    checkpoint_path: Path,
) -> int:
    if not _same_control_config(baseline_payload["config"], expected_control):
        return 0
    if (
        _normalized_contextual_config(baseline_payload["contextual_cluster"])
        != asdict(expected_contextual)
    ):
        return 0
    retained = 0
    for row in baseline_payload.get("results", []):
        seed = int(row["seed"])
        if seed not in seeds or seed in results:
            continue
        compact = compact_result(row)
        if float(compact["lam_g"]) != float(expected_control.lam_g):
            continue
        with checkpoint_path.open("a") as checkpoint:
            checkpoint.write(json.dumps(compact) + "\n")
        results[seed] = compact
        retained += 1
    return retained


def _save_lambda_summary(
    output_dir: Path,
    control_config: routed.RoutedDeePCConfig,
    contextual_config: contextual.ContextualClusterConfig,
    seeds: list[int],
    results: dict[int, dict],
) -> dict:
    missing = [seed for seed in seeds if seed not in results]
    payload = {
        "method": "triggered_contextual_atlas_deepc_lam_g_ablation",
        "parameter": "lam_g",
        "parameter_value": float(control_config.lam_g),
        "config": asdict(control_config),
        "contextual_cluster": asdict(contextual_config),
        "requested_seeds": seeds,
        "complete": not missing,
        "missing_seeds": missing,
        "results": [results[seed] for seed in seeds if seed in results],
    }
    _write_json(output_dir / "summary.json", payload)
    return payload


def run_lam_g_sweep(
    values,
    seeds,
    physical_atlas_dir=DEFAULT_PHYSICAL_ATLAS_DIR,
    contextual_baseline_dir=DEFAULT_CONTEXTUAL_BASELINE_DIR,
    output_root=DEFAULT_OUTPUT_ROOT,
    data_path=DEFAULT_DATA_PATH,
    workers=6,
    max_rollouts=None,
    overwrite=False,
) -> dict:
    """Run or resume a contextual ``lam_g`` sweep.

    ``max_rollouts`` limits newly computed rows, making repeated scheduled runs
    predictable. Saved and bootstrapped rows do not consume that budget.
    """
    values = [float(value) for value in values]
    seeds = [int(seed) for seed in seeds]
    if len(values) != len(set(values)):
        raise ValueError("lam_g values must be unique")
    if any(not np.isfinite(value) or value < 0 for value in values):
        raise ValueError("lam_g values must be finite and nonnegative")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    if int(workers) < 1:
        raise ValueError("workers must be at least one")
    if max_rollouts is not None and int(max_rollouts) < 1:
        raise ValueError("max_rollouts must be positive when supplied")

    physical_atlas_dir = Path(physical_atlas_dir).resolve()
    contextual_baseline_dir = Path(contextual_baseline_dir).resolve()
    output_root = Path(output_root).resolve()
    data_path = Path(data_path).resolve()
    physical_atlas_path = physical_atlas_dir / "behavioral_atlas.npz"
    contextual_artifact_path = (
        contextual_baseline_dir / "contextual_cluster_atlas.npz"
    )
    baseline_summary_path = contextual_baseline_dir / "summary.json"
    for path in (
        data_path,
        physical_atlas_path,
        contextual_artifact_path,
        baseline_summary_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    baseline_payload = json.loads(baseline_summary_path.read_text())
    base_control = routed.RoutedDeePCConfig(**baseline_payload["config"])
    contextual_config = contextual.ContextualClusterConfig(
        **baseline_payload["contextual_cluster"]
    )
    artifact = contextual.load_contextual_cluster_atlas(contextual_artifact_path)
    saved_contextual = _normalized_contextual_config(
        json.loads(artifact["cluster_config"].item())
    )
    if saved_contextual != asdict(contextual_config):
        raise RuntimeError("Frozen contextual artifact does not match baseline config")
    if contextual_config.activation_mode != "triggered":
        raise RuntimeError("This ablation requires the triggered contextual baseline")

    output_root.mkdir(parents=True, exist_ok=True)
    H = np.load(data_path, mmap_mode="r")
    manifest = {
        "method": "triggered_contextual_atlas_deepc_lam_g_ablation",
        "physical_atlas_source": str(physical_atlas_dir),
        "contextual_baseline_source": str(contextual_baseline_dir),
        "contextual_artifact_source": str(contextual_artifact_path),
        "data_path": str(data_path),
        "base_config": asdict(base_control),
        "contextual_cluster": asdict(contextual_config),
        "parameter": "lam_g",
        "parameter_values": values,
        "seeds": seeds,
        "workers": int(workers),
        "fairness_audit": audit_iid_library(H, eval_seeds=seeds),
        "lambda_directories": [
            lambda_directory_name(index, value)
            for index, value in enumerate(values)
        ],
    }
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists() and not overwrite:
        saved_manifest = json.loads(manifest_path.read_text())
        comparable = (
            "base_config",
            "contextual_cluster",
            "parameter",
            "parameter_values",
            "seeds",
            "physical_atlas_source",
            "contextual_baseline_source",
            "contextual_artifact_source",
            "data_path",
        )
        if any(saved_manifest.get(key) != manifest.get(key) for key in comparable):
            raise RuntimeError(f"Incompatible contextual sweep: {manifest_path}")
    _write_json(manifest_path, manifest)

    initargs = (
        str(data_path),
        str(physical_atlas_path),
        str(contextual_artifact_path),
        asdict(base_control),
        asdict(contextual_config),
    )
    executor = None
    if int(workers) > 1:
        executor = ProcessPoolExecutor(
            max_workers=int(workers),
            initializer=_initialize_worker,
            initargs=initargs,
        )
    else:
        _initialize_worker(*initargs)

    remaining_budget = None if max_rollouts is None else int(max_rollouts)
    summaries = []
    newly_computed = 0
    try:
        for index, value in enumerate(values):
            control_config = replace(base_control, lam_g=value)
            lambda_dir = output_root / lambda_directory_name(index, value)
            results, checkpoint_path = _load_lambda_checkpoint(
                lambda_dir,
                control_config,
                contextual_config,
                seeds,
                bool(overwrite),
            )
            if value == float(base_control.lam_g) and not overwrite:
                bootstrapped = _bootstrap_baseline_rows(
                    baseline_payload,
                    control_config,
                    contextual_config,
                    seeds,
                    results,
                    checkpoint_path,
                )
                if bootstrapped:
                    print(
                        f"lam_g={value:.8g}: bootstrapped {bootstrapped} baseline rows",
                        flush=True,
                    )
            pending = [seed for seed in seeds if seed not in results]
            if remaining_budget is not None:
                pending = pending[:remaining_budget]
            print(
                f"lam_g={value:.8g} | saved={len(results)} | "
                f"scheduled_now={len(pending)}",
                flush=True,
            )
            completed_this_value = 0

            def retain(row: dict) -> None:
                nonlocal completed_this_value, newly_computed, remaining_budget
                seed = int(row["seed"])
                with checkpoint_path.open("a") as checkpoint:
                    checkpoint.write(json.dumps(row) + "\n")
                results[seed] = row
                completed_this_value += 1
                newly_computed += 1
                if remaining_budget is not None:
                    remaining_budget -= 1
                if (
                    completed_this_value == 1
                    or completed_this_value % 25 == 0
                    or completed_this_value == len(pending)
                ):
                    strict = bool(row["success_30"] and row["success_100"])
                    print(
                        f"  progress={len(results)}/{len(seeds)} "
                        f"latest_seed={seed} strict={strict} "
                        f"triggered={row['contextual_triggered']}",
                        flush=True,
                    )

            if executor is None:
                for seed in pending:
                    retain(_run_worker((value, seed)))
            elif pending:
                futures = {
                    executor.submit(_run_worker, (value, seed)): seed
                    for seed in pending
                }
                for future in as_completed(futures):
                    seed = futures[future]
                    try:
                        retain(future.result())
                    except Exception as error:
                        print(
                            f"  seed={seed} failed and remains pending: {error!r}",
                            flush=True,
                        )
            summaries.append(
                _save_lambda_summary(
                    lambda_dir,
                    control_config,
                    contextual_config,
                    seeds,
                    results,
                )
            )
            if remaining_budget == 0:
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    saved_rows = 0
    completed_values = []
    for index, value in enumerate(values):
        summary_path = (
            output_root / lambda_directory_name(index, value) / "summary.json"
        )
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text())
        saved_rows += len(payload.get("results", []))
        if payload.get("complete"):
            completed_values.append(float(value))
    manifest.update({
        "complete": saved_rows == len(values) * len(seeds),
        "saved_rollouts": saved_rows,
        "total_rollouts": len(values) * len(seeds),
        "new_rollouts_this_run": newly_computed,
        "completed_parameter_values": completed_values,
    })
    _write_json(manifest_path, manifest)
    print(
        f"contextual sweep progress: {saved_rows}/{manifest['total_rollouts']} "
        f"rollouts; {len(completed_values)}/{len(values)} values complete",
        flush=True,
    )
    return manifest


def load_lam_g_sweep(output_root=DEFAULT_OUTPUT_ROOT) -> tuple[dict, dict[float, dict]]:
    output_root = Path(output_root)
    manifest = json.loads((output_root / "manifest.json").read_text())
    payloads = {}
    for index, value in enumerate(manifest["parameter_values"]):
        summary_path = (
            output_root / lambda_directory_name(index, value) / "summary.json"
        )
        if summary_path.exists():
            payloads[float(value)] = json.loads(summary_path.read_text())
    return manifest, payloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Triggered contextual Atlas-DeePC lam_g ablation"
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--max-rollouts",
        type=int,
        help="Maximum number of new rollouts to compute in this invocation.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock = _try_acquire_sweep_lock(Path(args.output_root).resolve())
    if lock is None:
        print(
            f"contextual sweep already active for {Path(args.output_root).resolve()}; "
            "exiting without starting a duplicate",
            flush=True,
        )
        return
    try:
        run_lam_g_sweep(
            lam_g_values(),
            range(211, 711),
            output_root=args.output_root,
            workers=args.workers,
            max_rollouts=args.max_rollouts,
            overwrite=args.overwrite,
        )
    finally:
        lock.close()


if __name__ == "__main__":
    main()
