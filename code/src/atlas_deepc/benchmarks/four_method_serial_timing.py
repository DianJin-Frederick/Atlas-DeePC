"""Controlled serial timing for the four Reacher DeePC comparison methods.

This script is intentionally separate from the large outcome benchmark.  It
never loads per-seed rollout results, never starts worker processes, and rotates
method order across seeds so machine drift does not systematically favor one
method.  Offline atlas/artifact loading and the optional warm-up are outside the
reported online timing boundary.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from atlas_deepc.common.paths import project_root

import argparse
import contextlib
import gc
import json
import os
import platform
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

# Keep each numerical library single-threaded before importing NumPy/CVXPY.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

import numpy as np

from atlas_deepc.atlas import contextual_cluster_atlas_experiment as contextual_atlas
from atlas_deepc.atlas import routed_deepc_experiment as routed
from atlas_deepc.atlas.contextual_chart_expansion_experiment import baseline_config
from atlas_deepc.benchmarks.online_selected_deepc_comparison import run_online_selected_deepc


ROOT = project_root()
DEFAULT_DATA_PATH = ROOT / "data/select_iid_offline_hankel.npy"
DEFAULT_BASELINE_DIR = (
    ROOT / "experiments/routed_deepc_iid/clustering_jointconfig48_fresh211_310"
)
DEFAULT_CONTEXTUAL_DIR = (
    ROOT
    / "experiments/routed_deepc_iid/contextual_cluster_eval211_710_global128_triggered"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "experiments/deepc_algorithm_comparison/four_method_controlled_serial_timing"
)

METHODS = (
    "Select-DPC",
    "Atlas-DeePC",
    "Contextual DPC",
    "Contextual Atlas-DeePC",
)

STAGE_KEYS = {
    "localization": "localization_time_s",
    "graph routing": "routing_time_s",
    "data / prototype selection": None,
    "matrix / SVD preparation": None,
    "QP solve": "solve_time_s",
    "total online control": "control_compute_time_s",
}


def parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    else:
        seeds = list(range(args.seed_start, args.seed_stop + 1))
    if not seeds:
        raise ValueError("At least one timing seed is required.")
    return seeds


def rotated_order(index: int) -> list[str]:
    shift = index % len(METHODS)
    order = list(METHODS[shift:] + METHODS[:shift])
    # Reverse alternating four-seed blocks to balance slow monotone drift.
    if (index // len(METHODS)) % 2:
        order.reverse()
    return order


def run_method(
    method: str,
    seed: int,
    config: routed.RoutedDeePCConfig,
    output_path: Path,
    H: np.ndarray,
    atlas: dict,
    atlas_cache: dict,
    artifact: dict,
    cluster_config: contextual_atlas.ContextualClusterConfig,
) -> dict:
    control_config = replace(config, seed=int(seed))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        if method == "Atlas-DeePC":
            return routed.run_routed_deepc(
                H,
                atlas,
                control_config,
                save_path=output_path,
                matrix_cache=atlas_cache,
            )
        if method == "Select-DPC":
            return run_online_selected_deepc(
                H,
                atlas,
                control_config,
                selected_columns=200,
                selected_model="rank_matched",
                selection_method="select_dpc",
                reference_mode="graph_routed",
                save_path=output_path,
            )
        if method == "Contextual DPC":
            return run_online_selected_deepc(
                H,
                atlas,
                control_config,
                selected_columns=200,
                selected_model="rank_matched",
                selection_method="contextual",
                reference_mode="final_only",
                save_path=output_path,
            )
        if method == "Contextual Atlas-DeePC":
            return contextual_atlas.run_contextual_cluster_atlas(
                H,
                atlas,
                artifact,
                control_config,
                cluster_config,
                save_path=output_path,
                physical_matrix_cache=atlas_cache,
            )
    raise ValueError(f"Unknown method: {method}")


def rollout_stage_medians(path: Path, method: str, discard: int) -> dict[str, float]:
    with np.load(path, allow_pickle=True) as data:
        steps = len(data["control_compute_time_s"])
        start = min(discard, max(steps - 1, 0))

        def milliseconds(key: str) -> np.ndarray:
            return 1000.0 * np.asarray(data[key], dtype=float)[start:]

        selection_key = (
            "cluster_scoring_time_s"
            if method == "Contextual Atlas-DeePC"
            else "selection_time_s"
        )
        if selection_key in data.files:
            selection = milliseconds(selection_key)
        else:
            selection = np.zeros(steps - start)

        if "svd_rebuild_time_s" in data.files:
            matrix = milliseconds("svd_rebuild_time_s")
        else:
            matrix = milliseconds("matrix_preparation_time_s")

        arrays = {
            "localization": milliseconds("localization_time_s"),
            "graph routing": milliseconds("routing_time_s"),
            "data / prototype selection": selection,
            "matrix / SVD preparation": matrix,
            "QP solve": milliseconds("solve_time_s"),
            "total online control": milliseconds("control_compute_time_s"),
        }
        return {name: float(np.median(values)) for name, values in arrays.items()}


def aggregate(records: list[dict]) -> list[dict]:
    rows = []
    for method in METHODS:
        selected = [record for record in records if record["method"] == method]
        row = {"method": method, "rollouts": len(selected)}
        for stage in STAGE_KEYS:
            row[f"median {stage} (ms)"] = float(
                np.median([record[stage] for record in selected])
            )
        rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", help="Comma-separated seed list; overrides the range.")
    parser.add_argument("--seed-start", type=int, default=211)
    parser.add_argument("--seed-stop", type=int, default=310)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-seed", type=int, default=210)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--discard-initial-steps", type=int, default=5)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--contextual-dir", type=Path, default=DEFAULT_CONTEXTUAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(args: argparse.Namespace) -> None:
    seeds = parse_seeds(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    H = np.load(args.data_path)
    baseline_payload = json.loads((args.baseline_dir / "summary.json").read_text())
    config = replace(
        baseline_config(baseline_payload),
        steps=args.steps,
        svd_rank=60,
    )
    atlas = routed.load_atlas(args.baseline_dir / "behavioral_atlas.npz")
    atlas_cache = routed.precompute_chart_matrix_cache(H, atlas, config)
    artifact = contextual_atlas.load_contextual_cluster_atlas(
        args.contextual_dir / "contextual_cluster_atlas.npz"
    )
    cluster_config = contextual_atlas.ContextualClusterConfig(
        **json.loads(artifact["cluster_config"].item())
    )
    assert cluster_config.activation_mode == "triggered"
    artifact_cache = contextual_atlas.physical_matrix_cache_from_artifact(artifact)
    for chart in range(config.clusters):
        assert np.allclose(
            atlas_cache[(chart,)][0]["basis"],
            artifact_cache[(chart,)][0]["basis"],
            atol=1e-12,
            rtol=1e-12,
        )

    # Warm every exact rollout path once. These files and times are excluded.
    warmup_config = replace(config, steps=args.warmup_steps)
    for method in METHODS:
        gc.collect()
        run_method(
            method,
            args.warmup_seed,
            warmup_config,
            args.output_dir / "warmup" / method / "rollout.npz",
            H,
            atlas,
            atlas_cache,
            artifact,
            cluster_config,
        )
    print(f"Warm-up complete. Timing {len(seeds)} matched seeds serially.", flush=True)

    records = []
    for index, seed in enumerate(seeds):
        order = rotated_order(index)
        for method in order:
            gc.collect()
            path = args.output_dir / "rollouts" / method / f"seed{seed}.npz"
            summary = run_method(
                method,
                seed,
                config,
                path,
                H,
                atlas,
                atlas_cache,
                artifact,
                cluster_config,
            )
            timing = rollout_stage_medians(path, method, args.discard_initial_steps)
            records.append(
                {
                    "seed": int(seed),
                    "method": method,
                    "execution_position": order.index(method),
                    "strict_success": bool(summary["success_30"] and summary["success_100"]),
                    **timing,
                }
            )
            print(
                f"seed={seed} method={method} "
                f"total={timing['total online control']:.2f} ms "
                f"QP={timing['QP solve']:.2f} ms",
                flush=True,
            )

    payload = {
        "protocol": {
            "description": "fresh serial, matched seeds, rotated method order",
            "seed_range": [min(seeds), max(seeds)],
            "seeds": seeds,
            "steps": args.steps,
            "warmup_seed": args.warmup_seed,
            "warmup_steps": args.warmup_steps,
            "discard_initial_steps_per_rollout": args.discard_initial_steps,
            "workers": 1,
            "thread_limits": {
                name: os.environ[name]
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
            "rank": config.svd_rank,
            "selected_columns": 200,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "python": sys.version,
        },
        "aggregate": aggregate(records),
        "per_seed": records,
    }
    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["aggregate"], indent=2))
    print(f"Saved controlled timing summary: {output_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
