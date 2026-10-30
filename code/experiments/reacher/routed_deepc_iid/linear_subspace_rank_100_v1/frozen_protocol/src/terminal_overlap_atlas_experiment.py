"""Evaluate terminal current-plus-goal chart overlap in graph-routed Atlas-DeePC."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

import numpy as np

import routed_deepc_experiment as routed


_H = None
_ATLAS = None
_CONFIG = None
_MATRIX_CACHE = None
_OUTPUT_DIR = None


def _load_summary(path: Path) -> dict:
    result = np.load(path, allow_pickle=True)
    return json.loads(result["summary"].item())


def _initialize_worker(
    data_path: str,
    atlas_path: str,
    config_values: dict,
    output_dir: str,
) -> None:
    global _H, _ATLAS, _CONFIG, _MATRIX_CACHE, _OUTPUT_DIR
    _H = np.load(data_path, mmap_mode="r")
    _ATLAS = routed.load_atlas(Path(atlas_path))
    _CONFIG = routed.RoutedDeePCConfig(**config_values)
    _MATRIX_CACHE = routed.precompute_chart_matrix_cache(_H, _ATLAS, _CONFIG)
    _OUTPUT_DIR = Path(output_dir)


def _run_seed(seed: int) -> dict:
    config = routed.RoutedDeePCConfig(
        **{**asdict(_CONFIG), "seed": int(seed)}
    )
    path = _OUTPUT_DIR / f"seed{seed}_control_results.npz"
    result = routed.run_routed_deepc(
        _H,
        _ATLAS,
        config,
        save_path=path,
        matrix_cache=_MATRIX_CACHE,
        summary_metadata={"ablation": f"terminal_{_CONFIG.target_matrix_mode}"},
    )
    result["result_path"] = str(path)
    return result


def run_panel(
    seeds: list[int],
    baseline_dir: Path,
    output_dir: Path,
    data_path: Path,
    workers: int = 1,
    overwrite: bool = False,
    mode: str = "current_goal_overlap",
    terminal_overlap_distance: float = 0.01,
    terminal_overlap_dwell: int = 5,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((baseline_dir / "summary.json").read_text())
    config = routed.RoutedDeePCConfig(
        **{
            **baseline["config"],
            "target_matrix_mode": mode,
            "terminal_overlap_distance": terminal_overlap_distance,
            "terminal_overlap_dwell": terminal_overlap_dwell,
        }
    )
    atlas_path = baseline_dir / "behavioral_atlas.npz"
    if not atlas_path.exists():
        raise FileNotFoundError(atlas_path)

    results = {}
    pending = []
    for seed in seeds:
        path = output_dir / f"seed{seed}_control_results.npz"
        if path.exists() and not overwrite:
            result = _load_summary(path)
            if (
                result.get("target_matrix_mode") != mode
                or result.get("terminal_overlap_distance") != terminal_overlap_distance
                or result.get("terminal_overlap_dwell") != terminal_overlap_dwell
            ):
                raise RuntimeError(f"Incompatible saved result: {path}")
            results[seed] = result
        else:
            pending.append(seed)

    initargs = (
        str(data_path),
        str(atlas_path),
        asdict(config),
        str(output_dir),
    )
    if workers == 1:
        _initialize_worker(*initargs)
        for seed in pending:
            results[seed] = _run_seed(seed)
            result = results[seed]
            print(
                f"seed={seed} success={result['success_30'] and result['success_100']} "
                f"final100={1000 * result['mean_last_100']:.3f} mm",
                flush=True,
            )
    elif pending:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=initargs,
        ) as executor:
            futures = {executor.submit(_run_seed, seed): seed for seed in pending}
            for future in as_completed(futures):
                seed = futures[future]
                results[seed] = future.result()
                result = results[seed]
                print(
                    f"seed={seed} success={result['success_30'] and result['success_100']} "
                    f"final100={1000 * result['mean_last_100']:.3f} mm",
                    flush=True,
                )

    ordered = [results[seed] for seed in seeds]
    H = np.load(data_path, mmap_mode="r")
    atlas = routed.load_atlas(atlas_path)
    payload = {
        "method": f"graph_routed_atlas_terminal_{mode}",
        "changed_component": "target matrix after final-reference latch only",
        "baseline_source": str(baseline_dir),
        "config": asdict(config),
        "fairness_audit": routed.audit_iid_library(H, eval_seeds=seeds),
        "chart_sizes": np.bincount(
            atlas["labels"].astype(int), minlength=config.clusters
        ).tolist(),
        "results": ordered,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-stop", type=int, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/select_iid_offline_hankel.npy"),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("current_goal_overlap", "dwell_current_goal_overlap"),
        default="current_goal_overlap",
    )
    parser.add_argument("--terminal-overlap-distance", type=float, default=0.01)
    parser.add_argument("--terminal-overlap-dwell", type=int, default=5)
    args = parser.parse_args()
    if args.seed_stop <= args.seed_start:
        raise ValueError("seed-stop must be greater than seed-start")
    run_panel(
        list(range(args.seed_start, args.seed_stop)),
        args.baseline_dir,
        args.output_dir,
        args.data_path,
        workers=args.workers,
        overwrite=args.overwrite,
        mode=args.mode,
        terminal_overlap_distance=args.terminal_overlap_distance,
        terminal_overlap_dwell=args.terminal_overlap_dwell,
    )


if __name__ == "__main__":
    main()
