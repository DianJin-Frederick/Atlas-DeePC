"""Ablate the derived cos(q1+q2), sin(q1+q2) Reacher output channels.

The original offline trajectories are retained.  Only the two derived output
rows are removed from every Hankel time block, after which the 48-chart atlas
and all chart SVD bases are refit.  Online control uses the corresponding
eight-channel measurement map.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, fields
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np

import expert_pipeline_search as pipe
import routed_deepc_experiment as routed
from local_affine_atlas_experiment import audit_iid_library


DEFAULT_BASELINE = Path(
    "experiments/routed_deepc_iid/clustering_jointconfig48_fresh211_310/summary.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/routed_deepc_iid/ablation_reacher_output8_eval211_310"
)

_H = None
_ATLAS = None
_CONFIG = None
_CACHE = None
_OUTPUT_DIR = None


def use_native_eight_output() -> None:
    pipe.P = 8


def remove_joint_sum_channels(H10: np.ndarray) -> np.ndarray:
    """Drop output rows 8:10 independently from every Hankel time block."""
    H10 = np.asarray(H10, dtype=float)
    input_rows = pipe.M * pipe.L
    expected_rows = (pipe.M + 10) * pipe.L
    if H10.shape[0] != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in the 10-output Hankel matrix.")
    outputs = H10[input_rows:].reshape(pipe.L, 10, H10.shape[1])
    return np.vstack([H10[:input_rows], outputs[:, :8].reshape(pipe.L * 8, H10.shape[1])])


def exact_mcnemar_p(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(first_only, second_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def strict_success(row: dict) -> bool:
    return bool(row["success_30"] and row["success_100"])


def initialize_worker(data_path: str, atlas_path: str, config: dict, output_dir: str) -> None:
    global _H, _ATLAS, _CONFIG, _CACHE, _OUTPUT_DIR
    use_native_eight_output()
    _H = np.load(data_path)
    _ATLAS = routed.load_atlas(Path(atlas_path))
    _CONFIG = routed.RoutedDeePCConfig(**config)
    _CACHE = routed.precompute_chart_matrix_cache(_H, _ATLAS, _CONFIG)
    _OUTPUT_DIR = Path(output_dir)


def run_seed(seed: int) -> dict:
    config = routed.RoutedDeePCConfig(**{**asdict(_CONFIG), "seed": int(seed)})
    path = _OUTPUT_DIR / f"seed{seed}_control_results.npz"
    summary = routed.run_routed_deepc(
        _H,
        _ATLAS,
        config,
        save_path=path,
        matrix_cache=_CACHE,
        summary_metadata={
            "output_dimension": 8,
            "removed_output_channels": ["cos(q1+q2)", "sin(q1+q2)"],
        },
    )
    summary["result_path"] = str(path)
    return summary


def make_figure(payload: dict, output_dir: Path) -> None:
    paired = payload["paired_comparison"]
    baseline = payload["baseline"]
    reduced = payload["reduced_output"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = ["10 outputs\n(baseline)", "8 native outputs"]
    rates = [baseline["strict_success_rate"], reduced["strict_success_rate"]]
    axes[0].bar(labels, rates, color=["#415a77", "#2a9d8f"], width=0.62)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("strict success rate")
    axes[0].set_title("Same 100 evaluation seeds")
    for index, (rate, count) in enumerate(zip(rates, [baseline["strict_success_count"], reduced["strict_success_count"]])):
        axes[0].text(index, rate + 0.025, f"{count}/100", ha="center", fontweight="bold")

    categories = ["repaired", "regressed"]
    values = [paired["repaired_count"], paired["regressed_count"]]
    axes[1].bar(categories, values, color=["#2a9d8f", "#c44536"], width=0.62)
    axes[1].set_ylabel("paired seeds")
    axes[1].set_title(f"Net change: {paired['net_change']:+d} seeds")
    for index, value in enumerate(values):
        axes[1].text(index, value + 0.25, str(value), ha="center", fontweight="bold")
    fig.suptitle("Ablation: remove derived joint-sum output channels", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "output8_ablation.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / "output8_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def main(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_payload = json.loads(args.baseline_summary.read_text())
    baseline_rows = baseline_payload["results"]
    baseline_by_seed = {int(row["seed"]): row for row in baseline_rows}
    seeds = list(range(args.seed_start, args.seed_end + 1))
    missing_baseline = sorted(set(seeds) - set(baseline_by_seed))
    if missing_baseline:
        raise ValueError(f"Baseline summary is missing requested seeds: {missing_baseline}")

    H10 = np.load(args.data_path)
    H8 = remove_joint_sum_channels(H10)
    reduced_data_path = args.output_dir / "offline_hankel_output8.npy"
    np.save(reduced_data_path, H8)
    use_native_eight_output()

    config_keys = {field.name for field in fields(routed.RoutedDeePCConfig)}
    config_payload = {
        key: value for key, value in baseline_payload["config"].items() if key in config_keys
    }
    config = routed.RoutedDeePCConfig(**config_payload)
    atlas = routed.fit_behavioral_atlas(H8, config)
    atlas_path = args.output_dir / "behavioral_atlas_output8.npz"
    routed.save_atlas(atlas_path, atlas, config)
    audit = audit_iid_library(H8, eval_seeds=seeds)
    (args.output_dir / "fairness_audit.json").write_text(json.dumps(audit, indent=2))

    completed = {}
    for seed in seeds:
        path = args.output_dir / f"seed{seed}_control_results.npz"
        if path.exists() and not args.overwrite:
            with np.load(path, allow_pickle=True) as data:
                row = json.loads(str(data["summary"].item()))
            if int(row.get("output_dimension", -1)) == 8:
                completed[seed] = row

    pending = [seed for seed in seeds if seed not in completed]
    if pending:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize_worker,
            initargs=(str(reduced_data_path), str(atlas_path), asdict(config), str(args.output_dir)),
        ) as pool:
            futures = {pool.submit(run_seed, seed): seed for seed in pending}
            for index, future in enumerate(as_completed(futures), 1):
                seed = futures[future]
                row = future.result()
                completed[seed] = row
                print(
                    f"[{index}/{len(pending)}] seed {seed}: "
                    f"{1000 * row['mean_last_30']:.3f} mm, strict={strict_success(row)}",
                    flush=True,
                )

    reduced_rows = [completed[seed] for seed in seeds]
    baseline_selected = [baseline_by_seed[seed] for seed in seeds]
    baseline_success = {seed: strict_success(baseline_by_seed[seed]) for seed in seeds}
    reduced_success = {seed: strict_success(completed[seed]) for seed in seeds}
    repaired = [seed for seed in seeds if not baseline_success[seed] and reduced_success[seed]]
    regressed = [seed for seed in seeds if baseline_success[seed] and not reduced_success[seed]]
    baseline_count = sum(baseline_success.values())
    reduced_count = sum(reduced_success.values())
    paired = {
        "repaired_count": len(repaired),
        "regressed_count": len(regressed),
        "net_change": reduced_count - baseline_count,
        "repaired_seeds": repaired,
        "regressed_seeds": regressed,
        "exact_mcnemar_p": exact_mcnemar_p(len(regressed), len(repaired)),
    }
    payload = {
        "method": "eight_native_output_graph_routed_atlas_ablation",
        "invariant_components": (
            "offline episodes, input trajectories, 48 charts, clustering algorithm, "
            "routing, goal policy, DeePC formulation, rank, horizons, costs, and seeds"
        ),
        "changed_component": (
            "remove cos(q1+q2), sin(q1+q2) from each offline/online output block; "
            "refit clustering and chart bases"
        ),
        "seeds": seeds,
        "config": asdict(config),
        "fairness_audit": audit,
        "chart_sizes": np.bincount(atlas["labels"].astype(int), minlength=config.clusters).tolist(),
        "baseline": {
            "output_dimension": 10,
            "strict_success_count": baseline_count,
            "strict_success_rate": baseline_count / len(seeds),
            "median_final_30_mm": float(1000 * np.median([row["mean_last_30"] for row in baseline_selected])),
        },
        "reduced_output": {
            "output_dimension": 8,
            "strict_success_count": reduced_count,
            "strict_success_rate": reduced_count / len(seeds),
            "median_final_30_mm": float(1000 * np.median([row["mean_last_30"] for row in reduced_rows])),
            "results": reduced_rows,
        },
        "paired_comparison": paired,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    make_figure(payload, args.output_dir)
    print(json.dumps({"baseline": payload["baseline"], "reduced_output": {k: v for k, v in payload["reduced_output"].items() if k != "results"}, "paired": paired}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=routed.DEFAULT_DATA_PATH)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed-start", type=int, default=211)
    parser.add_argument("--seed-end", type=int, default=310)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
