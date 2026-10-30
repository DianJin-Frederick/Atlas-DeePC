"""Run a rank-only contextual-cluster Atlas sweep with frozen supports."""

import argparse
import json
from pathlib import Path

import contextual_cluster_atlas_experiment as experiment


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "experiments/routed_deepc_iid"
DATA_PATH = ROOT / "data/select_iid_offline_hankel.npy"


def parse_ranks(value: str) -> list[int]:
    ranks = sorted({int(item) for item in value.split(",")})
    if not ranks or ranks[0] < 1:
        raise argparse.ArgumentTypeError("Ranks must be positive integers")
    return ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=parse_ranks, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-stop", type=int, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = experiment.ContextualClusterConfig(
        architecture="global_vq",
        contextual_clusters=128,
        support_columns=200,
        activation_mode="triggered",
        error_threshold=0.01,
        stall_window=15,
        minimum_progress=0.001,
    )
    seeds = range(args.seed_start, args.seed_stop + 1)
    outcomes = []
    for rank in args.ranks:
        output_dir = RESULT_ROOT / f"{args.output_prefix}{rank}"
        payload = experiment.run_panel(
            seeds,
            args.baseline_dir,
            output_dir,
            config,
            data_path=DATA_PATH,
            workers=args.workers,
            overwrite=args.overwrite,
            control_overrides={"svd_rank": rank},
            source_artifact_path=args.source_artifact,
        )
        successes = sum(
            row["success_30"] and row["success_100"]
            for row in payload["results"]
        )
        outcome = {
            "rank": rank,
            "successes": successes,
            "trials": len(payload["results"]),
            "output_dir": str(output_dir),
        }
        outcomes.append(outcome)
        print(json.dumps(outcome), flush=True)
    print(json.dumps({"rank_sweep": outcomes}, indent=2))


if __name__ == "__main__":
    main()
