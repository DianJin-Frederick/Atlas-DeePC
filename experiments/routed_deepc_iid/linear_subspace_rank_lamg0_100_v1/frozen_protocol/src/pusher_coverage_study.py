"""Collect, audit, and evaluate broad-coverage Pusher Atlas-DeePC data."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from pusher_atlas import PusherAtlasConfig, render_rollout_video, run_experiment
from pusher_atlas.data import collect_teacher_data, load_data


def parse_seeds(value: str):
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Collect, validate, and evaluate broad-coverage Pusher Atlas-DeePC data"
    )
    parser.add_argument(
        "--output-dir", default="experiments/pusher_atlas/coverage_study"
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--probe-episodes", type=int, default=24)
    parser.add_argument(
        "--test-seeds",
        default=",".join(map(str, range(101, 117))),
    )
    parser.add_argument(
        "--orientation-turn-threshold",
        type=float,
        default=PusherAtlasConfig().collection_orientation_turn_threshold,
        help="Use -2 to disable reset-routed alternate tool orientation",
    )
    parser.add_argument("--no-reuse-data", action="store_true")
    parser.add_argument("--render-videos", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = replace(
        PusherAtlasConfig(),
        offline_episodes=args.episodes,
        coverage_probe_episodes=args.probe_episodes,
        test_seeds=parse_seeds(args.test_seeds),
        collection_orientation_turn_threshold=args.orientation_turn_threshold,
    )
    probe_cfg = replace(
        cfg,
        seed=cfg.coverage_probe_seed,
        offline_episodes=cfg.coverage_probe_episodes,
        test_seeds=(),
        ik_restarts=min(cfg.ik_restarts, 4),
    )
    probe_path = output_dir / "heldout_coverage_probe_data.npz"
    if probe_path.exists() and not args.no_reuse_data:
        probe_data = load_data(probe_path)
    else:
        probe_data = collect_teacher_data(probe_cfg, probe_path)

    run = run_experiment(
        cfg,
        output_dir=output_dir,
        reuse_data=not args.no_reuse_data,
        probe_data=probe_data,
    )
    if args.render_videos:
        for result in run["results"]:
            seed = int(result["seed"])
            render_rollout_video(result, output_dir / f"rollout_seed{seed}.mp4")
    print(
        json.dumps(
            {
                "coverage": run["summary"].get("coverage_validation"),
                "evaluation": {
                    key: value
                    for key, value in run["summary"].items()
                    if key
                    in {
                        "success_rate",
                        "goal_reach_rate",
                        "reach_10cm_rate",
                        "mean_minimum_distance",
                        "mean_final_distance",
                        "mean_solve_rate",
                        "median_solve_ms",
                    }
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
