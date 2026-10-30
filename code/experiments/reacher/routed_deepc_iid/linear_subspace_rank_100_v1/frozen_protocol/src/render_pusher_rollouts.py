"""Render deterministic videos from saved Pusher DeePC actions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pusher_atlas.video import render_rollout_video


def main():
    parser = argparse.ArgumentParser(description="Render saved Pusher QP rollouts")
    parser.add_argument(
        "--artifact-dir",
        default="experiments/pusher_atlas/coverage_300_routed",
    )
    parser.add_argument(
        "--seeds", default=",".join(map(str, range(101, 117)))
    )
    args = parser.parse_args()
    artifact_dir = Path(args.artifact_dir)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    for seed in seeds:
        with np.load(
            artifact_dir / f"rollout_seed{seed}.npz", allow_pickle=True
        ) as payload:
            result = {key: payload[key] for key in payload.files}
        output = render_rollout_video(
            result, artifact_dir / f"rollout_seed{seed}.mp4"
        )
        print(output, flush=True)


if __name__ == "__main__":
    main()
