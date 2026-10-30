"""Fit the standard and routed offline intent priors used by Pusher DeePC."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from pusher_atlas.config import PusherAtlasConfig
from pusher_atlas.data import load_data
from pusher_atlas.policy import (
    RegionalBehaviorPolicy,
    save_behavior_policy,
    train_behavior_policy,
)


def main():
    parser = argparse.ArgumentParser(
        description="Fit standard and routed offline intent experts for the DeePC QP"
    )
    parser.add_argument(
        "--standard-data",
        default="experiments/pusher_atlas/coverage_300_target/offline_teacher_data.npz",
    )
    parser.add_argument(
        "--routed-data",
        default="experiments/pusher_atlas/coverage_300_routed/offline_teacher_data.npz",
    )
    parser.add_argument(
        "--output",
        default="experiments/pusher_atlas/coverage_300_routed/offline_behavior_policy_regional.joblib",
    )
    args = parser.parse_args()

    cfg = replace(PusherAtlasConfig(), offline_episodes=300)
    standard = train_behavior_policy(load_data(Path(args.standard_data)), cfg)
    routed = train_behavior_policy(load_data(Path(args.routed_data)), cfg)
    policy = RegionalBehaviorPolicy(standard, routed)
    save_behavior_policy(policy, Path(args.output))
    print(Path(args.output))


if __name__ == "__main__":
    main()
