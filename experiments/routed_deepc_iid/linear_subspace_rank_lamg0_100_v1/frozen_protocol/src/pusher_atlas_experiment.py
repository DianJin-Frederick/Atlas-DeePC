"""Run the coverage-validated Atlas-DeePC controller on Pusher-v5."""

from __future__ import annotations

import argparse
import json

from pusher_atlas import PusherAtlasConfig, run_experiment


def main():
    parser = argparse.ArgumentParser(description="Contact-aware data-driven QP for Pusher-v5")
    parser.add_argument("--smoke", action="store_true", help="Run the small notebook-sized experiment")
    parser.add_argument("--no-reuse-data", action="store_true")
    parser.add_argument("--output-dir", default="experiments/pusher_atlas/default")
    args = parser.parse_args()
    cfg = PusherAtlasConfig.smoke() if args.smoke else PusherAtlasConfig()
    run = run_experiment(cfg, args.output_dir, reuse_data=not args.no_reuse_data)
    print(json.dumps(run["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
