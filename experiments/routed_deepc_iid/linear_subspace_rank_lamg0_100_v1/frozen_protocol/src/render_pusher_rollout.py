"""Replay and render an exact saved Pusher Atlas-DeePC rollout."""

import argparse
from pathlib import Path

import pusher_routed_deepc_experiment as pusher
from render_saved_rollout import render_saved_rollout


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_path", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--warmup-zeros", type=int, default=pusher.T_INI)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    error = render_saved_rollout(
        args.result_path,
        args.seed,
        args.output,
        fps=args.fps,
        frame_stride=args.frame_stride,
        warmup_zeros=args.warmup_zeros,
        verification_path=args.verification_output,
        env_id=pusher.PUSHER_ENV_ID,
        action_dim=pusher.M,
        output_from_obs=pusher.pusher_output_from_obs,
    )
    print("saved", args.output, "maximum replay error", error)
