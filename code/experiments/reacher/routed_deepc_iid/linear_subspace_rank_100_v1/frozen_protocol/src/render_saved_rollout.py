import argparse
import json
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

import expert_pipeline_search as pipe


def render_saved_rollout(
    result_path,
    seed,
    output_path,
    fps=20,
    frame_stride=4,
    warmup_zeros=0,
    verification_path=None,
    env_id=pipe.REACHER_ENV_ID,
    action_dim=pipe.M,
    output_from_obs=pipe.reacher_output_from_obs,
):
    result = np.load(result_path, allow_pickle=True)
    actions = np.asarray(result["u"], dtype=float)
    expected_y = np.asarray(result["y"], dtype=float)
    env = gym.make(
        env_id,
        render_mode="rgb_array",
        max_episode_steps=actions.shape[0] + warmup_zeros + 5,
    )
    obs, _ = env.reset(seed=seed)
    for _ in range(warmup_zeros):
        obs, *_ = env.step(np.zeros(action_dim))
    frames = [np.asarray(env.render(), dtype=np.uint8)]
    replay_y = []
    for step, action in enumerate(actions):
        obs, *_ = env.step(action)
        replay_y.append(output_from_obs(obs))
        if (step + 1) % frame_stride == 0 or step + 1 == actions.shape[0]:
            frames.append(np.asarray(env.render(), dtype=np.uint8))
    env.close()
    replay_y = np.asarray(replay_y)
    maximum_error = float(np.max(np.abs(replay_y - expected_y)))
    if maximum_error > 1e-8:
        raise RuntimeError(f"Saved rollout replay mismatch: maximum output error {maximum_error}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
    if verification_path is not None:
        verification = {
            "result_path": str(result_path),
            "video_path": str(output_path),
            "seed": int(seed),
            "actions": int(actions.shape[0]),
            "warmup_zeros": int(warmup_zeros),
            "fps": int(fps),
            "frame_stride": int(frame_stride),
            "maximum_replay_output_error": maximum_error,
            "environment": env_id,
        }
        verification_path.write_text(json.dumps(verification, indent=2))
    return maximum_error


def parse_args():
    parser = argparse.ArgumentParser(description="Replay and render an exact saved Reacher action sequence")
    parser.add_argument("result_path", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--warmup-zeros", type=int, default=0)
    parser.add_argument("--verification-output", type=Path)
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
    )
    print("saved", args.output, "maximum replay error", error)
