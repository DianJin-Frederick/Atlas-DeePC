"""Deterministic MuJoCo replay rendering for saved Pusher actions."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from .environment import make_env


def render_rollout_video(
    result: dict,
    output_path: str | Path,
    width: int = 640,
    height: int = 480,
    fps: int = 20,
) -> Path:
    """Replay saved QP actions in Pusher-v5 and encode an RGB MP4."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    actions = np.asarray(result["action"], dtype=float)
    warmup_actions = np.asarray(result.get("warmup_action", []), dtype=float)
    seed = int(np.asarray(result["seed"]))
    env, _ = make_env(
        seed,
        max_episode_steps=len(warmup_actions) + len(actions) + 2,
        render_mode="rgb_array",
    )
    # Gymnasium forwards these renderer dimensions to MuJoCo through make().
    # Recreate if the default renderer size differs from the requested output.
    if width != 480 or height != 480:
        env.close()
        import gymnasium as gym

        env = gym.make(
            "Pusher-v5",
            max_episode_steps=len(warmup_actions) + len(actions) + 2,
            render_mode="rgb_array",
            width=int(width),
            height=int(height),
        )
        env.reset(seed=seed)

    frames = [np.asarray(env.render(), dtype=np.uint8)]
    replay_states = []
    for action in warmup_actions:
        env.step(np.asarray(action, dtype=np.float32))
        frames.append(np.asarray(env.render(), dtype=np.uint8))
    for action in actions:
        observation, *_ = env.step(np.asarray(action, dtype=np.float32))
        replay_states.append(np.asarray(observation, dtype=float))
        frames.append(np.asarray(env.render(), dtype=np.uint8))
    env.close()

    imageio.mimwrite(
        output_path,
        frames,
        fps=int(fps),
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    return output_path
