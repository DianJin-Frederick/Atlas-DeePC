"""Shared Reacher data decoding and atomic result-file utilities."""
import hashlib
import json
from pathlib import Path
import numpy as np


def timing_stats(seconds):
    ms = 1000 * np.asarray(seconds)
    return {"mean_ms": float(ms.mean()), "median_ms": float(np.median(ms)),
            "p95_ms": float(np.percentile(ms, 95)), "p99_ms": float(np.percentile(ms, 99)),
            "max_ms": float(ms.max())}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)

def output_from_obs(obs):
    # Post-action output ordering agrees with the Atlas offline library.
    obs = np.asarray(obs)
    return np.r_[obs[:4], obs[6:8], obs[4:6] + obs[8:10]]

def recover_episodes(hankel, episode_length=200):
    """Recover unique samples and verify every overlap; never cross episodes.

    Source protocol: 200-step episodes, 12-step windows, 2 inputs/10 outputs.
    Each recorded y is measured AFTER the input in the same sample row.
    """
    h = np.asarray(hankel)
    width = episode_length - 12 + 1
    if h.ndim != 2 or h.shape[0] != 144 or h.shape[1] % width:
        raise ValueError("Expected the Atlas 12-step, 2-input/10-output episodic library")
    if not np.isfinite(h).all():
        raise ValueError("Nonfinite offline data")
    us, ys = [], []
    for start in range(0, h.shape[1], width):
        block = h[:, start:start + width]
        u, y = block[:24].reshape(12, 2, width), block[24:].reshape(12, 10, width)
        for values in (u, y):
            if not np.allclose(values[1:, :, :-1], values[:-1, :, 1:], rtol=0, atol=1e-12):
                raise ValueError("Offline episode overlaps do not match the declared protocol")
        us.append(np.concatenate([u[:, :, 0], u[-1, :, 1:].T]))
        ys.append(np.concatenate([y[:, :8, 0], y[-1, :8, 1:].T]))
    return np.stack(us), np.stack(ys)

def windows(episodes, length):
    return np.concatenate([
        np.stack([ep[t:t + length].ravel() for t in range(len(ep) - length + 1)], axis=1)
        for ep in episodes], axis=1)
