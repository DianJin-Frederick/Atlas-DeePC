"""Diagnostic plots for Pusher datasets and rollouts."""

from __future__ import annotations

import os
from pathlib import Path

_CACHE_ROOT = Path(__file__).resolve().parents[2] / ".cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from .environment import OBJECT_XY, PHASE_NAMES, TIP


def plot_rollout(result: dict, output_path: Path | None = None):
    states = np.asarray(result["state"], dtype=float)
    goal = np.asarray(result["goal_xy"], dtype=float)
    time = np.arange(len(states)) * float(result["dt"])
    distance = np.asarray(result["distance"], dtype=float)
    phase = np.asarray(result["phase"], dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(states[:, OBJECT_XY.start], states[:, OBJECT_XY.start + 1], "o-", ms=2, label="object")
    axes[0, 0].plot(states[:, TIP.start], states[:, TIP.start + 1], alpha=0.7, label="tip")
    axes[0, 0].scatter(*goal, marker="*", s=180, color="red", label="goal")
    axes[0, 0].set(xlabel="x [m]", ylabel="y [m]", title="Task-space trajectory")
    axes[0, 0].axis("equal")
    axes[0, 0].legend()

    axes[0, 1].plot(time, distance, lw=2)
    axes[0, 1].axhline(float(result["success_radius"]), color="C3", ls="--", label="success radius")
    axes[0, 1].set(xlabel="time [s]", ylabel="object-goal XY [m]", title="Goal distance")
    axes[0, 1].legend()

    axes[1, 0].step(time, phase, where="post")
    axes[1, 0].set_yticks(np.arange(len(PHASE_NAMES)), PHASE_NAMES)
    axes[1, 0].set(xlabel="time [s]", ylabel="phase", title="Geometric task phase")

    axes[1, 1].plot(time, np.asarray(result["action"], dtype=float))
    axes[1, 1].set(xlabel="time [s]", ylabel="torque", title="QP actions")
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def plot_dataset_summary(data: dict, output_path: Path | None = None):
    summary = np.asarray(data["episode_summary"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(summary[:, 0], summary[:, 1])
    limit = max(summary[:, 0].max(), summary[:, 1].max())
    axes[0].plot([0, limit], [0, limit], "k--", lw=1)
    axes[0].set(xlabel="initial distance [m]", ylabel="minimum distance [m]", title="Teacher task progress")
    axes[1].hist(summary[:, 2], bins=min(15, len(summary)))
    axes[1].set(xlabel="maximum object displacement [m]", ylabel="episodes", title="Contact-rich coverage")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig
