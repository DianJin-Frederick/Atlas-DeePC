"""Plot saved MVC-BDPC development rollouts without running the controller."""
from pathlib import Path
import os
import json

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gs-deepc-failure-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments/reacher/mvc_bdpc_p8_smoke_211_213_t300_v1"
OUT = Path(__file__).resolve().parent
SEEDS = [211, 212, 213]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "svg.fonttype": "none"})
fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.0))
fig.subplots_adjust(left=.07, right=.91, top=.83, bottom=.15,
                    wspace=.22, hspace=.42)
fig.suptitle("BDPC with multi-view clustering: saved Reacher performance", y=.98,
             fontsize=16, fontweight="bold")
fig.text(.5, .939, "p = 8 · three development rollouts · 300 control steps each · no reruns",
         ha="center", color="#555555", fontsize=11)
handles = [Line2D([], [], marker="*", color="#c94437", ls="", markersize=11, label="Target"),
           Line2D([], [], marker="o", markerfacecolor="white", color="#222222", ls="", markersize=6, label="Start (step 1)"),
           Line2D([], [], marker="X", color="#222222", ls="", markersize=7, label="End (step 300)")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .92),
           ncol=3, frameon=False, fontsize=10)
metrics = []
for col, seed in enumerate(SEEDS):
    with np.load(SOURCE / f"seed{seed}_control_results.npz", allow_pickle=False) as z:
        y = z["y"].copy(); target = z["target_xy"].copy()
        error = z["distance"].copy() * 1000
        clipped = z["action_clip"].copy() > 1e-8
        status = z["status"].copy()
        command = z["command_u"].copy()
    assert len(y) == 300
    xy = y[:, 6:8] * 1000
    assert np.allclose(error, np.linalg.norm(xy - target*1000, axis=1))
    steps = np.arange(1, 301)
    q = np.unwrap(np.arctan2(y[:, 2:4], y[:, :2]), axis=0)
    net_turns = (q[-1, 0] - q[0, 0]) / (2*np.pi)
    ax = axes[0, col]
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    lc = LineCollection(segments, cmap="viridis", norm=Normalize(1, 300),
                        linewidth=1.25, alpha=.85)
    lc.set_array(steps[1:]); ax.add_collection(lc)
    ax.scatter(*target*1000, s=145, marker="*", color="#c94437", zorder=6)
    ax.scatter(*xy[0], s=45, marker="o", facecolor="white", edgecolor="#222222", zorder=7)
    ax.scatter(*xy[-1], s=65, marker="X", color="#222222", zorder=7)
    ax.set(xlim=(-225, 225), ylim=(-225, 225), aspect="equal", xlabel="Fingertip x (mm)")
    ax.set_title(f"Seed {seed}", fontweight="bold", pad=9)
    if col == 0:
        ax.set_ylabel("Fingertip y (mm)")
    ax.set_xticks([-200, 0, 200]); ax.set_yticks([-200, 0, 200]); ax.grid(alpha=.12)
    ax.text(.03, .03, f"Shoulder: {net_turns:.1f} net revolutions", transform=ax.transAxes,
            fontsize=9, bbox={"facecolor":"white", "alpha":.9, "edgecolor":"none", "pad":2})
    ax = axes[1, col]
    ax.axvspan(200.5, 300, color="#c8d9e7", alpha=.38, lw=0)
    ax.plot(steps, error, color="#246c91", lw=1.35)
    ax.axhline(10, color="#777777", ls="--", lw=1)
    # A separate rug above the curves marks steps at which a command was clipped.
    ax.plot(steps[clipped], np.full(int(clipped.sum()), 415), "|", color="#c94437",
            markersize=5, markeredgewidth=.7)
    ax.set(xlim=(1, 300), ylim=(0, 430), xlabel="Control step")
    ax.set_yticks([0, 100, 200, 300, 400]); ax.set_xticks([1, 100, 200, 300])
    if col == 0:
        ax.set_ylabel("Target distance (mm)")
    ax.set_title(f"Last-100 mean: {error[-100:].mean():.1f} mm\n"
                 f"Clipped inputs: {int(clipped.sum())}/300 steps", fontsize=10, loc="left")
    ax.grid(alpha=.15)
    metrics.append({"seed":seed, "steps":300, "mean_last100_mm":float(error[-100:].mean()),
                    "minimum_mm":float(error.min()), "minimum_step_one_based":int(error.argmin()+1),
                    "final_mm":float(error[-1]), "clipped_steps":int(clipped.sum()),
                    "shoulder_net_revolutions":float(net_turns), "statuses":np.unique(status).tolist(),
                    "max_absolute_command":float(np.abs(command).max())})

cax = fig.add_axes([.93, .535, .012, .29])
fig.colorbar(lc, cax=cax, label="Control step", ticks=[1, 100, 200, 300])
fig.text(.07, .085, "Blue shading: last 100 steps. Dashed line: 10 mm threshold. Red ticks: actuator clipping (command exceeds ±1).",
         fontsize=9, color="#555555")
fig.text(.07, .054, "0/3 rollouts satisfy the last-100-step 10 mm criterion; all 900 solves returned optimal. Development examples, not a 500-seed evaluation.",
         fontsize=9, color="#555555")
fig.text(.07, .023, "The plots describe observed behavior; they do not establish a unique cause of failure.", fontsize=9, color="#555555")
fig.savefig(OUT / "bdpc_p8_performance_seeds211_213.png", dpi=190, facecolor="white")
fig.savefig(OUT / "bdpc_p8_performance_seeds211_213.svg", facecolor="white")
print(json.dumps(metrics, indent=2))
