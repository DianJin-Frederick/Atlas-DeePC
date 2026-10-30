"""Plot one retained historical GS-DeePC rollout; no controller execution."""
from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gs-deepc-failure-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments/reacher/gs_deepc_control_tuning_tf10_v1/paper_seed211"
OUT = Path(__file__).resolve().parent

with np.load(SOURCE / "seed211_control_results.npz", allow_pickle=False) as data:
    y = data["y"].copy()
    target = data["target_xy"].copy()
    distance = data["distance"].copy() * 1000
    fallback = data["fallback"].copy()
    statuses = data["status"].copy()
    warmup = data["warmup_y"][-1].copy()
config = json.loads((SOURCE / "manifest.json").read_text())["config"]
n = len(y)
steps = np.arange(1, n + 1)
closest = int(np.argmin(distance))
q = np.unwrap(np.arctan2(y[:, 2:4], y[:, :2]), axis=0)
turns = (q - q[0]) / (2 * np.pi)
xy = y[:, 6:8] * 1000

# Reacher XML: shoulder-to-elbow 0.10 m; elbow-to-fingertip 0.11 m.
elbows = 100 * np.column_stack([np.cos(q[:, 0]), np.sin(q[:, 0])])
reconstructed = elbows + 110 * np.column_stack([
    np.cos(q[:, 0] + q[:, 1]), np.sin(q[:, 0] + q[:, 1])])
# Saved Cartesian observations and angle-derived kinematics differ slightly
# (under 0.4 mm here); keep measured XY in panel a and reconstructed poses in c.
kinematic_discrepancy_mm = np.max(np.linalg.norm(reconstructed - xy, axis=1))
assert kinematic_discrepancy_mm < .5
assert np.allclose(distance, np.linalg.norm(xy - 1000 * target, axis=1))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "svg.fonttype": "none"})
fig, axs = plt.subplots(2, 2, figsize=(11, 8.3))
fig.subplots_adjust(left=.08, right=.97, bottom=.115, top=.87,
                    wspace=.28, hspace=.42)
fig.suptitle("GS-DeePC failure: approaches the target, then rotates and folds inward",
             fontsize=15, fontweight="bold", y=.98)
fig.text(.5, .935, "Reacher · seed 211 · original configuration · retained 200-step development rollout",
         ha="center", color="#555555", fontsize=10)

ax = axs[0, 0]
segments = np.stack([xy[:-1], xy[1:]], axis=1)
lc = LineCollection(segments, cmap="viridis", norm=Normalize(1, n), linewidth=1.8)
lc.set_array(steps[1:]); ax.add_collection(lc)
ax.scatter(*target * 1000, marker="*", s=140, c="#d44836", zorder=6, label="Target")
ax.scatter(*xy[0], marker="o", facecolor="white", edgecolor="#222222", s=45, zorder=6, label="Step 1")
ax.scatter(*xy[closest], marker="D", color="#e59922", s=40, zorder=6, label=f"Closest: step {closest+1}")
ax.scatter(*xy[-1], marker="X", color="#111111", s=65, zorder=6, label=f"Step {n}")
ax.set(xlim=(-230, 230), ylim=(-230, 230), aspect="equal", xlabel="Fingertip x (mm)", ylabel="Fingertip y (mm)")
ax.set_title("a  Measured fingertip trajectory", loc="left")
ax.legend(loc="lower left", fontsize=8, framealpha=.95)
fig.colorbar(lc, ax=ax, pad=.03, fraction=.045, label="Control step")

ax = axs[0, 1]
ax.plot(steps, distance, color="#246c91", lw=1.8)
ax.axhline(10, color="#888888", ls="--", lw=1, label="10 mm threshold")
ax.scatter([closest+1, n], distance[[closest, -1]], color=["#e59922", "#111111"], zorder=5)
ax.annotate(f"Closest: {distance[closest]:.1f} mm", (closest+1, distance[closest]),
            xytext=(10, 105), textcoords="data", arrowprops={"arrowstyle": "-", "color": "#777777"})
ax.annotate(f"Final: {distance[-1]:.1f} mm", (n, distance[-1]),
            xytext=(120, 285), textcoords="data", arrowprops={"arrowstyle": "-", "color": "#777777"})
ax.set(xlim=(1, n), ylim=(0, 410), xlabel="Control step", ylabel="Target distance (mm)")
ax.set_title("b  Tracking error", loc="left"); ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=.15)

ax = axs[1, 0]
for index, color, style in [(0, "#8e9ba3", "--"), (closest, "#e59922", "-"), (n-1, "#246c91", "-")]:
    pose = np.vstack([np.zeros(2), elbows[index], reconstructed[index]])
    ax.plot(pose[:, 0], pose[:, 1], style, color=color, marker="o", lw=2.5,
            markersize=5, label=f"Step {index+1}")
ax.scatter(*target * 1000, marker="*", s=140, c="#d44836", zorder=6, label="Target")
ax.set(xlim=(-45, 235), ylim=(-50, 160), aspect="equal", xlabel="x (mm)", ylabel="y (mm)")
ax.set_title("c  Arm configurations from measured angles", loc="left")
ax.legend(loc="upper right", fontsize=8)
ax.text(.02, .97, f"Final elbow angle: {np.rad2deg(q[-1,1]):.1f}°\nArm folds back toward its base",
        transform=ax.transAxes, va="top", fontsize=9, color="#246c91")
ax.grid(alpha=.15)

ax = axs[1, 1]
ax.plot(steps, turns[:, 0], color="#246c91", lw=1.8, label="Shoulder joint")
ax.plot(steps, turns[:, 1], color="#e59922", lw=1.8, label="Elbow joint")
ax.set(xlim=(1, n), ylim=(min(-.15, float(turns.min())-.08), 2.5), xlabel="Control step", ylabel="Unwrapped angle change (revolutions)")
ax.set_title("d  Rotation relative to step 1", loc="left")
ax.legend(loc="upper left", fontsize=8)
ax.text(.97, .13, f"Shoulder: {turns[-1,0]:.2f} net revolutions", transform=ax.transAxes,
        ha="right", color="#246c91", fontsize=10)
ax.grid(alpha=.15)

fig.text(.08, .045, f"T_ini={config['tini']}, T_f={config['horizon']}, Q_xy={config['q_xy']:g}, R={config['r']:g}, "
         f"λ_g={config['lambda_g']:g}, λ_ini={config['lambda_ini']:.0e}; q₂ scheduler, 8 regions.  "
         f"Solver fallback steps: {int(fallback.sum())}/{n}.", fontsize=9, color="#555555")
fig.text(.08, .02, "Historical example—not a 300-step test-panel result. Motion illustrates failure behavior, not proof of its unique cause.",
         fontsize=9, color="#555555")
fig.savefig(OUT / "gs_deepc_failure_seed211.png", dpi=190, facecolor="white")
fig.savefig(OUT / "gs_deepc_failure_seed211.svg", facecolor="white")
print(json.dumps({"source": str(SOURCE), "steps": n, "closest_step_one_based": closest+1,
    "minimum_mm": float(distance[closest]), "final_mm": float(distance[-1]),
    "shoulder_net_revolutions": float(turns[-1,0]), "fallback_steps": int(fallback.sum()),
    "maximum_pose_vs_saved_xy_discrepancy_mm": float(kinematic_discrepancy_mm),
    "solver_statuses": np.unique(statuses).tolist()}, indent=2))
