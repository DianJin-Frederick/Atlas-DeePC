"""Create the paper version of the physical-chart cloud visualization."""

from __future__ import annotations

import os
import argparse
import json
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_paper_mpl")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "select_iid_offline_hankel.npy"
ATLAS_PATH = (
    ROOT
    / "experiments"
    / "routed_deepc_iid"
    / "clustering_jointconfig48_fresh211_310"
    / "behavioral_atlas.npz"
)
OUTPUT_DIR = ROOT / "paper" / "figures"

CHART_SELECTION_SEED = 20260827
CLOUD_SAMPLE_SEED = 17
CHART_COUNT = 15
GRID_ROWS = 3
GRID_COLUMNS = 5
CLOUD_MEMBERS_PER_CHART = 32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color", default="#3f6f9f")
    parser.add_argument("--output-stem", default="physical_chart_clouds_15")
    parser.add_argument("--complete-trajectories", action="store_true",
                        help="Show all 2 past and 10 future output poses per sampled column.")
    parser.add_argument("--trajectory-count", type=int, default=8,
                        help="Complete trajectories shown per chart (subset of original 32 samples).")
    args = parser.parse_args()
    if not 1 <= args.trajectory_count <= CLOUD_MEMBERS_PER_CHART:
        parser.error("--trajectory-count must be between 1 and 32")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 8,
        }
    )

    H = np.load(DATA_PATH, mmap_mode="r")
    with np.load(ATLAS_PATH, allow_pickle=False) as archive:
        atlas = {key: archive[key] for key in ("labels", "medoid_states", "centers")}
    labels = np.asarray(atlas["labels"], dtype=int)
    medoid_states = np.asarray(atlas["medoid_states"], dtype=float)
    total_charts = int(np.asarray(atlas["centers"]).shape[0])
    if total_charts != 48:
        raise ValueError(f"Expected 48 physical charts, found {total_charts}")

    selection_rng = np.random.default_rng(CHART_SELECTION_SEED)
    selected_chart_ids = np.sort(
        selection_rng.choice(total_charts, size=CHART_COUNT, replace=False)
    )
    cloud_rng = np.random.default_rng(CLOUD_SAMPLE_SEED)

    chart_sizes = np.bincount(labels, minlength=total_charts)
    # Baseline Reacher library: m=2, p=10, T_ini=2, T_f=10.
    # The saved matrix stores all input rows before all output rows.
    assert H.shape == (144, 37800)
    assert labels.shape == (H.shape[1],)
    trajectory_output_states = H[2 * 12 :].reshape(
        12,
        10,
        H.shape[1],
    ).transpose(2, 0, 1)
    chart_output_states = trajectory_output_states[:, 1, :]

    link_1 = 0.10
    medoid_elbows = link_1 * np.column_stack(
        [medoid_states[:, 0], medoid_states[:, 2]]
    )
    medoid_fingertips = medoid_states[:, 6:8]

    fig, axes = plt.subplots(
        GRID_ROWS,
        GRID_COLUMNS,
        figsize=(6.3, 3.9),
        constrained_layout=True,
    )
    # Paul Tol medium-contrast blue pair; preserve the paper's orange exactly.
    past_color, future_color = "#6699CC", "#DF7B20"
    representative_color = "#004488"
    sampled_columns = {}
    if args.complete_trajectories:
        fig.legend(handles=[
            Line2D([0], [0], color=past_color, lw=1.8, label="Past poses ($T_{\\mathrm{ini}}=2$)"),
            Line2D([0], [0], color=future_color, lw=1.8, label="Future poses ($T_{\\mathrm{f}}=10$)"),
            Line2D([0], [0], color=representative_color, lw=2, marker="o",
                   markersize=3.5, markerfacecolor="white", label="Nearest-centroid pose"),
        ], loc="outside upper center", ncol=3, frameon=False, fontsize=8,
            handlelength=1.8, columnspacing=1.4)
    for chart_id, ax in zip(selected_chart_ids, axes.flat, strict=True):
        color = args.color
        member_indices = np.flatnonzero(labels == chart_id)
        sampled_indices = cloud_rng.choice(
            member_indices,
            size=min(CLOUD_MEMBERS_PER_CHART, member_indices.size),
            replace=False,
        )
        if args.complete_trajectories:
            sampled_indices = sampled_indices[:args.trajectory_count]
        sampled_columns[str(chart_id + 1)] = sampled_indices.tolist()
        if args.complete_trajectories:
            # Display measured output poses from each complete Hankel column.
            # Future first, past above it: ten future poses must not hide two past poses.
            for time_slice, pose_color, opacity, layer in (
                (slice(2, 12), future_color, 0.22, 1),
                (slice(0, 2), past_color, 0.40, 2),
            ):
                states = trajectory_output_states[sampled_indices, time_slice, :].reshape(-1, 10)
                elbows = link_1 * states[:, [0, 2]]
                tips = states[:, 6:8]
                assert np.isfinite(states).all()
                assert np.max(np.abs(tips)) < .225
                segments = np.stack([np.zeros_like(elbows), elbows, tips], axis=1)
                ax.add_collection(LineCollection(segments, colors=[pose_color],
                    linewidths=0.60, alpha=opacity, zorder=layer))
                ax.scatter(tips[:, 0], tips[:, 1], s=2.3, color=pose_color,
                    alpha=min(opacity * 1.5, 1), linewidth=0, zorder=layer)
            # A feature centroid need not be a physically realizable arm pose.
            # Show the stored nearest member's last-past pose instead.
            elbow = medoid_elbows[chart_id]
            tip = medoid_fingertips[chart_id]
            # Narrow white underlay separates the representative from crossing poses.
            ax.plot([0, elbow[0], tip[0]], [0, elbow[1], tip[1]],
                    color="white", linewidth=3.0,
                    solid_capstyle="round", zorder=3)
            ax.plot([0, elbow[0], tip[0]], [0, elbow[1], tip[1]],
                    color=representative_color, linewidth=2.0,
                    solid_capstyle="round", zorder=4)
            ax.scatter([elbow[0], tip[0]], [elbow[1], tip[1]], s=[10, 18],
                       facecolor="white", edgecolor=representative_color,
                       linewidth=.8, zorder=5)
            ax.scatter([0], [0], s=12, color="#222222", zorder=4)
        else:
            draw_single_pose_cloud(ax, chart_output_states[sampled_indices],
                medoid_elbows[chart_id], medoid_fingertips[chart_id], color, link_1)

        ax.axhline(0.0, color="#e2e2e2", linewidth=0.40, zorder=0)
        ax.axvline(0.0, color="#e2e2e2", linewidth=0.40, zorder=0)
        ax.set_xlim(-0.225, 0.225)
        ax.set_ylim(-0.225, 0.225)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"Chart {chart_id + 1} | $n={chart_sizes[chart_id]}$",
            color="#111111",
            pad=2.0,
        )
        for spine in ax.spines.values():
            spine.set_color("#999999" if args.complete_trajectories else "#444444")
            spine.set_linewidth(0.55 if args.complete_trajectories else 0.75)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = OUTPUT_DIR / args.output_stem
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    if args.complete_trajectories:
        output_stem.with_suffix(".json").write_text(json.dumps({
            "data": str(DATA_PATH), "atlas": str(ATLAS_PATH),
            "past_time_indices": [0, 1], "future_time_indices": list(range(2, 12)),
            "sampled_columns_zero_based": sampled_columns,
            "chart_sizes": {str(k+1): int(chart_sizes[k]) for k in selected_chart_ids},
            "past_color": past_color, "future_color": future_color,
            "past_pose_alpha": .40, "future_pose_alpha": .22,
            "representative_color": representative_color,
            "palette_source": "Paul Tol medium-contrast blue pair; original paper orange retained",
            "displayed_trajectories_per_chart": args.trajectory_count,
            "representative": "Last-past pose of the recorded chart member nearest the standardized clustering centroid",
            "representative_states": {str(k+1): medoid_states[k].tolist() for k in selected_chart_ids},
        }, indent=2) + "\n")

    selected_one_based = selected_chart_ids + 1
    print("Selected charts (one-based):", ", ".join(map(str, selected_one_based)))
    print(output_stem.with_suffix(".pdf"))
    print(output_stem.with_suffix(".png"))


def draw_single_pose_cloud(ax, cloud_states, elbow, fingertip, color, link_1):
        """Preserve the original last-past-pose visualization."""
        cloud_elbows = link_1 * np.column_stack(
            [cloud_states[:, 0], cloud_states[:, 2]]
        )
        cloud_fingertips = cloud_states[:, 6:8]
        cloud_segments = np.stack(
            [np.zeros_like(cloud_elbows), cloud_elbows, cloud_fingertips], axis=1
        )

        ax.add_collection(
            LineCollection(
                cloud_segments,
                colors=[color],
                linewidths=0.85,
                alpha=0.22,
                zorder=1,
            )
        )
        ax.scatter(
            cloud_fingertips[:, 0],
            cloud_fingertips[:, 1],
            s=6,
            color=color,
            alpha=0.30,
            linewidth=0,
            zorder=1,
        )

        ax.plot(
            [0.0, elbow[0], fingertip[0]],
            [0.0, elbow[1], fingertip[1]],
            color=color,
            linewidth=2.7,
            solid_capstyle="round",
            zorder=3,
        )
        ax.scatter([0.0], [0.0], s=14, color="#111111", zorder=3)
        ax.scatter([elbow[0]], [elbow[1]], s=17, color="#444444", zorder=3)
        ax.scatter(
            [fingertip[0]],
            [fingertip[1]],
            s=24,
            color=color,
            edgecolor="#111111",
            linewidth=0.45,
            zorder=4,
        )



if __name__ == "__main__":
    main()
