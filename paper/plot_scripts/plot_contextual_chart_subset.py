"""Create a compact paper visualization of selected contextual charts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_paper_mpl")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import contextual_cluster_atlas_experiment as contextual
import routed_deepc_experiment as routed


DATA_PATH = ROOT / "data" / "select_iid_offline_hankel.npy"
CONTEXTUAL_DIR = (
    ROOT
    / "experiments"
    / "routed_deepc_iid"
    / "contextual_cluster_eval211_710_global128_triggered"
)
ATLAS_PATH = CONTEXTUAL_DIR / "contextual_cluster_atlas.npz"
SUMMARY_PATH = CONTEXTUAL_DIR / "summary.json"
OUTPUT_DIR = ROOT / "paper" / "figures"

CHART_SELECTION_SEED = 20260827
CLOUD_SAMPLE_SEED = 29
CHART_COUNT = 15
GRID_ROWS = 3
GRID_COLUMNS = 5
CLOUD_MEMBERS_PER_CHART = 18

CURRENT_COLOR = "#3f6f9f"
FUTURE_COLOR = "#b66a3c"


def arm_segments(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    elbows = 0.10 * np.column_stack([states[:, 0], states[:, 2]])
    fingertips = states[:, 6:8]
    return np.stack([np.zeros_like(elbows), elbows, fingertips], axis=1)


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 8,
            "legend.fontsize": 7.5,
        }
    )

    summary = json.loads(SUMMARY_PATH.read_text())
    config = routed.RoutedDeePCConfig(**summary["config"])
    H = np.load(DATA_PATH, mmap_mode="r")
    artifact = contextual.load_contextual_cluster_atlas(ATLAS_PATH)
    supports = np.asarray(artifact["support_indices"], dtype=int)
    assignment_counts = np.asarray(artifact["assignment_counts"], dtype=int)
    total_charts = supports.shape[0]
    if total_charts != 128:
        raise ValueError(f"Expected 128 contextual charts, found {total_charts}")

    library = contextual.prepare_contextual_library(H)
    embedding = contextual._contextual_embedding(
        library,
        contextual.output_cost(config),
    )
    output_windows = H[routed.pipe.M * routed.pipe.L :].reshape(
        routed.pipe.L,
        routed.pipe.P,
        H.shape[1],
    )
    current_states = output_windows[routed.pipe.T_INI - 1].T
    future_states = output_windows[-1].T

    selection_rng = np.random.default_rng(CHART_SELECTION_SEED)
    selected_chart_ids = np.sort(
        selection_rng.choice(total_charts, size=CHART_COUNT, replace=False)
    )
    cloud_rng = np.random.default_rng(CLOUD_SAMPLE_SEED)

    fig, axes = plt.subplots(
        GRID_ROWS,
        GRID_COLUMNS,
        figsize=(6.3, 4.15),
        constrained_layout=True,
    )
    for chart_id, ax in zip(selected_chart_ids, axes.flat, strict=True):
        chart_indices = supports[chart_id]
        sampled_indices = cloud_rng.choice(
            chart_indices,
            size=min(CLOUD_MEMBERS_PER_CHART, chart_indices.size),
            replace=False,
        )
        contextual_center = embedding[chart_indices].mean(axis=0)
        representative = chart_indices[
            np.argmin(
                np.linalg.norm(
                    embedding[chart_indices] - contextual_center,
                    axis=1,
                )
            )
        ]

        ax.add_collection(
            LineCollection(
                arm_segments(current_states[sampled_indices]),
                colors=CURRENT_COLOR,
                linewidths=0.75,
                alpha=0.14,
                zorder=1,
            )
        )
        ax.add_collection(
            LineCollection(
                arm_segments(future_states[sampled_indices]),
                colors=FUTURE_COLOR,
                linewidths=0.75,
                alpha=0.12,
                zorder=1,
            )
        )

        current_arm = arm_segments(current_states[[representative]])[0]
        future_arm = arm_segments(future_states[[representative]])[0]
        ax.plot(
            current_arm[:, 0],
            current_arm[:, 1],
            "-o",
            color=CURRENT_COLOR,
            linewidth=2.25,
            markersize=3.0,
            markeredgecolor="#222222",
            markeredgewidth=0.4,
            zorder=3,
        )
        ax.plot(
            future_arm[:, 0],
            future_arm[:, 1],
            "-o",
            color=FUTURE_COLOR,
            linewidth=2.25,
            markersize=3.0,
            markeredgecolor="#222222",
            markeredgewidth=0.4,
            zorder=3,
        )
        ax.annotate(
            "",
            xy=future_arm[-1],
            xytext=current_arm[-1],
            arrowprops={
                "arrowstyle": "->",
                "color": "#555555",
                "linewidth": 0.8,
                "alpha": 0.85,
            },
            zorder=2,
        )

        ax.axhline(0.0, color="#d0d0d0", linewidth=0.4, zorder=0)
        ax.axvline(0.0, color="#d0d0d0", linewidth=0.4, zorder=0)
        ax.set_xlim(-0.225, 0.225)
        ax.set_ylim(-0.225, 0.225)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"Chart {chart_id + 1} | $n={assignment_counts[chart_id]}$",
            color="#111111",
            pad=2.0,
        )
        for spine in ax.spines.values():
            spine.set_color("#444444")
            spine.set_linewidth(0.75)

    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=CURRENT_COLOR,
                linewidth=2.2,
                label=r"current posture (end of $Y_{\rm p}$)",
            ),
            Line2D(
                [0],
                [0],
                color=FUTURE_COLOR,
                linewidth=2.2,
                label=r"future posture (end of $Y_{\rm f}$)",
            ),
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
        handlelength=2.4,
        columnspacing=2.0,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = OUTPUT_DIR / "contextual_chart_clouds_15"
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    selected_one_based = selected_chart_ids + 1
    print("Selected charts (one-based):", ", ".join(map(str, selected_one_based)))
    print(output_stem.with_suffix(".pdf"))
    print(output_stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
