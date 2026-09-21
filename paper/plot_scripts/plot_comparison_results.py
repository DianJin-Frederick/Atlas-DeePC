"""Create the paper comparison of success and online computation."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_paper_mpl")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMPARISON = (
    ROOT
    / "experiments"
    / "deepc_algorithm_comparison"
    / "six_way_svd_heldout111_1210_n200_rank60"
    / "summary.json"
)
CONTEXTUAL_ATLAS_FIRST = (
    ROOT
    / "experiments"
    / "routed_deepc_iid"
    / "contextual_cluster_eval211_710_global128_triggered"
    / "summary.json"
)
CONTEXTUAL_ATLAS_EXTENSION = (
    ROOT
    / "experiments"
    / "routed_deepc_iid"
    / "contextual_cluster_eval711_1210_global128_triggered"
    / "summary.json"
)
OUTPUT_DIR = ROOT / "paper" / "figures"

SEEDS = list(range(211, 1211))
METHODS = [
    "Select-DPC",
    "Atlas-DeePC",
    "Contextual DPC",
    "Contextual Atlas-DeePC",
]
METHOD_TICK_LABELS = [
    "Select-DPC",
    "Atlas-DeePC",
    "Contextual DPC",
    "Contextual Atlas-\nDeePC",
]
SOURCE_KEYS = {
    "Select-DPC": "select_dpc_svd",
    "Atlas-DeePC": "atlas",
    "Contextual DPC": "contextual_final_svd",
}
MODULES = [
    "Localization",
    "Graph search",
    "Data/prototype selection",
    "Matrix/SVD preparation",
    "QP solve",
    "Other",
]
MODULE_COLORS = {
    "Localization": "#9db9cf",
    "Graph search": "#6f91ad",
    "Data/prototype selection": "#c8954d",
    "Matrix/SVD preparation": "#dfc27d",
    "QP solve": "#4f806f",
    "Other": "#aaaaaa",
}


def load_results() -> dict[str, list[dict]]:
    source = json.loads(SOURCE_COMPARISON.read_text())
    first = json.loads(CONTEXTUAL_ATLAS_FIRST.read_text())
    extension = json.loads(CONTEXTUAL_ATLAS_EXTENSION.read_text())

    results: dict[str, list[dict]] = {}
    for label, key in SOURCE_KEYS.items():
        by_seed = {
            int(row["seed"]): row for row in source["method_summaries"][key]
        }
        results[label] = [by_seed[seed] for seed in SEEDS]

    contextual_by_seed = {
        int(row["seed"]): row
        for row in first["results"] + extension["results"]
    }
    results["Contextual Atlas-DeePC"] = [
        contextual_by_seed[seed] for seed in SEEDS
    ]
    return results


def strict_success(row: dict) -> bool:
    return bool(row["success_30"] and row["success_100"])


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (rate + z**2 / (2.0 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(rate * (1.0 - rate) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return center - half_width, center + half_width


def stage_timing(row: dict, stage: str) -> float:
    current = f"median_{stage}_ms"
    legacy = f"median_{stage}_time_ms"
    return float(row.get(current, row.get(legacy, 0.0)))


def control_timing(row: dict) -> float:
    if "median_control_compute_ms" in row:
        return float(row["median_control_compute_ms"])
    return float(row["median_control_compute_time_ms"])


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    results = load_results()
    y = np.arange(len(METHODS))

    rates = []
    lows = []
    highs = []
    timing_components = []
    timing_totals = []
    for method in METHODS:
        rows = results[method]
        successes = sum(strict_success(row) for row in rows)
        rate = successes / len(rows)
        low, high = wilson_interval(successes, len(rows))
        rates.append(100.0 * rate)
        lows.append(100.0 * low)
        highs.append(100.0 * high)

        selection_stage = (
            "cluster_scoring"
            if method == "Contextual Atlas-DeePC"
            else "selection"
        )
        components = [
            np.median([stage_timing(row, "localization") for row in rows]),
            np.median([stage_timing(row, "routing") for row in rows]),
            np.median([stage_timing(row, selection_stage) for row in rows]),
            np.median(
                [stage_timing(row, "matrix_preparation") for row in rows]
            ),
            np.median([stage_timing(row, "solve") for row in rows]),
        ]
        total = np.median([control_timing(row) for row in rows])
        components.append(max(0.0, total - sum(components)))
        timing_components.append(components)
        timing_totals.append(total)

    rates = np.asarray(rates)
    lows = np.asarray(lows)
    highs = np.asarray(highs)
    timing_components = np.asarray(timing_components)
    timing_totals = np.asarray(timing_totals)

    fig, (success_ax, timing_ax) = plt.subplots(
        1,
        2,
        figsize=(6.3, 3.25),
        gridspec_kw={"width_ratios": [0.88, 1.35]},
        sharey=True,
        constrained_layout=True,
    )

    success_ax.barh(
        y,
        rates,
        height=0.58,
        color="#4b7396",
        edgecolor="none",
        zorder=2,
    )
    success_ax.errorbar(
        rates,
        y,
        xerr=[rates - lows, highs - rates],
        fmt="none",
        color="#222222",
        capsize=2.5,
        linewidth=0.8,
        zorder=3,
    )
    for position, rate in zip(y, rates, strict=True):
        success_ax.text(
            rate + 1.2,
            position,
            f"{rate:.1f}%",
            va="center",
            ha="left",
            fontsize=7.5,
        )
    success_ax.set_xlim(0.0, 106.0)
    success_ax.set_xticks([0, 25, 50, 75, 100])
    success_ax.set_xlabel("Strict success rate (%)")
    success_ax.set_title("(a) Sustained 1-cm accuracy", loc="left", pad=4.0)
    success_ax.set_yticks(y, METHOD_TICK_LABELS)
    success_ax.invert_yaxis()
    success_ax.grid(axis="x", color="#d8d8d8", linewidth=0.5, zorder=0)

    left = np.zeros(len(METHODS))
    for module_index, module in enumerate(MODULES):
        values = timing_components[:, module_index]
        timing_ax.barh(
            y,
            values,
            left=left,
            height=0.58,
            color=MODULE_COLORS[module],
            edgecolor="white",
            linewidth=0.3,
            label=module,
            zorder=2,
        )
        left += values
    for position, total in zip(y, timing_totals, strict=True):
        timing_ax.text(
            total + 1.2,
            position,
            f"{total:.1f}",
            va="center",
            ha="left",
            fontsize=7.5,
        )
    timing_ax.set_xlim(0.0, 86.0)
    timing_ax.set_xticks([0, 20, 40, 60, 80])
    timing_ax.set_xlabel("Median computation per step (ms)")
    timing_ax.set_title("(b) Computation by module", loc="left", pad=4.0)
    timing_ax.tick_params(axis="y", labelleft=False)
    timing_ax.grid(axis="x", color="#d8d8d8", linewidth=0.5, zorder=0)

    for axis in (success_ax, timing_ax):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#555555")
        axis.spines["bottom"].set_color("#555555")
        axis.spines["left"].set_linewidth(0.6)
        axis.spines["bottom"].set_linewidth(0.6)

    fig.legend(
        *timing_ax.get_legend_handles_labels(),
        loc="outside lower center",
        ncol=3,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.3,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = OUTPUT_DIR / "reacher_success_and_computation"
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(output_stem.with_suffix(".pdf"))
    print(output_stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
