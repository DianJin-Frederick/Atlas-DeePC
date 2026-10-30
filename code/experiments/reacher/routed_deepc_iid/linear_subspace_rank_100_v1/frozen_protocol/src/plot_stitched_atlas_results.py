import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_run(path):
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def plot_results(experiment_dir):
    summary = json.loads((experiment_dir / "summary.json").read_text())
    runs = {int(row["seed"]): load_run(experiment_dir / f"seed{row['seed']}_control_results.npz") for row in summary["results"]}
    method = summary.get("method", "")
    title = (
        "Graph-routed fixed-chart Atlas-DeePC"
        if method == "graph_routed_fixed_chart_deepc"
        else "Stitched fixed-chart atlas: held-out targets"
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))
    many_runs = len(summary["results"]) > 12
    success_label_used = False
    for result in summary["results"]:
        seed = int(result["seed"])
        success = result["success_30"] and result["success_100"]
        style = "-" if success else "--"
        if many_runs and success:
            label = "successful seeds" if not success_label_used else None
            ax.plot(
                runs[seed]["distance"] * 100.0,
                style,
                color="#2166ac",
                alpha=0.38,
                linewidth=1.2,
                label=label,
            )
            success_label_used = True
        else:
            color = "#b2182b" if many_runs else None
            ax.plot(
                runs[seed]["distance"] * 100.0,
                style,
                color=color,
                linewidth=2.0 if many_runs else 1.7,
                label=f"failed seed {seed}" if many_runs else f"seed {seed}",
            )
    ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.65, label="1 cm threshold")
    ax.set_xlabel("Control step")
    ax.set_ylabel("Fingertip-target distance (cm)")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(experiment_dir / "multi_seed_distance.png", dpi=180)
    plt.close(fig)

    preferred = [seed for seed in (1, 3) if seed in runs]
    example_seeds = preferred if preferred else [next(iter(runs))]
    for seed in example_seeds:
        run = runs[seed]
        xy = run["y"][:, 6:8]
        target = run["ref"][-1, 6:8]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].plot(xy[:, 0], xy[:, 1], color="#2166ac", linewidth=1.8)
        axes[0].scatter(xy[0, 0], xy[0, 1], color="#555555", s=35, label="start")
        axes[0].scatter(target[0], target[1], color="#b2182b", marker="x", s=70, linewidth=2.0, label="target")
        axes[0].set_xlabel("x (m)")
        axes[0].set_ylabel("y (m)")
        axes[0].set_title(f"Seed {seed} fingertip path")
        axes[0].axis("equal")
        axes[0].grid(alpha=0.2)
        axes[0].legend()
        axes[1].plot(run["distance"] * 100.0, color="#2166ac", linewidth=1.8)
        axes[1].axhline(1.0, color="#b2182b", linewidth=1.0, linestyle="--")
        axes[1].set_xlabel("Control step")
        axes[1].set_ylabel("Distance (cm)")
        axes[1].set_title("Tracking error")
        axes[1].grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(experiment_dir / f"seed{seed}_tracking.png", dpi=180)
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot saved stitched-atlas results")
    parser.add_argument("experiment_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    plot_results(parse_args().experiment_dir)
