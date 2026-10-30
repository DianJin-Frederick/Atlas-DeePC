"""Preview controller success versus compute cost from all 500 saved traces."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_comparison_mpl")
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/reacher_comparison_500"
CASES = [
    ("Atlas-DeePC (with planning)", "four_baselines_500_mixed_k_t300_v1/output8/graph", "#df771a", "^", (14, 10), "left"),
    ("Atlas-DeePC (without planning)", "four_baselines_500_mixed_k_t300_v1/output8/terminal", "#337caf", "o", (14, 8), "left"),
    ("Contextual", "contextual_dpc_k64_rank53_terminal8_10001_10500_t300_v1", "#378b7c", "s", (12, 8), "left"),
    ("Select", "select_dpc_upstream_final_clarabel_10001_10500_t300_v1", "#8067a5", "D", (-12, 12), "right"),
]


def main():
    rows = []
    for name, folder, color, marker, offset, align in CASES:
        path = ROOT / "experiments/reacher" / folder
        files = list(path.glob("seed*_control_results.npz"))
        assert sorted(int(f.name.split("_")[0][4:]) for f in files) == list(range(10001, 10501))
        times, successes = [], 0
        for seed in range(10001, 10501):
            with np.load(path / f"seed{seed}_control_results.npz", allow_pickle=True) as f:
                e = f["distance"] * 1000
                t = f["control_compute_time_s"] * 1000
                assert e.shape == t.shape == (300,)
                assert np.isfinite(e).all() and np.isfinite(t).all()
                successes += all(e[-n:].mean() < 10 and e[-n:].max() < 20 for n in (30, 100))
                times.extend(t[5:])
        rows.append(dict(method=name, source=str(path), successes=successes,
                         rollouts=500, success_percent=successes/5,
                         median_compute_ms=float(np.median(times)), timed_steps=len(times)))

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                         "mathtext.fontset": "cm", "font.size": 8,
                         "axes.labelsize": 9, "axes.edgecolor": "#666666",
                         "axes.linewidth": .7, "pdf.fonttype": 42,
                         "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(2.8, 2.65))
    fig.subplots_adjust(left=.19, right=.97, bottom=.18, top=.77)
    ax.set_xscale("log")
    ax.set(xlim=(12, 2300), ylim=(78, 100), yticks=[80, 85, 90, 95, 100],
           xlabel="Computation (ms/step)", ylabel="Success rate (%)")
    ax.xaxis.set_major_locator(FixedLocator([20, 100, 1000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e7e7e7", lw=.6)
    ax.set_axisbelow(True)
    handles = []
    for row, (_, _, color, marker, offset, align) in zip(rows, CASES):
        x, y = row["median_compute_ms"], row["success_percent"]
        handles.append(ax.scatter(x, y, s=43, marker=marker, color=color,
                                  edgecolors="white", linewidths=.7, zorder=3))
    legend = fig.legend([handles[i] for i in [1, 0, 3, 2]],
                        ["Atlas", "Atlas + planning", "Select", "Contextual"],
                        loc="upper center", bbox_to_anchor=(.53, .99), ncol=2,
                        frameon=False, fontsize=8, handletextpad=.35,
                        columnspacing=1, borderaxespad=0, labelspacing=.6)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in [*ax.texts, *fig.texts, legend, ax.xaxis.label, ax.yaxis.label]:
        box = text.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"success_vs_compute_500.{ext}", dpi=250)
    (OUT / "success_vs_compute_500.json").write_text(json.dumps({
        "seeds": [10001, 10500],
        "success_definition": "mean <10 mm and max <20 mm over both final 30 and final 100 steps",
        "timing_definition": "pooled online computation, excluding first 5 steps of each rollout",
        "results": rows}, indent=2) + "\n")
    print(OUT / "success_vs_compute_500.pdf")


if __name__ == "__main__":
    main()
