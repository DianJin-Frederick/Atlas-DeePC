"""Chart figures and media display for the current Reacher baseline notebook."""

import json
from pathlib import Path

import numpy as np

from atlas_deepc.baselines import unstandardized_atlas_deepc as experiment
from atlas_deepc.benchmarks import reacher_mixed_k as study

RESULT_ROOT = study.RESULT_ROOT
REPORT_ROOT = RESULT_ROOT / "baseline_report"


def chart_clouds(p, root=RESULT_ROOT, report_root=REPORT_ROOT, *, atlas_file=None):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    report_root = Path(report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    H = experiment.load_hankel(study.runner.DATA_PATH, p)
    atlas = experiment.routed.load_atlas(
        Path(atlas_file) if atlas_file is not None else Path(root) / f"output{p}/behavioral_atlas.npz")
    Y = H[24:].reshape(12, p, -1)
    past_color, future_color, representative_color = "#6699CC", "#DF7B20", "#004488"

    def poses(values):
        return np.stack([np.zeros((len(values), 2)), .10*values[:, [0, 2]], values[:, 6:8]], axis=1)

    chart_count = len(atlas["medoid_states"])
    rows = (chart_count + 7) // 8
    fig, axes = plt.subplots(rows, 8, figsize=(19, 2.5*rows), squeeze=False)
    rng, sampled_columns = np.random.default_rng(2026), {}
    for k, ax in enumerate(axes.flat):
        if k >= chart_count:
            ax.set_visible(False)
            continue
        members = np.flatnonzero(atlas["labels"] == k)
        sampled = np.sort(rng.choice(members, min(8, len(members)), replace=False))
        sampled_columns[k] = sampled.tolist()
        block = Y[:, :, sampled].transpose(2, 0, 1)
        for arms, color, alpha in [(poses(block[:, 2:].reshape(-1, p)), future_color, .22),
                                   (poses(block[:, :2].reshape(-1, p)), past_color, .5)]:
            ax.add_collection(LineCollection(arms, colors=color, linewidths=.8, alpha=alpha))
            ax.scatter(arms[:, 2, 0], arms[:, 2, 1], s=3, color=color, alpha=alpha, linewidths=0)
        representative = poses(atlas["medoid_states"][k][None])[0]
        ax.plot(representative[:, 0], representative[:, 1], color="white", lw=4.7)
        ax.plot(representative[:, 0], representative[:, 1], "-o", color=representative_color,
                lw=2.4, markersize=3.5, markerfacecolor="white", zorder=5)
        ax.scatter([0], [0], s=12, color="#333333", zorder=6)
        ax.axhline(0, color="#DDDDDD", lw=.6, zorder=0)
        ax.axvline(0, color="#DDDDDD", lw=.6, zorder=0)
        ax.set(xlim=(-.22, .22), ylim=(-.22, .22), aspect="equal", xticks=[], yticks=[])
        ax.set_title(rf"Chart {k} | $|\mathcal{{I}}_k|={len(members)}$", fontsize=10)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#AAAAAA")
    handles = [Line2D([], [], color=past_color, lw=2, label="Past poses (2 per trajectory)"),
               Line2D([], [], color=future_color, lw=2, label="Future poses (10 per trajectory)"),
               Line2D([], [], color=representative_color, lw=2.5, marker="o",
                      markerfacecolor="white", label="Nearest-centroid last-past pose")]
    fig.suptitle(f"Complete-trajectory chart configurations: {p} outputs, K={chart_count}", y=.997, fontsize=18)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .976), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .945), h_pad=1.1, w_pad=.7)
    for suffix in ("png", "pdf"):
        fig.savefig(report_root / f"physical_chart_clouds_output{p}.{suffix}", dpi=160, bbox_inches="tight")
    (report_root / f"cloud_samples_output{p}.json").write_text(json.dumps(sampled_columns, indent=2))
    return fig


def gallery(entries, kind):
    from IPython.display import HTML, Video, display
    cards = []
    for entry in entries:
        outcome = "SUCCESS" if entry["success"] else "FAILURE"
        color = "#007A5E" if entry["success"] else "#B33B24"
        mode = "graph" if entry["mode"] == "graph" else "no graph"
        player = Video(entry[kind], embed=True, width=620)._repr_html_()
        player = player.replace("<video ", '<video style="width:100%;height:auto" ')
        cards.append(f'<div style="min-width:0;background:white;color:#182238;border:1px solid #cbd0d6;padding:12px">'
                     f'<strong>p={entry["output_dimension"]}, {mode}; seed {entry["seed"]}: '
                     f'<span style="color:{color}">{outcome}</span></strong>'
                     f'<p>Final-100 mean: {entry["final100_mm"]:.3f} mm</p>{player}</div>')
    display(HTML('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:12px">'
                 + "".join(cards) + "</div>"))
