"""Paper plots from validated completed checkpoints; never run simulations."""
from pathlib import Path
import argparse
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_paper_mpl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import baseline_control_parameter_ablation as ablation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator

SOURCE = ROOT / "experiments/routed_deepc_iid/ablation_control_parameter_baseline_clarabel_100_v1"
OUT = ROOT / "paper/figures/reacher_lam_g_ablation"
ORANGE, BLUE, INK, GRAY = "#DF7B20", "#0077BB", "#273441", "#89949E"
BASELINE = .05


def wilson(k, n):
    z = 1.959963984540054
    p, den = k / n, 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return [100*(center-half), 100*(center+half)]


def summarize():
    manifest, payloads = ablation.load_lam_g_sweep(SOURCE)
    assert manifest["complete"] and len(payloads) == 52
    assert manifest["signature"]["seeds"] == list(range(211, 311))
    records = []
    for value, panel in sorted(payloads.items()):
        rows = panel["results"]
        assert panel["complete"] and len(rows) == 100
        successes = sum(r["success_30"] and r["success_100"] for r in rows)
        record = {"lam_g": value, "success_percent": successes,
                  "wilson95_percent": wilson(successes, 100),
                  "unsolved_steps": sum(r["unsolved_steps"] for r in rows)}
        for name, key, scale in (
            ("error_mm", "mean_last_100", 1000),
            ("task_cost", "closed_loop_task_cost", 1),
            ("prediction_mm", "mean_prediction_error_xy", 1000),
            ("rollout_median_time_ms", "median_control_compute_time_ms", 1),
        ):
            values = np.asarray([r[key] for r in rows])*scale
            assert np.all(np.isfinite(values))
            record[name] = dict(zip(("q25", "median", "q75", "p90", "p95", "p99"),
                                   np.percentile(values, [25, 50, 75, 90, 95, 99]).tolist()))
        records.append(record)
    return manifest, records


def style(ax, title, ylabel):
    ax.set_xscale("symlog", linthresh=.005, linscale=.65)
    ax.set_xlim(-.0007, 140)
    ax.set_xticks([0, .01, .1, 1, 10, 100],
                 ["0", r"$10^{-2}$", r"$10^{-1}$", "1", "10", r"$10^2$"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"$\lambda_g$", labelpad=1)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_title(title, loc="left", pad=4, fontsize=8.5)
    ax.axvline(BASELINE, color=INK, ls=":", lw=.85, alpha=.8, zorder=1)
    ax.grid(axis="y", color="#E6E8EA", lw=.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#71777B")
        ax.spines[side].set_linewidth(.65)
    ax.tick_params(length=2.5, width=.65, pad=2)


def distribution(ax, x, records, metric, scale=1):
    values = lambda key: np.asarray([r[metric][key] for r in records])/scale
    ax.fill_between(x, values("q25"), values("q75"), color=ORANGE, alpha=.16, lw=0)
    ax.plot(x, values("median"), color=ORANGE, lw=1.35, solid_capstyle="round")
    idx = np.flatnonzero(x == BASELINE)[0]
    ax.plot(BASELINE, values("median")[idx], "o", ms=3, color=ORANGE, zorder=4)


def save(fig, stem, enabled=True):
    # Keep the exact physical canvas; do not use bbox_inches='tight'.
    if enabled:
        fig.savefig(OUT / f"{stem}.pdf")
        fig.savefig(OUT / f"{stem}.png", dpi=350)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--performance-only", action="store_true")
    args = parser.parse_args()
    manifest, records = summarize()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                         "mathtext.fontset": "stix", "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "text.color": INK,
                         "axes.labelcolor": INK, "pdf.fonttype": 42, "ps.fonttype": 42})
    x = np.asarray([r["lam_g"] for r in records])
    fig = plt.figure(figsize=(6, 1.5))
    a = fig.add_axes([.067, .28, .385, .49])
    b = fig.add_axes([.58, .28, .395, .49])
    style(a, "(a) Success rate", "Success (%)")
    limits = np.asarray([r["wilson95_percent"] for r in records])
    a.fill_between(x, limits[:, 0], limits[:, 1], color=ORANGE, alpha=.16, lw=0)
    a.plot(x, [r["success_percent"] for r in records], color=ORANGE, lw=1.35)
    base = next(r for r in records if r["lam_g"] == BASELINE)
    a.plot(BASELINE, base["success_percent"], "o", ms=3, color=ORANGE)
    a.set(ylim=(79, 100), yticks=[80, 90, 100])
    style(b, "(b) Tracking error", "Error (mm)")
    distribution(b, x, records, "error_mm")
    b.plot(x, [r["error_mm"]["p90"] for r in records], color=BLUE, ls="--", lw=1.0)
    b.axhline(10, color=GRAY, lw=.7, ls=(0, (4, 3)))
    b.set(ylim=(0, 16), yticks=[0, 5, 10, 15])
    handles = [Line2D([], [], color=INK, ls=":", lw=.9, label=r"Baseline $\lambda_g=0.05$"),
               Line2D([], [], color=ORANGE, lw=1.35, label="Rate / median"),
               Line2D([], [], color=BLUE, ls="--", lw=1, label="90th percentile (b)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.025),
               ncol=3, frameon=False, fontsize=7, handlelength=2.1, columnspacing=1.8)
    save(fig, "lam_g_performance")

    fig = plt.figure(figsize=(6, 1.5))
    a = fig.add_axes([.067, .28, .385, .49])
    b = fig.add_axes([.58, .28, .395, .49])
    style(a, "(a) One-step prediction error", "Error (mm)")
    distribution(a, x, records, "prediction_mm")
    a.set_ylim(bottom=0)
    style(b, "(b) Online computation time", "Time (ms)")
    distribution(b, x, records, "rollout_median_time_ms")
    b.set(ylim=(0, 40), yticks=[0, 10, 20, 30, 40])
    fig.legend(handles=[handles[0], Line2D([], [], color=ORANGE, lw=1.35, label="Median"),
                        Patch(facecolor=ORANGE, alpha=.16, label="Interquartile range")],
               loc="upper center", bbox_to_anchor=(.5, 1.025), ncol=3,
               frameon=False, fontsize=7, handlelength=2.1, columnspacing=2.1)
    save(fig, "lam_g_prediction_timing", enabled=not args.performance_only)
    metadata = {
        "source_notebook": "notebooks/ablations/ablation_control_parameter.ipynb",
        "source_results": str(SOURCE), "source_signature": manifest["signature"],
        "validated_rollouts": 5200, "rank": 60, "seeds": [211, 310],
        "figure_inches": [6, 1.5], "base_font_pt": 8,
        "performance_panels": ["Success rate", "Tracking error"],
        "x_axis": "Symmetric-log, linear for abs(lambda_g)<=0.005; zero included",
        "tracking_error": "Across-rollout quantiles of each rollout's final-100 mean fingertip error, including failures",
        "success_band": "Pointwise 95% Wilson intervals; not simultaneous intervals across the sweep",
        "distribution_bands": "25th-75th percentiles across 100 rollouts, not confidence intervals",
        "task_cost": "sum_t [100000*||fingertip_t-target||^2 + 0.0001*||u_t||^2], not full QP objective",
        "prediction_error": "Across-rollout distribution of mean one-step fingertip prediction error",
        "time": "Across-rollout distribution of per-rollout median control time; first 5 steps excluded, not pooled per-step quantiles",
        "unsolved_qps": sum(r["unsolved_steps"] for r in records),
        "records": records,
    }
    (OUT / "plot_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Validated 52 complete panels / 5,200 paired rollouts; unsolved QPs:", metadata["unsolved_qps"])
    print("Baseline:", json.dumps(base))
    print("Saved", 1 if args.performance_only else 2, "6 x 1.5 inch PDF/PNG figures to", OUT)


if __name__ == "__main__":
    main()
