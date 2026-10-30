"""Plot measured closed-loop errors from full-length development validation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(directory, destination, extra=None):
    manifest = json.loads((directory / "manifest.json").read_text())
    reports = json.loads((directory / "report.json").read_text())
    seeds, steps = manifest["seeds"], manifest["steps"]
    assert steps == 300
    sources = {name: directory for name in reports}
    if extra is not None:
        extra_manifest = json.loads((extra / "manifest.json").read_text())
        assert extra_manifest["seeds"] == seeds and extra_manifest["steps"] == steps
        extra_reports = json.loads((extra / "report.json").read_text())
        assert not set(reports).intersection(extra_reports)
        reports.update(extra_reports)
        sources.update({name: extra for name in extra_reports})
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    labels = []
    for i, (name, report) in enumerate(reports.items()):
        errors = []
        for seed in seeds:
            with np.load(sources[name] / name / f"seed{seed}_control_results.npz", allow_pickle=False) as z:
                error = z["distance"] * 1000
                assert len(error) == steps and np.isfinite(error).all()
                errors.append(error)
        errors = np.stack(errors)
        lo, median, hi = np.percentile(errors, [25, 50, 75], axis=0)
        color = f"C{i}"
        label = name.replace("_", " ")
        axes[0].plot(np.arange(1, steps + 1), median, color=color, label=label, linewidth=1.6)
        axes[0].fill_between(np.arange(1, steps + 1), lo, hi, color=color, alpha=.10)
        tail = errors[:, -100:].mean(axis=1)
        axes[1].scatter(i + np.linspace(-.14, .14, len(seeds)), tail, color=color, s=35)
        axes[1].plot([i-.25, i+.25], [tail.mean()]*2, color="black", linewidth=2)
        labels.append(f"{label}\n{report['successes']}/{len(seeds)} successes")
    for ax in axes:
        ax.axhline(10, linestyle="--", color=".35", linewidth=1)
        ax.grid(alpha=.15)
        ax.set_ylim(bottom=0)
    axes[0].set(xlabel="Control step", ylabel="Fingertip–target distance (mm)",
                title="Across-seed median; shaded interquartile range")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].set(xticks=range(len(labels)), xticklabels=labels,
                ylabel="Mean distance over final 100 steps (mm)",
                title="Each dot is one seed; black line is the mean")
    axes[1].tick_params(axis="x", labelsize=8)
    fig.suptitle(f"MVC-BDPC adaptation: weight-only tuning · {len(seeds)} development seeds × 300 steps")
    fig.supxlabel("Dashed line: 10 mm mean-error threshold only. Success also requires max-error checks over the final 30 and 100 steps.", fontsize=8)
    destination.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(destination / f"bdpc_weight_validation.{suffix}", dpi=180)
    plt.close(fig)
    unique_weights, phase_results = set(), {}
    for phase in ("screen", "refine", "local300", "validation300", "local_best_validation300"):
        root = directory.parent / phase
        progress = json.loads((root / "progress.json").read_text())
        assert progress["state"] == "complete", f"Phase still incomplete: {phase}"
        stage_manifest = json.loads((root / "manifest.json").read_text())
        stage_reports = json.loads((root / "report.json").read_text())
        assert set(stage_reports) == set(stage_manifest["candidates"])
        for report in stage_reports.values():
            assert report["finished"] == len(stage_manifest["seeds"])
            unique_weights.add(json.dumps(report["weights"], sort_keys=True))
        phase_results[phase] = {
            "steps": stage_manifest["steps"], "seeds": stage_manifest["seeds"],
            "candidates": len(stage_reports),
            "attempted_rollouts": sum(r["finished"] for r in stage_reports.values()),
            "full_rollouts": sum(r["full_rollouts"] for r in stage_reports.values()),
            "aborted_rollouts": sum(r["aborted"] for r in stage_reports.values()),
            "successful_rollouts": sum(r["successes"] for r in stage_reports.values()),
        }
    summary = {"unique_weight_settings": len(unique_weights), "phases": phase_results,
               "validation_seeds": seeds, "validation_steps": steps,
               "validation": reports,
               "scope": "Paper-based Reacher adaptation, frozen offline model; weight-only development search, not a published-method impossibility result.",
               "success_criterion": "Mean <10 mm and max <20 mm in BOTH final 30 and final 100 steps.",
               "defaults_changed": False, "test_seeds_used": False}
    (destination / "tuning_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--destination", type=Path, default=Path(__file__).parent)
    parser.add_argument("--extra", type=Path)
    args = parser.parse_args()
    main(args.directory, args.destination, args.extra)
