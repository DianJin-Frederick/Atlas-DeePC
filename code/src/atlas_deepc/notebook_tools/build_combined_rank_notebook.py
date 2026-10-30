"""One read-only notebook for the four completed Reacher rank studies."""

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = ROOT / "notebooks/reacher/ablations/ablation_linear_subspace_rank.ipynb"


def build(output=NOTEBOOK):
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [md(r"""# Reacher Rank Ablation: K = 48, 64, 80, 96

One report for the three formerly separate rank notebooks. Each K tests every
integer rank 28--70 for p=8 and p=10, without and with graph guidance, on the
same 100 development seeds (211--310): 17,200 sweep evaluations per K.

This notebook **only reads saved results**. Run All refreshes plots and tables;
it never starts or resumes rollouts, fits atlases, or writes experiment data.
The original experiment folders remain separate and unchanged.

## Protocol and Interpretation

- Unstandardized linear charts, full-window projection-residual selection (L=12).
- Tracking + input effort + initial-output slack, lambda_ini=100000, Clarabel.
- Guided: chordal graph costs, waypoints, iterative N=10 terminal selection.
- Unguided: one fixed nearest eligible terminal configuration.
- Select highest success, then lowest median rollout RMSE, then lowest rank.
- Strict success requires final-30 and final-100 mean errors below 10 mm and
  maximum errors below 20 mm. Failed solves remain in the statistics.
- Final-error quantiles summarize each rollout's final-100 mean error. Rollout
  RMSE includes the entire controlled trajectory, excluding warmup.
- Only selected-rank serial reruns are used for timing comparisons; sweep
  timings are affected by parallel execution. Separate sessions are not a
  controlled timing benchmark.
- Selection and reporting use the same exploratory seeds, not a held-out test.

These saved curves were measured at 320 controlled steps, following 12 warmup
steps. Current runnable experiments use 300 steps. The historical measurements
are preserved, not silently relabeled or recomputed.

The LTI dimension benchmark is $mL+n=2\times12+4=28$ for both output dimensions;
it is not a nonlinear rank guarantee. Ambient dimensions are 120 and 144.

The ongoing [low-K batch study](ablation_chart_count_and_rank.ipynb) is separate.
"""), code('''import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")
from pathlib import Path
import json, sys
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src/atlas_deepc").is_dir())
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from atlas_deepc.reporting.rank_sweep_k48_k64 import load_saved, SOURCES
from IPython import get_ipython
get_ipython().run_line_magic("matplotlib", "inline")
RUN_EXPERIMENTS = False  # This notebook has no rollout-launch code.
panels, provenance = load_saved()
COUNTS = (48, 64, 80, 96)
CASES = ((8, "terminal"), (8, "graph"), (10, "terminal"), (10, "graph"))
LABELS = ("p=8, no guidance", "p=8, guided", "p=10, no guidance", "p=10, guided")
records = [dict(row, K=k) for (k, p, mode), (curve, winner) in panels.items() for row in curve]
frame = pd.DataFrame(records)
selected = pd.DataFrame([
    dict(row, K=k) for k in COUNTS
    for row in json.loads((SOURCES[k] / "best_rank_table.json").read_text())["columns"]
])
assert len(frame) == 688 and len(selected) == 16
print("Verified: 4 chart counts, 688 complete rank/method panels, 68,800 sweep seed evaluations.")
'''), md("## Selected Ranks\nSerially verified best configurations for each K and method."), code('''display(selected[["K", "output_dimension", "reference_mode", "rank", "success_rate",
                  "median_error_mm", "median_rollout_rmse_mm", "median_time_ms"]].round(3))
'''), md("## Success Versus Rank\nOutlined markers identify the selected ranks; tied maximum success rates use the RMSE tie-breaker."), code('''fig, axes = plt.subplots(4, 2, figsize=(12, 11), sharex=True, sharey=True, constrained_layout=True)
for row, k in enumerate(COUNTS):
    for col, p in enumerate((8, 10)):
        ax = axes[row, col]
        for mode, label, color, style in (("terminal", "Without guidance", "#377eb4", "--"),
                                           ("graph", "With guidance", "#e67e17", "-")):
            curve, winner = panels[k, p, mode]
            data = pd.DataFrame(curve)
            ax.plot(data["rank"], data.success_rate, marker=".", linestyle=style, color=color, label=label)
            ax.plot(winner["rank"], winner["success_rate"], "o", color=color,
                    markerfacecolor="white", markersize=9, markeredgewidth=2)
        ax.set(title=f"K={k}, p={p}", ylim=(0, 105), xlim=(27, 71))
        ax.grid(alpha=.2)
        if col == 0: ax.set_ylabel("Success rate (%)")
        if row == 3: ax.set_xlabel("Chart rank")
axes[0, 0].legend()
plt.show()
''')]
    for k in (48, 64, 80, 96):
        cells.extend([md(f"## K = {k}\nComplete success-rate table, selected-rank serial metrics, and tracking diagnostics."),
            code(f'''k = {k}
data = frame[frame.K == k].copy()
rates = data.pivot(index="rank", columns=["output_dimension", "reference_mode"], values="success_rate")
rates = rates.reindex(columns=pd.MultiIndex.from_tuples(CASES))
rates.columns = LABELS
display(rates.style.format("{{:.0f}}%"))
best = selected[selected.K == k].set_index(["output_dimension", "reference_mode"]).reindex(CASES)
metrics = ["rank", "success_rate", "median_error_mm", "p90_error_mm", "p95_error_mm", "p99_error_mm",
           "median_rollout_rmse_mm", "median_time_ms", "p95_time_ms", "p99_time_ms"]
table = best[metrics].T
table.columns = LABELS
display(table.round(3))
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), constrained_layout=True)
for (p, mode), label in zip(CASES, LABELS):
    curve = data[(data.output_dimension == p) & (data.reference_mode == mode)].sort_values("rank")
    for ax, metric, ylabel in ((axes[0], "median_rollout_rmse_mm", "Median rollout RMSE (mm)"),
                               (axes[1], "p95_error_mm", "P95 final-100 mean error (mm)")):
        ax.plot(curve["rank"], curve[metric], ".-", label=label)
        ax.set(xlabel="Chart rank", ylabel=ylabel, ylim=(0, None))
        ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
plt.show()
details = data.drop(columns=["timing_modules", "per_seed_rmse_mm"], errors="ignore")
display(HTML("<details><summary>All saved rank metrics and solver diagnostics</summary>" +
             details.round(3).to_html(index=False) + "</details>"))
if "failed_seeds" in best:
    display(best[["rank", "failed_seeds"]])
latex = SOURCES[k] / "table_best_ranks.tex"
if latex.exists():
    print(latex.read_text())
''')])
    cells.extend([md("## Data Provenance\nSource folders and hashes are checked when loading. No original results are modified."),
                  code('''display(pd.DataFrame([{"K": k, "source": str(SOURCES[k]),
                           "selection_rule": provenance[k]["selection_rule"]} for k in COUNTS]))
''')])
    nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {
        "display_name": "gym-robotics-mujoco", "language": "python", "name": "gym-robotics-mujoco"}})
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            compile(cell.source, str(output), "exec")
    nbformat.write(nb, output)
    return Path(output)


if __name__ == "__main__":
    print(build())
