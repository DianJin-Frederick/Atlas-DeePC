"""Build the unified Reacher chart-count/rank report and optional batch controls."""

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = ROOT / "notebooks/reacher/ablations/ablation_chart_count_and_rank.ipynb"


def build(output=NOTEBOOK):
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [md(r"""# Reacher: Complete Chart-Count and Rank Ablation

All tested chart counts in one notebook:
**K = 1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96**, for **p=8 and p=10,
without and with graph guidance**, using **development seeds 211--310**.
These are not the 500-seed evaluation results on seeds 10001--10500.

The overview combines all 44 method/K best-rank results. The detailed sections
retain the full K=48/64/80/96 rank tables, serial metrics, and diagnostic plots.
For the 21 low-K completion searches, losing-rank checkpoints were intentionally
deleted after selection; only the selected ranks and success rates remain.
The original p=10 guided low-K study used a coarse search and local refinement.
Do not interpret all points as having identical rank-search budgets.

**Run All is report-only by default.** Optional low-K batch controls are at the
end, with RUN_BATCH=False. They retain the previous bounded/resumable runner.
The original result folders and experiment data remain separate and unchanged.

## Protocol and Interpretation

- Unstandardized linear charts, full-window projection-residual selection (L=12).
- Tracking + input effort + initial-output slack, lambda_ini=100000, Clarabel.
- Guided: chordal graph costs, waypoints, iterative N=10 terminal selection.
- Unguided: one fixed nearest eligible terminal configuration.
- Select highest success, then lowest median rollout RMSE, then lowest rank.
  Exception: the success-only low-K completion uses lowest rank to break ties.
- Strict success requires final-30 and final-100 mean errors below 10 mm and
  maximum errors below 20 mm. Failed solves remain in the statistics.
- Final-error quantiles summarize each rollout's final-100 mean error. Rollout
  RMSE includes the entire controlled trajectory, excluding warmup.
- Only selected-rank serial reruns are used for timing comparisons; sweep
  timings are affected by parallel execution. Separate sessions are not a
  controlled timing benchmark.
- Selection and reporting use the same exploratory seeds, not a held-out test.

The original p=10 guided chart-count study and K=48/64/80/96 rank sweeps were
measured at 320 controlled steps. The three-method low-K completion uses 300
steps. The overview's provenance table identifies the horizon of every point;
the merge does not relabel or recompute historical measurements.

The LTI dimension benchmark is $mL+n=2\times12+4=28$ for both output dimensions;
it is not a nonlinear rank guarantee. Ambient dimensions are 120 and 144.

"""), code('''import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")
from pathlib import Path
import json, sys
ROOT = next(root for parent in (Path.cwd(), *Path.cwd().parents)
            for root in (parent, parent / "code") if (root / "src/atlas_deepc").is_dir())
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from atlas_deepc.reporting.rank_sweep_k48_k64 import load_saved, load_chart_counts, SOURCES
from IPython import get_ipython
get_ipython().run_line_magic("matplotlib", "inline")
RUN_BATCH = False  # Explicit opt-in only, in the optional batch section below.
panels, provenance = load_saved()
count_points, count_provenance = load_chart_counts(panels)
ALL_COUNTS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96)
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
count_frame = pd.DataFrame(count_points).rename(columns={"charts": "K"})
assert not count_frame.duplicated(["K", "output_dimension", "reference_mode"]).any()
print("Verified: 4 chart counts, 688 complete rank/method panels, 68,800 sweep seed evaluations.")
print(f"Combined chart-count coverage: {len(count_frame)}/44 selected configurations.")
'''), md("## All Chart Counts: Best-Rank Success\nEach cell shows success rate and selected rank. Missing entries remain blank; incomplete searches are not treated as winners."), code('''def show_count_table(points):
    values = points.copy()
    values["result"] = values.apply(lambda r: f"{r.success_rate:.0f}% (rank {int(r['rank'])})", axis=1)
    table = values.pivot(index="K", columns=["output_dimension", "reference_mode"], values="result")
    table = table.reindex(index=ALL_COUNTS, columns=pd.MultiIndex.from_tuples(CASES))
    table.columns = LABELS
    display(table.fillna("Not complete"))

show_count_table(count_frame)
'''), md("## Success Versus Number of Charts\nBest among the tested ranks, on development seeds. The combined curve includes different saved horizons and tie-break rules, listed below."), code('''fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
styles = (("#258774", "--", "s"), ("#9467ad", "-", "s"),
          ("#377eb4", "--", "o"), ("#e67e17", "-", "o"))
for (p, mode), label, (color, linestyle, marker) in zip(CASES, LABELS, styles):
    points = count_frame[(count_frame.output_dimension == p) & (count_frame.reference_mode == mode)].set_index("K").reindex(ALL_COUNTS)
    ax.plot(ALL_COUNTS, points.success_rate, linestyle=linestyle, marker=marker,
            color=color, label=label, markerfacecolor="white" if p == 8 else color)
ax.set(xscale="log", xlabel="Number of charts K", ylabel="Success rate (%)", ylim=(0, 105))
ax.set_xticks(ALL_COUNTS, labels=ALL_COUNTS)
ax.minorticks_off()
ax.grid(alpha=.2)
ax.legend(loc="lower right", ncol=2)
plt.show()
'''), md("## Search Coverage and Provenance\nLow-K completion retains only winners, not full rank curves. Original saved horizons are shown explicitly."), code('''coverage = count_frame[["K", "output_dimension", "reference_mode", "rank", "success_rate", "controlled_steps", "tested_ranks"]].copy()
coverage["tested_rank_count"] = coverage.tested_ranks.map(len)
coverage["tie_break"] = coverage.controlled_steps.map({300: "lowest rank", 320: "lowest median RMSE, then rank"})
display(coverage.sort_values(["K", "output_dimension", "reference_mode"]).reset_index(drop=True))
print("Original guided chart-count source:", count_provenance["source"])
if "completion" in count_provenance:
    print("Low-K completion source:", count_provenance["completion"]["source"])
'''), md("## Selected Ranks: K = 48, 64, 80, 96\nSerially verified best configurations for each K and method."), code('''display(selected[["K", "output_dimension", "reference_mode", "rank", "success_rate",
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
    cells.extend([md("""## Optional: Resume One Low-K Batch

This retains the previous batch interface for K=1,2,4,8,16,24,32 and the three
completion methods (p=8 without/with guidance; p=10 without guidance).
All 21 searches are already complete in the current saved results.
It does not rerun the historical p=10 guided study or the large-K rank sweeps.

Only set RUN_BATCH=True deliberately. BATCH_SIZE limits new rollouts per call;
WORKERS controls concurrent rank panels. T_sim=300; seeds 211--310; ranks 28--70.
Completed seeds are resumable, and losing rank checkpoints are deleted only
after the complete search selects a winner. Avoid running alongside timing studies.
"""), code('''from atlas_deepc.ablations import low_k_completion as study
RESULT_ROOT = ROOT / "experiments/reacher/low_k_missing_batches_t300_v3"
BATCH_SIZE = 100
WORKERS = 4
RUN_BATCH = False
assert study.CONTROL_STEPS == 300
if RUN_BATCH:
    batch = study.run_batch(RESULT_ROOT, batch_size=BATCH_SIZE, workers=WORKERS)
    print("New rollouts in this batch:", batch["batch_completed_rollouts"])
else:
    print("Report only; no experiments launched.")
payload = study.report(RESULT_ROOT)
print("Completed low-K rollouts:", payload["completed_rollouts"], "/ 90300")
print("Complete:", payload["complete"])
display(pd.DataFrame(payload["progress"]))
if RUN_BATCH:
    count_points, count_provenance = load_chart_counts(panels)
    count_frame = pd.DataFrame(count_points).rename(columns={"charts": "K"})
    show_count_table(count_frame)
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
