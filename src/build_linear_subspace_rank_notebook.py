"""Build the dense, current-baseline rank ablation report from saved checkpoints."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/ablations/ablation_linear_subspace_rank.ipynb"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [md(r"""# Atlas-DeePC: Linear-Subspace Rank

**Every integer rank from 20 through 80, each on the same 100 seeds (211–310):
6,100 full-duration rollouts. Rank 60 is the baseline at sweep launch.** These seeds have
been used for exploratory ablations; this is not an untouched test set.

Only `svd_rank` changes. The controller is the frozen launch version of
`baseline_atlas_deepc.BaselineAtlasDeePC`:
48 frozen physical-configuration charts, **uncentered linear** behavioral bases,
current-chart DeePC, graph waypoint references, and iterative N=10 terminal
configuration selection. The archived contextual affine-rank study is a separate
experiment; none of its results are used here.

The shared workspace's baseline changed to $\lambda_g=0$ after this sweep
started. This experiment retains its original **$\lambda_g=0.05$** specification.
Its source snapshot matches all seven launch-manifest source hashes exactly;
no new-baseline results are mixed into this sweep. The running workers retained
the original imported code, and the report uses the verified source snapshot.

## Fixed setup

- Same random-exploration offline data, seeds 2026–2225; no new trajectories.
- Full input–output Hankel columns, $H\in\mathbb R^{144\times37800}$, ten outputs,
  two inputs, past horizon 2 and prediction horizon 10.
- Frozen chart memberships, centers, graph, and representatives. Localization
  uses the latest past configuration; input-feature weight is zero.
- Rank-$r$ basis is computed from the **uncentered** chart matrix via the existing
  eigendecomposition of $XX^\top$. Offset is zero; no $\sum g=1$ constraint.
- Coefficient bounds are the projected offline coordinate mean ±3 standard
  deviations (with the baseline $10^{-8}$ scale floor), recomputed at each rank.
- $Q=\mathrm{diag}(5000,5000,5000,5000,20,20,10^5,10^5,5000,5000)$,
  $R=10^{-4}I$, $\lambda_g=.05$, $\lambda_{ini}=10^5$, $\lambda_{\Delta u}=5$;
  no additional terminal cost, input limits $[-1,1]$, 320 steps.
- Same ten nearest eligible offline terminal candidates, at least 6 cm source
  target separation; candidate choice minimizes directed route cost online.
  Exact target position, zero reference velocities, and the 7 cm latch are retained.
- No measurement noise, contextual rescue, fitted model, IK, or trajectory replay.

**QP failure behavior:** the unchanged baseline runner applies a zero input when
the QP returns no solution, then continues the rollout. Those steps are retained
and counted as unsolved; they are not QP-generated actions. This matters at low
ranks. `optimal_inaccurate` is accepted by the existing baseline and is excluded
from the unsolved count. Compact checkpoints retain aggregate counts by solver
status in their summary, while individual per-step status arrays are discarded.
Status counts alone do not identify the underlying numerical failure mechanism.

Changing rank changes the number of behavioral coefficients and the retained
subspace; it does not refit clustering or retune any control weights. Nominal rank
means the number of retained basis columns, not a thresholded numerical-rank estimate.
"""), code('''import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from IPython.display import HTML, display
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p / "src/linear_subspace_rank_ablation.py").exists())
FROZEN_SOURCE = ROOT / "experiments/routed_deepc_iid/linear_subspace_rank_100_v1/frozen_protocol/src"
assert FROZEN_SOURCE.is_dir(), "The verified launch-source snapshot is required."
sys.path.insert(0, str(FROZEN_SOURCE))
import linear_subspace_rank_ablation as experiment
assert Path(experiment.__file__).resolve().parent == FROZEN_SOURCE.resolve()
get_ipython().run_line_magic("matplotlib", "inline")
RUN_EXPERIMENTS = False
WORKERS = 6
RANKS = list(range(20, 81))
SEEDS = list(range(211, 311))
BASELINE_RANK = 60
if RUN_EXPERIMENTS:
    experiment.run_all(RANKS, SEEDS, workers=WORKERS)
RESULT_ROOT = experiment.RESULT_ROOT
payload = json.loads((RESULT_ROOT / "summary.json").read_text())
assert payload["version"] == experiment.VERSION
assert payload["ranks"] == RANKS and payload["seeds"] == SEEDS
assert payload["complete"], "Sweep is still in progress; do not report partial results as complete."
assert [row["rank"] for row in payload["completed"]] == RANKS
assert experiment.prepare() == payload["signature"], "Experiment sources/data changed."
panels, records = {}, []
for rank in RANKS:
    panel = json.loads((RESULT_ROOT / f"rank{rank:03d}" / "summary.json").read_text())
    assert panel["version"] == experiment.VERSION and panel["rank"] == rank
    assert panel["seeds"] == SEEDS and [r["seed"] for r in panel["results"]] == SEEDS
    assert panel["timing"]["timed_steps"] == 31500
    panels[rank] = panel
    for summary_row in panel["results"]:
        seed = summary_row["seed"]
        row = experiment.load_checkpoint(experiment.checkpoint_path(rank, seed), rank, seed)
        assert row == summary_row
        records.append({**row, "rank": rank, "strict": experiment.strict_success(row),
                        "tail100_mm": 1000*row["mean_last_100"]})
frame = pd.DataFrame(records)
assert len(frame) == 6100 and not frame.duplicated(["rank", "seed"]).any()
for seed in SEEDS:
    targets = np.array(frame.loc[frame.seed == seed, "exact_target_xy"].tolist())
    np.testing.assert_array_equal(targets, np.repeat(targets[:1], len(RANKS), axis=0))
audit = json.loads((RESULT_ROOT / "fairness_audit.json").read_text())
print("Verified: 61 ranks × 100 paired seeds = 6,100 rollouts; 320 steps each.")
print("All checkpoint metrics, identical targets, linear baseline parameters, source hashes, and timing counts verified.")
display(pd.Series({"Offline seed range": audit["data_seed_range"],
    "Stabilized offline episodes": audit["offline_stabilized_episode_count"],
    "Timed steps per rank": 31500,
    "Baseline rank": BASELINE_RANK}))
BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#667085"
def rank_axis(ax, ylabel):
    ax.axvline(BASELINE_RANK, color=GRAY, linestyle="--", linewidth=1, label="Rank 60 baseline")
    ax.set(xlim=(20, 80), xlabel="Retained linear-subspace rank", ylabel=ylabel)
    ax.grid(alpha=.2)
def save_show(fig, name):
    fig.savefig(RESULT_ROOT / name, dpi=160)
    plt.show()
'''), md("""## Success and final-window error

Strict success requires **both** the final-30 and final-100 windows to have mean
fingertip error <10 mm and maximum error <20 mm. All 320 steps are evaluated;
entering the 10 mm neighborhood once is reported separately. The 10 mm line in
mean-error plots represents only one part of this success definition.
"""), code('''summary = pd.DataFrame([{"rank": rank,
    "successes": panels[rank]["successes"],
    "median_tail100_mm": panels[rank]["median_tail100_mm"],
    "p90_tail100_mm": panels[rank]["p90_tail100_mm"],
    "ever_reached_10mm": int(frame.loc[frame["rank"] == rank, "ever_reached_10mm"].sum()),
    "unsolved_qps": panels[rank]["unsolved_qps"],
    "rollouts_with_unsolved_qps": int((frame.loc[frame["rank"] == rank, "unsolved_steps"] > 0).sum())
    } for rank in RANKS]).set_index("rank")
display(summary.loc[[20, 40, 60, 80]].round(3))
best_count = int(summary.successes.max())
best_ranks = summary.index[summary.successes == best_count].tolist()
print(f"Rank 60: {summary.loc[60, 'successes']:.0f}/100 strict successes.")
print(f"Highest observed success: {best_count}/100, ranks {best_ranks}. Exploratory comparison; no held-out selection claim.")
fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
axes = axes.ravel()
axes[0].plot(RANKS, summary.successes, color=BLUE, marker=".")
axes[0].scatter([60], [summary.loc[60, "successes"]], color=ORANGE, s=65, zorder=5)
rank_axis(axes[0], "Strict successes /100")
axes[0].set_ylim(0, 105)
axes[0].set_title("Sustained success")
for ax in axes[1:3]:
    ax.plot(RANKS, summary.median_tail100_mm, color=BLUE, marker=".", label="Median")
    ax.plot(RANKS, summary.p90_tail100_mm, color=ORANGE, marker=".", label="P90")
    ax.axhline(10, color=GRAY, linestyle=":")
    rank_axis(ax, "Final-100 mean error (mm)")
    ax.set_ylim(bottom=0)
axes[1].set_title("Across-seed error distribution")
axes[1].legend(fontsize=9)
axes[2].set(title="Near the mean-error criterion", ylim=(0, 30))
axes[3].plot(RANKS, 100*summary.unsolved_qps/32000, color=ORANGE, marker=".", label="Control steps with unsolved QP")
axes[3].plot(RANKS, summary.rollouts_with_unsolved_qps, color=BLUE, marker=".", label="Rollouts with any unsolved QP")
rank_axis(axes[3], "Percentage (%)")
axes[3].set(title="QP failures under unchanged zero-input handling", ylim=(0, 105))
axes[3].legend(fontsize=9)
save_show(fig, "rank_success_errors.png")
display(HTML("<details><summary>All 61 rank summaries</summary>" + summary.round(3).to_html() + "</details>"))
'''), md("""## All paired seeds

The left panel shows final-100 mean error, with logarithmic color scaling to
retain detail near 10 mm across the full error range. The right panel shows the
complete strict-success test. Each row contains the same 100 seeds. The dashed
horizontal line marks rank 60; no seed is dropped because of failure.
"""), code('''error_grid = frame.pivot(index="rank", columns="seed", values="tail100_mm").loc[RANKS, SEEDS]
success_grid = frame.pivot(index="rank", columns="seed", values="strict").loc[RANKS, SEEDS]
fig, axes = plt.subplots(1, 2, figsize=(17, 11), constrained_layout=True)
extent = (210.5, 310.5, 80.5, 19.5)
minimum = max(float(error_grid.min().min()), .001)
im = axes[0].imshow(error_grid, aspect="auto", interpolation="nearest", extent=extent,
    norm=LogNorm(vmin=minimum, vmax=max(10, float(error_grid.max().max()))), cmap="viridis")
fig.colorbar(im, ax=axes[0], label="Final-100 mean fingertip error (mm; log color)", shrink=.7)
im2 = axes[1].imshow(success_grid.astype(int), aspect="auto", interpolation="nearest", extent=extent,
    cmap=ListedColormap(["#D55E00", "#0072B2"]), vmin=-.5, vmax=1.5)
bar = fig.colorbar(im2, ax=axes[1], ticks=[0, 1], shrink=.4)
bar.ax.set_yticklabels(["Failure", "Success"])
for ax, title in zip(axes, ["Error for every rank and seed", "Strict sustained success"]):
    ax.axhline(60, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Rollout seed", ylabel="Retained linear-subspace rank", title=title,
           xticks=list(range(211, 311, 10)), yticks=list(range(20, 81, 5)))
save_show(fig, "rank_seed_heatmaps.png")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
for rank, color in zip([20, 40, 60, 80], ["#CC79A7", "#009E73", BLUE, ORANGE]):
    values = np.sort(error_grid.loc[rank].to_numpy())
    for ax in axes:
        ax.step(np.r_[0, values], np.r_[0, np.arange(1, 101)/100], where="post",
                label=f"Rank {rank}" + (" (baseline)" if rank == 60 else ""), color=color)
for ax, title in zip(axes, ["Selected-rank error distributions", "Near-target detail"]):
    ax.set(xlim=(0, None), ylim=(0, 1.02), xlabel="Final-100 mean error (mm)", ylabel="Fraction of seeds", title=title)
    ax.axvline(10, color=GRAY, linestyle="--")
    ax.grid(alpha=.2)
axes[1].set_xlim(0, 30)
axes[0].legend()
save_show(fig, "rank_error_distributions.png")
'''), md("""## Repaired and regressed seeds relative to rank 60

A repaired seed fails at rank 60 and succeeds at the comparison rank. A regressed
seed succeeds at rank 60 and fails at the comparison rank. These are paired
outcomes, not independent evaluation samples at each rank.
"""), code('''base = success_grid.loc[60].to_numpy(dtype=bool)
paired_records = []
for rank in RANKS:
    current = success_grid.loc[rank].to_numpy(dtype=bool)
    repaired = np.asarray(SEEDS)[current & ~base].tolist()
    regressed = np.asarray(SEEDS)[~current & base].tolist()
    paired_records.append({"rank": rank, "repaired_count": len(repaired), "regressed_count": len(regressed),
        "net_success_change": len(repaired)-len(regressed), "repaired_seeds": repaired, "regressed_seeds": regressed})
paired = pd.DataFrame(paired_records).set_index("rank")
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
axes[0].plot(RANKS, paired.repaired_count, color=BLUE, label="Repaired")
axes[0].plot(RANKS, paired.regressed_count, color=ORANGE, label="Regressed")
axes[0].set_ylim(bottom=0)
axes[0].legend()
rank_axis(axes[0], "Number of paired seeds")
axes[1].plot(RANKS, paired.net_success_change, color=BLUE)
axes[1].axhline(0, color=GRAY, linewidth=1)
rank_axis(axes[1], "Strict success change versus rank 60")
save_show(fig, "rank_paired_changes.png")
display(HTML("<details><summary>Repaired and regressed seed lists for every rank</summary>" + paired.to_html() + "</details>"))
display(HTML("<details><summary>All 6,100 per-seed results</summary>" + frame[["rank", "seed", "strict", "tail100_mm", "unsolved_steps"]].round(3).to_html(index=False) + "</details>"))
'''), md("""## Computation time

Six workers, one numerical-library thread per worker, on a shared computer.
Rank 60 runs first, followed by the other ranks in ascending order. Rank and run
order are therefore confounded with changes in system load: these measurements
are descriptive, not an isolated hardware benchmark.

Timings discard the first five steps of every rollout, leaving **31,500 timed
steps per rank**. QP time includes construction and solve. Offline SVD and
simulator stepping are excluded. One-time candidate setup is reported separately.
Module means add to mean total time; medians and P99 values generally do not add.
All computation-time axes are linear and start at zero.
"""), code('''timing = pd.DataFrame([{"rank": rank, **row} for rank in RANKS for row in panels[rank]["timing"]["modules"]])
modules = panels[60]["timing"]["modules"]
module_names = [row["module"] for row in modules]
for rank in RANKS:
    rows = timing[timing["rank"] == rank]
    total_mean = rows.loc[rows.module == "Total online control", "mean_ms"].item()
    np.testing.assert_allclose(rows.loc[rows.module != "Total online control", "mean_ms"].sum(), total_mean)
fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
for ax, name in zip(axes.flat, module_names):
    values = timing[timing.module == name].set_index("rank").loc[RANKS]
    ax.plot(RANKS, values.mean_ms, color=BLUE, label="Mean")
    ax.plot(RANKS, values.median_ms, color=ORANGE, label="Median")
    rank_axis(ax, "ms per control step")
    ax.set(title=name, ylim=(0, None))
axes.flat[0].legend(fontsize=8)
setup = pd.DataFrame([{"rank": rank, **panels[rank]["timing"]["one_time_goal_setup_ms"]} for rank in RANKS]).set_index("rank")
ax = axes.flat[-1]
ax.plot(RANKS, setup["mean"], color=BLUE, label="Mean")
ax.plot(RANKS, setup["median"], color=ORANGE, label="Median")
rank_axis(ax, "ms per rollout setup")
ax.set(title="One-time target candidate setup", ylim=(0, None))
save_show(fig, "rank_module_timings.png")
total = timing[timing.module == "Total online control"].set_index("rank").loc[RANKS]
fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
for ax, column, title in zip(axes, ["mean_ms", "median_ms", "p99_ms"], ["Mean", "Median", "P99"]):
    ax.plot(RANKS, total[column], color=BLUE, marker=".")
    ax.scatter([60], [total.loc[60, column]], color=ORANGE, zorder=5)
    rank_axis(ax, "Total online time (ms / step)")
    ax.set(title=title, ylim=(0, None))
save_show(fig, "rank_total_timings.png")
display(total.loc[[20, 40, 60, 80], ["mean_ms", "median_ms", "p99_ms"]].round(3))
display(HTML("<details><summary>Module means, medians, P95 and P99 for all ranks</summary>" + timing.round(4).to_html(index=False) + "</details>"))
report = {"complete": True, "rank_count": len(RANKS), "seeds_per_rank": len(SEEDS), "rollouts": len(frame),
    "baseline_rank": 60, "best_observed_success_count": best_count, "best_observed_ranks": best_ranks,
    "summary": summary.reset_index().to_dict(orient="records"),
    "paired": paired.reset_index().to_dict(orient="records"),
    "total_online_timing": total.reset_index().to_dict(orient="records")}
experiment.write_json(RESULT_ROOT / "notebook_report.json", report)
print("Complete report saved; all 6,100 checkpoints verified.")
'''), md("""## Follow-up: why ranks 59 and 60 differ

The success difference is concentrated in four seeds. The across-seed medians
differ by only 0.017 mm, while these four rank-60 runs settle beyond the 10 mm
mean-error criterion:

| Seed | Rank 59 final-100 mean (mm) | Rank 60 final-100 mean (mm) |
| --- | ---: | ---: |
| 232 | 0.945 | 13.481 |
| 247 | 1.969 | 13.375 |
| 288 | 2.207 | 13.428 |
| 296 | 2.080 | 13.670 |

Four diagnostic rank-59 reruns reproduced every saved distance within 1e-10 m.
The rank-60 full traces from the preceding baseline study also match the sweep.
All four pairs settle in **chart 19**, with identical final terminal references
and candidate indices. Every QP in these eight traces has status `optimal`.
All first 59 basis columns match exactly between the rank-59 and rank-60 caches.

At seed 232's rank-60 steady state, holding the observed history, chart and
reference fixed, the rank-60 QP selects essentially zero first input. Rank 59
selects approximately [0.005, -0.498], which reduces actual error on the next
simulator step from 13.481 to 11.669 mm. The rank-60 extra coefficient is at its
upper bound (0.158608). Its predicted ten-step error falls to 0.669 mm even
though the first input leaves the robot stationary. Re-solving at that state
therefore sustains the same offset.

This directly demonstrates sensitivity of the feedback action and equilibrium
to the additional behavioral direction. The binary success criterion amplifies
the four changed equilibria into a four-point success-rate difference. It does
not establish that rank 59 is globally preferable or that its predictions are
uniformly accurate. The fixed-history probe's full open-loop predicted and actual
trajectories differ substantially for both ranks; the actual controller replans
at every step.

Diagnostics are saved under `diagnostics_rank59_60/` in the experiment directory,
including `diagnosis.json` and `fixed_history_probe.json`. Diagnostic runs do not
replace the original sweep or its timing measurements. Rank 60 was timed first
under different system load, so the timing gap is not an isolated rank comparison.

"""), md("""## Interpretation and reproduction

The success curve tests sensitivity to retained behavioral rank with all other
baseline choices fixed. A larger basis admits more behavioral directions but need
not improve closed-loop stabilization under the same regularization and coefficient
bounds. This study does not isolate those mechanisms or tune their weights.

Differences across adjacent ranks are descriptive on these reused 100 seeds. Any
rank selected from this sweep requires a separate evaluation before claiming a
generalization improvement over rank 60. Timing differences may also reflect system
load and the fixed run order.

When a rank has unsolved QPs, its closed-loop results include the baseline's
zero-input failure handling. Inspect the QP-failure panel alongside success and
error, rather than interpreting every failure to stabilize as the performance of
a successfully solved controller. Unsolved-step counts alone do not establish
mathematical infeasibility or identify a numerical failure mechanism.

To resume this exact launch protocol, run
`python experiments/routed_deepc_iid/linear_subspace_rank_100_v1/frozen_protocol/src/linear_subspace_rank_ablation.py --workers 6`
from the project root in the existing `gym-robotics-mujoco` environment. The frozen
runner resumes validated compact checkpoints and retains all original hash checks.
Its data and experiment-directory links reuse the existing files without copying
datasets. The live baseline in `src/` now belongs to a newer protocol.
The notebook defaults to `RUN_EXPERIMENTS=False` and refuses to present a partial
sweep as complete. The archived contextual affine study remains at
`archive/notebooks/ablation_affine_rank_contextual.ipynb`.
""")]

def build(output=OUTPUT, notebook_cells=None):
    notebook_cells = cells if notebook_cells is None else notebook_cells
    for cell in notebook_cells:
        if cell.cell_type == "code":
            compile(cell.source, str(output), "exec")
    nbf.write(nbf.v4.new_notebook(cells=notebook_cells, metadata={"kernelspec": {
        "display_name": "Python (gym-robotics-mujoco)", "language": "python", "name": "gym-robotics-mujoco"}}), output)
    print(output)


if __name__ == "__main__":
    build()
