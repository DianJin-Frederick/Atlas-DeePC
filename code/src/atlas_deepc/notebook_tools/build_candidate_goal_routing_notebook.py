"""Build the reproducible candidate-goal routing ablation notebook."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from atlas_deepc.common.paths import project_root

from pathlib import Path

import nbformat as nbf


def build():
    cells = []

    def md(source):
        cells.append(nbf.v4.new_markdown_cell(source.strip()))

    def code(source):
        cells.append(nbf.v4.new_code_cell(source.strip()))

    md(r"""
# Candidate-Goal Graph Routing for Atlas-DeePC

This experiment compares baseline Atlas-DeePC with a terminal-configuration selector based on graph path cost. Both use the same frozen 48-chart physical atlas, cached rank-60 behavioral matrices, offline random-walk library, DeePC QP, input limits, 7 cm target-reference latch, 320 control steps, and strict success metric. There is no contextual atlas or contextual rescue.

The paired evaluation panel is **500 seeds, 211–710**. These seeds have appeared in prior experiments; this is a paired ablation, not a new untouched evaluation. Results appear after running the experiment cell. Existing results from other timing settings are not silently reused.
""")
    code(r"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import sys
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p / "src" / "atlas_deepc/atlas/candidate_goal_routing_experiment.py").exists()), None)
assert ROOT is not None, "Open this notebook from the atlas_deepc_reacher project"
sys.path.insert(0, str(ROOT / "src"))
from atlas_deepc.atlas import routed_deepc_experiment as baseline
from atlas_deepc.atlas import candidate_goal_routing_experiment as experiment
SEEDS = list(range(211, 711))  # Change this range to evaluate other seeds.
CANDIDATE_COUNTS = [10, 50, 100, 200, 400]
RUN_EXPERIMENTS = True
OVERWRITE = False  # True reruns all methods for fresh local timing.
DATA_PATH = ROOT / "data/reacher/select_iid_offline_hankel.npy"
RESULT_ROOT = ROOT / "experiments/reacher/routed_deepc_iid"
ATLAS_PATH = RESULT_ROOT / "clustering_jointconfig48_fresh211_310/behavioral_atlas.npz"
BASELINE_SUMMARY = RESULT_ROOT / "clustering_jointconfig48_baseline_cache/summary.json"
OUTPUT_DIR = RESULT_ROOT / "candidate_goal_routing_500"

source = json.loads(BASELINE_SUMMARY.read_text())
config = baseline.RoutedDeePCConfig(**source["config"])
assert config.clusters == 48 and config.svd_rank == 60
assert config.clustering_feature_mode == "plant_joint_configuration"
assert config.target_matrix_mode == "current" and config.formulation == "svd_current"
assert config.steps == 320
assert set(SEEDS).issubset({int(r["seed"]) for r in source["results"]}) or SEEDS != list(range(211, 711))
print(f"{len(SEEDS)} seeds: {SEEDS[0]}–{SEEDS[-1]}")
print("Candidate counts:", CANDIDATE_COUNTS)
display(pd.Series(vars(config), name="Shared controller settings"))
""")
    md(r"""
## 1. Goal candidates and repeated route selection

For target $p^*=(p_x,p_y)$, take the top-$N$ eligible offline configurations ranked by $\|p_i-p^*\|_2$. Eligibility uses the same baseline fairness rule: the source episode's own target must be at least 6 cm from the online target. This does not exclude offline fingertip positions near the online target.

The configuration for a Hankel column is taken at the **last sample of $Y_p$**, matching the state used to fit its physical chart label. The old baseline goal helper uses the first output sample; the unchanged baseline retains that convention. The candidate method aligns the candidate state with the stored label instead of introducing this temporal mismatch.

Let $\ell_i$ be each candidate's physical chart. The graph retains exactly the baseline directed edges and costs:

$$w_{ab}=\frac{\|\mu_a-\mu_b\|_2/\sqrt{d}}{\sqrt{n_{ab}}},\qquad n_{ab}\geq3.$$

Here $\mu$ denotes the standardized atlas feature center, $d$ its dimension, and $n_{ab}$ the observed offline transition count. At every step, localize the measured current chart $k_t$, then choose

$$i_t^*=\arg\min_{i\in\mathcal I_N(p^*)}D(k_t,\ell_i),$$

where $D$ is the sum of edge costs along a shortest directed path. Ties are broken by fingertip error, then offline sample index. Thus candidates in the same chart need only one comparison. Unreachable candidates have infinite cost; if all are unreachable, the experiment stops with an explicit error.

Copy the selected candidate's configuration channels, set its velocities to zero, and set its fingertip channels to the exact online target. This gives the full terminal reference. The reference has the baseline's approximate consistency: copied joint channels come from a nearby sampled position, while its position channels use the exact target. The candidate mismatch is reported below.

DeePC receives **only the cached full input-output basis of the current chart**. Before target-reference handoff it tracks the next route chart's medoid; after handoff it tracks the newly selected full terminal reference. The existing 7 cm latch selects which type of reference to use; it does **not freeze the selected candidate**. Candidate selection continues at every step, including near the target. No new dwell trigger, branch lock, or early termination is added.
""")
    md(r"""
## 2. Timing protocol

The graph is fixed, so all-pairs shortest paths are cached once, before any rollout. Online rerouting remains responsive to the measured current chart: it compares cached distances to candidate charts, chooses a goal, and reconstructs the selected route. It does not repeat a graph search unnecessarily.

We report separately:

- Shared offline SVD, graph-table preparation, and source-episode metadata loading. Source metadata is warmed once for both methods before rollout timing.
- Per-rollout goal setup: baseline branch construction, or candidate lookup and grouping.
- Per-step current-chart localization, goal selection, route reconstruction, matrix lookup, and DeePC QP solve.
- Other loop overhead, total computation, and goal setup amortized across the rollout.

Runs are sequential with one numerical-library thread. Method order rotates across seeds. Solver timing wraps the same `DeePCController.solve` call for every method, including its problem preparation. Environment stepping and file saving are outside control timing. Resume skips completed checkpoints in this experiment directory; use a new output directory or `OVERWRITE=True` for fresh machine timing.

The baseline uses its original Dijkstra routing implementation. Therefore routing-time savings combine cached shortest paths with the new goal-selection rule; they do not isolate an intrinsic speed advantage of the rule alone. The unchanged QP may dominate total time. Neither speedup nor success improvement is assumed.
""")
    code(r"""
if RUN_EXPERIMENTS:
    payload = experiment.run_panel(
        SEEDS, CANDIDATE_COUNTS, config, DATA_PATH, ATLAS_PATH, OUTPUT_DIR,
        overwrite=OVERWRITE, progress_every=10,
    )
else:
    payload = json.loads((OUTPUT_DIR / "summary.json").read_text())

METHODS = ["Baseline Atlas"] + [f"Candidate N={n}" for n in CANDIDATE_COUNTS]
rows = [r for r in payload["results"] if r["seed"] in SEEDS and r["method"] in METHODS]
frame = pd.DataFrame(rows)
assert not frame.duplicated(["method", "seed"]).any()
for method in METHODS:
    found = set(frame.loc[frame.method == method, "seed"])
    assert found == set(SEEDS), f"Incomplete panel: {method}, missing {sorted(set(SEEDS) - found)}"
frame["strict_success"] = frame.apply(experiment.strict_success, axis=1)
frame["amortized_control_ms"] = frame.mean_control_compute_time_ms + frame.goal_setup_time_ms / config.steps
display(pd.Series({k: v for k, v in payload["manifest"].items() if k.endswith("_s")}, name="Offline seconds"))
""")
    md("## 3. Paired success comparison")
    code(r"""
def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    half = z * np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return center-half, center+half

summary_rows = []
for method in METHODS:
    group = frame[frame.method == method]
    successes = int(group.strict_success.sum())
    lo, hi = wilson(successes, len(group))
    summary_rows.append({
        "method": method, "successes": successes, "rollouts": len(group),
        "success_pct": 100*successes/len(group), "CI_low_pct": 100*lo, "CI_high_pct": 100*hi,
        "median_tail100_mm": 1000*group.mean_last_100.median(),
        "mean_step_ms": group.mean_control_compute_time_ms.mean(),
        "median_rollout_median_step_ms": group.median_control_compute_time_ms.median(),
        "mean_amortized_step_ms": group.amortized_control_ms.mean(),
        "mean_goal_setup_ms": group.goal_setup_time_ms.mean(),
    })
summary = pd.DataFrame(summary_rows).set_index("method").loc[METHODS]
display(summary.round(3))

success = frame.pivot(index="seed", columns="method", values="strict_success").loc[SEEDS]
paired = []
for method in METHODS[1:]:
    repaired = success.index[~success["Baseline Atlas"] & success[method]].tolist()
    regressed = success.index[success["Baseline Atlas"] & ~success[method]].tolist()
    a, b = len(repaired), len(regressed)
    n = a+b
    p = 1.0 if n == 0 else min(1.0, 2*sum(math.comb(n, j) for j in range(min(a,b)+1)) / 2**n)
    paired.append({"method": method, "repaired": a, "regressed": b, "net": a-b,
                   "McNemar_p_unadjusted": p, "repaired_seeds": repaired, "regressed_seeds": regressed})
display(pd.DataFrame(paired).set_index("method"))
""")
    code(r"""
colors = ["#555555", "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), constrained_layout=True)
x = np.arange(len(METHODS))
axes[0].bar(x, summary.success_pct, color=[colors[i % len(colors)] for i in x])
axes[0].errorbar(x, summary.success_pct,
                yerr=np.vstack([summary.success_pct-summary.CI_low_pct, summary.CI_high_pct-summary.success_pct]),
                fmt="none", color="black", capsize=3)
axes[0].set(ylim=(0, 100), ylabel="Strict success (%)", title="Paired success and Wilson 95% CI")
for i, value in enumerate(summary.success_pct):
    axes[0].text(i, max(1, value-7), f"{value:.1f}%", ha="center", color="white", weight="bold")
for i, method in enumerate(METHODS):
    error = np.sort(1000*frame.loc[frame.method == method, "mean_last_100"])
    axes[1].plot(error, np.arange(1,len(error)+1)/len(error), label=method, color=colors[i % len(colors)])
axes[1].axvline(10, color="black", linestyle="--", alpha=.5)
axes[1].set(xlabel="Final-100 mean error (mm)", ylabel="Cumulative fraction", xlim=(0,None), ylim=(0,1))
axes[1].legend(fontsize=8)
axes[2].bar(x, summary.mean_amortized_step_ms, color=[colors[i % len(colors)] for i in x])
axes[2].set(ylabel="Mean ms / step, including goal setup / 320", ylim=(0,None), title="Total computation, linear scale")
for ax in (axes[0], axes[2]):
    ax.set_xticks(x, METHODS, rotation=30, ha="right")
for ax in axes:
    ax.grid(axis="y", alpha=.2)
fig.savefig(OUTPUT_DIR / "success_and_runtime.png", dpi=170)
plt.show()
""")
    md(r"""
## 4. Computational time by module

Stacked components are **means of per-rollout means**, with equal weight for every seed. They sum to the mean total control time; stacking separate medians would not preserve that identity. The separate median and p95 tables describe variation without treating medians as additive.
""")
    code(r"""
components = {
    "mean_localization_time_ms": "Current-chart localization",
    "mean_goal_selection_time_ms": "Candidate goal selection",
    "mean_routing_time_ms": "Route search / reconstruction",
    "mean_matrix_preparation_time_ms": "Behavioral matrix lookup",
    "mean_solve_time_ms": "DeePC solve call",
    "mean_other_time_ms": "Other control overhead",
}
times = frame.groupby("method")[list(components)].mean().reindex(METHODS)
display(times.rename(columns=components).round(4))
assert np.allclose(times.sum(axis=1), summary.mean_step_ms)
fig, axes = plt.subplots(1, 2, figsize=(15,5), constrained_layout=True)
times.rename(columns=components).plot.bar(stacked=True, ax=axes[0], color=["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#555555", "#D55E00"])
axes[0].set(ylabel="Mean computation time (ms / step)", ylim=(0,None), xlabel="", title="Full computation")
small = times.drop(columns="mean_solve_time_ms")
small.rename(columns=components).plot.bar(stacked=True, ax=axes[1], color=["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"])
axes[1].set(ylabel="Mean computation time excluding solve (ms / step)", ylim=(0,None), xlabel="", title="Routing and selection detail")
for ax in axes:
    ax.tick_params(axis="x", labelrotation=30)
    ax.legend(fontsize=8)
fig.savefig(OUTPUT_DIR / "module_runtime.png", dpi=170)
plt.show()
p95_columns = [f"p95_{key[:-1]}ms" for key in experiment.TIMING_KEYS]
display(frame.groupby("method")[p95_columns].mean().reindex(METHODS).round(4).rename_axis("Mean of per-rollout p95 times"))
display(summary[["mean_goal_setup_ms", "mean_step_ms", "mean_amortized_step_ms"]].round(4))
""")
    md(r"""
## 5. Candidate coverage and reference switching

Increasing $N$ can offer more reachable goal charts, but can also admit configurations farther from the target. A candidate in the current chart has zero graph path cost even if its posture is not the easiest physical target. Edge costs are empirical graph costs, not certificates of joint-limit feasibility or control effort.

The switching diagnostics test the consequence of recomputing the terminal configuration at every step, including after first entering the 1 cm region. This experiment does not hide later drift by stopping early.
""")
    code(r"""
diagnostics = ["goal_sample_switches", "goal_chart_switches", "goal_switches_after_first_reach",
               "candidate_unique_charts", "candidate_max_xy_error_mm"]
display(frame.groupby("method")[diagnostics].mean().reindex(METHODS).round(3))
assert (frame.minimum_selected_goal_source_separation >= config.goal_source_target_min_distance - 1e-12).all()
assert not frame.fitted_dynamics_model.any()
assert not frame.online_nearest_trajectory_retrieval.any()
print("Offline target-separation check passed for every selected goal, including later reselections.")
print("The target-conditioned candidate lookup occurs once per rollout; online comparisons use candidate chart IDs.")

FAILURE_SEED = 277
if FAILURE_SEED in SEEDS:
    display(frame.loc[frame.seed == FAILURE_SEED, ["method", "strict_success", "mean_last_30", "mean_last_100",
                                                    "goal_sample_switches", "goal_chart_switches"]])
    fig, axes = plt.subplots(2,1, figsize=(12,7), sharex=True, constrained_layout=True)
    for i, (method, n) in enumerate([(METHODS[0], None)] + [(f"Candidate N={n}", n) for n in CANDIDATE_COUNTS]):
        folder = "baseline" if n is None else f"candidate_{n}"
        with np.load(OUTPUT_DIR / folder / f"seed{FAILURE_SEED}.npz") as trace:
            axes[0].plot(1000*trace["distance"], label=method, color=colors[i % len(colors)])
            axes[1].step(np.arange(config.steps), trace["goal_chart"], where="post", label=method, color=colors[i % len(colors)])
    axes[0].axhline(10, color="black", linestyle="--", alpha=.5)
    axes[0].set(ylabel="Fingertip error (mm)", title=f"Seed {FAILURE_SEED}", ylim=(0,None))
    axes[0].legend(ncol=2)
    axes[1].set(xlabel="Control step", ylabel="Selected goal chart")
    plt.show()
""")
    md(r"""
## 6. Implementation validation

Before the 500-seed sweep, a full 320-step paired check was run on the previously failing **seed 277**. Baseline final-100 mean fingertip error was **58.574 mm**; candidate routing with **N=400** gave **10.394 mm**. Both failed the unchanged strict success criterion. This is a single-seed implementation check, not an estimate of success rate or speedup. The baseline error reproduced its saved original result to numerical precision.

Focused tests also verify cached paths against the baseline Dijkstra implementation, top-N configuration selection at the correct Hankel time index, deterministic within-chart tie breaking, and goal reselection as the current chart changes. Every analysis and plotting cell was checked against the small paired result set. The full 500-seed results are computed by Section 2's experiment cell.
""")
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3 (gym-robotics-mujoco)", "language": "python", "name": "gym-robotics-mujoco"}
    notebook.metadata.language_info = {"name": "python", "version": "3.10"}
    nbf.validate(notebook)
    path = project_root() / "notebooks/reacher/ablations/ablation_candidate_goal_routing.ipynb"
    nbf.write(notebook, path)
    print(path)


if __name__ == "__main__":
    build()
