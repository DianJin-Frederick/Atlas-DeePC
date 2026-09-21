"""Build the lambda_g=0 baseline notebook with a paired waypoint comparison."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/baselines/baseline_graph_routed_atlas_deepc.ipynb"
ARCHIVE = ROOT / "archive/notebooks/baseline_linear_iterative_n10_lamg005_before_waypoint_comparison.ipynb"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def build():
    previous = nbf.read(ARCHIVE, as_version=4)
    cells = [md(r"""# Baseline Atlas-DeePC

The current nominal baseline uses **$\lambda_g=0$**, 48 physical-configuration
charts, rank-60 uncentered linear subspaces, and iterative N=10 terminal-reference
augmentation. This notebook compares two algorithms on the same evaluation seeds:

1. **Atlas-DeePC without graph waypoints:** track the selected terminal reference
   from the first control step.
2. **Atlas-DeePC with graph waypoints:** track the next chart representative until
   the 7 cm terminal-mode latch, then track the selected terminal reference.

The main results and videos below use Clarabel. **Section 8** reports the matched
OSQP solver test, keeping the Clarabel results intact.

**Scope:** both algorithms retain graph-route-cost selection of the N=10 terminal
candidates. "Without graph waypoints" does not mean all graph information is
removed. Only the use of intermediate waypoint references changes. Because the
closed-loop trajectories differ, their subsequently selected terminal configurations
can differ too, despite using the same selection rule and candidate pool.

There is no contextual rescue, fitted dynamics, IK controller, added offline data,
or measurement noise. Every successful solve supplies an action through DeePC.
Unsolved QPs, if any, are reported rather than omitted. The old $\lambda_g=0.05$
notebook and its results remain archived; none of its outcome labels is reused here.
"""), code('''import json
import os
import sys
from pathlib import Path
from dataclasses import asdict
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display
get_ipython().run_line_magic("matplotlib", "inline")
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p / "src/baseline_atlas_deepc.py").exists())
sys.path.insert(0, str(ROOT / "src"))
import baseline_atlas_deepc as baseline
import routed_deepc_experiment as experiment
import baseline_waypoint_comparison as comparison
DATA_PATH, ATLAS_PATH = baseline.DATA_PATH, baseline.ATLAS_PATH
RESULT_ROOT = comparison.RESULT_ROOT
EVALUATION_SEEDS = list(range(211, 1211))  # Stop is exclusive; edit this range.
RUN_EVALUATION = False
WORKERS = 4
assert EVALUATION_SEEDS and len(set(EVALUATION_SEEDS)) == len(EVALUATION_SEEDS)
config = baseline.default_config()
assert config.lam_g == 0
if RUN_EVALUATION:
    comparison.run_panel(EVALUATION_SEEDS, RESULT_ROOT, WORKERS)
H = np.load(DATA_PATH, mmap_mode="r")
atlas = experiment.load_atlas(ATLAS_PATH)
METHODS = comparison.METHODS
COLORS = {"terminal": "#D55E00", "graph": "#0072B2"}
print("Baseline version:", baseline.BASELINE_VERSION)
print(f"Evaluation: {len(EVALUATION_SEEDS)} paired seeds, {min(EVALUATION_SEEDS)}--{max(EVALUATION_SEEDS)}")
'''), md(r"""## 1. Linear-Subspace DeePC and References

For each chart, collect its complete offline columns $X_k$ and retain the leading
60 left singular vectors **without mean subtraction**. The QP uses the current
chart's cached basis $B_k$:

$$\begin{bmatrix}u_{\rm ini}\\u\\y_{\rm ini}+\sigma\\y\end{bmatrix}=B_k g.$$

There is no affine offset, sum-to-one constraint, or coefficient-norm penalty.
The empirical coefficient bounds remain in place. All $U_p,U_f,Y_p,Y_f$ rows are
retained. Q, R, past-output slack and input-smoothness penalties remain unchanged:

$$Q=\operatorname{diag}(5000I_4,20I_2,10^5I_2,5000I_2),\quad R=10^{-4}I_2,$$
$$\lambda_g=0,\quad\lambda_{\rm ini}=10^5,\quad\lambda_{\Delta u}=5.$$

Both methods use two past steps, a ten-step prediction horizon, input limits
$[-1,1]$, and 320 control steps. The additional terminal cost is zero.

### Common Iterative Terminal Reference

1. Once the target is known, find the ten eligible offline configurations with
   closest fingertips. Exclude source episodes with original targets within 6 cm
   of the online target. Deduplicate candidates by chart.
2. Each step, locate the current chart and select the candidate chart with minimum
   directed route cost. Ties use fingertip error and source index.
3. The selected candidate supplies the posture; velocities are set to zero and
   fingertip coordinates to the requested target. Candidate selection continues
   throughout both rollouts, including after the waypoint method's 7 cm latch.

The one-time candidate search is shared in definition, not an online trajectory
retrieval. The graph provides references, never control actions.
"""), code('''from threadpoolctl import threadpool_limits
with threadpool_limits(limits=1):
    controller = baseline.BaselineAtlasDeePC(H, atlas, config)
assert len(controller.matrix_cache) == 48
assert all(not np.any(m["offset"]) and m["basis"].shape == (144, 60)
           and "g_sum" not in m for m, _ in controller.matrix_cache.values())
display(pd.Series({"charts": config.clusters, "linear-subspace rank": config.svd_rank,
    "lambda_g": config.lam_g, "candidate count": baseline.CANDIDATE_COUNT,
    "DeePC matrix": "current chart only", "terminal-reference selection": "iterative minimum route cost",
    "changed component between methods": "intermediate graph waypoint references only"}))
''')]
    # Retain the established configuration clouds, workspace and topology views.
    cells.extend(previous.cells[4:9])
    cells += [md("""## 3. Paired Results and Fairness

Strict success requires mean fingertip error below 10 mm and maximum error below
20 mm in **both final-30 and final-100 windows**. Every rollout retains all 320
steps. These repeatedly used evaluation seeds are not a new untouched test set.
"""), code('''payload = json.loads((RESULT_ROOT / "summary.json").read_text())
assert payload["version"] == comparison.VERSION
assert payload["signature"]["baseline_version"] == baseline.BASELINE_VERSION
assert payload["signature"]["config"] == asdict(config)
assert payload["signature"]["terminal_selection_uses_graph_in_both_methods"]
audit = payload["fairness_audit"]
assert audit["offline_stabilized_episode_count"] == 0
rows_by_mode, records, errors = {}, [], {}
for mode in METHODS:
    by_seed = {int(r["seed"]): r for r in payload["panels"][mode]["results"]}
    missing = sorted(set(EVALUATION_SEEDS)-set(by_seed))
    assert not missing, f"{mode}: missing {len(missing)} requested seeds. Enable RUN_EVALUATION."
    rows_by_mode[mode] = {seed: by_seed[seed] for seed in EVALUATION_SEEDS}
    for seed, row in rows_by_mode[mode].items():
        assert row["baseline_version"] == baseline.BASELINE_VERSION and row["lam_g"] == 0
        assert row["reference_mode"] == mode and row["graph_waypoints_enabled"] == (mode == "graph")
        assert row["matrix_geometry"] == "linear" and row["candidate_count"] == 10 and row["steps"] == 320
        assert row["minimum_selected_goal_source_separation"] >= .06
        assert not row["fitted_dynamics_model"] and not row["feedback_fallback"]
        path = RESULT_ROOT / mode / f"seed{seed}_control_results.npz"
        with np.load(path, allow_pickle=True) as trace:
            target = trace["final_goal"].reshape(-1)[6:8]
            np.testing.assert_allclose(trace["distance"], np.linalg.norm(trace["y"][:, 6:8]-target, axis=1))
            error = 1000*trace["distance"]
        errors[mode, seed] = error
        records.append({"mode": mode, "method": METHODS[mode], "seed": seed,
            "strict": bool(row["success_30"] and row["success_100"]),
            "mean_error_mm": 1000*row["mean_last_100"], "minimum_mm": float(error.min()),
            "tail_time_below10_pct": 100*np.mean(error[-100:] < 10),
            "unsolved_qps": row["unsolved_steps"]})
for seed in EVALUATION_SEEDS:
    np.testing.assert_array_equal(rows_by_mode["terminal"][seed]["exact_target_xy"], rows_by_mode["graph"][seed]["exact_target_xy"])
frame = pd.DataFrame(records)
summary = pd.DataFrame([{"Method": METHODS[mode], "Successes": int((g:=frame[frame['mode']==mode]).strict.sum()),
    "Trials": len(g), "Success rate (%)": 100*g.strict.mean(),
    "Median final-100 mean (mm)": g.mean_error_mm.median(),
    "P90 final-100 mean (mm)": g.mean_error_mm.quantile(.9),
    "Mean final-100 time below 10 mm (%)": g.tail_time_below10_pct.mean(),
    "Unsolved QPs": int(g.unsolved_qps.sum())} for mode in METHODS]).set_index("Method")
display(summary.round(3))
display(pd.Series({"offline seed range": str(audit["data_seed_range"]),
    "stabilized offline episodes": audit["offline_stabilized_episode_count"],
    "minimum selected source-target separation (m)": min(r["minimum_selected_goal_source_separation"] for rows in rows_by_mode.values() for r in rows.values()),
    "lambda_g, both methods": config.lam_g, "per-step full-library trajectory retrieval": "none"}))
paired = frame.pivot(index="seed", columns="mode", values="strict")
graph_only = paired.index[paired.graph & ~paired.terminal].tolist()
terminal_only = paired.index[paired.terminal & ~paired.graph].tolist()
display(pd.Series({"with-waypoints-only successes": len(graph_only),
    "without-waypoints-only successes": len(terminal_only),
    "net extra successes with waypoints": len(graph_only)-len(terminal_only)}))
display(HTML("<details><summary>Paired outcome changes</summary>" +
    pd.Series({"with-waypoints-only seeds": graph_only, "without-waypoints-only seeds": terminal_only}).to_frame().to_html() + "</details>"))
display(HTML("<details><summary>All per-seed results</summary>"+frame.round(3).to_html(index=False)+"</details>"))
summary.to_json(RESULT_ROOT / "comparison_report.json", orient="index", indent=2)
'''), code('''fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
for i, mode in enumerate(METHODS):
    rows = frame[frame['mode'] == mode]
    rate = 100*rows.strict.mean()
    axes[0].bar(i, rate, color=COLORS[mode], width=.6)
    axes[0].text(i, rate+2, f"{rows.strict.sum()}/{len(rows)}", ha="center")
    values = np.sort(rows.mean_error_mm)
    for ax in axes[1:]:
        ax.step(np.r_[0, values], np.r_[0, np.arange(1, len(values)+1)/len(values)],
                where="post", color=COLORS[mode], label=METHODS[mode])
axes[0].set(xticks=[0,1], xticklabels=["Without graph\\nwaypoints", "With graph\\nwaypoints"],
            ylim=(0,110), ylabel="strict success (%)")
for ax in axes[1:]:
    ax.axvline(10, color="#667085", linestyle="--")
    ax.set(xlim=(0,None), ylim=(0,1.02), xlabel="final-100 mean fingertip error (mm)", ylabel="fraction of seeds")
axes[1].set_title("Full error distribution")
axes[2].set(xlim=(0,30), title="Near the 10 mm mean-error criterion")
axes[1].legend(fontsize=8)
for ax in axes: ax.grid(axis="y", alpha=.2)
fig.suptitle(f"Atlas-DeePC, lambda_g=0 | {len(EVALUATION_SEEDS)} paired seeds")
fig.savefig(RESULT_ROOT / "paired_success_and_error.png", dpi=160)
plt.show()
'''), md("""## 4. Computation Time by Module

The methods use matching workers and settings; method order alternates across
seeds. The first five steps per rollout are excluded. Timings are pooled over the
requested seeds. QP time includes construction, canonicalization and solve.
Offline SVD, simulator stepping, saving, and video rendering are excluded.

Stacked **means** add to the mean total. Median and P99 module times are shown
separately because percentiles do not add. Terminal-reference selection still
uses graph costs in both methods. These are shared-computer wall-clock timings,
not an isolated hardware benchmark.

In the no-waypoint variant, the small time recorded under "Graph routing" is
branch/reference bookkeeping, not a waypoint-route search. The common route-cost
terminal selection is timed separately under "Terminal-reference selection".
"""), code('''from baseline_1000_benchmark import summarize_timing
timings = {}
for mode in METHODS:
    if set(EVALUATION_SEEDS) == set(payload["seeds"]):
        timings[mode] = payload["panels"][mode]["timing"]
    else:
        timings[mode] = summarize_timing([RESULT_ROOT/mode/f"seed{s}_control_results.npz" for s in EVALUATION_SEEDS])
    assert timings[mode]["timed_steps"] == len(EVALUATION_SEEDS)*(config.steps-5)
timing = pd.DataFrame([{"mode": mode, "Method": METHODS[mode], **entry}
    for mode, data in timings.items() for entry in data["modules"]])
display(timing.pivot(index="module", columns="Method", values=["mean_ms", "median_ms", "p99_ms"]).round(4))
module_names = [x for x in timing.module.unique() if x != "Total online control"]
palette = ["#56B4E9", "#E69F00", "#0072B2", "#CC79A7", "#009E73", "#777777"]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.7), constrained_layout=True)
bottom = np.zeros(2)
for name, color in zip(module_names, palette):
    values = np.array([timing.loc[(timing['mode']==mode)&(timing.module==name), "mean_ms"].item() for mode in METHODS])
    axes[0].bar([0,1], values, bottom=bottom, color=color, label=name, width=.55)
    bottom += values
axes[0].set(xticks=[0,1], xticklabels=["Without waypoints", "With waypoints"], ylim=(0,bottom.max()*1.15), ylabel="mean time per step (ms)")
axes[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02,1))
overhead = [name for name in module_names if name != "QP construction and solve"]
y = np.arange(len(overhead))
for i, mode in enumerate(METHODS):
    values = [timing.loc[(timing['mode']==mode)&(timing.module==name), "mean_ms"].item() for name in overhead]
    axes[1].barh(y+(i-.5)*.35, values, height=.35, color=COLORS[mode], label=METHODS[mode])
axes[1].set(yticks=y, yticklabels=overhead, xlabel="mean time per step (ms)", title="Non-QP modules")
axes[1].set_ylim(-.6, len(overhead)+.7)
axes[1].legend(fontsize=8, loc="upper right")
fig.savefig(RESULT_ROOT/"paired_module_mean_times.png", dpi=160)
plt.show()
for statistic, label in [("median_ms","Median"),("p99_ms","P99")]:
    fig, axes = plt.subplots(1,2,figsize=(14,4.7),constrained_layout=True)
    for i, mode in enumerate(METHODS):
        values = [timing.loc[(timing['mode']==mode)&(timing.module==name), statistic].item() for name in ("Total online control","QP construction and solve")]
        bars = axes[0].bar(np.arange(2)+(i-.5)*.32,values,width=.32,color=COLORS[mode],label=METHODS[mode])
        axes[0].bar_label(bars,fmt="%.2f",padding=3)
        values = [timing.loc[(timing['mode']==mode)&(timing.module==name), statistic].item() for name in overhead]
        axes[1].barh(y+(i-.5)*.35,values,height=.35,color=COLORS[mode])
    axes[0].set(xticks=[0,1],xticklabels=["Total online control","QP construction and solve"],ylabel=f"{label} ms per step")
    axes[0].set_ylim(bottom=0)
    axes[0].margins(y=.2)
    axes[0].legend(fontsize=8)
    axes[1].set(yticks=y,yticklabels=overhead,xlabel=f"{label} ms per step",title="Non-QP modules")
    fig.savefig(RESULT_ROOT/f"paired_module_{statistic}.png",dpi=160)
    plt.show()
display(pd.DataFrame({METHODS[mode]: data["one_time_goal_setup_ms"] for mode,data in timings.items()}).rename_axis("One-time candidate setup (ms)"))
'''), md("""## 5. Paired Rollout Videos

Each pair uses the same seed and target. Videos replay saved actions, with measured
outputs checked against the numerical traces. Outcome labels use the complete
320-step run, not the displayed video duration. These selected examples are
diagnostic illustrations; the full evaluation above determines the success rate.
"""), code('''import base64
from baseline_waypoint_media import prepare_media, GALLERY_SEEDS, DASHBOARD_SEEDS
from render_routed_deepc_dashboard import file_sha256
PREPARE_MEDIA = False
if PREPARE_MEDIA:
    prepare_media(RESULT_ROOT)
MEDIA_DIR = RESULT_ROOT / "videos"
def media_uri(path, mime):
    return "data:"+mime+";base64,"+base64.b64encode(Path(path).read_bytes()).decode("ascii")
def show_pair(seed, dashboard=False):
    cards = []
    for mode in METHODS:
        row = rows_by_mode[mode][seed]
        name = f"{mode}_seed{seed}" + ("_025x" if dashboard else "_replay")
        video = MEDIA_DIR / (name+".mp4")
        metadata = json.loads((MEDIA_DIR/(name+".json")).read_text())
        trace_path = RESULT_ROOT/mode/f"seed{seed}_control_results.npz"
        assert metadata["source_result_sha256"] == file_sha256(trace_path)
        assert metadata["lam_g"] == 0 and metadata["reference_mode"] == mode
        success = bool(row["success_30"] and row["success_100"])
        status = "STRICT SUCCESS" if success else "STRICT FAILURE"
        color = "#166534" if success else "#b91c1c"
        poster = ""
        detail = f"Final-100 mean: {1000*row['mean_last_100']:.3f} mm | lambda_g=0"
        if dashboard:
            assert metadata["play_speed"] == .25
            assert metadata["dashboard_renderer_sha256"] == file_sha256(ROOT/"src/render_routed_deepc_dashboard.py")
            detail += f" | 0.25x | {metadata['displayed_control_steps']}/{metadata['saved_control_steps']} steps shown"
            poster_path = MEDIA_DIR/(name+"_poster.png")
            poster = f' poster="{media_uri(poster_path, "image/png")}"'
        else:
            assert metadata["maximum_replay_output_error"] <= 1e-8
        cards.append(f'<div style="background:#fff;color:#111827;padding:10px;border:1px solid #d1d5db;min-width:0">'
            f'<h4 style="margin:0 0 6px">Seed {seed}: {METHODS[mode]}</h4>'
            f'<p style="margin:0 0 8px;color:{color}">{status}</p><p style="font-size:12px">{detail}</p>'
            f'<video controls preload="metadata"{poster} src="{media_uri(video,"video/mp4")}" style="width:100%;display:block"></video></div>')
    display(HTML('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(480px,100%),1fr));gap:12px;margin-bottom:16px">'+''.join(cards)+'</div>'))
for seed in GALLERY_SEEDS:
    if seed in EVALUATION_SEEDS:
        show_pair(seed)
'''), md("""## 6. Slowed Control and Chart Diagnostics

The 0.25x dashboards show the arm, DeePC prediction, active reference, measured
charts, true fingertip error, and inputs. A successful rollout may have its settled
tail omitted for display only; failures retain the full rollout so late drift is
visible. Playback speed follows the existing dashboard convention (20 control
samples per second at 1x), not a claim about real-time solver execution.
"""), code('''for seed in DASHBOARD_SEEDS:
    if seed in EVALUATION_SEEDS:
        show_pair(seed, dashboard=True)
'''), md("""## 7. Reproduction and Version Boundary

Edit `EVALUATION_SEEDS` at the top and enable `RUN_EVALUATION` to run both methods
on another range. Results resume per seed and incompatible caches are rejected.
`PREPARE_MEDIA` regenerates the gallery from the saved numerical traces.

The nominal `baseline.default_config()` now sets lambda_g=0. The default
`BaselineAtlasDeePC.run(...)` uses graph waypoints; pass `reference_mode="terminal"`
to omit intermediate waypoints while retaining N=10 route-cost goal selection.
The historical lower-level `RoutedDeePCConfig` default remains lambda_g=0.05 to
avoid silently changing archived experiments that explicitly use it.

The new result root is `baseline.RESULT_ROOT`; old results and the previous
baseline notebook are archived, not relabeled. The separate comparison notebooks
and the rank-ablation experiments are not rerun or mixed into this report.
""")]
    from baseline_osqp_notebook import notebook_cells as osqp_cells
    cells.extend(osqp_cells())
    for cell in cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            compile(cell.source, str(OUTPUT), "exec")
    notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {
        "display_name": "Python (gym-robotics-mujoco)", "language": "python", "name": "gym-robotics-mujoco"}})
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
