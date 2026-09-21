"""Migrate the primary baseline notebook without reusing legacy rollout claims."""

import shutil
from pathlib import Path

import nbformat as nbf
from baseline_notebook_layout import restore_media_sections
from baseline_success_rate import notebook_cells as success_rate_cells
from nearest_terminal_ablation_notebook import notebook_cells as nearest_ablation_cells


ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "notebooks/baselines/baseline_graph_routed_atlas_deepc.ipynb"
archive = ROOT / "archive/notebooks/baseline_affine_fixed_reference_before_linear_iterative.ipynb"
archive.parent.mkdir(parents=True, exist_ok=True)
if not archive.exists():
    shutil.copy2(path, archive)
old = nbf.read(archive, as_version=4)
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [md(r"""# Baseline Atlas-DeePC

The current baseline combines **uncentered linear-subspace charts** with the user's
**iterative terminal-reference augmentation, N=10**. The 48 physical charts, rank 60,
offline random-action library, DeePC objective, constraints, and 7 cm waypoint-to-terminal
switch are unchanged. Every control input comes from DeePC; there is no fitted dynamics,
contextual rescue, inverse-kinematics controller, or feedback fallback.

All new baseline runs use `src/baseline_atlas_deepc.py`. Historical affine/fixed-reference
experiments are preserved separately and are not evidence for this combined method.
The comparison notebook is maintained separately and is not modified here.
"""), code('''import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from IPython import get_ipython
get_ipython().run_line_magic("matplotlib", "inline")

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src" / "baseline_atlas_deepc.py").is_file())
assert (ROOT / "src/baseline_atlas_deepc.py").exists(), "Open from the project root"
sys.path.insert(0, str(ROOT / "src"))
import baseline_atlas_deepc as baseline
import routed_deepc_experiment as experiment

DATA_PATH, ATLAS_PATH = baseline.DATA_PATH, baseline.ATLAS_PATH
RESULT_ROOT = baseline.RESULT_ROOT
RESULT_ROOT.mkdir(parents=True, exist_ok=True)
EVALUATION_SEEDS = list(range(211, 1211))  # Python stop is exclusive; change this range.
assert EVALUATION_SEEDS and len(EVALUATION_SEEDS) == len(set(EVALUATION_SEEDS))
from baseline_notebook_media import ALL_SEEDS
DIAGNOSTIC_SEEDS = list(ALL_SEEDS)
RUN_EVALUATION = False
RUN_DIAGNOSTICS = False
OVERWRITE = False
EVAL_DIR = RESULT_ROOT / "evaluation_1000"
DIAGNOSTIC_DIR = RESULT_ROOT / "diagnostics"
config = baseline.default_config()
H = np.load(DATA_PATH, mmap_mode="r")
atlas = experiment.load_atlas(ATLAS_PATH)
if RUN_DIAGNOSTICS:
    baseline.run_panel(DIAGNOSTIC_SEEDS, DIAGNOSTIC_DIR, config, overwrite=OVERWRITE)
if RUN_EVALUATION:
    from baseline_1000_benchmark import run_panel as run_timed_baseline
    run_timed_baseline(EVALUATION_SEEDS, EVAL_DIR, config)
print("Baseline version:", baseline.BASELINE_VERSION)
print("Full evaluation requested:", len(EVALUATION_SEEDS), "seeds; enabled:", RUN_EVALUATION)
'''), md(r"""## 1. Local linear-subspace DeePC

For each chart, collect its complete offline columns $X_k$ and compute the rank-60
left SVD basis **without subtracting a mean**: $X_k=U_k\Sigma_k V_k^\top$.
The cached matrix fed to DeePC is $B_k=U_k(:,1:60)$, with zero affine offset:

$$\begin{bmatrix}u_{\mathrm{ini}}\\u\\y_{\mathrm{ini}}+\sigma\\y\end{bmatrix}=B_k g.$$

The matrix includes all $U_p,U_f,Y_p,Y_f$ rows. The empirical coefficient bounds
and regularization use the same rules as before, evaluated in the new coordinates.
There is no centering or sum-to-one constraint. Only the current chart's basis is used.

### Iterative terminal-reference augmentation
1. Once the target is known, find the 10 eligible offline configurations with nearest
   fingertip positions. The original 6 cm source-episode-target separation rule remains.
2. At every time step, locate the current physical chart and choose the candidate chart
   with smallest directed route cost from that current chart. Within a chart, keep the
   nearest fingertip candidate; ties use the offline index.
3. Construct the full terminal reference from that candidate's posture, zero velocity,
   and the requested fingertip target. Reconstruct the route using cached shortest paths.
4. Track the next chart's medoid until the fingertip comes within 7 cm; then permanently
   use the terminal-reference mode. **Candidate selection continues after this switch**, so
   the terminal posture is allowed to change online. No new goal-freezing rule is added.

The top-10 target-neighborhood search occurs once per rollout. Online reselection compares
only the candidate charts, not all Hankel columns. The graph supplies references, never actions.
"""), code('''from threadpoolctl import threadpool_limits
with threadpool_limits(limits=1):
    controller = baseline.BaselineAtlasDeePC(H, atlas, config)
chart0_matrix, chart0_columns = controller.matrix_cache[(0,)]
assert len(controller.matrix_cache) == 48
assert all(not np.any(matrix["offset"]) for matrix, _ in controller.matrix_cache.values())
assert all(matrix["basis"].shape[1] == 60 for matrix, _ in controller.matrix_cache.values())
display(pd.Series({
    "charts": config.clusters, "chart geometry": "uncentered linear subspace",
    "SVD rank": config.svd_rank, "candidate count": baseline.CANDIDATE_COUNT,
    "terminal selection": "minimum route cost, recomputed every step",
    "chart 0 matrix": str(chart0_matrix["basis"].shape),
    "chart 0 support columns": len(chart0_columns),
    "DeePC matrix": "current chart only", "contextual rescue": False,
}))
''')]
cells += [old.cells[i] for i in range(4, 9)]
cells += [md("""## 3. Current-version result and fairness audit

Only `linear_iterative_n10_v1` rollouts are accepted here. If the requested evaluation
panel has not been run, the notebook shows the explicitly labeled diagnostic set.
Neither old fixed-reference results nor the affine iterative experiment are loaded.
"""), code('''evaluation_runs = []
evaluation_path = EVAL_DIR / "summary.json"
if evaluation_path.exists():
    payload = json.loads(evaluation_path.read_text())
    assert payload["method"] == baseline.BASELINE_VERSION
    by_seed = {int(r["seed"]): r for r in payload["results"]}
    evaluation_runs = [by_seed[s] for s in EVALUATION_SEEDS if s in by_seed]
missing_seeds = sorted(set(EVALUATION_SEEDS) - {r["seed"] for r in evaluation_runs})
if missing_seeds:
    print(f"Requested evaluation is incomplete: {len(missing_seeds)} missing seeds. No full-panel success estimate is reported.")
    result_summary = json.loads((DIAGNOSTIC_DIR / "summary.json").read_text())
    panel_name = "Diagnostic validation only (not a success-rate benchmark)"
    runs = result_summary["results"]
else:
    result_summary = payload
    panel_name = "Requested baseline evaluation"
    runs = evaluation_runs
assert result_summary["method"] == baseline.BASELINE_VERSION
from dataclasses import asdict
assert result_summary["signature"]["config"] == {k: v for k, v in asdict(config).items() if k != "seed"}
assert all(r["baseline_version"] == baseline.BASELINE_VERSION and r["matrix_geometry"] == "linear" and r["candidate_count"] == 10 for r in runs)
assert all(r["dynamic_terminal_reference"] for r in runs)
audit = result_summary["fairness_audit"]
assert audit["offline_stabilized_episode_count"] == 0
assert all(r["minimum_selected_goal_source_separation"] >= config.goal_source_target_min_distance for r in runs)
assert all(not r["fitted_dynamics_model"] and not r["feedback_fallback"] for r in runs)
display(pd.Series({
    "displayed panel": panel_name,
    "offline seed range": str(audit["data_seed_range"]),
    "stabilized offline episodes": audit["offline_stabilized_episode_count"],
    "minimum source-target separation (m)": min(r["minimum_selected_goal_source_separation"] for r in runs),
    "per-step full-library retrieval": "none",
    "one-time target-conditioned state search": "top 10 eligible configurations",
}))
'''), md("""## 4. Baseline results and timing

Strict success requires mean error below 1 cm and maximum error below 2 cm over both
the final-30 and final-100 windows. The initial diagnostic panel is intentionally small
and includes previously investigated cases; it cannot establish a general success rate.
Timings wrap the complete online controller call; QP time includes problem construction
and solution. One-time goal-neighborhood setup is separate.
"""), code('''results = pd.DataFrame([{
    "seed": r["seed"], "strict success": bool(r["success_30"] and r["success_100"]),
    "final-30 (mm)": 1000*r["mean_last_30"], "final-100 (mm)": 1000*r["mean_last_100"],
    "minimum (mm)": 1000*r["minimum"],
    "goal sample changes": r["goal_sample_switches"],
    "online median (ms)": r["median_control_compute_time_ms"],
    "QP median (ms)": r["median_solve_time_ms"],
    "goal selection median (ms)": r["median_goal_selection_time_ms"],
    "one-time goal setup (ms)": r["goal_setup_time_ms"],
    "unsolved steps": r["unsolved_steps"],
} for r in runs])
print(panel_name)
display(HTML("<details><summary>Show per-seed results</summary>"
             + results.style.format({c: "{:.3f}" for c in results if "(mm)" in c or "(ms)" in c}).to_html()
             + "</details>"))
if not missing_seeds:
    print(f"Strict success: {results['strict success'].sum()}/{len(results)}")
failures = results.loc[~results["strict success"]].copy()
failures["failure type"] = np.select(
    [failures["minimum (mm)"] < 10, failures["minimum (mm)"] < 20],
    ["Failed strict test; minimum error <1 cm", "Failed strict test; 1 cm <= minimum error <2 cm"],
    default="Failed strict test; minimum error >=2 cm")
display(failures[["seed", "failure type", "minimum (mm)", "final-30 (mm)", "final-100 (mm)"]])
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
axes[0].scatter(results["seed"], results["final-30 (mm)"], s=35,
                c=np.where(results["strict success"], "#0072B2", "#D55E00"))
axes[0].axhline(10, color="black", linestyle="--", label="mean threshold (10 mm)")
axes[0].set(xlabel="seed", ylabel="final-30 mean error (mm, log scale)", yscale="log")
axes[0].legend(frameon=False)
failure_counts = pd.Series({
    "Passed strict test": int(results["strict success"].sum()),
    "Failed: min. error <1 cm": int((failures["minimum (mm)"] < 10).sum()),
    "Failed: 1 <= min. error <2 cm": int(((failures["minimum (mm)"] >= 10) & (failures["minimum (mm)"] < 20)).sum()),
    "Failed: min. error >=2 cm": int((failures["minimum (mm)"] >= 20).sum()),
})
axes[1].bar(failure_counts.index, failure_counts.values, color=["#0072B2", "#D55E00", "#E69F00", "#b91c1c"])
axes[1].set_ylabel("number of displayed rollouts")
axes[1].tick_params(axis="x", rotation=20)
for i, value in enumerate(failure_counts):
    axes[1].text(i, value + 0.1, str(value), ha="center")
for ax in axes:
    ax.grid(alpha=0.2, axis="y")
fig.suptitle(panel_name)
fig.savefig(RESULT_ROOT / "current_baseline_results.png", dpi=160)
plt.show()
'''), md("""## 5. Current-baseline routing videos

These dashboards are generated only from the new combined baseline's saved diagnostic
rollouts. The selected goal-chart curve can change over time. Playback is **0.25x**;
the full 320-step rollout is retained so late drift is not hidden. Old videos are in the
archived notebook and are not shown as current-baseline demonstrations.
"""), code('''from render_routed_deepc_dashboard import render_dashboard
VIDEO_SEEDS = [222, 277]
RENDER_VIDEOS = False
VIDEO_DIR = RESULT_ROOT / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
for seed in VIDEO_SEEDS:
    trace = DIAGNOSTIC_DIR / f"seed{seed}_control_results.npz"
    video = VIDEO_DIR / f"seed{seed}_025x.mp4"
    if RENDER_VIDEOS:
        render_dashboard(trace, ATLAS_PATH, video, play_speed=0.25, hold_seconds=0,
                         evaluation_label="Reacher diagnostic")
    if video.exists():
        metadata = json.loads(video.with_suffix(".json").read_text())
        from render_routed_deepc_dashboard import file_sha256
        assert metadata["source_result_sha256"] == file_sha256(trace)
        poster = video.with_name(video.stem + "_poster.png")
        display(HTML(f'<h3>Seed {seed}: linear charts + iterative N=10</h3><video src="{video.resolve()}" poster="{poster.resolve()}" controls preload="metadata" style="width:100%;max-width:1280px"></video>'))
    else:
        print(f"Seed {seed}: enable RENDER_VIDEOS to render its current-baseline dashboard.")
'''), md("""## 6. Reproduction and version boundary

Change `EVALUATION_SEEDS` and enable `RUN_EVALUATION` near the top to run this baseline
on any seed range. Compatible results resume per seed; incompatible or unversioned
rollout caches are rejected. Use `BaselineAtlasDeePC` or `baseline_atlas_deepc.run_panel`
in new baseline experiments, including externally maintained comparison jobs.

The legacy lower-level `routed_deepc_experiment.run_routed_deepc` is intentionally kept
backward-compatible for archived affine/fixed-reference and contextual experiments.
It is not the current baseline entry point. Cross-task and contextual controllers are
not silently changed. The archived baseline notebook preserves the old figures and videos.

This combination has not inherited the 89.2% pure-subspace-only or 91.4% affine-iterative
success claims. Its large-scale comparison is a separate job. N=10 and rank 60 were chosen
using earlier experiments, and diagnostic seeds shown here are not untouched evaluation.
""")]
cells = restore_media_sections(cells)
test_section_index = next(i for i, cell in enumerate(cells) if cell.source.startswith("## 5. Atlas routing replays"))
cells[test_section_index:test_section_index] = success_rate_cells() + nearest_ablation_cells()
for cell in cells:
    if cell.cell_type == "code":
        compile(cell.source, str(path), "exec")
        cell.outputs = []
        cell.execution_count = None
notebook = nbf.v4.new_notebook(cells=cells, metadata=old.metadata)
nbf.write(notebook, path)
print(path)
