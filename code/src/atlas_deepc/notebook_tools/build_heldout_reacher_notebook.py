"""Build the main 500-seed, mixed-chart-count Reacher evaluation notebook."""

import shutil

import nbformat as nbf

from atlas_deepc.benchmarks import reacher_mixed_k as study

NOTEBOOK = study.ROOT / "notebooks/reacher/baselines/baseline_graph_routed_atlas_deepc.ipynb"
ARCHIVE = study.ROOT / "archive/notebooks/baseline_heldout500_k48_before_mixed_k.ipynb"


def matched_guidance_cells():
    cells = [nbf.v4.new_markdown_cell(r"""## Matched-Atlas Guidance: Seven Figure Seeds

Run **only seven guided rollouts**, with **p=8, K=64, rank=53, T_sim=300**:
**10206, 10349, 10144, 10445, 10330, 10255, 10449**. Seed 10449 is the
matched-atlas success-to-failure regression example. Reuse the exact frozen atlas
and controller weights of the unguided baseline; do not recluster or retune.
The existing unguided traces provide the paired comparison and are not rerun.
All other settings remain unchanged: full-window projection selection, Clarabel,
12 zero-input warmup steps, and chordal graph costs.

This matches K and rank, but tests the **guidance package**, including iterative
N=10 terminal selection, not waypoints alone. These figure-selected seeds are
illustrative and must not be used as an unbiased success-rate estimate.

Run the setup cell above, then set `RUN_MATCHED_GUIDANCE = True` below and run
this section only. Wait until the Select-DPC timing run finishes. Full trajectories
are saved separately in `experiments/reacher/figure_guidance_p8_k64_r53_t300_v1`.
Completed seeds are reused after interruption; the 500-seed results are untouched.
"""), nbf.v4.new_code_cell('''from atlas_deepc.ablations import figure_guidance_matched as matched_guidance
importlib.reload(matched_guidance)
RUN_MATCHED_GUIDANCE = False
print("Figure seeds:", matched_guidance.SEEDS)
print("Guided traces:", matched_guidance.RESULT_ROOT / "output8/graph")
if RUN_MATCHED_GUIDANCE:
    import signal, subprocess
    matched_process = subprocess.Popen(
        [sys.executable, "-u", "-m", "atlas_deepc.ablations.figure_guidance_matched"],
        cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, start_new_session=True)
    try:
        for line in matched_process.stdout:
            print(line, end="", flush=True)
        if matched_process.wait():
            raise RuntimeError("Matched guidance stopped; see output above. Completed seeds are retained.")
    except KeyboardInterrupt:
        try:
            os.killpg(matched_process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            matched_process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(matched_process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            matched_process.wait()
        print("Stopped; rerun this cell to resume the seven-seed panel.")
    finally:
        matched_process.stdout.close()
else:
    print("No new rollouts launched. Enable RUN_MATCHED_GUIDANCE when ready.")
'''), nbf.v4.new_code_cell('''matched_rows = matched_guidance.paired_rows()
display(pd.DataFrame(matched_rows).reindex(columns=["seed", "unguided_success", "guided_success",
                   "unguided_final100_mm", "guided_final100_mm"]))
print("Saved paired results:", matched_guidance.RESULT_ROOT / "paired_results.json")
''')]
    for cell in cells:
        cell.metadata["study"] = "matched_guidance_figure_seeds"
    return cells


def append_matched_guidance():
    notebook = nbf.read(NOTEBOOK, as_version=4)
    notebook.cells = [c for c in notebook.cells
                      if c.metadata.get("study") != "matched_guidance_figure_seeds"]
    notebook.cells.extend(matched_guidance_cells())
    nbf.validate(notebook)
    temporary = NOTEBOOK.with_suffix(".pending.ipynb")
    nbf.write(notebook, temporary)
    temporary.replace(NOTEBOOK)
    return NOTEBOOK


def build():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md(r"""# Reacher Atlas-DeePC: 500-Seed Four-Method Evaluation

**500 evaluation seeds: 10001--10500. Four methods, 2,000 rollouts.**

Ranks were selected using **development seeds 211--310** at 320 steps. This
evaluation freezes those ranks and uses **T_sim = 300 controlled steps**.
The chart counts below were chosen by the user. Source development studies:
[K=64 and K=80/96](../ablations/ablation_linear_subspace_rank.ipynb).

| Output dimension | Graph guidance | Frozen rank | Charts |
|---|---|---:|---:|
| 8 | No | 53 | 64 |
| 8 | Yes | 57 | 96 |
| 10 | No | 64 | 80 |
| 10 | Yes | 62 | 64 |

Each case uses its own frozen development atlas. No reclustering or rank search
is performed here. Controller weights and reference policies remain unchanged.
These 500 seeds were already evaluated with K=48: they are disjoint from
development, but this is a **repeated evaluation**, not a new untouched test set.
Do not retune using these evaluation results. Since both chart counts and ranks
differ, the four-method comparison is not a controlled graph-only ablation.

## Protocol

The objective contains only tracking error, input effort, and initial-output
slack, with $\lambda_{\mathrm{ini}}=10^5$. There are no coefficient or input-change
penalties, coefficient boxes, affine charts, or extra terminal penalties.
Clarabel uses the existing solver settings. All coordinates are unstandardized.
Charts are selected by projection residual using the most recent **L=12**
complete input/output samples, with no coordinate truncation.

Unguided methods use a fixed nearest eligible offline terminal configuration.
Guided methods retain chordal graph costs divided by square-root transition
counts, graph waypoints, and iterative **N=10** terminal-candidate selection.
Thus the guidance comparison includes the existing terminal-reference policy
difference; it is not an isolated waypoint-only ablation.

Each seed uses the same Reacher reset across methods, 12 zero-input warmup
steps, then 300 controlled steps. Default resets cover the target disk but only
small initial joint-angle perturbations near zero; this is not a test over
arbitrary arm configurations. Evaluation seeds are disjoint from development
and offline episode seeds.

Strict success requires mean fingertip error below 10 mm and maximum error
below 20 mm in **both** the final 30 and final 100 steps. Error quantiles are
across rollout final-100 mean errors. Median rollout RMSE is the median of
500 individual RMSEs, each using all 300 controlled steps relative to the true
target, excluding warmup. Failed solves remain in the statistics.

## Run or Resume

Select the **gym-robotics-mujoco** kernel. Set `RUN_EXPERIMENTS = True` and Run All.
Runs are **serial with one numerical thread**, for comparable timings. Wait for
other heavy experiments to finish first. Interrupt
the run cell to stop; complete seed files are reused on the next run.
With the flag False, the notebook loads the saved protocol and displays progress
(or prepares a protocol if none exists). No test rollouts or videos run by default.

The shared controller now exposes only tracking, input effort and initial-slack
costs. Existing results and their frozen source signatures are retained unchanged.
Saved results can be viewed without revalidating them against edited source;
a new run still enforces source compatibility and may require a fresh result root.
"""), code('''from pathlib import Path
import os, sys
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src/atlas_deepc").is_dir())
sys.path.insert(0, str(ROOT / "src"))
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
import importlib
from atlas_deepc.benchmarks import reacher_heldout as shared_runner
from atlas_deepc.benchmarks import reacher_mixed_k as study
from atlas_deepc.reporting import current_reacher_baseline as visuals

# Reload dependencies first: an existing kernel may still hold the K=48 runner.
importlib.reload(shared_runner)
importlib.reload(study)
importlib.reload(visuals)

RUN_EXPERIMENTS = False
PLOT_CHART_CLOUDS = False
RENDER_MEDIA = False
RESULT_ROOT = study.RESULT_ROOT
REPORT_ROOT = RESULT_ROOT / "baseline_report"
T_SIM = study.CONTROL_STEPS
assert T_SIM == 300
saved_manifest = RESULT_ROOT / "manifest.json"
protocol = (json.loads(saved_manifest.read_text())
            if saved_manifest.exists() and not RUN_EXPERIMENTS else study.prepare())
print(f"T_sim={T_SIM}; seeds {study.SEEDS[0]}--{study.SEEDS[-1]}; results: {RESULT_ROOT}")
display(pd.DataFrame(study.report()["progress"]))
'''), code('''if RUN_EXPERIMENTS:
    import signal, subprocess
    child_env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "atlas_deepc.benchmarks.reacher_mixed_k"],
        cwd=ROOT, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, start_new_session=True)
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        if process.wait():
            raise RuntimeError("Evaluation stopped. See the error above; saved seeds are retained.")
    except KeyboardInterrupt:
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        print("Stopped; completed seeds remain saved. Run this cell again to resume.")
    finally:
        process.stdout.close()
else:
    print("No rollouts launched. Set RUN_EXPERIMENTS = True to run/resume.")
'''), md("""## Evaluation Results

The final table is generated only after all four 500-seed panels are complete.
Partial progress is labeled below. Success rates retain 0.2-percentage-point
resolution. The LaTeX fragment requires `booktabs` in the paper preamble.
"""), code('''report = study.report()
display(pd.DataFrame(report["progress"]))
if report["complete"]:
    columns = report["columns"]
    labels = [f"p={c['output_dimension']}, " + ("graph" if c["graph_guidance"] else "no graph") for c in columns]
    fields = ["charts", "rank", "control_steps", "success_rate", "median_error_mm", "p90_error_mm", "p95_error_mm", "p99_error_mm",
              "median_rollout_rmse_mm", "median_time_ms", "p95_time_ms", "p99_time_ms"]
    display(pd.DataFrame({label: {k: c[k] for k in fields} for label, c in zip(labels, columns)}))
    print((RESULT_ROOT / "table_heldout500.tex").read_text())
else:
    display(Markdown("**Incomplete evaluation. No final test-performance table yet.**"))
for c in report["columns"]:
    print(f"p={c['output_dimension']}, {c['reference_mode']}: {len(c['failed_seeds'])} failed seeds")
    print(c["failed_seeds"])
'''), md("""## Complete-Trajectory Physical Chart Clouds

Optional offline visualization of the frozen atlas. Blue shows the two past
poses and orange the ten future poses of each sampled trajectory. The dark-blue
representative is for visualization only; online selection uses projection
residuals. The same color palette is used for every chart.
"""), code('''if PLOT_CHART_CLOUDS:
    plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm"})
    for p, mode, rank in study.CASES:
        figure = visuals.chart_clouds(p, RESULT_ROOT, REPORT_ROOT / f"output{p}/{mode}",
                                     atlas_file=study.atlas_path(RESULT_ROOT, p, mode))
        plt.show()
        plt.close(figure)
'''), md("""## Success and Failure Replays

Optional: after evaluation, render the lowest-numbered success and failure seed
for each method (where available). These are post-evaluation illustrations,
not additional selection or tuning. Simulator replays use physical 1x speed;
diagnostic dashboards use 0.25x. Saved actions are replayed with 12 zero-input
warmup steps. No optimization is rerun to create the videos.
"""), code('''if RENDER_MEDIA and report["complete"]:
    from atlas_deepc.reporting import render_saved_rollout as replay
    from atlas_deepc.reporting import render_routed_deepc_dashboard as dashboard
    entries = []
    for c in report["columns"]:
        p, mode = c["output_dimension"], c["reference_mode"]
        failed = set(c["failed_seeds"])
        picked = ([next(s for s in study.SEEDS if s not in failed)] if len(failed) < 500 else [])
        picked += [min(failed)] if failed else []
        directory = RESULT_ROOT / f"output{p}/{mode}"
        panel = json.loads((directory / "summary.json").read_text())
        rows = {r["seed"]: r for r in panel["results"]}
        for seed in picked:
            source = directory / f"seed{seed}_control_results.npz"
            media_dir = REPORT_ROOT / f"videos/output{p}/{mode}"
            media_dir.mkdir(parents=True, exist_ok=True)
            simulator = media_dir / f"seed{seed}_simulator.mp4"
            movie = media_dir / f"seed{seed}_dashboard_025x.mp4"
            replay.render_saved_rollout(source, seed, simulator, fps=50, frame_stride=1,
                warmup_zeros=12, verification_path=simulator.with_suffix(".json"),
                output_from_obs=lambda obs, p=p: replay.pipe.reacher_output_from_obs(obs)[:p])
            dashboard.render_dashboard(source, study.atlas_path(RESULT_ROOT, p, mode), movie,
                fps=25, frame_stride=2, play_speed=.25, end_step=T_SIM,
                evaluation_label=f"500-seed evaluation | p={p} | K={c['charts']} | rank={c['rank']}")
            entries.append({"output_dimension": p, "mode": mode, "seed": seed,
                "success": seed not in failed, "final100_mm": 1000*rows[seed]["mean_last_100"],
                "simulator": str(simulator), "dashboard": str(movie)})
    visuals.gallery(entries, "simulator")
    visuals.gallery(entries, "dashboard")
elif RENDER_MEDIA:
    print("Finish the evaluation before selecting illustrative replays.")
'''), md("""## Saved Artifacts

- `manifest.json`: frozen seeds, ranks, configurations, data/atlas/source hashes.
- `frozen_source/`: controller and evaluation source snapshot.
- `output8/` and `output10/`: per-seed traces, graph/projection geometry, summaries.
- `evaluation_report.json`: completion state, metrics, failed seeds and rollout RMSEs.
- `table_heldout500.tex`: final four-column paper table, written only on completion.

Results are stored separately in `four_baselines_500_mixed_k_t300_v1`.
Development results and offline data remain intact. The previous K=48 notebook
is archived as `archive/notebooks/baseline_heldout500_k48_before_mixed_k.ipynb`.
Its raw 500-seed rollout data and generated videos were removed to free storage;
summary tables and provenance remain. Old raw-trace analyses require rerunning.
""")]
    if not ARCHIVE.exists():
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(NOTEBOOK, ARCHIVE)
    notebook = nbf.v4.new_notebook(cells=cells + matched_guidance_cells(), metadata={"kernelspec": {
        "display_name": "gym-robotics-mujoco", "language": "python", "name": "gym-robotics-mujoco"}})
    nbf.validate(notebook)
    temporary = NOTEBOOK.with_suffix(".pending.ipynb")
    nbf.write(notebook, temporary)
    temporary.replace(NOTEBOOK)
    return NOTEBOOK


if __name__ == "__main__":
    print(build())
