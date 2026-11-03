"""Four baseline cases with full-window projection selection and chordal routing."""

from dataclasses import replace
import json
from pathlib import Path
import shutil

from atlas_deepc.ablations import basic_four_cases_osqp as four
from atlas_deepc.ablations import four_baseline_chordal as chordal
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = four.ROOT
RESULT_ROOT = ROOT / "experiments/reacher/four_baselines_projection_chordal_lamg0_clarabel_v1"
SEEDS = list(range(211, 311))
TAG = "four_baselines_projection_chordal"


def prepare(root=RESULT_ROOT, seeds=SEEDS):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = replace(four.experiment.default_config(), lam_g=0., svd_rank=60)
    signatures = {}
    for p, mode in four.CASES:
        _, atlas_path = four.verify_source(p, mode)
        destination = root / f"output{p}/behavioral_atlas.npz"
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            shutil.copy2(atlas_path, destination)
        assert four.file_digest(destination) == four.file_digest(atlas_path)
        signature = four.experiment.execution_signature(
            config, four.DATA_PATH, destination, p, mode,
            chart_selection_method="full_window_projection", warmup_steps=12)
        previous = json.loads((chordal.RESULT_ROOT / f"output{p}/{mode}/manifest.json").read_text())
        for key in ("config", "source_sha256", "data_sha256", "atlas_sha256", "solver",
                    "standardize", "coefficient_bounds_enabled", "terminal_reference_policy"):
            assert signature[key] == previous[key], f"Unexpected baseline change: {key}"
        signatures[f"{p}_{mode}"] = signature
    source_paths = [Path(__file__), Path(four.__file__),
                    Path(__file__).with_name("chordal_graph_cost.py")]
    signature = {"seeds": list(seeds), "case_signatures": signatures,
                 "sources": {str(p.relative_to(ROOT)): four.file_digest(p) for p in source_paths},
                 "rank": 60, "lambda_g": 0, "solver": "CLARABEL", "workers": 1,
                 "numerical_threads": 1, "chart_selection": "full_window_projection",
                 "warmup_steps": 12, "graph_cost": "chordal / sqrt(transition_count)",
                 "comparison_caveat": "Earlier centroid-selection controls use two warmup steps, not twelve."}
    manifest = root / "experiment_manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == signature, "Incompatible experiment root"
    else:
        four.save_json(manifest, signature)
        for path in source_paths:
            destination = root / "frozen_source" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def verify(root=RESULT_ROOT, seeds=SEEDS):
    root = Path(root)
    records = []
    with threadpool_limits(limits=1):
        for p, mode in four.CASES:
            directory = root / f"output{p}/{mode}"
            panel = json.loads((directory / "summary.json").read_text())
            assert panel["seeds"] == list(seeds)
            with np.load(directory / "selection_geometry.npz") as geometry:
                bases, chart_ids = geometry["bases"], geometry["chart_ids"]
                assert bases.shape == (48, (2+p)*12, 60)
                np.testing.assert_array_equal(geometry["ranks"], np.full(48, 60))
                np.testing.assert_allclose(bases.transpose(0, 2, 1) @ bases,
                                           np.broadcast_to(np.eye(60), (48, 60, 60)), atol=1e-12)
            maximum_residual_difference = 0.
            for row in panel["results"]:
                assert row["lam_g"] == 0 and row["solver"] == "CLARABEL"
                assert row["localization_history_length"] == 12
                assert row["chart_selection_method"] == "full_window_projection"
                with np.load(directory / f"seed{row['seed']}_control_results.npz", allow_pickle=True) as trace:
                    z = trace["selection_windows"]
                    residuals = trace["selection_residuals"]
                    assert z.shape == (320, (2+p)*12) and residuals.shape == (320, 48)
                    # Reconstruct all later windows from the initial measured window and saved rollout.
                    u = np.concatenate((z[0, :24].reshape(12, 2), trace["u"]))
                    y = np.concatenate((z[0, 24:].reshape(12, p), trace["y"]))
                    np.testing.assert_array_equal(u[:12], np.zeros((12, 2)))
                    for t in range(320):
                        np.testing.assert_array_equal(z[t], np.r_[u[t:t+12].ravel(), y[t:t+12].ravel()])
                    direct = np.stack([np.linalg.norm(z-(z @ b) @ b.T, axis=1) for b in bases], axis=1)
                    np.testing.assert_allclose(residuals, direct, atol=1e-7, rtol=1e-7)
                    maximum_residual_difference = max(maximum_residual_difference, float(np.max(np.abs(residuals-direct))))
                    tolerance = 100*np.finfo(float).eps*np.maximum(1., np.linalg.norm(z, axis=1))
                    ties = residuals <= residuals.min(axis=1, keepdims=True)+tolerance[:, None]
                    np.testing.assert_array_equal(trace["chart"], chart_ids[np.argmax(ties, axis=1)])
                    np.testing.assert_array_equal(trace["selection_tie_count"], ties.sum(axis=1))
                    metrics = four.true_error_metrics(trace["distance"])
                    for key in ("success_30", "success_100", "mean_last_100"):
                        assert row[key] == metrics[key]
                    target = trace["final_goal"].reshape(-1)[6:8]
                    np.testing.assert_allclose(trace["distance"], np.linalg.norm(trace["y"][:, 6:8]-target, axis=1))
                    if mode == "terminal":
                        assert len(np.unique(trace["goal_index"])) == 1
                        np.testing.assert_array_equal(trace["ref"], np.tile(trace["final_goal"], (320, 1)))
            if mode == "graph":
                with np.load(directory / "graph_geometry.npz") as a, np.load(
                        chordal.RESULT_ROOT / f"output{p}/graph/graph_geometry.npz") as b:
                    for key in ("bases", "labels", "transition_counts", "chordal_cost", "chordal_distance"):
                        np.testing.assert_array_equal(a[key], b[key])
            records.append({"output_dimension": p, "reference_mode": mode,
                            "checked_rollouts": len(panel["results"]),
                            "maximum_projection_residual_difference": maximum_residual_difference,
                            "successes": sum(four.strict_success(r) for r in panel["results"]),
                            "unsolved_steps": sum(r["unsolved_steps"] for r in panel["results"])})
    report = {"complete": True, "seeds": list(seeds), "records": records,
              "verified": "All 48 projection residuals, measured history alignment, selected chart, graph geometry, and error metrics."}
    four.save_json(root / "verification.json", report)
    return report


def notebook_cells():
    import nbformat as nbf
    cells = [nbf.v4.new_markdown_cell(r"""## 9. Four-Baselines: Projection Selection and Chordal Routing

**Current experiment:** rank 60, $\lambda_g=0$, Clarabel, 48 frozen charts,
100 seeds (211--310), and 320 control steps. Offline data, clustering, linear
chart bases, Q/R weights, and reference policies are unchanged. There is no
standardization, affine constraint, or coefficient box constraint.

At every step, use the complete latest $L=12$ measured input-output samples,
$z_t\in\mathbb R^{(m+p)L}$, and choose
$$k_t=\arg\min_k\|z_t-\Phi_k\Phi_k^\top z_t\|_2.$$
All 120 ($p=8$) or 144 ($p=10$) raw coordinates enter this residual. Only the
chart bases are rank-truncated. No future observations or reference enter selection.
The DeePC initial-condition constraint still uses $T_{\rm ini}=2$ samples.

Graph-guided cases use chordal distance divided by $\sqrt{n_{kk'}}$ for both
waypoint paths and iterative $N=10$ terminal-candidate selection. No-graph cases
use one fixed nearest eligible terminal configuration.

**Warmup caveat:** these cases use 12 zero-input steps to obtain the measured
window. Earlier four-baselines centroid-selection runs use only two. Comparisons
to those earlier results are not an isolated chart-selection ablation.
Runs are serial with one numerical thread. Timings exclude offline construction
and the first five online steps. The QP module includes construction and solving.
"""), nbf.v4.new_code_cell('''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML, display
PROJECTION_ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src/atlas_deepc").is_dir())
sys.path.insert(0, str(PROJECTION_ROOT / "src"))
from atlas_deepc.ablations import four_baseline_projection_chordal as projection_experiment
RUN_PROJECTION_CHORDAL = False
if RUN_PROJECTION_CHORDAL:
    projection_experiment.run_all(update_notebook=False)
PROJECTION_RESULTS = projection_experiment.RESULT_ROOT
projection_audit = json.loads((PROJECTION_RESULTS / "verification.json").read_text())
assert projection_audit["complete"] and projection_audit["seeds"] == list(range(211, 311))
projection_table = json.loads((PROJECTION_RESULTS / "four_case_table.json").read_text())
projection_frame = pd.DataFrame(projection_table["columns"])
projection_labels = [f"p={r.output_dimension}, " + ("graph" if r.graph_guidance else "no graph") for r in projection_frame.itertuples()]
projection_frame.index = projection_labels
display(projection_frame[["success_rate", "median_error_mm", "p90_error_mm", "p95_error_mm", "p99_error_mm", "unsolved_steps"]].round(3))
display(HTML("<details><summary>Failed seeds</summary>" + projection_frame[["failed_seeds"]].to_html() + "</details>"))
print(projection_audit["verified"])
'''), nbf.v4.new_code_cell('''fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
colors = ["#D55E00", "#0072B2", "#CC79A7", "#009E73"]
bars = axes[0].bar(projection_labels, projection_frame.success_rate, color=colors)
axes[0].bar_label(bars, fmt="%.0f%%", padding=3)
axes[0].set(ylabel="Strict success (%)", ylim=(0, 105))
axes[0].tick_params(axis="x", labelrotation=15)
for (p, mode), label, color in zip(projection_experiment.four.CASES, projection_labels, colors):
    panel = json.loads((PROJECTION_RESULTS / f"output{p}/{mode}/summary.json").read_text())
    errors = np.sort([1000*r["mean_last_100"] for r in panel["results"]])
    axes[1].step(errors, np.arange(1, 101)/100, where="post", label=label, color=color)
axes[1].axvline(10, color="0.4", linestyle=":")
axes[1].set(xlabel="Final-100 mean true error (mm)", ylabel="Cumulative fraction", xlim=(0, None), ylim=(0, 1.02))
axes[1].legend()
for ax in axes:
    ax.grid(axis="y", alpha=.2)
fig.savefig(PROJECTION_RESULTS / "projection_chordal_success_error.png", dpi=160)
plt.show()
'''), nbf.v4.new_code_cell('''timing_rows = []
for (p, mode), label in zip(projection_experiment.four.CASES, projection_labels):
    panel = json.loads((PROJECTION_RESULTS / f"output{p}/{mode}/summary.json").read_text())
    timing_rows.extend({"case": label, **r} for r in panel["timing"]["modules"])
projection_timing = pd.DataFrame(timing_rows)
fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
for ax, metric, title in zip(axes, ["median_time_ms", "p99_time_ms"], ["Median online time (ms)", "P99 online time (ms)"]):
    bars = ax.bar(projection_labels, projection_frame[metric], color=colors)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set(ylabel=title, ylim=(0, projection_frame[metric].max()*1.18))
    ax.tick_params(axis="x", labelrotation=15)
    ax.grid(axis="y", alpha=.2)
fig.savefig(PROJECTION_RESULTS / "projection_chordal_timing.png", dpi=160)
plt.show()
display(projection_timing.pivot(index="case", columns="module", values="median_ms").round(3))
''')]
    for cell in cells:
        cell.metadata["tags"] = [TAG]
    return cells


def update_notebook():
    import nbformat as nbf
    from nbclient import NotebookClient
    backup = ROOT / "archive/notebooks/baseline_atlas_deepc_no_standardization_before_projection_chordal_20261012.ipynb"
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(chordal.NOTEBOOK, backup)
    report = nbf.v4.new_notebook(cells=notebook_cells())
    NotebookClient(report, timeout=600, kernel_name="gym-robotics-mujoco",
                   resources={"metadata": {"path": str(chordal.NOTEBOOK.parent)}}).execute()
    notebook = nbf.read(chordal.NOTEBOOK, as_version=4)
    notebook.cells = [c for c in notebook.cells if TAG not in c.metadata.get("tags", [])] + report.cells
    nbf.validate(notebook)
    temporary = chordal.NOTEBOOK.with_suffix(".ipynb.tmp")
    nbf.write(notebook, temporary)
    temporary.replace(chordal.NOTEBOOK)


def run_all(root=RESULT_ROOT, seeds=SEEDS, update_notebook=True):
    root = Path(root)
    prepare(root, seeds)
    for p, mode in four.CASES:
        four.run_case(p, mode, list(seeds), root, "CLARABEL", graph_cost="chordal",
                      chart_selection_method="full_window_projection", warmup_steps=12)
    four.report(root, list(seeds), "CLARABEL")
    result = verify(root, seeds)
    if update_notebook:
        globals()["update_notebook"]()
    return result


if __name__ == "__main__":
    run_all()
