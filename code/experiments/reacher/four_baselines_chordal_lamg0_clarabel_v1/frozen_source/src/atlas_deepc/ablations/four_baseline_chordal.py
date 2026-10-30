"""Chordal graph costs for the four rank-60, lambda_g=0 Clarabel baselines."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil

from atlas_deepc.ablations import basic_four_cases_osqp as four
from atlas_deepc.ablations.chordal_graph_cost import chordal_distances, graph_costs
from atlas_deepc.atlas.candidate_goal_routing_experiment import shortest_path_tables
import numpy as np

ROOT = four.ROOT
RESULT_ROOT = ROOT / "experiments/reacher/four_baselines_chordal_lamg0_clarabel_v1"
REFERENCE_ROOT = ROOT / "experiments/reacher/basic_four_cases_lamg0_clarabel_v1"
NOTEBOOK = ROOT / "notebooks/reacher/baselines/baseline_atlas_deepc_no_standardization.ipynb"
SEEDS = list(range(211, 311))
TAG = "four_baselines_chordal"


def prepare(root=RESULT_ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    references = {}
    config = replace(four.experiment.default_config(), lam_g=0., svd_rank=60)
    for p, mode in four.CASES:
        reference = REFERENCE_ROOT / f"output{p}/{mode}/manifest.json"
        prior = json.loads(reference.read_text())
        _, atlas = four.verify_source(p, mode)
        current = four.experiment.execution_signature(config, four.DATA_PATH, atlas, p, mode,
                    terminal_reference_policy=four.experiment.terminal_policy(mode))
        for key in ("source_sha256", "config", "data_sha256", "atlas_sha256", "solver",
                    "standardize", "coefficient_bounds_enabled", "terminal_reference_policy"):
            assert current[key] == prior[key], f"Baseline changed: {p}/{mode}/{key}"
        references[f"{p}_{mode}"] = four.file_digest(reference)
        destination = root / f"output{p}/behavioral_atlas.npz"
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            shutil.copy2(atlas, destination)
        assert four.file_digest(destination) == prior["atlas_sha256"]
    sources = {str(path.relative_to(ROOT)): four.file_digest(path) for path in
               (Path(__file__), Path(four.__file__), Path(__file__).with_name("chordal_graph_cost.py"))}
    signature = {"seeds": SEEDS, "rank": 60, "lambda_g": 0, "solver": "CLARABEL",
                 "workers": 1, "numerical_threads": 1, "sources": sources,
                 "reference_manifests": references,
                 "edge_cost": "||Phi_k Phi_k.T - Phi_j Phi_j.T||_F / sqrt(2 * n_kj)",
                 "no_graph_policy": "fixed_nearest_eligible_fingertip",
                 "graph_policy": "iterative_minimum_route_cost_N10"}
    manifest = root / "experiment_manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == signature, "Incompatible experiment root"
    else:
        four.save_json(manifest, signature)
        for relative in sources:
            destination = root / "frozen_source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)


def compare(root=RESULT_ROOT, seeds=SEEDS):
    root = Path(root)
    records, verification = [], []
    for p, mode in four.CASES:
        folder = root / f"output{p}/{mode}"
        old_folder = REFERENCE_ROOT / f"output{p}/{mode}"
        panel = json.loads((folder / "summary.json").read_text())
        old_panel = json.loads((old_folder / "summary.json").read_text())
        new_rows = {r["seed"]: r for r in panel["results"]}
        old_rows = {r["seed"]: r for r in old_panel["results"]}
        assert set(seeds) == set(new_rows) and set(seeds) <= set(old_rows)
        repaired, regressed = [], []
        step_differences = {k: 0 for k in ("chart", "goal_chart", "goal_index", "next_chart")}
        max_error_difference = 0.
        for seed in seeds:
            row, old = new_rows[seed], old_rows[seed]
            if four.strict_success(row) and not four.strict_success(old):
                repaired.append(seed)
            if four.strict_success(old) and not four.strict_success(row):
                regressed.append(seed)
            name = f"seed{seed}_control_results.npz"
            with np.load(folder / name, allow_pickle=True) as a, np.load(old_folder / name, allow_pickle=True) as b:
                metrics = four.true_error_metrics(a["distance"])
                for key in ("success_30", "success_100", "mean_last_100"):
                    assert row[key] == metrics[key]
                assert a["y"].shape == (320, p) and np.all(np.isfinite(a["distance"]))
                target = a["final_goal"].reshape(-1)[6:8]
                np.testing.assert_array_equal(target, b["final_goal"].reshape(-1)[6:8])
                np.testing.assert_allclose(a["distance"], np.linalg.norm(a["y"][:, 6:8]-target, axis=1))
                max_error_difference = max(max_error_difference, float(np.max(np.abs(a["distance"]-b["distance"]))))
                for key in step_differences:
                    step_differences[key] += int(np.count_nonzero(a[key] != b[key]))
                if mode == "terminal":
                    for key in ("y", "u", "ref", "chart", "goal_index", "distance"):
                        np.testing.assert_array_equal(a[key], b[key])
        geometry_audit = None
        if mode == "graph":
            with np.load(folder / "graph_geometry.npz") as geometry:
                bases = geometry["bases"]
                assert bases.shape == (48, (2+p)*12, 60)
                np.testing.assert_allclose(chordal_distances(bases), geometry["chordal_distance"], atol=1e-12)
                for k in range(48):
                    for j in range(k):
                        direct = np.linalg.norm(bases[k] @ bases[k].T-bases[j] @ bases[j].T, "fro")/np.sqrt(2)
                        np.testing.assert_allclose(direct, geometry["chordal_distance"][k, j], atol=1e-7)
                np.testing.assert_array_equal(np.isfinite(geometry["centroid_cost"]), np.isfinite(geometry["chordal_cost"]))
                atlas = four.experiment.routed.load_atlas(folder.parent / "behavioral_atlas.npz")
                np.testing.assert_array_equal(geometry["labels"], atlas["labels"])
                np.testing.assert_array_equal(geometry["transition_counts"], atlas["transition_counts"])
                old_dist, old_pred = shortest_path_tables(atlas, 3)
                atlas["routing_edge_costs"] = geometry["chordal_cost"]
                new_dist, new_pred = shortest_path_tables(atlas, 3)
                np.testing.assert_array_equal(np.isfinite(old_dist), np.isfinite(new_dist))
                geometry_audit = {"directed_edges": int(np.isfinite(geometry["chordal_cost"]).sum()),
                                  "changed_predecessor_entries": int(np.count_nonzero(old_pred != new_pred))}
        errors = np.array([new_rows[s]["mean_last_100"]*1000 for s in seeds])
        total = next(m for m in panel["timing"]["modules"] if m["module"] == "Total online control")
        records.append({"output_dimension": p, "reference_mode": mode,
                        "centroid_successes": sum(four.strict_success(old_rows[s]) for s in seeds),
                        "chordal_successes": sum(four.strict_success(new_rows[s]) for s in seeds),
                        "median_error_mm": float(np.median(errors)),
                        **{f"p{q}_error_mm": float(np.percentile(errors, q)) for q in (90, 95, 99)},
                        "median_time_ms": total["median_ms"], "p99_time_ms": total["p99_ms"],
                        "unsolved_steps": sum(new_rows[s]["unsolved_steps"] for s in seeds),
                        "repaired_seeds": repaired, "regressed_seeds": regressed,
                        "failed_seeds": [s for s in seeds if not four.strict_success(new_rows[s])],
                        "changed_control_steps": step_differences,
                        "maximum_error_trace_difference_m": max_error_difference,
                        "geometry_audit": geometry_audit})
        verification.append({"output_dimension": p, "mode": mode, "checked_rollouts": len(seeds)})
    result = {"complete": True, "seeds": seeds, "records": records, "verification": verification,
              "timing_protocol": "Serial, one numerical thread; exclude first five steps; shared machine.",
              "reference_timing_note": "Centroid controls were run earlier; not a contemporaneous solver timing comparison."}
    four.save_json(root / "comparison.json", result)
    return result


def notebook_cells():
    import nbformat as nbf
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md(r"""## 8. Four-Baselines: Chordal Graph Edge Cost

This section uses the latest **four-baselines** protocol: **rank 60, $\lambda_g=0$,
Clarabel, seeds 211--310**, 48 frozen unstandardized charts and 320 control steps.
The earlier sections retain their historical $\lambda_g=0.05$ results and are not
the controls for this ablation. Centroid-cost controls here come from the saved
100-seed Clarabel/$\lambda_g=0$ four-baselines experiment.

For every existing directed edge, replace only its numerator:
$$w_{kk'}=\frac{1}{\sqrt{n_{kk'}}}\,
\frac{\|\Phi_k\Phi_k^\top-\Phi_{k'}\Phi_{k'}^\top\|_F}{\sqrt{2}}.$$
The bases are the rank-60 **complete input-output** linear chart bases (120 rows
for $p=8$, 144 for $p=10$), not clustering centroids or past-only projections.
Keep transition counts and the minimum-count threshold (3) unchanged. Compute
the chordal costs offline, then use them in both waypoint paths and iterative
$N=10$ minimum-route-cost terminal-reference selection.

The no-graph cases keep their fixed nearest eligible terminal configuration and
do not use edge costs; they are rerun as invariance controls. No $g$ boxes, affine
sum constraint, standardization, new data, or reclustering is introduced.
All 400 runs are serial with one numerical thread. Strict success and error
definitions match the earlier four-baselines table. These are exploratory seeds.
"""), code('''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML, display
CHORDAL_ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src/atlas_deepc").is_dir())
sys.path.insert(0, str(CHORDAL_ROOT / "src"))
from atlas_deepc.ablations import four_baseline_chordal as chordal_experiment
RUN_CHORDAL_EXPERIMENTS = False
if RUN_CHORDAL_EXPERIMENTS:
    chordal_experiment.run_all(update_notebook=False)
CHORDAL_RESULT_ROOT = chordal_experiment.RESULT_ROOT
chordal_report = json.loads((CHORDAL_RESULT_ROOT / "comparison.json").read_text())
assert chordal_report["complete"] and chordal_report["seeds"] == list(range(211, 311))
chordal_frame = pd.DataFrame(chordal_report["records"])
chordal_labels = [f"p={row.output_dimension}, " + ("graph" if row.reference_mode == "graph" else "no graph") for row in chordal_frame.itertuples()]
chordal_frame.index = chordal_labels
display(chordal_frame[["centroid_successes", "chordal_successes", "median_error_mm", "p90_error_mm", "p95_error_mm", "p99_error_mm", "unsolved_steps"]].round(3))
print("400 rollouts complete and verified; no-graph trajectories reproduce the centroid controls exactly.")
print(chordal_report["timing_protocol"])
'''), code('''fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
x = np.arange(4)
a = axes[0].bar(x-.18, chordal_frame.centroid_successes, .36, label="Centroid-cost control", color="#0072B2")
b = axes[0].bar(x+.18, chordal_frame.chordal_successes, .36, label="Chordal-cost experiment", color="#D55E00")
axes[0].bar_label(a, padding=3, fmt="%.0f")
axes[0].bar_label(b, padding=3, fmt="%.0f")
axes[0].set(xticks=x, xticklabels=chordal_labels, ylim=(0, 105), ylabel="Strict success (%)")
axes[0].tick_params(axis="x", labelrotation=15)
axes[0].legend(fontsize=9)
for p, color in [(8, "#0072B2"), (10, "#009E73")]:
    for directory, style, name in [(chordal_experiment.REFERENCE_ROOT, "--", "centroid"), (CHORDAL_RESULT_ROOT, "-", "chordal")]:
        panel = json.loads((directory / f"output{p}/graph/summary.json").read_text())
        errors = np.sort([r["mean_last_100"]*1000 for r in panel["results"]])
        axes[1].step(errors, np.arange(1, 101)/100, where="post", color=color, linestyle=style, label=f"p={p}, graph, {name}")
axes[1].axvline(10, color="0.4", linestyle=":")
axes[1].set(xlabel="Final-100 mean true error (mm)", ylabel="Cumulative fraction", xlim=(0, None), ylim=(0, 1.02))
axes[1].legend(fontsize=9)
for ax in axes:
    ax.grid(axis="y", alpha=.2)
fig.savefig(CHORDAL_RESULT_ROOT / "chordal_success_error.png", dpi=160)
plt.show()
for label, row in chordal_frame.iterrows():
    print(f"{label}: repaired {row.repaired_seeds}; regressed {row.regressed_seeds}")
display(HTML("<details><summary>Failed seeds and changed chart decisions</summary>" + chordal_frame[["failed_seeds", "changed_control_steps", "geometry_audit"]].to_html() + "</details>"))
'''), md("""### Serial Computation Time

Fresh serial-run timings, excluding offline basis/cost construction and the first
five online steps per rollout. The QP module includes construction and solving.
Module medians need not sum to the total median. Earlier centroid-run timings
were collected in a different session and are not a contemporaneous speed test.
"""), code('''timing_records = []
for p, mode in chordal_experiment.four.CASES:
    panel = json.loads((CHORDAL_RESULT_ROOT / f"output{p}/{mode}/summary.json").read_text())
    label = f"p={p}, " + ("graph" if mode == "graph" else "no graph")
    for module in panel["timing"]["modules"]:
        timing_records.append({"case": label, **module})
timing_frame = pd.DataFrame(timing_records)
fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
totals = timing_frame[timing_frame.module == "Total online control"]
for ax, metric, label in zip(axes, ["median_ms", "p99_ms"], ["Median online time (ms)", "P99 online time (ms)"]):
    bars = ax.bar(totals["case"], totals[metric], color=["#D55E00", "#0072B2", "#CC79A7", "#009E73"])
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set(ylabel=label, ylim=(0, totals[metric].max()*1.18))
    ax.tick_params(axis="x", labelrotation=15)
    ax.grid(axis="y", alpha=.2)
fig.savefig(CHORDAL_RESULT_ROOT / "chordal_serial_timing.png", dpi=160)
plt.show()
display(timing_frame.pivot(index="case", columns="module", values="median_ms").round(3))
''')]
    for cell in cells:
        cell.metadata["tags"] = [TAG]
    return cells


def update_notebook(execute=True):
    import nbformat as nbf
    from nbclient import NotebookClient
    notebook = nbf.read(NOTEBOOK, as_version=4)
    backup = ROOT / "archive/notebooks/baseline_atlas_deepc_no_standardization_before_chordal_20261011.ipynb"
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(NOTEBOOK, backup)
    cells = notebook_cells()
    if execute:
        report = nbf.v4.new_notebook(cells=cells)
        NotebookClient(report, timeout=600, kernel_name="gym-robotics-mujoco",
                       resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
        cells = report.cells
    notebook.cells = [c for c in notebook.cells if TAG not in c.metadata.get("tags", [])] + cells
    nbf.validate(notebook)
    temporary = NOTEBOOK.with_suffix(".ipynb.tmp")
    nbf.write(notebook, temporary)
    temporary.replace(NOTEBOOK)


def run_all(root=RESULT_ROOT, seeds=SEEDS, update_notebook=True):
    root = Path(root)
    prepare(root)
    for p, mode in four.CASES:
        four.run_case(p, mode, list(seeds), root, "CLARABEL", graph_cost="chordal")
    four.report(root, list(seeds), "CLARABEL")
    result = compare(root, list(seeds))
    if update_notebook:
        globals()["update_notebook"]()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        compare()
        update_notebook()
    else:
        run_all()
