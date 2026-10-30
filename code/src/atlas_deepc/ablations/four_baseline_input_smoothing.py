"""Paired lambda_du=0 ablation of the frozen projection/chordal baseline."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np

from atlas_deepc.ablations import four_baseline_projection_chordal as baseline

four = baseline.four
RESULT_ROOT = four.ROOT / "experiments/reacher/four_baselines_projection_chordal_lamdu0_clarabel_v1"
NOTEBOOK = four.ROOT / "notebooks/reacher/ablations/ablation_input_smoothing_four_baselines.ipynb"
SEEDS = list(range(211, 311))


def assert_only_smoothing_changed(current, previous):
    expected = json.loads(json.dumps(previous))
    assert expected["config"]["lam_du"] == 5.0
    expected["config"]["lam_du"] = 0.0
    assert current == expected, "A setting other than lambda_du differs from the current baseline"


def prepare(root=RESULT_ROOT):
    root = Path(root)
    assert root.resolve() != baseline.RESULT_ROOT.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = replace(four.experiment.default_config(), lam_g=0., lam_du=0., svd_rank=60)
    baseline_manifests = {}
    for p, mode in four.CASES:
        path = baseline.RESULT_ROOT / f"output{p}/{mode}/manifest.json"
        previous = json.loads(path.read_text())
        summary = json.loads(path.with_name("summary.json").read_text())
        assert summary["seeds"] == SEEDS and len(summary["results"]) == 100
        source_atlas = path.parent.parent / "behavioral_atlas.npz"
        destination = root / f"output{p}/behavioral_atlas.npz"
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            shutil.copy2(source_atlas, destination)
        assert four.file_digest(destination) == previous["atlas_sha256"]
        signature = four.experiment.execution_signature(
            config, four.DATA_PATH, destination, p, mode,
            chart_selection_method="full_window_projection", warmup_steps=12)
        # The shared run_case adds its runner version after execution_signature.
        expected = {key: previous[key] for key in signature}
        expected["version"] = signature["version"]
        assert_only_smoothing_changed(signature, expected)
        assert four.file_digest(four.__file__) == previous["runner_sha256"]
        baseline_manifests[f"{p}_{mode}"] = four.file_digest(path)
    manifest = {
        "baseline_root": str(baseline.RESULT_ROOT), "baseline_manifests": baseline_manifests,
        "seeds": SEEDS, "rollouts": 400, "lambda_du": 0., "baseline_lambda_du": 5.,
        "lambda_g": 0., "solver": "CLARABEL", "rank": 60, "charts": 48,
        "chart_selection": "full_window_projection", "graph_cost": "chordal",
        "warmup_steps": 12, "control_steps": 300, "workers": 1, "numerical_threads": 1,
        "data_sha256": four.file_digest(four.DATA_PATH),
        "ablation_source_sha256": four.file_digest(__file__),
        "changed_component": "Input-increment penalty weight only: lambda_du 5 -> 0",
    }
    path = root / "experiment_manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "Incompatible ablation root"
    else:
        four.save_json(path, manifest)
        shutil.copytree(baseline.RESULT_ROOT / "frozen_source", root / "frozen_source")
        shutil.copy2(__file__, root / "frozen_source" / Path(__file__).name)
    return manifest


def run_case(p, mode, root=RESULT_ROOT):
    # Validate the historical source with its original defaults, then scope the
    # one-parameter override to this run. Shared baseline files remain unchanged.
    source = four.verify_source(p, mode)
    default = four.experiment.default_config()
    with patch.object(four, "verify_source", return_value=source), patch.object(
            four.experiment, "default_config", return_value=replace(default, lam_du=0.)):
        panel = four.run_case(
            p, mode, SEEDS, Path(root), "CLARABEL", graph_cost="chordal",
            chart_selection_method="full_window_projection", warmup_steps=12)
    previous = json.loads((baseline.RESULT_ROOT / f"output{p}/{mode}/manifest.json").read_text())
    assert_only_smoothing_changed(panel["signature"], previous)
    return panel


def finish(root=RESULT_ROOT):
    root = Path(root)
    baseline.verify(root, SEEDS)
    report = four.report(root, SEEDS, "CLARABEL")
    report["lambda_du"] = 0.
    four.save_json(root / "four_case_table.json", report)
    comparisons = []
    for p, mode in four.CASES:
        old_dir = baseline.RESULT_ROOT / f"output{p}/{mode}"
        new_dir = root / f"output{p}/{mode}"
        previous = json.loads((old_dir / "summary.json").read_text())
        current = json.loads((new_dir / "summary.json").read_text())
        assert_only_smoothing_changed(current["signature"], previous["signature"])
        old_rows = {r["seed"]: r for r in previous["results"]}
        new_rows = {r["seed"]: r for r in current["results"]}
        for filename in ("selection_geometry.npz", "graph_geometry.npz"):
            if filename.startswith("graph") and mode != "graph":
                continue
            with np.load(old_dir / filename) as a, np.load(new_dir / filename) as b:
                assert a.files == b.files
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
        for seed in SEEDS:
            with np.load(old_dir / f"seed{seed}_control_results.npz") as a, np.load(
                    new_dir / f"seed{seed}_control_results.npz") as b:
                np.testing.assert_array_equal(a["selection_windows"][0], b["selection_windows"][0])
                np.testing.assert_array_equal(a["final_goal"].reshape(-1)[6:8], b["final_goal"].reshape(-1)[6:8])
        old_success = {s for s in SEEDS if four.strict_success(old_rows[s])}
        new_success = {s for s in SEEDS if four.strict_success(new_rows[s])}
        comparisons.append({
            "output_dimension": p, "graph_guidance": mode == "graph",
            "baseline_successes": len(old_success), "lamdu0_successes": len(new_success),
            "repaired_seeds": sorted(new_success-old_success),
            "regressed_seeds": sorted(old_success-new_success),
        })
    four.save_json(root / "paired_comparison.json", comparisons)
    # Keep the common table layout, but identify the additional ablated weight.
    for filename in ("table_clarabel_lamg0.tex", "table_clarabel_lamg0_preview.tex"):
        path = root / filename
        text = path.read_text().replace(r"with $\lambda_g=0$ and", r"with $\lambda_g=0$, $\lambda_{\Delta u}=0$, and")
        path.write_text(text.replace("tab:basic-cases-clarabel-lamg0", "tab:basic-cases-clarabel-lamg0-lamdu0"))
    write_notebook(root)
    print(json.dumps(comparisons, indent=2), flush=True)
    return comparisons


def write_notebook(root=RESULT_ROOT):
    import nbformat as nbf
    root = Path(root)
    complete = (root / "paired_comparison.json").exists()
    cells = [nbf.v4.new_markdown_cell(r"""# Input-Smoothing Ablation: Four Current Baselines

Only $\lambda_{\Delta u}$ changes from **5 to 0**. All four cases use
$\lambda_g=0$, Clarabel, rank 60, 48 frozen unstandardized linear charts,
full-window projection-residual chart selection, and seeds **211--310**.
Graph cases retain chordal edge costs, iterative $N=10$ terminal selection,
and graph waypoints. No-graph cases retain one fixed nearest eligible terminal
configuration. Each rollout uses 12 zero-input warmup steps and 300 control steps.
No affine or empirical coefficient-box constraints are used.

The removed term is $\lambda_{\Delta u}\sum_{j=0}^{T_f-2}
\|u_{j+1|t}-u_{j|t}\|_2^2$. Input effort and input bounds remain unchanged.
Runs are serial with one numerical thread. Timings include online construction
and solving, discard the first five controlled steps, and exclude offline setup.
The two weight settings are timed in separate sessions.
"""), nbf.v4.new_code_cell('''from pathlib import Path
import sys, json
import pandas as pd
from IPython.display import display
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src/atlas_deepc").is_dir())
sys.path.insert(0, str(ROOT / "src"))
from atlas_deepc.ablations import four_baseline_input_smoothing as study
RUN_EXPERIMENTS = False
if RUN_EXPERIMENTS:
    study.run_all()
RESULTS = study.RESULT_ROOT
if (RESULTS / "paired_comparison.json").exists():
    display(pd.DataFrame(json.loads((RESULTS / "paired_comparison.json").read_text())))
    display(pd.DataFrame(json.loads((RESULTS / "four_case_table.json").read_text())["columns"]))
else:
    print("400-rollout experiment is not complete; per-case progress is saved under RESULTS.")
''')]
    if complete:
        comparison = json.loads((root / "paired_comparison.json").read_text())
        lines = ["## Completed Results", "", "| Output | Graph | Success, lambda_du=5 | Success, lambda_du=0 |",
                 "|---|---|---:|---:|"]
        for r in comparison:
            lines.append(f"| {r['output_dimension']} | {'Yes' if r['graph_guidance'] else 'No'} | {r['baseline_successes']}/100 | {r['lamdu0_successes']}/100 |")
        cells.insert(1, nbf.v4.new_markdown_cell("\n".join(lines)))
    notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {
        "display_name": "gym-robotics-mujoco", "language": "python", "name": "gym-robotics-mujoco"}})
    nbf.validate(notebook)
    nbf.write(notebook, NOTEBOOK)


def run_all(root=RESULT_ROOT):
    prepare(root)
    for p, mode in four.CASES:
        run_case(p, mode, root)
    return finish(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        prepare()
        write_notebook()
    elif args.report_only:
        finish()
    else:
        run_all()
