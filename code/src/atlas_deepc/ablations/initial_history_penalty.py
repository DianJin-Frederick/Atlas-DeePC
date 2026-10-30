"""Remove the output-history slack penalty in the latest 10-output graph case."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np

from atlas_deepc.ablations import four_baseline_input_smoothing as smoothing

four = smoothing.four
SEEDS = list(range(211, 311))
RESULT_ROOT = four.ROOT / "experiments/reacher/projection_chordal_all_penalties_zero_output10_graph_v1"
CONTROL_DIR = smoothing.RESULT_ROOT / "output10/graph"


def check_signature(current, control, initial_version=False):
    expected = json.loads(json.dumps(control))
    assert expected["config"]["lam_ini"] == 1e5
    assert expected["config"]["lam_g"] == expected["config"]["lam_du"] == 0.
    expected["config"]["lam_ini"] = 0.
    if initial_version:
        expected = {key: expected[key] for key in current}
        expected["version"] = current["version"]
    assert current == expected, "Unexpected change beyond lambda_ini"


def prepare(root=RESULT_ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    previous = json.loads((CONTROL_DIR / "manifest.json").read_text())
    assert json.loads((CONTROL_DIR / "summary.json").read_text())["seeds"] == SEEDS
    source_atlas = CONTROL_DIR.parent / "behavioral_atlas.npz"
    atlas = root / "output10/behavioral_atlas.npz"
    atlas.parent.mkdir(exist_ok=True)
    if not atlas.exists():
        shutil.copy2(source_atlas, atlas)
    assert four.file_digest(atlas) == previous["atlas_sha256"]
    config = replace(four.experiment.default_config(), lam_g=0., lam_du=0., lam_ini=0., svd_rank=60)
    signature = four.experiment.execution_signature(
        config, four.DATA_PATH, atlas, 10, "graph",
        chart_selection_method="full_window_projection", warmup_steps=12)
    check_signature(signature, previous, initial_version=True)
    assert four.file_digest(four.__file__) == previous["runner_sha256"]
    manifest = {
        "control_directory": str(CONTROL_DIR),
        "control_manifest_sha256": four.file_digest(CONTROL_DIR / "manifest.json"),
        "ablation_source_sha256": four.file_digest(__file__),
        "data_sha256": four.file_digest(four.DATA_PATH), "atlas_sha256": four.file_digest(atlas),
        "seeds": SEEDS, "rollouts": 100, "output_dimension": 10, "reference_mode": "graph",
        "lambda_ini": 0., "lambda_g": 0., "lambda_du": 0.,
        "changed_component": "Output-history slack penalty only: lambda_ini 100000 -> 0",
        "solver": "CLARABEL", "rank": 60, "charts": 48, "warmup_steps": 12,
        "control_steps": 300, "workers": 1, "numerical_threads": 1,
        "chart_selection": "full_window_projection", "graph_cost": "chordal",
        "interpretation": "Output-history slack is free; hard past-input matching remains."
    }
    path = root / "experiment_manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "Incompatible experiment root"
    else:
        four.save_json(path, manifest)
        shutil.copytree(smoothing.RESULT_ROOT / "frozen_source", root / "frozen_source")
        shutil.copy2(__file__, root / "frozen_source" / Path(__file__).name)
    return previous


def finish(root=RESULT_ROOT):
    root = Path(root)
    directory = root / "output10/graph"
    current = json.loads((directory / "summary.json").read_text())
    control = json.loads((CONTROL_DIR / "summary.json").read_text())
    check_signature(current["signature"], control["signature"])
    assert current["seeds"] == control["seeds"] == SEEDS
    with patch.object(four, "CASES", ((10, "graph"),)):
        smoothing.baseline.verify(root, SEEDS)
    for filename in ("selection_geometry.npz", "graph_geometry.npz"):
        with np.load(CONTROL_DIR / filename) as a, np.load(directory / filename) as b:
            assert a.files == b.files
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
    for seed in SEEDS:
        filename = f"seed{seed}_control_results.npz"
        with np.load(CONTROL_DIR / filename) as a, np.load(directory / filename) as b:
            np.testing.assert_array_equal(a["selection_windows"][0], b["selection_windows"][0])
            np.testing.assert_array_equal(a["final_goal"].reshape(-1)[6:8], b["final_goal"].reshape(-1)[6:8])
    def metrics(panel):
        rows = panel["results"]
        errors = np.array([1000*r["mean_last_100"] for r in rows])
        return {
            "successes": sum(four.strict_success(r) for r in rows),
            "median_error_mm": float(np.median(errors)), "mean_error_mm": float(errors.mean()),
            **{f"p{q}_error_mm": float(np.percentile(errors, q)) for q in (90, 95, 99)},
            "unsolved_steps": sum(r["unsolved_steps"] for r in rows),
            "failed_seeds": [r["seed"] for r in rows if not four.strict_success(r)],
            "timing": panel["timing"],
        }
    old_success = {r["seed"] for r in control["results"] if four.strict_success(r)}
    new_success = {r["seed"] for r in current["results"] if four.strict_success(r)}
    comparison = {
        "seeds": SEEDS, "output_dimension": 10, "reference_mode": "graph",
        "lambda_g": 0., "lambda_du": 0.,
        "control_lambda_ini_100000": metrics(control), "ablation_lambda_ini_0": metrics(current),
        "repaired_seeds": sorted(new_success-old_success),
        "regressed_seeds": sorted(old_success-new_success),
        "verification": "Only lambda_ini changed; identical atlas, graph, measured initial windows, and targets.",
    }
    four.save_json(root / "comparison.json", comparison)
    print(json.dumps(comparison, indent=2), flush=True)
    return comparison


def run(root=RESULT_ROOT):
    previous = prepare(root)
    source = four.verify_source(10, "graph")
    config = replace(four.experiment.default_config(), lam_ini=0., lam_du=0., lam_g=0.)
    # Keep the shared baseline defaults and existing cache signatures untouched.
    with patch.object(four, "verify_source", return_value=source), patch.object(
            four.experiment, "default_config", return_value=config):
        panel = four.run_case(
            10, "graph", SEEDS, Path(root), "CLARABEL", graph_cost="chordal",
            chart_selection_method="full_window_projection", warmup_steps=12)
    check_signature(panel["signature"], previous)
    return finish(root)


if __name__ == "__main__":
    run()
