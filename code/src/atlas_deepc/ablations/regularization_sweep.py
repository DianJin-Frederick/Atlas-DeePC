"""Lambda-g sweep on the saved primitive eight-output Reacher baseline."""

from dataclasses import asdict, replace
import argparse
import json
from pathlib import Path

import numpy as np

from atlas_deepc.baselines import unstandardized_atlas_deepc as experiment
from atlas_deepc.common.files import file_digest
from atlas_deepc.common.metrics import strict_success

ROOT = experiment.baseline.ROOT
SAVED_ROOT = ROOT / "experiments/reacher/routed_deepc_iid/unstandardized_shared_runner_v6_output8"
ATLAS_PATH = SAVED_ROOT / "behavioral_atlas.npz"
DATA_PATH = ROOT / "data/reacher/select_iid_offline_hankel.npy"
RESULT_ROOT = ROOT / "experiments/reacher/primitive_lambda_g_seed252_v1"
INITIAL_LAMBDAS = (0., 1e-4, 1e-3, .01, .05, .1, .5, 1., 5., 10., 50., 100.,
                   1e3, 1e4, 1e5, 1e6)
# Retain the original diagnostic values and add 34 logarithmically spaced values.
LAMBDAS = tuple(sorted((*INITIAL_LAMBDAS, *map(float, np.geomspace(2e-4, 5e5, 34)))))
assert len(LAMBDAS) == len(set(LAMBDAS)) == 50
KWARGS = dict(output_dimension=8, reference_mode="terminal",
              terminal_reference_policy="fixed_nearest_eligible_fingertip",
              chart_selection_method="nearest_centroid", warmup_steps=2)


def level_directory(value, root=RESULT_ROOT):
    return Path(root) / ("lambda_" + format(float(value), ".12g").replace(".", "p"))


def verify_baseline():
    config = experiment.default_config()
    original = json.loads((SAVED_ROOT / "rank060_steps320/terminal/manifest.json").read_text())
    assert original["config"] == {k: v for k, v in asdict(config).items() if k != "seed"}
    assert original["data_sha256"] == file_digest(DATA_PATH)
    assert original["atlas_sha256"] == file_digest(ATLAS_PATH)
    assert original["reference_mode"] == "terminal" and original["output_dimension"] == 8
    assert original["standardize"] is False and original["coefficient_bounds_enabled"] is False
    return config


def run(seeds=(252,), values=LAMBDAS, root=RESULT_ROOT):
    values = [float(v) for v in values]
    if not values or any(not np.isfinite(v) or v < 0 for v in values) or len(set(values)) != len(values):
        raise ValueError("Lambda values must be finite, distinct, and nonnegative")
    base = verify_baseline()
    for value in values:
        print(f"Lambda_g={value:g}", flush=True)
        _, missing = experiment.load_panel(seeds, replace(base, lam_g=value), ATLAS_PATH,
                                            level_directory(value, root), DATA_PATH, **KWARGS)
        if not missing:
            print("Reusing verified completed rollouts", flush=True)
            continue
        experiment.run_panel(seeds, replace(base, lam_g=value), ATLAS_PATH,
                             level_directory(value, root), DATA_PATH, **KWARGS)
    report = summarize(seeds, values, root)
    (Path(root) / "sweep_summary.json").write_text(json.dumps(report, indent=2))
    return report


def summarize(seeds=(252,), values=LAMBDAS, root=RESULT_ROOT):
    base = verify_baseline()
    records, reproduction = [], []
    for seed in seeds:
        original_path = SAVED_ROOT / f"rank060_steps320/terminal/seed{seed}_control_results.npz"
        with np.load(original_path, allow_pickle=True) as original:
            goal = original["final_goal"].copy()
            original_y = original["y"].copy()
            original_u = original["u"].copy()
            original_row = json.loads(original["summary"].item())
        for value in values:
            rows, missing = experiment.load_panel(
                [seed], replace(base, lam_g=float(value)), ATLAS_PATH,
                level_directory(value, root), DATA_PATH, **KWARGS)
            if missing:
                raise ValueError(f"Missing seed {seed}, lambda {value}")
            row = rows[0]
            with np.load(row["result_path"], allow_pickle=True) as trace:
                np.testing.assert_array_equal(trace["final_goal"], goal)
                np.testing.assert_array_equal(trace["ref"], np.tile(goal, (base.steps, 1)))
                g_norm = np.linalg.norm(trace["g_coefficients"], axis=1)
                errors = trace["distance"]
                records.append({
                    "seed": seed, "lambda_g": float(value), "strict_success": strict_success(row),
                    "final100_mean_mm": 1000*row["mean_last_100"],
                    "final100_max_mm": 1000*row["max_last_100"],
                    "minimum_mm": 1000*row["minimum"], "last_mm": 1000*row["last"],
                    "mean_rollout_mm": 1000*float(np.mean(errors)),
                    "time_below_10mm_fraction": float(np.mean(errors < .01)),
                    "charts": np.unique(trace["chart"]).astype(int).tolist(),
                    "chart_switches": int(np.sum(np.diff(trace["chart"]) != 0)),
                    "median_g_norm": float(np.nanmedian(g_norm)),
                    "median_prediction_error_mm": 1000*float(np.nanmedian(trace["prediction_error_xy"])),
                    "unsolved_steps": row["unsolved_steps"],
                    "median_online_ms": 1000*float(np.median(trace["control_compute_time_s"][5:])),
                    "median_solve_ms": 1000*float(np.median(trace["solve_time_s"][5:])),
                    "result_path": row["result_path"],
                })
                if value == base.lam_g:
                    reproduction.append({"seed": seed,
                        "saved_final100_mean_mm": 1000*original_row["mean_last_100"],
                        "rerun_final100_mean_mm": 1000*row["mean_last_100"],
                        "max_abs_output_difference": float(np.max(abs(trace["y"]-original_y))),
                        "max_abs_input_difference": float(np.max(abs(trace["u"]-original_u)))})
    return {"seeds": list(seeds), "lambda_g_values": list(values),
            "data_sha256": file_digest(DATA_PATH), "atlas_sha256": file_digest(ATLAS_PATH),
            "reference_mode": "terminal", "outputs": 8, "coefficient_bounds_enabled": False,
            "only_changed_parameter": "lam_g", "serial_single_numerical_thread": True,
            "records": records, "baseline_reproduction": reproduction}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[252])
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(LAMBDAS))
    parser.add_argument("--root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()
    result = run(args.seeds, args.lambdas, args.root)
    print(json.dumps(result, indent=2))
