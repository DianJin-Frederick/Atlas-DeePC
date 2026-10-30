"""The four raw-clustering baseline cases with lambda_g=0 and a chosen solver."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import time
from unittest.mock import patch
import warnings

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.baselines import unstandardized_atlas_deepc as experiment
from atlas_deepc.benchmarks.baseline_1000_benchmark import summarize_timing
from atlas_deepc.common.files import file_digest
from atlas_deepc.common.metrics import strict_success, true_error_metrics


ROOT = experiment.baseline.ROOT
RESULT_ROOT = ROOT / "experiments/reacher/basic_four_cases_lamg0_osqp_v1"
DATA_PATH = ROOT / "data/reacher/select_iid_offline_hankel.npy"
VERSION = "basic_four_cases_lamg0_osqp_v1"
CASES = ((8, "terminal"), (8, "graph"), (10, "terminal"), (10, "graph"))


def save_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def source_panel(dimension, mode):
    return (ROOT / "experiments/reacher/routed_deepc_iid" /
            f"unstandardized_shared_runner_v6_output{dimension}" / "rank060_steps320" / mode)


def verify_source(dimension, mode):
    directory = source_panel(dimension, mode)
    original = json.loads((directory / "manifest.json").read_text())
    original_config = {k: v for k, v in asdict(experiment.default_config()).items() if k != "seed"}
    assert original["config"] == original_config
    assert original["solver"] == "CLARABEL" and original_config["lam_g"] == .05
    assert original["standardize"] is False and original["coefficient_bounds_enabled"] is False
    assert original["data_sha256"] == file_digest(DATA_PATH)
    atlas_path = directory.parents[1] / "behavioral_atlas.npz"
    assert original["atlas_sha256"] == file_digest(atlas_path)
    policy = experiment.terminal_policy(mode)
    assert original["terminal_reference_policy"] == policy
    return original, atlas_path


def run_case(dimension, mode, seeds, root, solver="OSQP"):
    if solver not in ("OSQP", "CLARABEL"):
        raise ValueError("Expected OSQP or CLARABEL")
    selected_solver = solver
    version = f"basic_four_cases_lamg0_{solver.lower()}_v1"
    original, source_atlas = verify_source(dimension, mode)
    config = replace(experiment.default_config(), lam_g=0.)
    directory = root / f"output{dimension}" / mode
    directory.mkdir(parents=True, exist_ok=True)
    atlas_path = directory.parent / "behavioral_atlas.npz"
    if not atlas_path.exists():
        shutil.copy2(source_atlas, atlas_path)
    assert file_digest(atlas_path) == original["atlas_sha256"]
    policy = original["terminal_reference_policy"]
    signature = experiment.execution_signature(
        config, DATA_PATH, atlas_path, dimension, mode,
        terminal_reference_policy=policy)
    signature.update(version=version, solver=solver, runner_sha256=file_digest(__file__),
                     solver_options={"max_iter": 10000},
                     other_solver_options=f"CVXPY {solver} defaults; no solver switching or retries",
                     source_panel=str(source_panel(dimension, mode)),
                     source_manifest_sha256=file_digest(source_panel(dimension, mode) / "manifest.json"),
                     package_versions={name: importlib.metadata.version(name)
                                       for name in ("numpy", "scipy", "cvxpy", "osqp", "clarabel", "mujoco", "gymnasium")})
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == signature, "Incompatible cache"
    else:
        assert not list(directory.glob("seed*_control_results.npz")), "Unversioned cache"
        save_json(manifest_path, signature)
    rows = []
    with experiment.output_context(dimension), threadpool_limits(limits=1):
        H = experiment.load_hankel(DATA_PATH, dimension)
        atlas = experiment.routed.load_atlas(atlas_path)
        assert np.all(atlas["feature_mean"] == 0) and np.all(atlas["feature_scale"] == 1)
        controller = experiment.baseline.BaselineAtlasDeePC(
            H, atlas, config, terminal_reference_policy=policy)
        for matrix, _ in controller.matrix_cache.values():
            matrix.pop("g_min", None)
            matrix.pop("g_max", None)
            assert matrix["basis"].shape == ((2 + dimension) * 12, 60)
            assert np.all(matrix["offset"] == 0) and matrix.get("g_sum") is None
        original_class = experiment.routed.pipe.DeePCController
        coefficients = []

        class SelectedSolverController(original_class):
            def __init__(self, Q, R, Q_terminal, lam_g, lam_ini, lam_du, u_limit, solver, **kwargs):
                assert solver == "CLARABEL" and lam_g == 0
                super().__init__(Q, R, Q_terminal, lam_g, lam_ini, lam_du, u_limit, selected_solver, **kwargs)

            def solve(self, matrix, *args, **kwargs):
                assert self.solver == selected_solver and self.lam_g == 0
                assert matrix.get("g_min") is None and matrix.get("g_max") is None
                value = super().solve(matrix, *args, **kwargs)
                coefficients.append(np.full(config.svd_rank, np.nan) if value[2] is None
                                    else value[2].ravel().copy())
                return value

        with patch.object(experiment.routed.pipe, "DeePCController", SelectedSolverController):
            for index, seed in enumerate(seeds, 1):
                path = directory / f"seed{seed}_control_results.npz"
                if path.exists():
                    with np.load(path, allow_pickle=True) as trace:
                        row = json.loads(trace["summary"].item())
                        assert row["experiment_version"] == version
                        assert row["solver"] == selected_solver and row["lam_g"] == 0
                        assert trace["y"].shape == (320, dimension)
                else:
                    coefficients.clear()
                    start = time.perf_counter()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        row = controller.run(seed, path, reference_mode=mode,
                                             localization_history_length=2)
                    assert len(coefficients) == config.steps
                    row.update(experiment_version=version, solver=selected_solver,
                               coefficient_bounds_enabled=False, feature_standardization=False,
                               output_dimension=dimension, chart_selection_method="nearest_centroid",
                               elapsed_rollout_wall_s=time.perf_counter() - start,
                               solver_warnings=dict(Counter(str(w.message) for w in caught)))
                    with np.load(path, allow_pickle=True) as trace:
                        arrays = {k: trace[k] for k in trace.files if k != "summary"}
                    assert row["terminal_reference_policy"] == policy
                    if mode == "terminal":
                        np.testing.assert_array_equal(arrays["ref"], np.tile(arrays["final_goal"], (320, 1)))
                        with np.load(source_panel(dimension, mode) / path.name, allow_pickle=True) as reference:
                            np.testing.assert_array_equal(arrays["final_goal"], reference["final_goal"])
                    metrics = true_error_metrics(arrays["distance"])
                    for key in ("success_30", "success_100", "mean_last_100"):
                        assert row[key] == metrics[key]
                    np.savez_compressed(path, **arrays, g_coefficients=np.asarray(coefficients),
                                        summary=np.array(json.dumps(row)))
                row["result_path"] = str(path)
                rows.append(row)
                print(f"p={dimension} {mode}: {index}/{len(seeds)}, seed={seed}, "
                      f"success={strict_success(row)}, final100={1000*row['mean_last_100']:.3f} mm, "
                      f"unsolved={row['unsolved_steps']}", flush=True)
                save_json(directory / "progress.json", {"seeds_completed": [r["seed"] for r in rows],
                          "requested_seeds": seeds, "successes_so_far": sum(strict_success(r) for r in rows),
                          "last_result": row})
    timing = summarize_timing([r["result_path"] for r in rows])
    result = {"signature": signature, "seeds": seeds, "results": rows, "timing": timing}
    save_json(directory / "summary.json", result)
    return result


def report(root, seeds, solver="OSQP"):
    columns = []
    for dimension, mode in CASES:
        data = json.loads((root / f"output{dimension}" / mode / "summary.json").read_text())
        assert data["seeds"] == seeds
        assert data["signature"]["solver"] == solver
        rows = data["results"]
        errors = np.array([1000*r["mean_last_100"] for r in rows])
        timing = next(m for m in data["timing"]["modules"] if m["module"] == "Total online control")
        columns.append({"output_dimension": dimension, "graph_guidance": mode == "graph",
                        "terminal_reference_policy": data["signature"]["terminal_reference_policy"],
                        "success_rate": 100*sum(strict_success(r) for r in rows)/len(rows),
                        "median_error_mm": float(np.median(errors)),
                        **{f"p{q}_error_mm": float(np.percentile(errors, q)) for q in (90, 95, 99)},
                        "median_time_ms": timing["median_ms"], "p95_time_ms": timing["p95_ms"],
                        "p99_time_ms": timing["p99_ms"],
                        "unsolved_steps": sum(r["unsolved_steps"] for r in rows),
                        "rollouts_with_unsolved_steps": sum(r["unsolved_steps"] > 0 for r in rows),
                        "failed_seeds": [r["seed"] for r in rows if not strict_success(r)]})
    payload = {"seeds": seeds, "lambda_g": 0, "solver": solver, "columns": columns,
               "error_definition": "Across-seed quantiles of each rollout final-100 mean true error",
               "time_definition": "Pooled online step times after discarding first 5 per rollout; milliseconds",
               "timing_caveat": "Serial, one numerical thread; other machine workloads not controlled"}
    save_json(root / "four_case_table.json", payload)
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Output dimension & \multicolumn{2}{c}{$p=8$} & \multicolumn{2}{c}{$p=10$} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Graph guidance & No & Yes & No & Yes \\", r"\midrule"]
    metrics = [(r"Success rate $\uparrow$", "success_rate", 0, True),
               (r"Median error (mm) $\downarrow$", "median_error_mm", 2, False),
               (r"90th-percentile error (mm) $\downarrow$", "p90_error_mm", 3, False),
               (r"95th-percentile error (mm) $\downarrow$", "p95_error_mm", 3, False),
               (r"99th-percentile error (mm) $\downarrow$", "p99_error_mm", 3, False)]
    for label, key, decimals, maximize in metrics:
        values = [c[key] for c in columns]
        best = max(values) if maximize else min(values)
        cells = []
        for value in values:
            cell = f"{value:.{decimals}f}" + (r"\%" if key == "success_rate" else "")
            is_best = np.isclose(value, best, rtol=0, atol=1e-12)
            if key == "success_rate":
                cells.append("$" + (r"\mathbf{" + cell + "}" if is_best else cell) + "$")
            else:
                cells.append(r"\textbf{" + cell + "}" if is_best else cell)
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\midrule", r"\multicolumn{5}{l}{\textit{Total online computation time (ms)}} \\"])
    for label, key in (("Median", "median_time_ms"), ("95th percentile", "p95_time_ms"),
                       ("99th percentile", "p99_time_ms")):
        lines.append(label + " & " + " & ".join(f"{c[key]:.2f}" for c in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}",
                  r"\caption{Four basic Reacher cases with $\lambda_g=0$ and " + solver + ", evaluated on "
                  + f"{len(seeds)} seeds ({seeds[0]}--{seeds[-1]}). "
                  + r"Errors summarize the final-100 mean error across seeds. Timings pool online steps "
                  r"after discarding the first five per rollout. Unguided cases use one fixed nearest "
                  r"terminal configuration; guided cases retain iterative $N=10$ terminal selection.}",
                  r"\label{tab:basic-cases-" + solver.lower() + r"-lamg0}", r"\end{table}"])
    table = "\n".join(lines) + "\n"
    (root / f"table_{solver.lower()}_lamg0.tex").write_text(table)
    document = (r"\documentclass[10pt]{article}" + "\n" +
                r"\usepackage[landscape,margin=1in]{geometry}" + "\n" +
                r"\usepackage{booktabs}" + "\n" + r"\usepackage{times}" + "\n" +
                r"\pagestyle{empty}" + "\n" + r"\begin{document}" + "\n" + table +
                r"\end{document}" + "\n")
    (root / f"table_{solver.lower()}_lamg0_preview.tex").write_text(document)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("OSQP", "CLARABEL"), default="OSQP")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(211, 311)))
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.root is None:
        args.root = ROOT / f"experiments/reacher/basic_four_cases_lamg0_{args.solver.lower()}_v1"
    assert len(args.seeds) == len(set(args.seeds))
    args.root.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        save_json(args.root / f"session_{time.time_ns()}.json", {
            "started_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
            "workers": 1, "numerical_threads": 1, "seeds": args.seeds, "solver": args.solver})
        for dimension, mode in CASES:
            run_case(dimension, mode, args.seeds, args.root, args.solver)
    report(args.root, args.seeds, args.solver)


if __name__ == "__main__":
    main()
