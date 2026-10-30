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


def run_case(dimension, mode, seeds, root, solver="OSQP", *, rank=60,
             record_coefficients=True, timing_workers=1, graph_cost="centroid",
             chart_selection_method="nearest_centroid", warmup_steps=None):
    if solver not in ("OSQP", "CLARABEL"):
        raise ValueError("Expected OSQP or CLARABEL")
    selected_solver = solver
    if graph_cost not in ("centroid", "chordal"):
        raise ValueError("Expected centroid or chordal graph cost")
    version = f"basic_four_cases_lamg0_{solver.lower()}_v1"
    original, source_atlas = verify_source(dimension, mode)
    if not 1 <= rank < (2 + dimension) * 12:
        raise ValueError("Rank must be positive and below the Hankel row dimension")
    config = replace(experiment.default_config(), lam_g=0., svd_rank=rank)
    if warmup_steps is None:
        warmup_steps = (experiment.routed.pipe.L if chart_selection_method == "full_window_projection"
                        else experiment.routed.pipe.T_INI)
    directory = root / f"output{dimension}" / mode
    directory.mkdir(parents=True, exist_ok=True)
    atlas_path = directory.parent / "behavioral_atlas.npz"
    if not atlas_path.exists():
        shutil.copy2(source_atlas, atlas_path)
    assert file_digest(atlas_path) == original["atlas_sha256"]
    policy = original["terminal_reference_policy"]
    signature = experiment.execution_signature(
        config, DATA_PATH, atlas_path, dimension, mode,
        terminal_reference_policy=policy, chart_selection_method=chart_selection_method,
        warmup_steps=warmup_steps)
    signature.update(version=version, solver=solver, runner_sha256=file_digest(__file__),
                     record_coefficients=record_coefficients, timing_workers=timing_workers,
                     graph_cost_method=graph_cost if mode == "graph" else "not_used",
                     graph_metric_sha256=(file_digest(Path(__file__).with_name("chordal_graph_cost.py"))
                                          if graph_cost == "chordal" and mode == "graph" else None),
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
        if mode == "graph" and graph_cost == "chordal":
            from atlas_deepc.ablations.chordal_graph_cost import chordal_distances, graph_costs
            cache = experiment.routed.precompute_chart_matrix_cache(H, atlas, config, geometry="linear")
            bases = np.stack([cache[(k,)][0]["basis"] for k in range(config.clusters)])
            distances = chordal_distances(bases)
            atlas["routing_edge_costs"] = graph_costs(
                distances, atlas["transition_counts"], config.edge_count_threshold)
            centroid_costs = graph_costs(atlas["center_distance"], atlas["transition_counts"],
                                         config.edge_count_threshold)
            np.testing.assert_array_equal(np.isfinite(centroid_costs), np.isfinite(atlas["routing_edge_costs"]))
            np.savez_compressed(directory / "graph_geometry.npz", bases=bases,
                                labels=atlas["labels"], transition_counts=atlas["transition_counts"],
                                chordal_distance=distances, chordal_cost=atlas["routing_edge_costs"],
                                centroid_cost=centroid_costs)
            # Install the costs before the controller constructs its all-pairs routes.
            with patch.object(experiment.routed, "precompute_chart_matrix_cache", return_value=cache):
                controller = experiment.baseline.BaselineAtlasDeePC(
                    H, atlas, config, terminal_reference_policy=policy)
        else:
            controller = experiment.baseline.BaselineAtlasDeePC(
                H, atlas, config, terminal_reference_policy=policy)
        for matrix, _ in controller.matrix_cache.values():
            matrix.pop("g_min", None)
            matrix.pop("g_max", None)
            assert matrix["basis"].shape == ((2 + dimension) * 12, rank)
            assert np.all(matrix["offset"] == 0) and matrix.get("g_sum") is None
        selector = None
        selection_windows = []
        if chart_selection_method == "full_window_projection":
            class LoggedProjectionSelector(experiment.FullWindowProjectionSelector):
                def select(self, window):
                    selection_windows.append(np.asarray(window).copy())
                    return super().select(window)

            selector = LoggedProjectionSelector(controller.matrix_cache, experiment.routed.pipe.M,
                                                dimension, experiment.routed.pipe.L)
            np.savez_compressed(directory / "selection_geometry.npz", bases=selector.bases,
                                chart_ids=selector.chart_ids, ranks=selector.ranks)
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
                if record_coefficients:
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
                        assert row["svd_rank"] == rank
                        assert row["chart_selection_method"] == chart_selection_method
                        assert row["localization_history_length"] == warmup_steps
                        assert trace["y"].shape == (320, dimension)
                else:
                    coefficients.clear()
                    selection_windows.clear()
                    if selector is not None:
                        selector.reset()
                    start = time.perf_counter()
                    pending_path = path.with_suffix(".pending.npz")
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        row = controller.run(seed, pending_path, reference_mode=mode,
                                             chart_selector=selector,
                                             localization_history_length=warmup_steps)
                    assert not record_coefficients or len(coefficients) == config.steps
                    row.update(experiment_version=version, solver=selected_solver,
                               coefficient_bounds_enabled=False, feature_standardization=False,
                               graph_cost_method=graph_cost if mode == "graph" else "not_used",
                               output_dimension=dimension, chart_selection_method=chart_selection_method,
                               elapsed_rollout_wall_s=time.perf_counter() - start,
                               solver_warnings=dict(Counter(str(w.message) for w in caught)))
                    with np.load(pending_path, allow_pickle=True) as trace:
                        arrays = {k: trace[k] for k in trace.files if k != "summary"}
                    assert row["terminal_reference_policy"] == policy
                    if mode == "terminal":
                        np.testing.assert_array_equal(arrays["ref"], np.tile(arrays["final_goal"], (320, 1)))
                        with np.load(source_panel(dimension, mode) / path.name, allow_pickle=True) as reference:
                            np.testing.assert_array_equal(arrays["final_goal"], reference["final_goal"])
                    metrics = true_error_metrics(arrays["distance"])
                    for key in ("success_30", "success_100", "mean_last_100"):
                        assert row[key] == metrics[key]
                    if record_coefficients:
                        arrays["g_coefficients"] = np.asarray(coefficients)
                    if selector is not None:
                        assert len(selector.residual_history) == config.steps
                        arrays.update(selection_windows=np.asarray(selection_windows),
                                      selection_residuals=np.asarray(selector.residual_history),
                                      selection_tie_count=np.asarray(selector.tie_history))
                    temporary = path.with_suffix(".npz.tmp")
                    with temporary.open("wb") as stream:
                        np.savez_compressed(stream, **arrays, summary=np.array(json.dumps(row)))
                    temporary.replace(path)
                    pending_path.unlink()
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
                        "chart_selection_method": data["signature"].get("chart_selection_method", "nearest_centroid"),
                        "warmup_steps": data["signature"].get("warmup_steps", 2),
                        "graph_cost_method": data["signature"].get("graph_cost_method", "centroid" if mode == "graph" else "not_used"),
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
                  r"terminal configuration; guided cases retain iterative $N=10$ terminal selection. "
                  + (r"Graph edge costs use chordal subspace distance divided by the square root of the transition count."
                     if any(c["graph_cost_method"] == "chordal" for c in columns) else "")
                  + (r" All cases select charts by the minimum full-window input-output projection residual,"
                     r" using 12 measured samples and a 12-step zero-input warmup."
                     if all(c["chart_selection_method"] == "full_window_projection" for c in columns) else "") + "}",
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
