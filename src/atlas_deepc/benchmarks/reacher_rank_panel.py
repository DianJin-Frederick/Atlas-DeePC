"""Shared execution of paper rank panels with explicit atlas/protocol inputs."""

from collections import Counter
from dataclasses import replace
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch
import warnings

from atlas_deepc.common.timing import summarize_timing
import numpy as np
from threadpoolctl import threadpool_limits
from atlas_deepc.baselines import unstandardized_atlas_deepc as experiment
from atlas_deepc.ablations import chordal_graph_cost as graph
from atlas_deepc.common.files import file_digest
from atlas_deepc.common.metrics import strict_success, true_error_metrics

ROOT = experiment.baseline.ROOT
DATA_PATH = ROOT / "data/reacher/select_iid_offline_hankel.npy"
CASES = ((8, "terminal"), (8, "graph"), (10, "terminal"), (10, "graph"))


def save_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)



def run_case(dimension, mode, seeds, root, solver="CLARABEL", *, source_atlas, source_manifest,
             config, reference_directory=None, rank=60,
             record_coefficients=True, timing_workers=1, graph_cost="chordal",
             chart_selection_method="full_window_projection", warmup_steps=None):
    if solver != "CLARABEL":
        raise ValueError("Paper rank panels use CLARABEL")
    selected_solver = solver
    if graph_cost != "chordal" or chart_selection_method != "full_window_projection":
        raise ValueError("Paper rank panels use chordal routing and projection-residual selection")
    version = "reacher_rank_panel_v1"
    root = Path(root)
    source_atlas, source_manifest = Path(source_atlas), Path(source_manifest)
    original = json.loads(source_manifest.read_text())
    if (original["output_dimension"], original["reference_mode"]) != (dimension, mode):
        raise ValueError("Source manifest does not describe this output/guidance case")
    if original["data_sha256"] != file_digest(DATA_PATH):
        raise ValueError("Source manifest uses a different offline dataset")
    original["atlas_sha256"] = file_digest(source_atlas)
    if not 1 <= rank < (2 + dimension) * 12:
        raise ValueError("Rank must be positive and below the Hankel row dimension")
    config = replace(config, svd_rank=rank)
    if config.lam_g != 0 or config.lam_du != 0 or config.steps != 300:
        raise ValueError("Paper rank panels require the unregularized 300-step protocol")
    if warmup_steps is None:
        warmup_steps = experiment.routed.pipe.L
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
                     graph_metric_sha256=(file_digest(graph.__file__)
                                          if graph_cost == "chordal" and mode == "graph" else None),
                     solver_options={"max_iter": 10000},
                     other_solver_options=f"CVXPY {solver} defaults; no solver switching or retries",
                     source_panel=str(source_manifest.parent),
                     source_manifest_sha256=file_digest(source_manifest),
                     package_versions={name: importlib.metadata.version(name)
                                       for name in ("numpy", "scipy", "cvxpy", "clarabel", "mujoco", "gymnasium")})
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
            np.savez_compressed(directory / "graph_geometry.npz", bases=bases,
                                labels=atlas["labels"], transition_counts=atlas["transition_counts"],
                                chordal_distance=distances, chordal_cost=atlas["routing_edge_costs"])
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
            def __init__(self, Q, R, lam_ini, u_limit, solver, **kwargs):
                assert solver == "CLARABEL"
                super().__init__(Q, R, lam_ini, u_limit, selected_solver, **kwargs)

            def solve(self, matrix, *args, **kwargs):
                assert self.solver == selected_solver
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
                        assert trace["y"].shape == (300, dimension)
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
                        np.testing.assert_array_equal(arrays["ref"], np.tile(arrays["final_goal"], (300, 1)))
                        if reference_directory is not None:
                            with np.load(Path(reference_directory) / path.name, allow_pickle=True) as reference:
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

