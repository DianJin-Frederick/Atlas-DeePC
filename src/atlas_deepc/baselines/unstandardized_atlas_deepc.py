"""Shared eight/ten-output Atlas-DeePC without feature standardization."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from atlas_deepc.common.paths import source_hashes, source_path

from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.baselines import baseline_atlas_deepc as baseline
from atlas_deepc.atlas import routed_deepc_experiment as routed
from atlas_deepc.atlas.chart_selection import FullWindowProjectionSelector
from atlas_deepc.common.timing import summarize_timing
from atlas_deepc.common.files import file_digest
from atlas_deepc.common.hankel import remove_joint_sum_channels


VERSION = "unstandardized_shared_runner_v7"


def terminal_policy(reference_mode, override=None):
    if reference_mode not in ("graph", "terminal"):
        raise ValueError("reference_mode must be graph or terminal")
    if override is not None:
        if override not in ("iterative_minimum_route_cost", "fixed_nearest_eligible_fingertip"):
            raise ValueError("Unknown terminal reference policy")
        return override
    return ("iterative_minimum_route_cost" if reference_mode == "graph"
            else "fixed_nearest_eligible_fingertip")

def result_root(output_dimension=10):
    validate_dimension(output_dimension)
    return baseline.ROOT / "experiments/reacher/atlas_builds" / f"output{output_dimension}"


def validate_dimension(dimension):
    if dimension not in (8, 10):
        raise ValueError("output_dimension must be 8 or 10")


def output_context(output_dimension=10):
    """Scope the legacy controller's output dimension without changing other runs."""
    validate_dimension(output_dimension)
    return patch.object(routed.pipe, "P", output_dimension)


def load_hankel(data_path=baseline.DATA_PATH, output_dimension=10):
    """Keep all original samples/inputs; remove rows 8:10 in every output block."""
    validate_dimension(output_dimension)
    source = np.load(data_path, mmap_mode="r")
    input_rows = routed.pipe.M * routed.pipe.L
    if source.ndim != 2 or source.shape[0] != (routed.pipe.M + 10) * routed.pipe.L:
        raise ValueError("Expected the original ten-output offline Hankel matrix")
    if output_dimension == 10:
        return source
    return remove_joint_sum_channels(source)


def default_config():
    """The paper controller defaults, without coefficient or input-increment penalties."""
    return baseline.default_config()


def prepare_atlas(config, output_dir=None, data_path=baseline.DATA_PATH, *, output_dimension=10):
    """Reuse the offline Hankel data; refit memberships, graph and medoids only."""
    if config.clustering_method != "minibatch_kmeans":
        raise ValueError("This notebook tests the baseline MiniBatchKMeans algorithm")
    validate_dimension(output_dimension)
    output_dir = Path(output_dir) if output_dir is not None else result_root(output_dimension)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = {
        "version": VERSION, "standardize": False, "data_sha256": file_digest(data_path),
        "output_dimension": output_dimension, "retained_output_rows": list(range(output_dimension)),
        "runner_sha256": file_digest(__file__),
        "clustering_config": {k: v for k, v in asdict(config).items()
                              if k.startswith("clustering_") or k in ("clusters", "cluster_seed")},
        "feature_code_sha256": file_digest(routed.__file__),
    }
    manifest_path = output_dir / "atlas_manifest.json"
    atlas_path = output_dir / "behavioral_atlas.npz"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["signature"] != signature or not atlas_path.exists():
            raise ValueError("Incompatible atlas cache; choose a fresh OUTPUT_ROOT")
        assert file_digest(atlas_path) == manifest["atlas_sha256"]
        return atlas_path, manifest
    if atlas_path.exists():
        raise ValueError("Unversioned atlas cache; choose a fresh OUTPUT_ROOT")
    with output_context(output_dimension), threadpool_limits(limits=1):
        H = load_hankel(data_path, output_dimension)
        start = time.perf_counter()
        atlas = routed.fit_behavioral_atlas(H, config, standardize=False)
        fit_seconds = time.perf_counter() - start
        assert np.all(atlas["feature_mean"] == 0.)
        assert np.all(atlas["feature_scale"] == 1.)
        atlas["feature_standardization"] = np.array(False)
        atlas["output_dimension"] = np.array(output_dimension)
        assert atlas["medoid_states"].shape == (config.clusters, output_dimension)
        routed.save_atlas(atlas_path, atlas, config)
    manifest = {"signature": signature, "atlas_sha256": file_digest(atlas_path),
                "offline_clustering_and_graph_seconds": fit_seconds}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return atlas_path, manifest


def execution_signature(config, data_path, atlas_path, output_dimension=10, reference_mode="graph",
                        *, chart_selection_method="nearest_centroid", warmup_steps=None,
                        terminal_reference_policy=None):
    validate_dimension(output_dimension)
    if reference_mode not in ("graph", "terminal"):
        raise ValueError("reference_mode must be graph or terminal")
    if chart_selection_method not in ("nearest_centroid", "full_window_projection"):
        raise ValueError("Unknown chart selection method")
    warmup_steps = routed.pipe.T_INI if warmup_steps is None else int(warmup_steps)
    if warmup_steps < routed.pipe.T_INI or (chart_selection_method == "full_window_projection"
                                           and warmup_steps != routed.pipe.L):
        raise ValueError("Full-window projection requires exactly L warmup steps")
    return {
        "version": VERSION, "standardize": False, "coefficient_bounds_enabled": False,
        "output_dimension": output_dimension, "retained_output_rows": list(range(output_dimension)),
        "config": {k: v for k, v in asdict(config).items() if k != "seed"},
        "solver": "CLARABEL", "reference_mode": reference_mode,
        "candidate_count": (1 if terminal_policy(reference_mode, terminal_reference_policy)
                            == "fixed_nearest_eligible_fingertip" else 10),
        "terminal_reference_policy": terminal_policy(reference_mode, terminal_reference_policy),
        "chart_selection_method": chart_selection_method, "warmup_steps": warmup_steps,
        "workers": 1, "numerical_threads": 1,
        "data_sha256": file_digest(data_path), "atlas_sha256": file_digest(atlas_path),
        "source_sha256": source_hashes((
            "unstandardized_atlas_deepc.py", "baseline_atlas_deepc.py", "routed_deepc_experiment.py",
            "expert_pipeline_search.py", "candidate_goal_routing_experiment.py",
            "chart_selection.py")),
    }


def load_panel(seeds, config, atlas_path, output_dir, data_path=baseline.DATA_PATH, *, output_dimension=10,
               reference_mode="graph", chart_selection_method="nearest_centroid", warmup_steps=None,
               terminal_reference_policy=None):
    seeds = [int(s) for s in seeds]
    validate_dimension(output_dimension)
    output_dir = Path(output_dir) if output_dir is not None else result_root(output_dimension)
    manifest = output_dir / "manifest.json"
    signature = execution_signature(config, data_path, atlas_path, output_dimension, reference_mode,
                                    chart_selection_method=chart_selection_method, warmup_steps=warmup_steps,
                                    terminal_reference_policy=terminal_reference_policy)
    if not manifest.exists():
        return [], seeds
    if json.loads(manifest.read_text()) != signature:
        raise ValueError("Incompatible rollout cache; choose a fresh EVALUATION_DIR")
    rows, missing = [], []
    for seed in seeds:
        path = output_dir / f"seed{seed}_control_results.npz"
        if not path.exists():
            missing.append(seed)
            continue
        with np.load(path, allow_pickle=True) as trace:
            row = json.loads(trace["summary"].item())
            assert row["experiment_version"] == VERSION and row["seed"] == seed
            assert row["coefficient_bounds_enabled"] is False and row["feature_standardization"] is False
            assert row["output_dimension"] == output_dimension
            assert row["graph_waypoints_enabled"] == (reference_mode == "graph")
            assert row["terminal_reference_policy"] == signature["terminal_reference_policy"]
            assert row["candidate_count"] == signature["candidate_count"]
            assert row["chart_selection_method"] == chart_selection_method
            assert row["localization_history_length"] == signature["warmup_steps"]
            if chart_selection_method == "full_window_projection":
                assert trace["selection_residuals"].shape == (config.steps, config.clusters)
                assert trace["selection_tie_count"].shape == (config.steps,)
            if signature["terminal_reference_policy"] == "fixed_nearest_eligible_fingertip":
                assert row["dynamic_terminal_reference"] is False
                assert row["graph_route_cost_selection_enabled"] is False
                assert np.all(trace["goal_index"] == trace["goal_index"][0])
                assert np.all(trace["goal_chart"] == trace["goal_chart"][0])
            if reference_mode == "terminal":
                assert np.all(trace["ref"] == trace["ref"][0])
                assert np.all(trace["path_length"] == 1)
                assert np.all(np.isnan(trace["goal_route_cost"]))
            assert trace["y"].shape == trace["ref"].shape == (config.steps, output_dimension)
            assert trace["g_coefficients"].shape == (config.steps, config.svd_rank)
            for key, value in signature["config"].items():
                assert row[key] == value, key
        rows.append({**row, "result_path": str(path.resolve())})
    return rows, missing


def run_panel(seeds, config, atlas_path, output_dir, data_path=baseline.DATA_PATH, *, output_dimension=10,
              reference_mode="graph", chart_selection_method="nearest_centroid", warmup_steps=None,
              terminal_reference_policy=None):
    seeds = [int(s) for s in seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be a nonempty set of distinct integers")
    validate_dimension(output_dimension)
    output_dir = Path(output_dir) if output_dir is not None else result_root(output_dimension)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = execution_signature(config, data_path, atlas_path, output_dimension, reference_mode,
                                    chart_selection_method=chart_selection_method, warmup_steps=warmup_steps,
                                    terminal_reference_policy=terminal_reference_policy)
    manifest = output_dir / "manifest.json"
    if not manifest.exists():
        if any(output_dir.glob("seed*.npz")):
            raise ValueError("Unversioned rollout cache")
        manifest.write_text(json.dumps(signature, indent=2))
    _, missing = load_panel(seeds, config, atlas_path, output_dir, data_path,
                           output_dimension=output_dimension, reference_mode=reference_mode,
                           chart_selection_method=chart_selection_method, warmup_steps=warmup_steps,
                           terminal_reference_policy=terminal_reference_policy)
    with output_context(output_dimension), threadpool_limits(limits=1):
        H = load_hankel(data_path, output_dimension)
        atlas = routed.load_atlas(atlas_path)
        assert int(atlas["output_dimension"]) == output_dimension
        assert np.all(atlas["feature_mean"] == 0.) and np.all(atlas["feature_scale"] == 1.)
        controller = baseline.BaselineAtlasDeePC(
            H, atlas, config, terminal_reference_policy=signature["terminal_reference_policy"])
        selector = (FullWindowProjectionSelector(controller.matrix_cache, routed.pipe.M,
                                                 output_dimension, routed.pipe.L)
                    if chart_selection_method == "full_window_projection" else None)
        # Drop legacy box metadata as well as relying on the shared no-box QP.
        for matrix, _ in controller.matrix_cache.values():
            assert matrix["basis"].shape == ((routed.pipe.M + output_dimension) * routed.pipe.L,
                                             config.svd_rank)
            matrix.pop("g_min", None)
            matrix.pop("g_max", None)
        coefficients = []
        solve = routed.pipe.DeePCController.solve

        def logged_solve(instance, matrix, *args, **kwargs):
            assert instance.solver == "CLARABEL"
            assert instance.Q.shape == (output_dimension, output_dimension)
            assert matrix.get("g_min") is None and matrix.get("g_max") is None
            result = solve(instance, matrix, *args, **kwargs)
            coefficients.append(np.full(config.svd_rank, np.nan)
                                if result[2] is None else result[2].ravel().copy())
            return result

        with patch.object(routed.pipe.DeePCController, "solve", logged_solve):
            for index, seed in enumerate(missing, 1):
                coefficients.clear()
                if selector is not None:
                    selector.reset()
                path = output_dir / f"seed{seed}_control_results.npz"
                row = controller.run(seed, path, reference_mode=reference_mode,
                                     chart_selector=selector,
                                     localization_history_length=signature["warmup_steps"])
                assert len(coefficients) == config.steps
                row.update(experiment_version=VERSION, coefficient_bounds_enabled=False,
                           feature_standardization=False, output_dimension=output_dimension,
                           chart_selection_method=chart_selection_method,
                           removed_output_channels=["cos(q1+q2)", "sin(q1+q2)"] if output_dimension == 8 else [])
                with np.load(path, allow_pickle=True) as trace:
                    arrays = {k: trace[k] for k in trace.files if k != "summary"}
                if selector is not None:
                    arrays.update(selection_residuals=np.asarray(selector.residual_history),
                                  selection_tie_count=np.asarray(selector.tie_history))
                np.savez_compressed(path, **arrays, g_coefficients=np.asarray(coefficients),
                                    summary=np.array(json.dumps(row)))
                print(f"{index}/{len(missing)} new rollouts: seed {seed}, "
                      f"success={row['success_30'] and row['success_100']}, "
                      f"final100={1000*row['mean_last_100']:.3f} mm", flush=True)
    rows, missing = load_panel(seeds, config, atlas_path, output_dir, data_path,
                              output_dimension=output_dimension, reference_mode=reference_mode,
                              chart_selection_method=chart_selection_method, warmup_steps=warmup_steps,
                              terminal_reference_policy=terminal_reference_policy)
    assert not missing
    summary = {"signature": signature, "seeds": seeds, "results": rows,
               "timing": summarize_timing([r["result_path"] for r in rows])}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
