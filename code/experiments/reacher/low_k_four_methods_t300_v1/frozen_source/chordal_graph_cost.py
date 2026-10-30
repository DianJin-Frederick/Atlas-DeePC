"""Paired centroid/chordal graph costs with frozen nearest terminal references."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.baselines import unstandardized_atlas_deepc as experiment
from atlas_deepc.atlas.candidate_goal_routing_experiment import shortest_path_tables
from atlas_deepc.benchmarks.baseline_1000_benchmark import summarize_timing
from atlas_deepc.common.files import file_digest
from atlas_deepc.common.metrics import strict_success

ROOT = experiment.baseline.ROOT
RESULT_ROOT = ROOT / "experiments/reacher/chordal_graph_cost_v1"
DATA_PATH = ROOT / "data/reacher/select_iid_offline_hankel.npy"
VERSION = "fixed_terminal_chordal_graph_v1"
METHODS = ("centroid", "chordal")
TERMINAL_POLICY = "fixed_nearest_eligible_fingertip"


def chordal_distances(bases):
    """Projection chordal distance; equals sqrt(r - ||U.T V||_F^2) at equal rank."""
    bases = [np.asarray(b, dtype=float) for b in bases]
    if not bases or any(b.ndim != 2 or b.shape[0] != bases[0].shape[0] for b in bases):
        raise ValueError("Bases must have a common ambient dimension")
    for b in bases:
        if not np.all(np.isfinite(b)) or not np.allclose(b.T @ b, np.eye(b.shape[1]), atol=1e-8):
            raise ValueError("Chart bases must have orthonormal columns")
    distance = np.zeros((len(bases), len(bases)))
    for k, u in enumerate(bases):
        for j in range(k):
            v = bases[j]
            squared = (u.shape[1] + v.shape[1]) / 2 - np.sum((u.T @ v)**2)
            distance[k, j] = distance[j, k] = np.sqrt(max(0., squared))
    return distance


def graph_costs(distance, counts, threshold):
    distance, counts = np.asarray(distance), np.asarray(counts)
    if distance.shape != counts.shape or distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("Distance/count matrices must be matching square matrices")
    edge = (counts >= threshold) & ~np.eye(len(counts), dtype=bool)
    if np.any(~np.isfinite(distance[edge])) or np.any(distance[edge] < 0):
        raise ValueError("Edge distances must be finite and nonnegative")
    cost = np.full(counts.shape, np.inf)
    cost[edge] = distance[edge] / np.sqrt(counts[edge])
    return cost


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def prepare_dimension(dimension, config=None, root=RESULT_ROOT):
    config = experiment.default_config() if config is None else config
    folder = Path(root) / f"output{dimension}"
    atlas_path, atlas_manifest = experiment.prepare_atlas(
        config, folder, DATA_PATH, output_dimension=dimension)
    signature = {
        "version": VERSION, "config": asdict(config), "output_dimension": dimension,
        "atlas_sha256": file_digest(atlas_path), "data_sha256": file_digest(DATA_PATH),
        "source_sha256": experiment.execution_signature(
            config, DATA_PATH, atlas_path, dimension, "graph",
            terminal_reference_policy=TERMINAL_POLICY)["source_sha256"],
        "ablation_sha256": file_digest(__file__),
    }
    manifest_path = folder / "graph_manifest.json"
    artifact_path = folder / "graph_geometry.npz"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["signature"] != signature or saved["artifact_sha256"] != file_digest(artifact_path):
            raise ValueError("Incompatible graph artifacts; choose a fresh root")
        return folder
    with experiment.output_context(dimension), threadpool_limits(limits=1):
        H = experiment.load_hankel(DATA_PATH, dimension)
        atlas = experiment.routed.load_atlas(atlas_path)
        started = time.perf_counter()
        cache = experiment.routed.precompute_chart_matrix_cache(H, atlas, config, geometry="linear")
        bases = np.stack([cache[(k,)][0]["basis"] for k in range(config.clusters)])
        if bases.shape != (config.clusters, (2 + dimension) * experiment.routed.pipe.L, config.svd_rank):
            raise ValueError("Unexpected full-trajectory basis dimensions")
        svd_seconds = time.perf_counter() - started
        started = time.perf_counter()
        chordal = chordal_distances(bases)
        chordal_seconds = time.perf_counter() - started
        counts = atlas["transition_counts"]
        centroid_cost = graph_costs(atlas["center_distance"], counts, config.edge_count_threshold)
        chordal_cost = graph_costs(chordal, counts, config.edge_count_threshold)
        np.testing.assert_array_equal(np.isfinite(centroid_cost), np.isfinite(chordal_cost))
        np.savez_compressed(artifact_path, bases=bases, labels=atlas["labels"],
                            transition_counts=counts, centroid_distance=atlas["center_distance"],
                            chordal_distance=chordal, centroid_cost=centroid_cost,
                            chordal_cost=chordal_cost)
    write_json(manifest_path, {"signature": signature, "artifact_sha256": file_digest(artifact_path),
                              "basis_precomputation_seconds": svd_seconds,
                              "chordal_distance_seconds": chordal_seconds,
                              "atlas_fit_seconds": atlas_manifest["offline_clustering_and_graph_seconds"]})
    return folder


def run_condition(dimension, method, seeds, root=RESULT_ROOT, workers=1):
    if method not in METHODS:
        raise ValueError(method)
    config = experiment.default_config()
    folder = Path(root) / f"output{dimension}"
    output = folder / method
    output.mkdir(parents=True, exist_ok=True)
    geometry_manifest = json.loads((folder / "graph_manifest.json").read_text())
    if geometry_manifest["signature"]["ablation_sha256"] != file_digest(__file__):
        raise ValueError("Ablation code changed after preparation")
    if geometry_manifest["artifact_sha256"] != file_digest(folder / "graph_geometry.npz"):
        raise ValueError("Graph geometry changed")
    signature = {"geometry": geometry_manifest, "method": method,
                 "terminal_reference_policy": TERMINAL_POLICY, "graph_waypoints_enabled": True,
                 "workers": workers, "numerical_threads_per_worker": 1,
                 "timing_note": "Shared-machine timings; not an isolated latency benchmark"}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != signature:
        raise ValueError("Incompatible rollout cache")
    if not manifest_path.exists() and list(output.glob("seed*.npz")):
        raise ValueError("Unversioned rollout cache")
    write_json(manifest_path, signature)
    with experiment.output_context(dimension), threadpool_limits(limits=1):
        H = experiment.load_hankel(DATA_PATH, dimension)
        atlas = experiment.routed.load_atlas(folder / "behavioral_atlas.npz")
        with np.load(folder / "graph_geometry.npz") as saved:
            bases = saved["bases"].copy()
            atlas["routing_edge_costs"] = saved[f"{method}_cost"].copy()
            np.testing.assert_array_equal(saved["labels"], atlas["labels"])
        cache = {(k,): ({"basis": bases[k], "offset": np.zeros((bases.shape[1], 1))},
                         np.flatnonzero(atlas["labels"] == k)) for k in range(config.clusters)}
        # Both conditions consume the exact same persisted bases, with no g boxes.
        with patch.object(experiment.routed, "precompute_chart_matrix_cache", return_value=cache):
            controller = experiment.baseline.BaselineAtlasDeePC(
                H, atlas, config, terminal_reference_policy=TERMINAL_POLICY)
        started = time.perf_counter()
        controller.route_tables = shortest_path_tables(atlas, config.edge_count_threshold)
        routing_setup_seconds = time.perf_counter() - started
        rows = []
        for i, seed in enumerate(seeds, 1):
            path = output / f"seed{seed}_control_results.npz"
            if path.exists():
                with np.load(path, allow_pickle=True) as saved:
                    row = json.loads(saved["summary"].item())
                assert row["graph_cost_method"] == method and row["output_dimension"] == dimension
            else:
                row = controller.run(seed, path, reference_mode="graph")
                with np.load(path, allow_pickle=True) as saved:
                    arrays = {k: saved[k] for k in saved.files if k != "summary"}
                assert np.all(arrays["goal_index"] == arrays["goal_index"][0])
                assert np.all(arrays["goal_chart"] == arrays["goal_chart"][0])
                row.update(graph_cost_method=method, output_dimension=dimension,
                           experiment_version=VERSION, coefficient_bounds_enabled=False,
                           feature_standardization=False)
                temporary = path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, **arrays, summary=np.array(json.dumps(row)))
                temporary.replace(path)
            row["result_path"] = str(path.resolve())
            rows.append(row)
            print(f"R{dimension} {method} {i}/{len(seeds)} seed {seed}: "
                  f"success={strict_success(row)}, tail={1000*row['mean_last_100']:.3f} mm", flush=True)
        summary = {"signature": signature, "results": rows, "seeds": list(seeds),
                   "routing_setup_seconds": routing_setup_seconds,
                   "timing": summarize_timing([r["result_path"] for r in rows])}
        write_json(output / "summary.json", summary)
    return {"dimension": dimension, "method": method, "successes": sum(map(strict_success, rows))}


def run_all(seeds=range(211, 311), root=RESULT_ROOT, workers=4):
    seeds = list(map(int, seeds))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and distinct")
    for dimension in (8, 10):
        prepare_dimension(dimension, root=root)
    jobs = [(dimension, method) for dimension in (8, 10) for method in METHODS]
    if workers == 1:
        return [run_condition(d, m, seeds, root, workers) for d, m in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_condition, d, m, seeds, root, workers) for d, m in jobs]
        return [future.result() for future in futures]


def load_results(seeds=range(211, 311), root=RESULT_ROOT):
    panels = {}
    for dimension in (8, 10):
        for method in METHODS:
            summary = json.loads((Path(root) / f"output{dimension}" / method / "summary.json").read_text())
            by_seed = {r["seed"]: r for r in summary["results"]}
            if set(seeds) - by_seed.keys():
                raise ValueError("Incomplete panel")
            panels[dimension, method] = [by_seed[s] for s in seeds]
    for dimension in (8, 10):
        for a, b in zip(panels[dimension, "centroid"], panels[dimension, "chordal"]):
            with np.load(a["result_path"], allow_pickle=True) as first, np.load(b["result_path"], allow_pickle=True) as second:
                np.testing.assert_array_equal(first["final_goal"], second["final_goal"])
                np.testing.assert_array_equal(first["goal_index"], second["goal_index"])
                np.testing.assert_array_equal(first["goal_chart"], second["goal_chart"])
    return panels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=211)
    parser.add_argument("--seed-stop", type=int, default=310)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()
    print(run_all(range(args.seed_start, args.seed_stop + 1), args.root, args.workers))
