"""Paired chart-count/rank search for the current ten-output graph baseline."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import json
import multiprocessing
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from threadpoolctl import threadpool_limits

from atlas_deepc.ablations import projection_chordal_rank as previous
from atlas_deepc.reporting.four_case_tracking_table import full_rollout_rmse

four = previous.four
ROOT = four.ROOT
RESULT_ROOT = ROOT / "experiments/reacher/chart_count_rank_projection_chordal_v1"
SEEDS = previous.SEEDS
COUNTS = [1, 2, 4, 8, 16, 24, 32, 48, 64]
COARSE_RANKS = [48, 54, 60, 66, 72]
LIMITS = (28, 80)


def config_for(k, rank=60):
    return replace(four.experiment.default_config(), clusters=k, svd_rank=rank,
                   lam_g=0., lam_du=0.)


def prepare(root=RESULT_ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    reference = previous.control_signature(10, "graph")
    current = {k: v for k, v in asdict(config_for(48)).items() if k != "seed"}
    assert current == reference["config"]
    assert four.file_digest(four.__file__) == reference["runner_sha256"]
    assert four.file_digest(four.DATA_PATH) == reference["data_sha256"]
    manifest = {
        "chart_counts": COUNTS, "coarse_ranks": COARSE_RANKS, "rank_limits": LIMITS,
        "seeds": SEEDS, "output_dimension": 10, "config_at_k48_rank60": current,
        "data_sha256": reference["data_sha256"], "clustering_seed": current["cluster_seed"],
        "chart_selection": "full_window_projection", "graph_cost": "chordal",
        "selection_rule": previous.SELECTION_RULE, "workers": 4,
        "rank_search": "Extend a winning boundary by six (clamped to 28--80), then test all ranks within two of the coarse winner.",
        "k1": "One global chart, no inter-chart edges/waypoints; candidate policy collapses to nearest eligible posture.",
        "caveat": "Exploratory rank selection and reporting on the same 100 seeds; not exhaustive rank optimization or held-out testing.",
        "runner_sha256": four.file_digest(four.__file__),
        "study_sha256": four.file_digest(__file__), "source_sha256": reference["source_sha256"],
        "timing": "Parallel search, followed by serial one-thread timing on all 100 seeds per selected K/rank.",
    }
    manifest = json.loads(json.dumps(manifest))
    path = root / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "Study protocol changed"
    else:
        four.save_json(path, manifest)
        shutil.copytree(previous.control.RESULT_ROOT / "frozen_source", root / "frozen_source")
        shutil.copy2(__file__, root / "frozen_source" / Path(__file__).name)
    for k in COUNTS:
        folder = root / f"atlases/k{k:03d}"
        if k == 48:
            folder.mkdir(parents=True, exist_ok=True)
            source = previous.control.RESULT_ROOT / "output10/behavioral_atlas.npz"
            destination = folder / "behavioral_atlas.npz"
            if not destination.exists():
                shutil.copy2(source, destination)
            assert four.file_digest(destination) == reference["atlas_sha256"]
        else:
            destination, _ = four.experiment.prepare_atlas(
                config_for(k), folder, four.DATA_PATH, output_dimension=10)
        atlas = four.experiment.routed.load_atlas(destination)
        counts = np.bincount(atlas["labels"].astype(int), minlength=k)
        assert len(counts) == k and counts.sum() == 37800 and counts.min() >= 80
        edges = atlas["transition_counts"] >= current["edge_count_threshold"]
        np.fill_diagonal(edges, False)
        n_components = connected_components(csr_matrix(edges), directed=True,
                                            connection="strong", return_labels=False)
        source = {
            "kind": "frozen_offline_atlas_for_chart_count_ablation", "clusters": k,
            "atlas_sha256": four.file_digest(destination), "data_sha256": reference["data_sha256"],
            "terminal_reference_policy": "iterative_minimum_route_cost",
            "chart_sizes": counts.tolist(), "directed_edges": int(edges.sum()),
            "strongly_connected_components": int(n_components),
        }
        source_dir = folder / "source_graph"
        source_dir.mkdir(exist_ok=True)
        source_path = source_dir / "manifest.json"
        if source_path.exists():
            assert json.loads(source_path.read_text()) == source
        else:
            four.save_json(source_path, source)
    return manifest


def reused_directory(k, rank):
    if k == 48 and rank in previous.RANKS:
        return previous.panel_directory(previous.RESULT_ROOT, rank, 10, "graph")
    return None


def panel_path(root, k, rank, serial=False):
    reused = reused_directory(k, rank)
    if reused is not None and not serial:
        return reused
    return Path(root) / ("selected_serial" if serial else "sweep") / f"k{k:03d}_r{rank:03d}/output10/graph"


def validate_signature(signature, root, k, rank):
    expected = previous.control_signature(10, "graph")
    config = expected["config"].copy()
    config.update(clusters=k, svd_rank=rank)
    assert signature["config"] == config
    for key in ("solver", "solver_options", "source_sha256", "data_sha256",
                "chart_selection_method", "warmup_steps", "graph_cost_method",
                "terminal_reference_policy", "coefficient_bounds_enabled", "standardize"):
        assert signature[key] == expected[key], key
    assert signature["atlas_sha256"] == four.file_digest(
        Path(root) / f"atlases/k{k:03d}/behavioral_atlas.npz")


def summarize(directory, root, k, rank):
    panel = json.loads((directory / "summary.json").read_text())
    assert panel["seeds"] == SEEDS and len(panel["results"]) == 100
    validate_signature(panel["signature"], root, k, rank)
    with np.load(directory / "selection_geometry.npz") as geometry:
        assert geometry["bases"].shape == (k, 144, rank)
        np.testing.assert_array_equal(geometry["ranks"], np.full(k, rank))
        ids = geometry["chart_ids"].copy()
    distances, status_counts = [], {}
    waypoint_steps = 0
    for row in panel["results"]:
        name = f"seed{row['seed']}_control_results.npz"
        with np.load(directory / name, allow_pickle=True) as trace:
            distance = trace["distance"]
            assert distance.shape == (300,) and np.isfinite(distance).all()
            np.testing.assert_allclose(distance, np.linalg.norm(
                trace["y"][:, 6:8] - trace["final_goal"].reshape(-1)[6:8], axis=1))
            metrics = four.true_error_metrics(distance)
            for key in ("success_30", "success_100", "mean_last_100"):
                assert row[key] == metrics[key]
            for status in trace["status"]:
                status_counts[str(status)] = status_counts.get(str(status), 0) + 1
            z, residuals = trace["selection_windows"], trace["selection_residuals"]
            assert z.shape == (300, 144) and residuals.shape == (300, k)
            tolerance = 100 * np.finfo(float).eps * np.maximum(1., np.linalg.norm(z, axis=1))
            selected = ids[np.argmax(residuals <= residuals.min(axis=1, keepdims=True)
                                     + tolerance[:, None], axis=1)]
            np.testing.assert_array_equal(trace["chart"], selected)
            with np.load(previous.control.RESULT_ROOT / "output10/graph" / name) as original:
                np.testing.assert_array_equal(z[0], original["selection_windows"][0])
                np.testing.assert_array_equal(trace["final_goal"].reshape(-1)[6:8],
                                              original["final_goal"].reshape(-1)[6:8])
            if k == 1:
                assert np.all(trace["chart"] == 0)
                assert np.unique(trace["goal_index"]).size == 1
                np.testing.assert_allclose(trace["ref"], np.tile(trace["final_goal"], (300, 1)))
            waypoint_steps += int(np.any(np.abs(trace["ref"] - trace["final_goal"].reshape(1, -1))
                                        > 1e-10, axis=1).sum())
            distances.append(distance.copy())
    errors = 1000*np.mean(np.asarray(distances)[:, -100:], axis=1)
    _, per_seed = full_rollout_rmse(distances)
    modules = {r["module"]: r for r in panel["timing"]["modules"]}
    total = modules["Total online control"]
    successes = sum(four.strict_success(r) for r in panel["results"])
    return {"clusters": k, "rank": rank, "successes": successes, "success_rate": successes,
            "median_error_mm": float(np.median(errors)),
            **{f"p{q}_error_mm": float(np.percentile(errors, q)) for q in (90, 95, 99)},
            "median_rollout_rmse_mm": float(np.median(per_seed)),
            "per_seed_rmse_mm": dict(zip(map(str, SEEDS), per_seed.tolist())),
            "solver_status_counts": status_counts, "waypoint_step_fraction": waypoint_steps/32000,
            "failed_seeds": [r["seed"] for r in panel["results"] if not four.strict_success(r)],
            "unsolved_steps": sum(r["unsolved_steps"] for r in panel["results"]),
            "median_time_ms": total["median_ms"], "p95_time_ms": total["p95_ms"],
            "p99_time_ms": total["p99_ms"], "timing_modules": modules,
            "source_directory": str(directory), "summary_sha256": four.file_digest(directory / "summary.json")}


def metric_path(root, k, rank, serial=False):
    return Path(root) / ("serial_metrics" if serial else "metrics") / f"k{k:03d}_r{rank:03d}.json"


def run_case(root, k, rank, serial=False):
    root = Path(root)
    metric = metric_path(root, k, rank, serial)
    if metric.exists():
        record = json.loads(metric.read_text())
        assert four.file_digest(Path(record["source_directory"]) / "summary.json") == record["summary_sha256"]
        return record
    assert shutil.disk_usage(root).free > 3 * 2**30, "Less than 3 GiB free; stop safely"
    directory = panel_path(root, k, rank, serial)
    if reused_directory(k, rank) is None or serial:
        source_dir = root / f"atlases/k{k:03d}/source_graph"
        source = json.loads((source_dir / "manifest.json").read_text())
        atlas = source_dir.parent / "behavioral_atlas.npz"
        assert four.file_digest(atlas) == source["atlas_sha256"]
        # Reuse the unchanged rollout/QP implementation with a different frozen atlas.
        with patch.object(four, "verify_source", return_value=(source, atlas)), patch.object(
                four, "source_panel", return_value=source_dir), patch.object(
                four.experiment, "default_config", return_value=config_for(k, rank)):
            four.run_case(10, "graph", SEEDS, directory.parents[1], "CLARABEL", rank=rank,
                          record_coefficients=False, timing_workers=1 if serial else 4,
                          graph_cost="chordal", chart_selection_method="full_window_projection", warmup_steps=12)
    with threadpool_limits(limits=1):
        record = summarize(directory, root, k, rank)
    record["reused"] = reused_directory(k, rank) is not None and not serial
    record["serial"] = serial
    metric.parent.mkdir(exist_ok=True)
    four.save_json(metric, record)
    print(f"PANEL COMPLETE K={k} rank={rank}: {record['successes']}/100", flush=True)
    return record


def extend_ranks(records):
    best = previous.best_record(records)
    tested = {r["rank"] for r in records}
    rank = best["rank"]
    if rank == min(tested) and rank > LIMITS[0]:
        return [max(LIMITS[0], rank-6)]
    if rank == max(tested) and rank < LIMITS[1]:
        return [min(LIMITS[1], rank+6)]
    return []


def records_at(root, k):
    return [json.loads(p.read_text()) for p in sorted((Path(root)/"metrics").glob(f"k{k:03d}_r*.json"))]


def report(root=RESULT_ROOT):
    root = Path(root)
    records = [r for k in COUNTS for r in records_at(root, k)]
    complete = (root / "selection.json").exists() and all(
        len(list((root/"serial_metrics").glob(f"k{k:03d}_r*.json"))) == 1 for k in COUNTS)
    payload = {"records": records, "complete": complete, "evaluated_rollouts": 100*len(records),
               "new_rollouts": 100*sum(not r["reused"] for r in records),
               "best": [previous.best_record(records_at(root, k)) for k in COUNTS if records_at(root, k)],
               "serial": [json.loads(p.read_text()) for p in sorted((root/"serial_metrics").glob("*.json"))]}
    four.save_json(root / "summary.json", payload)
    return payload


def run_all(root=RESULT_ROOT):
    root = Path(root)
    prepare(root)
    def batch(jobs):
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(run_case, root, k, r) for k, r in jobs]
            for future in as_completed(futures):
                future.result()
                report(root)
    batch([(k, r) for k in COUNTS for r in COARSE_RANKS])
    plan_path = root / "refinement_plan.json"
    while not plan_path.exists():
        jobs = [(k, r) for k in COUNTS for r in extend_ranks(records_at(root, k))]
        if not jobs:
            break
        batch(jobs)
    if plan_path.exists():
        plans = json.loads(plan_path.read_text())
    else:
        plans = {str(k): list(range(max(LIMITS[0], previous.best_record(records_at(root, k))["rank"]-2),
                                   min(LIMITS[1], previous.best_record(records_at(root, k))["rank"]+2)+1))
                 for k in COUNTS}
        four.save_json(plan_path, plans)
    batch([(int(k), r) for k, ranks in plans.items() for r in ranks])
    selected = [previous.best_record(records_at(root, k)) for k in COUNTS]
    four.save_json(root / "selection.json", {"rule": previous.SELECTION_RULE, "selected": selected})
    for row in selected:
        serial = run_case(root, row["clusters"], row["rank"], serial=True)
        assert serial["failed_seeds"] == row["failed_seeds"]
        for key in ("median_error_mm", "median_rollout_rmse_mm", "p99_error_mm"):
            np.testing.assert_allclose(serial[key], row[key], rtol=1e-6, atol=1e-5)
        report(root)
    from atlas_deepc.notebook_tools.build_chart_count_rank_notebook import build
    build(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        prepare()
    elif args.report_only:
        print(json.dumps(report(), indent=2))
    else:
        run_all()
