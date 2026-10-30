"""Rank sweep for the current four baselines, with independent serial timing."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.ablations import four_baseline_input_smoothing as control
from atlas_deepc.reporting.four_case_tracking_table import full_rollout_rmse

four = control.four
ROOT = four.ROOT
RESULT_ROOT = ROOT / "experiments/reacher/projection_chordal_rank_lamg0_lamdu0_clarabel_v1"
NOTEBOOK = ROOT / "notebooks/reacher/ablations/ablation_linear_subspace_rank.ipynb"
RANKS = list(range(28, 71))
SEEDS = list(range(211, 311))
SELECTION_RULE = "Highest strict success count, then lowest median rollout RMSE, then lowest rank"


def best_record(records):
    if not records:
        raise ValueError("No completed ranks")
    return min(records, key=lambda r: (-r["successes"], r["median_rollout_rmse_mm"], r["rank"]))


def control_signature(p, mode):
    return json.loads((control.RESULT_ROOT / f"output{p}/{mode}/manifest.json").read_text())


def check_signature(signature, p, mode, rank, workers, initial=False):
    expected = control_signature(p, mode)
    assert expected["config"]["lam_du"] == expected["config"]["lam_g"] == 0.
    assert expected["config"]["lam_ini"] == 1e5
    expected["config"]["svd_rank"] = rank
    expected["record_coefficients"] = False
    expected["timing_workers"] = workers
    if initial:
        expected = {k: expected[k] for k in signature}
        expected["version"] = signature["version"]
    assert signature == expected, "Rank sweep changed another baseline setting"


def prepare(root, ranks, seeds, workers):
    root = Path(root)
    ranks, seeds = list(ranks), list(seeds)
    if not ranks or len(set(ranks)) != len(ranks) or any(r < 1 or r >= 120 for r in ranks):
        raise ValueError("Use unique ranks between 1 and 119")
    if seeds != SEEDS or workers not in range(1, 5):
        raise ValueError("This paired study uses seeds 211--310 and one to four workers")
    root.mkdir(parents=True, exist_ok=True)
    controls = {}
    for p, mode in four.CASES:
        source = control.RESULT_ROOT / f"output{p}/{mode}/manifest.json"
        previous = control_signature(p, mode)
        config = replace(four.experiment.default_config(), lam_g=0., lam_du=0., svd_rank=60)
        atlas = source.parent.parent / "behavioral_atlas.npz"
        signature = four.experiment.execution_signature(
            config, four.DATA_PATH, atlas, p, mode,
            chart_selection_method="full_window_projection", warmup_steps=12)
        check_signature(signature, p, mode, 60, workers, initial=True)
        assert four.file_digest(four.__file__) == previous["runner_sha256"]
        controls[f"{p}_{mode}"] = four.file_digest(source)
    manifest = {
        "ranks": ranks, "seeds": seeds, "workers": workers, "numerical_threads_per_worker": 1,
        "lambda_g": 0., "lambda_du": 0., "lambda_ini": 1e5, "solver": "CLARABEL",
        "charts": 48, "chart_selection": "full_window_projection", "graph_cost": "chordal",
        "warmup_steps": 12, "controlled_steps": 320, "mL_plus_n": 28,
        "selection_rule": SELECTION_RULE, "control_manifests": controls,
        "data_sha256": four.file_digest(four.DATA_PATH),
        "sweep_source_sha256": four.file_digest(__file__),
        "timing_protocol": "Parallel rank search; selected ranks rerun serially on all 100 seeds",
        "selection_caveat": "Rank selected and reported on the same exploratory seeds, not held-out performance",
        "rank60_reuse": str(control.RESULT_ROOT),
    }
    path = root / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "Incompatible sweep root"
    else:
        four.save_json(path, manifest)
        shutil.copytree(control.RESULT_ROOT / "frozen_source", root / "frozen_source")
        shutil.copy2(__file__, root / "frozen_source" / Path(__file__).name)
    # Prepare shared atlas paths before workers start. Never recluster for rank changes.
    for rank in ranks:
        if rank == 60:
            continue
        for p in (8, 10):
            source = control.RESULT_ROOT / f"output{p}/behavioral_atlas.npz"
            destination = root / f"rank{rank:03d}/output{p}/behavioral_atlas.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(source, destination)
            assert four.file_digest(source) == four.file_digest(destination)
    return manifest


def panel_directory(root, rank, p, mode):
    if rank == 60:
        return control.RESULT_ROOT / f"output{p}/{mode}"
    return Path(root) / f"rank{rank:03d}/output{p}/{mode}"


def summarize_panel(directory, rank, p, mode):
    directory = Path(directory)
    panel = json.loads((directory / "summary.json").read_text())
    assert panel["seeds"] == SEEDS and len(panel["results"]) == len(SEEDS)
    distances, statuses = [], {}
    with np.load(directory / "selection_geometry.npz") as geometry:
        assert geometry["bases"].shape == (48, (2+p)*12, rank)
        np.testing.assert_array_equal(geometry["ranks"], np.full(48, rank))
        chart_ids = geometry["chart_ids"]
    for row in panel["results"]:
        seed = row["seed"]
        assert row["svd_rank"] == rank and row["lam_g"] == 0 and row["solver"] == "CLARABEL"
        with np.load(directory / f"seed{seed}_control_results.npz", allow_pickle=True) as trace:
            distance = trace["distance"]
            assert distance.shape == (320,) and np.all(np.isfinite(distance))
            np.testing.assert_allclose(distance, np.linalg.norm(
                trace["y"][:, 6:8]-trace["final_goal"].reshape(-1)[6:8], axis=1))
            measured = four.true_error_metrics(distance)
            for key in ("success_30", "success_100", "mean_last_100"):
                assert row[key] == measured[key]
            status = [str(s) for s in trace["status"]]
            assert row["unsolved_steps"] == sum(s not in ("optimal", "optimal_inaccurate") for s in status)
            for s in status:
                statuses[s] = statuses.get(s, 0)+1
            z, residuals = trace["selection_windows"], trace["selection_residuals"]
            assert z.shape == (320, (2+p)*12) and residuals.shape == (320, 48)
            tolerance = 100*np.finfo(float).eps*np.maximum(1., np.linalg.norm(z, axis=1))
            ties = residuals <= residuals.min(axis=1, keepdims=True)+tolerance[:, None]
            np.testing.assert_array_equal(trace["chart"], chart_ids[np.argmax(ties, axis=1)])
            if mode == "terminal":
                np.testing.assert_array_equal(trace["ref"], np.tile(trace["final_goal"], (320, 1)))
            with np.load(control.RESULT_ROOT / f"output{p}/{mode}/seed{seed}_control_results.npz") as old:
                np.testing.assert_array_equal(z[0], old["selection_windows"][0])
                np.testing.assert_array_equal(trace["final_goal"].reshape(-1)[6:8], old["final_goal"].reshape(-1)[6:8])
            distances.append(distance.copy())
    errors = 1000*np.mean(np.asarray(distances)[:, -100:], axis=1)
    pooled, per_seed = full_rollout_rmse(distances)
    modules = {r["module"]: r for r in panel["timing"]["modules"]}
    total = modules["Total online control"]
    successes = sum(four.strict_success(r) for r in panel["results"])
    return {"rank": rank, "output_dimension": p, "reference_mode": mode,
            "graph_guidance": mode == "graph", "seeds": len(SEEDS), "successes": successes,
            "success_rate": 100*successes/len(SEEDS), "median_error_mm": float(np.median(errors)),
            **{f"p{q}_error_mm": float(np.percentile(errors, q)) for q in (90, 95, 99)},
            "median_rollout_rmse_mm": float(np.median(per_seed)), "pooled_rmse_mm": pooled,
            "per_seed_rmse_mm": {str(s): float(e) for s, e in zip(SEEDS, per_seed)},
            "unsolved_steps": sum(r["unsolved_steps"] for r in panel["results"]),
            "solver_status_counts": statuses,
            "failed_seeds": [r["seed"] for r in panel["results"] if not four.strict_success(r)],
            "median_time_ms": total["median_ms"], "p95_time_ms": total["p95_ms"],
            "p99_time_ms": total["p99_ms"], "timing_modules": modules,
            "timing_workers": panel["signature"]["timing_workers"],
            "summary_sha256": four.file_digest(directory / "summary.json"),
            "source_directory": str(directory), "rank60_reused": rank == 60}


def run_case(root, rank, p, mode, workers, serial=False):
    root = Path(root)
    directory = (root / "selected_serial" / f"rank{rank:03d}/output{p}/{mode}" if serial
                 else panel_directory(root, rank, p, mode))
    if not (rank == 60 and not serial):
        source = four.verify_source(p, mode)
        config = replace(four.experiment.default_config(), lam_du=0.)
        case_root = directory.parents[1]
        with patch.object(four, "verify_source", return_value=source), patch.object(
                four.experiment, "default_config", return_value=config):
            panel = four.run_case(p, mode, SEEDS, case_root, "CLARABEL", rank=rank,
                record_coefficients=False, timing_workers=workers, graph_cost="chordal",
                chart_selection_method="full_window_projection", warmup_steps=12)
        check_signature(panel["signature"], p, mode, rank, workers)
    with threadpool_limits(limits=1):
        record = summarize_panel(directory, rank, p, mode)
    record["rank60_reused"] = rank == 60 and not serial
    destination = (root / "selected_serial" if serial else root / "panel_metrics")
    destination.mkdir(parents=True, exist_ok=True)
    four.save_json(destination / f"rank{rank:03d}_p{p}_{mode}.json", record)
    return {k: record[k] for k in ("rank", "output_dimension", "reference_mode", "successes", "median_rollout_rmse_mm")}


def report(root=RESULT_ROOT):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    records, missing = [], []
    for rank in manifest["ranks"]:
        for p, mode in four.CASES:
            path = root / "panel_metrics" / f"rank{rank:03d}_p{p}_{mode}.json"
            if not path.exists():
                missing.append([rank, p, mode])
                continue
            record = json.loads(path.read_text())
            directory = Path(record["source_directory"])
            assert four.file_digest(directory / "summary.json") == record["summary_sha256"]
            records.append(record)
    best = [best_record([r for r in records if (r["output_dimension"], r["reference_mode"]) == (p, mode)])
            for p, mode in four.CASES if any((r["output_dimension"], r["reference_mode"]) == (p, mode) for r in records)]
    result = {"manifest": manifest, "records": records, "missing_panels": missing,
              "complete": not missing, "best_completed_ranks": best,
              "completed_rollouts": 100*len(records), "requested_rollouts": 400*len(manifest["ranks"])}
    four.save_json(root / "summary.json", result)
    return result


def serial_winners(root=RESULT_ROOT):
    root = Path(root)
    sweep = report(root)
    assert sweep["complete"], "Do not finalize best ranks from an incomplete sweep"
    selected = []
    for record in sweep["best_completed_ranks"]:
        rank, p, mode = (record[k] for k in ("rank", "output_dimension", "reference_mode"))
        run_case(root, rank, p, mode, 1, serial=True)
        serial = json.loads((root / "selected_serial" / f"rank{rank:03d}_p{p}_{mode}.json").read_text())
        assert serial["failed_seeds"] == record["failed_seeds"]
        for key in ("median_error_mm", "p90_error_mm", "p95_error_mm", "p99_error_mm", "median_rollout_rmse_mm"):
            np.testing.assert_allclose(serial[key], record[key], rtol=1e-6, atol=1e-5)
        serial["sweep_source_directory"] = record["source_directory"]
        selected.append(serial)
    result = {"selection_rule": SELECTION_RULE, "ranks_tested": sweep["manifest"]["ranks"],
              "seeds": SEEDS, "columns": selected, "lambda_g": 0., "lambda_du": 0.,
              "lambda_ini": 1e5, "solver": "CLARABEL", "timing_workers": 1,
              "caveat": sweep["manifest"]["selection_caveat"]}
    four.save_json(root / "best_rank_table.json", result)
    from atlas_deepc.notebook_tools.build_projection_chordal_rank_notebook import write_table
    write_table(root, result)
    return result


def run_all(root=RESULT_ROOT, ranks=RANKS, workers=4):
    root = Path(root)
    prepare(root, ranks, SEEDS, workers)
    four.save_json(root / "latest_session.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
                   "workers": workers, "numerical_threads_per_worker": 1})
    # Reuse the exact current rank-60 experiment, including both histories and metrics.
    if 60 in ranks:
        for p, mode in four.CASES:
            run_case(root, 60, p, mode, workers)
        report(root)
    order = sorted((r for r in ranks if r != 60), key=lambda r: (abs(r-60), r))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for rank in order:
            if shutil.disk_usage(root).free < 3*1024**3:
                raise RuntimeError("Less than 3 GiB free; stopping safely with resumable saved results")
            futures = [pool.submit(run_case, root, rank, p, mode, workers) for p, mode in four.CASES]
            for future in as_completed(futures):
                print(json.dumps(future.result()), flush=True)
                report(root)
    result = serial_winners(root)
    from atlas_deepc.notebook_tools.build_projection_chordal_rank_notebook import build
    build(root)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.root, args.ranks, SEEDS, args.workers)
    elif args.report_only:
        report(args.root)
    else:
        run_all(args.root, args.ranks, args.workers)
