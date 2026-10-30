"""Full 28--70 rank sweep of four K=64 Reacher configurations."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.ablations import projection_chordal_rank as previous
from atlas_deepc.common.paths import source_hashes, source_path

four = previous.four
ROOT = four.ROOT
RESULT_ROOT = ROOT / "experiments/reacher/four_baselines_k64_rank_projection_chordal_v1"
NOTEBOOK = ROOT / "notebooks/reacher/ablations/ablation_linear_subspace_rank_k64.ipynb"
RANKS = list(range(28, 71))
SEEDS = list(range(211, 311))
K = 64


def config_for(rank=60):
    return replace(four.experiment.default_config(), clusters=K, svd_rank=rank,
                   lam_g=0., lam_du=0.)


def code_hashes():
    values = source_hashes(("unstandardized_atlas_deepc.py", "baseline_atlas_deepc.py",
        "routed_deepc_experiment.py", "expert_pipeline_search.py",
        "candidate_goal_routing_experiment.py", "chart_selection.py"))
    for path in (Path(__file__), Path(previous.__file__), Path(four.__file__),
                 Path(__file__).with_name("chordal_graph_cost.py")):
        values[path.name] = four.file_digest(path)
    return values


def validate_config(config, p, mode, rank):
    expected = previous.control_signature(p, mode)["config"]
    expected.update(clusters=K, svd_rank=rank)
    assert config == expected, "Only chart count and rank may differ from K=48"


def atlas_path(root, p):
    return Path(root) / f"atlases/output{p}/behavioral_atlas.npz"


def prepare(root=RESULT_ROOT, workers=4):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in asdict(config_for()).items() if k != "seed"}
    for p, mode in four.CASES:
        validate_config(config, p, mode, 60)
    data_hash = four.file_digest(four.DATA_PATH)
    assert data_hash == previous.control_signature(10, "graph")["data_sha256"]
    manifest = {
        "charts": K, "ranks": RANKS, "seeds": SEEDS, "config_rank60": config,
        "workers": workers, "numerical_threads_per_worker": 1,
        "source_sha256": code_hashes(), "data_sha256": data_hash,
        "objective": "tracking + input effort + initial-output slack",
        "selection": "full_window_projection", "graph_cost": "chordal",
        "selection_rule": previous.SELECTION_RULE,
        "prior_rollouts_reused": False,
        "reuse_note": "Reuse the frozen p=10 K=64 atlas, not old rollouts: controller source changed.",
        "timing": "Parallel sweep; all 100 seeds of each selected rank rerun serially with one numerical thread.",
        "caveat": "Ranks selected and reported on the same exploratory seeds; not held-out evaluation.",
    }
    path = root / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "Protocol/source changed; use a fresh root"
    else:
        four.save_json(path, manifest)
        frozen = root / "frozen_source"
        frozen.mkdir(exist_ok=True)
        for name in manifest["source_sha256"]:
            if name.startswith("common/"):
                source = ROOT / "src/atlas_deepc" / name
            elif name == Path(__file__).name:
                source = Path(__file__)
            elif (Path(__file__).parent / name).is_file():
                source = Path(__file__).parent / name
            else:
                source = source_path(name)
            destination = frozen / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    original = ROOT / "experiments/reacher/chart_count_rank_projection_chordal_v1/atlases/k064"
    original_meta = json.loads((original / "source_graph/manifest.json").read_text())
    assert original_meta["data_sha256"] == data_hash
    target = atlas_path(root, 10)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(original / "behavioral_atlas.npz", target)
    assert four.file_digest(target) == original_meta["atlas_sha256"]
    four.experiment.prepare_atlas(config_for(), atlas_path(root, 8).parent,
                                 four.DATA_PATH, output_dimension=8)
    info = {}
    for p in (8, 10):
        atlas = four.experiment.routed.load_atlas(atlas_path(root, p))
        counts = np.bincount(atlas["labels"].astype(int), minlength=K)
        assert len(counts) == K and counts.sum() == 37800 and counts.min() >= max(RANKS)
        assert atlas["medoid_states"].shape == (K, p)
        assert np.all(atlas["feature_mean"] == 0) and np.all(atlas["feature_scale"] == 1)
        info[str(p)] = {"sha256": four.file_digest(atlas_path(root, p)),
                        "chart_sizes": counts.tolist(), "min_columns": int(counts.min())}
    four.save_json(root / "atlas_inventory.json", info)
    return manifest


def panel_directory(root, rank, p, mode, serial=False):
    return Path(root) / ("selected_serial" if serial else "sweep") / f"rank{rank:03d}/output{p}/{mode}"


def metric_path(root, rank, p, mode, serial=False):
    return Path(root) / ("serial_metrics" if serial else "panel_metrics") / f"rank{rank:03d}_p{p}_{mode}.json"


def run_case(root, rank, p, mode, workers=4, serial=False, seeds=None):
    root = Path(root)
    seeds = SEEDS if seeds is None else list(seeds)
    manifest = json.loads((root / "manifest.json").read_text())
    assert code_hashes() == manifest["source_sha256"], "Source changed during sweep"
    metric = metric_path(root, rank, p, mode, serial)
    if metric.exists():
        record = json.loads(metric.read_text())
        assert four.file_digest(Path(record["source_directory"]) / "summary.json") == record["summary_sha256"]
        return record
    assert shutil.disk_usage(root).free > 3 * 2**30, "Less than 3 GiB free; resumable stop"
    atlas = atlas_path(root, p)
    inventory = json.loads((root / "atlas_inventory.json").read_text())
    assert four.file_digest(atlas) == inventory[str(p)]["sha256"]
    # The old panel is only a reference for fixed terminal poses and seed histories.
    source = previous.control_signature(p, mode)
    source["atlas_sha256"] = inventory[str(p)]["sha256"]
    source_dir = previous.control.RESULT_ROOT / f"output{p}/{mode}"
    directory = panel_directory(root, rank, p, mode, serial)
    with patch.object(four, "verify_source", return_value=(source, atlas)), patch.object(
            four, "source_panel", return_value=source_dir), patch.object(
            four.experiment, "default_config", return_value=config_for(rank)):
        panel = four.run_case(p, mode, seeds, directory.parents[1], "CLARABEL", rank=rank,
            record_coefficients=False, timing_workers=1 if serial else workers,
            graph_cost="chordal", chart_selection_method="full_window_projection", warmup_steps=12)
    validate_config(panel["signature"]["config"], p, mode, rank)
    assert panel["signature"]["atlas_sha256"] == inventory[str(p)]["sha256"]
    assert panel["signature"]["chart_selection_method"] == "full_window_projection"
    assert panel["signature"]["warmup_steps"] == 12
    assert panel["signature"]["graph_cost_method"] == ("chordal" if mode == "graph" else "not_used")
    if seeds != SEEDS:
        return {"rank": rank, "p": p, "mode": mode, "pilot_seeds": seeds,
                "successes": sum(four.strict_success(r) for r in panel["results"])}
    with threadpool_limits(limits=1):
        record = previous.summarize_panel(directory, rank, p, mode, clusters=K)
    record.update(clusters=K, rank60_reused=False, serial=serial)
    metric.parent.mkdir(exist_ok=True)
    four.save_json(metric, record)
    return record


def report(root=RESULT_ROOT):
    root = Path(root)
    records = [json.loads(path.read_text()) for path in sorted((root / "panel_metrics").glob("*.json"))]
    completed = {(r["rank"], r["output_dimension"], r["reference_mode"]) for r in records}
    missing = [(rank, p, mode) for rank in RANKS for p, mode in four.CASES
               if (rank, p, mode) not in completed]
    best = [previous.best_record([r for r in records if (r["output_dimension"], r["reference_mode"]) == case])
            for case in four.CASES if any((r["output_dimension"], r["reference_mode"]) == case for r in records)]
    partial_seeds = sum(len(json.loads(path.read_text())["seeds_completed"])
                        for path in (root / "sweep").glob("rank*/output*/*/progress.json")
                        if not (path.parent / "summary.json").exists())
    payload = {"complete": not missing, "records": records, "best_completed_ranks": best,
               "missing_panels": missing, "completed_rollouts": 100*len(records),
               "in_progress_rollouts": partial_seeds, "requested_rollouts": 17200,
               "updated_utc": datetime.now(timezone.utc).isoformat()}
    four.save_json(root / "summary.json", payload)
    four.save_json(root / "rank_results.json", records)
    return payload


def run_all(root=RESULT_ROOT, workers=4):
    root = Path(root)
    prepare(root, workers)
    from atlas_deepc.notebook_tools.build_k64_rank_notebook import build
    build(root)
    order = sorted(RANKS, key=lambda r: (abs(r-62), r))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for rank in order:
            futures = [pool.submit(run_case, root, rank, p, mode, workers) for p, mode in four.CASES]
            for future in as_completed(futures):
                record = future.result()
                print("PANEL COMPLETE", json.dumps({k: record[k] for k in
                      ("rank", "output_dimension", "reference_mode", "successes")}), flush=True)
                report(root)
                build(root)
    sweep = report(root)
    assert sweep["complete"]
    serial = []
    for winner in sweep["best_completed_ranks"]:
        r, p, mode = (winner[k] for k in ("rank", "output_dimension", "reference_mode"))
        record = run_case(root, r, p, mode, workers, serial=True)
        assert record["failed_seeds"] == winner["failed_seeds"]
        for key in ("median_error_mm", "median_rollout_rmse_mm", "p99_error_mm"):
            np.testing.assert_allclose(record[key], winner[key], rtol=1e-6, atol=1e-5)
        serial.append(record)
    four.save_json(root / "best_rank_table.json", {"columns": serial, "ranks_tested": RANKS,
                   "selection_rule": previous.SELECTION_RULE, "charts": K, "seeds": SEEDS})
    build(root)
    print("COMPLETE: 17,200 sweep rollouts and 400 serial timing rollouts", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.root, args.workers)
    elif args.report_only:
        from atlas_deepc.notebook_tools.build_k64_rank_notebook import build
        build(args.root)
    elif args.pilot:
        prepare(args.root, args.workers)
        for p, mode in four.CASES:
            print(run_case(args.root, 62, p, mode, args.workers, seeds=[211]), flush=True)
    else:
        run_all(args.root, args.workers)
