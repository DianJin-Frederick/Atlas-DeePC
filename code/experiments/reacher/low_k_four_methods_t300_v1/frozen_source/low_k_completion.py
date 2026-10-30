"""Compact, resumable low-chart-count development study at 300 steps."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np

from atlas_deepc.benchmarks import reacher_heldout as runner
from atlas_deepc.ablations import k64_rank
from atlas_deepc.common.metrics import strict_success, true_error_metrics

ROOT = runner.ROOT
RESULT_ROOT = ROOT / "experiments/reacher/low_k_four_methods_t300_v1"
COUNTS = (1, 2, 4, 8, 16, 24, 32)
CASES = ((8, "terminal"), (8, "graph"), (10, "terminal"))
ALL_CASES = (*CASES, (10, "graph"))
SEEDS = tuple(range(211, 311))
COARSE_RANKS = (48, 54, 60, 66, 72)
LIMITS = (28, 80)
CONTROL_STEPS = 300
SELECTION_RULE = k64_rank.previous.SELECTION_RULE
save_json = runner.save_json


def config_for(k, rank):
    return replace(runner.experiment.baseline.default_config(), clusters=k,
                   svd_rank=rank, steps=CONTROL_STEPS)


def code_hashes():
    return {**k64_rank.code_hashes(), Path(__file__).name: runner.file_digest(__file__),
            Path(runner.__file__).name: runner.file_digest(runner.__file__)}


def atlas_path(root, k, p):
    return Path(root) / f"atlases/k{k:03d}/output{p}/behavioral_atlas.npz"


def case_path(root, k, p, mode):
    return Path(root) / f"cases/k{k:03d}/output{p}/{mode}"


def best_record(records):
    return min(records, key=lambda r: (-r["successes"], r["median_rollout_rmse_mm"], r["rank"]))


def boundary_extension(records):
    ranks = {r["rank"] for r in records}
    winner = best_record(records)["rank"]
    if winner == min(ranks) and winner > LIMITS[0]:
        return [max(LIMITS[0], winner-6)]
    if winner == max(ranks) and winner < LIMITS[1]:
        return [min(LIMITS[1], winner+6)]
    return []


def prepare(root=RESULT_ROOT, cases=CASES, counts=COUNTS, seeds=SEEDS, workers=4, pilot=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cases = tuple(tuple(c) for c in cases)
    if not cases or len(set(cases)) != len(cases) or not set(cases) <= set(ALL_CASES):
        raise ValueError("Unknown or repeated case")
    if not counts or not set(counts) <= set(COUNTS):
        raise ValueError("Unsupported chart count")
    if tuple(seeds) != ((211,) if pilot else SEEDS):
        raise ValueError("Use the frozen development panel, or the separate one-seed pilot")
    manifest = dict(version="low_k_t300_v1", cases=cases, chart_counts=counts, seeds=seeds,
        controlled_steps=CONTROL_STEPS, warmup_steps=12, workers=workers,
        config_at_k48_rank60=asdict(config_for(48, 60)),
        coarse_ranks=[60] if pilot else COARSE_RANKS, rank_limits=LIMITS,
        pilot=pilot, source_sha256=code_hashes(), data_sha256=runner.file_digest(runner.DATA_PATH),
        selection_rule=SELECTION_RULE,
        rank_search="Coarse ranks, extend winning boundary by six, then every rank within two of the coarse winner",
        storage="Per-seed metrics only; temporary validated rollout traces removed after checkpoint",
        timing="Parallel search timings are not reported as serial controller latency",
        caveat="Exploratory development seeds; best among tested ranks, not a held-out optimum")
    manifest = json.loads(json.dumps(manifest))
    path = root / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Study protocol/source changed; choose a fresh result directory")
    else:
        save_json(path, manifest)
        for name in manifest["source_sha256"]:
            if name == Path(__file__).name:
                source = Path(__file__)
            elif name == Path(runner.__file__).name:
                source = Path(runner.__file__)
            elif name.startswith("common/"):
                source = ROOT / "src/atlas_deepc" / name
            elif (Path(__file__).parent / name).exists():
                source = Path(__file__).parent / name
            else:
                source = k64_rank.source_path(name)
            destination = root / "frozen_source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    inventory = {}
    for k in counts:
        for p in sorted({p for p, _ in cases}):
            path, _ = runner.experiment.prepare_atlas(config_for(k, 60),
                atlas_path(root, k, p).parent, runner.DATA_PATH, output_dimension=p)
            atlas = runner.experiment.routed.load_atlas(path)
            sizes = np.bincount(atlas["labels"].astype(int), minlength=k)
            assert len(sizes) == k and sizes.sum() == 37800 and sizes.min() >= LIMITS[1]
            assert np.all(atlas["feature_mean"] == 0) and np.all(atlas["feature_scale"] == 1)
            inventory[f"{k}_{p}"] = dict(sha256=runner.file_digest(path), chart_sizes=sizes.tolist())
    save_json(root / "atlas_inventory.json", inventory)
    return manifest


def seed_metrics(trace, seed, p, mode, k):
    distance = np.asarray(trace["distance"])
    assert distance.shape == (CONTROL_STEPS,) and np.isfinite(distance).all()
    assert trace["y"].shape == (CONTROL_STEPS, p)
    np.testing.assert_allclose(distance, np.linalg.norm(
        trace["y"][:, 6:8] - trace["final_goal"].reshape(-1)[6:8], axis=1))
    residuals, windows = trace["selection_residuals"], trace["selection_windows"]
    assert residuals.shape == (CONTROL_STEPS, k)
    assert windows.shape == (CONTROL_STEPS, (2+p)*12)
    assert np.isfinite(residuals).all() and np.isfinite(windows).all()
    tolerance = 100*np.finfo(float).eps*np.maximum(1., np.linalg.norm(windows, axis=1))
    selected = np.argmax(residuals <= residuals.min(axis=1, keepdims=True)+tolerance[:, None], axis=1)
    np.testing.assert_array_equal(trace["chart"], selected)
    if mode == "terminal" or k == 1:
        np.testing.assert_allclose(trace["ref"], np.tile(trace["final_goal"], (CONTROL_STEPS, 1)))
        assert np.unique(trace["goal_index"]).size == 1
    metrics = true_error_metrics(distance)
    return dict(seed=seed, **metrics, success=strict_success(metrics),
                rollout_rmse_mm=float(1000*np.sqrt(np.mean(distance**2))),
                unsolved_steps=sum(str(s) not in ("optimal", "optimal_inaccurate") for s in trace["status"]))


def summarize(rows, k, rank, p, mode):
    errors = np.array([1000*r["mean_last_100"] for r in rows])
    return dict(clusters=k, rank=rank, output_dimension=p, reference_mode=mode,
        controlled_steps=CONTROL_STEPS, seeds=len(rows), successes=sum(r["success"] for r in rows),
        success_rate=100*sum(r["success"] for r in rows)/len(rows),
        median_error_mm=float(np.median(errors)),
        **{f"p{q}_error_mm": float(np.percentile(errors, q)) for q in (90, 95, 99)},
        median_rollout_rmse_mm=float(np.median([r["rollout_rmse_mm"] for r in rows])),
        failed_seeds=[r["seed"] for r in rows if not r["success"]],
        unsolved_steps=sum(r["unsolved_steps"] for r in rows))


def run_panel(root, k, rank, p, mode):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    assert code_hashes() == manifest["source_sha256"], "Source changed during experiment"
    assert k in manifest["chart_counts"] and [p, mode] in manifest["cases"]
    assert LIMITS[0] <= rank <= LIMITS[1]
    seeds = manifest["seeds"]
    directory = case_path(root, k, p, mode)
    path = directory / f"rank{rank:03d}.json"
    signature = dict(manifest_sha256=runner.file_digest(root / "manifest.json"),
                     clusters=k, rank=rank, output_dimension=p, reference_mode=mode)
    payload = json.loads(path.read_text()) if path.exists() else dict(signature=signature, results=[], complete=False)
    assert payload["signature"] == signature
    rows = payload["results"]
    assert [r["seed"] for r in rows] == seeds[:len(rows)]
    temporary = directory / f"rank{rank:03d}_temporary.npz"
    if payload["complete"]:
        assert len(rows) == len(seeds)
        if temporary.exists():
            temporary.unlink()
        return summarize(rows, k, rank, p, mode)
    assert shutil.disk_usage(root).free > 2*2**30, "Less than 2 GiB free; resumable stop"
    inventory = json.loads((root / "atlas_inventory.json").read_text())
    assert runner.file_digest(atlas_path(root, k, p)) == inventory[f"{k}_{p}"]["sha256"]
    directory.mkdir(parents=True, exist_ok=True)
    # Reuse the active baseline controller verbatim, changing only K, rank and horizon.
    with patch.object(runner, "config_for", return_value=config_for(k, rank)), patch.object(
            runner, "atlas_path", side_effect=lambda root_, p_, mode_=None: atlas_path(root_, k, p_)):
        with runner.controller_context(root, p, mode) as (controller, selector, _, _):
            for seed in seeds[len(rows):]:
                selector.reset()
                controller.run(seed, temporary, reference_mode=mode,
                               chart_selector=selector, localization_history_length=12)
                with np.load(temporary, allow_pickle=True) as original:
                    trace = {key: original[key] for key in original.files if key != "summary"}
                trace.update(selection_windows=np.asarray(selector.windows),
                             selection_residuals=np.asarray(selector.residual_history))
                rows.append(seed_metrics(trace, seed, p, mode, k))
                payload.update(complete=len(rows) == len(seeds), updated_utc=datetime.now(timezone.utc).isoformat())
                save_json(path, payload)
                temporary.unlink()
                if len(rows) % 10 == 0 or payload["complete"]:
                    print(f"K={k} p={p} {mode} rank={rank}: {len(rows)}/{len(seeds)} seeds", flush=True)
    return summarize(rows, k, rank, p, mode)


def run_case(root, k, p, mode):
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    records = [run_panel(root, k, r, p, mode) for r in manifest["coarse_ranks"]]
    if not manifest["pilot"]:
        while boundary_extension(records):
            records += [run_panel(root, k, r, p, mode) for r in boundary_extension(records)]
        center = best_record(records)["rank"]
        for rank in range(max(LIMITS[0], center-2), min(LIMITS[1], center+2)+1):
            if rank not in {r["rank"] for r in records}:
                records.append(run_panel(root, k, rank, p, mode))
    winner = best_record(records)
    save_json(case_path(root, k, p, mode) / "selection.json",
              dict(complete=True, selection_rule=SELECTION_RULE, tested_ranks=sorted(r["rank"] for r in records),
                   winner=winner, records=records))
    return winner


def report(root=RESULT_ROOT):
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return dict(complete=False, records=[], selected=[], completed_rollouts=0, progress=[])
    manifest = json.loads(manifest_path.read_text())
    records, selected, progress = [], [], []
    completed = 0
    for k in manifest["chart_counts"]:
        for p, mode in manifest["cases"]:
            directory = case_path(root, k, p, mode)
            panel_rows = []
            total = 0
            for path in sorted(directory.glob("rank*.json")):
                panel = json.loads(path.read_text())
                total += len(panel["results"])
                if panel["complete"]:
                    panel_rows.append(summarize(panel["results"], k, panel["signature"]["rank"], p, mode))
            completed += total
            records.extend(panel_rows)
            done = (directory / "selection.json").exists()
            if done:
                selected.append(json.loads((directory / "selection.json").read_text())["winner"])
            progress.append(dict(clusters=k, output_dimension=p, reference_mode=mode,
                completed_rollouts=total, complete=done, completed_ranks=len(panel_rows)))
    return dict(complete=all(r["complete"] for r in progress), completed_rollouts=completed,
                records=records, selected=selected, progress=progress, controlled_steps=CONTROL_STEPS)


def run_all(root=RESULT_ROOT, cases=CASES, workers=4, pilot=False):
    counts, seeds = ((1,), (211,)) if pilot else (COUNTS, SEEDS)
    prepare(root, cases, counts, seeds, workers, pilot)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = [pool.submit(run_case, root, k, p, mode) for k in counts for p, mode in cases]
        for future in as_completed(jobs):
            print("CASE COMPLETE", json.dumps(future.result()), flush=True)
            save_json(Path(root) / "summary.json", report(root))
    result = report(root)
    assert result["complete"]
    save_json(Path(root) / "summary.json", result)
    save_json(Path(root) / "best_rank_table.json", dict(columns=result["selected"],
        seeds=list(seeds), controlled_steps=CONTROL_STEPS, selection_rule=SELECTION_RULE))
    return result
