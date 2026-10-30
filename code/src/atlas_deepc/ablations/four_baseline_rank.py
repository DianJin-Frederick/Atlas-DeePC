"""Rank-only ablation of the four unstandardized, unbounded-g baselines."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

from atlas_deepc.ablations import basic_four_cases_osqp as four

import numpy as np

ROOT = four.ROOT
RESULT_ROOT = ROOT / "experiments/reacher/four_baseline_rank_lamg0_clarabel_v1"
NOTEBOOK = ROOT / "notebooks/reacher/ablations/ablation_linear_subspace_rank.ipynb"
RANKS = tuple(range(24, 81))
SEEDS = tuple(range(211, 311))
ML_PLUS_N = 2 * 12 + 4
VERSION = "four_baseline_rank_lamg0_clarabel_v1"


def prepare(root, ranks, seeds, workers):
    root = Path(root)
    if not ranks or len(set(ranks)) != len(ranks) or any(r < 1 or r >= 120 for r in ranks):
        raise ValueError("Ranks must be unique integers in [1, 119]")
    if not seeds or len(set(seeds)) != len(seeds) or workers not in range(1, 5):
        raise ValueError("Use unique seeds and one to four workers")
    root.mkdir(parents=True, exist_ok=True)
    package = ROOT / "src/atlas_deepc"
    sources = {str(p.relative_to(package)): four.file_digest(p)
               for folder in ("control", "atlas", "baselines", "common", "benchmarks")
               for p in sorted((package / folder).glob("*.py"))}
    for p in (Path(__file__), Path(four.__file__)):
        sources[str(p.relative_to(package))] = four.file_digest(p)
    atlases = {}
    for dimension, mode in four.CASES:
        original, _ = four.verify_source(dimension, mode)
        atlases[f"{dimension}_{mode}"] = original["atlas_sha256"]
    manifest = {"version": VERSION, "ranks": ranks, "seeds": seeds,
                "m": 2, "L": 12, "n_dynamic": 4, "mL_plus_n": ML_PLUS_N,
                "lambda_g": 0, "solver": "CLARABEL", "workers": workers,
                "numerical_threads_per_worker": 1,
                "data_sha256": four.file_digest(four.DATA_PATH), "atlas_sha256": atlases,
                "sources": sources,
                "timing_note": ("Parallel workers: contention-affected timings, not comparable to serial table."
                                if workers > 1 else "Serial, single numerical thread; shared-machine timing."),
                "reference_policies": {mode: four.experiment.terminal_policy(mode)
                                       for mode in ("terminal", "graph")}}
    path = root / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Incompatible sweep cache; choose a fresh result root")
    else:
        four.save_json(path, manifest)
        for relative in sources:
            target = root / "frozen_source/atlas_deepc" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(package / relative, target)
        shutil.copy2(package / "__init__.py", root / "frozen_source/atlas_deepc/__init__.py")
    # Both reference policies share one atlas file per dimension. Prepare it
    # before spawning workers so no worker observes another worker's partial copy.
    for rank in ranks:
        for dimension in (8, 10):
            source = four.source_panel(dimension, "terminal").parents[1] / "behavioral_atlas.npz"
            target = root / f"rank{rank:03d}/output{dimension}/behavioral_atlas.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                temporary = target.with_suffix(".npz.tmp")
                shutil.copy2(source, temporary)
                temporary.replace(target)
            assert four.file_digest(target) == atlases[f"{dimension}_terminal"]
    return manifest


def run_case(root, rank, dimension, mode, seeds, workers):
    directory = Path(root) / f"rank{rank:03d}"
    started = time.monotonic()
    panel = four.run_case(dimension, mode, seeds, directory, "CLARABEL", rank=rank,
                          record_coefficients=False, timing_workers=workers)
    return {"rank": rank, "output_dimension": dimension, "reference_mode": mode,
            "successes": sum(four.strict_success(r) for r in panel["results"]),
            "elapsed_s": time.monotonic() - started}


def report(root=RESULT_ROOT):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    records, missing = [], []
    for rank in manifest["ranks"]:
        for dimension, mode in four.CASES:
            path = root / f"rank{rank:03d}/output{dimension}/{mode}/summary.json"
            if not path.exists():
                missing.append([rank, dimension, mode])
                continue
            panel = json.loads(path.read_text())
            assert panel["seeds"] == manifest["seeds"]
            rows = panel["results"]
            assert [r["seed"] for r in rows] == manifest["seeds"]
            assert all(r["svd_rank"] == rank and r["lam_g"] == 0 and r["solver"] == "CLARABEL"
                       for r in rows)
            errors = np.array([1000*r["mean_last_100"] for r in rows])
            modules = {r["module"]: r for r in panel["timing"]["modules"]}
            total = modules["Total online control"]
            records.append({"rank": rank, "output_dimension": dimension, "reference_mode": mode,
                            "seeds": len(rows), "successes": sum(four.strict_success(r) for r in rows),
                            "success_rate": 100*np.mean([four.strict_success(r) for r in rows]),
                            "median_error_mm": float(np.median(errors)),
                            **{f"p{q}_error_mm": float(np.percentile(errors, q)) for q in (90, 95, 99)},
                            "unsolved_steps": sum(r["unsolved_steps"] for r in rows),
                            "unsolved_rollouts": sum(r["unsolved_steps"] > 0 for r in rows),
                            "failed_seeds": [r["seed"] for r in rows if not four.strict_success(r)],
                            "median_time_ms": total["median_ms"], "p99_time_ms": total["p99_ms"],
                            "timing_modules": modules})
    payload = {"version": VERSION, "manifest": manifest, "records": records,
               "missing_panels": missing, "complete": not missing,
               "completed_rollouts": sum(r["seeds"] for r in records),
               "requested_rollouts": len(manifest["ranks"])*len(manifest["seeds"])*4}
    four.save_json(root / "summary.json", payload)
    return payload


def validate(root=RESULT_ROOT):
    """Recompute metrics from every stored trace, including unsuccessful QPs."""
    root = Path(root)
    payload = report(root)
    assert payload["complete"]
    statuses, checked = {}, 0
    targets = {}
    rank60_differences = []
    for record in payload["records"]:
        rank, dimension, mode = (record[k] for k in ("rank", "output_dimension", "reference_mode"))
        for seed in payload["manifest"]["seeds"]:
            path = root / f"rank{rank:03d}/output{dimension}/{mode}/seed{seed}_control_results.npz"
            with np.load(path, allow_pickle=True) as trace:
                row = json.loads(trace["summary"].item())
                assert trace["y"].shape == (300, dimension)
                np.testing.assert_allclose(trace["distance"],
                    np.linalg.norm(trace["y"][:, 6:8] - trace["final_goal"].reshape(-1)[6:8], axis=1))
                assert np.all(np.isfinite(trace["distance"]))
                metrics = four.true_error_metrics(trace["distance"])
                for key in ("success_30", "success_100", "mean_last_100"):
                    assert row[key] == metrics[key]
                actual_status = [str(s) for s in trace["status"]]
                assert row["unsolved_steps"] == sum(s not in ("optimal", "optimal_inaccurate") for s in actual_status)
                for status in actual_status:
                    statuses[status] = statuses.get(status, 0) + 1
                target = trace["final_goal"].reshape(-1)[6:8]
                if seed in targets:
                    np.testing.assert_array_equal(target, targets[seed])
                targets[seed] = target.copy()
                if mode == "terminal":
                    np.testing.assert_array_equal(trace["ref"], np.tile(trace["final_goal"], (300, 1)))
                if rank == 60:
                    old = ROOT / f"experiments/reacher/basic_four_cases_lamg0_clarabel_v1/output{dimension}/{mode}/{path.name}"
                    if old.exists():
                        with np.load(old, allow_pickle=True) as prior:
                            difference = float(np.max(np.abs(prior["distance"]-trace["distance"])))
                            if difference > 1e-8:
                                rank60_differences.append([dimension, mode, seed, difference])
            checked += 1
    result = {"checked_rollouts": checked, "solver_status_counts": statuses,
              "rank60_trace_differences_above_1e_8_m": rank60_differences,
              "complete": True}
    four.save_json(root / "verification.json", result)
    return result


def refresh_notebook():
    import nbformat
    from nbclient import NotebookClient
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    assert any("RUN_EXPERIMENTS = False" in c.source for c in notebook.cells if c.cell_type == "code")
    NotebookClient(notebook, timeout=600, kernel_name="gym-robotics-mujoco",
                   resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    temporary = NOTEBOOK.with_suffix(".ipynb.tmp")
    nbformat.write(notebook, temporary)
    temporary.replace(NOTEBOOK)


def run_all(ranks=RANKS, seeds=SEEDS, workers=4, root=RESULT_ROOT, update_notebook=True):
    ranks, seeds, root = list(map(int, ranks)), list(map(int, seeds)), Path(root)
    prepare(root, ranks, seeds, workers)
    four.save_json(root / f"session_{time.time_ns()}.json",
                   {"started_utc": datetime.now(timezone.utc).isoformat(), "workers": workers,
                    "load_average_at_start": os.getloadavg()})
    # Rank 60 first supplies a direct reproduction check against the four-baselines table.
    order = ([60] if 60 in ranks else []) + [r for r in ranks if r != 60]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for rank in order:
            futures = [pool.submit(run_case, root, rank, p, mode, seeds, workers) for p, mode in four.CASES]
            for future in as_completed(futures):
                print(json.dumps(future.result()), flush=True)
                report(root)
            if update_notebook:
                refresh_notebook()
    verification = validate(root)
    if update_notebook:
        refresh_notebook()
    print(json.dumps(verification, indent=2), flush=True)
    return report(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--no-notebook", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        report(args.root)
        if not args.no_notebook:
            refresh_notebook()
    else:
        run_all(args.ranks, args.seeds, args.workers, args.root, not args.no_notebook)


if __name__ == "__main__":
    main()
