"""Frozen development-selected Reacher methods on a separate 500-seed panel."""

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import time
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.ablations import chordal_graph_cost as graph
from atlas_deepc.baselines import unstandardized_atlas_deepc as experiment
from atlas_deepc.benchmarks.baseline_1000_benchmark import summarize_timing
from atlas_deepc.common.files import file_digest
from atlas_deepc.common.metrics import strict_success, true_error_metrics
from atlas_deepc.common.paths import source_path
from atlas_deepc.reporting.four_case_tracking_table import format_table, full_rollout_rmse

ROOT = experiment.baseline.ROOT
RESULT_ROOT = ROOT / "experiments/reacher/four_baselines_heldout500_k48_v1"
DEVELOPMENT_ROOT = ROOT / "experiments/reacher/projection_chordal_rank_lamg0_lamdu0_clarabel_v1"
DATA_PATH = ROOT / "data/reacher/select_iid_offline_hankel.npy"
SEEDS = tuple(range(10001, 10501))
DEVELOPMENT_SEEDS = tuple(range(211, 311))
CASES = ((8, "terminal", 53), (8, "graph", 53), (10, "terminal", 57), (10, "graph", 60))
VERSION = "four_baselines_heldout500_k48_v1"
CHART_COUNTS = {(p, mode): 48 for p, mode, _ in CASES}
DEVELOPMENT_ROOTS = {(p, mode): DEVELOPMENT_ROOT for p, mode, _ in CASES}
CONTROL_STEPS = 320
PER_CASE_ATLASES = False
EVALUATION_NOTE = "Frozen from development; no test-set tuning"


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending.json")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def validate_seeds(seeds):
    if tuple(seeds) != SEEDS or set(seeds) & set(DEVELOPMENT_SEEDS):
        raise ValueError("The frozen test panel is seeds 10001--10500, disjoint from development.")


def development_cases():
    columns = []
    for p, mode, rank in CASES:
        selected = json.loads((DEVELOPMENT_ROOTS[p, mode] / "best_rank_table.json").read_text())
        assert selected["seeds"] == list(DEVELOPMENT_SEEDS)
        column = next(c for c in selected["columns"]
                      if (c["output_dimension"], c["reference_mode"]) == (p, mode))
        assert column["rank"] == rank
        columns.append(column)
    return columns


def config_for(p, mode):
    rank = next(rank for dimension, reference, rank in CASES if (dimension, reference) == (p, mode))
    config = replace(experiment.baseline.default_config(), clusters=CHART_COUNTS[p, mode],
                     svd_rank=rank, steps=CONTROL_STEPS)
    selected = next(c for c in development_cases()
                    if (c["output_dimension"], c["reference_mode"]) == (p, mode))
    signature = json.loads((Path(selected["source_directory"]) / "manifest.json").read_text())
    expected = dict(signature["config"], steps=CONTROL_STEPS)
    assert {k: v for k, v in asdict(config).items() if k != "seed"} == expected
    assert config.lam_g == config.lam_du == 0 and config.lam_ini == 1e5
    return config


def atlas_path(root, p, mode=None):
    if PER_CASE_ATLASES:
        if (p, mode) not in CHART_COUNTS:
            raise ValueError("Mixed-K evaluation requires the reference mode for its atlas")
        return Path(root) / f"output{p}/{mode}/behavioral_atlas.npz"
    return Path(root) / f"output{p}/behavioral_atlas.npz"


def case_signature(root, p, mode):
    signature = experiment.execution_signature(config_for(p, mode), DATA_PATH,
        atlas_path(root, p, mode), p, mode, chart_selection_method="full_window_projection", warmup_steps=12)
    signature.update(evaluation_version=VERSION, evaluation_seeds=list(SEEDS),
        graph_cost_method="chordal" if mode == "graph" else "not_used",
        graph_code_sha256=file_digest(graph.__file__), evaluation_code_sha256=file_digest(__file__),
        solver_options={"max_iter": 10000}, timing_workers=1,
        package_versions={name: importlib.metadata.version(name) for name in
                          ("numpy", "scipy", "cvxpy", "clarabel", "mujoco", "gymnasium")})
    return signature


def prepare(root=RESULT_ROOT):
    """Freeze protocol and copy only the small atlases, never the offline dataset."""
    root = Path(root)
    validate_seeds(SEEDS)
    root.mkdir(parents=True, exist_ok=True)
    for c in development_cases():
        p, mode = c["output_dimension"], c["reference_mode"]
        config_for(p, mode)
        directory = Path(c["source_directory"])
        source = directory.parent / "behavioral_atlas.npz"
        original = json.loads((directory / "manifest.json").read_text())
        assert original["data_sha256"] == file_digest(DATA_PATH)
        assert original["atlas_sha256"] == file_digest(source)
        destination = atlas_path(root, p, mode)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temporary = destination.with_suffix(".pending.npz")
            shutil.copy2(source, temporary)
            temporary.replace(destination)
        assert file_digest(destination) == original["atlas_sha256"]
    signatures = {f"{p}_{mode}": case_signature(root, p, mode) for p, mode, _ in CASES}
    protocol = {"version": VERSION, "seeds": list(SEEDS), "development_seeds": list(DEVELOPMENT_SEEDS),
        "cases": [list(case) for case in CASES], "control_steps": CONTROL_STEPS,
        "charts": {f"{p}_{mode}": CHART_COUNTS[p, mode] for p, mode, _ in CASES},
        "development_selection_sha256": {f"{p}_{mode}": file_digest(
            DEVELOPMENT_ROOTS[p, mode] / "best_rank_table.json") for p, mode, _ in CASES},
        "case_signatures": signatures, "rank_selection": EVALUATION_NOTE,
        "reset_distribution": "Reacher-v5 default reset, followed by 12 zero-input steps",
        "timing": "Serial, one numerical thread; discard first five controlled steps per rollout",
        "initial_pose_caveat": "Joint angles start near zero, not arbitrary arm configurations"}
    manifest = root / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != protocol:
            raise ValueError("Frozen evaluation protocol/source changed; do not mix cached results.")
    else:
        if any(root.glob("output*/*/seed*_control_results.npz")):
            raise ValueError("Unversioned held-out results")
        frozen = root / "frozen_source"
        frozen.mkdir(exist_ok=True)
        for name in signatures["8_terminal"]["source_sha256"]:
            source = (ROOT / "src/atlas_deepc" / name if name.startswith("common/") else source_path(name))
            destination = frozen / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        for source in (Path(__file__), Path(graph.__file__)):
            shutil.copy2(source, frozen / source.name)
        save_json(manifest, protocol)
    for p, mode, _ in CASES:
        path = root / f"output{p}/{mode}/manifest.json"
        signature = signatures[f"{p}_{mode}"]
        if path.exists() and json.loads(path.read_text()) != signature:
            raise ValueError(f"Incompatible panel: {path}")
        save_json(path, signature)
    return protocol


class LoggedSelector(experiment.FullWindowProjectionSelector):
    def reset(self):
        super().reset()
        self.windows = []

    def select(self, window):
        self.windows.append(np.asarray(window).copy())
        return super().select(window)


@contextmanager
def controller_context(root, p, mode):
    config = config_for(p, mode)
    with experiment.output_context(p), threadpool_limits(limits=1):
        H = experiment.load_hankel(DATA_PATH, p)
        atlas = experiment.routed.load_atlas(atlas_path(root, p, mode))
        assert atlas["medoid_states"].shape == (config.clusters, p)
        assert np.all(atlas["feature_mean"] == 0) and np.all(atlas["feature_scale"] == 1)
        cache = experiment.routed.precompute_chart_matrix_cache(H, atlas, config, geometry="linear")
        bases = np.stack([cache[(k,)][0]["basis"] for k in range(config.clusters)])
        assert bases.shape == (config.clusters, (2+p)*12, config.svd_rank)
        if mode == "graph":
            distances = graph.chordal_distances(bases)
            atlas["routing_edge_costs"] = graph.graph_costs(
                distances, atlas["transition_counts"], config.edge_count_threshold)
        for matrix, _ in cache.values():
            matrix.pop("g_min", None)
            matrix.pop("g_max", None)
            assert not np.any(matrix["offset"]) and matrix.get("g_sum") is None
        with patch.object(experiment.routed, "precompute_chart_matrix_cache", return_value=cache):
            controller = experiment.baseline.BaselineAtlasDeePC(
                H, atlas, config, terminal_reference_policy=experiment.terminal_policy(mode))
        selector = LoggedSelector(cache, 2, p, 12)
        yield controller, selector, atlas, bases


def validate_trace(path, p, mode, signature_hash):
    config = config_for(p, mode)
    with np.load(path, allow_pickle=True) as trace:
        row = json.loads(trace["summary"].item())
        assert row["evaluation_signature_sha256"] == signature_hash
        assert row["seed"] in SEEDS and path.name == f"seed{row['seed']}_control_results.npz"
        assert row["svd_rank"] == config.svd_rank
        assert row["lam_g"] == row["lam_du"] == 0 and row["lam_ini"] == 1e5
        assert trace["y"].shape == (config.steps, p)
        distance = trace["distance"].copy()
        assert distance.shape == (config.steps,) and np.all(np.isfinite(distance))
        np.testing.assert_allclose(distance, np.linalg.norm(trace["y"][:, 6:8] -
                                   trace["final_goal"].reshape(-1)[6:8], axis=1))
        for key, value in true_error_metrics(distance).items():
            if key in row:
                assert np.isclose(row[key], value), key
        assert trace["selection_windows"].shape == (config.steps, (2+p)*12)
        residuals = trace["selection_residuals"]
        assert residuals.shape == (config.steps, config.clusters) and np.all(np.isfinite(residuals))
        tolerance = 100*np.finfo(float).eps*np.maximum(1., np.linalg.norm(trace["selection_windows"], axis=1))
        ties = residuals <= residuals.min(axis=1, keepdims=True) + tolerance[:, None]
        np.testing.assert_array_equal(trace["chart"], np.argmax(ties, axis=1))
        if mode == "terminal":
            np.testing.assert_array_equal(trace["ref"], np.tile(trace["final_goal"], (config.steps, 1)))
            assert np.all(trace["goal_index"] == trace["goal_index"][0])
        statuses = Counter(str(s) for s in trace["status"])
        assert row["unsolved_steps"] == sum(v for k, v in statuses.items()
                                            if k not in ("optimal", "optimal_inaccurate"))
    return {**row, "result_path": str(path.resolve())}, distance, statuses


def run_case(root, p, mode):
    root = Path(root)
    directory = root / f"output{p}/{mode}"
    signature_hash = file_digest(directory / "manifest.json")
    missing = [s for s in SEEDS if not (directory / f"seed{s}_control_results.npz").exists()]
    if not missing:
        return summarize_case(root, p, mode)
    with controller_context(root, p, mode) as (controller, selector, atlas, bases):
        np.savez_compressed(directory / "selection_geometry.npz", bases=selector.bases,
                            chart_ids=selector.chart_ids, ranks=selector.ranks)
        if mode == "graph":
            np.savez_compressed(directory / "graph_geometry.npz", bases=bases,
                labels=atlas["labels"], transition_counts=atlas["transition_counts"],
                chordal_cost=atlas["routing_edge_costs"])
        for seed in missing:
            if shutil.disk_usage(root).free < 2*2**30:
                raise RuntimeError("Less than 2 GiB free; stopped with completed seeds saved.")
            selector.reset()
            path = directory / f"seed{seed}_control_results.npz"
            pending = path.with_suffix(".pending.npz")
            started = time.perf_counter()
            row = controller.run(seed, pending, reference_mode=mode,
                                 chart_selector=selector, localization_history_length=12)
            row.update(evaluation_version=VERSION, evaluation_signature_sha256=signature_hash,
                solver="CLARABEL", output_dimension=p, chart_selection_method="full_window_projection",
                graph_cost_method="chordal" if mode == "graph" else "not_used",
                elapsed_rollout_wall_s=time.perf_counter()-started)
            with np.load(pending, allow_pickle=True) as trace:
                arrays = {k: trace[k] for k in trace.files if k != "summary"}
            np.savez_compressed(pending, **arrays, selection_windows=np.asarray(selector.windows),
                selection_residuals=np.asarray(selector.residual_history),
                selection_tie_count=np.asarray(selector.tie_history), summary=np.array(json.dumps(row)))
            pending.replace(path)
            validate_trace(path, p, mode, signature_hash)
            completed = sum((directory / f"seed{s}_control_results.npz").exists() for s in SEEDS)
            save_json(directory / "progress.json", {"completed": completed, "requested": 500,
                       "last_seed": seed, "updated_utc": datetime.now(timezone.utc).isoformat()})
            print(f"p={p}, {mode}: {completed}/500, seed={seed}, success={strict_success(row)}, "
                  f"final100={1000*row['mean_last_100']:.3f} mm", flush=True)
    return summarize_case(root, p, mode)


def summarize_case(root, p, mode):
    directory = Path(root) / f"output{p}/{mode}"
    signature_hash = file_digest(directory / "manifest.json")
    rows, distances, statuses = [], [], Counter()
    for seed in SEEDS:
        path = directory / f"seed{seed}_control_results.npz"
        row, distance, counts = validate_trace(path, p, mode, signature_hash)
        rows.append(row)
        distances.append(distance)
        statuses.update(counts)
    timing = summarize_timing([r["result_path"] for r in rows])
    save_json(directory / "summary.json", {"seeds": list(SEEDS), "results": rows, "timing": timing,
              "signature": json.loads((directory / "manifest.json").read_text())})
    errors = 1000*np.mean(np.asarray(distances)[:, -100:], axis=1)
    _, per_seed_rmse = full_rollout_rmse(distances)
    total = next(r for r in timing["modules"] if r["module"] == "Total online control")
    record = {"output_dimension": p, "reference_mode": mode, "graph_guidance": mode == "graph",
        "rank": config_for(p, mode).svd_rank, "charts": config_for(p, mode).clusters,
        "control_steps": config_for(p, mode).steps, "seeds": len(SEEDS),
        "successes": sum(strict_success(r) for r in rows),
        "success_rate": 100*sum(strict_success(r) for r in rows)/len(SEEDS),
        "median_error_mm": float(np.median(errors)),
        **{f"p{q}_error_mm": float(np.percentile(errors, q)) for q in (90, 95, 99)},
        "median_rollout_rmse_mm": float(np.median(per_seed_rmse)),
        "failed_seeds": [r["seed"] for r in rows if not strict_success(r)],
        "per_seed_rmse_mm": {str(s): float(e) for s, e in zip(SEEDS, per_seed_rmse)},
        "solver_status_counts": dict(statuses), "unsolved_steps": sum(r["unsolved_steps"] for r in rows),
        "median_time_ms": total["median_ms"], "p95_time_ms": total["p95_ms"],
        "p99_time_ms": total["p99_ms"], "timing_modules": timing["modules"],
        "signature_sha256": signature_hash, "summary_sha256": file_digest(directory / "summary.json")}
    save_json(directory / "metrics.json", record)
    return record


def report(root=RESULT_ROOT):
    root = Path(root)
    columns, progress = [], []
    for p, mode, rank in CASES:
        directory = root / f"output{p}/{mode}"
        count = sum((directory / f"seed{s}_control_results.npz").exists() for s in SEEDS)
        progress.append({"outputs": p, "graph": mode == "graph", "rank": rank,
                         "charts": CHART_COUNTS[p, mode], "control_steps": CONTROL_STEPS,
                         "completed": count, "requested": 500})
        if (directory / "metrics.json").exists():
            record = json.loads((directory / "metrics.json").read_text())
            assert record["signature_sha256"] == file_digest(directory / "manifest.json")
            assert record["summary_sha256"] == file_digest(directory / "summary.json")
            assert count == len(SEEDS)
            columns.append(record)
    payload = {"complete": len(columns) == 4, "seeds": list(SEEDS), "columns": columns, "progress": progress}
    if payload["complete"]:
        verify_paired_resets(root)
        caption = (r"Four-method comparison on 500 Reacher seeds (10001--10500). "
            r"Ranks were selected using development seeds 211--310. "
            r"All methods use unstandardized linear charts, full-window projection selection, "
            r"the tracking/input-effort/initial-slack objective, $\lambda_{\mathrm{ini}}=10^5$, and Clarabel. "
            r"Guided methods use chordal costs and iterative $N=10$ terminal selection; unguided methods "
            r"use a fixed nearest eligible terminal configuration. Error quantiles summarize final-100 "
            f"mean fingertip error across rollouts. Median rollout RMSE includes all {CONTROL_STEPS} controlled steps. "
            r"Timings pool serial single-thread online steps, excluding the first five per rollout. "
            r"Evaluation uses the default reset distribution, not arbitrary initial arm configurations. "
            + EVALUATION_NOTE + ".")
        table = format_table(columns, caption, "tab:reacher-heldout500", include_rank=True)
        # At 500 seeds, success-rate resolution is 0.2 percentage points.
        best = max(c["successes"] for c in columns)
        cells = [f"{c['success_rate']:.1f}" + r"\%" for c in columns]
        cells = [r"\textbf{"+cell+"}" if c["successes"] == best else cell for cell, c in zip(cells, columns)]
        table = "\n".join("        Success rate $\\uparrow$ & " + " & ".join(cells) + r" \\"
                          if line.strip().startswith("Success rate") else line for line in table.splitlines()) + "\n"
        table = table.replace("Total online computation time (ms)", "Online computation time (ms/step)")
        chart_row = "        Charts $K$ & " + " & ".join(
            str(CHART_COUNTS[p, mode]) for p, mode, _ in CASES) + r" \\" + "\n"
        table = "\n".join(chart_row + line if line.strip().startswith("Selected rank")
                          else line for line in table.splitlines()) + "\n"
        (root / "table_heldout500.tex").write_text(table)
    if root.exists():
        save_json(root / "evaluation_report.json", payload)
    return payload


def verify_paired_resets(root):
    """Check paired targets and measured warmup histories across all four methods."""
    root = Path(root)
    for seed in SEEDS:
        initial, target = None, None
        for p, mode, _ in CASES:
            with np.load(root / f"output{p}/{mode}/seed{seed}_control_results.npz") as trace:
                z = trace["selection_windows"][0]
                common = np.r_[z[:24], z[24:].reshape(12, p)[:, :8].ravel()]
                xy = trace["final_goal"].reshape(-1)[6:8]
                if initial is None:
                    initial, target = common.copy(), xy.copy()
                else:
                    np.testing.assert_array_equal(common, initial)
                    np.testing.assert_array_equal(xy, target)


def run_all(root=RESULT_ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This held-out evaluation is already running.") from None
        prepare(root)
        session = {"started_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                   "platform": platform.platform(), "workers": 1, "numerical_threads": 1,
                   "load_average_at_start": os.getloadavg(), "status": "running",
                   "timing_caveat": "Avoid simultaneous sweeps; shared-computer load affects wall-clock timing."}
        save_json(root / "run_status.json", session)
        try:
            for p, mode, _ in CASES:
                run_case(root, p, mode)
                report(root)
            session["status"] = "complete"
        except BaseException as exc:
            session.update(status="stopped", error=str(exc))
            raise
        finally:
            session["updated_utc"] = datetime.now(timezone.utc).isoformat()
            save_json(root / "run_status.json", session)
    return report(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        prepare()
        print("Protocol frozen. No evaluation rollouts launched.")
    else:
        run_all()
