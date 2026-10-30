"""Figure seeds with guidance added to the p=8, K=64, rank-53 baseline."""

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np

from atlas_deepc.benchmarks import reacher_mixed_k as baseline

runner = baseline.runner
SEEDS = (10206, 10349, 10144, 10445, 10330, 10255, 10449)
VERSION = "figure_guidance_p8_k64_r53_t300_v1"
RESULT_ROOT = baseline.ROOT / "experiments/reacher" / VERSION


def config():
    value = baseline.config_for(8, "terminal")
    assert (value.clusters, value.svd_rank, value.steps) == (64, 53, 300)
    return value


@contextmanager
def protocol_context():
    value = config()
    source = baseline.atlas_path(baseline.RESULT_ROOT, 8, "terminal")
    with patch.object(runner, "SEEDS", SEEDS), patch.object(runner, "VERSION", VERSION),\
            patch.object(runner, "config_for", return_value=value),\
            patch.object(runner, "atlas_path", return_value=source):
        yield


def prepare(root):
    root = Path(root)
    original = baseline.RESULT_ROOT / "output8/terminal/manifest.json"
    previous = json.loads(original.read_text())
    value = config()
    source = baseline.atlas_path(baseline.RESULT_ROOT, 8, "terminal")
    assert previous["config"] == {k: v for k, v in asdict(value).items() if k != "seed"}
    assert previous["atlas_sha256"] == runner.file_digest(source)
    assert previous["data_sha256"] == runner.file_digest(runner.DATA_PATH)
    for seed in SEEDS:
        if not (original.parent / f"seed{seed}_control_results.npz").exists():
            raise FileNotFoundError(f"Missing paired unguided trace for seed {seed}")
    with protocol_context():
        signature = runner.case_signature(root, 8, "graph")
    signature.update(figure_study_sha256=runner.file_digest(__file__),
                     paired_manifest_sha256=runner.file_digest(original),
                     paired_directory=str(original.parent), atlas_path=str(source),
                     caveat="Selected illustrative seeds, not a success-rate estimate. Guidance includes iterative N=10 terminal selection.")
    path = root / "output8/graph/manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != signature:
            raise ValueError("Figure study source/protocol changed; use a fresh result directory")
    else:
        if list(path.parent.glob("seed*_control_results.npz")):
            raise ValueError("Unversioned figure traces")
        runner.save_json(path, signature)
    return path


def paired_rows(root=RESULT_ROOT):
    root = Path(root)
    result = []
    for seed in SEEDS:
        paths = {"unguided": baseline.RESULT_ROOT / f"output8/terminal/seed{seed}_control_results.npz",
                 "guided": root / f"output8/graph/seed{seed}_control_results.npz"}
        row = {"seed": seed}
        initial, target = None, None
        for mode, path in paths.items():
            row[f"{mode}_path"] = str(path)
            if not path.exists():
                row[f"{mode}_success"] = None
                continue
            with np.load(path, allow_pickle=True) as trace:
                if mode == "guided":
                    np.testing.assert_array_equal(initial, trace["selection_windows"][0])
                    np.testing.assert_array_equal(target, trace["final_goal"].reshape(-1)[6:8])
                else:
                    initial = trace["selection_windows"][0].copy()
                    target = trace["final_goal"].reshape(-1)[6:8].copy()
                distance = trace["distance"]
                assert distance.shape == (300,)
                row[f"{mode}_success"] = bool(runner.strict_success(runner.true_error_metrics(distance)))
                row[f"{mode}_final100_mm"] = float(1000 * distance[-100:].mean())
        result.append(row)
    return result


def run(root=RESULT_ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = prepare(root)
        signature = runner.file_digest(manifest)
        directory = manifest.parent
        with protocol_context():
            missing = []
            for seed in SEEDS:
                path = directory / f"seed{seed}_control_results.npz"
                if path.exists():
                    runner.validate_trace(path, 8, "graph", signature)
                else:
                    missing.append(seed)
            if missing:
                with runner.controller_context(root, 8, "graph") as (controller, selector, atlas, bases):
                    np.savez_compressed(directory / "graph_geometry.npz", bases=bases,
                        labels=atlas["labels"], transition_counts=atlas["transition_counts"],
                        chordal_cost=atlas["routing_edge_costs"])
                    for seed in missing:
                        if shutil.disk_usage(root).free < 2 * 2**30:
                            raise RuntimeError("Less than 2 GiB free; saved seeds retained")
                        selector.reset()
                        path = directory / f"seed{seed}_control_results.npz"
                        pending = path.with_suffix(".pending.npz")
                        started = time.perf_counter()
                        row = controller.run(seed, pending, reference_mode="graph",
                                             chart_selector=selector, localization_history_length=12)
                        row.update(evaluation_version=VERSION, evaluation_signature_sha256=signature,
                            solver="CLARABEL", output_dimension=8,
                            chart_selection_method="full_window_projection", graph_cost_method="chordal",
                            elapsed_rollout_wall_s=time.perf_counter()-started)
                        with np.load(pending, allow_pickle=True) as trace:
                            arrays = {k: trace[k] for k in trace.files if k != "summary"}
                        np.savez_compressed(pending, **arrays,
                            selection_windows=np.asarray(selector.windows),
                            selection_residuals=np.asarray(selector.residual_history),
                            selection_tie_count=np.asarray(selector.tie_history), summary=np.array(json.dumps(row)))
                        # Keep invalid traces out of the resumable checkpoint set.
                        pending.replace(path)
                        try:
                            runner.validate_trace(path, 8, "graph", signature)
                        except Exception:
                            path.replace(pending)
                            raise
                        completed = sum((directory / f"seed{s}_control_results.npz").exists() for s in SEEDS)
                        print(f"Matched guidance: {completed}/{len(SEEDS)}, seed={seed}, success={runner.strict_success(row)}", flush=True)
            runner.summarize_case(root, 8, "graph")
        rows = paired_rows(root)
        runner.save_json(root / "paired_results.json", rows)
        return rows


if __name__ == "__main__":
    run()
