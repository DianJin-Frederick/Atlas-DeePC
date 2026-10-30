"""Remove joint-sum outputs from the current linear, iterative-N10 baseline."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

import numpy as np
from threadpoolctl import threadpool_limits

import baseline_atlas_deepc as baseline
import expert_pipeline_search as pipe
import routed_deepc_experiment as routed
from baseline_1000_benchmark import summarize_timing
from candidate_goal_routing_experiment import file_digest, strict_success
from reduced_output_ablation import remove_joint_sum_channels

VERSION = "baseline_output_channels_linear_n10_v1"
ROOT = baseline.ROOT
RESULT_ROOT = ROOT / "experiments/routed_deepc_iid" / VERSION
SEEDS = list(range(211, 311))
_controllers = {}
_thread_limit = None


def prepare():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    config = baseline.default_config()
    signature = {
        "version": VERSION,
        "config": asdict(config),
        "data_sha256": file_digest(baseline.DATA_PATH),
        "baseline_atlas_sha256": file_digest(baseline.ATLAS_PATH),
        "source_sha256": {name: file_digest(ROOT / "src" / name) for name in (
            "baseline_output_channels_ablation.py", "reduced_output_ablation.py",
            "baseline_atlas_deepc.py", "candidate_goal_routing_experiment.py",
            "routed_deepc_experiment.py", "expert_pipeline_search.py",
            "stitched_atlas_mpc_experiment.py")},
        "noise_sd_ratio": 0.0, "dimensions": [10, 8],
        "matrix_geometry": "linear", "candidate_count": 10,
    }
    manifest = RESULT_ROOT / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous != signature:
            corrected = json.loads(json.dumps(previous))
            old_runner = corrected["source_sha256"]["baseline_output_channels_ablation.py"]
            corrected["source_sha256"]["baseline_output_channels_ablation.py"] = signature["source_sha256"]["baseline_output_channels_ablation.py"]
            if old_runner != "fc5466a34eaa16d323d99cae07af6cf36c067a09ed4232bf5f6797f678ec28de" or corrected != signature:
                raise ValueError("Incompatible ablation artifacts; choose a new result directory")
            # This exact earlier wrapper failed after saving a valid baseline trace.
            (RESULT_ROOT / "metadata_wrapper_migration.json").write_text(json.dumps({
                "previous_signature": previous, "new_signature": signature,
                "change": "Read target from saved final_goal; unchanged control and offline artifacts",
            }, indent=2))
            manifest.write_text(json.dumps(signature, indent=2))
    else:
        if any(RESULT_ROOT.glob("output*/seed*.npz")):
            raise ValueError("Unversioned ablation traces")
        manifest.write_text(json.dumps(signature, indent=2))
    atlas_path = RESULT_ROOT / "atlas_output8.npz"
    metadata_path = RESULT_ROOT / "offline_setup.json"
    with threadpool_limits(limits=1):
        if not metadata_path.exists():
            H10 = np.load(baseline.DATA_PATH)
            H8 = remove_joint_sum_channels(H10)
            pipe.P = 8
            started = time.perf_counter()
            atlas = routed.fit_behavioral_atlas(H8, config)
            fit_seconds = time.perf_counter() - started
            routed.save_atlas(atlas_path, atlas, config)
            audit = routed.audit_iid_library(H8, eval_seeds=SEEDS)
            metadata = {"hankel10_shape": list(H10.shape), "hankel8_shape": list(H8.shape),
                        "atlas8_fit_seconds": fit_seconds,
                        "atlas8_sha256": file_digest(atlas_path), "fairness_audit": audit,
                        "chart_sizes8": np.bincount(atlas["labels"], minlength=48).tolist()}
            metadata_path.write_text(json.dumps(metadata, indent=2))
        metadata = json.loads(metadata_path.read_text())
        assert metadata["atlas8_sha256"] == file_digest(atlas_path)
    pipe.P = 10
    return signature


def initialize_worker():
    global _controllers, _thread_limit
    _thread_limit = threadpool_limits(limits=1)
    H10 = np.load(baseline.DATA_PATH)
    for dimension in (10, 8):
        pipe.P = dimension
        H = H10 if dimension == 10 else remove_joint_sum_channels(H10)
        path = baseline.ATLAS_PATH if dimension == 10 else RESULT_ROOT / "atlas_output8.npz"
        atlas = routed.load_atlas(path)
        controller = baseline.BaselineAtlasDeePC(H, atlas)
        assert atlas["medoid_states"].shape == (48, dimension)
        assert atlas["centers"].shape[1] == pipe.M * pipe.T_INI + dimension - 4
        for matrix, _ in controller.matrix_cache.values():
            assert matrix["basis"].shape == ((pipe.M + dimension) * pipe.L, 60)
            assert not np.any(matrix["offset"])
        _controllers[dimension] = controller


def run_pair(seed):
    rows = []
    # Alternate method order to reduce systematic order effects on wall-clock time.
    for dimension in ((10, 8) if seed % 2 else (8, 10)):
        pipe.P = dimension
        folder = RESULT_ROOT / f"output{dimension}"
        folder.mkdir(exist_ok=True)
        path = folder / f"seed{seed}_control_results.npz"
        if not path.exists():
            _controllers[dimension].run(seed, path)
        rows.append(finalize_trace(path, dimension, seed))
    a, b = sorted(rows, key=lambda row: -row["output_dimension"])
    np.testing.assert_array_equal(a["exact_target_xy"], b["exact_target_xy"])
    return rows


def finalize_trace(path, dimension, seed, steps=320):
    with np.load(path, allow_pickle=True) as trace:
        row = json.loads(trace["summary"].item())
        arrays = {key: trace[key] for key in trace.files if key != "summary"}
    expected = {**asdict(baseline.default_config()), "seed": seed, "steps": steps}
    assert all(row[key] == value for key, value in expected.items())
    assert row["baseline_version"] == baseline.BASELINE_VERSION
    assert row["matrix_geometry"] == "linear" and row["candidate_count"] == 10
    assert arrays["y"].shape == arrays["ref"].shape == (steps, dimension)
    target = arrays["final_goal"].reshape(-1)[6:8]
    np.testing.assert_allclose(arrays["distance"], np.linalg.norm(arrays["y"][:, 6:8] - target, axis=1))
    assert row.get("output_ablation_version", VERSION) == VERSION
    assert row.get("output_dimension", dimension) == dimension
    row.update(output_ablation_version=VERSION, output_dimension=dimension,
               noise_sd_ratio=0.0, result_path=str(path), exact_target_xy=target.tolist(),
               removed_output_channels=[] if dimension == 10 else ["cos(q1+q2)", "sin(q1+q2)"])
    np.savez_compressed(path, **arrays, summary=np.array(json.dumps(row)))
    return row


def run_all(workers=4):
    signature = prepare()
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker) as pool:
        futures = [pool.submit(run_pair, seed) for seed in SEEDS]
        for i, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            if i % 10 == 0 or i == len(SEEDS):
                counts = {p: sum(strict_success(r) for r in rows if r["output_dimension"] == p) for p in (10, 8)}
                print(f"{i}/100 paired seeds: 10-output successes={counts[10]}, 8-output successes={counts[8]}", flush=True)
    payload = {"version": VERSION, "signature": signature, "seeds": SEEDS, "workers": workers,
               "timing_note": "Fresh paired runs, alternating method order, one numerical thread per worker; shared-computer timing",
               "offline": json.loads((RESULT_ROOT / "offline_setup.json").read_text()), "panels": {}}
    for dimension in (10, 8):
        panel = sorted([row for row in rows if row["output_dimension"] == dimension], key=lambda row: row["seed"])
        assert [row["seed"] for row in panel] == SEEDS
        payload["panels"][str(dimension)] = {
            "results": panel,
            "timing": summarize_timing([row["result_path"] for row in panel]),
            "successes": sum(strict_success(row) for row in panel),
            "median_tail100_mm": float(1000*np.median([row["mean_last_100"] for row in panel])),
        }
    baseline_rows = payload["panels"]["10"]["results"]
    reduced_rows = payload["panels"]["8"]["results"]
    payload["paired"] = {
        "repaired": [a["seed"] for a, b in zip(baseline_rows, reduced_rows) if not strict_success(a) and strict_success(b)],
        "regressed": [a["seed"] for a, b in zip(baseline_rows, reduced_rows) if strict_success(a) and not strict_success(b)],
    }
    (RESULT_ROOT / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({p: {k: v for k, v in panel.items() if k not in ("results", "timing")}
                      for p, panel in payload["panels"].items()}, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    run_all(parser.parse_args().workers)
