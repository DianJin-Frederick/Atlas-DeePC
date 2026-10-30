"""Paired rank-62 noise study with frozen rank-59 atlas memberships and graphs."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.noise import measurement_noise_rank59_ablation as rank59
from atlas_deepc.benchmarks.baseline_1000_benchmark import summarize_timing
from atlas_deepc.atlas.candidate_goal_routing_experiment import file_digest

noise, baseline, routed = rank59.noise, rank59.baseline, rank59.routed
ROOT, SEEDS = rank59.ROOT, rank59.SEEDS
RATIOS = (0., .01, .02, .03, .04, .05, .10)
level_name = rank59.level_name
RESULT_ROOT = ROOT / "experiments/routed_deepc_iid/measurement_noise_rank62_lamg005_100_v1"
VERSION = "gaussian_output_offline_online_rank62_lamg005_v1"


def experiment_config():
    return replace(rank59.experiment_config(), svd_rank=62)


@contextmanager
def settings():
    with patch.multiple(noise, VERSION=VERSION, experiment_config=experiment_config):
        yield


def prepare_level(ratio, result_root=RESULT_ROOT):
    source = rank59.RESULT_ROOT / level_name(ratio)
    source_manifest = source / "artifact_manifest.json"
    old = json.loads(source_manifest.read_text())
    assert old["signature"] == {**rank59.signature(), "sd_ratio": ratio, "power_nsr": ratio**2}
    for name, digest in old["artifact_sha256"].items():
        assert file_digest(source / name) == digest
    config = experiment_config()
    wanted = {
        **old["signature"], "version": VERSION,
        "config": {k: v for k, v in asdict(config).items() if k != "seed"},
        "rank59_artifact_manifest_sha256": file_digest(source_manifest),
        "rank62_runner_sha256": file_digest(Path(__file__)),
    }
    folder = Path(result_root) / level_name(ratio)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "artifact_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["signature"] != wanted:
            raise ValueError(f"Incompatible rank-62 cache: {folder}")
        for name, digest in manifest["artifact_sha256"].items():
            assert file_digest(folder / name) == digest
        return manifest
    with threadpool_limits(limits=1):
        start = time.perf_counter()
        H, scale, _ = noise.noisy_library(np.load(baseline.DATA_PATH, mmap_mode="r"), ratio,
                                         old["signal_std"])
        reconstruction_time = time.perf_counter() - start
        atlas = routed.load_atlas(source / "behavioral_atlas.npz")
        start = time.perf_counter()
        controller = baseline.BaselineAtlasDeePC(H, atlas, config)
        cache_time = time.perf_counter() - start
        matrices = [controller.matrix_cache[(k,)][0] for k in range(48)]
        cache = {key: np.stack([m[key] for m in matrices])
                 for key in ("basis", "offset", "g_min", "g_max")}
        assert cache["basis"].shape == (48, 144, 62)
        assert not np.any(cache["offset"])
        with np.load(source / "chart_cache.npz") as previous:
            np.testing.assert_allclose(cache["basis"][:, :, :59], previous["basis"], rtol=1e-10, atol=1e-10)
            for name in ("g_min", "g_max"):
                np.testing.assert_allclose(cache[name][:, :59], previous[name], rtol=1e-10, atol=1e-10)
        np.savez_compressed(folder / "chart_cache.npz", **cache)
        routed.save_atlas(folder / "behavioral_atlas.npz", atlas, config)
        with np.load(source / "behavioral_atlas.npz", allow_pickle=True) as previous:
            with np.load(folder / "behavioral_atlas.npz", allow_pickle=True) as current:
                assert previous.files == current.files
                for key in previous.files:
                    if key != "config":
                        np.testing.assert_array_equal(previous[key], current[key])
        manifest = {**old, "signature": wanted,
            "offline_preparation_s": {"corruption_and_reconstruction": reconstruction_time,
                                      "chart_fit": 0., "svd_cache_and_routes": cache_time},
            "rank59_source": str(source.resolve()), "atlas_reused_from_rank59": True,
            "nested_basis_prefix_verified": True, "same_chart_memberships_and_graph_verified": True,
            "artifact_sha256": {name: file_digest(folder / name)
                                for name in ("chart_cache.npz", "behavioral_atlas.npz")}}
    path.write_text(json.dumps(manifest, indent=2))
    print(f"Prepared {ratio:.0%} noise: rank 62; identical atlas and nested rank-59 basis verified", flush=True)
    return manifest


def initialize_worker(ratio, folder):
    with settings():
        noise.initialize_worker(ratio, folder)
    assert noise._controller.config.svd_rank == 62 and noise._controller.config.lam_g == .05


def run_seed(seed, folder):
    with settings():
        row = noise.run_seed(seed, folder)
    assert row["svd_rank"] == 62 and row["lam_g"] == .05 and row["graph_waypoints_enabled"]
    previous_path = rank59.RESULT_ROOT / level_name(row["noise_sd_ratio"]) / f"seed{seed}_control_results.npz"
    with np.load(previous_path, allow_pickle=True) as previous, np.load(row["result_path"], allow_pickle=True) as current:
        previous_row = json.loads(previous["summary"].item())
        np.testing.assert_array_equal(row["exact_target_xy"], previous_row["exact_target_xy"])
        n = len(current["sensor_true"])
        np.testing.assert_allclose(current["sensor_measured"]-current["sensor_true"],
                                  (previous["sensor_measured"]-previous["sensor_true"])[:n],
                                  rtol=0, atol=1e-13)
    return row


def run_level(ratio, seeds=SEEDS, workers=4, result_root=RESULT_ROOT):
    folder = Path(result_root) / level_name(ratio)
    manifest = prepare_level(ratio, result_root)
    rows, pending = [], []
    for seed in seeds:
        path = folder / f"seed{seed}_control_results.npz"
        if path.exists():
            with np.load(path, allow_pickle=True) as trace:
                row = json.loads(trace["summary"].item())
            assert row["noise_version"] == VERSION and row["noise_sd_ratio"] == ratio and row["seed"] == seed
            assert row["steps"] == 300 and row["evaluation_uses_true_outputs"]
            assert row["svd_rank"] == 62 and row["lam_g"] == .05 and row["graph_waypoints_enabled"]
            rows.append(row)
        else:
            pending.append(seed)
    print(f"Rank 62, noise {ratio:.0%}: {len(rows)} cached, {len(pending)} new", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker,
                                 initargs=(ratio, folder)) as pool:
            futures = [pool.submit(run_seed, seed, folder) for seed in pending]
            for future in as_completed(futures):
                rows.append(future.result())
                if len(rows) % 10 == 0 or len(rows) == len(seeds):
                    count = sum(r["success_30"] and r["success_100"] for r in rows)
                    print(f"Rank 62, noise {ratio:.0%}: {len(rows)}/{len(seeds)}, {count} strict successes", flush=True)
    rows.sort(key=lambda r: r["seed"])
    assert [r["seed"] for r in rows] == sorted(seeds)
    payload = {"version": VERSION, "artifact": manifest, "seeds": list(seeds), "results": rows,
               "timing": summarize_timing([r["result_path"] for r in rows]),
               "workers": workers, "numerical_threads": 1,
               "timing_note": "Shared-computer wall clock; first five steps excluded; four workers; rank panels measured in separate batches; no dedicated warmup."}
    (folder / "summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def run_all(ratios=RATIOS, seeds=SEEDS, workers=4, result_root=RESULT_ROOT):
    seeds = list(map(int, seeds))
    if len(seeds) != 100 or len(set(seeds)) != 100:
        raise ValueError("This paired ablation requires exactly 100 distinct seeds")
    rows = []
    for ratio in ratios:
        rows.extend(run_level(float(ratio), seeds, workers, result_root)["results"])
        (Path(result_root) / "summary.json").write_text(json.dumps({
            "version": VERSION, "requested_sd_ratios": list(ratios), "seeds": seeds, "results": rows,
        }, indent=2))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ratios", type=float, nargs="+", default=RATIOS)
    args = parser.parse_args()
    run_all(args.ratios, workers=args.workers)
