"""Gaussian output-measurement noise in the offline and online baseline pipeline."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

import numpy as np
from threadpoolctl import threadpool_limits

import baseline_atlas_deepc as baseline
import routed_deepc_experiment as routed
from baseline_1000_benchmark import summarize_timing
from candidate_goal_routing_experiment import file_digest, shortest_path_tables

pipe = routed.pipe
ROOT = baseline.ROOT
PREVIOUS_RESULT_ROOT = ROOT / "experiments/routed_deepc_iid/measurement_noise_baseline_100_v1"
RESULT_ROOT = ROOT / "experiments/routed_deepc_iid/measurement_noise_lam_g0_100_v1"
VERSION = "gaussian_output_offline_online_lam_g0_v1"
RATIOS = (0.0, 0.01, 0.02, 0.05, 0.10)
SEEDS = tuple(range(211, 311))
OFFLINE_NOISE_SEED = 1941701
ONLINE_NOISE_SEED = 1941702
CHANNELS = ("cos(q1)", "cos(q2)", "sin(q1)", "sin(q2)", "dq1", "dq2",
            "fingertip x", "fingertip y", "cos(q1+q2)", "sin(q1+q2)")
_controller = None
_sigma = None
_ratio = None


def experiment_config():
    return replace(baseline.default_config(), lam_g=0.0)


def level_name(ratio):
    return f"sd_ratio_{ratio:.4f}".replace(".", "p")


def episode_outputs(H):
    """Recover unique time samples rather than independently corrupting overlaps."""
    windows = routed.WINDOWS_PER_EPISODE
    assert H.shape[1] % windows == 0
    blocks = H[pipe.M*pipe.L:].reshape(pipe.L, pipe.P, -1, windows).transpose(2, 3, 0, 1)
    outputs = np.concatenate([blocks[:, :, 0], blocks[:, -1, 1:]], axis=1)
    rebuilt = output_hankel(outputs)
    np.testing.assert_array_equal(rebuilt, H[pipe.M*pipe.L:])
    return outputs


def output_hankel(outputs):
    windows = outputs.shape[1] - pipe.L + 1
    blocks = np.stack([outputs[:, lag:lag+windows] for lag in range(pipe.L)])
    return blocks.transpose(0, 3, 1, 2).reshape(pipe.P*pipe.L, -1)


def noisy_library(H, ratio, signal_std=None):
    if not np.isfinite(ratio) or ratio < 0:
        raise ValueError("Noise standard-deviation ratio must be finite and nonnegative")
    clean = episode_outputs(H)
    scale = clean.std(axis=(0, 1)) if signal_std is None else np.asarray(signal_std)
    assert scale.shape == (pipe.P,) and np.all(scale > 0)
    standard_noise = np.random.default_rng(OFFLINE_NOISE_SEED).normal(size=clean.shape)
    noise = ratio * scale * standard_noise
    measured = clean + noise
    noisy = np.array(H, copy=True)
    noisy[pipe.M*pipe.L:] = output_hankel(measured)
    np.testing.assert_array_equal(noisy[:pipe.M*pipe.L], H[:pipe.M*pipe.L])
    # This round trip also validates every overlapping sample after corruption.
    np.testing.assert_array_equal(episode_outputs(noisy), measured)
    return noisy, scale, noise


def signature():
    return {
        "version": VERSION, "baseline_version": baseline.BASELINE_VERSION,
        "config": {k: v for k, v in asdict(experiment_config()).items() if k != "seed"},
        "data_sha256": file_digest(baseline.DATA_PATH), "clean_atlas_sha256": file_digest(baseline.ATLAS_PATH),
        "offline_noise_seed": OFFLINE_NOISE_SEED, "online_noise_seed": ONLINE_NOISE_SEED,
        "noise_coordinates": "independent additive Gaussian noise in all ten output channels; inputs exact",
        "source_sha256": {name: file_digest(ROOT / "src" / name) for name in (
            "measurement_noise_ablation.py", "baseline_atlas_deepc.py", "routed_deepc_experiment.py",
            "candidate_goal_routing_experiment.py", "expert_pipeline_search.py")},
    }


def prepare_level(ratio, result_root=RESULT_ROOT):
    folder = Path(result_root) / level_name(ratio)
    folder.mkdir(parents=True, exist_ok=True)
    wanted = {**signature(), "sd_ratio": ratio, "power_nsr": ratio**2}
    path = folder / "artifact_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["signature"] != wanted:
            raise ValueError(f"Incompatible noise artifact: {folder}")
        for name, digest in manifest["artifact_sha256"].items():
            assert file_digest(folder / name) == digest
        return manifest
    with threadpool_limits(limits=1):
        start = time.perf_counter()
        H = np.load(baseline.DATA_PATH, mmap_mode="r")
        noisy, scale, noise = noisy_library(H, ratio)
        corruption_time = time.perf_counter() - start
        start = time.perf_counter()
        if ratio == 0:
            atlas = routed.load_atlas(baseline.ATLAS_PATH)
        else:
            atlas = routed.fit_behavioral_atlas(noisy, experiment_config())
        fit_time = time.perf_counter() - start
        start = time.perf_counter()
        controller = baseline.BaselineAtlasDeePC(noisy, atlas, experiment_config())
        cache_time = time.perf_counter() - start
        assert len(controller.matrix_cache) == 48
        matrices = [controller.matrix_cache[(k,)][0] for k in range(48)]
        assert all(m["basis"].shape == (H.shape[0], 60) and not np.any(m["offset"]) for m in matrices)
        np.savez_compressed(folder / "chart_cache.npz", **{
            key: np.stack([m[key] for m in matrices]) for key in ("basis", "offset", "g_min", "g_max")})
        routed.save_atlas(folder / "behavioral_atlas.npz", atlas, controller.config)
        manifest = {
            "signature": wanted, "signal_std": scale.tolist(), "noise_std": (ratio*scale).tolist(),
            "empirical_offline_power_nsr": (np.mean(noise**2, axis=(0, 1))/scale**2).tolist(),
            "offline_unique_samples": int(np.prod(noise.shape[:2])), "offline_episodes": noise.shape[0],
            "steps_per_offline_episode": noise.shape[1], "channels": list(CHANNELS),
            "chart_sizes": np.bincount(atlas["labels"], minlength=48).tolist(),
            "unreachable_chart_pairs": int(np.count_nonzero(~np.isfinite(controller.route_tables[0]))),
            "offline_preparation_s": {"corruption_and_reconstruction": corruption_time,
                                      "chart_fit": fit_time, "svd_cache_and_routes": cache_time},
            "artifact_sha256": {name: file_digest(folder / name) for name in ("chart_cache.npz", "behavioral_atlas.npz")},
            "overlap_consistency_verified": True, "applied_inputs_unchanged": True,
        }
    path.write_text(json.dumps(manifest, indent=2))
    print(f"Prepared SD ratio {ratio:.1%}: fingertip noise SD {1000*ratio*scale[6:8]} mm", flush=True)
    return manifest


def initialize_worker(ratio, folder):
    global _controller, _sigma, _ratio
    folder = Path(folder)
    manifest = json.loads((folder / "artifact_manifest.json").read_text())
    clean = np.load(baseline.DATA_PATH, mmap_mode="r")
    H, _, _ = noisy_library(clean, ratio, manifest["signal_std"])
    atlas = routed.load_atlas(folder / "behavioral_atlas.npz")
    # Load the offline bases instead of repeating SVD inside each worker.
    controller = object.__new__(baseline.BaselineAtlasDeePC)
    controller.config, controller.H, controller.atlas = experiment_config(), H, atlas
    with np.load(folder / "chart_cache.npz") as cache:
        controller.matrix_cache = {
            (k,): ({key: cache[key][k] for key in cache.files}, np.flatnonzero(atlas["labels"] == k))
            for k in range(48)
        }
    controller.route_tables = shortest_path_tables(atlas, controller.config.edge_count_threshold)
    _controller, _sigma, _ratio = controller, np.array(manifest["noise_std"]), ratio


class NoisyOutputObserver:
    """Independent sensor noise; true outputs are logged but never returned to control."""

    def __init__(self, seed, sigma, output_map):
        self.rng = np.random.default_rng(np.random.SeedSequence([ONLINE_NOISE_SEED, int(seed)]))
        self.sigma, self.output_map = np.asarray(sigma), output_map
        self.true, self.measured = [], []
        self.target = None

    def __call__(self, obs):
        true = self.output_map(obs)
        measured = true + self.sigma*self.rng.normal(size=pipe.P)
        self.true.append(true.copy())
        self.measured.append(measured.copy())
        target = np.asarray(obs)[4:6].copy()
        if self.target is not None:
            np.testing.assert_array_equal(target, self.target)
        self.target = target
        return measured


def true_error_metrics(distance):
    result = {"minimum": float(distance.min()), "minimum_step": int(distance.argmin()), "last": float(distance[-1])}
    for n in (30, 100):
        tail = distance[-n:]
        result.update({f"mean_last_{n}": float(tail.mean()), f"max_last_{n}": float(tail.max()),
                       f"success_{n}": bool(tail.mean() < .01 and tail.max() < .02)})
    return result


def run_seed(seed, folder):
    path = Path(folder) / f"seed{seed}_control_results.npz"
    observer = NoisyOutputObserver(seed, _sigma, pipe.reacher_output_from_obs)
    # Worker-local patch injects noise at the existing sensor boundary. The rollout,
    # chart/goal policies, optimization, and actuator handling are reused verbatim.
    with patch.object(pipe, "reacher_output_from_obs", observer):
        row = _controller.run(seed, path)
    with np.load(path, allow_pickle=True) as trace:
        arrays = {key: trace[key] for key in trace.files if key != "summary"}
    truth, measured = np.asarray(observer.true), np.asarray(observer.measured)
    steps = _controller.config.steps
    assert len(truth) == 1 + pipe.T_INI + steps
    np.testing.assert_array_equal(arrays["y"], measured[-steps:])
    arrays["y_measured"] = arrays["y"]
    arrays["y"] = truth[-steps:]
    arrays["sensor_true"] = truth
    arrays["sensor_measured"] = measured
    arrays["distance_measured"] = arrays["distance"]
    arrays["distance"] = np.linalg.norm(arrays["y"][:, 6:8]-observer.target, axis=1)
    arrays["prediction_error_xy_measured"] = arrays["prediction_error_xy"]
    arrays["prediction_error_xy"] = np.linalg.norm(arrays["pred"][:, 6:8]-arrays["y"][:, 6:8], axis=1)
    row["measured_error_metrics"] = true_error_metrics(arrays["distance_measured"])
    row.update(true_error_metrics(arrays["distance"]))
    row["measurement_based_costs"] = {key: row.pop(key) for key in list(row) if "cost" in key}
    row["mean_prediction_error_xy"] = float(np.nanmean(arrays["prediction_error_xy"]))
    row.update({"noise_version": VERSION, "noise_sd_ratio": _ratio, "noise_power_nsr": _ratio**2,
                "measurement_noise_std": _sigma.tolist(), "evaluation_uses_true_outputs": True,
                "offline_and_online_noise": True, "exact_target_xy": observer.target.tolist(),
                "noise_on_inputs": False, "result_path": str(path.resolve()),
                "online_noise_empirical_rms": np.sqrt(np.mean((measured-truth)**2, axis=0)).tolist()})
    np.savez_compressed(path, **arrays, summary=np.array(json.dumps(row)))
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
            assert row["steps"] == 320 and row["evaluation_uses_true_outputs"]
            assert row["lam_g"] == 0.0
            rows.append(row)
        else:
            pending.append(seed)
    print(f"SD ratio {ratio:.1%}: {len(pending)} new runs; {len(rows)} cached", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker, initargs=(ratio, folder)) as pool:
            futures = [pool.submit(run_seed, seed, folder) for seed in pending]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                if len(rows) % 10 == 0 or len(rows) == len(seeds):
                    successes = sum(r["success_30"] and r["success_100"] for r in rows)
                    print(f"SD ratio {ratio:.1%}: {len(rows)}/{len(seeds)} complete, {successes} successful so far", flush=True)
    rows.sort(key=lambda row: row["seed"])
    assert [r["seed"] for r in rows] == sorted(seeds)
    payload = {"version": VERSION, "artifact": manifest, "seeds": list(seeds), "results": rows,
               "timing": summarize_timing([r["result_path"] for r in rows]),
               "workers": workers, "numerical_threads": 1,
               "timing_note": "Shared-computer wall clock; four workers by default; first five steps excluded, no dedicated warmup; sensor noise injection excluded from control timer."}
    (folder / "summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def run_all(ratios=RATIOS, seeds=SEEDS, workers=4, result_root=RESULT_ROOT):
    seeds = list(map(int, seeds))
    if len(seeds) != 100 or len(set(seeds)) != 100:
        raise ValueError("This notebook ablation uses exactly 100 distinct seeds")
    result_root = Path(result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    all_results = []
    for ratio in ratios:
        payload = run_level(float(ratio), seeds, workers, result_root)
        all_results.extend(payload["results"])
        (result_root / "summary.json").write_text(json.dumps({
            "version": VERSION, "requested_sd_ratios": list(ratios), "seeds": seeds,
            "results": all_results, "success_definition": "true-error means <1 cm and maxima <2 cm in both final-30 and final-100 windows",
        }, indent=2))
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ratios", type=float, nargs="+", default=RATIOS)
    args = parser.parse_args()
    run_all(args.ratios, workers=args.workers)
