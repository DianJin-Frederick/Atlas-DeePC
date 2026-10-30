"""Gaussian output-measurement noise in the offline and online baseline pipeline."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "atlas_deepc_mpl"))

import numpy as np

from atlas_deepc.baselines import baseline_atlas_deepc as baseline
from atlas_deepc.atlas import routed_deepc_experiment as routed
from atlas_deepc.atlas.candidate_goal_routing_experiment import shortest_path_tables

pipe = routed.pipe
ROOT = baseline.ROOT
VERSION = "measurement_norm_gaussian_v1"
RATIOS = (0.0, 0.01, 0.02, 0.05, 0.10)
SEEDS = tuple(range(211, 311))
OFFLINE_NOISE_SEED = 1941701
ONLINE_NOISE_SEED = 1941702
CHANNELS = ("cos(q1)", "cos(q2)", "sin(q1)", "sin(q2)", "dq1", "dq2",
            "fingertip x", "fingertip y", "cos(q1+q2)", "sin(q1+q2)")
_controller = None
_ratio = None


def experiment_config():
    return replace(baseline.default_config(), lam_g=0.0)


def level_name(ratio):
    return f"sd_ratio_{ratio:.4f}".replace(".", "p")


from atlas_deepc.common.hankel import episode_outputs


from atlas_deepc.common.hankel import output_hankel


def measurement_sigma(y, ratio):
    """Per-coordinate SD: E[||epsilon||² | y] = ratio² ||y||²."""
    y = np.asarray(y, dtype=float)
    if not np.isfinite(ratio) or ratio < 0:
        raise ValueError("Relative noise amplitude must be finite and nonnegative")
    if y.ndim < 1 or y.shape[-1] != pipe.P or not np.isfinite(y).all():
        raise ValueError("Expected finite ten-channel measurement vectors")
    return ratio * np.linalg.norm(y, axis=-1, keepdims=True) / np.sqrt(pipe.P)


def noisy_library(H, ratio):
    clean = episode_outputs(H)
    sigma = measurement_sigma(clean, ratio)
    z = np.random.default_rng(OFFLINE_NOISE_SEED).normal(size=clean.shape)
    eps = sigma * z
    noisy = np.array(H, copy=True)
    noisy[pipe.M*pipe.L:] = output_hankel(clean + eps)
    np.testing.assert_array_equal(noisy[:pipe.M*pipe.L], H[:pipe.M*pipe.L])
    np.testing.assert_array_equal(episode_outputs(noisy), clean + eps)
    return noisy, sigma, eps


def initialize_worker(ratio, folder):
    global _controller, _ratio
    folder = Path(folder)
    clean = np.load(baseline.DATA_PATH, mmap_mode="r")
    H, _, _ = noisy_library(clean, ratio)
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
    _controller, _ratio = controller, ratio


class NoisyOutputObserver:
    """Independent sensor noise; true outputs are logged but never returned to control."""

    def __init__(self, seed, ratio, output_map):
        self.rng = np.random.default_rng(np.random.SeedSequence([ONLINE_NOISE_SEED, int(seed)]))
        self.ratio, self.output_map = ratio, output_map
        self.standard_normal, self.sigmas = [], []
        self.true, self.measured = [], []
        self.target = None

    def __call__(self, obs):
        true = self.output_map(obs)
        sigma = measurement_sigma(true, self.ratio)
        z = self.rng.normal(size=pipe.P)
        measured = true + sigma*z
        self.standard_normal.append(z.copy())
        self.sigmas.append(float(sigma.item()))
        self.true.append(true.copy())
        self.measured.append(measured.copy())
        target = np.asarray(obs)[4:6].copy()
        if self.target is not None:
            np.testing.assert_array_equal(target, self.target)
        self.target = target
        return measured


from atlas_deepc.common.metrics import true_error_metrics


def run_seed(seed, folder):
    final_path = Path(folder) / f"seed{seed}_control_results.npz"
    path = Path(folder) / f".seed{seed}_pending.npz"
    observer = NoisyOutputObserver(seed, _ratio, pipe.reacher_output_from_obs)
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
    arrays["sensor_standard_normal"] = np.asarray(observer.standard_normal)
    arrays["sensor_noise_sigma"] = np.asarray(observer.sigmas)
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
                "noise_model": "measurement_norm_gaussian_v1",
                "measurement_noise_sigma_min": float(min(observer.sigmas)),
                "measurement_noise_sigma_max": float(max(observer.sigmas)), "evaluation_uses_true_outputs": True,
                "offline_and_online_noise": True, "exact_target_xy": observer.target.tolist(),
                "noise_on_inputs": False, "result_path": str(final_path.resolve()),
                "online_noise_empirical_rms": np.sqrt(np.mean((measured-truth)**2, axis=0)).tolist()})
    np.savez_compressed(path, **arrays, summary=np.array(json.dumps(row)))
    path.replace(final_path)
    return row

