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
VERSION = "physical_sensor_gaussian_v1"
RATIOS = (0., 1., 2., 3., 4., 5., 10.)
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
    return f"multiplier_{ratio:.4f}".replace(".", "p")


from atlas_deepc.common.hankel import episode_outputs


from atlas_deepc.common.hankel import output_hankel


def configure(angle_sd_deg=0.1, position_sd_mm=1.0, velocity_sd_rad_s=0.01):
    """Fixed one-times SDs; environment propagation makes spawned workers agree."""
    values = [angle_sd_deg, position_sd_mm, velocity_sd_rad_s]
    if not np.isfinite(values).all() or min(values) < 0:
        raise ValueError("Sensor standard deviations must be finite and nonnegative")
    for key, value in zip(("ATLAS_SENSOR_ANGLE_DEG", "ATLAS_SENSOR_POSITION_MM", "ATLAS_SENSOR_VELOCITY_RAD_S"), values):
        os.environ[key] = str(float(value))


def sensor_settings():
    return {"angle_sd_deg": float(os.environ.get("ATLAS_SENSOR_ANGLE_DEG", ".1")),
            "position_sd_mm": float(os.environ.get("ATLAS_SENSOR_POSITION_MM", "1")),
            "velocity_sd_rad_s": float(os.environ.get("ATLAS_SENSOR_VELOCITY_RAD_S", ".01"))}


def measurement_sigma(multiplier):
    if not np.isfinite(multiplier) or multiplier < 0:
        raise ValueError("Noise multiplier must be finite and nonnegative")
    s = sensor_settings()
    values = [np.deg2rad(s["angle_sd_deg"])]*2 + [s["velocity_sd_rad_s"]]*2 + [s["position_sd_mm"]/1000]*2
    if not np.isfinite(values).all() or min(values) < 0:
        raise ValueError("Invalid sensor standard deviations")
    return multiplier*np.asarray(values)


def corrupt_outputs(clean, multiplier, z):
    """z columns: q1,q2,dq1,dq2,x,y. Derived features share angle errors."""
    clean, z = np.asarray(clean, dtype=float), np.asarray(z, dtype=float)
    if clean.shape[-1:] != (10,) or z.shape != clean.shape[:-1]+(6,):
        raise ValueError("Expected ten outputs and six primitive sensor draws")
    if not np.isfinite(clean).all() or not np.isfinite(z).all():
        raise ValueError("Nonfinite sensor inputs")
    delta = measurement_sigma(multiplier)*z
    if not np.any(delta):
        return clean.copy()  # Exact identity, including the zero-noise control.
    q = np.arctan2(clean[...,2:4], clean[...,0:2]) + delta[...,:2]
    measured = clean.copy()
    measured[...,0:2], measured[...,2:4] = np.cos(q), np.sin(q)
    measured[...,4:6] += delta[...,2:4]
    measured[...,6:8] += delta[...,4:6]
    measured[...,8], measured[...,9] = np.cos(q.sum(axis=-1)), np.sin(q.sum(axis=-1))
    return measured


def noisy_library(H, multiplier):
    clean = episode_outputs(H)
    z = np.random.default_rng(OFFLINE_NOISE_SEED).normal(size=clean.shape[:-1]+(6,))
    measured = corrupt_outputs(clean, multiplier, z)
    noisy = np.array(H, copy=True)
    noisy[pipe.M*pipe.L:] = output_hankel(measured)
    np.testing.assert_array_equal(noisy[:pipe.M*pipe.L], H[:pipe.M*pipe.L])
    np.testing.assert_array_equal(episode_outputs(noisy), measured)
    return noisy, measurement_sigma(multiplier), z*measurement_sigma(multiplier)


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
        sigma = measurement_sigma(self.ratio)
        z = self.rng.normal(size=6)
        measured = corrupt_outputs(true, self.ratio, z)
        self.standard_normal.append(z.copy())
        self.sigmas.append(sigma.copy())
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
    row.update({"noise_version": VERSION, "noise_multiplier": _ratio, "sensor_settings": sensor_settings(),
                "noise_model": "physical_sensor_gaussian_v1",
                "primitive_noise_sigma": measurement_sigma(_ratio).tolist(), "evaluation_uses_true_outputs": True,
                "offline_and_online_noise": True, "exact_target_xy": observer.target.tolist(),
                "noise_on_inputs": False, "result_path": str(final_path.resolve()),
                "online_noise_empirical_rms": np.sqrt(np.mean((measured-truth)**2, axis=0)).tolist()})
    np.savez_compressed(path, **arrays, summary=np.array(json.dumps(row)))
    path.replace(final_path)
    return row

