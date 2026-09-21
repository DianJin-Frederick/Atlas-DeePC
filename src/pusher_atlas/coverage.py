"""Offline and held-out coverage diagnostics for Pusher behavioral data."""

from __future__ import annotations

from pathlib import Path
import json
import os

_CACHE_ROOT = Path(__file__).resolve().parents[2] / ".cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .atlas import build_behavior_library
from .environment import PHASE_NAMES


def _gate_features(library, atlas):
    normalized = atlas["past_scaler"].transform(library["past_features"])
    return np.column_stack(
        [normalized, atlas["direction_feature_weight"] * library["direction"]]
    )


def _phase_counts(values):
    values = np.asarray(values, dtype=int)
    return {
        str(name): int(np.sum(values == phase))
        for phase, name in enumerate(PHASE_NAMES)
    }


def validate_offline_coverage(data, library, atlas, cfg, probe_data=None):
    """Quantify task coverage, excitation, chart support, and held-out novelty."""
    metadata = np.asarray(data["episode_metadata"], dtype=float)
    summary = np.asarray(data["episode_summary"], dtype=float)
    directions = metadata[:, 5:7]
    angles = np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2.0 * np.pi)
    direction_histogram, _ = np.histogram(angles, bins=12, range=(0.0, 2.0 * np.pi))
    action = np.asarray(data["u"], dtype=float)
    action_centered = action - action.mean(axis=0)
    action_singular = np.linalg.svd(action_centered, compute_uv=False)
    action_rank = int(np.sum(action_singular > 1e-6 * max(action_singular[0], 1e-12)))
    phase_counts = _phase_counts(data["phase"])
    chart_sizes = np.asarray([chart["sample_count"] for chart in atlas["charts"]])
    chart_ranks = np.asarray([chart["rank"] for chart in atlas["charts"]])
    transition_counts = np.asarray(atlas["transition_counts"], dtype=int)
    phase_transitions = np.zeros((len(PHASE_NAMES), len(PHASE_NAMES)), dtype=int)
    episode = np.asarray(data["episode"], dtype=int)
    phase = np.asarray(data["phase"], dtype=int)
    step = np.asarray(data["step"], dtype=int)
    for index in range(len(phase) - 1):
        if episode[index + 1] == episode[index] and step[index + 1] == step[index] + 1:
            phase_transitions[phase[index], phase[index + 1]] += 1

    train_features = _gate_features(library, atlas)
    rng = np.random.default_rng(cfg.seed + 818)
    train_query_indices = rng.choice(
        len(train_features), size=min(5000, len(train_features)), replace=False
    )
    neighbors = NearestNeighbors(n_neighbors=2).fit(train_features)
    train_nn = neighbors.kneighbors(
        train_features[train_query_indices], n_neighbors=2, return_distance=True
    )[0][:, 1]

    probe_nn = np.array([], dtype=float)
    probe_gate_ratio = np.array([], dtype=float)
    probe_phase_ratio = {}
    probe_library = None
    if probe_data is not None:
        probe_library = build_behavior_library(probe_data, cfg)
        probe_features = _gate_features(probe_library, atlas)
        probe_nn = neighbors.kneighbors(
            probe_features, n_neighbors=1, return_distance=True
        )[0][:, 0]
        ratios = []
        for feature, phase_id in zip(probe_features, probe_library["phase"]):
            candidates = atlas["phase_chart_ids"].get(int(phase_id), [])
            chart_ratios = [
                np.linalg.norm(feature - atlas["charts"][chart_id]["gate_center"])
                / atlas["charts"][chart_id]["gate_radius"]
                for chart_id in candidates
            ]
            ratios.append(min(chart_ratios) if chart_ratios else np.inf)
        probe_gate_ratio = np.asarray(ratios)
        for phase_id, name in enumerate(PHASE_NAMES):
            mask = probe_library["phase"] == phase_id
            probe_phase_ratio[str(name)] = (
                float(np.mean(probe_gate_ratio[mask] <= 1.0)) if np.any(mask) else 0.0
            )

    train_p95 = float(np.quantile(train_nn, 0.95))
    probe_p95 = float(np.quantile(probe_nn, 0.95)) if probe_nn.size else np.nan
    novelty_ratio = probe_p95 / max(train_p95, 1e-9) if probe_nn.size else np.nan
    contact_fractions = metadata[:, 2]
    offline_success = metadata[:, 3]
    checks = {
        "all_action_directions_excited": action_rank == action.shape[1],
        "most_episodes_make_contact": float(np.mean(contact_fractions >= 0.02)) >= 0.85,
        "offline_success_rate_at_least_65pct": float(np.mean(offline_success)) >= 0.65,
        "at_least_half_direction_bins_occupied": int(np.count_nonzero(direction_histogram)) >= 6,
        "each_phase_has_minimum_windows": min(phase_counts.values()) >= cfg.min_chart_rank,
        "each_chart_has_more_samples_than_rank": bool(np.all(chart_sizes > chart_ranks)),
        "approach_to_push_observed": bool(phase_transitions[0, 1] > 0),
        "push_to_hold_observed": bool(phase_transitions[1, 2] > 0),
    }
    if probe_nn.size:
        checks.update(
            {
                "probe_p95_novelty_below_3x_train": novelty_ratio <= 3.0,
                "probe_inside_chart_radius_at_least_75pct": float(
                    np.mean(probe_gate_ratio <= 1.0)
                )
                >= 0.75,
                "every_probe_phase_inside_radius_at_least_60pct": min(
                    probe_phase_ratio.values()
                )
                >= 0.60,
            }
        )

    metrics = {
        "offline_episodes": int(summary.shape[0]),
        "offline_success_rate": float(np.mean(offline_success)),
        "offline_reach_10cm_rate": float(np.mean(summary[:, 1] <= 0.10)),
        "episode_contact_rate": float(np.mean(contact_fractions >= 0.02)),
        "mean_contact_fraction": float(np.mean(contact_fractions)),
        "mean_object_motion_m": float(np.mean(summary[:, 2])),
        "direction_bins_occupied": int(np.count_nonzero(direction_histogram)),
        "direction_bin_count": int(direction_histogram.size),
        "initial_distance_min_m": float(np.min(summary[:, 0])),
        "initial_distance_max_m": float(np.max(summary[:, 0])),
        "action_covariance_rank": action_rank,
        "action_dimension": int(action.shape[1]),
        "action_saturation_fraction": float(
            np.mean(np.abs(action) >= 0.99 * cfg.torque_limit)
        ),
        "phase_window_counts": phase_counts,
        "chart_count": int(len(chart_sizes)),
        "minimum_chart_size": int(chart_sizes.min()),
        "median_chart_size": float(np.median(chart_sizes)),
        "minimum_chart_rank_margin": int(np.min(chart_sizes - chart_ranks)),
        "observed_chart_edges": int(np.count_nonzero(transition_counts)),
        "observed_cross_chart_edges": int(
            np.count_nonzero(transition_counts - np.diag(np.diag(transition_counts)))
        ),
        "phase_transition_counts": phase_transitions.tolist(),
        "train_nn_p95": train_p95,
        "probe_nn_p95": probe_p95,
        "probe_to_train_p95_novelty_ratio": float(novelty_ratio),
        "probe_inside_chart_radius_rate": (
            float(np.mean(probe_gate_ratio <= 1.0)) if probe_gate_ratio.size else np.nan
        ),
        "probe_phase_inside_chart_radius_rate": probe_phase_ratio,
    }
    if "orientation_turn" in data:
        episode_turn = np.asarray(data["orientation_turn"], dtype=bool)[
            np.asarray(data["step"], dtype=int) == 0
        ]
        metrics.update(
            orientation_turn_episode_count=int(np.sum(episode_turn)),
            orientation_turn_episode_rate=float(np.mean(episode_turn)),
            orientation_turn_success_rate=(
                float(np.mean(offline_success[episode_turn]))
                if np.any(episode_turn)
                else np.nan
            ),
            standard_orientation_success_rate=(
                float(np.mean(offline_success[~episode_turn]))
                if np.any(~episode_turn)
                else np.nan
            ),
        )
    return {
        "coverage_sufficient": bool(all(checks.values())),
        "checks": checks,
        "metrics": metrics,
        "diagnostics": {
            "direction_histogram": direction_histogram,
            "train_nn_distance": train_nn,
            "probe_nn_distance": probe_nn,
            "probe_gate_ratio": probe_gate_ratio,
            "chart_sizes": chart_sizes,
            "chart_ranks": chart_ranks,
        },
    }


def serializable_coverage(report):
    """Return the compact JSON part of a coverage report."""
    return {
        "coverage_sufficient": report["coverage_sufficient"],
        "checks": report["checks"],
        "metrics": report["metrics"],
    }


def save_coverage_report(report, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable_coverage(report), indent=2, sort_keys=True))
    return path


def plot_coverage(data, report, output_path: str | Path | None = None):
    metadata = np.asarray(data["episode_metadata"], dtype=float)
    summary = np.asarray(data["episode_summary"], dtype=float)
    diagnostics = report["diagnostics"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    success = metadata[:, 3].astype(bool)
    axes[0, 0].quiver(
        metadata[:, 7],
        metadata[:, 8],
        metadata[:, 5],
        metadata[:, 6],
        success.astype(float),
        angles="xy",
        scale_units="xy",
        scale=8,
        cmap="coolwarm",
    )
    axes[0, 0].set(title="Stratified reset directions (red = success)", xlabel="object x", ylabel="object y")
    axes[0, 0].axis("equal")

    for variant in np.unique(metadata[:, 1].astype(int)):
        mask = metadata[:, 1] == variant
        axes[0, 1].scatter(
            summary[mask, 0], summary[mask, 1], label=f"variant {variant}", alpha=0.75
        )
    limit = float(max(summary[:, :2].max(), 0.1))
    axes[0, 1].plot([0, limit], [0, limit], "k--", lw=1)
    axes[0, 1].axhline(0.08, color="C3", ls=":")
    axes[0, 1].set(title="Offline teacher progress", xlabel="initial distance [m]", ylabel="minimum distance [m]")
    axes[0, 1].legend(fontsize=8)

    indices = np.arange(len(diagnostics["chart_sizes"]))
    axes[1, 0].bar(indices, diagnostics["chart_sizes"], label="samples")
    axes[1, 0].plot(indices, diagnostics["chart_ranks"], "o-", color="C3", label="rank")
    axes[1, 0].set(title="Behavioral chart support", xlabel="chart", ylabel="count")
    axes[1, 0].legend()

    axes[1, 1].hist(diagnostics["train_nn_distance"], bins=35, alpha=0.65, density=True, label="train self-NN")
    if diagnostics["probe_nn_distance"].size:
        axes[1, 1].hist(diagnostics["probe_nn_distance"], bins=35, alpha=0.65, density=True, label="held-out probe")
    axes[1, 1].set(title="Normalized past-behavior novelty", xlabel="nearest-neighbor distance", ylabel="density")
    axes[1, 1].legend()
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig
