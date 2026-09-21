"""Behavioral-window construction, chart fitting, and routing for Pusher."""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = "8"

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import PusherAtlasConfig
from .environment import ACTION_DIM, PHASE_NAMES, STATE_DIM, behavior_output


@dataclass
class PastScaler:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values):
        return (np.asarray(values, dtype=float) - self.mean) / self.scale


def build_behavior_library(data: dict, cfg: PusherAtlasConfig):
    """Build time-major DeePC windows without crossing episode resets."""
    length = cfg.t_ini + cfg.t_f
    columns, past_features, phases, directions, episodes, starts = [], [], [], [], [], []
    episode_ids = np.asarray(data["episode"], dtype=int)
    step_ids = np.asarray(data["step"], dtype=int)
    for episode in np.unique(episode_ids):
        indices = np.flatnonzero(episode_ids == episode)
        indices = indices[np.argsort(step_ids[indices])]
        u_episode = np.asarray(data["u"], dtype=float)[indices]
        # Pair every input with the measured output after that MuJoCo step.  This
        # makes the last y_ini sample the controller's current measured state and
        # the first optimized future output the one-step-ahead state.
        y_episode = np.asarray(
            [
                behavior_output(state, goal)
                for state, goal in zip(data["x_next"][indices], data["goal"][indices])
            ],
            dtype=float,
        )
        phase_episode = np.asarray(data["phase"], dtype=int)[indices]
        direction_episode = np.asarray(data["direction"], dtype=float)[indices]
        for start in range(len(indices) - length + 1):
            uw = u_episode[start : start + length]
            yw = y_episode[start : start + length]
            column = np.r_[uw.ravel(), yw.ravel()]
            past = np.r_[uw[: cfg.t_ini].ravel(), yw[: cfg.t_ini].ravel()]
            columns.append(column)
            past_features.append(past)
            past_end = start + cfg.t_ini - 1
            phases.append(int(phase_episode[past_end]))
            directions.append(direction_episode[past_end])
            episodes.append(int(episode))
            starts.append(int(start))

    h = np.asarray(columns, dtype=float).T
    z_past = np.asarray(past_features, dtype=float)
    expected_rows = (ACTION_DIM + STATE_DIM) * length
    if h.shape[0] != expected_rows:
        raise RuntimeError(f"Behavior matrix has {h.shape[0]} rows; expected {expected_rows}")
    return {
        "H": h,
        "past_features": z_past,
        "phase": np.asarray(phases, dtype=int),
        "direction": np.asarray(directions, dtype=float),
        "episode": np.asarray(episodes, dtype=int),
        "start": np.asarray(starts, dtype=int),
    }


def split_behavior_rows(h, cfg: PusherAtlasConfig):
    a = ACTION_DIM * cfg.t_ini
    b = a + ACTION_DIM * cfg.t_f
    c = b + STATE_DIM * cfg.t_ini
    return h[:a], h[a:b], h[b:c], h[c:]


def fit_atlas(library: dict, cfg: PusherAtlasConfig):
    """Fit centered affine behavioral charts, exactly without A/B models."""
    raw_past = np.asarray(library["past_features"], dtype=float)
    sklearn_scaler = StandardScaler().fit(raw_past)
    scaler = PastScaler(sklearn_scaler.mean_, np.maximum(sklearn_scaler.scale_, 1e-8))
    normalized = scaler.transform(raw_past)
    gate_features = np.column_stack(
        [normalized, cfg.direction_feature_weight * library["direction"]]
    )
    phase = np.asarray(library["phase"], dtype=int)
    h = np.asarray(library["H"], dtype=float)
    # SVD in physical units would devote nearly all modes to torque/joint-speed
    # variance and discard the centimeter-scale object motion that matters most.
    # Standardize each trajectory row before finding the behavioral subspace,
    # then map its basis back to physical units for the online equality.
    behavior_scale = np.maximum(np.std(h, axis=1), 1e-4)

    charts = []
    labels = np.full(h.shape[1], -1, dtype=int)
    phase_chart_ids = {}
    for phase_id in range(len(PHASE_NAMES)):
        phase_indices = np.flatnonzero(phase == phase_id)
        if phase_indices.size == 0:
            continue
        n_charts = min(cfg.charts_per_phase, max(1, phase_indices.size // 100))
        gate = KMeans(
            n_clusters=n_charts,
            n_init=10,
            random_state=cfg.seed + phase_id,
        ).fit(gate_features[phase_indices])
        chart_ids = []
        for local_id in range(n_charts):
            members = phase_indices[gate.labels_ == local_id]
            chart_id = len(charts)
            labels[members] = chart_id
            hk = h[:, members]
            h_bar = hk.mean(axis=1)
            centered = hk - h_bar[:, None]
            centered_scaled = centered / behavior_scale[:, None]
            u_svd, singular, _ = np.linalg.svd(centered_scaled, full_matrices=False)
            max_components = min(cfg.max_chart_rank, u_svd.shape[1], max(1, members.size - 1))
            energy = np.cumsum(singular**2) / max(np.sum(singular**2), 1e-12)
            energy_rank = int(np.searchsorted(energy, cfg.svd_energy) + 1)
            rank = min(max(energy_rank, min(cfg.min_chart_rank, max_components)), max_components)
            basis = behavior_scale[:, None] * u_svd[:, :rank]
            coordinates = u_svd[:, :rank].T @ centered_scaled
            coefficient_bound = np.maximum(
                cfg.coefficient_scale * np.quantile(np.abs(coordinates), 0.995, axis=1),
                1e-4,
            )
            center = gate.cluster_centers_[local_id]
            gate_distance = np.linalg.norm(gate_features[members] - center, axis=1)
            _, u_f, y_p, y_f = split_behavior_rows(hk, cfg)
            representative_y = np.median(y_p[-STATE_DIM:, :], axis=1)
            direction_center = np.mean(library["direction"][members], axis=0)
            direction_center = direction_center / max(
                np.linalg.norm(direction_center), 1e-9
            )
            reconstruction = h_bar[:, None] + basis @ coordinates
            relative_residual = np.linalg.norm(
                (hk - reconstruction) / behavior_scale[:, None], "fro"
            ) / (
                np.linalg.norm((hk - h_bar[:, None]) / behavior_scale[:, None], "fro")
                + 1e-12
            )
            charts.append(
                {
                    "id": chart_id,
                    "phase": phase_id,
                    "indices": members,
                    "h_bar": h_bar,
                    "basis": basis,
                    "behavior_scale": behavior_scale,
                    "coefficient_bound": coefficient_bound,
                    "member_gate_features": gate_features[members],
                    "member_coordinates": coordinates,
                    "member_future_y": y_f.T.reshape(members.size, cfg.t_f, STATE_DIM),
                    "member_first_action": u_f[:ACTION_DIM].T,
                    "rank": rank,
                    "gate_center": center,
                    "gate_radius": float(np.quantile(gate_distance, 0.99) + 1e-6),
                    "representative_y": representative_y,
                    "direction_center": direction_center,
                    "sample_count": int(members.size),
                    "relative_reconstruction_residual": float(relative_residual),
                    "singular_values": singular,
                }
            )
            chart_ids.append(chart_id)
        phase_chart_ids[phase_id] = chart_ids

    if np.any(labels < 0):
        raise RuntimeError("Every behavioral window must belong to one chart")

    counts = np.zeros((len(charts), len(charts)), dtype=int)
    for k in range(len(labels) - 1):
        if (
            library["episode"][k + 1] == library["episode"][k]
            and library["start"][k + 1] == library["start"][k] + 1
        ):
            counts[labels[k], labels[k + 1]] += 1
    return {
        "charts": charts,
        "labels": labels,
        "transition_counts": counts,
        "phase_chart_ids": phase_chart_ids,
        "past_scaler": scaler,
        "behavior_scale": behavior_scale,
        "direction_feature_weight": float(cfg.direction_feature_weight),
        "direction_candidate_margin": float(cfg.direction_candidate_margin),
        "action_scale": np.maximum(
            np.std(library["H"][:ACTION_DIM], axis=1), 1e-3
        ),
    }


def assign_chart(
    atlas,
    u_ini,
    y_ini,
    direction,
    phase,
    previous_chart=None,
    min_transition_count=1,
    chart_exit_ratio=1.50,
    chart_switch_advantage=1.25,
    force_previous=False,
):
    raw = np.r_[np.asarray(u_ini).ravel(), np.asarray(y_ini).ravel()]
    feature = np.r_[
        atlas["past_scaler"].transform(raw),
        atlas["direction_feature_weight"] * np.asarray(direction),
    ]
    candidates = atlas["phase_chart_ids"].get(int(phase), [])
    if not candidates:
        candidates = list(range(len(atlas["charts"])))
    direction = np.asarray(direction, dtype=float)
    direction_scores = np.asarray(
        [
            direction @ atlas["charts"][chart_id]["direction_center"]
            for chart_id in candidates
        ]
    )
    best_direction_score = float(np.max(direction_scores))
    directional_candidates = [
        chart_id
        for chart_id, score in zip(candidates, direction_scores)
        if score >= best_direction_score - atlas["direction_candidate_margin"]
    ]
    if directional_candidates:
        candidates = directional_candidates
    if previous_chart is not None:
        observed = [
            chart_id
            for chart_id in candidates
            if chart_id == int(previous_chart)
            or atlas["transition_counts"][int(previous_chart), chart_id] >= min_transition_count
        ]
        if observed:
            candidates = observed
    if (
        force_previous
        and previous_chart is not None
        and int(previous_chart) in candidates
        and atlas["charts"][int(previous_chart)]["phase"] == int(phase)
    ):
        chart = atlas["charts"][int(previous_chart)]
        ratio = np.linalg.norm(feature - chart["gate_center"]) / chart["gate_radius"]
        return int(previous_chart), float(ratio)
    distances = np.asarray(
        [
            np.linalg.norm(feature - atlas["charts"][chart_id]["gate_center"])
            for chart_id in candidates
        ]
    )
    ratios = np.asarray(
        [
            distance / atlas["charts"][chart_id]["gate_radius"]
            for distance, chart_id in zip(distances, candidates)
        ]
    )
    position = int(np.argmin(ratios))
    chart_id = int(candidates[position])
    if previous_chart is not None and int(previous_chart) in candidates:
        previous_position = candidates.index(int(previous_chart))
        previous_ratio = float(ratios[previous_position])
        best_ratio = float(ratios[position])
        if (
            atlas["charts"][int(previous_chart)]["phase"] == int(phase)
            and previous_ratio <= chart_exit_ratio
            and previous_ratio <= chart_switch_advantage * max(best_ratio, 1e-9)
        ):
            return int(previous_chart), previous_ratio
    return chart_id, float(ratios[position])


def validate_atlas(library: dict, atlas: dict):
    residuals = np.asarray(
        [chart["relative_reconstruction_residual"] for chart in atlas["charts"]]
    )
    weighted = np.average(
        residuals,
        weights=[chart["sample_count"] for chart in atlas["charts"]],
    )
    return {
        "weighted_relative_reconstruction_residual": float(weighted),
        "max_chart_relative_reconstruction_residual": float(np.max(residuals)),
        "chart_relative_reconstruction_residuals": residuals,
        "behavior_matrix_shape": np.asarray(library["H"].shape, dtype=int),
    }


def nearest_behavior(atlas, chart_id, u_ini, y_ini, direction):
    """Return g and future outputs for the closest demonstrated local behavior."""
    raw = np.r_[np.asarray(u_ini).ravel(), np.asarray(y_ini).ravel()]
    feature = np.r_[
        atlas["past_scaler"].transform(raw),
        atlas["direction_feature_weight"] * np.asarray(direction),
    ]
    chart = atlas["charts"][int(chart_id)]
    index = int(np.argmin(np.linalg.norm(chart["member_gate_features"] - feature, axis=1)))
    return (
        chart["member_coordinates"][:, index],
        chart["member_future_y"][index],
    )


def nearest_behavior_coordinate(atlas, chart_id, u_ini, y_ini, direction):
    """Backward-compatible coefficient-only nearest behavior lookup."""
    return nearest_behavior(atlas, chart_id, u_ini, y_ini, direction)[0]


def select_behavior_with_policy(
    atlas,
    u_ini,
    y_ini,
    direction,
    phase,
    u_prior,
    action_weight=2.0,
    previous_chart=None,
    min_transition_count=1,
):
    """Select a local behavior matching measured past and the offline action prior."""
    raw = np.r_[np.asarray(u_ini).ravel(), np.asarray(y_ini).ravel()]
    feature = np.r_[
        atlas["past_scaler"].transform(raw),
        atlas["direction_feature_weight"] * np.asarray(direction),
    ]
    candidates = list(atlas["phase_chart_ids"].get(int(phase), []))
    if not candidates:
        candidates = list(range(len(atlas["charts"])))
    direction = np.asarray(direction, dtype=float)
    scores = np.asarray(
        [
            direction @ atlas["charts"][chart_id]["direction_center"]
            for chart_id in candidates
        ]
    )
    best_direction = float(np.max(scores))
    candidates = [
        chart_id
        for chart_id, score in zip(candidates, scores)
        if score >= best_direction - atlas["direction_candidate_margin"]
    ]
    if previous_chart is not None:
        observed = [
            chart_id
            for chart_id in candidates
            if chart_id == int(previous_chart)
            or atlas["transition_counts"][int(previous_chart), chart_id]
            >= min_transition_count
        ]
        if observed:
            candidates = observed

    u_prior = np.asarray(u_prior, dtype=float)
    best = None
    for chart_id in candidates:
        chart = atlas["charts"][chart_id]
        past_error = np.mean(
            (chart["member_gate_features"] - feature) ** 2, axis=1
        )
        action_error = np.mean(
            (
                (chart["member_first_action"] - u_prior)
                / atlas["action_scale"]
            )
            ** 2,
            axis=1,
        )
        total = past_error + float(action_weight) * action_error
        member = int(np.argmin(total))
        candidate = (float(total[member]), int(chart_id), member)
        if best is None or candidate[0] < best[0]:
            best = candidate
    _, chart_id, member = best
    chart = atlas["charts"][chart_id]
    gate_ratio = np.linalg.norm(feature - chart["gate_center"]) / chart["gate_radius"]
    return (
        chart_id,
        float(gate_ratio),
        chart["member_coordinates"][:, member],
        chart["member_future_y"][member],
    )
