"""Original-waypoint reference geometry for the Pure-Pursuit-free experiment.

This module deliberately has no dependency on the saved Pure Pursuit rollout.
It converts the original F1TENTH waypoint CSV into a dense, cyclic reference
whose sample spacing is consistent with the DeePC prediction timestep.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .outputs import wrap_angle


WAYPOINT_COLUMNS = (
    "arc_length",
    "x",
    "y",
    "heading_csv",
    "curvature_csv",
    "speed_csv",
    "acceleration_csv",
)


def periodic_smooth(values: np.ndarray, width: int) -> np.ndarray:
    """Circular moving average with an odd window width."""
    values = np.asarray(values, dtype=float)
    width = max(1, min(int(width), len(values) - (1 - len(values) % 2)))
    if width % 2 == 0:
        width -= 1
    if width <= 1:
        return values.copy()
    half = width // 2
    padded = np.r_[values[-half:], values, values[:half]]
    return np.convolve(padded, np.ones(width) / width, mode="valid")


class OriginalWaypointRaceline:
    """Dense cyclic raceline derived only from the original waypoint CSV.

    The CSV heading is approximately pi/2 behind the standard gym yaw
    convention.  It is corrected before periodic interpolation.  The repeated
    closing row is retained only as the interpolation endpoint and is not made
    a second cyclic sample.
    """

    def __init__(
        self,
        waypoint_file: str | Path,
        sample_spacing: float,
        *,
        wheelbase: float = 0.3302,
        heading_offset: float = np.pi / 2.0,
        curvature_smoothing_samples: int = 21,
    ):
        self.waypoint_file = Path(waypoint_file).resolve()
        self.sample_spacing_requested = float(sample_spacing)
        self.wheelbase = float(wheelbase)
        self.heading_offset = float(heading_offset)

        if self.sample_spacing_requested <= 0.0:
            raise ValueError("sample_spacing must be positive")

        raw = np.loadtxt(self.waypoint_file, delimiter=";", skiprows=3)
        if raw.ndim != 2 or raw.shape[1] != len(WAYPOINT_COLUMNS):
            raise ValueError(
                f"Expected a seven-column F1TENTH waypoint CSV, got shape {raw.shape}"
            )
        if len(raw) < 4:
            raise ValueError("At least four waypoint rows are required")
        if not np.all(np.isfinite(raw)):
            raise ValueError("Waypoint CSV contains non-finite values")
        if not np.all(np.diff(raw[:, 0]) > 0.0):
            raise ValueError("Waypoint arc length must increase strictly")
        if not np.allclose(raw[0, 1:3], raw[-1, 1:3], atol=1e-6):
            raise ValueError("Waypoint CSV must finish with a duplicate closing position")

        self.source = raw
        self.source_unique = raw[:-1].copy()
        self.lap_length = float(raw[-1, 0] - raw[0, 0])
        source_s = raw[:, 0] - raw[0, 0]

        # Convert the optimizer/CSV convention into the gym's standard yaw
        # convention, then unwrap through the duplicated closing row.
        source_psi = np.unwrap(raw[:, 3] + self.heading_offset)
        source_psi[-1] = source_psi[0] + 2.0 * np.pi

        n_samples = max(4, int(np.round(self.lap_length / self.sample_spacing_requested)))
        self.s = np.linspace(0.0, self.lap_length, n_samples, endpoint=False)
        self.sample_spacing = float(self.lap_length / n_samples)
        self.n = n_samples

        self.xy = np.column_stack(
            [
                np.interp(self.s, source_s, raw[:, 1]),
                np.interp(self.s, source_s, raw[:, 2]),
            ]
        )
        self.psi = np.interp(self.s, source_s, source_psi)
        self.source_curvature = np.interp(self.s, source_s, raw[:, 4])
        self.source_speed = np.interp(self.s, source_s, raw[:, 5])
        self.source_acceleration = np.interp(self.s, source_s, raw[:, 6])

        segment = np.roll(self.xy, -1, axis=0) - self.xy
        self.ds = np.maximum(np.linalg.norm(segment, axis=1), 1e-8)
        geometric_curvature = wrap_angle(np.roll(self.psi, -1) - self.psi) / self.ds
        self.curvature = periodic_smooth(
            geometric_curvature, curvature_smoothing_samples
        )

        # Validate the corrected CSV heading against the actual x-y tangent.
        source_xy = self.source_unique[:, 1:3]
        source_delta = np.roll(source_xy, -1, axis=0) - source_xy
        source_tangent = np.arctan2(source_delta[:, 1], source_delta[:, 0])
        corrected_source_heading = self.source_unique[:, 3] + self.heading_offset
        source_heading_error = wrap_angle(source_tangent - corrected_source_heading)
        self.heading_validation_rmse = float(
            np.sqrt(np.mean(source_heading_error**2))
        )
        self.heading_validation_max = float(np.max(np.abs(source_heading_error)))
        if self.heading_validation_max > 0.1:
            raise ValueError(
                "Corrected waypoint headings disagree with the x-y tangent; "
                "check the heading convention"
            )

    def nearest_global(self, xy: np.ndarray) -> int:
        return int(np.argmin(np.sum((self.xy - np.asarray(xy)) ** 2, axis=1)))

    def localize(self, xy, previous=None, backward=10, forward=300):
        """Find the closest nearby unwrapped progress index."""
        if previous is None:
            return self.nearest_global(xy)
        offsets = np.arange(-int(backward), int(forward) + 1)
        candidates = int(previous) + offsets
        wrapped = candidates % self.n
        distance = np.sum((self.xy[wrapped] - np.asarray(xy)) ** 2, axis=1)
        return int(candidates[int(np.argmin(distance))])

    def initial_state(self, index, speed, lateral=0.0, heading=0.0):
        idx = int(index) % self.n
        normal = np.array([-np.sin(self.psi[idx]), np.cos(self.psi[idx])])
        state = np.zeros(7)
        state[:2] = self.xy[idx] + float(lateral) * normal
        state[2] = np.clip(np.arctan(self.wheelbase * self.curvature[idx]), -0.3, 0.3)
        state[3] = float(speed)
        state[4] = float(wrap_angle(self.psi[idx] + float(heading)))
        state[5] = float(speed) * self.curvature[idx]
        return state
