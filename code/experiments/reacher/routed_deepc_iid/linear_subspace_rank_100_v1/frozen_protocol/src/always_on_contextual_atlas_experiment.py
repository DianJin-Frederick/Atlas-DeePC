"""Always-on Q-weighted chart-union Atlas-DeePC.

At every control step, the current physical chart is localized from Y_p. The
same chart-local Q-weighted Y_f selector used by the triggered method chooses
an observed outgoing chart. DeePC then receives the cached rank-matched basis
built from all full Hankel columns in the union of the two charts.

This module deliberately shares the selector, pair-basis artifact, controller,
and rollout implementation with ``future_output_atlas_experiment``. The only
difference from triggered Atlas is that contextual chart union is active at
every step instead of being enabled by terminal stagnation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from future_output_atlas_experiment import (
    DEFAULT_DATA_PATH,
    FutureOutputAtlasConfig,
    run_future_output_atlas,
    run_panel as run_activation_panel,
)


AlwaysOnContextualConfig = FutureOutputAtlasConfig


def run_always_on_contextual_atlas(
    H,
    atlas,
    artifact,
    control_config,
    contextual_config: AlwaysOnContextualConfig,
    save_path: Path | None = None,
) -> dict:
    return run_future_output_atlas(
        H,
        atlas,
        artifact,
        control_config,
        contextual_config,
        save_path=save_path,
        activation_mode="always_on",
    )


def run_panel(
    seeds,
    baseline_dir: Path,
    output_dir: Path,
    contextual_config: AlwaysOnContextualConfig,
    data_path: Path = DEFAULT_DATA_PATH,
    workers: int = 1,
    overwrite: bool = False,
) -> dict:
    return run_activation_panel(
        seeds,
        baseline_dir,
        output_dir,
        contextual_config,
        data_path=data_path,
        workers=workers,
        overwrite=overwrite,
        activation_mode="always_on",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-stop", type=int, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--modes-per-chart", type=int, default=8)
    parser.add_argument("--score-margin", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_panel(
        list(range(args.seed_start, args.seed_stop + 1)),
        args.baseline_dir,
        args.output_dir,
        AlwaysOnContextualConfig(
            modes_per_chart=args.modes_per_chart,
            score_margin=args.score_margin,
        ),
        workers=args.workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
