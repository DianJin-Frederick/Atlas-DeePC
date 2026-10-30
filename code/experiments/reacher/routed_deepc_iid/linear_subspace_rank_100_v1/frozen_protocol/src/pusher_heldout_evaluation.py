"""Resumable held-out evaluation for the coverage-validated Pusher controller."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from pusher_atlas import PusherAtlasConfig
from pusher_atlas.atlas import build_behavior_library, fit_atlas
from pusher_atlas.data import load_data
from pusher_atlas.experiment import (
    _summary_from_rollouts,
    load_atlas,
    run_rollout,
    save_atlas,
)
from pusher_atlas.policy import load_behavior_policy


def parse_seeds(value: str) -> tuple[int, ...]:
    """Parse comma-separated seeds and inclusive ranges such as ``201-300``."""
    seeds: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first, last = (int(part) for part in item.split("-", maxsplit=1))
            seeds.extend(range(first, last + 1))
        else:
            seeds.append(int(item))
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and unique")
    return tuple(seeds)


def load_rollout(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054):
    if trials == 0:
        return np.nan, np.nan
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    radius = z * np.sqrt(
        rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return float(center - radius), float(center + radius)


def write_summary(
    path: Path,
    results: list[dict],
    seeds: tuple[int, ...],
    source_dir: Path,
    cfg: PusherAtlasConfig,
    started: float,
    method: str,
    controller_variant: str,
):
    summary = _summary_from_rollouts(results)
    successes = sum(bool(np.asarray(result["success"]).item()) for result in results)
    ci_low, ci_high = wilson_interval(successes, len(results))
    rows = [
        {
            "seed": int(np.asarray(result["seed"]).item()),
            "success": bool(np.asarray(result["success"]).item()),
            "reached_goal": bool(np.asarray(result["reached_goal"]).item()),
            "minimum_distance_m": float(np.asarray(result["minimum_distance"]).item()),
            "final_distance_m": float(np.asarray(result["final_distance"]).item()),
            "solve_rate": float(np.asarray(result["solve_rate"]).item()),
            "median_control_compute_ms": float(
                np.asarray(result["median_control_compute_ms"]).item()
            ),
        }
        for result in results
    ]
    summary.update(
        method=method,
        controller_variant=controller_variant,
        policy_prior_weight=float(cfg.policy_prior_weight),
        behavior_anchor_weight=float(cfg.lam_behavior_anchor),
        behavior_reference_weight=float(cfg.behavior_reference_weight),
        source_artifact_dir=str(source_dir),
        requested_seeds=list(seeds),
        completed_seeds=[row["seed"] for row in rows],
        completed_trials=len(rows),
        requested_trials=len(seeds),
        complete=len(rows) == len(seeds),
        successes=successes,
        success_rate_wilson_95=[ci_low, ci_high],
        success_definition=(
            f"the final {cfg.success_hold_steps} consecutive steps within "
            f"{cfg.success_radius:.3f} m"
        ),
        evaluation_wall_time_s_this_invocation=perf_counter() - started,
        results=rows,
    )
    path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("experiments/pusher_atlas/rebuild_300_routed"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/pusher_atlas/heldout201_300"),
    )
    parser.add_argument("--seeds", default="201-300")
    parser.add_argument(
        "--pure-atlas",
        action="store_true",
        help=(
            "Disable the learned action policy, action-prior penalty, "
            "demonstrated-behavior anchor, and demonstrated reference blending."
        ),
    )
    args = parser.parse_args()

    started = perf_counter()
    seeds = parse_seeds(args.seeds)
    data_path = args.source_dir / "offline_teacher_data.npz"
    atlas_path = args.source_dir / "behavioral_atlas.npz"
    for required in (data_path, atlas_path):
        if not required.exists():
            raise FileNotFoundError(required)

    data = load_data(data_path)
    stored_config = json.loads(str(np.asarray(data["config"]).item()))
    cfg = replace(PusherAtlasConfig(**stored_config), test_seeds=seeds)
    if args.pure_atlas:
        cfg = replace(
            cfg,
            policy_prior_weight=0.0,
            lam_behavior_anchor=0.0,
            behavior_reference_weight=0.0,
        )
        behavior_policy = None
        method = "phase-conditioned fixed-chart Pusher Atlas-DeePC (no policy prior)"
        controller_variant = "pure_atlas_no_policy_no_behavior_anchor"
    else:
        regional_policy_path = (
            args.source_dir / "offline_behavior_policy_regional.joblib"
        )
        policy_path = args.source_dir / "offline_behavior_policy.joblib"
        selected_policy_path = (
            regional_policy_path if regional_policy_path.exists() else policy_path
        )
        if not selected_policy_path.exists():
            raise FileNotFoundError(selected_policy_path)
        behavior_policy = load_behavior_policy(selected_policy_path)
        method = "coverage-validated policy-guided Pusher Atlas-DeePC"
        controller_variant = "policy_guided"
    offline_seeds = set(np.asarray(data.get("selected_seeds", []), dtype=int).tolist())
    probe_path = args.source_dir / "heldout_coverage_probe_data.npz"
    probe_seeds = set()
    if probe_path.exists():
        probe = load_data(probe_path)
        probe_seeds = set(np.asarray(probe.get("selected_seeds", []), dtype=int).tolist())
    forbidden = offline_seeds | probe_seeds | set(range(101, 117)) | set(range(131, 161))
    overlap = sorted(set(seeds).intersection(forbidden))
    if overlap:
        raise ValueError(f"Evaluation seeds overlap prior data/development seeds: {overlap}")

    try:
        atlas = load_atlas(atlas_path)
    except ValueError:
        library = build_behavior_library(data, cfg)
        atlas = fit_atlas(library, cfg)
        save_atlas(atlas_path, atlas, cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"

    results: list[dict] = []
    for position, seed in enumerate(seeds, start=1):
        result_path = args.output_dir / f"rollout_seed{seed}.npz"
        if result_path.exists():
            result = load_rollout(result_path)
        else:
            result = run_rollout(atlas, cfg, seed, behavior_policy=behavior_policy)
            np.savez_compressed(result_path, **result)
        results.append(result)
        summary = write_summary(
            summary_path,
            results,
            seeds,
            args.source_dir,
            cfg,
            started,
            method,
            controller_variant,
        )
        print(
            json.dumps(
                {
                    "completed": position,
                    "seed": seed,
                    "success": bool(np.asarray(result["success"]).item()),
                    "successes": summary["successes"],
                    "success_rate": summary["success_rate"],
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise
