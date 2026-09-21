"""Verified paired media from saved zero-regularization baseline rollouts."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

import numpy as np

import baseline_atlas_deepc as baseline
import baseline_waypoint_comparison as comparison


GALLERY_SEEDS = [222, 237, 277, 232, 269, 302]
DASHBOARD_SEEDS = [222, 232, 277]


def display_end_step(distance, strict_success):
    """Trim only a successful settled tail, retaining a short arrival view."""
    distance = np.asarray(distance)
    if not strict_success:
        return len(distance)
    outside = np.flatnonzero(distance >= .01)
    arrival = int(outside[-1] + 1) if outside.size else 0
    return min(len(distance), max(60, arrival + 15))


def prepare_media(output_dir=comparison.RESULT_ROOT):
    from render_saved_rollout import render_saved_rollout
    from render_routed_deepc_dashboard import file_sha256, render_dashboard

    output_dir = Path(output_dir)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    for seed in sorted(set(GALLERY_SEEDS + DASHBOARD_SEEDS)):
        for mode in comparison.METHODS:
            trace_path = output_dir / mode / f"seed{seed}_control_results.npz"
            with np.load(trace_path, allow_pickle=True) as trace:
                summary = json.loads(trace["summary"].item())
                assert summary["baseline_version"] == baseline.BASELINE_VERSION
                assert summary["lam_g"] == 0 and summary["reference_mode"] == mode
                success = bool(summary["success_30"] and summary["success_100"])
                end_step = display_end_step(trace["distance"], success)
            digest = file_sha256(trace_path)
            common = dict(source_result_sha256=digest, lam_g=0,
                          reference_mode=mode, baseline_version=baseline.BASELINE_VERSION)
            if seed in GALLERY_SEEDS:
                video = video_dir / f"{mode}_seed{seed}_replay.mp4"
                metadata_path = video.with_suffix(".json")
                old = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
                valid = (video.exists() and all(old.get(k) == v for k, v in common.items())
                         and old.get("maximum_replay_output_error", 1) <= 1e-8)
                if not valid:
                    render_saved_rollout(trace_path, seed, video, fps=20,
                                         frame_stride=4, warmup_zeros=2,
                                         verification_path=metadata_path)
                    metadata = json.loads(metadata_path.read_text())
                    metadata.update(common)
                    metadata_path.write_text(json.dumps(metadata, indent=2))
                print(f"Verified replay: {mode}, seed {seed}", flush=True)
            if seed in DASHBOARD_SEEDS:
                video = video_dir / f"{mode}_seed{seed}_025x.mp4"
                metadata_path = video.with_suffix(".json")
                poster = video.with_name(video.stem + "_poster.png")
                old = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
                valid = (video.exists() and poster.exists()
                         and all(old.get(k) == v for k, v in common.items())
                         and old.get("play_speed") == .25
                         and old.get("displayed_control_steps") == end_step)
                if not valid:
                    render_dashboard(trace_path, baseline.ATLAS_PATH, video,
                                     play_speed=.25, end_step=end_step, hold_seconds=0,
                                     poster_step=min(35, end_step - 1),
                                     evaluation_label="Reacher | lambda_g=0")
                    metadata = json.loads(metadata_path.read_text())
                    metadata.update(common)
                    metadata_path.write_text(json.dumps(metadata, indent=2))
                print(f"Verified 0.25x dashboard: {mode}, seed {seed}", flush=True)


if __name__ == "__main__":
    prepare_media()
