"""Version-checked media for the current baseline's original notebook layout."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

import baseline_atlas_deepc as baseline


GALLERY_SEEDS = [215, 219, 222, 242, 250, 258, 268, 273, 294, 302]
DIAGNOSTIC_VIDEO_SEEDS = [211, 232, 277]
ADDITIONAL_DASHBOARD_SEEDS = [211, 213, 237, 269, 302]
ALL_SEEDS = sorted(set(GALLERY_SEEDS + DIAGNOSTIC_VIDEO_SEEDS + ADDITIONAL_DASHBOARD_SEEDS + [116, 193]))
TRACE_DIR = baseline.RESULT_ROOT / "diagnostics"
VIDEO_DIR = baseline.RESULT_ROOT / "videos"


def prepare_media():
    # The renderer selects the Agg backend; do not change notebook plotting
    # merely by importing the seed lists.
    from render_saved_rollout import render_saved_rollout
    from render_routed_deepc_dashboard import file_sha256, render_dashboard

    payload = baseline.run_panel(ALL_SEEDS, TRACE_DIR)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    for seed in GALLERY_SEEDS + DIAGNOSTIC_VIDEO_SEEDS:
        trace = TRACE_DIR / f"seed{seed}_control_results.npz"
        video = VIDEO_DIR / f"seed{seed}_control.mp4"
        verification = video.with_name(video.stem + "_verification.json")
        digest = file_sha256(trace)
        old = json.loads(verification.read_text()) if verification.exists() else {}
        if not (video.exists() and old.get("source_result_sha256") == digest
                and old.get("maximum_replay_output_error", 1) <= 1e-8):
            render_saved_rollout(trace, seed, video, fps=20, frame_stride=4,
                                 warmup_zeros=2, verification_path=verification)
            metadata = json.loads(verification.read_text())
            metadata.update(source_result_sha256=digest, baseline_version=baseline.BASELINE_VERSION)
            verification.write_text(json.dumps(metadata, indent=2))
        print(f"Verified simulator replay: seed {seed}", flush=True)
    for seed in [222] + ADDITIONAL_DASHBOARD_SEEDS + [232, 277]:
        trace = TRACE_DIR / f"seed{seed}_control_results.npz"
        video = VIDEO_DIR / f"seed{seed}_025x.mp4"
        metadata_path = video.with_suffix(".json")
        old = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        # Existing full-length dashboards remain valid; new gallery dashboards
        # keep the original 60-step view, while diagnostic cases remain uncut.
        if not (video.exists() and old.get("source_result_sha256") == file_sha256(trace)
                and old.get("play_speed") == 0.25):
            render_dashboard(trace, baseline.ATLAS_PATH, video, play_speed=0.25,
                             end_step=None if seed in (232, 277) else 60,
                             hold_seconds=0, evaluation_label="Reacher diagnostic")
        print(f"Verified slowed dashboard: seed {seed}", flush=True)
    return payload


if __name__ == "__main__":
    prepare_media()
