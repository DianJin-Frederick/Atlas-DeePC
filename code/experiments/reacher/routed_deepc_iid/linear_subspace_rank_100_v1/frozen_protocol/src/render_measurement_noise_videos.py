"""Render paired, full-length diagnostics from the saved lambda_g=0 noise study."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

import measurement_noise_ablation as experiment
from render_routed_deepc_dashboard import file_sha256, render_dashboard


VIDEO_DIR = experiment.RESULT_ROOT / "videos"
PAIRS = [(212, .01, .05), (222, .01, .05), (237, .02, .10)]
RENDER_VERSION = "noise_true_sensor_full_v1"


def render_one(job):
    seed, ratio = job
    folder = experiment.RESULT_ROOT / experiment.level_name(ratio)
    source = folder / f"seed{seed}_control_results.npz"
    atlas = folder / "behavioral_atlas.npz"
    output = VIDEO_DIR / f"seed{seed}_{experiment.level_name(ratio)}_025x.mp4"
    metadata_path = output.with_suffix(".json")
    with np.load(source, allow_pickle=True) as trace:
        row = json.loads(trace["summary"].item())
        assert row["lam_g"] == 0 and row["noise_sd_ratio"] == ratio
        assert row["evaluation_uses_true_outputs"] and len(trace["distance"]) == 320
        limit = max(30, float(10*np.ceil(1.15*np.quantile(1000*trace["distance"][-100:], .95)/10)))
    signature = {
        "render_version": RENDER_VERSION,
        "renderer_sha256": file_sha256(Path(__file__).with_name("render_routed_deepc_dashboard.py")),
        "source_result_sha256": file_sha256(source), "source_atlas_sha256": file_sha256(atlas),
        "displayed_control_steps": 320, "error_panel_limit_mm": limit,
        "fps": 10, "frame_stride": 2, "play_speed": .25,
    }
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if not output.exists() or any(metadata.get(k) != v for k, v in signature.items()):
        print(f"Rendering seed {seed}, noise SD {100*ratio:g}%", flush=True)
        render_dashboard(source, atlas, output, fps=10, frame_stride=2,
                         play_speed=.25, end_step=320, hold_seconds=1, poster_step=240,
                         evaluation_label=f"{100*ratio:g}% SD noise | lambda_g=0",
                         arm_legend_location="upper left" if seed == 212 else "lower left",
                         error_limit_mm=limit)
        metadata = json.loads(metadata_path.read_text())
        metadata.update(signature)
        metadata["clip_rule"] = "All 320 saved steps; no trimming after first target contact"
        metadata["noise_display"] = "True arm and true error; separate noisy fingertip and noisy measured error"
        metadata["video_sha256"] = file_sha256(output)
        metadata["poster_sha256"] = file_sha256(Path(metadata["poster"]))
        metadata_path.write_text(json.dumps(metadata, indent=2))
    metadata["playback_note"] = "0.25x relative to the existing 20-fps, one-sample-per-frame dashboard convention; not a claim about simulator real-time speed"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved {output.name}", flush=True)
    return {"seed": seed, "noise_sd_ratio": ratio, "video": str(output.resolve()),
            "poster": metadata["poster"], "metadata": metadata,
            "strict_success": bool(row["success_30"] and row["success_100"]),
            "mean_last_100_mm": 1000*row["mean_last_100"]}


def render_all(workers=2):
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(seed, ratio) for seed, low, high in PAIRS for ratio in (low, high)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        entries = list(pool.map(render_one, jobs))
    (VIDEO_DIR / "manifest.json").write_text(json.dumps(entries, indent=2))
    return entries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    render_all(parser.parse_args().workers)
