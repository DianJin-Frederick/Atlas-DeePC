"""Render paired diagnostics, trimming only settled successful rollouts."""

import json
import numpy as np

from small_atlas_ablations import BASIS_DIR, RESULT_ROOT, baseline
from render_routed_deepc_dashboard import file_sha256, render_dashboard


VIDEO_JOBS = [
    (237, "linear_terminal", "No waypoints: large-error failure"),
    (237, "linear_graph", "Baseline: matching success"),
    (306, "linear_terminal", "No waypoints: opposite-side target failure"),
    (306, "linear_graph", "Baseline: matching success"),
    (232, "linear_terminal", "No waypoints: near-target failure"),
    (232, "linear_graph", "Baseline: matching near-target failure"),
    (212, "linear_graph", "Both succeed: lower-left target, with waypoints"),
    (212, "linear_terminal", "Both succeed: lower-left target, without waypoints"),
    (213, "linear_graph", "Both succeed: upper target, with waypoints"),
    (213, "linear_terminal", "Both succeed: upper target, without waypoints"),
    (222, "linear_graph", "Both succeed: near-base target, with waypoints"),
    (222, "linear_terminal", "Both succeed: near-base target, without waypoints"),
]
VIDEO_DIR = RESULT_ROOT / "videos"


def display_stop(source):
    with np.load(source, allow_pickle=True) as trace:
        summary = json.loads(trace["summary"].item())
        y, distance = trace["y"], trace["distance"]
        total = len(y)
        if summary["success_30"] and summary["success_100"]:
            # Inspect the full suffix so later motion or drift is never hidden.
            for step in range(total - 10):
                if (np.all(distance[step:] < 0.01)
                        and np.max(np.linalg.norm(y[step:, 6:8] - y[step, 6:8], axis=1)) < 0.001
                        and np.max(np.linalg.norm(y[step:, :4] - y[step, :4], axis=1)) < 0.01):
                    return step + 1, 1.0
        return total, 0.0


def render_all():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for seed, variant, label in VIDEO_JOBS:
        source = BASIS_DIR / variant / f"seed{seed}_control_results.npz"
        output = VIDEO_DIR / f"seed{seed}_{variant}_025x.mp4"
        metadata_path = output.with_suffix(".json")
        stop, hold = display_stop(source)
        legend_location = "upper left" if seed == 212 else "lower left"
        valid = False
        if output.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            valid = (metadata["source_result_sha256"] == file_sha256(source)
                     and metadata["displayed_control_steps"] == stop
                     and metadata.get("graph_waypoints_used") == (variant == "linear_graph")
                     and metadata.get("arm_legend_location", "lower left") == legend_location)
        if not valid:
            print(f"Rendering seed {seed}: {variant}", flush=True)
            render_dashboard(source, baseline.ATLAS_PATH, output, play_speed=0.25,
                             end_step=stop, hold_seconds=hold, poster_step=min(100, stop-1),
                             evaluation_label="100-seed ablation", arm_legend_location=legend_location)
            metadata = json.loads(metadata_path.read_text())
        metadata["clip_rule"] = "Settled success only: suffix error <1 cm, fingertip displacement <1 mm, cosine/sine posture displacement <0.01; failures full length"
        metadata["end_hold_seconds"] = hold
        metadata_path.write_text(json.dumps(metadata, indent=2))
        entries.append({"seed": seed, "variant": variant, "label": label,
                        "video": str(output.resolve()), "poster": metadata["poster"],
                        "source": str(source.resolve()), "metadata": metadata})
        print(f"Saved {output}", flush=True)
    (VIDEO_DIR / "manifest.json").write_text(json.dumps(entries, indent=2))
    return entries


if __name__ == "__main__":
    render_all()
