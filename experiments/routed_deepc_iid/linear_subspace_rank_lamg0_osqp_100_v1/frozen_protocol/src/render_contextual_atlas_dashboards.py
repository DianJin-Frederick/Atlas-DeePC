"""Render the synchronized dashboards used by the contextual Atlas notebook."""

import argparse
from pathlib import Path

from render_routed_deepc_dashboard import render_dashboard


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "experiments" / "routed_deepc_iid"
EXPERIMENT_DIR = RESULT_ROOT / "contextual_cluster_eval211_710_global128_triggered"
ATLAS_PATH = RESULT_ROOT / "clustering_jointconfig48_fresh211_310" / "behavioral_atlas.npz"
OUTPUT_DIR = RESULT_ROOT / "contextual_cluster_dashboard_videos"

# seed: (label, displayed control steps, poster step)
DASHBOARDS = {
    232: ("rescued", 140, 105),
    215: ("nominal", 60, 35),
    247: ("rescued", 100, 72),
    266: ("rescued", 90, 58),
    404: ("rescued", 100, 68),
    542: ("rescued", 90, 58),
    231: ("failure_unstable", 110, 72),
    277: ("failure_joint_limit", 80, 56),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DASHBOARDS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        if seed not in DASHBOARDS:
            raise ValueError(f"No dashboard specification for seed {seed}")
        label, end_step, poster_step = DASHBOARDS[seed]
        output = OUTPUT_DIR / f"contextual128_seed{seed}_{label}_025x.mp4"
        poster = output.with_name(f"{output.stem}_poster.png")
        metadata = output.with_suffix(".json")
        if not args.force and output.exists() and poster.exists() and metadata.exists():
            print("dashboard exists", output, flush=True)
            continue
        render_dashboard(
            EXPERIMENT_DIR / f"seed{seed}_control_results.npz",
            ATLAS_PATH,
            output,
            fps=20,
            frame_stride=1,
            play_speed=0.25,
            end_step=end_step,
            hold_seconds=1.0,
            poster_step=poster_step,
        )
        print("rendered", output, flush=True)


if __name__ == "__main__":
    main()
