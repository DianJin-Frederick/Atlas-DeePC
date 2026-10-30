import argparse
import json
from pathlib import Path

from render_saved_rollout import render_saved_rollout


DEFAULT_EXPERIMENT_DIRS = (
    Path("experiments/routed_deepc_iid/clustering_jointconfig48_fresh211_310"),
)


def verified_video_exists(result_path, video_path, verification_path):
    if not video_path.exists() or not verification_path.exists():
        return False
    verification = json.loads(verification_path.read_text())
    return (
        verification.get("result_path") == str(result_path)
        and verification.get("video_path") == str(video_path)
        and verification.get("maximum_replay_output_error") == 0.0
    )


def render_experiment_videos(experiment_dirs, seeds=None, frame_stride=4, force=False):
    selected_seeds = None if seeds is None else set(seeds)
    for experiment_dir in experiment_dirs:
        summary = json.loads((experiment_dir / "summary.json").read_text())
        for result in summary["results"]:
            seed = int(result["seed"])
            if selected_seeds is not None and seed not in selected_seeds:
                continue
            result_path = experiment_dir / f"seed{seed}_control_results.npz"
            video_path = experiment_dir / f"seed{seed}_control.mp4"
            verification_path = experiment_dir / f"seed{seed}_control_video_verification.json"
            if not force and verified_video_exists(result_path, video_path, verification_path):
                print("verified video exists", video_path, flush=True)
                continue
            error = render_saved_rollout(
                result_path,
                seed,
                video_path,
                fps=20,
                frame_stride=frame_stride,
                warmup_zeros=2,
                verification_path=verification_path,
            )
            print("rendered", video_path, "maximum replay error", error, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Render selected frozen routed-DeePC rollouts")
    parser.add_argument(
        "experiment_dirs",
        type=Path,
        nargs="*",
        default=DEFAULT_EXPERIMENT_DIRS,
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render_experiment_videos(
        args.experiment_dirs,
        seeds=args.seeds,
        frame_stride=args.frame_stride,
        force=args.force,
    )
