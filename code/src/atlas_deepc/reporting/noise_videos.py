"""Shared noisy-rollout video renderer; study entry points supply only settings."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from importlib import import_module
import json
from pathlib import Path
import numpy as np
from atlas_deepc.common.paths import source_path
from atlas_deepc.reporting.render_routed_deepc_dashboard import file_sha256, render_dashboard


@dataclass(frozen=True)
class NoiseVideoRenderer:
    experiment_module: str
    pairs: tuple
    render_version: str
    ratio_key: str = "noise_sd_ratio"
    display_scale: float = 100.
    display_unit: str = "%"
    noise_label: str = "relative-amplitude noise"

    @property
    def video_dir(self):
        return import_module(self.experiment_module).RESULT_ROOT / "videos"

    def render_one(self, job):
        experiment = import_module(self.experiment_module)
        seed, ratio = job
        folder = experiment.RESULT_ROOT / experiment.level_name(ratio)
        source = folder / f"seed{seed}_control_results.npz"
        atlas = folder / "behavioral_atlas.npz"
        output = self.video_dir / f"seed{seed}_{experiment.level_name(ratio)}_025x.mp4"
        metadata_path = output.with_suffix(".json")
        with np.load(source, allow_pickle=True) as trace:
            row = json.loads(trace["summary"].item())
            assert row["lam_g"] == .05 and row["svd_rank"] == 59 and row[self.ratio_key] == ratio
            assert row["evaluation_uses_true_outputs"] and len(trace["distance"]) == 320
            limit = max(30, float(10*np.ceil(1.15*np.quantile(1000*trace["distance"][-100:], .95)/10)))
        signature = {
            "render_version": self.render_version,
            "noise_video_helper_sha256": file_sha256(Path(__file__)),
            "renderer_sha256": file_sha256(source_path("render_routed_deepc_dashboard.py")),
            "source_result_sha256": file_sha256(source), "source_atlas_sha256": file_sha256(atlas),
            "displayed_control_steps": 320, "error_panel_limit_mm": limit,
            "fps": 10, "frame_stride": 2, "play_speed": .25,
        }
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        if not output.exists() or any(metadata.get(k) != v for k, v in signature.items()):
            print(f"Rendering seed {seed}, noise SD {self.display_scale*ratio:g}{self.display_unit}", flush=True)
            render_dashboard(source, atlas, output, fps=10, frame_stride=2,
                             play_speed=.25, end_step=320, hold_seconds=1, poster_step=240,
                             evaluation_label=f"{self.display_scale*ratio:g}{self.display_unit} {self.noise_label} | rank 59 | lambda_g=0.05",
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
        return {"seed": seed, self.ratio_key: ratio, "video": str(output.resolve()),
                "poster": metadata["poster"], "metadata": metadata,
                "strict_success": bool(row["success_30"] and row["success_100"]),
                "mean_last_100_mm": 1000*row["mean_last_100"]}

    def render_all(self, workers=2):
        self.video_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(seed, ratio) for seed, low, high in self.pairs for ratio in (low, high)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            entries = list(pool.map(self.render_one, jobs))
        (self.video_dir / "manifest.json").write_text(json.dumps(entries, indent=2))
        return entries
