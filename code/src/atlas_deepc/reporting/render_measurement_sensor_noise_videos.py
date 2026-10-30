"""Video settings for measurement_sensor_noise_rank59; rendering is shared."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from atlas_deepc.reporting.noise_videos import NoiseVideoRenderer

renderer = NoiseVideoRenderer(
    "atlas_deepc.noise.measurement_sensor_noise_rank59",
    ((212, 1.0, 5.0), (222, 1.0, 5.0), (237, 2.0, 10.0)), 'measurement_sensor_noise_rank59_full_v1',
    'noise_multiplier', 1.0, 'x', 'relative-amplitude noise',
)
VIDEO_DIR, PAIRS, RENDER_VERSION = renderer.video_dir, renderer.pairs, renderer.render_version
render_one, render_all = renderer.render_one, renderer.render_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    render_all(parser.parse_args().workers)
