"""Render a saved F1TENTH lap; never run or modify the control experiment."""

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import yaml

from atlas_deepc.common.paths import ROOT

DEFAULT_RUN = (ROOT / "experiments/f1tenth/paper_nsr_20261027/sigma_0p001"
               / "atlas_k12_d40/seed542")
DEFAULT_VIDEO = ROOT / "experiments/f1tenth/video_gallery_v1/atlas_seed542_lap.mp4"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_lap(run_dir=DEFAULT_RUN, output=DEFAULT_VIDEO, *, fps=30,
               playback_speed=1.0, overwrite=False):
    """Animate true saved poses, including warmup, with display-only interpolation.

    Timing and errors are taken from recorded controller samples. The car outline
    is the simulator's 0.58 x 0.31 m footprint; the overview dot is a locator only.
    A manifest records source hashes and allows safe reuse of an existing render.
    """
    run_dir, output = Path(run_dir), Path(output)
    if fps <= 0 or playback_speed <= 0:
        raise ValueError("fps and playback_speed must be positive")
    files = {name: run_dir / name for name in
             ("rollout.npz", "summary.json", "collision_replay.json", "calls.json")}
    files["protocol.json"] = run_dir.parent.parent / "protocol.json"
    files["map.png"] = ROOT / "data/f1tenth/maps/example_map.png"
    files["map.yaml"] = ROOT / "data/f1tenth/maps/example_map.yaml"
    files["waypoints.csv"] = ROOT / "data/f1tenth/example_waypoints.csv"
    files["renderer.py"] = Path(__file__)
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing saved lap/asset: {path}. Connect the experiment SSD.")
    signature = dict(source=str(run_dir.relative_to(ROOT)),
                     hashes={k: digest(v) for k, v in files.items()},
                     fps=fps, playback_speed=playback_speed)
    manifest_file = output.with_suffix(".json")
    if output.is_file() and manifest_file.is_file() and not overwrite:
        previous = json.loads(manifest_file.read_text())
        if previous.get("signature") == signature and previous.get("video_sha256") == digest(output):
            return output

    summary = json.loads(files["summary.json"].read_text())
    replay = json.loads(files["collision_replay.json"].read_text())
    protocol = json.loads(files["protocol.json"].read_text())
    calls = json.loads(files["calls.json"].read_text())
    if not (summary["verified_success"] and summary["lap_completed"] and
            replay["verified"] and replay["collision_free"] and replay["maximum_state_mismatch"] < 1e-9):
        raise ValueError("Choose a saved, replay-verified collision-free completed lap")
    with np.load(files["rollout.npz"]) as saved:
        trace = {k: saved[k].copy() for k in
                 ("X", "warmup_states", "final_state", "reference", "cross_track_error")}
    n, nw = len(trace["X"]), len(trace["warmup_states"])
    dt = float(protocol["dt"])
    assert n == summary["steps"] and nw == summary["warmup_steps"]
    assert trace["reference"].shape[0] == n and len(calls) >= n
    np.testing.assert_allclose(100 * np.sqrt(np.mean(trace["cross_track_error"]**2)),
                               summary["rmse_cm"], rtol=1e-10)
    states = np.vstack((trace["warmup_states"], trace["X"], trace["final_state"]))
    if not np.isfinite(states).all():
        raise ValueError("Nonfinite saved poses")
    yaw = np.unwrap(states[:, 4])
    elapsed = np.arange(len(states)) * dt
    sample_t = (nw + np.arange(n)) * dt
    duration = elapsed[-1]
    sigma = float(summary["noise_sigma"])
    config = summary["config"]
    method = (f"Atlas-DeePC | K={config['clusters']}, d={config['rank']}"
              if "clusters" in config else f"Contextual-DPC | {config['columns']} columns")
    map_cfg = yaml.safe_load(files["map.yaml"].read_text())
    if not np.isclose(map_cfg["origin"][2], 0):
        raise ValueError("This renderer expects an axis-aligned map")
    pixels = imageio.imread(files["map.png"])
    x0, y0 = map_cfg["origin"][:2]
    resolution = map_cfg["resolution"]
    extent = (x0, x0 + pixels.shape[1]*resolution, y0, y0 + pixels.shape[0]*resolution)
    waypoints = np.loadtxt(files["waypoints.csv"], delimiter=";", comments="#")
    track = np.vstack((waypoints[:, 1:3], waypoints[:1, 1:3]))
    blue, orange = "#31659b", "#df8123"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig = plt.figure(figsize=(12.8, 8), dpi=100, facecolor="white")
    grid = fig.add_gridspec(3, 2, height_ratios=(3.5, 1, 1), width_ratios=(1.25, 1),
                            left=.065, right=.975, bottom=.065, top=.85, hspace=.6, wspace=.22)
    overview = fig.add_subplot(grid[:, 0])
    zoom = fig.add_subplot(grid[0, 1])
    speed = fig.add_subplot(grid[1, 1])
    error = fig.add_subplot(grid[2, 1])
    fig.suptitle(f"{method} | seed {summary['seed']}\n"
                 f"Collision-free full lap | cross-track RMSE {summary['rmse_cm']:.2f} cm | "
                 f"noise amplitude σ={sigma:g}", fontsize=15, y=.98)
    clock = fig.text(.5, .895, "", ha="center", fontsize=11)
    fig.text(.5, .012, f"Saved-state replay · {playback_speed:g}× playback · pose interpolation for display only",
             ha="center", color="#687580", fontsize=9)
    trails, bodies = [], []
    footprint = np.array([[.29, .155], [.29, -.155], [-.29, -.155], [-.29, .155]])
    for ax in (overview, zoom):
        ax.imshow(pixels, extent=extent, origin="upper", cmap="gray", vmin=0, vmax=255,
                  interpolation="nearest", zorder=0)
        ax.plot(*track.T, "--", color=orange, lw=1.1, alpha=.85, label="Official reference")
        trail, = ax.plot([], [], color=blue, lw=1.8, label="Closed-loop trajectory")
        trails.append(trail)
        body = Polygon(footprint, closed=True, color=blue, ec="white", lw=.7, zorder=6)
        ax.add_patch(body)
        nose, = ax.plot([], [], color="white", lw=1.5, zorder=7)
        bodies.append((body, nose))
        ax.set_aspect("equal")
        ax.set(xlabel="x (m)", ylabel="y (m)")
    overview.set(xlim=(track[:, 0].min()-3, track[:, 0].max()+3),
                 ylim=(track[:, 1].min()-3, track[:, 1].max()+3), title="Complete track")
    overview.plot(*states[0, :2], "x", color="#59636b", ms=7, label="Start")
    marker, = overview.plot([], [], "o", ms=7, mfc="none", mec=blue, mew=1.5, zorder=8)
    overview.legend(loc="upper right", fontsize=9, framealpha=.95)
    zoom.set_title("Car footprint at true scale")
    horizon, = zoom.plot([], [], ":o", color=orange, lw=1.4, ms=2.5)
    speed_actual, = speed.plot([], [], color=blue, lw=1.8, label="True speed")
    speed_ref, = speed.plot([], [], "--", color=orange, lw=1.4, label="Active reference")
    speed.legend(loc="upper right", fontsize=8, ncol=2)
    all_speeds = np.r_[states[:, 3], trace["reference"][:, 0, 2]]
    speed.set(ylabel="Speed (m/s)", ylim=(all_speeds.min()-.3, all_speeds.max()+.3))
    signed_error = 100 * trace["cross_track_error"]
    error_line, = error.plot([], [], color=blue, lw=1.6)
    error.axhline(0, color="#9ba4ab", lw=.7)
    limit = max(5, 1.15 * np.max(np.abs(signed_error)))
    error.set(xlabel="Elapsed time including warmup (s)", ylabel="Track error (cm)", ylim=(-limit, limit))
    for ax in (speed, error):
        ax.set_xlim(0, duration)
        ax.axvspan(0, nw*dt, color="#e9edf0", zorder=0)
        ax.grid(alpha=.2)
    cursors = [ax.axvline(0, color="#606c76", lw=.8) for ax in (speed, error)]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".rendering.mp4")
    frame_count = int(np.ceil(duration / playback_speed * fps)) + 1
    hold_frames = fps
    try:
        with imageio.get_writer(temporary, fps=fps, codec="libx264", quality=8,
                                pixelformat="yuv420p", macro_block_size=1,
                                ffmpeg_params=["-movflags", "+faststart"]) as writer:
            for frame in range(frame_count + hold_frames):
                t = min(frame / fps * playback_speed, duration)
                index = min(int(t / dt), len(states)-2)
                fraction = np.clip((t-elapsed[index])/dt, 0, 1)
                pose = (1-fraction)*states[index] + fraction*states[index+1]
                angle = (1-fraction)*yaw[index] + fraction*yaw[index+1]
                rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                     [np.sin(angle), np.cos(angle)]])
                path = np.vstack((states[:index+1, :2], pose[:2]))
                for line, (body, nose) in zip(trails, bodies):
                    line.set_data(*path.T)
                    body.set_xy(footprint @ rotation.T + pose[:2])
                    nose.set_data(*(np.array([[0, 0], [.29, 0]]) @ rotation.T + pose[:2]).T)
                marker.set_data([pose[0]], [pose[1]])
                zoom.set_xlim(pose[0]-3.5, pose[0]+3.5)
                zoom.set_ylim(pose[1]-2.5, pose[1]+2.5)
                control_index = min(index-nw, n-1)
                count = max(0, control_index+1)
                if count:
                    horizon.set_data(*trace["reference"][control_index, :, :2].T)
                    phase = f"Controller | chart {calls[control_index].get('chart', '—')}"
                else:
                    horizon.set_data([], [])
                    phase = "Shared initialization (warmup)"
                speed_actual.set_data(elapsed[:index+1], states[:index+1, 3])
                speed_ref.set_data(sample_t[:count], trace["reference"][:count, 0, 2])
                error_line.set_data(sample_t[:count], signed_error[:count])
                for cursor in cursors:
                    cursor.set_xdata([t, t])
                clock.set_text(f"{t:4.1f} / {duration:.1f} s  |  {pose[3]:.2f} m/s  |  "
                               + ("Full lap completed — collision-free" if t >= duration else phase))
                fig.canvas.draw()
                pixels_rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
                writer.append_data(pixels_rgb)
                if frame == frame_count//2:
                    imageio.imwrite(output.with_suffix(".png"), pixels_rgb)
                if frame % (fps*5) == 0:
                    print(f"Rendering {frame}/{frame_count+hold_frames} frames", flush=True)
        temporary.replace(output)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)
    manifest = dict(signature=signature, seed=summary["seed"], config=config,
                    noise_sigma=sigma, noise_amplitude_percent=100*sigma,
                    noise_power_percent=100*sigma*sigma,
                    verified_success=True, rmse_cm=summary["rmse_cm"],
                    controller_steps=n, warmup_steps=nw, simulation_duration_s=duration,
                    frames=frame_count+hold_frames, video_duration_s=(frame_count+hold_frames)/fps,
                    video_sha256=digest(output), collision_replay=replay,
                    note="True states, interpolated for display; no simulation or controller rerun.")
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    return output
