"""Slow-motion galleries rendered from saved Reacher traces, without control solves."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, ScalarFormatter
import numpy as np

from atlas_deepc.common.files import file_digest
from atlas_deepc.common.metrics import strict_success, true_error_metrics
from atlas_deepc.reporting.render_routed_deepc_dashboard import arm_points


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "experiments/reacher/video_gallery_v1"
GUIDED = ROOT / "experiments/reacher/matched_guidance_500_p8_k64_r53_t300_v1/output8/graph"
UNGUIDED = ROOT / "experiments/reacher/four_baselines_500_mixed_k_t300_v1/output8/terminal"
DEMO_SEEDS = (10339, 10258, 10215, 10143)
REPAIR_SEEDS = (10206, 10349, 10445)
COLORS = dict(actual="#20252a", prediction="#2879b5", reference="#e47a12",
              target="#b82936", trail="#359b89", goal="#87649c")


def load_trace(directory, seed):
    path = Path(directory) / f"seed{seed}_control_results.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Connect the experiment SSD; missing saved trace: {path}")
    with np.load(path, allow_pickle=True) as saved:
        names = ("y", "u", "pred", "ref", "distance", "chart", "next_chart",
                 "goal_chart", "final_goal", "selection_windows")
        trace = {name: saved[name].copy() for name in names}
        trace["summary"] = json.loads(saved["summary"].item())
    summary = trace["summary"]
    if (summary["seed"], summary["output_dimension"], summary["clusters"],
            summary["svd_rank"], summary["steps"]) != (seed, 8, 64, 53, 300):
        raise ValueError("This gallery requires the saved p=8, K=64, rank=53, 300-step study")
    trace["initial"] = trace["selection_windows"][0, -8:]
    trace["target"] = trace["final_goal"].reshape(-1)[6:8]
    for name in ("y", "u", "ref", "distance"):
        if not np.isfinite(trace[name]).all():
            raise ValueError(f"Nonfinite {name}: {path}")
    np.testing.assert_allclose(trace["distance"], np.linalg.norm(
        trace["y"][:, 6:8] - trace["target"], axis=1), atol=1e-10)
    trace["metrics"] = true_error_metrics(trace["distance"])
    trace["success"] = strict_success(trace["metrics"])
    if trace["success"] != strict_success(summary):
        raise ValueError(f"Saved outcome differs from recomputed outcome: {path}")
    trace["manifest"] = json.loads((Path(directory) / "manifest.json").read_text())
    trace["path"] = path
    return trace


def verify_pair(unguided, guided):
    np.testing.assert_array_equal(unguided["selection_windows"][0], guided["selection_windows"][0])
    np.testing.assert_array_equal(unguided["target"], guided["target"])
    for key in ("atlas_sha256", "data_sha256", "config", "warmup_steps", "solver",
                "solver_options", "chart_selection_method"):
        if unguided["manifest"][key] != guided["manifest"][key]:
            raise ValueError(f"Comparison is not matched: {key}")
    if unguided["summary"]["reference_mode"] != "terminal" or guided["summary"]["reference_mode"] != "graph":
        raise ValueError("Expected unguided then guided traces")


def control_interval():
    import gymnasium as gym
    from atlas_deepc.control.expert_pipeline_search import REACHER_ENV_ID

    env = gym.make(REACHER_ENV_ID)
    try:
        return float(env.unwrapped.dt)
    finally:
        env.close()


def settled_display_steps(traces, tolerance_m=.001, padding_steps=10):
    """Keep all visible motion and chart switches, plus a short settled tail."""
    last_active = 1
    for trace in traces:
        active = np.zeros(len(trace["y"]), dtype=bool)
        for key in ("y", "ref", "pred"):
            poses = np.asarray([arm_points(state) for state in trace[key]])
            if not np.isfinite(poses).all():
                return len(trace["y"])
            active |= np.max(np.linalg.norm(poses - poses[-1], axis=2), axis=1) > tolerance_m
        for key in ("chart", "next_chart", "goal_chart"):
            active |= trace[key] != trace[key][-1]
        indices = np.flatnonzero(active)
        if indices.size:
            last_active = max(last_active, int(indices[-1]) + 2)
    return min(len(traces[0]["y"]), int(np.ceil((last_active + padding_steps) / 10) * 10))


def _panel(fig, axes, trace, dt, error_max, duration):
    arm, charts, errors = axes
    guided = trace["summary"]["reference_mode"] == "graph"
    outcome = "SUCCESS" if trace["success"] else "FAILURE"
    heading = "With graph guidance" if guided else "Without graph guidance"
    mean_error = 1000 * trace["metrics"]["mean_last_100"]
    arm.set_title(f"{heading} | {outcome}\nFinal-100 mean error: {mean_error:.2f} mm",
                  fontsize=13, pad=10)
    arm.set(xlim=(-23, 23), ylim=(-23, 23), xlabel="Fingertip x (cm)", ylabel="Fingertip y (cm)")
    arm.set_aspect("equal")
    arm.add_patch(plt.Circle((0, 0), 21, fill=False, linestyle=":", color="#ccd2d7", lw=1))
    arm.scatter(*[100 * v for v in trace["target"]], marker="x", s=115,
                color=COLORS["target"], linewidth=2.5, zorder=8)
    initial_arm = 100 * arm_points(trace["initial"])
    arm.plot(initial_arm[:, 0], initial_arm[:, 1], ":o", lw=1.4, ms=4, color="#a4aaaf", zorder=2)
    arm.scatter([0], [0], color=COLORS["actual"], s=55, zorder=8)
    actual, = arm.plot([], [], "-o", color=COLORS["actual"], lw=4, ms=7,
                       markerfacecolor="white", markeredgewidth=2, zorder=7, animated=True)
    prediction, = arm.plot([], [], "--o", color=COLORS["prediction"], lw=2, ms=4,
                           alpha=.85, zorder=6, animated=True)
    reference, = arm.plot([], [], "-o", color=COLORS["reference"], lw=3, ms=5,
                          alpha=.7, zorder=5, animated=True)
    trail, = arm.plot([], [], color=COLORS["trail"], lw=2, alpha=.8, zorder=3, animated=True)
    reference_trail, = arm.plot([], [], ":", color=COLORS["reference"], lw=1.3,
                                alpha=.6, zorder=2, animated=True)
    for ax in axes:
        ax.grid(True, color="#e2e6e8", lw=.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#a4aaaf")
        ax.spines["bottom"].set_color("#a4aaaf")
    charts.set(xlim=(0, duration), ylim=(-2, 66), ylabel="Chart ID")
    charts.set_yticks((0, 16, 32, 48, 63))
    current_text = charts.set_title("", fontsize=10, loc="left", pad=9)
    current_text.set_animated(True)
    selected, = charts.step([], [], where="post", color=COLORS["trail"], lw=1.8,
                            label="Selected", animated=True)
    next_chart, = charts.step([], [], where="post", color=COLORS["reference"], lw=1.3,
                              linestyle="--", label="Next waypoint", animated=True)
    goal, = charts.step([], [], where="post", color=COLORS["goal"], lw=1.2,
                        linestyle=":", label="Goal", animated=True)
    next_chart.set_visible(guided)
    handles = [selected, next_chart, goal] if guided else [selected, goal]
    chart_legend = charts.legend(handles=handles, loc="upper right", ncol=len(handles), fontsize=8,
                                 framealpha=.95, borderpad=.3)
    chart_legend.set_animated(True)
    charts.tick_params(labelbottom=False)
    errors.set(xlim=(0, duration), ylim=(0, error_max), xlabel="Simulation time (s)", ylabel="Error (mm)")
    errors.set_yscale("symlog", linthresh=10, linscale=1)
    ticks = [v for v in (0, 10, 50, 100, 200, 400) if v <= error_max]
    errors.yaxis.set_major_locator(FixedLocator(ticks))
    errors.yaxis.set_major_formatter(ScalarFormatter())
    errors.axhline(10, color="#adb5ba", linestyle=":", lw=.9)
    target_error, = errors.plot([], [], color=COLORS["target"], lw=2, label="Target", animated=True)
    ref_error, = errors.plot([], [], color=COLORS["reference"], lw=1.3,
                             linestyle="--", label="Active reference", animated=True)
    pred_error, = errors.plot([], [], color=COLORS["prediction"], lw=1.2,
                              linestyle=":", label="Prediction", animated=True)
    error_legend = errors.legend(loc="upper right", ncol=3, fontsize=8, framealpha=.95, borderpad=.3)
    error_legend.set_animated(True)
    errors.set_title("Tracking and one-step prediction errors", fontsize=11, loc="left", pad=6)
    chart_cursor = charts.axvline(0, color="#90979c", lw=.9, animated=True)
    error_cursor = errors.axvline(0, color="#90979c", lw=.9, animated=True)
    poses = np.vstack([trace["initial"], trace["y"]])
    distances = 1000 * np.linalg.norm(poses[:, 6:8] - trace["target"], axis=1)
    reference_errors = 1000 * np.linalg.norm(trace["y"][:, 6:8] - trace["ref"][:, 6:8], axis=1)
    prediction_errors = 1000 * np.linalg.norm(trace["y"][:, 6:8] - trace["pred"][:, 6:8], axis=1)
    sample_times = dt * np.arange(len(poses))
    artists = [trail, reference_trail, reference, prediction, actual, current_text,
               selected, next_chart, goal, target_error, ref_error, pred_error, chart_cursor, error_cursor,
               chart_legend, error_legend]

    def update(time_s):
        position = np.clip(time_s / dt, 0, len(poses) - 1)
        lower = int(np.floor(position))
        upper = min(lower + 1, len(poses) - 1)
        fraction = position - lower
        state = (1 - fraction) * poses[lower] + fraction * poses[upper]
        action = min(lower, len(trace["y"]) - 1)
        for line, value in ((actual, state), (prediction, trace["pred"][action]),
                            (reference, trace["ref"][action])):
            points = 100 * arm_points(value)
            line.set_data(points[:, 0], points[:, 1])
        xy = 100 * np.vstack([poses[:lower + 1, 6:8], state[6:8]])
        trail.set_data(xy[:, 0], xy[:, 1])
        ref_xy = 100 * trace["ref"][:action + 1, 6:8]
        reference_trail.set_data(ref_xy[:, 0], ref_xy[:, 1])
        chart = int(trace["chart"][action])
        requested = int(trace["next_chart"][action])
        goal_id = int(trace["goal_chart"][action])
        waypoint = np.linalg.norm(trace["ref"][action, 6:8] - trace["target"]) > 1e-9
        mode = (f"Waypoint chart {requested}" if waypoint else "Terminal reference") if guided else "Fixed terminal reference"
        current_error = 1000 * np.linalg.norm(state[6:8] - trace["target"])
        current_text.set_text(f"Selected chart {chart} | goal {goal_id} | {mode}\nTarget error {current_error:.2f} mm")
        chart_times = np.r_[dt * np.arange(action + 1), time_s]
        for line, values in ((selected, trace["chart"]), (next_chart, trace["next_chart"]), (goal, trace["goal_chart"])):
            line.set_data(chart_times, np.r_[values[:action + 1], values[action]])
        target_error.set_data(np.r_[sample_times[:lower + 1], time_s],
                              np.r_[distances[:lower + 1], current_error])
        ref_error.set_data(sample_times[1:lower + 1], reference_errors[:lower])
        pred_error.set_data(sample_times[1:lower + 1], prediction_errors[:lower])
        chart_cursor.set_xdata([time_s, time_s])
        error_cursor.set_xdata([time_s, time_s])
        return artists

    return update, artists


def render_video(traces, output, *, fps=20, speed=.2, hold_seconds=1.0, preview_only=False):
    if not 0 < speed <= 1 or fps < 1:
        raise ValueError("Use 0 < speed <= 1 and positive fps")
    if len(traces) not in (1, 2):
        raise ValueError("Expected a single example or matched pair")
    if len(traces) == 2:
        verify_pair(*traces)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dt = control_interval()
    recorded_steps = len(traces[0]["y"])
    displayed_steps = settled_display_steps(traces)
    duration = displayed_steps * dt
    seed = traces[0]["summary"]["seed"]
    paired = len(traces) == 2
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig = plt.figure(figsize=(16, 10.8) if paired else (12.8, 7.2), dpi=100)
    if paired:
        grid = fig.add_gridspec(3, 2, height_ratios=(4.5, .9, 1.4), hspace=.45, wspace=.22,
                               left=.065, right=.975, bottom=.085, top=.765)
        axes = [(fig.add_subplot(grid[0, i]), fig.add_subplot(grid[1, i]), fig.add_subplot(grid[2, i])) for i in range(2)]
    else:
        grid = fig.add_gridspec(2, 2, width_ratios=(1.15, 1), hspace=.45, wspace=.23,
                               left=.065, right=.975, bottom=.11, top=.71)
        axes = [(fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1]))]
    title = f"Atlas-DeePC | {'Graph-guidance repair' if paired else 'Target reaching'} | seed {seed}"
    fig.suptitle(title, fontsize=19, fontweight="bold", y=.98)
    fig.text(.5, .925 if paired else .91, f"p = 8   |   K = 64   |   rank = 53   |   {1/speed:g}x slower ({speed:g}x speed)",
             ha="center", fontsize=12, color="#4b5359")
    legend = [Line2D([], [], color=COLORS["actual"], lw=3, label="Actual arm"),
              Line2D([], [], color=COLORS["prediction"], lw=2, ls="--", label="One-step predicted pose"),
              Line2D([], [], color=COLORS["reference"], lw=3, label="Active reference pose"),
              Line2D([], [], color=COLORS["trail"], lw=2, label="Fingertip path"),
              Line2D([], [], color=COLORS["target"], marker="x", ls="", label="Target")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .90 if paired else .885), ncol=5,
               frameon=False, fontsize=10)
    clock = fig.text(.5, .845 if paired else .815, "", ha="center", fontsize=11, animated=True)
    if paired:
        fig.text(.5, .016, "Matched atlas, rank, initial history and target. Guidance includes waypoints + iterative terminal-reference selection.",
                 ha="center", fontsize=9, color="#4b5359")
    else:
        fig.text(.5, .025, "Settled tail omitted | interpolated display | metrics use all 300 recorded steps", ha="center", fontsize=9, color="#4b5359")
    error_max = max(1000 * np.linalg.norm(t["initial"][6:8] - t["target"]) for t in traces)
    for t in traces:
        error_max = max(error_max, 1000 * np.nanmax(t["distance"]),
                        1000 * np.nanmax(np.linalg.norm(t["y"][:, 6:8] - t["ref"][:, 6:8], axis=1)),
                        1000 * np.nanmax(np.linalg.norm(t["y"][:, 6:8] - t["pred"][:, 6:8], axis=1)))
    panels = [_panel(fig, ax, t, dt, max(50, error_max * 1.15), duration) for ax, t in zip(axes, traces)]
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)

    def draw(time_s):
        fig.canvas.restore_region(background)
        clock.set_text(f"Displayed time {time_s:.2f} / {duration:.2f} s    |    step {min(round(time_s/dt), displayed_steps)} / {displayed_steps}    |    {recorded_steps} steps recorded")
        fig.draw_artist(clock)
        for update, _ in panels:
            for artist in update(time_s):
                if artist.get_visible():
                    artist.axes.draw_artist(artist)
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

    try:
        # A moving-state poster shows both overlays and completed tracking history.
        poster_time = min(1.0, duration)
        poster = output.with_name(output.stem + "_poster.png")
        imageio.imwrite(poster, draw(poster_time))
        imageio.imwrite(output.with_name(output.stem + "_final.png"), draw(duration))
        if preview_only:
            return dict(seed=seed, poster=str(poster), preview=True)
        frame_count = int(np.ceil(duration / speed * fps))
        times = np.arange(frame_count) * speed / fps
        pending = output.with_name(output.stem + ".pending.mp4")
        with imageio.get_writer(pending, fps=fps, codec="libx264", quality=8,
                                macro_block_size=2, ffmpeg_params=["-movflags", "+faststart", "-threads", "2",
                                "-metadata", f"title={title}"]) as writer:
            for index, time_s in enumerate(times):
                writer.append_data(draw(float(time_s)))
                if index % 200 == 0:
                    print(f"seed {seed}: frame {index}/{frame_count}", flush=True)
            final = draw(duration)
            for _ in range(round(hold_seconds * fps)):
                writer.append_data(final)
        pending.replace(output)
        sources = [dict(path=str(t["path"].relative_to(ROOT)), sha256=file_digest(t["path"]),
                        reference_mode=t["summary"]["reference_mode"], success=t["success"],
                        final100_mean_mm=1000*t["metrics"]["mean_last_100"]) for t in traces]
        metadata = dict(seed=seed, title=title, video=output.name, poster=poster.name,
                        sources=sources, p=8, K=64, rank=53, steps=300,
                        displayed_control_steps=displayed_steps,
                        recorded_simulated_duration_s=recorded_steps * dt,
                        trim_rule="Omit tail after actual/reference/predicted arm points stay within 1 mm of final values and chart IDs stay fixed; retain 10 extra steps and round up to 10 steps. Pairs use the later cutoff.",
                        control_dt_s=dt, simulated_duration_s=duration, play_speed=speed,
                        fps=fps, motion_frames=frame_count, hold_seconds=hold_seconds,
                        total_frames=frame_count+round(hold_seconds*fps), renderer_sha256=file_digest(__file__),
                        visualization="Actual pose interpolated between saved samples; prediction/reference/chart held per action. No controller rerun.",
                        comparison_scope="Guidance policy package: chordal waypoints and iterative N=10 terminal selection, versus fixed terminal reference.")
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
        return metadata
    finally:
        plt.close(fig)


def gallery_jobs():
    return [(f"target_seed{seed}_5x_slow.mp4", [load_trace(GUIDED, seed)]) for seed in DEMO_SEEDS] + [
        (f"guidance_repair_seed{seed}_5x_slow.mp4", [load_trace(UNGUIDED, seed), load_trace(GUIDED, seed)]) for seed in REPAIR_SEEDS]


def render_gallery(output_dir=DEFAULT_OUTPUT, *, preview_only=False, overwrite=False):
    output_dir = Path(output_dir)
    records = []
    for name, traces in gallery_jobs():
        if len(traces) == 1 and not traces[0]["success"]:
            raise ValueError("Demonstration seed is not successful")
        if len(traces) == 2 and (traces[0]["success"] or not traces[1]["success"]):
            raise ValueError("Comparison seed is not a repair")
        path = output_dir / name
        if path.exists() and path.with_suffix(".json").exists() and not overwrite and not preview_only:
            metadata = json.loads(path.with_suffix(".json").read_text())
            if metadata["renderer_sha256"] != file_digest(__file__):
                raise ValueError("Renderer changed; set overwrite=True to regenerate videos")
            if [s["sha256"] for s in metadata["sources"]] != [file_digest(t["path"]) for t in traces]:
                raise ValueError("Source trace changed; refusing stale video reuse")
            records.append(metadata)
        else:
            records.append(render_video(traces, path, preview_only=preview_only))
        print(f"Ready: {path.name}", flush=True)
    if not preview_only:
        (output_dir / "gallery.json").write_text(json.dumps(records, indent=2)+"\n")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    render_gallery(args.output_dir, preview_only=args.preview_only, overwrite=args.overwrite)
