import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D

from routed_deepc_experiment import load_atlas, shortest_chart_path


LINK_1 = 0.10
WORKSPACE_RADIUS = 0.21


def arm_points(state):
    state = np.asarray(state, dtype=float)
    elbow = LINK_1 * np.array([state[0], state[2]])
    fingertip = state[6:8]
    return np.vstack([np.zeros(2), elbow, fingertip])


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_dashboard(result_path, atlas_path, output_path, fps=20, frame_stride=1,
                     play_speed=1.0, end_step=None, hold_seconds=1.5, poster_step=12,
                     evaluation_label="Held-out Reacher", arm_legend_location="lower left",
                     error_limit_mm=None):
    result_path = Path(result_path)
    atlas_path = Path(atlas_path)
    output_path = Path(output_path)
    poster_path = output_path.with_name(f"{output_path.stem}_poster.png")
    metadata_path = output_path.with_suffix(".json")

    result = np.load(result_path, allow_pickle=True)
    atlas = load_atlas(atlas_path)
    summary = json.loads(result["summary"].item())
    noisy_run = bool(summary.get("offline_and_online_noise", False))
    terminal_only = summary.get("reference_mode") == "terminal"
    total_saved_steps = int(result["y"].shape[0])
    target = np.asarray(result["final_goal"], dtype=float)[6:8].copy() if "final_goal" in result.files else np.asarray(result["ref"], dtype=float)[-1, 6:8].copy()
    stop = total_saved_steps if end_step is None else min(int(end_step), total_saved_steps)
    if stop < 2:
        raise ValueError("end_step must retain at least two control steps.")
    selected = slice(0, stop)
    y = np.asarray(result["y"], dtype=float)[selected]
    u = np.asarray(result["u"], dtype=float)[selected]
    prediction = np.asarray(result["pred"], dtype=float)[selected]
    reference = np.asarray(result["ref"], dtype=float)[selected]
    distance = np.asarray(result["distance"], dtype=float)[selected]
    sensed_y = np.asarray(result["y_measured"], dtype=float)[selected] if noisy_run else None
    sensed_distance = np.asarray(result["distance_measured"], dtype=float)[selected] if noisy_run else None
    charts = np.asarray(result["chart"], dtype=int)[selected]
    requested = np.asarray(result["next_chart"], dtype=int)[selected]
    goal_charts = np.asarray(result["goal_chart"], dtype=int)[selected]
    statuses = np.asarray(result["status"], dtype=str)[selected]
    matrix_columns = np.asarray(result["matrix_columns"], dtype=int)[selected]
    contextual_run = "contextual_active" in result.files
    contextual_active = (
        np.asarray(result["contextual_active"], dtype=bool)[selected]
        if contextual_run else np.zeros(y.shape[0], dtype=bool)
    )
    contextual_clusters = (
        np.asarray(result["contextual_cluster"], dtype=int)[selected]
        if contextual_run else np.full(y.shape[0], -1, dtype=int)
    )
    trigger_locations = np.flatnonzero(
        np.asarray(result["contextual_triggered"], dtype=bool)[selected]
    ) if contextual_run else np.array([], dtype=int)
    trigger_step = int(trigger_locations[0]) if trigger_locations.size else -1
    measured_next = np.r_[charts[1:], -1]
    if stop < total_saved_steps:
        measured_next[-1] = int(result["chart"][stop])
    steps = np.arange(y.shape[0])

    route_cache = {}
    dynamic_goal = bool(summary.get("dynamic_terminal_reference", False))
    if dynamic_goal and not terminal_only:
        from candidate_goal_routing_experiment import shortest_path_tables
        route_distances, route_predecessors = shortest_path_tables(atlas, summary["edge_count_threshold"])
    for current, goal in set(zip(charts.tolist(), goal_charts.tolist())):
        if terminal_only:
            route_cache[(current, goal)] = []
            continue
        if dynamic_goal:
            route = [goal] if np.isfinite(route_distances[current, goal]) else []
            while route and route[-1] != current:
                parent = int(route_predecessors[current, route[-1]])
                if parent < 0 or len(route) > len(atlas["centers"]):
                    raise ValueError("Invalid saved-goal route")
                route.append(parent)
            route.reverse()
        else:
            route = shortest_chart_path(atlas, current, goal, summary["edge_count_threshold"])
        route_cache[(current, goal)] = route if route else [current]

    colors = {
        "actual": "#172238",
        "prediction": "#0284C7",
        "reference": "#E8790C",
        "target": "#C81E1E",
        "route": "#7C3AED",
        "measured": "#07825A",
        "grid": "#D8DEE8",
        "muted": "#667085",
    }
    success = bool(summary["success_30"] and summary["success_100"])
    reached_target = float(summary["minimum"]) < 0.01
    if success:
        outcome_label = "STRICT SUCCESS"
    elif reached_target:
        outcome_label = "FAILED STRICT TEST (reached <1 cm)"
    else:
        outcome_label = "FAILED STRICT TEST"
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.facecolor": "#F5F7FA",
        "axes.facecolor": "white",
        "axes.edgecolor": "#AAB3C2",
        "axes.linewidth": 0.8,
        "grid.color": colors["grid"],
        "grid.linewidth": 0.7,
        "grid.alpha": 0.8,
    })

    fig = plt.figure(figsize=(16, 9), dpi=120)
    grid = fig.add_gridspec(
        2, 3, width_ratios=(1.25, 1.0, 1.0), height_ratios=(1.05, 0.95)
    )
    arm_ax = fig.add_subplot(grid[:, 0])
    chart_ax = fig.add_subplot(grid[0, 1:])
    distance_ax = fig.add_subplot(grid[1, 1])
    control_ax = fig.add_subplot(grid[1, 2])
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.075, top=0.855,
                        wspace=0.27, hspace=0.34)

    method_title = (
        "Graph-routed Contextual Atlas-DeePC"
        if contextual_run else "Graph-routed Atlas-DeePC"
    )
    if summary.get("baseline_version") == "linear_iterative_n10_v1":
        method_title = "Atlas-DeePC | Linear charts + iterative N=10"
    if terminal_only:
        method_title = "Atlas-DeePC | no graph waypoints"
    if noisy_run:
        method_title = "Atlas-DeePC"
        outcome_label = "STRICT SUCCESS" if success else "STRICT FAILURE"
    title_text = fig.suptitle(
        f"{method_title} | {evaluation_label} | {outcome_label} | "
        f"{play_speed:g}x play speed",
        x=0.055, y=0.965, ha="left", fontsize=14 if dynamic_goal else (16 if contextual_run else 19),
        fontweight="bold",
        color=colors["actual"] if success else "#991B1B",
    )
    status_text = fig.text(
        0.055, 0.915, "", ha="left", va="center", fontsize=11,
        color="#344054",
    )
    arm_ax.set_title("Physical motion and terminal reference" if terminal_only
                     else "Physical motion and chart-medoid route", pad=12)
    arm_ax.set_aspect("equal")
    arm_ax.set_xlim(-0.225, 0.225)
    arm_ax.set_ylim(-0.225, 0.225)
    arm_ax.set_xlabel("workspace x (m)")
    arm_ax.set_ylabel("workspace y (m)")
    arm_ax.grid(True)
    arm_ax.axhline(0, color="#CBD2DC", linewidth=0.7)
    arm_ax.axvline(0, color="#CBD2DC", linewidth=0.7)
    arm_ax.add_patch(plt.Circle(
        (0, 0), WORKSPACE_RADIUS, fill=False, linestyle="--", linewidth=1.0,
        color="#98A2B3", alpha=0.8,
    ))
    medoid_xy = np.asarray(atlas["medoid_states"], dtype=float)[:, 6:8]
    arm_ax.scatter(
        medoid_xy[:, 0], medoid_xy[:, 1], s=13,
        c=np.arange(medoid_xy.shape[0]), cmap="turbo", alpha=0.24,
        linewidth=0, zorder=1,
    )
    arm_ax.scatter(
        [target[0]], [target[1]], s=160, facecolor="none",
        edgecolor=colors["target"], linewidth=2.2, zorder=7,
    )
    arm_ax.scatter(
        [target[0]], [target[1]], s=24, color=colors["target"], zorder=7,
    )
    route_line, = arm_ax.plot(
        [], [], "--", color=colors["route"], linewidth=1.5, alpha=0.65,
        marker="o", markersize=5, zorder=2,
    )
    trail_line, = arm_ax.plot(
        [], [], color=colors["measured"], linewidth=2.0, alpha=0.55, zorder=3,
    )
    reference_line, = arm_ax.plot(
        [], [], "-o", color=colors["reference"], linewidth=4.0,
        markersize=7, alpha=0.58, zorder=4,
    )
    prediction_line, = arm_ax.plot(
        [], [], "--o", color=colors["prediction"], linewidth=2.2,
        markersize=5, alpha=0.65, zorder=5,
    )
    actual_line, = arm_ax.plot(
        [], [], "-o", color=colors["actual"], linewidth=5.5,
        markerfacecolor="white", markeredgewidth=2.0, markersize=9, zorder=6,
    )
    sensor_point = arm_ax.scatter([], [], marker="x", s=65, color="#667085",
                                  linewidths=1.7, zorder=9) if noisy_run else None
    active_reference_point = arm_ax.scatter(
        [], [], s=80, marker="D", color=colors["reference"],
        edgecolor="white", linewidth=1.2, zorder=8,
    )
    route_labels = [
        arm_ax.text(0, 0, "", fontsize=8, fontweight="bold",
                    color=colors["route"], zorder=8)
        for _ in range(max(2, int(np.max(result["path_length"]))))
    ]
    arm_legend = [
        Line2D([0], [0], color=colors["actual"], linewidth=4,
               label="true arm" if noisy_run else "measured arm"),
        Line2D([0], [0], color=colors["prediction"], linewidth=2,
               linestyle="--", label="DeePC one-step prediction"),
        Line2D([0], [0], color=colors["reference"], linewidth=3,
               label="selected terminal posture" if terminal_only else "active waypoint posture"),
        Line2D([0], [0], marker="o", color="none",
               markeredgecolor=colors["target"], markerfacecolor="none",
               markersize=9, label="online target"),
    ]
    if not terminal_only:
        arm_legend.insert(-1, Line2D([0], [0], color=colors["route"], linewidth=1.5,
                                    linestyle="--", marker="o", label="route through chart medoids"))
    if noisy_run:
        arm_legend.append(Line2D([0], [0], color="#667085", marker="x", linestyle="none",
                                 label="noisy fingertip measurement"))
    arm_ax.legend(handles=arm_legend, loc=arm_legend_location, framealpha=0.94)

    chart_ax.set_title(
        "Current and selected goal charts" if terminal_only else (
            "Physical routing charts and contextual activation" if contextual_run
            else "Requested chart versus next measured chart"),
        loc="left",
        pad=10,
    )
    transition_text = chart_ax.text(
        0.995, 1.04, "", transform=chart_ax.transAxes, ha="right", va="bottom",
        fontsize=10, fontweight="bold",
    )
    chart_ax.set_xlim(0, y.shape[0] - 1)
    chart_ax.set_ylim(-2, max(int(np.max(charts)), int(np.max(requested)),
                              int(np.max(goal_charts))) + 4)
    chart_ax.set_xlabel("control step")
    chart_ax.set_ylabel("chart ID")
    chart_ax.grid(True)
    chart_ax.plot(steps, charts, color=colors["actual"], linewidth=1.0, alpha=0.12)
    if not terminal_only:
        chart_ax.plot(steps, requested, color=colors["reference"], linewidth=1.0, alpha=0.12)
    chart_ax.plot(steps, measured_next, color=colors["measured"], linewidth=1.0, alpha=0.12)
    chart_ax.step(steps, goal_charts, where="post", color=colors["target"], linewidth=1.4,
                  linestyle=":", alpha=0.8,
                  label="selected goal chart" if dynamic_goal else f"goal chart {goal_charts[0]}")
    if trigger_step >= 0:
        chart_ax.axvspan(
            trigger_step, y.shape[0] - 1, color=colors["measured"], alpha=0.08,
            label=f"contextual basis active from step {trigger_step + 1}",
        )
        chart_ax.axvline(
            trigger_step, color=colors["measured"], linewidth=1.2,
            linestyle="--", alpha=0.8,
        )
    current_chart_line, = chart_ax.step(
        [], [], where="post", color=colors["actual"], linewidth=2.2,
        label="current chart",
    )
    requested_chart_line, = chart_ax.step(
        [], [], where="post", color=colors["reference"], linewidth=2.2,
        label="_nolegend_" if terminal_only else "requested next",
    )
    measured_chart_line, = chart_ax.step(
        [], [], where="post", color=colors["measured"], linewidth=1.8,
        label="measured after action",
    )
    chart_cursor = chart_ax.axvline(0, color="#475467", linewidth=1.2, alpha=0.8)
    chart_markers = [
        chart_ax.scatter([], [], s=48, color=color, edgecolor="white",
                         linewidth=0.8, zorder=5)
        for color in (colors["actual"], colors["reference"], colors["measured"])
    ]
    if terminal_only:
        requested_chart_line.set_visible(False)
        chart_markers[1].set_visible(False)
    chart_ax.legend(loc="upper right", ncol=2, framealpha=0.94)

    distance_scale = 1000 if noisy_run else 100
    distance_unit = "mm" if noisy_run else "cm"
    distance_cm = distance_scale * distance
    distance_ax.set_title("Fingertip error (zoomed)" if error_limit_mm else "Target convergence", pad=10)
    distance_ax.set_xlim(0, y.shape[0] - 1)
    distance_ax.set_ylim(0, error_limit_mm * distance_scale / 1000 if error_limit_mm
                         else max(.085 * distance_scale, 1.08 * float(np.max(distance_cm))))
    distance_ax.set_xlabel("control step")
    distance_ax.set_ylabel(f"fingertip error ({distance_unit})")
    distance_ax.grid(True)
    distance_ax.plot(steps, distance_cm, color=colors["target"], alpha=0.15,
                     linewidth=1.2)
    closest_step = int(np.argmin(distance))
    distance_ax.scatter(
        [closest_step], [distance_cm[closest_step]], s=85, marker="*",
        color="#7C3AED", edgecolor="white", linewidth=0.8, zorder=4,
        label="_nolegend_" if noisy_run else f"closest displayed: {distance_cm[closest_step]:.2f} cm",
    )
    if not terminal_only and (error_limit_mm is None or error_limit_mm > 70):
        distance_ax.axhline(.07 * distance_scale, color="#D99A12", linestyle="--", linewidth=1.4,
                            label="target reference latch (7 cm)")
    distance_ax.axhline(.01 * distance_scale, color=colors["measured"], linestyle="--",
                        linewidth=1.4, label="mean-error criterion (10 mm)" if noisy_run else "mean-error criterion (1 cm)")
    distance_line, = distance_ax.plot([], [], color=colors["target"], linewidth=2.4,
                                      label="true error" if noisy_run else "_nolegend_")
    sensor_distance_line = None
    if noisy_run:
        sensor_distance_line, = distance_ax.plot([], [], color="#667085", linewidth=1,
                                                 alpha=.65, label="noisy measured error")
    distance_cursor = distance_ax.axvline(0, color="#475467", linewidth=1.2, alpha=0.8)
    distance_marker = distance_ax.scatter([], [], s=52, color=colors["target"],
                                          edgecolor="white", linewidth=0.8, zorder=5)
    distance_ax.legend(loc="upper right", framealpha=0.94)

    control_ax.set_title("DeePC control applied to the two joints", pad=10)
    control_ax.set_xlim(0, y.shape[0] - 1)
    control_ax.set_ylim(-1.12, 1.12)
    control_ax.set_xlabel("control step")
    control_ax.set_ylabel("normalized torque")
    control_ax.grid(True)
    control_ax.axhline(1.0, color="#98A2B3", linestyle=":", linewidth=1.0)
    control_ax.axhline(-1.0, color="#98A2B3", linestyle=":", linewidth=1.0)
    control_ax.plot(steps, u[:, 0], color="#2563EB", alpha=0.12, linewidth=1.0)
    control_ax.plot(steps, u[:, 1], color="#D97706", alpha=0.12, linewidth=1.0)
    u1_line, = control_ax.plot([], [], color="#2563EB", linewidth=2.0,
                               label="joint 1 input")
    u2_line, = control_ax.plot([], [], color="#D97706", linewidth=2.0,
                               label="joint 2 input")
    control_cursor = control_ax.axvline(0, color="#475467", linewidth=1.2, alpha=0.8)
    u1_marker = control_ax.scatter([], [], s=42, color="#2563EB",
                                   edgecolor="white", linewidth=0.8, zorder=5)
    u2_marker = control_ax.scatter([], [], s=42, color="#D97706",
                                   edgecolor="white", linewidth=0.8, zorder=5)
    control_ax.legend(loc="upper right", framealpha=0.94)

    def interpolate(values, position):
        lower = int(np.floor(position))
        upper = min(lower + 1, values.shape[0] - 1)
        fraction = position - lower
        return (1.0 - fraction) * values[lower] + fraction * values[upper]

    def update(position):
        step = int(np.floor(position))
        actual_state = interpolate(y, position)
        predicted_state = interpolate(prediction, position)
        current_distance = float(interpolate(distance, position))
        current_u = interpolate(u, position)
        actual_points = arm_points(actual_state)
        reference_points = arm_points(reference[step])
        predicted_points = arm_points(predicted_state)
        actual_line.set_data(actual_points[:, 0], actual_points[:, 1])
        if noisy_run:
            sensor_point.set_offsets(sensed_y[step, 6:8][None, :])
        reference_line.set_data(reference_points[:, 0], reference_points[:, 1])
        prediction_line.set_data(predicted_points[:, 0], predicted_points[:, 1])
        trail_start = max(0, step - 90)
        trail_xy = np.vstack([y[trail_start:step + 1, 6:8], actual_state[6:8]])
        trail_line.set_data(trail_xy[:, 0], trail_xy[:, 1])
        active_reference_point.set_offsets(reference[step, 6:8][None, :])

        current = int(charts[step])
        goal = int(goal_charts[step])
        route = route_cache[(current, goal)]
        route_xy = medoid_xy[route]
        route_line.set_data(route_xy[:, 0], route_xy[:, 1])
        for label_index, label in enumerate(route_labels):
            if label_index < len(route):
                point = route_xy[label_index]
                label.set_position((point[0] + 0.006, point[1] + 0.006))
                label.set_text(str(route[label_index]))
                label.set_visible(True)
            else:
                label.set_visible(False)

        prefix = np.r_[steps[:step + 1], position]
        current_chart_line.set_data(prefix, np.r_[charts[:step + 1], charts[step]])
        requested_chart_line.set_data(prefix, np.r_[requested[:step + 1], requested[step]])
        measured_chart_line.set_data(
            prefix, np.where(np.r_[measured_next[:step + 1], measured_next[step]] < 0,
                             np.nan, np.r_[measured_next[:step + 1], measured_next[step]])
        )
        chart_cursor.set_xdata([position, position])
        for marker, value in zip(
            chart_markers, (charts[step], requested[step], measured_next[step])
        ):
            marker.set_offsets(np.empty((0, 2)) if value < 0 else np.array([[position, value]]))

        distance_line.set_data(prefix, np.r_[distance_cm[:step + 1], distance_scale * current_distance])
        if noisy_run:
            sensor_distance_line.set_data(steps[:step + 1], distance_scale*sensed_distance[:step + 1])
        distance_cursor.set_xdata([position, position])
        distance_marker.set_offsets(np.array([[position, distance_scale * current_distance]]))
        u1_line.set_data(prefix, np.r_[u[:step + 1, 0], current_u[0]])
        u2_line.set_data(prefix, np.r_[u[:step + 1, 1], current_u[1]])
        control_cursor.set_xdata([position, position])
        u1_marker.set_offsets(np.array([[position, current_u[0]]]))
        u2_marker.set_offsets(np.array([[position, current_u[1]]]))

        target_latched = np.linalg.norm(reference[step, 6:8] - target) < 1e-10
        reference_mode = "exact target latched" if target_latched else f"chart {requested[step]} waypoint"
        if dynamic_goal and target_latched:
            reference_mode = f"terminal candidate {int(result['goal_index'][step])}"
        route_text = " -> ".join(str(node) for node in route)
        if terminal_only:
            route_text = "disabled"
        if contextual_active[step]:
            matrix_text = (
                f"contextual matrix: {contextual_clusters[step]} "
                f"({matrix_columns[step]} columns, SVD rank {summary['svd_rank']})"
            )
        else:
            matrix_text = (
                f"physical matrix: {current} "
                f"({matrix_columns[step]} columns, SVD rank {summary['svd_rank']})"
            )
        status_text.set_text(
            f"seed {summary['seed']} | step {step + 1}/{y.shape[0]} | "
            f"route {route_text} | reference: {reference_mode} | "
            f"{matrix_text} | "
            f"{statuses[step]} | error {1000 * current_distance:.2f} mm"
        )
        transition_observed = int(measured_next[step]) == int(requested[step])
        if measured_next[step] < 0:
            transition_text.set_text(f"current {current} | goal {goal} | next chart not recorded")
            transition_text.set_color(colors["muted"])
        elif terminal_only:
            transition_text.set_text(f"current {current} | goal {goal} | next measured {measured_next[step]}")
            transition_text.set_color(colors["muted"])
        elif transition_observed:
            transition_text.set_text(
                f"requested {current} -> {requested[step]} | measured next {measured_next[step]}"
            )
            transition_text.set_color(colors["measured"])
        else:
            transition_text.set_text(
                f"requested {current} -> {requested[step]} | measured next {measured_next[step]} | replan"
            )
            transition_text.set_color(colors["reference"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    update(int(np.clip(poster_step, 0, y.shape[0] - 1)))
    fig.savefig(poster_path, dpi=120, facecolor=fig.get_facecolor())

    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=1800 if noisy_run else 5000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={
            "title": f"{method_title} | {evaluation_label} | seed {summary['seed']}",
            "artist": "Atlas-DeePC experiment renderer",
        },
    )
    if not 0 < play_speed <= 1:
        raise ValueError("play_speed must be in the interval (0, 1].")
    frame_increment = frame_stride * play_speed
    frame_positions = np.arange(0.0, y.shape[0] - 1, frame_increment).tolist()
    if not frame_positions or frame_positions[-1] != y.shape[0] - 1:
        frame_positions.append(float(y.shape[0] - 1))
    with writer.saving(fig, str(output_path), dpi=120):
        for position in frame_positions:
            update(position)
            writer.grab_frame(facecolor=fig.get_facecolor())
        for _ in range(round(hold_seconds * fps)):
            writer.grab_frame(facecolor=fig.get_facecolor())
    plt.close(fig)

    metadata = {
        "source_result": str(result_path.resolve()),
        "source_result_sha256": file_sha256(result_path),
        "source_atlas": str(atlas_path.resolve()),
        "source_atlas_sha256": file_sha256(atlas_path),
        "output_video": str(output_path.resolve()),
        "poster": str(poster_path.resolve()),
        "seed": int(summary["seed"]),
        "saved_control_steps": total_saved_steps,
        "displayed_control_steps": int(y.shape[0]),
        "rendered_frames": int(len(frame_positions) + round(hold_seconds * fps)),
        "fps": int(fps),
        "frame_stride": int(frame_stride),
        "play_speed": float(play_speed),
        "target_xy_m": target.tolist(),
        "final_error_mm": float(1000 * distance[-1]),
        "success_30": bool(summary["success_30"]),
        "success_100": bool(summary["success_100"]),
        "chart_semantics": {
            "current": "chart selected from the pre-action measured history",
            "requested_next": "unused: terminal-reference-only mode" if terminal_only
                              else "next chart on the graph route toward the goal chart",
            "measured_next": "chart selected at the following control step after applying the action",
        },
        "contextual_run": bool(contextual_run),
        "graph_waypoints_used": not terminal_only,
        "arm_legend_location": arm_legend_location,
        "noise_sd_ratio": summary.get("noise_sd_ratio"),
        "lam_g": summary.get("lam_g"),
        "true_arm_displayed": noisy_run,
        "noisy_fingertip_displayed": noisy_run,
        "error_panel_limit_mm": error_limit_mm,
        "contextual_trigger_step": int(trigger_step),
        "contextual_usage_fraction_displayed": float(np.mean(contextual_active)),
        "controller": summary["controller"],
        "fitted_dynamics_model": bool(summary["fitted_dynamics_model"]),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return output_path, poster_path, metadata_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a synchronized dashboard for a saved routed Atlas-DeePC rollout"
    )
    parser.add_argument("result_path", type=Path)
    parser.add_argument("--atlas-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--play-speed", type=float, default=1.0)
    parser.add_argument("--end-step", type=int)
    parser.add_argument("--hold-seconds", type=float, default=1.5)
    parser.add_argument("--poster-step", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    outputs = render_dashboard(
        args.result_path,
        args.atlas_path,
        args.output,
        fps=args.fps,
        frame_stride=args.frame_stride,
        play_speed=args.play_speed,
        end_step=args.end_step,
        hold_seconds=args.hold_seconds,
        poster_step=args.poster_step,
    )
    print("saved", *(str(path) for path in outputs))
