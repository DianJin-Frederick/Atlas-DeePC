import argparse
import heapq
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from sklearn.cluster import MiniBatchKMeans

import expert_pipeline_search as pipe
from local_affine_atlas_experiment import DEFAULT_EVAL_SEEDS, audit_iid_library
from stitched_atlas_mpc_experiment import (
    StitchedAtlasConfig,
    StitchedAtlasController,
    SoftStitchedAtlasController,
    chart_compatible_goal,
    load_atlas,
    nearest_charts,
    transition_data,
)


EXPERIMENT_DIR = Path("experiments/routed_atlas_iid")
DEFAULT_DATA_PATH = Path("data/select_iid_offline_hankel.npy")


@dataclass
class RoutedAtlasConfig:
    clusters: int = 64
    graph_clusters: int = 64
    graph_seed: int = 11
    route_mode: str = "hybrid"
    edge_count_threshold: int = 3
    route_horizon: int = 10
    target_horizon: int = 12
    near_horizon: int = 3
    target_switch_distance: float = 0.07
    goal_chart_handoff: bool = True
    latch_local_controller: bool = True
    goal_chart_override: int | None = None
    goal_chart_candidates: int = 400
    goal_charts: int = 2
    goal_branch_selection: str = "closest_xy"
    local_controller_mode: str = "hard"
    soft_temperature: float = 0.1
    soft_top_charts: int = 4
    adaptive_hybrid_on_stall: bool = False
    stall_window: int = 20
    stall_progress: float = 0.002
    edge_action_scale: float = 1.0
    hybrid_edge_blend: float = 0.35
    schedule_iterations: int = 3
    q_xy: float = 100000.0
    q_dq: float = 20.0
    q_data_goal_joint: float = 5000.0
    r: float = 0.0001
    lam_du: float = 5.0
    u_limit: float = 1.0
    steps: int = 320
    seed: int = 1


def graph_chart_ids(graph, z):
    z = np.asarray(z)
    if z.ndim == 1:
        z = z[None, :]
    delta = z[:, None, :] - graph["gate_centers"][None, :, :]
    return np.argmin(np.mean(delta * delta, axis=2), axis=1)


def build_transition_graph(H, atlas, graph_clusters=None, graph_seed=11):
    y, u, y_next = transition_data(H)
    z = (y - atlas["y_mean"]) / atlas["y_scale"]
    z_next = (y_next - atlas["y_mean"]) / atlas["y_scale"]
    model_clusters = atlas["gate_centers"].shape[0]
    clusters = model_clusters if graph_clusters is None else int(graph_clusters)
    if clusters == model_clusters:
        gate_centers = atlas["gate_centers"]
        source = nearest_charts(atlas, z)
    else:
        gate = MiniBatchKMeans(
            n_clusters=clusters,
            random_state=graph_seed,
            n_init=5,
            batch_size=2048,
            max_iter=300,
            reassignment_ratio=0.002,
        )
        source = gate.fit_predict(z)
        gate_centers = gate.cluster_centers_
    delta_next = z_next[:, None, :] - gate_centers[None, :, :]
    destination = np.argmin(np.mean(delta_next * delta_next, axis=2), axis=1)
    counts = np.zeros((clusters, clusters), dtype=int)
    action_sum = np.zeros((clusters, clusters, pipe.M), dtype=float)
    for i, j, action in zip(source, destination, u):
        counts[i, j] += 1
        action_sum[i, j] += action
    mean_action = action_sum / np.maximum(counts[:, :, None], 1)

    medoid_states = []
    for cluster_id in range(clusters):
        idx = np.where(source == cluster_id)[0]
        error = np.mean((z[idx] - gate_centers[cluster_id]) ** 2, axis=1)
        medoid = y[idx[int(np.argmin(error))]].copy()
        medoid[4:6] = 0.0
        medoid_states.append(medoid)
    center_delta = gate_centers[:, None, :] - gate_centers[None, :, :]
    center_distance = np.sqrt(np.mean(center_delta * center_delta, axis=2))
    return {
        "counts": counts,
        "gate_centers": gate_centers,
        "mean_action": mean_action,
        "medoid_states": np.asarray(medoid_states),
        "center_distance": center_distance,
    }


def shortest_chart_path(graph, source, destination, count_threshold):
    if source == destination:
        return [source]
    counts = graph["counts"]
    distances = [np.inf] * counts.shape[0]
    previous = [-1] * counts.shape[0]
    distances[source] = 0.0
    queue = [(0.0, source)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        if node == destination:
            break
        for neighbor in np.where(counts[node] >= count_threshold)[0]:
            neighbor = int(neighbor)
            if neighbor == node:
                continue
            reliability = np.sqrt(float(counts[node, neighbor]))
            edge_cost = graph["center_distance"][node, neighbor] / reliability
            candidate = distance + edge_cost
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))
    if previous[destination] < 0:
        return []
    path = [destination]
    while path[-1] != source:
        path.append(previous[path[-1]])
    return path[::-1]


def controller_config(config, horizon):
    return StitchedAtlasConfig(
        clusters=config.clusters,
        schedule_iterations=config.schedule_iterations,
        horizon=horizon,
        near_horizon=horizon,
        q_xy=config.q_xy,
        q_dq=config.q_dq,
        q_data_goal_joint=config.q_data_goal_joint,
        r=config.r,
        lam_du=config.lam_du,
        u_limit=config.u_limit,
        steps=config.steps,
        seed=config.seed,
    )


def run_routed_atlas(H, atlas, graph, config, save_path=None, fine_graph=None):
    route_controller = StitchedAtlasController(atlas, controller_config(config, config.route_horizon))
    target_controller = StitchedAtlasController(atlas, controller_config(config, config.target_horizon))
    if config.local_controller_mode == "soft":
        near_controller = SoftStitchedAtlasController(
            atlas,
            controller_config(config, config.near_horizon),
            temperature=config.soft_temperature,
            top_charts=config.soft_top_charts,
        )
    else:
        near_controller = StitchedAtlasController(atlas, controller_config(config, config.near_horizon))
    env = gym.make(pipe.REACHER_ENV_ID, max_episode_steps=config.steps + 5)
    obs, _ = env.reset(seed=config.seed)
    target_xy = np.asarray(obs[4:6], dtype=float).copy()
    current_y = pipe.reacher_output_from_obs(obs)
    goal, goal_metadata = chart_compatible_goal(H, target_xy, current_y, config)
    goal_z = (goal - atlas["y_mean"]) / atlas["y_scale"]
    goal_chart = int(graph_chart_ids(graph, goal_z)[0])
    fine_goal_chart = int(graph_chart_ids(fine_graph, goal_z)[0]) if fine_graph is not None else -1

    records = {
        "y": [], "u": [], "pred": [], "ref": [], "distance": [],
        "prediction_error_xy": [], "status": [], "chart": [], "goal_chart": [],
        "next_chart": [], "path_length": [], "action_source": [],
        "route_level": [],
    }
    local_controller_latched = False
    hybrid_route_latched = False
    for _ in range(config.steps):
        current_z = (current_y - atlas["y_mean"]) / atlas["y_scale"]
        current_chart = int(graph_chart_ids(graph, current_z)[0])
        distance = float(np.linalg.norm(current_y[6:8] - target_xy))
        active_graph = graph
        active_chart = current_chart
        active_goal_chart = goal_chart
        active_threshold = config.edge_count_threshold
        route_level = "coarse"
        if fine_graph is not None and current_chart == goal_chart:
            active_graph = fine_graph
            active_chart = int(graph_chart_ids(fine_graph, current_z)[0])
            active_goal_chart = fine_goal_chart
            active_threshold = 2
            route_level = "fine"
        path = shortest_chart_path(
            active_graph, active_chart, active_goal_chart, active_threshold
        )
        if not path:
            path = [active_chart]
        next_chart = path[1] if len(path) > 1 else active_chart

        should_handoff = distance < config.target_switch_distance
        reached_final_chart = active_chart == active_goal_chart
        if config.goal_chart_handoff and reached_final_chart:
            should_handoff = True
        if config.latch_local_controller and should_handoff:
            local_controller_latched = True
        use_local_controller = should_handoff or local_controller_latched

        if use_local_controller:
            active_reference = goal
            active_controller = near_controller
            action_source = "near_target_mpc"
        elif reached_final_chart:
            active_reference = goal
            active_controller = target_controller
            action_source = "target_chart_mpc"
        else:
            active_reference = active_graph["medoid_states"][next_chart]
            active_controller = route_controller
            action_source = "route_mpc"
        reference = np.tile(active_reference, (active_controller.config.horizon, 1))
        u_solution, z_prediction, _, _, status = active_controller.solve(current_y, reference)
        if u_solution is None:
            mpc_action = np.zeros(pipe.M)
            prediction = np.full(pipe.P, np.nan)
        else:
            mpc_action = np.asarray(u_solution[0], dtype=float)
            prediction = atlas["y_mean"] + atlas["y_scale"] * z_prediction[1]

        if (
            config.adaptive_hybrid_on_stall
            and action_source == "route_mpc"
            and len(records["distance"]) >= config.stall_window
        ):
            previous_distance = float(records["distance"][-config.stall_window])
            recent_best = min(float(x) for x in records["distance"][-config.stall_window:])
            if previous_distance - recent_best < config.stall_progress:
                hybrid_route_latched = True

        edge_action = active_graph["mean_action"][active_chart, next_chart] * config.edge_action_scale
        if action_source == "route_mpc" and config.route_mode == "edge_action":
            action = edge_action
            action_source = "edge_action"
            prediction = np.full(pipe.P, np.nan)
        elif action_source == "route_mpc" and (
            config.route_mode == "hybrid" or hybrid_route_latched
        ):
            blend = config.hybrid_edge_blend
            action = (1.0 - blend) * mpc_action + blend * edge_action
            action_source = "hybrid_route"
        else:
            action = mpc_action
        action = np.clip(action, -config.u_limit, config.u_limit)
        obs, *_ = env.step(action)
        y_now = pipe.reacher_output_from_obs(obs)
        records["y"].append(y_now)
        records["u"].append(action)
        records["pred"].append(prediction)
        records["ref"].append(active_reference)
        records["distance"].append(float(np.linalg.norm(y_now[6:8] - target_xy)))
        records["prediction_error_xy"].append(float(np.linalg.norm(prediction[6:8] - y_now[6:8])))
        records["status"].append(status)
        records["chart"].append(active_chart)
        records["goal_chart"].append(goal_chart)
        records["next_chart"].append(next_chart)
        records["path_length"].append(len(path))
        records["action_source"].append(action_source)
        records["route_level"].append(route_level)
        current_y = y_now
    env.close()

    for key in records:
        dtype = object if key in ("status", "action_source", "route_level") else None
        records[key] = np.asarray(records[key], dtype=dtype)
    distance = records["distance"]
    tail_30 = distance[-30:]
    tail_100 = distance[-min(100, distance.size):]
    summary = {
        **asdict(config),
        "minimum": float(distance.min()),
        "minimum_step": int(distance.argmin()),
        "last": float(distance[-1]),
        "mean_last_30": float(tail_30.mean()),
        "max_last_30": float(tail_30.max()),
        "mean_last_100": float(tail_100.mean()),
        "max_last_100": float(tail_100.max()),
        "success_30": bool(tail_30.mean() < 0.01 and tail_30.max() < 0.02),
        "success_100": bool(tail_100.mean() < 0.01 and tail_100.max() < 0.02),
        "mean_prediction_error_xy": float(np.nanmean(records["prediction_error_xy"])),
        "charts_visited": int(np.unique(records["chart"]).size),
        "goal_chart": goal_chart,
        "fine_goal_chart": fine_goal_chart,
        "statuses": {str(s): int(np.sum(records["status"] == s)) for s in np.unique(records["status"])},
        "action_sources": {str(s): int(np.sum(records["action_source"] == s)) for s in np.unique(records["action_source"])},
        "online_trajectory_retrieval": False,
        "use_ik_ref": False,
        "feedback_blend": 0.0,
        "feedback_fallback": False,
        **goal_metadata,
    }
    if save_path is not None:
        np.savez(save_path, **records, summary=np.array(json.dumps(summary), dtype=object))
    return summary


def main(args):
    output_dir = EXPERIMENT_DIR / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    H = np.load(args.data_path)
    atlas = load_atlas(args.atlas_path)
    audit = audit_iid_library(H, eval_seeds=args.seeds)
    if args.graph_path is not None and args.graph_path.exists():
        graph_data = np.load(args.graph_path, allow_pickle=True)
        graph = {key: graph_data[key] for key in graph_data.files}
        if "gate_centers" not in graph:
            if graph["counts"].shape[0] != atlas["gate_centers"].shape[0]:
                raise ValueError("Legacy graph chart count does not match the dynamics atlas.")
            graph["gate_centers"] = atlas["gate_centers"]
        if graph["gate_centers"].shape[0] != args.graph_clusters:
            raise ValueError(
                f"Loaded graph has {graph['gate_centers'].shape[0]} charts, "
                f"but --graph-clusters is {args.graph_clusters}."
            )
    else:
        graph = build_transition_graph(
            H, atlas, graph_clusters=args.graph_clusters, graph_seed=args.graph_seed
        )
    (output_dir / "fairness_audit.json").write_text(json.dumps(audit, indent=2))
    config = RoutedAtlasConfig(
        route_mode=args.route_mode,
        graph_clusters=args.graph_clusters,
        graph_seed=args.graph_seed,
        edge_count_threshold=args.edge_count_threshold,
        route_horizon=args.route_horizon,
        target_horizon=args.target_horizon,
        near_horizon=args.near_horizon,
        target_switch_distance=args.target_switch_distance,
        goal_chart_handoff=args.goal_chart_handoff,
        latch_local_controller=args.latch_local_controller,
        goal_chart_override=args.goal_chart_override,
        goal_branch_selection=args.goal_branch_selection,
        local_controller_mode=args.local_controller_mode,
        soft_temperature=args.soft_temperature,
        soft_top_charts=args.soft_top_charts,
        adaptive_hybrid_on_stall=args.adaptive_hybrid_on_stall,
        stall_window=args.stall_window,
        stall_progress=args.stall_progress,
        edge_action_scale=args.edge_action_scale,
        hybrid_edge_blend=args.hybrid_edge_blend,
        steps=args.steps,
    )
    np.savez(output_dir / "transition_graph.npz", **graph)
    summaries = []
    fine_graph = None
    if args.fine_graph_path is not None:
        fine_data = np.load(args.fine_graph_path, allow_pickle=True)
        fine_graph = {key: fine_data[key] for key in fine_data.files}
        np.savez(output_dir / "fine_transition_graph.npz", **fine_graph)
    for seed in args.seeds:
        run_config = RoutedAtlasConfig(**{**asdict(config), "seed": seed})
        result_path = output_dir / f"seed{seed}_control_results.npz"
        summary = run_routed_atlas(
            H, atlas, graph, run_config, save_path=result_path, fine_graph=fine_graph
        )
        summary["result_path"] = str(result_path)
        summaries.append(summary)
        print(
            "seed", seed,
            "mean30", f"{summary['mean_last_30']:.6f}",
            "minimum", f"{summary['minimum']:.6f}",
            "charts", summary["charts_visited"],
            "success", summary["success_30"] and summary["success_100"],
            flush=True,
        )
    payload = {
        "method": "offline_chart_transition_graph_with_local_atlas_control",
        "config": asdict(config),
        "fairness_audit": audit,
        "graph_edge_count": int(np.sum(graph["counts"] >= config.edge_count_threshold)),
        "results": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Graph-routed stitched Atlas control")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--atlas-path", type=Path, default=Path("experiments/stitched_atlas_iid/candidate_v2/stitched_atlas.npz"))
    parser.add_argument("--route-mode", choices=("waypoint_mpc", "edge_action", "hybrid"), default="hybrid")
    parser.add_argument("--graph-clusters", type=int, default=64)
    parser.add_argument("--graph-seed", type=int, default=11)
    parser.add_argument("--graph-path", type=Path)
    parser.add_argument("--fine-graph-path", type=Path)
    parser.add_argument("--edge-count-threshold", type=int, default=3)
    parser.add_argument("--route-horizon", type=int, default=10)
    parser.add_argument("--target-horizon", type=int, default=12)
    parser.add_argument("--near-horizon", type=int, default=3)
    parser.add_argument("--target-switch-distance", type=float, default=0.07)
    parser.add_argument("--goal-chart-handoff", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--latch-local-controller", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--goal-chart-override", type=int)
    parser.add_argument("--goal-branch-selection", choices=("compatible", "closest_xy"), default="closest_xy")
    parser.add_argument("--local-controller-mode", choices=("hard", "soft"), default="hard")
    parser.add_argument("--soft-temperature", type=float, default=0.1)
    parser.add_argument("--soft-top-charts", type=int, default=4)
    parser.add_argument("--adaptive-hybrid-on-stall", action="store_true")
    parser.add_argument("--stall-window", type=int, default=20)
    parser.add_argument("--stall-progress", type=float, default=0.002)
    parser.add_argument("--edge-action-scale", type=float, default=1.0)
    parser.add_argument("--hybrid-edge-blend", type=float, default=0.35)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_EVAL_SEEDS)
    parser.add_argument("--output-name", default="candidate")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
