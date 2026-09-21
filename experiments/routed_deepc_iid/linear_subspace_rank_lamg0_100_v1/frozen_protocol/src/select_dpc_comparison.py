"""Matched Reacher benchmark for Select-DPC and graph-routed Atlas-DeePC.

The Select-DPC loop follows Algorithm 1 and the supplied reference code:
the most recent complete predicted input-output trajectory is used as a query,
the closest full Hankel columns are selected with the raw L1 metric, and a DPC
problem is solved repeatedly before the first action is applied.  For a fair
computational comparison, both methods use this repository's output vector,
horizons, target construction, costs, input constraints, solver, and frozen
offline Hankel library.  Thus the experiment preserves Select-DPC's defining
selection/refinement idea while avoiding a confound from unrelated controller
settings in the paper's original Reacher notebook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import cvxpy as cp
import gymnasium as gym
import numpy as np
import psutil

import expert_pipeline_search as pipe
from routed_deepc_experiment import (
    DEFAULT_DATA_PATH,
    RoutedDeePCConfig,
    load_atlas,
    run_routed_deepc,
)
from stitched_atlas_mpc_experiment import chart_compatible_goal


DEFAULT_SELECT_CODE_ZIP = Path("/Users/kumiko/Projects/choose-wisely-paper-main.zip")
DEFAULT_ATLAS_PATH = Path(
    "experiments/routed_deepc_iid/heldout100_frozen_target_separated/behavioral_atlas.npz"
)
DEFAULT_OUTPUT_DIR = Path("experiments/select_dpc_comparison")
REFERENCE_CONTROLLER_MEMBER = (
    "choose-wisely-paper-main/select_deepc/deepc_controller.py"
)
REFERENCE_SELECTOR_MEMBER = "choose-wisely-paper-main/select_deepc/data_selectors.py"


@dataclass
class SelectDPCConfig:
    num_columns: int = 100
    refinement_iterations: int = 5
    steps: int = 320
    q_xy: float = 100000.0
    q_dq: float = 20.0
    q_data_goal_joint: float = 5000.0
    r: float = 0.0001
    lam_g: float = 0.05
    lam_ini: float = 100000.0
    lam_du: float = 5.0
    u_limit: float = 1.0
    goal_chart_candidates: int = 400
    goal_charts: int = 2
    goal_source_target_min_distance: float = 0.06
    goal_branch_selection: str = "compatible"
    seed: int = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_code_provenance(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Select-DPC source archive not found: {path}")
    with zipfile.ZipFile(path) as archive:
        controller_source = archive.read(REFERENCE_CONTROLLER_MEMBER)
        selector_source = archive.read(REFERENCE_SELECTOR_MEMBER)
    return {
        "archive_path": str(path.resolve()),
        "archive_sha256": sha256_file(path),
        "controller_member": REFERENCE_CONTROLLER_MEMBER,
        "controller_sha256": hashlib.sha256(controller_source).hexdigest(),
        "selector_member": REFERENCE_SELECTOR_MEMBER,
        "selector_sha256": hashlib.sha256(selector_source).hexdigest(),
        "adaptation": (
            "Exact raw-L1 full-trajectory ranking and iterative selection/DPC "
            "structure, with the shared Atlas Reacher protocol and DeePC subproblem."
        ),
    }


def output_cost(config: SelectDPCConfig) -> np.ndarray:
    return np.diag(
        [
            config.q_data_goal_joint,
            config.q_data_goal_joint,
            config.q_data_goal_joint,
            config.q_data_goal_joint,
            config.q_dq,
            config.q_dq,
            config.q_xy,
            config.q_xy,
            config.q_data_goal_joint,
            config.q_data_goal_joint,
        ]
    )


def matched_deepc_controller(config: SelectDPCConfig) -> pipe.DeePCController:
    return pipe.DeePCController(
        output_cost(config),
        config.r * np.eye(pipe.M),
        np.zeros((pipe.P, pipe.P)),
        config.lam_g,
        config.lam_ini,
        config.lam_du,
        config.u_limit,
        cp.CLARABEL,
    )


def select_l1_full_trajectory(
    H_u: np.ndarray,
    H_y: np.ndarray,
    input_trajectory: np.ndarray,
    output_trajectory: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The supplied LkSelector(order=1) ranking, without its unused torch imports."""
    input_distance = np.linalg.norm(H_u - input_trajectory, ord=1, axis=0)
    output_distance = np.linalg.norm(H_y - output_trajectory, ord=1, axis=0)
    distances = input_distance + output_distance
    order = np.argsort(distances, kind="stable")
    return order[:count], distances


def percentile_summary(values: np.ndarray, prefix: str) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        f"median_{prefix}_ms": float(1000 * np.median(values)),
        f"p95_{prefix}_ms": float(1000 * np.percentile(values, 95)),
        f"max_{prefix}_ms": float(1000 * np.max(values)),
    }


def run_select_dpc_matched(
    H: np.ndarray,
    config: SelectDPCConfig,
    save_path: Path | None = None,
) -> dict:
    """Run protocol-matched Select-DPC with detailed end-to-end timing."""
    if not 1 <= config.num_columns <= H.shape[1]:
        raise ValueError("num_columns must be between 1 and the library size")
    if config.refinement_iterations < 1:
        raise ValueError("refinement_iterations must be positive")

    H_u = H[: pipe.M * pipe.L]
    H_y = H[pipe.M * pipe.L :]
    controller = matched_deepc_controller(config)
    goal_config = RoutedDeePCConfig(
        q_xy=config.q_xy,
        q_dq=config.q_dq,
        q_data_goal_joint=config.q_data_goal_joint,
        r=config.r,
        lam_g=config.lam_g,
        lam_ini=config.lam_ini,
        lam_du=config.lam_du,
        u_limit=config.u_limit,
        goal_chart_candidates=config.goal_chart_candidates,
        goal_charts=config.goal_charts,
        goal_source_target_min_distance=config.goal_source_target_min_distance,
        goal_branch_selection=config.goal_branch_selection,
        steps=config.steps,
        seed=config.seed,
    )

    env = gym.make(pipe.REACHER_ENV_ID, max_episode_steps=config.steps + pipe.T_INI + 5)
    obs, _ = env.reset(seed=config.seed)
    target_xy = np.asarray(obs[4:6], dtype=float).copy()
    initial_y = pipe.reacher_output_from_obs(obs)
    goal, goal_metadata = chart_compatible_goal(H, target_xy, initial_y, goal_config)
    reference = np.tile(goal.reshape(pipe.P, 1), (pipe.T_F, 1))

    u_history: list[np.ndarray] = []
    y_history: list[np.ndarray] = []
    for _ in range(pipe.T_INI):
        action = np.zeros(pipe.M)
        obs, *_ = env.step(action)
        u_history.append(action.copy())
        y_history.append(pipe.reacher_output_from_obs(obs))

    planned_u = np.zeros((pipe.M * pipe.T_F, 1))
    planned_y = np.tile(y_history[-1].reshape(pipe.P, 1), (pipe.T_F, 1))
    previous_selection: np.ndarray | None = None
    process = psutil.Process()
    records: dict[str, list] = {
        "y": [],
        "u": [],
        "pred": [],
        "ref": [],
        "distance": [],
        "prediction_error_xy": [],
        "status": [],
        "qp_statuses": [],
        "selection_time_s": [],
        "matrix_preparation_time_s": [],
        "solve_time_s": [],
        "control_compute_time_s": [],
        "selected_overlap_fraction": [],
        "selected_distance_median": [],
        "rss_bytes": [],
    }

    for _ in range(config.steps):
        control_start = time.perf_counter()
        u_ini = np.asarray(u_history[-pipe.T_INI :], dtype=float).reshape(
            pipe.M * pipe.T_INI, 1
        )
        y_ini = np.asarray(y_history[-pipe.T_INI :], dtype=float).reshape(
            pipe.P * pipe.T_INI, 1
        )
        query_u = np.vstack([u_ini, planned_u])
        query_y = np.vstack([y_ini, planned_y])
        selection_time = 0.0
        matrix_time = 0.0
        solve_time = 0.0
        iteration_statuses: list[str] = []
        selected = None
        distances = None
        u_solution = y_solution = g_solution = None

        for _iteration in range(config.refinement_iterations):
            stage_start = time.perf_counter()
            selected, distances = select_l1_full_trajectory(
                H_u, H_y, query_u, query_y, config.num_columns
            )
            selection_time += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            matrix = {
                "offset": np.zeros((H.shape[0], 1)),
                "basis": np.asarray(H[:, selected]),
                "g_sum": 1.0,
            }
            matrix_time += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            u_solution, y_solution, g_solution, status, _ = controller.solve(
                matrix, u_ini, y_ini, reference
            )
            solve_time += time.perf_counter() - stage_start
            iteration_statuses.append(str(status))
            if u_solution is None:
                break
            planned_u = np.asarray(u_solution, dtype=float).reshape(-1, 1)
            planned_y = np.asarray(y_solution, dtype=float).reshape(-1, 1)
            query_u = np.vstack([u_ini, planned_u])
            query_y = np.vstack([y_ini, planned_y])

        if u_solution is None:
            action = np.zeros(pipe.M)
            prediction = np.full(pipe.P, np.nan)
        else:
            action = np.clip(
                np.asarray(u_solution[: pipe.M, 0]).reshape(pipe.M),
                -config.u_limit,
                config.u_limit,
            )
            prediction = np.asarray(y_solution[: pipe.P, 0]).reshape(pipe.P)
        control_compute_time = time.perf_counter() - control_start

        obs, *_ = env.step(action)
        y_now = pipe.reacher_output_from_obs(obs)
        u_history.append(action.copy())
        y_history.append(y_now.copy())
        overlap = np.nan
        if previous_selection is not None and selected is not None:
            overlap = np.intersect1d(previous_selection, selected).size / config.num_columns
        if selected is not None:
            previous_selection = selected.copy()

        records["y"].append(y_now)
        records["u"].append(action)
        records["pred"].append(prediction)
        records["ref"].append(goal)
        records["distance"].append(float(np.linalg.norm(y_now[6:8] - target_xy)))
        records["prediction_error_xy"].append(
            float(np.linalg.norm(prediction[6:8] - y_now[6:8]))
        )
        records["status"].append(iteration_statuses[-1])
        records["qp_statuses"].append("|".join(iteration_statuses))
        records["selection_time_s"].append(selection_time)
        records["matrix_preparation_time_s"].append(matrix_time)
        records["solve_time_s"].append(solve_time)
        records["control_compute_time_s"].append(control_compute_time)
        records["selected_overlap_fraction"].append(overlap)
        records["selected_distance_median"].append(
            float(np.median(distances[selected])) if selected is not None else np.nan
        )
        records["rss_bytes"].append(process.memory_info().rss)

    env.close()
    arrays = {
        key: np.asarray(value, dtype=object if key in ("status", "qp_statuses") else None)
        for key, value in records.items()
    }
    distance = arrays["distance"]
    tail_30 = distance[-30:]
    tail_100 = distance[-min(100, distance.size) :]
    all_qp_statuses = [status for group in records["qp_statuses"] for status in group.split("|")]
    status_counts = {status: all_qp_statuses.count(status) for status in sorted(set(all_qp_statuses))}
    summary = {
        **asdict(config),
        "method": "select_dpc_matched_l1",
        "selector": "raw L1 over complete predicted input-output trajectories",
        "prediction_query": "measured-past anchored most-recent prediction",
        "dpc_subproblem": "shared with Atlas-DeePC; affine raw-column span with sum(g)=1",
        "minimum": float(distance.min()),
        "minimum_step": int(distance.argmin()),
        "last": float(distance[-1]),
        "mean_last_30": float(tail_30.mean()),
        "max_last_30": float(tail_30.max()),
        "mean_last_100": float(tail_100.mean()),
        "max_last_100": float(tail_100.max()),
        "success_30": bool(tail_30.mean() < 0.01 and tail_30.max() < 0.02),
        "success_100": bool(tail_100.mean() < 0.01 and tail_100.max() < 0.02),
        "mean_prediction_error_xy": float(np.nanmean(arrays["prediction_error_xy"])),
        "mean_selected_overlap_fraction": float(
            np.nanmean(arrays["selected_overlap_fraction"])
        ),
        "qp_count": len(all_qp_statuses),
        "qp_statuses": status_counts,
        "peak_observed_rss_mb": float(np.max(arrays["rss_bytes"]) / 1024**2),
        "library_columns": int(H.shape[1]),
        "optimized_coefficient_dimension": config.num_columns,
        **percentile_summary(arrays["selection_time_s"], "selection_time"),
        **percentile_summary(
            arrays["matrix_preparation_time_s"], "matrix_preparation_time"
        ),
        **percentile_summary(arrays["solve_time_s"], "solve_time"),
        **percentile_summary(arrays["control_compute_time_s"], "control_compute_time"),
        **goal_metadata,
    }
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path,
            **arrays,
            summary=np.array(json.dumps(summary), dtype=object),
        )
    return summary


def parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = (int(item) for item in part.split("-", 1))
            seeds.extend(range(start, stop + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    H = np.load(args.data_path)
    atlas = load_atlas(args.atlas_path)
    provenance = reference_code_provenance(args.select_code_zip)
    select_base = SelectDPCConfig(
        num_columns=args.select_columns,
        refinement_iterations=args.select_iterations,
        steps=args.steps,
    )
    atlas_base = RoutedDeePCConfig(steps=args.steps)

    results: list[dict] = []
    for seed in args.seeds:
        if "atlas" in args.methods:
            config = RoutedDeePCConfig(**{**asdict(atlas_base), "seed": seed})
            result_path = output_dir / f"atlas_seed{seed}_control_results.npz"
            summary = run_routed_deepc(H, atlas, config, save_path=result_path)
            summary.update(
                method="atlas_deepc",
                qp_count=args.steps,
                library_columns=int(H.shape[1]),
                optimized_coefficient_dimension=config.svd_rank,
                result_path=str(result_path),
            )
            results.append(summary)
            print(
                f"atlas seed={seed} success={summary['success_30'] and summary['success_100']} "
                f"median_total_ms={summary['median_control_compute_time_ms']:.2f}",
                flush=True,
            )
        if "select" in args.methods:
            config = SelectDPCConfig(**{**asdict(select_base), "seed": seed})
            result_path = output_dir / f"select_seed{seed}_control_results.npz"
            summary = run_select_dpc_matched(H, config, save_path=result_path)
            summary["result_path"] = str(result_path)
            results.append(summary)
            print(
                f"select seed={seed} success={summary['success_30'] and summary['success_100']} "
                f"median_total_ms={summary['median_control_compute_time_ms']:.2f}",
                flush=True,
            )

    payload = {
        "benchmark": "matched Reacher Select-DPC versus graph-routed Atlas-DeePC",
        "data_path": str(args.data_path),
        "data_sha256": sha256_file(args.data_path),
        "atlas_path": str(args.atlas_path),
        "select_dpc_reference": provenance,
        "shared_protocol": {
            "environment": pipe.REACHER_ENV_ID,
            "T_ini": pipe.T_INI,
            "T_f": pipe.T_F,
            "output_dimension": pipe.P,
            "input_dimension": pipe.M,
            "steps": args.steps,
            "solver": "CLARABEL",
            "target_and_goal_construction": "identical chart_compatible_goal call",
            "costs_constraints_and_regularization": "identical",
            "strict_success": (
                "final-30 and final-100 mean distance <1 cm and maximum distance <2 cm"
            ),
        },
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--atlas-path", type=Path, default=DEFAULT_ATLAS_PATH)
    parser.add_argument("--select-code-zip", type=Path, default=DEFAULT_SELECT_CODE_ZIP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--methods", nargs="+", choices=("atlas", "select"), default=["atlas", "select"])
    parser.add_argument("--seeds", type=parse_seeds, default=[1])
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--select-columns", type=int, default=100)
    parser.add_argument("--select-iterations", type=int, default=5)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
