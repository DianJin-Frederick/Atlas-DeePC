"""Shared Reacher controller with linear charts and N=10 terminal goals.

Experiment preparation and resumable evaluation live in the benchmark modules.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from atlas_deepc.common.paths import project_root
from atlas_deepc.atlas import routed_deepc_experiment as routed
from atlas_deepc.atlas.candidate_goal_routing_experiment import (
    CandidateGoalPolicy, FixedNearestGoalPolicy, shortest_path_tables,
)


ROOT = project_root()


DATA_PATH = ROOT / "data/select_iid_offline_hankel.npy"


BASELINE_VERSION = "linear_iterative_n10_lamg0_v2"


CANDIDATE_COUNT = 10


def default_config():
    # These legacy fields are retained in serialized configurations only.
    return routed.RoutedDeePCConfig(lam_g=0.0, lam_du=0.0)


class BaselineAtlasDeePC:
    """One frozen atlas/cache shared by all rollouts; goal state is per-rollout."""

    def __init__(self, H, atlas, config=None, *, terminal_reference_policy="iterative_minimum_route_cost"):
        if terminal_reference_policy not in ("iterative_minimum_route_cost", "fixed_nearest_eligible_fingertip"):
            raise ValueError("Unknown terminal reference policy")
        self.terminal_reference_policy = terminal_reference_policy
        self.config = default_config() if config is None else config
        if self.config.formulation != "svd_current" or self.config.target_matrix_mode != "current":
            raise ValueError("The baseline uses only the current chart's SVD matrix")
        self.H, self.atlas = H, atlas
        self.matrix_cache = routed.precompute_chart_matrix_cache(H, atlas, self.config, geometry="linear")
        self.route_tables = (
            shortest_path_tables(atlas, self.config.edge_count_threshold)
            if terminal_reference_policy == "iterative_minimum_route_cost" else None
        )
        for matrix, _ in self.matrix_cache.values():
            assert not np.any(matrix["offset"]), "Baseline chart offset must be zero"

    def run(self, seed, save_path=None, reference_mode="graph", *, chart_selector=None,
            localization_history_length=None):
        if reference_mode not in ("graph", "terminal"):
            raise ValueError("reference_mode must be graph or terminal")
        config = replace(self.config, seed=int(seed))
        fixed_nearest = self.terminal_reference_policy == "fixed_nearest_eligible_fingertip"
        if fixed_nearest and reference_mode == "graph" and self.route_tables is None:
            self.route_tables = shortest_path_tables(self.atlas, config.edge_count_threshold)
        policy = (FixedNearestGoalPolicy(
            self.H, self.atlas, config,
            self.route_tables if reference_mode == "graph" else None) if fixed_nearest else
                  CandidateGoalPolicy(self.H, self.atlas, config, CANDIDATE_COUNT, self.route_tables))
        row = routed.run_routed_deepc(
            self.H, self.atlas, config, save_path=save_path,
            matrix_cache=self.matrix_cache, goal_policy=policy, reference_mode=reference_mode,
            chart_selector=chart_selector, localization_history_length=localization_history_length,
            summary_metadata={"baseline_version": BASELINE_VERSION, "matrix_geometry": "linear",
                              "candidate_count": 1 if fixed_nearest else CANDIDATE_COUNT,
                              "terminal_reference_policy": self.terminal_reference_policy,
                              "dynamic_terminal_reference": not fixed_nearest,
                              "graph_route_cost_selection_enabled": not fixed_nearest,
                              "graph_waypoints_enabled": reference_mode == "graph"},
        )
        if save_path is not None:
            path = Path(save_path)
            with np.load(path, allow_pickle=True) as trace:
                indices = trace["goal_index"].astype(int)
                targets = policy.source_targets[indices // routed.WINDOWS_PER_EPISODE]
                row["minimum_selected_goal_source_separation"] = float(
                    np.min(np.linalg.norm(targets - policy.target, axis=1)))
                row["unsolved_steps"] = int(sum(str(s) not in ("optimal", "optimal_inaccurate") for s in trace["status"]))
                arrays = {k: trace[k] for k in trace.files if k != "summary"}
            assert row["minimum_selected_goal_source_separation"] >= config.goal_source_target_min_distance
            np.savez_compressed(path, **arrays, summary=np.array(json.dumps(row)))
        return row
