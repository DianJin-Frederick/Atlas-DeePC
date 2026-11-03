"""Terminal-goal selection and cached shortest paths for Atlas-DeePC."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from atlas_deepc.control import expert_pipeline_search as pipe
from atlas_deepc.atlas import routed_deepc_experiment as routed
from atlas_deepc.common.reacher_data import DEFAULT_DATA_SEED
from atlas_deepc.atlas.goal_reference import offline_environment_targets


def shortest_path_tables(atlas, threshold):
    """Cache exact baseline directed graph costs once, before any rollout."""
    counts = atlas["transition_counts"]
    rows, cols = np.where((counts >= threshold) & ~np.eye(len(counts), dtype=bool))
    costs = atlas["center_distance"][rows, cols] / np.sqrt(counts[rows, cols])
    if "routing_edge_costs" in atlas:
        costs = atlas["routing_edge_costs"][rows, cols]
    valid = np.isfinite(costs)
    graph = csr_matrix((costs[valid], (rows[valid], cols[valid])), shape=counts.shape)
    distances, predecessors = dijkstra(graph, directed=True, return_predecessors=True)
    return distances, predecessors


class CandidateGoalPolicy:
    def __init__(self, H, atlas, config, candidate_count, tables):
        self.atlas = atlas
        self.config = config
        self.candidate_count = int(candidate_count)
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        # Labels are fitted to the latest Y_p configuration, not the first Y_p row.
        self.outputs = H[pipe.M * pipe.L:].reshape(pipe.L, pipe.P, H.shape[1])[pipe.T_INI - 1].T
        self.distances, self.predecessors = tables
        windows = routed.WINDOWS_PER_EPISODE
        if H.shape[1] % windows:
            raise ValueError("The offline library must contain complete episodes")
        self.source_targets = offline_environment_targets(H.shape[1] // windows)

    def prepare(self, target):
        self.target = np.asarray(target).copy()
        separation = np.linalg.norm(self.source_targets - target, axis=1)
        eligible = np.flatnonzero(np.repeat(
            separation >= self.config.goal_source_target_min_distance,
            routed.WINDOWS_PER_EPISODE,
        ))
        if len(eligible) < self.candidate_count:
            raise ValueError("Requested N exceeds eligible offline configurations")
        errors = np.linalg.norm(self.outputs[eligible, 6:8] - target, axis=1)
        # Exact top-N, with deterministic offline-index tie breaking.
        order = np.lexsort((eligible, errors))[:self.candidate_count]
        candidates = eligible[order]
        self.candidates = candidates
        self.candidate_errors = errors[order]
        labels = self.atlas["labels"][candidates].astype(int)
        # Within a chart all route costs are identical; keep the closest fingertip.
        _, first = np.unique(labels, return_index=True)
        first.sort()
        self.indices = candidates[first]
        self.charts = labels[first]
        self.xy_errors = self.candidate_errors[first]
        self.references = self.outputs[self.indices].copy()
        self.references[:, 4:6] = 0.0
        self.references[:, 6:8] = target
        self.separations = separation[self.indices // routed.WINDOWS_PER_EPISODE]

    def select(self, current_chart):
        costs = self.distances[current_chart, self.charts]
        if not np.any(np.isfinite(costs)):
            raise RuntimeError(f"No candidate chart reachable from chart {current_chart}")
        # Indices are already ordered by fingertip error, then source index.
        best = int(np.argmin(costs))
        index = int(self.indices[best])
        metadata = {
            "data_goal_index": index,
            "data_goal_xy_error": float(self.xy_errors[best]),
            "data_goal_source_seed": int(DEFAULT_DATA_SEED + index // routed.WINDOWS_PER_EPISODE),
            "data_goal_source_target_distance": float(self.separations[best]),
            "data_goal_chart": int(self.charts[best]),
            "data_goal_chart_count": int(len(self.charts)),
        }
        return self.references[best], metadata, int(self.charts[best]), float(costs[best])

    def path(self, source, destination):
        if not np.isfinite(self.distances[source, destination]):
            return []
        path = [int(destination)]
        while path[-1] != source:
            parent = int(self.predecessors[source, path[-1]])
            if parent < 0 or len(path) > len(self.atlas["centers"]):
                raise RuntimeError("Invalid shortest-path predecessor table")
            path.append(parent)
        return path[::-1]


class FixedNearestGoalPolicy(CandidateGoalPolicy):
    """One nearest eligible posture; optional routing never reselects the goal."""

    def __init__(self, H, atlas, config, tables=None):
        super().__init__(H, atlas, config, candidate_count=1,
                         tables=(None, None) if tables is None else tables)

    def prepare(self, target):
        super().prepare(target)
        self.fixed_selection = (
            self.references[0].copy(),
            {
                "data_goal_index": int(self.indices[0]),
                "data_goal_xy_error": float(self.xy_errors[0]),
                "data_goal_source_seed": int(DEFAULT_DATA_SEED + self.indices[0] // routed.WINDOWS_PER_EPISODE),
                "data_goal_source_target_distance": float(self.separations[0]),
                "data_goal_chart": int(self.charts[0]),
                "data_goal_chart_count": 1,
            },
            int(self.charts[0]),
            np.nan,  # No route is evaluated, so there is no route cost.
        )

    def select(self, current_chart):
        if self.distances is not None:
            goal, metadata, chart, _ = self.fixed_selection
            return goal, metadata, chart, float(self.distances[current_chart, chart])
        return self.fixed_selection

    def path(self, source, destination):
        if self.distances is None:
            raise RuntimeError("Fixed nearest-reference control does not use graph paths")
        return super().path(source, destination)
