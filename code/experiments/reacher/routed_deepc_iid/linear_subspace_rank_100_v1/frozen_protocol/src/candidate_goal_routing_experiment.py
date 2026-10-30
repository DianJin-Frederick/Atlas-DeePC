"""Route-cost goal selection for the unchanged current-chart Atlas-DeePC QP."""

import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from threadpoolctl import threadpool_limits

import expert_pipeline_search as pipe
import routed_deepc_experiment as routed
from local_affine_atlas_experiment import DEFAULT_DATA_SEED
from stitched_atlas_mpc_experiment import offline_environment_targets


VERSION = 2
TIMING_KEYS = (
    "localization_time_s", "goal_selection_time_s", "routing_time_s",
    "matrix_preparation_time_s", "solve_time_s", "control_compute_time_s",
)


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


def strict_success(row):
    return bool(row["success_30"] and row["success_100"])


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_panel(seeds, candidate_counts, config, data_path, atlas_path, output_dir,
              overwrite=False, progress_every=10):
    """Serial paired runs for comparable local timing, with per-rollout checkpoints."""
    seeds = [int(seed) for seed in seeds]
    candidate_counts = [int(n) for n in candidate_counts]
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Seeds must be nonempty and unique")
    if not candidate_counts or min(candidate_counts) < 1 or len(set(candidate_counts)) != len(candidate_counts):
        raise ValueError("Candidate counts must be distinct positive integers")
    if config.formulation != "svd_current" or config.target_matrix_mode != "current":
        raise ValueError("This ablation requires baseline current-chart DeePC")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = {
        "version": VERSION,
        "controller_config": {k: v for k, v in asdict(config).items() if k != "seed"},
        "data_sha256": file_digest(data_path), "atlas_sha256": file_digest(atlas_path),
        "timing_protocol": "serial_rollouts_one_blas_thread",
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["signature"] != signature:
        raise RuntimeError("Experiment settings changed; choose a new output directory")
    with threadpool_limits(limits=1):
        started = time.perf_counter()
        H = np.load(data_path, mmap_mode="r")
        atlas = routed.load_atlas(atlas_path)
        metadata_start = time.perf_counter()
        offline_environment_targets(H.shape[1] // routed.WINDOWS_PER_EPISODE)
        metadata_time = time.perf_counter() - metadata_start
        cache_start = time.perf_counter()
        matrix_cache = routed.precompute_chart_matrix_cache(H, atlas, config)
        svd_time = time.perf_counter() - cache_start
        graph_start = time.perf_counter()
        tables = shortest_path_tables(atlas, config.edge_count_threshold)
        graph_time = time.perf_counter() - graph_start
        preparation_time = time.perf_counter() - started
        manifest = {
            "signature": signature, "seeds": seeds, "candidate_counts": candidate_counts,
            "shared_offline_preparation_s": preparation_time,
            "shared_chart_svd_s": svd_time, "shared_all_pairs_routing_s": graph_time,
            "shared_source_metadata_s": metadata_time,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        jobs = [("Baseline Atlas", None)] + [(f"Candidate N={n}", n) for n in candidate_counts]
        rows = []
        total = len(seeds) * len(jobs)
        finished = 0
        for position, seed in enumerate(seeds):
            # Rotate order across seeds to reduce systematic warm-up/order effects.
            offset = position % len(jobs)
            for method, n in jobs[offset:] + jobs[:offset]:
                directory = output_dir / ("baseline" if n is None else f"candidate_{n}")
                directory.mkdir(exist_ok=True)
                checkpoint = directory / f"seed{seed}.json"
                if checkpoint.exists() and not overwrite:
                    row = json.loads(checkpoint.read_text())
                    if row.get("signature") != signature or row["seed"] != seed or row["candidate_count"] != n:
                        raise RuntimeError(f"Incompatible checkpoint: {checkpoint}")
                else:
                    cfg = replace(config, seed=seed)
                    policy = None if n is None else CandidateGoalPolicy(H, atlas, cfg, n, tables)
                    trace_path = directory / f"seed{seed}.npz"
                    row = routed.run_routed_deepc(
                        H, atlas, cfg, save_path=trace_path,
                        matrix_cache=matrix_cache, goal_policy=policy,
                    )
                    with np.load(trace_path, allow_pickle=True) as trace:
                        for key in TIMING_KEYS:
                            values = np.asarray(trace[key], dtype=float) * 1000
                            row[f"mean_{key[:-1]}ms"] = float(values.mean())
                            row[f"p95_{key[:-1]}ms"] = float(np.percentile(values, 95))
                        accounted = sum(trace[key] for key in TIMING_KEYS[:-1])
                        row["mean_other_time_ms"] = float(1000 * np.mean(trace["control_compute_time_s"] - accounted))
                        row["goal_switches_after_first_reach"] = 0
                        near = np.flatnonzero(trace["distance"] < 0.01)
                        if near.size:
                            row["goal_switches_after_first_reach"] = int(np.count_nonzero(np.diff(trace["goal_index"][near[0]:])))
                        row["minimum_selected_goal_source_separation"] = row["data_goal_source_target_distance"]
                        if policy is not None:
                            indices = np.asarray(trace["goal_index"], dtype=int)
                            source_targets = policy.source_targets[indices // routed.WINDOWS_PER_EPISODE]
                            row["minimum_selected_goal_source_separation"] = float(np.min(np.linalg.norm(source_targets - policy.target, axis=1)))
                            row["candidate_unique_charts"] = int(len(policy.charts))
                            row["candidate_max_xy_error_mm"] = float(1000 * policy.candidate_errors.max())
                        # Retain numerical traces without video or pickle-only metadata.
                        compact = {key: trace[key] for key in trace.files if key != "summary"}
                    np.savez_compressed(trace_path, **compact)
                    row.update(method=method, candidate_count=n, signature=signature)
                    checkpoint.write_text(json.dumps(row, indent=2))
                rows.append(row)
                finished += 1
                if finished % progress_every == 0 or finished == total:
                    print(f"{finished}/{total}: seed {seed}, {method}, strict={strict_success(row)}, tail100={1000 * row['mean_last_100']:.3f} mm", flush=True)
        payload = {"manifest": manifest, "results": rows}
        (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    return payload
