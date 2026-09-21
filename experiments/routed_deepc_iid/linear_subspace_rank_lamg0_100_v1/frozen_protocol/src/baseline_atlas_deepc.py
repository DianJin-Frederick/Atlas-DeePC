"""Current Reacher baseline: linear charts and iterative N=10 terminal goals.

Use this entry point for new baseline runs. The lower-level routed experiment
retains its historical defaults so existing ablations remain reproducible.
"""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import routed_deepc_experiment as routed
from candidate_goal_routing_experiment import CandidateGoalPolicy, file_digest, shortest_path_tables


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/select_iid_offline_hankel.npy"
ATLAS_PATH = ROOT / "experiments/routed_deepc_iid/clustering_jointconfig48_fresh211_310/behavioral_atlas.npz"
RESULT_ROOT = ROOT / "experiments/routed_deepc_iid/baseline_linear_iterative_n10_lamg0_v2"
BASELINE_VERSION = "linear_iterative_n10_lamg0_v2"
CANDIDATE_COUNT = 10


def default_config():
    return routed.RoutedDeePCConfig(lam_g=0.0)


class BaselineAtlasDeePC:
    """One frozen atlas/cache shared by all rollouts; goal state is per-rollout."""

    def __init__(self, H, atlas, config=None):
        self.config = default_config() if config is None else config
        if self.config.formulation != "svd_current" or self.config.target_matrix_mode != "current":
            raise ValueError("The baseline uses only the current chart's SVD matrix")
        self.H, self.atlas = H, atlas
        self.matrix_cache = routed.precompute_chart_matrix_cache(H, atlas, self.config, geometry="linear")
        self.route_tables = shortest_path_tables(atlas, self.config.edge_count_threshold)
        for matrix, _ in self.matrix_cache.values():
            assert not np.any(matrix["offset"]), "Baseline chart offset must be zero"

    def run(self, seed, save_path=None):
        config = replace(self.config, seed=int(seed))
        policy = CandidateGoalPolicy(self.H, self.atlas, config, CANDIDATE_COUNT, self.route_tables)
        row = routed.run_routed_deepc(
            self.H, self.atlas, config, save_path=save_path,
            matrix_cache=self.matrix_cache, goal_policy=policy, reference_mode="graph",
            summary_metadata={"baseline_version": BASELINE_VERSION, "matrix_geometry": "linear",
                              "candidate_count": CANDIDATE_COUNT,
                              "terminal_reference_policy": "iterative_minimum_route_cost"},
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


def run_panel(seeds, output_dir=RESULT_ROOT / "evaluation", config=None,
              data_path=DATA_PATH, atlas_path=ATLAS_PATH, overwrite=False):
    seeds = list(map(int, seeds))
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and unique")
    config = default_config() if config is None else config
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = {
        "baseline_version": BASELINE_VERSION, "matrix_geometry": "linear", "candidate_count": CANDIDATE_COUNT,
        "config": {k: v for k, v in asdict(config).items() if k != "seed"},
        "data_sha256": file_digest(data_path), "atlas_sha256": file_digest(atlas_path),
        "runner_sha256": file_digest(Path(routed.__file__)),
        "policy_sha256": file_digest(Path(__file__).with_name("candidate_goal_routing_experiment.py")),
    }
    manifest = output_dir / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != signature:
            raise ValueError("Incompatible baseline cache; choose a new output directory")
    elif any(output_dir.glob("seed*_control_results.npz")):
        raise ValueError("Unversioned rollout cache is not a current-baseline result")
    manifest.write_text(json.dumps(signature, indent=2))
    with threadpool_limits(limits=1):
        H = np.load(data_path, mmap_mode="r")
        atlas = routed.load_atlas(atlas_path)
        controller = BaselineAtlasDeePC(H, atlas, config)
        rows = []
        for seed in seeds:
            path = output_dir / f"seed{seed}_control_results.npz"
            if path.exists() and not overwrite:
                with np.load(path, allow_pickle=True) as trace:
                    row = json.loads(trace["summary"].item())
                assert row["baseline_version"] == BASELINE_VERSION and row["seed"] == seed
                assert row["matrix_geometry"] == "linear" and row["candidate_count"] == CANDIDATE_COUNT
            else:
                row = controller.run(seed, path)
            row["result_path"] = str(path.resolve())
            rows.append(row)
            print(f"seed {seed}: success={row['success_30'] and row['success_100']}, final100={1000*row['mean_last_100']:.3f} mm", flush=True)
        payload = {"method": BASELINE_VERSION, "config": asdict(config), "signature": signature,
                   "fairness_audit": routed.audit_iid_library(H, eval_seeds=seeds),
                   "results": rows}
        (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
        return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=211)
    parser.add_argument("--seed-stop", type=int, default=710)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=RESULT_ROOT / "evaluation")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_panel(args.seeds or range(args.seed_start, args.seed_stop + 1), args.output_dir, overwrite=args.overwrite)
