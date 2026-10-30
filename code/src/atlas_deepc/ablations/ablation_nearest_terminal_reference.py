"""Paired nearest-fingertip versus iterative route-cost terminal reference test."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from atlas_deepc.common.paths import source_hashes, source_path

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

import numpy as np

from atlas_deepc.baselines import baseline_atlas_deepc as baseline
from atlas_deepc.atlas import routed_deepc_experiment as routed
from atlas_deepc.atlas.candidate_goal_routing_experiment import CandidateGoalPolicy, file_digest, strict_success
from atlas_deepc.reporting.baseline_success_rate import validate_panel


OUTPUT_DIR = baseline.ROOT / "experiments/routed_deepc_iid/ablation_nearest_terminal_linear_v1"
BASELINE_DIR = baseline.RESULT_ROOT / "evaluation_1000"
METHOD = "linear_fixed_nearest_eligible_terminal_v1"
_controller = None


def initialize():
    global _controller
    H = np.load(baseline.DATA_PATH, mmap_mode="r")
    atlas = routed.load_atlas(baseline.ATLAS_PATH)
    _controller = baseline.BaselineAtlasDeePC(H, atlas)


def run_seed(seed, output_dir):
    controller = _controller
    config = replace(controller.config, seed=int(seed))
    # N=1 reduces the existing policy to the single nearest eligible fingertip.
    # Its reference is fixed, while the route from the current chart still updates.
    policy = CandidateGoalPolicy(controller.H, controller.atlas, config, 1, controller.route_tables)
    path = Path(output_dir) / f"seed{seed}_control_results.npz"
    row = routed.run_routed_deepc(
        controller.H, controller.atlas, config, save_path=path,
        matrix_cache=controller.matrix_cache, goal_policy=policy, reference_mode="graph",
        summary_metadata={"ablation_version": METHOD, "matrix_geometry": "linear",
                          "candidate_count": 1, "terminal_reference_policy": "fixed_nearest_eligible_fingertip"},
    )
    with np.load(path, allow_pickle=True) as trace:
        indices = trace["goal_index"].astype(int)
        assert len(np.unique(indices)) == 1 and indices[0] == policy.candidates[0]
        targets = policy.source_targets[indices // routed.WINDOWS_PER_EPISODE]
        row["minimum_selected_goal_source_separation"] = float(np.min(np.linalg.norm(targets-policy.target, axis=1)))
        row["unsolved_steps"] = sum(str(s) not in ("optimal", "optimal_inaccurate") for s in trace["status"])
        row["dynamic_terminal_reference"] = False
        arrays = {k: trace[k] for k in trace.files if k != "summary"}
    assert row["minimum_selected_goal_source_separation"] >= config.goal_source_target_min_distance
    np.savez_compressed(path, **arrays, summary=np.array(json.dumps(row)))
    row["result_path"] = str(path.resolve())
    return row


def compare(rows, baseline_rows, output_dir):
    old = {int(row["seed"]): row for row in baseline_rows}
    paired = []
    for row in rows:
        seed = row["seed"]
        with np.load(BASELINE_DIR / f"seed{seed}_control_results.npz", allow_pickle=True) as trace:
            old_indices = trace["goal_index"].astype(int)
        with np.load(row["result_path"], allow_pickle=True) as trace:
            nearest_index = int(trace["goal_index"][0])
        paired.append({"seed": seed, "baseline_success": strict_success(old[seed]),
                       "nearest_success": strict_success(row),
                       "baseline_final100_mm": 1000*old[seed]["mean_last_100"],
                       "nearest_final100_mm": 1000*row["mean_last_100"],
                       "same_initial_reference": bool(old_indices[0] == nearest_index),
                       "same_reference_all_steps": bool(np.all(old_indices == nearest_index))})
    report = {
        "seeds": [row["seed"] for row in rows],
        "baseline_successes": sum(p["baseline_success"] for p in paired),
        "nearest_successes": sum(p["nearest_success"] for p in paired),
        "repaired_seeds": [p["seed"] for p in paired if p["nearest_success"] and not p["baseline_success"]],
        "regressed_seeds": [p["seed"] for p in paired if p["baseline_success"] and not p["nearest_success"]],
        "same_initial_reference_count": sum(p["same_initial_reference"] for p in paired),
        "same_reference_all_steps_count": sum(p["same_reference_all_steps"] for p in paired),
        "nearest_unsolved_steps": sum(row["unsolved_steps"] for row in rows),
        "paired_results": paired,
        "timing_note": "New rollouts run concurrently; timings are not a controlled comparison to the saved serial baseline.",
    }
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2))
    compact = {k: v for k, v in report.items() if k not in ("seeds", "paired_results")}
    print(json.dumps(compact, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=211)
    parser.add_argument("--seed-stop", type=int, default=310)
    args = parser.parse_args()
    seeds = list(range(args.seed_start, args.seed_stop+1))
    config = baseline.default_config()
    payload = json.loads((BASELINE_DIR / "summary.json").read_text())
    baseline_rows, missing = validate_panel(payload, seeds, config)
    assert not missing
    signature = {
        "method": METHOD, "config": {k: v for k, v in asdict(config).items() if k != "seed"},
        "data_sha256": file_digest(baseline.DATA_PATH), "atlas_sha256": file_digest(baseline.ATLAS_PATH),
        "source_sha256": source_hashes(payload["signature"]["source_sha256"]),
    }
    for key in ("config", "data_sha256", "atlas_sha256", "source_sha256"):
        assert signature[key] == payload["signature"][key], f"Baseline mismatch: {key}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = OUTPUT_DIR / "manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == signature, "Incompatible cached ablation"
    elif any(OUTPUT_DIR.glob("seed*_control_results.npz")):
        raise ValueError("Unversioned ablation cache")
    manifest.write_text(json.dumps(signature, indent=2))
    rows, pending = [], []
    for seed in seeds:
        path = OUTPUT_DIR / f"seed{seed}_control_results.npz"
        if path.exists():
            with np.load(path, allow_pickle=True) as trace:
                row = json.loads(trace["summary"].item())
            assert row["seed"] == seed and row["ablation_version"] == METHOD
            row["result_path"] = str(path.resolve())
            rows.append(row)
        else:
            pending.append(seed)
    print(f"Reusing {len(baseline_rows)} matching baseline results; running {len(pending)} nearest-reference seeds.", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as executor:
        futures = [executor.submit(run_seed, seed, OUTPUT_DIR) for seed in pending]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{len(rows)}/{len(seeds)} seed {row['seed']}: success={strict_success(row)}; final100={1000*row['mean_last_100']:.3f} mm", flush=True)
    rows.sort(key=lambda row: row["seed"])
    assert [r["seed"] for r in rows] == seeds
    (OUTPUT_DIR / "summary.json").write_text(json.dumps({"method": METHOD, "signature": signature,
                                                        "results": rows, "workers": args.workers}, indent=2))
    compare(rows, baseline_rows, OUTPUT_DIR)


if __name__ == "__main__":
    main()
