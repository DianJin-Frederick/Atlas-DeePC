"""Paired noise study: remove non-position tracking weights, retaining full outputs."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

import measurement_noise_ablation as noise
from baseline_1000_benchmark import summarize_timing
from candidate_goal_routing_experiment import file_digest


VERSION = "position_only_tracking_q_noise_100_v1"
RESULT_ROOT = noise.ROOT / "experiments/routed_deepc_iid/position_only_tracking_q_noise_100_v1"
BASELINE_ROOT = noise.RESULT_ROOT
SEEDS, RATIOS = noise.SEEDS, noise.RATIOS


def experiment_config():
    return replace(noise.experiment_config(), q_data_goal_joint=0.0, q_dq=0.0)


def prepare_level(ratio):
    source = BASELINE_ROOT / noise.level_name(ratio)
    artifact = json.loads((source / "artifact_manifest.json").read_text())
    expected = {**noise.signature(), "sd_ratio": ratio, "power_nsr": ratio**2}
    if artifact["signature"] != expected:
        raise ValueError(f"Source noise experiment no longer matches current code/settings: {source}")
    for name, digest in artifact["artifact_sha256"].items():
        assert file_digest(source / name) == digest
    folder = RESULT_ROOT / noise.level_name(ratio)
    folder.mkdir(parents=True, exist_ok=True)
    signature = {
        "version": VERSION, "noise_sd_ratio": ratio,
        "config": {k: v for k, v in asdict(experiment_config()).items() if k != "seed"},
        "source_manifest_sha256": file_digest(source / "artifact_manifest.json"),
        "source_signature": artifact["signature"],
        "runner_sha256": file_digest(Path(__file__)),
        "reused_offline_artifacts": artifact["artifact_sha256"],
    }
    path = folder / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != signature:
            raise ValueError(f"Incompatible position-only Q results: {folder}")
    elif any(folder.glob("seed*_control_results.npz")):
        raise ValueError(f"Unversioned results: {folder}")
    else:
        path.write_text(json.dumps(signature, indent=2))
    return signature


def initialize_worker(ratio):
    # Reuse the exact cached noisy atlas and full-row bases. Only Q weights change.
    noise.initialize_worker(ratio, BASELINE_ROOT / noise.level_name(ratio))
    noise._controller.config = experiment_config()


def run_seed(seed, folder):
    row = noise.run_seed(seed, folder)
    assert row["q_data_goal_joint"] == row["q_dq"] == row["lam_g"] == 0
    row.update({"cost_ablation_version": VERSION, "tracking_cost": "position_only",
                "tracking_Q_diagonal": [0, 0, 0, 0, 0, 0, row["q_xy"], row["q_xy"], 0, 0],
                "augmented_output_dimension": noise.pipe.P, "nonposition_outputs_retained": True,
                "offline_atlas_and_bases_reused": True})
    path = Path(row["result_path"])
    with np.load(path, allow_pickle=True) as trace:
        arrays = {key: trace[key] for key in trace.files if key != "summary"}
    np.savez_compressed(path, **arrays, summary=np.array(json.dumps(row)))
    return row


def run_level(ratio, seeds=SEEDS, workers=4):
    signature = prepare_level(ratio)
    folder = RESULT_ROOT / noise.level_name(ratio)
    rows, pending = [], []
    for seed in seeds:
        path = folder / f"seed{seed}_control_results.npz"
        if path.exists():
            with np.load(path, allow_pickle=True) as trace:
                row = json.loads(trace["summary"].item())
            assert row["cost_ablation_version"] == VERSION and row["seed"] == seed
            assert row["steps"] == 320 and row["noise_sd_ratio"] == ratio
            assert row["q_data_goal_joint"] == row["q_dq"] == row["lam_g"] == 0
            rows.append(row)
        else:
            pending.append(seed)
    print(f"Position-only Q, noise SD {ratio:.1%}: {len(pending)} new; {len(rows)} cached", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker, initargs=(ratio,)) as pool:
            futures = [pool.submit(run_seed, seed, folder) for seed in pending]
            for future in as_completed(futures):
                rows.append(future.result())
                if len(rows) % 10 == 0 or len(rows) == len(seeds):
                    count = sum(r["success_30"] and r["success_100"] for r in rows)
                    print(f"Noise SD {ratio:.1%}: {len(rows)}/{len(seeds)} completed, {count} strict successes", flush=True)
    rows.sort(key=lambda r: r["seed"])
    assert [r["seed"] for r in rows] == sorted(seeds)
    payload = {"version": VERSION, "signature": signature, "seeds": list(seeds), "results": rows,
               "timing": summarize_timing([r["result_path"] for r in rows]),
               "workers": workers, "numerical_threads": 1,
               "timing_note": "Shared-computer wall clock, first five steps excluded; includes QP construction; excludes noise generation and simulator stepping"}
    (folder / "summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def run_all(ratios=RATIOS, seeds=SEEDS, workers=4):
    seeds = list(map(int, seeds))
    if len(seeds) != 100 or len(set(seeds)) != 100:
        raise ValueError("This paired ablation requires 100 distinct seeds")
    results = []
    for ratio in ratios:
        payload = run_level(float(ratio), seeds, workers)
        results.extend(payload["results"])
        (RESULT_ROOT / "summary.json").write_text(json.dumps({
            "version": VERSION, "ratios": list(ratios), "seeds": seeds, "results": results,
        }, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ratios", type=float, nargs="+", default=RATIOS)
    args = parser.parse_args()
    run_all(args.ratios, workers=args.workers)
