"""Repeat the coefficient-bound ablation at selected ranks, preserving rank 60."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import argparse
from dataclasses import asdict, replace
import json
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.ablations import ablation_coefficient_bounds as experiment
HISTORICAL = experiment.ROOT / "experiments/routed_deepc_iid/selected_ranks_1000_clarabel_v1"
OUTPUT = experiment.ROOT / "experiments/routed_deepc_iid/coefficient_bounds_removed_rank59_62_311_410_v1"


def metrics(rows):
    errors = np.array([r["mean_last_100"] for r in rows]) * 1000
    return {
        "successes": sum(experiment.successful(r) for r in rows),
        "median_final100_mean_error_mm": float(np.median(errors)),
        "mean_final100_mean_error_mm": float(errors.mean()),
        "p90_final100_mean_error_mm": float(np.percentile(errors, 90)),
        "unsolved_steps": sum(r["unsolved_steps"] for r in rows),
    }


def run_rank(rank):
    base_config = experiment.comparison.default_config()
    config = replace(base_config, svd_rank=rank)
    historical_manifest = json.loads((HISTORICAL / "manifest.json").read_text())
    assert historical_manifest["base_config"] == asdict(base_config)
    assert historical_manifest["solver"] == "CLARABEL"
    rows = []
    for seed in experiment.SEEDS:
        with np.load(HISTORICAL / f"rank{rank:03}" / f"seed{seed}.npz", allow_pickle=True) as trace:
            row = json.loads(trace["summary"].item())
            assert row["seed"] == seed and row["reference_mode"] == "graph"
            assert row["candidate_count"] == 10 and row["matrix_geometry"] == "linear"
            for key, value in asdict(config).items():
                if key != "seed":
                    assert row[key] == value, key
            for n in (30, 100):
                tail = trace["distance"][-n:]
                assert row[f"success_{n}"] == bool(tail.mean() < .01 and tail.max() < .02)
            rows.append(row)
    output = OUTPUT / f"rank{rank:03}"
    output.mkdir(parents=True, exist_ok=True)
    reference = {
        "signature": {"config": {k: v for k, v in asdict(config).items() if k != "seed"},
                      "solver": "CLARABEL", **{key: historical_manifest[key] for key in
                      ("data_sha256", "atlas_sha256")}},
        "source": str(HISTORICAL), "results": rows,
    }
    reference_path = output / "bounded_reference.json"
    reference_path.write_text(json.dumps(reference, indent=2))
    provenance = {"rank_runner_sha256": experiment.file_digest(__file__),
                  "historical_manifest_sha256": experiment.file_digest(HISTORICAL / "manifest.json"),
                  "rank": rank, "seeds": experiment.SEEDS}
    provenance_path = output / "rank_provenance.json"
    if provenance_path.exists():
        assert json.loads(provenance_path.read_text()) == provenance
    else:
        provenance_path.write_text(json.dumps(provenance, indent=2))

    # Confirm that the current bounded implementation reproduces the archived rank.
    verification_path = output / "bounded_replay_verification.json"
    if not verification_path.exists():
        with threadpool_limits(limits=1):
            H = np.load(experiment.baseline.DATA_PATH, mmap_mode="r")
            atlas = experiment.routed.load_atlas(experiment.baseline.ATLAS_PATH)
            controller = experiment.baseline.BaselineAtlasDeePC(H, atlas, config)
            path = output / "verification_bounded_seed311.npz"
            replay = controller.run(311, path)
        with np.load(path, allow_pickle=True) as new, np.load(
            HISTORICAL / f"rank{rank:03}" / "seed311.npz", allow_pickle=True
        ) as old:
            difference = float(np.max(np.abs(new["distance"] - old["distance"])))
            np.testing.assert_allclose(new["distance"], old["distance"], atol=1e-7, rtol=1e-7)
        verification_path.write_text(json.dumps({"seed": 311, "rank": rank,
            "maximum_distance_difference_m": difference, "success": experiment.successful(replay)}, indent=2))
    print(f"Rank {rank}: bounded baseline {metrics(rows)['successes']}/100; starting no-bound panel", flush=True)
    with patch.object(experiment, "OUTPUT", output), patch.object(experiment, "BASELINE", reference_path),\
         patch.object(experiment.comparison, "default_config", return_value=config):
        experiment.main()
    result = json.loads((output / "summary.json").read_text())
    assert result["completed"] == 100
    for row in result["results"]:
        with np.load(row["result_path"], allow_pickle=True) as trace:
            assert trace["g_coefficients"].shape == (300, rank)
            target = trace["final_goal"].reshape(-1)[6:8]
            np.testing.assert_allclose(trace["distance"], np.linalg.norm(trace["y"][:, 6:8] - target, axis=1))
            reference_row = rows[row["seed"] - experiment.SEEDS[0]]
            np.testing.assert_array_equal(target, reference_row["exact_target_xy"])
            for n in (30, 100):
                tail = trace["distance"][-n:]
                np.testing.assert_allclose(tail.mean(), row[f"mean_last_{n}"])
                assert row[f"success_{n}"] == bool(tail.mean() < .01 and tail.max() < .02)
    analysis = {"rank": rank, "seeds": experiment.SEEDS, "bounded": metrics(rows),
                "unbounded": metrics(result["results"]), "repaired_seeds": result["repaired_seeds"],
                "regressed_seeds": result["regressed_seeds"], "all_traces_validated": True}
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2))
    print(json.dumps(analysis), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, nargs="+", choices=[59, 62], default=[59, 62])
    args = parser.parse_args()
    for rank in args.ranks:
        run_rank(rank)
