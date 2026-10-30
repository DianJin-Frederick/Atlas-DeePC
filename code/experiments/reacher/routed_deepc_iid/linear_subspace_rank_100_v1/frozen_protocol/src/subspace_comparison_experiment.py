"""Add only the uncentered Atlas variant to the saved comparison panel."""

import argparse
import contextlib
import gc
import json
import os
import platform
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

import numpy as np
from threadpoolctl import threadpool_limits

import routed_deepc_experiment as routed
from candidate_goal_routing_experiment import file_digest, strict_success
from four_method_serial_timing import STAGE_KEYS, rollout_stage_medians
from stitched_atlas_mpc_experiment import offline_environment_targets


ROOT = Path(__file__).resolve().parents[1]
METHOD = "Subspace Atlas-DeePC"
SOURCE = ROOT / "experiments/deepc_algorithm_comparison/six_way_svd_heldout111_1210_n200_rank60/summary.json"
DATA = ROOT / "data/select_iid_offline_hankel.npy"
ATLAS = ROOT / "experiments/routed_deepc_iid/clustering_jointconfig48_fresh211_310/behavioral_atlas.npz"
OUTPUT = ROOT / "experiments/deepc_algorithm_comparison/subspace_atlas_serial_211_1210"


def run_panel(seeds, output_dir=OUTPUT, source_path=SOURCE, data_path=DATA, atlas_path=ATLAS):
    seeds = [int(seed) for seed in seeds]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and unique")
    source = json.loads(Path(source_path).read_text())
    config = routed.RoutedDeePCConfig(**source["config"])
    if config.formulation != "svd_current" or config.target_matrix_mode != "current":
        raise ValueError("The comparison requires the current-chart baseline")
    baseline = {int(row["seed"]): row for row in source["method_summaries"]["atlas"]}
    if set(seeds) - set(baseline):
        raise ValueError("Requested seeds are missing from the saved comparison")
    signature = {
        "version": 1,
        "config": {k: v for k, v in asdict(config).items() if k != "seed"},
        "geometry": "linear",
        "reference_mode": "graph",
        "data_sha256": file_digest(data_path),
        "atlas_sha256": file_digest(atlas_path),
        "controller_sha256": file_digest(Path(routed.__file__)),
        "discard_initial_steps": 5,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["signature"] != signature:
            raise RuntimeError("Incompatible cached settings; choose a new output directory")
    else:
        manifest = {
            "signature": signature,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "python": sys.version,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

    with threadpool_limits(limits=1):
        H = np.load(data_path, mmap_mode="r")
        atlas = routed.load_atlas(atlas_path)
        offline_environment_targets(H.shape[1] // routed.WINDOWS_PER_EPISODE)
        cache = routed.precompute_chart_matrix_cache(H, atlas, config, geometry="linear")
        assert len(cache) == config.clusters
        assert all(np.count_nonzero(value[0]["offset"]) == 0 for value in cache.values())
        assert all(value[0]["basis"].shape[1] == config.svd_rank for value in cache.values())

        def run(seed, steps, path):
            gc.collect()
            with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
                return routed.run_routed_deepc(
                    H, atlas, replace(config, seed=seed, steps=steps), save_path=path,
                    matrix_cache=cache, reference_mode="graph",
                    summary_metadata={"matrix_geometry": "linear", "method": METHOD},
                )

        warmup = output_dir / "warmup.npz"
        run(210, 10, warmup)
        print(f"Warm-up complete; only {METHOD} will run ({len(seeds)} seeds).", flush=True)
        folder = output_dir / "rollouts"
        folder.mkdir(exist_ok=True)
        rows, records = [], []
        for index, seed in enumerate(seeds):
            checkpoint = folder / f"seed{seed}.json"
            if checkpoint.exists():
                saved = json.loads(checkpoint.read_text())
                if saved["signature"] != signature or saved["result"]["seed"] != seed:
                    raise RuntimeError(f"Incompatible checkpoint: {checkpoint}")
                row, timing = saved["result"], saved["timing"]
            else:
                path = folder / f"seed{seed}.npz"
                row = run(seed, config.steps, path)
                assert row["data_goal_index"] == baseline[seed]["data_goal_index"]
                timing = {
                    "seed": seed, "method": METHOD, "strict_success": strict_success(row),
                    **rollout_stage_medians(path, METHOD, discard=5),
                }
                with np.load(path, allow_pickle=True) as trace:
                    assert np.all(trace["goal_index"] == trace["goal_index"][0])
                    assert len(trace["u"]) == config.steps
                    row["unsolved_steps"] = int(sum(str(s) not in ("optimal", "optimal_inaccurate") for s in trace["status"]))
                    compact = {k: trace[k] for k in trace.files if k != "summary"}
                np.savez_compressed(path, **compact)
                saved = {"signature": signature, "result": row, "timing": timing}
                checkpoint.write_text(json.dumps(saved, indent=2))
            rows.append(row)
            records.append(timing)
            if (index + 1) % 10 == 0 or index + 1 == len(seeds):
                print(f"{index+1}/{len(seeds)}; successes={sum(strict_success(r) for r in rows)}; seed={seed}; tail100={1000*row['mean_last_100']:.3f} mm", flush=True)
        aggregate = {"method": METHOD, "rollouts": len(records)}
        aggregate.update({f"median {stage} (ms)": float(np.median([r[stage] for r in records])) for stage in STAGE_KEYS})
        payload = {
            "config": asdict(config), "signature": signature, "results": rows,
            "aggregate": [aggregate], "per_seed": records,
            "protocol": {
                "description": "new-method-only serial batch; original four methods retained without rerunning",
                "seeds": seeds, "steps": config.steps, "workers": 1, "numerical_threads": 1,
                "warmup_seed": 210, "warmup_steps": 10, "discard_initial_steps_per_rollout": 5,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "batch_started_utc": manifest["created_utc"],
                "timing_caveat": "Same timing boundary, but a separate measurement batch from the existing four-method benchmark.",
            },
        }
        (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps(aggregate, indent=2), flush=True)
        return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=211)
    parser.add_argument("--seed-stop", type=int, default=1210)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run_panel(range(args.seed_start, args.seed_stop + 1), output_dir=args.output_dir)
