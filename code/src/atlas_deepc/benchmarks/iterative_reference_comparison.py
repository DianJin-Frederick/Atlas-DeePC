"""Extend the user's iterative terminal augmentation to the comparison seeds."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import argparse
import contextlib
import gc
import json
import os
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.atlas import candidate_goal_routing_experiment as candidate
from atlas_deepc.atlas import routed_deepc_experiment as routed
from atlas_deepc.benchmarks.four_method_serial_timing import STAGE_KEYS, rollout_stage_medians
from atlas_deepc.benchmarks.subspace_comparison_experiment import ATLAS, DATA, ROOT, SOURCE


METHOD = "Iterative-reference Atlas-DeePC"
ARCHIVE = ROOT / "experiments/routed_deepc_iid/candidate_goal_routing_500"
OUTPUT = ROOT / "experiments/deepc_algorithm_comparison/iterative_reference_n10_211_1210"


def timing_record(path, row, batch):
    timing = rollout_stage_medians(path, METHOD, discard=5)
    with np.load(path, allow_pickle=True) as trace:
        # The shared four-method timing helper has no dynamic-goal selection stage.
        timing["data / prototype selection"] = float(1000 * np.median(trace["goal_selection_time_s"][5:]))
        switches = int(np.count_nonzero(np.diff(trace["goal_index"])))
        assert switches == row["goal_sample_switches"]
        assert len(trace["u"]) == row["steps"]
    return {"seed": int(row["seed"]), "method": METHOD,
            "strict_success": candidate.strict_success(row), "timing_batch": batch,
            "goal setup (ms)": row["goal_setup_time_ms"], **timing}


def run_panel(seeds, output_dir=OUTPUT, candidate_count=10):
    seeds = list(map(int, seeds))
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and unique")
    config = routed.RoutedDeePCConfig(**json.loads(SOURCE.read_text())["config"])
    controller_config = {k: v for k, v in asdict(config).items() if k != "seed"}
    archive_manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    archive_signature = archive_manifest["signature"]
    assert archive_signature["controller_config"] == controller_config
    assert archive_signature["version"] == candidate.VERSION
    assert archive_signature["data_sha256"] == candidate.file_digest(DATA)
    assert archive_signature["atlas_sha256"] == candidate.file_digest(ATLAS)
    signature = {
        "version": 1, "candidate_count": candidate_count, "geometry": "affine",
        "archive_signature": archive_signature,
        "controller_sha256": candidate.file_digest(Path(routed.__file__)),
        "policy_sha256": candidate.file_digest(Path(candidate.__file__)),
        "discard_initial_steps": 5,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())["signature"] == signature, "Incompatible saved settings"
    else:
        manifest_path.write_text(json.dumps({"signature": signature, "created_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    rows, records = [], []
    folder = output_dir / "rollouts"
    folder.mkdir(exist_ok=True)
    pending = []
    for seed in seeds:
        archived = ARCHIVE / f"candidate_{candidate_count}" / f"seed{seed}.json"
        checkpoint = folder / f"seed{seed}.json"
        if archived.exists():
            row = json.loads(archived.read_text())
            assert row["signature"] == archive_signature
            assert row["candidate_count"] == candidate_count
            timing = timing_record(archived.with_suffix(".npz"), row, "archived candidate ablation")
        elif checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            assert saved["signature"] == signature
            row, timing = saved["result"], saved["timing"]
        else:
            pending.append(seed)
            continue
        assert row["seed"] == seed
        rows.append(row)
        records.append(timing)
    print(f"Reusing {len(rows)} seeds; running only iterative reference on {len(pending)} missing seeds.", flush=True)
    with threadpool_limits(limits=1):
        if pending:
            H = np.load(DATA, mmap_mode="r")
            atlas = routed.load_atlas(ATLAS)
            cache = routed.precompute_chart_matrix_cache(H, atlas, config, geometry="affine")
            tables = candidate.shortest_path_tables(atlas, config.edge_count_threshold)

            def run(seed, steps, path):
                cfg = replace(config, seed=seed, steps=steps)
                policy = candidate.CandidateGoalPolicy(H, atlas, cfg, candidate_count, tables)
                gc.collect()
                with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
                    row = routed.run_routed_deepc(H, atlas, cfg, save_path=path,
                        matrix_cache=cache, goal_policy=policy, reference_mode="graph")
                with np.load(path, allow_pickle=True) as trace:
                    indices = trace["goal_index"].astype(int)
                    source_targets = policy.source_targets[indices // routed.WINDOWS_PER_EPISODE]
                    row["minimum_selected_goal_source_separation"] = float(np.min(np.linalg.norm(source_targets - policy.target, axis=1)))
                    row["unsolved_steps"] = sum(str(s) not in ("optimal", "optimal_inaccurate") for s in trace["status"])
                    compact = {k: trace[k] for k in trace.files if k != "summary"}
                np.savez_compressed(path, **compact)
                row.update(method=METHOD, candidate_count=candidate_count)
                assert row["minimum_selected_goal_source_separation"] >= config.goal_source_target_min_distance
                return row

            run(210, 10, output_dir / "warmup.npz")
            for index, seed in enumerate(pending):
                path = folder / f"seed{seed}.npz"
                row = run(seed, config.steps, path)
                timing = timing_record(path, row, "new serial extension")
                path.with_suffix(".json").write_text(json.dumps({"signature": signature, "result": row, "timing": timing}, indent=2))
                rows.append(row)
                records.append(timing)
                if (index + 1) % 10 == 0 or index + 1 == len(pending):
                    print(f"{len(rows)}/{len(seeds)} total; {index+1}/{len(pending)} new; successes={sum(candidate.strict_success(r) for r in rows)}; seed={seed}", flush=True)
    rows.sort(key=lambda r: r["seed"])
    records.sort(key=lambda r: r["seed"])
    aggregate = {"method": METHOD, "rollouts": len(records)}
    aggregate.update({f"median {stage} (ms)": float(np.median([r[stage] for r in records])) for stage in STAGE_KEYS})
    payload = {
        "config": asdict(config), "signature": signature, "candidate_count": candidate_count,
        "results": rows, "aggregate": [aggregate], "per_seed": records,
        "protocol": {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "seeds": seeds, "steps": config.steps, "workers": 1, "numerical_threads": 1,
            "discard_initial_steps_per_rollout": 5, "extension_warmup_seed": 210, "extension_warmup_steps": 10,
            "timing_caveat": "Iterative timing pools archived serial ablation and a new serial extension, not an interleaved six-method benchmark.",
            "selection_caveat": "N=10 was selected from the earlier 211-710 candidate-count ablation; those 500 seeds are not untouched evaluation.",
            "archived_seeds": [r["seed"] for r in records if r["timing_batch"] == "archived candidate ablation"],
            "archive": str(ARCHIVE)},
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(aggregate, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=211)
    parser.add_argument("--seed-stop", type=int, default=1210)
    parser.add_argument("--candidate-count", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run_panel(range(args.seed_start, args.seed_stop + 1), args.output_dir, args.candidate_count)
