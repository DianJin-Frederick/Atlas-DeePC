"""Retest all p=8 mixed-K guidance regressions with the unguided atlas/rank."""

import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

import numpy as np
from atlas_deepc.ablations import figure_guidance_matched as study

VERSION = "guidance_regressions_p8_k64_r53_t300_v1"
RESULT_ROOT = ROOT / "experiments/reacher" / VERSION


def discover():
    panels = {}
    provenance = {}
    for mode in ("terminal", "graph"):
        directory = study.baseline.RESULT_ROOT / f"output8/{mode}"
        summary = directory / "summary.json"
        payload = json.loads(summary.read_text())
        rows = {int(row["seed"]): row for row in payload["results"]}
        assert len(payload["results"]) == len(rows) == 500
        assert set(rows) == set(range(10001, 10501))
        failed = {s for s, row in rows.items() if not study.runner.strict_success(row)}
        assert failed == set(json.loads((directory / "metrics.json").read_text())["failed_seeds"])
        panels[mode] = rows
        provenance[mode] = {"summary_sha256": study.runner.file_digest(summary),
                            "directory": str(directory)}
    seeds = tuple(s for s in sorted(panels["terminal"])
                  if study.runner.strict_success(panels["terminal"][s])
                  and not study.runner.strict_success(panels["graph"][s]))
    return seeds, panels, provenance


def main():
    seeds, panels, sources = discover()
    assert seeds
    provenance = dict(seeds=list(seeds), sources=sources,
        script_sha256=study.runner.file_digest(__file__),
        selection="Unguided p8 K64 rank53 success and guided p8 K96 rank57 failure",
        retest="Guided p8 K64 rank53, 300 steps; all other controller settings unchanged",
        caveat="Outcome-selected subset, not an unbiased success-rate estimate")
    path = RESULT_ROOT / "selection_provenance.json"
    if path.exists() and json.loads(path.read_text()) != provenance:
        raise ValueError("Regression selection/protocol changed")
    study.runner.save_json(path, provenance)
    original_root = study.RESULT_ROOT
    with patch.object(study, "SEEDS", seeds), patch.object(study, "VERSION", VERSION):
        manifest = study.prepare(RESULT_ROOT)
        signature = study.runner.file_digest(manifest)
        old_manifest = original_root / "output8/graph/manifest.json"
        old = json.loads(old_manifest.read_text())
        new = json.loads(manifest.read_text())
        for key in old.keys() | new.keys():
            if key not in ("evaluation_version", "evaluation_seeds"):
                assert old.get(key) == new.get(key), key
        # Reuse an identical rollout only after validating its original signature.
        with study.protocol_context():
            for seed in seeds:
                source = old_manifest.parent / f"seed{seed}_control_results.npz"
                destination = manifest.parent / source.name
                if not source.exists() or destination.exists():
                    continue
                study.runner.validate_trace(source, 8, "graph", study.runner.file_digest(old_manifest))
                with np.load(source, allow_pickle=True) as trace:
                    arrays = {k: trace[k] for k in trace.files if k != "summary"}
                    row = json.loads(trace["summary"].item())
                row.update(evaluation_version=VERSION, evaluation_signature_sha256=signature,
                    reused_from=str(source), reused_trace_sha256=study.runner.file_digest(source))
                temporary = destination.with_suffix(".pending.npz")
                np.savez_compressed(temporary, **arrays, summary=np.array(json.dumps(row)))
                temporary.replace(destination)
                study.runner.validate_trace(destination, 8, "graph", signature)
                print(f"Reused matched seed {seed}", flush=True)
        rows = study.run(RESULT_ROOT)
    for row in rows:
        previous = panels["graph"][row["seed"]]
        row.update(previous_guided_success=False,
                   previous_guided_final100_mm=1000*previous["mean_last_100"])
    study.runner.save_json(RESULT_ROOT / "regression_comparison.json", rows)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
