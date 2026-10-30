"""Remove only the empirical coefficient box from the notebook baseline."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from atlas_deepc.common.paths import source_hashes, source_path

from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.baselines import baseline_atlas_deepc as baseline
from atlas_deepc.benchmarks import baseline_clarabel_waypoint_comparison as comparison
from atlas_deepc.atlas import routed_deepc_experiment as routed
from atlas_deepc.atlas.candidate_goal_routing_experiment import file_digest


ROOT = baseline.ROOT
OUTPUT = ROOT / "experiments/routed_deepc_iid/coefficient_bounds_removed_311_410_v1"
BASELINE = ROOT / "experiments/routed_deepc_iid/clarabel_lamg005_panel_311_1310_v1/graph/summary.json"
SEEDS = list(range(311, 411))


def successful(row):
    return bool(row["success_30"] and row["success_100"])


def main():
    config = comparison.default_config()
    historical = json.loads(BASELINE.read_text())
    expected = {k: v for k, v in asdict(config).items() if k != "seed"}
    assert historical["signature"]["config"] == expected
    assert historical["signature"]["solver"] == "CLARABEL"
    bounded = {int(r["seed"]): r for r in historical["results"]}
    signature = {
        "config": asdict(config), "seeds": SEEDS, "solver": "CLARABEL",
        "reference_mode": "graph", "coefficient_bounds_enabled": False,
        "changed_component": "remove g_min and g_max inequalities only",
        "workers": 1, "numerical_threads": 1,
        "data_sha256": file_digest(baseline.DATA_PATH),
        "atlas_sha256": file_digest(baseline.ATLAS_PATH),
        "source_sha256": source_hashes((
            "ablation_coefficient_bounds.py", "baseline_atlas_deepc.py",
            "routed_deepc_experiment.py", "expert_pipeline_search.py",
            "candidate_goal_routing_experiment.py")),
    }
    for key in ("data_sha256", "atlas_sha256"):
        assert signature[key] == historical["signature"][key]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = OUTPUT / "manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == signature
    else:
        manifest.write_text(json.dumps(signature, indent=2))
    rows = []
    with threadpool_limits(limits=1):
        H = np.load(baseline.DATA_PATH, mmap_mode="r")
        atlas = routed.load_atlas(baseline.ATLAS_PATH)
        controller = baseline.BaselineAtlasDeePC(H, atlas, config)
        old_boxes = {}
        for charts, (matrix, _) in controller.matrix_cache.items():
            old_boxes[charts] = (matrix.pop("g_min"), matrix.pop("g_max"))
            assert set(matrix) == {"offset", "basis"}
        np.savez_compressed(OUTPUT / "original_coefficient_boxes.npz", **{
            f"chart{charts[0]}_{side}": bound
            for charts, bounds in old_boxes.items()
            for side, bound in zip(("lower", "upper"), bounds)
        })
        original_solve = routed.pipe.DeePCController.solve
        coefficients = []

        def logged_solve(instance, matrix, *args, **kwargs):
            assert instance.solver == "CLARABEL"
            assert matrix.get("g_min") is None and matrix.get("g_max") is None
            result = original_solve(instance, matrix, *args, **kwargs)
            g = result[2]
            coefficients.append(np.full(config.svd_rank, np.nan) if g is None else g.ravel().copy())
            return result

        with patch.object(routed.pipe.DeePCController, "solve", logged_solve):
            for seed in SEEDS:
                path = OUTPUT / f"seed{seed}_control_results.npz"
                if path.exists():
                    with np.load(path, allow_pickle=True) as trace:
                        row = json.loads(trace["summary"].item())
                        assert row["coefficient_bounds_enabled"] is False
                        assert trace["g_coefficients"].shape == (config.steps, config.svd_rank)
                else:
                    coefficients.clear()
                    row = controller.run(seed, path, reference_mode="graph")
                    assert len(coefficients) == config.steps
                    row["coefficient_bounds_enabled"] = False
                    row["ablation"] = "coefficient_bounds_removed_311_410_v1"
                    with np.load(path, allow_pickle=True) as trace:
                        arrays = {k: trace[k] for k in trace.files if k != "summary"}
                    np.savez_compressed(path, **arrays, g_coefficients=np.asarray(coefficients),
                                        summary=np.array(json.dumps(row)))
                row["result_path"] = str(path)
                rows.append(row)
                repaired = [r["seed"] for r in rows if successful(r) and not successful(bounded[r["seed"]])]
                regressed = [r["seed"] for r in rows if not successful(r) and successful(bounded[r["seed"]])]
                report = {
                    "signature": signature, "completed": len(rows), "requested": len(SEEDS),
                    "bounded_successes": sum(successful(bounded[r["seed"]]) for r in rows),
                    "unbounded_successes": sum(map(successful, rows)),
                    "repaired_seeds": repaired, "regressed_seeds": regressed,
                    "unsolved_steps": sum(r["unsolved_steps"] for r in rows), "results": rows,
                }
                (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2))
                print(f"{len(rows)}/100 seed={seed}: success={successful(row)}, "
                      f"final100={1000*row['mean_last_100']:.3f} mm; "
                      f"paired totals bounded={report['bounded_successes']} "
                      f"unbounded={report['unbounded_successes']}", flush=True)


if __name__ == "__main__":
    main()
