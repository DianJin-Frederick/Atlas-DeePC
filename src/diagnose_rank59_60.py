"""Trace the four paired success changes, without changing the frozen sweep."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
import sys
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments/routed_deepc_iid/linear_subspace_rank_100_v1"
sys.path.insert(0, str(RESULT / "frozen_protocol/src"))
import linear_subspace_rank_ablation as experiment


def main():
    experiment.prepare()
    experiment.initialize_worker()
    output = RESULT / "diagnostics_rank59_60"
    output.mkdir(exist_ok=True)
    controller59 = experiment.controller_for_rank(59)
    controller60 = experiment.baseline.BaselineAtlasDeePC(experiment._H, experiment._atlas, experiment.rank_config(60))
    nested = max(np.max(np.abs(controller59.matrix_cache[k][0]["basis"] - m["basis"][:, :59]))
                 for k, (m, _) in controller60.matrix_cache.items())
    records = []
    for seed in (232, 247, 288, 296):
        path59 = output / f"rank59_seed{seed}.npz"
        if not path59.exists():
            controller59.run(seed, path59)
        path60 = ROOT / f"experiments/routed_deepc_iid/baseline_output_channels_linear_n10_v1/output10/seed{seed}_control_results.npz"
        pair = {}
        for rank, path in ((59, path59), (60, path60)):
            with np.load(path, allow_pickle=True) as t, np.load(experiment.checkpoint_path(rank, seed)) as checkpoint:
                np.testing.assert_allclose(t["distance"], checkpoint["distance"], rtol=0, atol=1e-10)
                row = json.loads(t["summary"].item())
                target = t["final_goal"].reshape(-1)[6:8]
                pair[rank] = {k: np.array(t[k]) for k in ("u", "y", "ref", "chart", "goal_chart", "goal_index", "pred")}
                records.append({"seed": seed, "rank": rank, "mean_tail100_mm": 1000*row["mean_last_100"],
                    "tail_charts": np.unique(t["chart"][-100:]).tolist(),
                    "tail_goal_indices": np.unique(t["goal_index"][-100:]).tolist(),
                    "tail_goal_charts": np.unique(t["goal_chart"][-100:]).tolist(),
                    "tail_prediction_error_mm": float(1000*np.mean(np.linalg.norm(t["pred"][-100:, 6:8]-t["y"][-100:, 6:8], axis=1))),
                    "tail_predicted_target_error_mm": float(1000*np.mean(np.linalg.norm(t["pred"][-100:, 6:8]-target, axis=1))),
                    "final_action": t["u"][-1].tolist(), "statuses": row["statuses"]})
        print(json.dumps(records[-2:]), flush=True)
        print("Final references identical:", np.array_equal(pair[59]["ref"][-1], pair[60]["ref"][-1]), flush=True)
    report = {"max_nested_basis_difference": float(nested), "rollouts": records,
              "source": "Rank 59 rerun; rank 60 existing full traces. Every distance matches the frozen sweep within 1e-10 m."}
    experiment.write_json(output / "diagnosis.json", report)


if __name__ == "__main__":
    main()
