"""One bounded (10-trial) development check of a later recovery disturbance."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from pusher_contribution_study import (
    METHODS, SOURCE, STUDY, config, load_atlas, load_behavior_policy,
    metrics, provenance, trial, write_json,
)


def main():
    output = STUDY / "recovery_pilot"
    specification = {"step": 160, "displacement_m": .10}
    cfg = config()
    with threadpool_limits(limits=1):
        output.mkdir(exist_ok=True)
        if (output / "summary.json").exists():
            raise FileExistsError("Recovery check already completed; preserve it")
        write_json(output / "protocol.json", {
            "development_seeds": [51001, 51002], "condition": specification,
            "reason": "Original 6-second pilot had zero pre-disturbance sustained successes. "
                      "Check 8 seconds without altering controller or remaining horizon.",
            "provenance": provenance(),
        })
        atlas = load_atlas(SOURCE / "behavioral_atlas.npz")
        policy = load_behavior_policy(SOURCE / "offline_behavior_policy_regional.joblib")
        rows = []
        started = perf_counter()
        for offset, seed in enumerate((51001, 51002)):
            methods = list(METHODS)
            for method in methods[offset:] + methods[:offset]:
                path = output / "disturbance" / method / f"seed{seed}.npz"
                if path.exists():
                    raise FileExistsError("Partial recovery check exists; do not overwrite trials")
                result = trial(atlas, cfg, policy, method, "disturbance", specification, seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **result)
                row = metrics(result, cfg, specification)
                write_json(path.with_suffix(".json"), row)
                rows.append(row)
                print(json.dumps({k: row[k] for k in (
                    "seed", "method", "success", "pre_disturbance_sustained_success",
                    "recovery_eligible", "recovery_success", "post_disturbance_settling_time_s",
                )}), flush=True)
        write_json(output / "summary.json", {
            "completed_trials": len(rows), "development_only": True,
            "wall_time_s": perf_counter() - started, "condition": specification, "results": rows,
        })


if __name__ == "__main__":
    main()
