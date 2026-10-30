"""Read-only matched-seed summaries for contextual DPC parameter development."""
import json
from pathlib import Path

import numpy as np

from atlas_deepc.benchmarks import contextual_dpc_reacher as c

BASELINE = c.OUTPUT
TUNING = c.ROOT / "experiments/reacher/contextual_dpc_tuning_v1"
PILOT = [211, 222, 258, 281]
EXPANDED = [211, 214, 222, 236, 241, 245, 258, 271, 281, 288, 290, 310]


def candidates():
    paths = [BASELINE] + sorted(p.parent for p in TUNING.glob("*/manifest.json"))
    reference = json.loads((BASELINE / "manifest.json").read_text())
    fixed = {k: v for k, v in reference["config"].items()
             if k not in ("q_xy", "r", "lambda_g_bar", "lambda_sigma")}
    result = {}
    for path in paths:
        manifest = json.loads((path / "manifest.json").read_text())
        if (manifest["data_sha256"] != reference["data_sha256"]
                or manifest.get("controller_sha256", manifest.get("source_sha256")) != reference["source_sha256"]
                or {k: manifest["config"][k] for k in fixed} != fixed):
            raise ValueError(f"Incompatible controller/data/protocol: {path}")
        rows, _ = c.load_results(path)
        result[path.name] = {"path": str(path), "config": manifest["config"],
                            "rows": {r["seed"]: r for r in rows}}
    return result


def matched_summary(seeds):
    table = []
    for name, candidate in candidates().items():
        config = candidate["config"]
        rows = [candidate["rows"][s] for s in seeds if s in candidate["rows"]]
        if not rows:
            continue
        # All candidates have the same training normalization and physical stage
        # definitions. Rescore to fixed Q_xy=1, R=0.1; never rank native costs.
        common_cost = [r["closed_loop_reference_cost"] / config["q_xy"]
                       + .1 / config["r"] * (r["closed_loop_input_cost"] + r["closed_loop_smoothing_cost"])
                       for r in rows]
        table.append({"candidate": name, "q_xy": config["q_xy"], "r": config["r"],
            "lambda_g": config["lambda_g_bar"] * config["columns"],
            "lambda_sigma": config["lambda_sigma"], "completed": len(rows), "requested": len(seeds),
            "matched_panel_complete": len(rows) == len(seeds),
            "successes": sum(r["strict_success"] for r in rows),
            "success_pct": float(np.mean([r["strict_success"] for r in rows])*100),
            "mean_final100_mm": float(np.mean([r["mean_last_100"] for r in rows])*1000),
            "median_final100_mm": float(np.median([r["mean_last_100"] for r in rows])*1000),
            "worst_final100_mm": float(max(r["mean_last_100"] for r in rows)*1000),
            "median_prediction_mm": float(np.median([r["mean_prediction_error_xy"] for r in rows
                                                     if r["mean_prediction_error_xy"] is not None])*1000),
            "mean_xy_squared_error_sum_m2": float(np.mean([r["xy_squared_error_sum_m2"] for r in rows])),
            "median_common_cost_q1_r01": float(np.median(common_cost)),
            "fallback_steps": sum(r["fallback_steps"] for r in rows)})
    return table


def export_final(selected="q100_r01"):
    """Export only a complete paired development report; never run simulations."""
    data = candidates()
    old, new = data[BASELINE.name], data[selected]
    seeds = list(range(211, 311))
    if set(new["rows"]) != set(seeds) or set(old["rows"]) != set(seeds):
        raise ValueError("The full 100-seed panel is not complete")
    old_success = {s for s in seeds if old["rows"][s]["strict_success"]}
    new_success = {s for s in seeds if new["rows"][s]["strict_success"]}
    payload = {
        "selected_candidate": selected, "selected_config": new["config"],
        "scope": "Adaptive development tuning, not held-out evaluation or global optimality",
        "timing_caveat": "Concurrent tuning timings must not be compared with serial baseline latency",
        "four_seed_pilot": matched_summary(PILOT),
        "twelve_seed_panel": matched_summary(EXPANDED),
        "full_development": matched_summary(seeds),
        "recovered_seeds": sorted(new_success-old_success),
        "regressed_seeds": sorted(old_success-new_success),
        "remaining_failures": sorted(set(seeds)-new_success),
    }
    c.write_json(TUNING / "tuning_report.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps({"four_seed_pilot": matched_summary(PILOT),
                      "twelve_seed_panel": matched_summary(EXPANDED),
                      "full_development": matched_summary(list(range(211, 311)))}, indent=2))
