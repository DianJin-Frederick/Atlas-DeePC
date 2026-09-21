"""Summarize archived plot data and replay an unchanged notebook subset.

No source archive or original experimental results are modified.
Usage: python audit.py /path/to/prediction_error_bound.zip --replay-systems 3
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from math import sqrt
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--replay-systems", type=int, default=3)
    args = parser.parse_args()
    out = Path(__file__).resolve().parent
    with zipfile.ZipFile(args.archive) as z:
        prefix = "prediction_error_bound/"
        csv_bytes = z.read(prefix + "results/data/paper_fig4_theorem2_vs_our_bound_with_theorem9.csv")
        df = pd.read_csv(BytesIO(csv_bytes))
        notebook = json.loads(z.read(prefix + "bound_comparison.ipynb"))
        ns = {"__name__": __name__, "np": np, "pd": pd, "sqrt": sqrt,
              "dataclass": dataclass, "time": time}
        for name in ("core/common.py", "core/subspace.py"):
            exec(compile(z.read(prefix + name), name, "exec"), ns)
        # Reviewed configuration and function-definition cells only. Do not run
        # notebook cells that launch the full experiment or overwrite its files.
        for index in (4, 6, 8, 10, 12):
            source = "".join(notebook["cells"][index]["source"])
            exec(compile(source, f"notebook_cell_{index}", "exec"), ns)

    a = "worst_relative_gap_kaviani_theorem2_pct"
    b = "worst_relative_gap_our_theorem2_pct"
    c = "worst_relative_gap_our_theorem2_theorem9_pct"
    valid = df[np.isfinite(df[a]) & np.isfinite(df[b]) & (df[a] > 0)].copy()
    per_noise = valid.groupby("N")[[a, b]].median()
    per_noise["median_gap_reduction_pct"] = 100 * (1 - per_noise[b] / per_noise[a])
    per_noise["accepted_systems"] = valid.groupby("N").size()
    per_noise.to_csv(out / "per_noise_median_gaps.csv")
    paired_reduction = 100 * (1 - valid[b] / valid[a])
    computable = df.dropna(subset=[a, c])
    summary = {
        "source_archive": str(args.archive),
        "source_csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "metric": "Worst normalized bound slack per system/noise pair; not MSE or raw bound reduction",
        "saved_systems": int(df.system_id.nunique()),
        "noise_levels": int(df.N.nunique()),
        "noise_range": [float(df.N.min()), float(df.N.max())],
        "planned_noise_draws": int(len(df) * ns["cfg"].noise_realizations),
        "accepted_noise_draws": int(df.accepted_realizations.sum()),
        "accepted_system_noise_pairs": len(valid),
        "systems_with_any_accepted_draw": int(valid.system_id.nunique()),
        "accepted_systems_per_noise_range": [int(per_noise.accepted_systems.min()), int(per_noise.accepted_systems.max())],
        "per_noise_median_gap_reduction_pct_min": float(per_noise.median_gap_reduction_pct.min()),
        "per_noise_median_gap_reduction_pct_max": float(per_noise.median_gap_reduction_pct.max()),
        "per_noise_median_gap_reduction_pct_median": float(per_noise.median_gap_reduction_pct.median()),
        "median_paired_gap_reduction_pct": float(paired_reduction.median()),
        "oracle_gap_smaller_fraction": float((valid[b] < valid[a]).mean()),
        "computable_comparable_pairs": len(computable),
        "computable_gap_ratio_median": float((computable[c] / computable[a]).median()),
        "computable_gap_smaller_fraction": float((computable[c] < computable[a]).mean()),
        "saved_multistep_comparison_pairs": int((valid.T_f > 1).sum()),
        "saved_past_horizon_below_order_pairs": int((valid.T_p < valid.n).sum()),
    }

    # Independent rerun of the original algorithm; retain its exact RNG stream.
    # Generating all 1000 systems before evaluating is essential to reproduction.
    rng = np.random.default_rng(ns["cfg"].seed)
    systems = ns["generate_systems"](rng, ns["cfg"])
    original_our = ns["our_theorem2_bound"]
    original_baseline = ns["kaviani_theorem2_bound"]
    latest_baseline = {}
    current_system = {}
    diagnostic_rows = []

    def instrument_baseline(*pos, **kw):
        result = original_baseline(*pos, **kw)
        latest_baseline.update(bound=result[0], N=pos[-1])
        return result

    def instrument_our(U_est, U_true, T_p, T_f, m, p, u_ini, u_fut, y_ini):
        args_our = (U_est, U_true, T_p, T_f, m, p, u_ini, u_fut, y_ini)
        bound, kappa = original_our(*args_our)
        sigma, _ = ns["our_predictor_quantities"](U_est, T_p, T_f, m, p)
        est = ns["subspace_predictor"](U_est, T_p, T_f, m, p, u_ini, u_fut, y_ini).ravel()
        true = ns["subspace_predictor"](U_true, T_p, T_f, m, p, u_ini, u_fut, y_ini).ravel()
        first_error = float(np.linalg.norm((est - true)[:p]))
        full_error = float(np.linalg.norm(est - true))
        diagnostic_rows.append({
            "system_id": current_system["id"], "N": latest_baseline["N"],
            "n": current_system["n"], "T_p": T_p, "T_f": T_f,
            "baseline_bound": latest_baseline["bound"], "oracle_bound": bound,
            "raw_bound_reduction_pct": 100 * (1 - bound / latest_baseline["bound"]),
            "first_step_error": first_error, "full_horizon_error": full_error,
            "sigma_min": sigma, "kappa": kappa,
            "distance_condition_holds": bool(kappa <= sigma / (2 * sqrt(2))),
            "past_horizon_condition_holds": bool(T_p >= current_system["n"]),
            "bound_covers_first_step_empirically": bool(bound + 1e-12 >= first_error),
        })
        return bound, kappa

    ns["kaviani_theorem2_bound"] = instrument_baseline
    ns["our_theorem2_bound"] = instrument_our
    rerun_rows = []
    start = time.monotonic()
    for system in systems[:args.replay_systems]:
        current_system.update(id=system.system_id, n=system.n)
        rerun_rows.extend(ns["evaluate_system"](system, ns["cfg"], rng))
        print(f"Replayed system {system.system_id}; elapsed {time.monotonic()-start:.1f}s", flush=True)
    if rerun_rows:
        replay = pd.DataFrame(rerun_rows)
        replay.to_csv(out / "unchanged_notebook_replay.csv", index=False)
        diag = pd.DataFrame(diagnostic_rows)
        diag.to_csv(out / "replay_diagnostics.csv", index=False)
        expected = df[df.system_id < args.replay_systems].reset_index(drop=True)
        checks = {}
        for col in (a, b, c, "accepted_realizations", "accepted_theorem9_realizations"):
            checks[col] = bool(np.allclose(replay[col], expected[col], rtol=2e-3, atol=1e-9, equal_nan=True))
        summary["replay"] = {
            "systems": args.replay_systems,
            "noise_draws": len(replay) * ns["cfg"].noise_realizations,
            "accepted_draws": len(diag),
            "comparison_tolerance": "rtol=0.002, atol=1e-9 (SVD sensitivity across runtimes)",
            "saved_csv_matches": checks,
            "distance_condition_failures": int((~diag.distance_condition_holds).sum()),
            "past_horizon_condition_failures": int((~diag.past_horizon_condition_holds).sum()),
            "one_step_coverage_failures": int((~diag.bound_covers_first_step_empirically).sum()),
            "median_raw_bound_reduction_pct_diagnostic_only": float(diag.raw_bound_reduction_pct.median()),
            "elapsed_seconds": time.monotonic() - start,
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
