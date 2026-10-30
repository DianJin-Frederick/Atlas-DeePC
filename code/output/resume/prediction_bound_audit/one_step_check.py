"""Supplementary, assumption-screened comparison; does not alter the notebook.

Uses all structurally eligible systems from the archived 1000-system generator,
50 original noise levels, and 20 fresh noise realizations per level. This is a
NEW experiment, not a replay of the paper figure. Bounds use the notebook's
unchanged formula implementations. The subspace distance is oracle information.
"""
import argparse
from dataclasses import dataclass
import gzip
import csv
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
    args = parser.parse_args()
    out = Path(__file__).resolve().parent
    ns = {"__name__": __name__, "np": np, "pd": pd, "sqrt": sqrt,
          "dataclass": dataclass, "time": time}
    with zipfile.ZipFile(args.archive) as z:
        prefix = "prediction_error_bound/"
        notebook = json.loads(z.read(prefix + "bound_comparison.ipynb"))
        for name in ("core/common.py", "core/subspace.py"):
            exec(compile(z.read(prefix + name), name, "exec"), ns)
        for index in (4, 6, 8):
            exec(compile("".join(notebook["cells"][index]["source"]),
                         f"notebook_cell_{index}", "exec"), ns)
    cfg = ns["cfg"]
    systems = ns["generate_systems"](np.random.default_rng(cfg.seed), cfg)
    # Structural selection, independent of observed performance. Scalar outputs
    # ensure coordinatewise [-N,N] noise also has per-time Euclidean norm <= N.
    eligible = [s for s in systems if s.T_f == 1 and s.T_p >= s.n and s.p == 1]
    print(f"Structurally eligible systems: {len(eligible)} / {len(systems)}", flush=True)
    rng = np.random.default_rng(20260903)
    all_pairs = []
    rejected_baseline = rejected_local = accepted = coverage_ours = coverage_base = 0
    start = time.monotonic()
    fields = ["system_id", "noise_id", "realization", "N", "baseline_bound",
              "oracle_bound", "baseline_error", "oracle_error", "kappa",
              "sigma_min", "delta_sn", "raw_bound_reduction_pct"]
    with gzip.open(out / "one_step_raw.csv.gz", "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, s in enumerate(eligible):
            clean_history = s.y_ini_clean
            h = np.concatenate([s.u_ini.reshape(-1, order="F"),
                                s.u_fut.reshape(-1, order="F"),
                                clean_history.reshape(-1, order="F")])
            reference = ns["subspace_predictor"](
                s.U_clean, s.T_p, 1, s.m, 1, s.u_ini, s.u_fut, clean_history).ravel()
            reference_base = ns["behavioral_predict"](
                s.H_clean, s.T_p, 1, s.m, 1, s.u_ini, s.u_fut, clean_history)
            assert np.allclose(reference, reference_base, rtol=1e-7, atol=1e-9)
            for noise_id, N in enumerate(cfg.noise_levels):
                records = []
                for realization in range(20):
                    noisy = s.y_off_clean + rng.uniform(-N, N, s.y_off_clean.shape)
                    H_noisy = ns["build_io_hankel"](s.u_off, noisy, s.T_p + 1)
                    H_hat, U_hat = ns["tsvd"](H_noisy, s.rank)
                    baseline, delta = ns["kaviani_theorem2_bound"](
                        H_noisy, H_hat, h, s.rank, s.T_p, 1, s.m, 1, N)
                    if not np.isfinite(baseline) or delta <= cfg.delta_sn_min:
                        rejected_baseline += 1
                        continue
                    oracle, kappa = ns["our_theorem2_bound"](
                        U_hat, s.U_clean, s.T_p, 1, s.m, 1,
                        s.u_ini, s.u_fut, clean_history)
                    sigma, _ = ns["our_predictor_quantities"](U_hat, s.T_p, 1, s.m, 1)
                    if sigma <= 0 or kappa > sigma / (2 * sqrt(2)):
                        rejected_local += 1
                        continue
                    pred = ns["subspace_predictor"](
                        U_hat, s.T_p, 1, s.m, 1, s.u_ini, s.u_fut, clean_history).ravel()
                    pred_base = ns["behavioral_predict"](
                        H_hat, s.T_p, 1, s.m, 1, s.u_ini, s.u_fut, clean_history)
                    err = float(np.linalg.norm(pred - reference))
                    err_base = float(np.linalg.norm(pred_base - reference_base))
                    accepted += 1
                    coverage_ours += int(err <= oracle + 1e-12)
                    coverage_base += int(err_base <= baseline + 1e-12)
                    reduction = 100 * (1 - oracle / baseline)
                    row = dict(zip(fields, [s.system_id, noise_id, realization, N,
                                           baseline, oracle, err_base, err, kappa,
                                           sigma, delta, reduction]))
                    writer.writerow(row)
                    records.append(row)
                if records:
                    rdf = pd.DataFrame(records)
                    norm = max(float(np.linalg.norm(reference)), cfg.eps)
                    all_pairs.append({
                        "system_id": s.system_id, "noise_id": noise_id, "N": N,
                        "accepted": len(rdf),
                        "median_raw_bound_reduction_pct": rdf.raw_bound_reduction_pct.median(),
                        "baseline_worst_relative_slack_pct": float(100 * ((rdf.baseline_bound - rdf.baseline_error) / norm).max()),
                        "oracle_worst_relative_slack_pct": float(100 * ((rdf.oracle_bound - rdf.oracle_error) / norm).max()),
                    })
            if (i + 1) % 10 == 0 or i + 1 == len(eligible):
                print(f"Completed {i+1}/{len(eligible)} systems; accepted {accepted}; elapsed {time.monotonic()-start:.1f}s", flush=True)
    pairs = pd.DataFrame(all_pairs)
    pairs.to_csv(out / "one_step_pairs.csv", index=False)
    raw = pd.read_csv(out / "one_step_raw.csv.gz")
    per_noise = pairs.groupby("N").agg(
        median_raw_reduction_pct=("median_raw_bound_reduction_pct", "median"),
        median_baseline_slack=("baseline_worst_relative_slack_pct", "median"),
        median_oracle_slack=("oracle_worst_relative_slack_pct", "median"))
    per_noise["median_slack_reduction_pct"] = 100 * (1 - per_noise.median_oracle_slack / per_noise.median_baseline_slack)
    per_noise.to_csv(out / "one_step_per_noise.csv")
    summary = {
        "experiment": "NEW supplementary scalar-output, one-step comparison; NOT paper-figure replay",
        "generator_seed": cfg.seed, "noise_seed": 20260903,
        "generated_systems": len(systems), "structurally_eligible_systems": len(eligible),
        "structural_filter": "T_f=1, T_p>=n, p=1; exact online initial history",
        "noise_levels": len(cfg.noise_levels), "realizations_per_level": 20,
        "attempted_draws": len(eligible) * len(cfg.noise_levels) * 20,
        "baseline_rejections_delta_sn_le_0p6": rejected_baseline,
        "local_distance_condition_rejections": rejected_local,
        "accepted_draws": accepted, "accepted_system_noise_pairs": len(pairs),
        "systems_with_accepted_draws": int(pairs.system_id.nunique()),
        "raw_bound_reduction_pct_median": float(raw.raw_bound_reduction_pct.median()),
        "raw_bound_reduction_pct_quartiles": raw.raw_bound_reduction_pct.quantile([.25, .75]).tolist(),
        "raw_bound_reduction_pct_range": [float(raw.raw_bound_reduction_pct.min()), float(raw.raw_bound_reduction_pct.max())],
        "raw_bound_smaller_fraction": float((raw.oracle_bound < raw.baseline_bound).mean()),
        "per_noise_median_raw_reduction_range": [float(per_noise.median_raw_reduction_pct.min()), float(per_noise.median_raw_reduction_pct.max())],
        "per_noise_median_slack_reduction_range": [float(per_noise.median_slack_reduction_pct.min()), float(per_noise.median_slack_reduction_pct.max())],
        "empirical_coverage_oracle": coverage_ours / accepted,
        "empirical_coverage_baseline": coverage_base / accepted,
        "interpretation": "Oracle-distance bound vs notebook implementation of Kaviani et al. Theorem 2; not fully data-computable or MSE improvement; baseline retains its online-noise allowance although history is exact.",
        "elapsed_seconds": time.monotonic() - start,
    }
    (out / "one_step_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
