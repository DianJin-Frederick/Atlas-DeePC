"""Post-run descriptive details; does not import or modify frozen control code."""

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/pusher_atlas/contribution_study_20260903"


def quantiles_ms(values):
    x = np.asarray(values) * 1000
    return dict(zip(("median", "p95", "p99", "max"),
                    map(float, np.quantile(x, [.5, .95, .99, 1]))))


def main():
    summary = json.loads((STUDY / "final/summary.json").read_text())
    verification = json.loads((STUDY / "final/independent_verification.json").read_text())
    assert summary["completed_trials"] == 2000 and verification["verified"]
    protocol = json.loads((STUDY / "final_protocol.json").read_text())
    groups, rows_by_group = {}, {}
    files = []
    for condition in protocol["conditions"]:
        for method in protocol["methods"]:
            paths = sorted((STUDY / "final" / condition / method).glob("seed*.npz"))
            rows = [json.loads(p.with_suffix(".json").read_text()) for p in paths]
            rows_by_group[condition, method] = rows
            timings, all_timings, sim, feedback, total = [], [], [], [], []
            statuses = Counter()
            excluded = 0
            for path in paths:
                with np.load(path, allow_pickle=False) as r:
                    mask = r["steady_timing_mask"].astype(bool)
                    c, s, f = (r[k] for k in ("control_compute_time_s", "environment_step_time_s", "feedback_time_s"))
                    timings.extend(c[mask]); all_timings.extend(c)
                    sim.extend(s[mask]); feedback.extend(f[mask]); total.extend((c+s+f)[mask])
                    statuses.update(r["status"].astype(str).tolist())
                    excluded += len(mask) - int(mask.sum())
                files.append((path.with_suffix(".json").stat().st_mtime, str(path.relative_to(STUDY))))
            settling = [r["settling_time_s"] for r in rows if r["success"]]
            recovery = [r["post_disturbance_settling_time_s"] for r in rows if r["recovery_success"]]
            groups[f"{condition}:{method}"] = {
                "controller_compute_steady_ms": quantiles_ms(timings),
                "controller_compute_all_ms": quantiles_ms(all_timings),
                "simulation_steady_ms": quantiles_ms(sim),
                "feedback_steady_ms": quantiles_ms(feedback),
                "sum_of_measured_components_steady_ms": quantiles_ms(total),
                "sum_of_components_note": "Not a full loop timer: excludes instrumentation, list appends, intervention and setup.",
                "steady_calls": len(timings), "excluded_calls": excluded,
                "all_calls_over_50ms_percent": float(100*np.mean(np.asarray(all_timings) > .05)),
                "steady_calls_over_50ms_percent": float(100*np.mean(np.asarray(timings) > .05)),
                "mean_setup_s": float(np.mean([r["setup_time_s"] for r in rows])),
                "status_counts": dict(statuses),
                "episodes_with_joint_limit_excursion": sum(r["joint_limit_violation_steps"] > 0 for r in rows),
                "maximum_joint_limit_excess_rad": max(r["max_joint_limit_excess_rad"] for r in rows),
                "mean_maximum_joint_limit_excess_rad": float(np.mean([r["max_joint_limit_excess_rad"] for r in rows])),
                "mean_settling_s_among_successes": float(np.mean(settling)) if settling else None,
                "mean_recovery_s_among_recovered": float(np.mean(recovery)) if recovery else None,
                "median_recovery_s_among_recovered": float(np.median(recovery)) if recovery else None,
                "settling_observed_count": len(settling), "recovery_observed_count": len(recovery),
            }
    rng = np.random.default_rng(932608)
    comparisons = {}
    for condition in protocol["conditions"]:
        c = rows_by_group[condition, "C_combined"]
        a = rows_by_group[condition, "A_policy"]
        assert [r["seed"] for r in a] == [r["seed"] for r in c]
        # Exploratory extension of predeclared actual-constraint monitoring.
        delta = np.array([100*(x["joint_limit_violation_steps"]/x["completed_steps"] -
                              y["joint_limit_violation_steps"]/y["completed_steps"])
                          for x, y in zip(c, a)])
        draws = rng.integers(0, len(delta), (10000, len(delta)))
        comparisons[condition] = {
            "joint_excursion_sample_fraction_difference_pp": float(delta.mean()),
            "paired_bootstrap95_pp": np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist(),
            "note": "C minus A; exploratory, pointwise seed bootstrap, not safety or multiplicity-adjusted evidence.",
        }
    files.sort()
    start = (STUDY / "final/manifest.json").stat().st_mtime
    end = max(t for t, _ in files)
    gaps = sorted([{"between_checkpoint_seconds": t - previous,
                    "following_trial": path}
                   for previous, (t, path) in zip([start] + [t for t, _ in files[:-1]], files)],
                  key=lambda x: x["between_checkpoint_seconds"], reverse=True)[:10]
    run = json.loads((STUDY / "final/run_environment.json").read_text())
    output = {
        "groups": groups, "exploratory_C_minus_A": comparisons,
        "resources": {
            "manifest_mtime_utc": datetime.fromtimestamp(start, timezone.utc).isoformat(),
            "last_trial_checkpoint_mtime_utc": datetime.fromtimestamp(end, timezone.utc).isoformat(),
            "checkpoint_epoch_elapsed_s": end-start,
            "sum_recorded_trial_time_s": sum(r["trial_wall_time_s"] for rows in rows_by_group.values() for r in rows),
            "run_environment": run,
            "largest_checkpoint_intervals": gaps,
            "rollout_npz_bytes": sum(p.stat().st_size for p in (STUDY / "final").glob("*/*/seed*.npz")),
            "timing_caveat": "Filesystem epoch elapsed and perf_counter timers need not agree across host pauses/suspend/clock changes. Cause of observed gaps was not isolated. Shared-host empirical timing is not a real-time guarantee.",
        },
    }
    destination = STUDY / "final/detailed_metrics.json"
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    lines = ["# Complete secondary metrics", "", "100 paired seeds per row; 24,000 control samples per row. All failures retained.", "",
             "| Condition | Method | Success % [Wilson 95% CI] | Mean final cm | Mean capped settle s | Mean effort | RMS action increment | Joint-excursion samples | Recovered / eligible |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for a in summary["aggregates"]:
        lo, hi = a["success_wilson95_percent"]
        lines.append(f"| {a['condition']} | {a['method']} | {a['success_percent']:.0f} [{lo:.1f}, {hi:.1f}] | {100*a['mean_final_distance_m']:.2f} | {a['mean_capped_settling_time_s']:.3f} | {a['mean_control_effort_integral']:.2f} | {a['mean_rms_action_increment']:.3f} | {a['joint_limit_violation_steps']} / 24000 | {a['recovery_successes']} / {a['recovery_eligible_trials']} |")
    lines += ["", "Effort = dt × sum of squared applied torques; not energy. RMS increments include the change from zero initial control. Capped settling assigns failures 12 s; it is not an uncensored arrival-time mean. Recovery denominators are selected pre-disturbance goal-hold cases displaced outside the goal.", "",
              "## Computation and solver diagnostics", "", "Steady timing excludes the first 20 calls and each chart's first call. Times in ms; no simulation/rendering included.", "",
              "| Condition | Method | Median / p95 / p99 | All-call p95 | Steady >50 ms % | Inaccurate / retries / fallbacks | Raw QP overshoot samples |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for a in summary["aggregates"]:
        lines.append(f"| {a['condition']} | {a['method']} | {a['median_compute_ms']:.2f} / {a['p95_compute_ms']:.2f} / {a['p99_compute_ms']:.2f} | {a['all_step_p95_compute_ms']:.2f} | {a['compute_over_50ms_percent']:.2f} | {a['solver_inaccurate_steps']} / {a['solver_retry_steps']} / {a['fallback_steps']} | {a['raw_qp_torque_exceedance_steps']} |")
    lines += ["", "All applied torques respected ±2 within tolerance. Raw overshoots are before the common clipping stage; they are not applied-actuator violations.", "",
              "## Paired success comparisons", "", "C minus comparator, percentage points; pointwise 10,000-resample seed bootstrap intervals. Only C versus A/B belongs to the eight-test Holm family.", "",
              "| Condition : comparison | Difference pp [95% CI] | C-only / comparator-only successes | Exact McNemar p | Holm p (primary only) |",
              "|---|---:|---:|---:|---:|"]
    for key, value in summary["paired_comparisons"].items():
        estimate = value["success"]; lo, hi = estimate["paired_bootstrap95"]
        adjusted = f"{value['holm_adjusted_p']:.6g}" if "holm_adjusted_p" in value else "exploratory"
        lines.append(f"| {key} | {estimate['mean_difference']:+.0f} [{lo:+.0f}, {hi:+.0f}] | {value['combined_only']} / {value['comparator_only']} | {value['mcnemar_exact_p']:.6g} | {adjusted} |")
    lines += ["", "A degenerate [0, 0] empirical bootstrap interval with no discordances does not establish population equivalence. Secondary continuous-outcome paired estimates and confidence intervals are in `summary.json`; additional timing/recovery/resource details are in `detailed_metrics.json`."]
    (STUDY / "final/secondary_metrics.md").write_text("\n".join(lines) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
