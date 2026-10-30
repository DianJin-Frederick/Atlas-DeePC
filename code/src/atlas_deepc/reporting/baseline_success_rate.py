"""Version-checked success statistics for the current baseline notebook."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from dataclasses import asdict
import math

from atlas_deepc.baselines import baseline_atlas_deepc as baseline
def validate_panel(payload, seeds, config):
    seeds = list(map(int, seeds))
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Requested seeds must be nonempty and unique")
    signature = payload["signature"]
    expected = {k: v for k, v in asdict(config).items() if k != "seed"}
    if (payload["method"] != baseline.BASELINE_VERSION
            or signature.get("matrix_geometry") != "linear"
            or signature.get("candidate_count") != 10
            or signature.get("config") != expected):
        raise ValueError("Results do not match the current linear + iterative-N=10 baseline")
    rows = payload["results"]
    by_seed = {int(row["seed"]): row for row in rows}
    if len(rows) != len(by_seed):
        raise ValueError("Duplicate seeds in saved results")
    selected = []
    for seed in seeds:
        if seed not in by_seed:
            continue
        row = by_seed[seed]
        if (row.get("baseline_version") != baseline.BASELINE_VERSION
                or row.get("matrix_geometry") != "linear"
                or row.get("candidate_count") != 10
                or row.get("dynamic_terminal_reference") is not True
                or row.get("steps") != config.steps
                or row.get("minimum_selected_goal_source_separation", -1) < config.goal_source_target_min_distance):
            raise ValueError(f"Invalid baseline provenance for seed {seed}")
        for window in (30, 100):
            mean, maximum = row[f"mean_last_{window}"], row[f"max_last_{window}"]
            if not (math.isfinite(mean) and math.isfinite(maximum)
                    and mean >= 0 and maximum >= 0
                    and (mean <= maximum or math.isclose(mean, maximum, rel_tol=1e-14, abs_tol=1e-15))):
                raise ValueError(f"Invalid distance statistics for seed {seed}")
            success = mean < 0.01 and maximum < 0.02
            if bool(row[f"success_{window}"]) != success:
                raise ValueError(f"Success flag disagrees with distances for seed {seed}")
        selected.append(row)
    missing = [seed for seed in seeds if seed not in by_seed]
    return selected, missing


def success_statistics(rows):
    n = len(rows)
    if not n:
        raise ValueError("Cannot estimate success rate without completed trials")
    successes = sum(bool(r["success_30"] and r["success_100"]) for r in rows)
    p = successes / n
    z = 1.959963984540054
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return {"trials": n, "successes": successes, "failures": n-successes,
            "success_rate": p, "wilson_low": max(0.0, center-half),
            "wilson_high": min(1.0, center+half)}


def notebook_cells():
    import nbformat as nbf
    cells = [nbf.v4.new_markdown_cell("""### 4.1. 1,000-seed success-rate test

This test uses seeds **211--1210** for the current **linear-subspace + iterative-N=10**
baseline. Both final-30 and final-100 windows must have mean fingertip error below
1 cm and maximum error below 2 cm. The success count and rate are reported only
when all 1,000 distinct seeds are available.

Set `RUN_1000_SEED_TEST=True` below to run this baseline only. Compatible completed
rollouts resume; the comparison algorithms and videos are not run. `TEST_RESULT_DIR`
can point to an already completed current-baseline panel in the same result format.
Old affine or fixed-reference results are explicitly rejected. Until a complete panel
exists, the section reports missing results, not a success-rate estimate.

These seeds have been inspected in earlier ablations, so this is not a pristine
held-out test. The reported rate describes these 1,000 trials with the frozen
offline dataset.
"""), nbf.v4.new_code_cell('''from baseline_success_rate import validate_panel, success_statistics

TEST_SEEDS = list(range(211, 1211))
assert len(TEST_SEEDS) == 1000 and len(set(TEST_SEEDS)) == 1000
RUN_1000_SEED_TEST = False
TEST_RESULT_DIR = RESULT_ROOT / "evaluation_1000"
TEST_SUMMARY_PATH = TEST_RESULT_DIR / "summary.json"

if RUN_1000_SEED_TEST:
    from atlas_deepc.benchmarks.baseline_1000_benchmark import run_panel as run_timed_baseline
    run_timed_baseline(TEST_SEEDS, TEST_RESULT_DIR, config)

test_rows, test_missing = [], TEST_SEEDS.copy()
if TEST_SUMMARY_PATH.exists():
    test_payload = json.loads(TEST_SUMMARY_PATH.read_text())
    test_rows, test_missing = validate_panel(test_payload, TEST_SEEDS, config)
    from atlas_deepc.atlas.candidate_goal_routing_experiment import file_digest
    assert test_payload["signature"]["data_sha256"] == file_digest(DATA_PATH)
    assert test_payload["signature"]["atlas_sha256"] == file_digest(ATLAS_PATH)

display(pd.Series({
    "baseline": baseline.BASELINE_VERSION,
    "seed range": f"{TEST_SEEDS[0]}--{TEST_SEEDS[-1]}",
    "requested trials": len(TEST_SEEDS),
    "completed trials in summary": len(test_rows),
    "missing trials": len(test_missing),
    "result source": str(TEST_SUMMARY_PATH),
}))
if test_missing:
    print("1,000-seed success rate: not available yet.")
    print("Set RUN_1000_SEED_TEST=True to run/resume the baseline panel, or point TEST_RESULT_DIR to completed current-baseline results.")
else:
    test_stats = success_statistics(test_rows)
    test_table = pd.DataFrame([{
        "trials": test_stats["trials"],
        "successes": test_stats["successes"],
        "failures": test_stats["failures"],
        "strict success rate": f"{test_stats['success_rate']:.1%}",
    }])
    display(test_table)
    print("Failed seeds:", [r["seed"] for r in test_rows if not (r["success_30"] and r["success_100"])])
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    rate = 100 * test_stats["success_rate"]
    ax.bar([0], [rate], color="#0072B2", width=0.5)
    ax.set(ylim=(0, 105), ylabel="strict success rate (%)",
           xticks=[0], xticklabels=["Atlas-DeePC with linear-subspace charts\\nIterative reference selection (N=10)"],
           title="1,000-seed baseline test")
    ax.text(0, max(3, rate/2), f"{test_stats['successes']}/1000\\n{rate:.1f}%",
            ha="center", color="white", fontsize=14)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(TEST_RESULT_DIR / "success_rate_1000.png", dpi=160)
    plt.show()
''')]
    cells += [nbf.v4.new_markdown_cell("""### 4.2. Computation time by module

This benchmark runs one baseline rollout at a time with one numerical thread.
All 320 steps count toward success; timing statistics exclude a 10-step warm-up
on seed 210 and the first five steps of each evaluated rollout. The raw trace files
retain **every step's timers**, including those excluded from the timing aggregate.

The table reports pooled **mean, median, p95, and p99 milliseconds per step**.
The stacked plot uses means, so the module contributions add to the mean total.
Separate median and p99 plots use unstacked bars: module percentiles do not add
to the total percentile. Total latency percentiles are computed directly from
the measured total time at each step. P99 is the 99th percentile of timed steps.
The QP timer includes constructing the optimization problem, canonicalization,
and solution, not just the solver's internal iterations. Simulator stepping,
file saving, and video rendering are excluded from online computation.

One-time target-neighborhood construction and offline cache preparation are shown
separately. Wall-clock timings were taken on this shared computer and may be affected
by other running jobs; they are not an isolated cross-algorithm speed comparison.
"""), nbf.v4.new_code_cell('''TIMING_SUMMARY_PATH = TEST_RESULT_DIR / "timing_summary.json"
if test_missing or not TIMING_SUMMARY_PATH.exists():
    print("Complete 1,000-seed module timings are not available yet.")
else:
    test_timing = json.loads(TIMING_SUMMARY_PATH.read_text())
    if any("p99_ms" not in row for row in test_timing["modules"]):
        from atlas_deepc.benchmarks.baseline_1000_benchmark import summarize_timing
        refreshed = summarize_timing([TEST_RESULT_DIR / f"seed{s}_control_results.npz" for s in TEST_SEEDS])
        test_timing.update(refreshed)
        TIMING_SUMMARY_PATH.write_text(json.dumps(test_timing, indent=2))
    assert {int(r["seed"]) for r in test_timing["per_seed"]} == set(TEST_SEEDS)
    assert test_timing["timed_steps"] == len(TEST_SEEDS) * (config.steps - 5)
    module_table = pd.DataFrame(test_timing["modules"]).set_index("module")
    display(module_table.style.format({"mean_ms": "{:.4f}", "median_ms": "{:.4f}",
                                      "p95_ms": "{:.4f}", "p99_ms": "{:.4f}",
                                      "share_of_mean_total": "{:.2%}"}))
    modules = module_table.drop(index="Total online control")
    np.testing.assert_allclose(modules["mean_ms"].sum(), module_table.loc["Total online control", "mean_ms"], atol=1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    palette = ["#56B4E9", "#E69F00", "#0072B2", "#CC79A7", "#009E73", "#777777"]
    bottom = 0.0
    for (name, row), color in zip(modules.iterrows(), palette):
        axes[0].bar([0], [row["mean_ms"]], bottom=bottom, color=color, label=name)
        bottom += row["mean_ms"]
    axes[0].set(xticks=[0], xticklabels=["Baseline Atlas-DeePC"], ylabel="mean time per step (ms)", ylim=(0, bottom*1.15))
    axes[0].text(0, bottom*1.03, f"{bottom:.3f} ms", ha="center")
    axes[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    overhead = modules.drop(index="QP construction and solve")
    axes[1].barh(overhead.index, overhead["mean_ms"], color=[palette[i] for i in [0,1,2,3,5]])
    axes[1].set(xlabel="mean time per step (ms)", title="Non-QP modules (separate linear scale)")
    axes[1].grid(axis="x", alpha=0.2)
    fig.savefig(TEST_RESULT_DIR / "module_computation_time.png", dpi=160, bbox_inches="tight")
    plt.show()
    for statistic, label in [("median_ms", "Median"), ("p99_ms", "P99")]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
        main = module_table.loc[["Total online control", "QP construction and solve"], statistic]
        bars = axes[0].bar(["Total online control", "QP construction and solve"], main,
                           color=["#0072B2", "#009E73"], width=0.55)
        axes[0].bar_label(bars, fmt="%.3f", padding=4)
        axes[0].set(ylabel=f"{label} time per step (ms)", ylim=(0, main.max()*1.18))
        bars = axes[1].barh(overhead.index, overhead[statistic],
                            color=[palette[i] for i in [0, 1, 2, 3, 5]])
        axes[1].bar_label(bars, fmt="%.4f", padding=4)
        axes[1].set(xlabel=f"{label} time per step (ms)",
                    xlim=(0, overhead[statistic].max()*1.25),
                    title="Non-QP modules (separate linear scale)")
        axes[0].grid(axis="y", alpha=0.2)
        axes[1].grid(axis="x", alpha=0.2)
        fig.suptitle(f"{label} computation time across {test_timing['timed_steps']:,} timed steps")
        fig.savefig(TEST_RESULT_DIR / f"module_computation_time_{statistic}.png", dpi=160, bbox_inches="tight")
        plt.show()
    display(pd.Series(test_timing["one_time_goal_setup_ms"], name="One-time goal-neighborhood setup per rollout (ms)"))
    display(pd.DataFrame([{
        "session start UTC": s["started_utc"], **s["offline_setup_seconds"],
    } for s in test_timing["sessions"]]).rename(columns={
        "data_loading": "data loading (s)", "atlas_loading": "atlas loading (s)",
        "source_episode_metadata": "source metadata (s)",
        "linear_chart_cache_and_all_pairs_routes": "chart SVD + graph cache (s)",
    }))
    print("Timed control steps:", test_timing["timed_steps"])
    print("Median across rollout-level medians (ms):", round(test_timing["median_of_rollout_medians_ms"], 4))
''')]
    return cells
