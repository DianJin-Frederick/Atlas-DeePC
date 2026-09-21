"""Create the offline/online Gaussian noise ablation report."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/ablations/ablation_measurement_noise.ipynb"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [md(r"""# Atlas-DeePC: Gaussian Measurement Noise

**100 paired environment seeds per noise level: 211--310.** The controller is the
graph-routed baseline: **48 physical charts, rank-59 uncentered linear-subspace bases,
iterative N=10 terminal selection, graph waypoints, and current-chart DeePC**.
The coefficient regularization is **$\lambda_g=0.05$**, matching the nominal
baseline. Rank 59 was selected from the rank ablation. Both rank and regularization
change relative to the previous noise notebook; this is not a single-factor
comparison to that notebook. All settings are fixed across noise levels.
Only the graph-routed variant is evaluated. There is no contextual rescue, filtering,
model fitting, IK, or noise-dependent controller retuning. The bounded SVD
coefficients and past-output slack penalty remain in place.

Gaussian measurement noise affects **both offline data and online observations**.
Success is evaluated against the **true simulator fingertip position**, never a
noisy sensor distance. All 320 steps are retained. These previously inspected
seeds form an exploratory paired ablation, not a new untouched test set.

The 3% and 4% panels extend the completed 0%, 1%, 2%, 5%, and 10% study.
Existing compatible rollouts are reused; each added level has 100 new rollouts
and its own refitted noisy atlas.

Sections 1--6 report rank 59. Section 7 adds rank 62 on the same data, noise
streams, and seeds, with paired error and computation-time comparisons.
"""), code('''import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML, display
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p / "src/measurement_noise_rank59_ablation.py").exists())
sys.path.insert(0, str(ROOT / "src"))
import measurement_noise_rank59_ablation as experiment
get_ipython().run_line_magic("matplotlib", "inline")
SEEDS = list(range(211, 311))
SD_RATIOS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10]
RUN_EXPERIMENTS = False
WORKERS = 4
RESULT_ROOT = experiment.RESULT_ROOT
if RUN_EXPERIMENTS:
    experiment.run_all(SD_RATIOS, SEEDS, WORKERS)
payloads = {}
for ratio in SD_RATIOS:
    path = RESULT_ROOT / experiment.level_name(ratio) / "summary.json"
    assert path.exists(), f"Missing completed level {ratio}: enable RUN_EXPERIMENTS to run/resume."
    payload = json.loads(path.read_text())
    assert payload["version"] == experiment.VERSION and payload["seeds"] == SEEDS
    rows = payload["results"]
    assert len(rows) == 100 and {r["seed"] for r in rows} == set(SEEDS)
    assert all(r["steps"] == 320 and r["evaluation_uses_true_outputs"] for r in rows)
    assert all(r["candidate_count"] == 10 and r["clusters"] == 48 and r["svd_rank"] == 59 for r in rows)
    assert all(r["lam_g"] == .05 and r["graph_waypoints_enabled"] for r in rows)
    assert all(not r["noise_on_inputs"] and not r["feedback_fallback"] and not r["fitted_dynamics_model"] for r in rows)
    assert all(r["minimum_selected_goal_source_separation"] >= .06 for r in rows)
    payloads[ratio] = payload
clean_signature = payloads[0.0]["artifact"]["signature"]
for payload in payloads.values():
    signature = payload["artifact"]["signature"]
    for key in clean_signature:
        if key not in ("sd_ratio", "power_nsr"):
            assert signature[key] == clean_signature[key], key
frame = pd.DataFrame([r for payload in payloads.values() for r in payload["results"]])
frame["strict_success"] = frame.success_30 & frame.success_100
percent = 100*np.array(SD_RATIOS)
COLORS = ["#172238", "#0072B2", "#009E73", "#6A3D9A", "#CC79A7", "#E69F00", "#D55E00"]
trace_errors = {}
error_rows = []
for row in frame.to_dict("records"):
    ratio, seed = row["noise_sd_ratio"], row["seed"]
    path = RESULT_ROOT / experiment.level_name(ratio) / f"seed{seed}_control_results.npz"
    with np.load(path, allow_pickle=True) as trace:
        error = 1000*trace["distance"]
    trace_errors[ratio, seed] = error
    tail = error[-100:]
    error_rows.append({"noise_sd_ratio": ratio, "seed": seed,
                       "tail_rmse_mm": np.sqrt(np.mean(tail**2)),
                       "tail_std_mm": np.std(tail), "tail_p95_mm": np.quantile(tail, .95),
                       "tail_within_10_pct": 100*np.mean(tail < 10),
                       "tail_within_20_pct": 100*np.mean(tail < 20)})
frame = frame.merge(pd.DataFrame(error_rows), on=["noise_sd_ratio", "seed"], validate="one_to_one")
print(f"Loaded {len(frame)} rollouts: {len(SD_RATIOS)} noise levels x 100 paired seeds.")
controls = pd.Series({"T_ini": 2, "T_f": 10, "steps": 320, "charts": 48,
    "rank": 59, "terminal candidates": 10, "q_xy": 100000, "q_joint": 5000,
    "q_dq": 20, "r": .0001, "lambda_g": .05, "lambda_ini": 100000,
    "lambda_delta_u": 5, "u_limit": 1, "coefficient bound scale": 3,
    "target latch (m)": .07, "solver": "CLARABEL"}, name="Value")
display(HTML("<details><summary>Fixed control parameters</summary>" + controls.to_frame().to_html() + "</details>"))
'''), md(r"""### Noise Definition and Placement

For channel $j$, let $s_j$ be its standard deviation across the **40,000 unique
clean offline measurements**, not repeated Hankel entries. Apply

$$\widetilde y_{t,j}=y_{t,j}+\epsilon_{t,j},\qquad
\epsilon_{t,j}\sim\mathcal N(0,\rho^2s_j^2).$$

$\rho$ is the **noise-SD / signal-SD ratio**. The equivalent **power NSR** is
$\rho^2$ relative to centered signal variance, and
$\mathrm{SNR}_{\mathrm{dB}}=-20\log_{10}\rho$. We report both conventions.
Online noise scales remain fixed from offline calibration; they do not depend on
the online trajectory, velocity, target, or observed tracking error.

- **Offline:** reconstruct 200 episodes of 200 samples, add one draw per time/channel,
  and rebuild overlapping windows. Every occurrence of the same physical sample
  receives the same noise. Applied inputs are unchanged. At nonzero levels, refit
  normalization, clustering, transition graph, medoid references, and SVD bases
  from noisy data. Zero noise uses the original clean atlas. Terminal candidates
  also come from the noisy library.
- **Online:** noise enters the existing output-map boundary, including reset and
  the two initialization samples. Localization, DeePC past matching, and the 7 cm
  latch all use noisy outputs. Each measurement is stored once in the history.
- **Exact quantities:** requested target and applied input commands. No process
  noise or actuator disturbance is introduced. Simulator truth is logged for
  post-run evaluation but never passed to the controller.

Noise is independent across the baseline's **ten lifted output channels**: four
joint cosine/sine channels, two angular velocities, fingertip x/y, and the two
augmented cosine/sine channels for $q_1+q_2$. Thus noisy periodic outputs need not
lie exactly on a unit circle. This is an output-space Gaussian model, not noise
on encoder angles before lifting; no projection or denoising is applied.

One offline noise realization is shared across seeds. Each environment seed has
an independent online stream. The same standard-normal draws are scaled across
levels (common random numbers). We do not average over independently recollected
noisy atlases. The original offline data and baseline artifacts are not overwritten.
"""), code('''calibration = []
for ratio, payload in payloads.items():
    sigma = np.asarray(payload["artifact"]["noise_std"])
    calibration.append({"Noise SD (%)": 100*ratio, "Power NSR": ratio**2,
                        "Power NSR (%)": 100*ratio**2,
                        "SNR (dB)": np.inf if ratio == 0 else -20*np.log10(ratio),
                        "Fingertip x noise SD (mm)": 1000*sigma[6],
                        "Fingertip y noise SD (mm)": 1000*sigma[7]})
display(pd.DataFrame(calibration).round(5))
scale = np.array(payloads[0.0]["artifact"]["signal_std"])
channels = pd.DataFrame({"Channel": experiment.CHANNELS,
                         "Unit": ["unitless"]*4 + ["rad/s"]*2 + ["m"]*2 + ["unitless"]*2,
                         "Offline signal SD": scale})
for ratio in SD_RATIOS[1:]:
    channels[f"Noise SD at {100*ratio:g}%"] = ratio*scale
display(HTML("<details><summary>Per-channel calibration</summary>" + channels.to_html(index=False) + "</details>"))
'''), md("""## 4. Results and Tracking Accuracy

Strict success requires **mean true error below 1 cm and maximum true error below
2 cm in both final-30 and final-100 windows**. The horizons, run length, QP weights,
input limits, coefficient-bound rule, and success thresholds are fixed across
noise levels. Rank is 59 and coefficient regularization is 0.05 throughout.
Unsolved QPs retain the baseline's zero-action handling and are not discarded.
"""), code('''summary_rows = []
for ratio in SD_RATIOS:
    rows = frame[frame.noise_sd_ratio == ratio]
    summary_rows.append({"Noise SD (%)": 100*ratio, "Power NSR": ratio**2,
                         "Successes": int(rows.strict_success.sum()), "Trials": len(rows),
                         "Success rate (%)": 100*rows.strict_success.mean(),
                         "Median final-100 true error (mm)": 1000*rows.mean_last_100.median(),
                         "P90 final-100 true error (mm)": 1000*rows.mean_last_100.quantile(.9),
                         "Median tail RMSE (mm)": rows.tail_rmse_mm.median(),
                         "Median within-seed tail SD (mm)": rows.tail_std_mm.median(),
                         "Mean tail time below 10 mm (%)": rows.tail_within_10_pct.mean(),
                         "Mean tail time below 20 mm (%)": rows.tail_within_20_pct.mean(),
                         "Unsolved QP steps": int(rows.unsolved_steps.sum()),
                         "Median goal-sample switches": rows.goal_sample_switches.median()})
summary = pd.DataFrame(summary_rows).set_index("Noise SD (%)")
display(summary[["Successes", "Success rate (%)", "Median final-100 true error (mm)",
                 "P90 final-100 true error (mm)", "Mean tail time below 10 mm (%)",
                 "Unsolved QP steps"]].round(3))
display(HTML("<details><summary>All error summary statistics</summary>" + summary.round(3).to_html() + "</details>"))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
fig.suptitle("Atlas-DeePC measurement noise: rank 59, lambda_g = 0.05")
axes[0].plot(percent, summary["Success rate (%)"], "o-", color="#0072B2")
for x, rate, count in zip(percent, summary["Success rate (%)"], summary.Successes):
    offset = (0, -20) if x in (2, 4) else (0, 8)
    axes[0].annotate(f"{count}/100", (x, rate), xytext=offset, textcoords="offset points", ha="center")
axes[0].set(ylim=(0, 108), title="Strict success on true state", ylabel="success rate (%)")
for column, color, label in [("Median final-100 true error (mm)", "#009E73", "median across seeds"),
                              ("P90 final-100 true error (mm)", "#D55E00", "90th percentile across seeds")]:
    axes[1].plot(percent, summary[column], "o-", color=color, label=label)
axes[1].axhline(10, color="#666666", linestyle=":", label="mean-error criterion (10 mm)")
axes[1].set(ylim=(0, None), title="Final-100 true fingertip error", ylabel="mean fingertip error (mm)")
axes[1].legend()
for ax in axes:
    ax.set_xlabel("noise SD / offline signal SD (%)")
    ax.set_xticks(percent)
    ax.grid(alpha=.2)
fig.savefig(RESULT_ROOT / "noise_success_and_accuracy.png", dpi=170)
plt.show()
(RESULT_ROOT / "noise_report.json").write_text(summary.reset_index().to_json(orient="records", indent=2))
clean = frame[frame.noise_sd_ratio == 0].set_index("seed")
paired = []
for ratio in SD_RATIOS[1:]:
    noisy = frame[frame.noise_sd_ratio == ratio].set_index("seed").loc[clean.index]
    repaired = noisy.index[noisy.strict_success & ~clean.strict_success].tolist()
    regressed = noisy.index[~noisy.strict_success & clean.strict_success].tolist()
    paired.append({"Noise SD (%)": 100*ratio, "Repaired": len(repaired), "Regressed": len(regressed),
                   "Repaired seeds": repaired, "Regressed seeds": regressed})
paired = pd.DataFrame(paired).set_index("Noise SD (%)")
display(paired[["Repaired", "Regressed"]])
display(HTML("<details><summary>Paired seed lists</summary>" + paired.to_html() + "</details>"))
'''), md("""### Error Distributions and Time Near the Target

Each seed contributes equally. The CDF shows the distribution of each seed's
**final-100 mean true fingertip error**, first over its complete range and then
zoomed to 0--30 mm. A CDF value at 10 mm is a mean-error pass fraction, **not**
the stricter success rate above.

The lower panels show every seed: the fraction of its final 100 steps below
10 mm, and the temporal standard deviation of its fingertip-error magnitude.
Low temporal SD alone does not imply accuracy: an arm stuck away from the target
can have constant error. Boxes show the median and interquartile range; whiskers
span the 5th--95th percentiles, and dots include all 100 seeds, including outliers.
These are empirical distributions, not confidence intervals.
"""), code('''fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
fig.suptitle("Atlas-DeePC measurement noise: rank 59, lambda_g = 0.05")
for ratio, color in zip(SD_RATIOS, COLORS):
    values = np.sort(1000*frame.loc[frame.noise_sd_ratio == ratio, "mean_last_100"].to_numpy())
    x = np.r_[0, values]
    cumulative = np.r_[0, np.arange(1, len(values)+1)/len(values)]
    for ax in axes[0]:
        ax.step(x, cumulative, where="post", color=color, label=f"{100*ratio:g}% SD")
for ax, title in zip(axes[0], ["Final-100 mean error: full distribution", "Final-100 mean error: near-target zoom"]):
    ax.axvline(10, color="#666666", linestyle=":")
    ax.set(xlim=(0, None), ylim=(0, 1.02), xlabel="final-100 mean true error (mm)",
           ylabel="fraction of seeds", title=title)
    ax.grid(alpha=.2)
axes[0, 1].set_xlim(0, 30)
axes[0, 0].legend(loc="lower right")
for ax, metric, title, ylabel in [
    (axes[1, 0], "tail_within_10_pct", "Time near the target", "final-100 steps below 10 mm (%)"),
    (axes[1, 1], "tail_std_mm", "Error fluctuation within each seed", "SD of final-100 true error (mm)")]:
    groups = [frame.loc[frame.noise_sd_ratio == r, metric].to_numpy() for r in SD_RATIOS]
    bp = ax.boxplot(groups, positions=np.arange(len(SD_RATIOS)), widths=.5, patch_artist=True, whis=(5, 95), showfliers=False,
                    medianprops={"color": "#172238", "linewidth": 2})
    jitter = np.random.default_rng(710).uniform(-.16, .16, 100)
    for i, (values, color, box) in enumerate(zip(groups, COLORS, bp["boxes"])):
        box.set(facecolor=color, alpha=.18)
        ax.scatter(i+jitter, values, s=12, color=color, alpha=.55, zorder=3)
    ax.set(ylim=(0, 105) if metric.endswith("pct") else (0, None),
           xlabel="noise SD / offline signal SD (%)", ylabel=ylabel, title=title)
    ax.set_xticks(np.arange(len(SD_RATIOS)), [f"{100*r:g}" for r in SD_RATIOS])
    ax.grid(axis="y", alpha=.2)
fig.savefig(RESULT_ROOT / "noise_error_distributions.png", dpi=170)
plt.show()
'''), md(r"""## 3. Result and Fairness Audit

The zero-noise condition uses the same observer wrapper as the noisy conditions.
Its sensor outputs must exactly equal truth when noise is zero. All noise levels
use the same clean source data and noise seeds, with separate noisy atlases.
We verify rank-59 bases, noise calibration, true-state error, and source hashes.
The archived rank-60 results are not reused for this new configuration.
"""), code('''zero_matches = 0
for seed in SEEDS:
    zero_path = RESULT_ROOT / experiment.level_name(0) / f"seed{seed}_control_results.npz"
    with np.load(zero_path, allow_pickle=True) as zero:
        np.testing.assert_array_equal(zero["sensor_true"], zero["sensor_measured"])
        zero_matches += 1
atlas_rows = []
for ratio, payload in payloads.items():
    artifact = payload["artifact"]
    assert artifact["overlap_consistency_verified"] and artifact["applied_inputs_unchanged"]
    assert artifact["offline_unique_samples"] == 40000
    from candidate_goal_routing_experiment import file_digest
    for name, digest in artifact["artifact_sha256"].items():
        assert file_digest(RESULT_ROOT / experiment.level_name(ratio) / name) == digest
    with np.load(RESULT_ROOT / experiment.level_name(ratio) / "chart_cache.npz") as cache:
        assert cache["basis"].shape == (48, 144, 59)
        assert not np.any(cache["offset"])
    if ratio:
        np.testing.assert_allclose(artifact["empirical_offline_power_nsr"], ratio**2, rtol=.04)
    online_noise = []
    for row in payload["results"]:
        path = RESULT_ROOT / experiment.level_name(ratio) / f"seed{row['seed']}_control_results.npz"
        with np.load(path, allow_pickle=True) as trace:
            np.testing.assert_array_equal(trace["sensor_measured"][-320:], trace["y_measured"])
            np.testing.assert_array_equal(trace["sensor_true"][-320:], trace["y"])
            np.testing.assert_allclose(trace["distance"], np.linalg.norm(trace["y"][:, 6:8]-row["exact_target_xy"], axis=1))
            assert len(trace["sensor_true"]) == 323
            rng = np.random.default_rng(np.random.SeedSequence([experiment.noise.ONLINE_NOISE_SEED, row["seed"]]))
            expected_noise = rng.normal(size=trace["sensor_true"].shape)*np.array(artifact["noise_std"])
            np.testing.assert_allclose(trace["sensor_measured"]-trace["sensor_true"], expected_noise, rtol=0, atol=1e-14)
            online_noise.append(trace["sensor_measured"]-trace["sensor_true"])
    realized = np.sqrt(np.mean(np.concatenate(online_noise)**2, axis=0))/np.array(artifact["signal_std"])
    if ratio:
        np.testing.assert_allclose(realized, ratio, rtol=.03)
    else:
        np.testing.assert_array_equal(realized, np.zeros(10))
    sizes = artifact["chart_sizes"]
    atlas_rows.append({"Noise SD (%)": 100*ratio, "Charts": len(sizes), "Smallest chart": min(sizes),
                       "Largest chart": max(sizes), "Unreachable chart pairs": artifact["unreachable_chart_pairs"],
                       "Min observed online SD ratio (%)": 100*realized.min(),
                       "Max observed online SD ratio (%)": 100*realized.max(),
                       "Offline fit (s)": artifact["offline_preparation_s"]["chart_fit"],
                       "Offline bases/routes (s)": artifact["offline_preparation_s"]["svd_cache_and_routes"]})
print(f"Zero-noise sensor outputs exactly match truth for {zero_matches}/100 seeds.")
print(f"All chart hashes, rank-59 bases, and {len(frame)} sensor-noise streams verified.")
audit = experiment.routed.audit_iid_library(np.load(experiment.baseline.DATA_PATH, mmap_mode="r"), eval_seeds=SEEDS)
assert audit["offline_stabilized_episode_count"] == 0
display(pd.Series({"offline stabilized episodes": audit["offline_stabilized_episode_count"],
    "minimum selected source-target separation (m)": frame.minimum_selected_goal_source_separation.min(),
    "fitted dynamics / contextual rescue": "none", "evaluation": "true simulator outputs"}))
display(pd.DataFrame(atlas_rows).set_index("Noise SD (%)").round(3))
'''), md("""### 4.1. Computation Time by Module

Local wall-clock timings use four rollout workers and one numerical-library thread
per worker. Five initial steps are excluded per rollout; there is no separate
warm-up. This shared-computer run is not an isolated hardware benchmark. The
existing control-loop timers exclude simulator stepping and sensor-noise generation.
QP time includes problem construction and solve. Chart fitting and SVD are offline.
Levels were collected in separate batches; these are descriptive timings rather
than a controlled cross-level speed comparison.
The stacked plot uses means, which add to the mean total. Median and p99 module
plots are unstacked because module percentiles do not add to total percentiles.
"""), code('''timing_rows = []
for ratio, payload in payloads.items():
    assert payload["timing"]["timed_steps"] == 31500
    timing_rows.extend({"Noise SD (%)": 100*ratio, **r} for r in payload["timing"]["modules"])
timing = pd.DataFrame(timing_rows)
display(timing[timing.module.isin(["Total online control", "QP construction and solve"])]
        .set_index(["Noise SD (%)", "module"])[["mean_ms", "median_ms", "p99_ms"]].round(3))
display(HTML("<details><summary>All computation modules</summary>" + timing.set_index(["Noise SD (%)", "module"])
             [["mean_ms", "median_ms", "p95_ms", "p99_ms"]].round(4).to_html() + "</details>"))
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
fig.suptitle("Atlas-DeePC measurement noise: rank 59, lambda_g = 0.05")
for ax, metric, label in zip(axes, ["median_ms", "p99_ms"], ["Median", "P99"]):
    for module, color in [("Total online control", "#0072B2"), ("QP construction and solve", "#D55E00")]:
        rows = timing[timing.module == module]
        ax.plot(rows["Noise SD (%)"], rows[metric], "o-", label=module, color=color)
    ax.set(ylim=(0, None), xlabel="noise SD / offline signal SD (%)", ylabel="ms per step", title=label)
    ax.set_xticks(percent)
    ax.grid(alpha=.2)
    ax.legend()
fig.savefig(RESULT_ROOT / "noise_computation_time.png", dpi=170)
plt.show()
modules = [m for m in timing.module.unique() if m != "Total online control"]
module_colors = ["#0072B2", "#009E73", "#E69F00", "#CC79A7", "#D55E00", "#666666"]
fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
bottom = np.zeros(len(SD_RATIOS))
for module, color in zip(modules, module_colors):
    values = timing[timing.module == module].set_index("Noise SD (%)").loc[percent, "mean_ms"].to_numpy()
    ax.bar(np.arange(len(SD_RATIOS)), values, bottom=bottom, color=color, label=module)
    bottom += values
ax.set(ylim=(0, None), ylabel="mean ms per control step", xlabel="noise SD / signal SD (%)",
       title="Mean computation time by module")
ax.set_xticks(np.arange(len(SD_RATIOS)), [f"{x:g}" for x in percent])
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
fig.savefig(RESULT_ROOT / "noise_timing_mean_modules.png", dpi=170)
plt.show()
fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
for ax, module in zip(axes.flat, modules):
    rows = timing[timing.module == module].set_index("Noise SD (%)").loc[percent]
    ax.bar(np.arange(len(SD_RATIOS))-.18, rows.median_ms, width=.36, color="#0072B2", label="Median")
    ax.bar(np.arange(len(SD_RATIOS))+.18, rows.p99_ms, width=.36, color="#D55E00", label="P99")
    ax.set(ylim=(0, None), title=module, ylabel="ms per step", xlabel="noise SD / signal SD (%)")
    ax.set_xticks(np.arange(len(SD_RATIOS)), [f"{x:g}" for x in percent])
    ax.grid(axis="y", alpha=.2)
axes.flat[0].legend()
fig.savefig(RESULT_ROOT / "noise_timing_module_percentiles.png", dpi=170)
plt.show()
display(pd.DataFrame([{"Noise SD (%)": 100*r, **payloads[r]["timing"]["one_time_goal_setup_ms"]}
                      for r in SD_RATIOS]).set_index("Noise SD (%)")
        .rename(columns=lambda c: f"One-time terminal candidate search: {c} (ms)").round(3))
'''), md("""## 5. Diagnostic Trajectories: Twelve Seeds

Twelve fixed seeds are shown at every noise level, expanding the previous three
examples. They are fixed before completing this rerun, not chosen for favorable
outcomes. The first figure **zooms to 0--30 mm**, emphasizing the 10 mm criterion.
Small triangles at the upper boundary flag samples outside the zoom; they are
not substituted error values. The second figure retains the full vertical range.
The shaded final-100 window is the main evaluation period. All curves show true
fingertip error, with no smoothing or rollout truncation.
"""), code('''EXAMPLE_SEEDS = [212, 222, 232, 237, 247, 255, 266, 277, 288, 296, 302, 310]
for zoom in (True, False):
    fig, axes = plt.subplots(4, 3, figsize=(15, 13), constrained_layout=True)
    for ax, seed in zip(axes.flat, EXAMPLE_SEEDS):
        for ratio, color in zip(SD_RATIOS, COLORS):
            error = trace_errors[ratio, seed]
            ax.plot(error, color=color, label=f"{100*ratio:g}% SD", linewidth=1)
            if zoom:
                outside = np.flatnonzero(error > 30)
                ax.scatter(outside, np.full(len(outside), 29.6), marker="^", s=8,
                           color=color, alpha=.25, linewidths=0)
        ax.axvspan(220, 319, color="#666666", alpha=.06)
        ax.axhline(10, color="#666666", linestyle="--", linewidth=1.3)
        ax.set(ylim=(0, 30) if zoom else (0, None), xlim=(0, 319),
               xlabel="control step", ylabel="true fingertip error (mm)", title=f"Seed {seed}")
        if zoom:
            ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
        ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(SD_RATIOS),
               title="rank 59, lambda_g = 0.05; noise SD / signal SD")
    suffix = "zoom" if zoom else "full"
    fig.savefig(RESULT_ROOT / f"noise_example_traces_{suffix}.png", dpi=150)
    plt.show()
example_table = frame[frame.seed.isin(EXAMPLE_SEEDS)].copy()
example_table["Final-100 mean true error (mm)"] = 1000*example_table.mean_last_100
example_table["Noise SD (%)"] = 100*example_table.noise_sd_ratio
display(example_table.pivot(index="seed", columns="Noise SD (%)",
                            values="Final-100 mean true error (mm)").round(3))
'''), md(r"""## 6. Slowed Atlas-Routing Dashboards

These six dashboards replay **saved rank-59, $\lambda_g=0.05$ rollouts**, not new simulations:

- **Seed 212, 1% versus 5%.**
- **Seed 222, 1% versus 5%.**
- **Seed 237, 2% versus 10%.**

These are the same diagnostic examples as before; all outcome labels and error
statistics below are recomputed from the new experiment, not copied from old videos.

The dark arm is simulator truth; the gray cross is the noisy fingertip reading.
The red error curve is true error, while the gray curve is sensed error. Reference
postures and chart routes use the matching noisy offline atlas. Noisy references
and predictions need not be kinematically consistent.

All **320 steps** remain visible, including late fluctuations. Playback uses the
same **0.25x dashboard-speed convention** as the earlier videos (relative to one
sample per frame at 20 fps, not simulator real time). Interpolation is for display
only; the controller did not smooth its measurements or actions. The error panel
zooms near the target; initial errors above its limit are outside that panel, but
their magnitude remains in the header and the full-range plots in Section 5.
"""), code('''import base64
from candidate_goal_routing_experiment import file_digest
RENDER_VIDEOS = False
if RENDER_VIDEOS:
    from render_measurement_noise_videos import render_all
    render_all()
    get_ipython().run_line_magic("matplotlib", "inline")
VIDEO_DIR = RESULT_ROOT / "videos"
entries = json.loads((VIDEO_DIR / "manifest.json").read_text())
lookup = {(entry["seed"], entry["noise_sd_ratio"]): entry for entry in entries}
def noisy_video_card(seed, ratio):
    entry = lookup[seed, ratio]
    metadata = entry["metadata"]
    source_dir = RESULT_ROOT / experiment.level_name(ratio)
    source = source_dir / f"seed{seed}_control_results.npz"
    video = VIDEO_DIR / Path(entry["video"]).name
    poster = VIDEO_DIR / Path(entry["poster"]).name
    assert metadata["source_result_sha256"] == file_digest(source)
    assert metadata["source_atlas_sha256"] == file_digest(source_dir / "behavioral_atlas.npz")
    assert metadata["video_sha256"] == file_digest(video)
    assert metadata["poster_sha256"] == file_digest(poster)
    assert metadata["lam_g"] == .05 and metadata["noise_sd_ratio"] == ratio
    assert metadata["saved_control_steps"] == metadata["displayed_control_steps"] == 320
    assert metadata["true_arm_displayed"] and metadata["noisy_fingertip_displayed"]
    row = frame[(frame.seed == seed) & (frame.noise_sd_ratio == ratio)].iloc[0]
    outcome = "STRICT SUCCESS" if row.strict_success else "FAILED STRICT TEST"
    color = "#087443" if row.strict_success else "#B42318"
    mp4 = base64.b64encode(video.read_bytes()).decode("ascii")
    png = base64.b64encode(poster.read_bytes()).decode("ascii")
    return (f'<div style="min-width:0;background:white;color:#172238;padding:12px;border:1px solid #cbd5e1">'
        f'<h3 style="margin:0 0 8px;color:#172238">Seed {seed} | {100*ratio:g}% SD noise</h3>'
        f'<p style="margin:0 0 8px;color:{color}">{outcome}</p>'
        f'<p style="margin:0 0 8px;color:#344054">Final-100 true error: mean {1000*row.mean_last_100:.3f} mm; '
        f'maximum {1000*row.max_last_100:.3f} mm</p>'
        f'<video controls preload="none" style="width:100%" poster="data:image/png;base64,{png}" '
        f'src="data:video/mp4;base64,{mp4}"></video></div>')
for seed, lower, higher in ((212, .01, .05), (222, .01, .05), (237, .02, .10)):
    display(HTML('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:14px;margin-bottom:20px">'
                 + noisy_video_card(seed, lower) + noisy_video_card(seed, higher) + '</div>'))
'''), md("""## 7. Reproduction and Preservation

From the project root, using the existing environment:

```bash
python src/measurement_noise_rank59_ablation.py --workers 4 --ratios 0 0.01 0.02 0.03 0.04 0.05 0.10
python src/render_measurement_noise_videos.py --workers 2
```

Completed levels/seeds are resumed; incompatible artifact settings are rejected.
The notebook defaults to reading saved results with rank 59 and coefficient
regularization 0.05. This override is local to the noise experiment; other baseline notebooks
are not changed. No duplicate full Hankel matrices
are saved: noise is regenerated deterministically, while atlases and reduced bases
are cached. Baseline data, notebooks, and results remain unchanged.

This combined ablation does not isolate offline versus online noise. It does not
establish robustness to correlated noise, sensor bias, encoder noise before output
lifting, process disturbances, or independently repeated offline datasets.
Noise also affects the posture retrieved for terminal-reference augmentation:
its fingertip entries are replaced with the exact target as in the baseline,
but its joint entries remain noisy. Consequently, results include reference
inconsistency as well as changes in localization and behavioral prediction.
The previous rank-60, lambda_g=0 notebook is preserved in
`archive/notebooks/ablation_measurement_noise_rank60_lamg0_before_rank59.ipynb`.
Its data remain in their original directory. The new experiment uses a separate
`measurement_noise_rank59_lamg005_100_v1` directory and does not overwrite old results.
""")]
formulation = md(r"""## 1. Local Linear-Subspace DeePC

Each chart contains full noisy input-output Hankel columns, not only the
configuration coordinates used for clustering. Compute an **uncentered rank-59
SVD basis** $B_k$ for each chart. There is no affine offset or sum-to-one constraint.

$$[u_{\rm ini}^\top\;u^\top\;(\widetilde y_{\rm ini}+\sigma)^\top\;y^\top]^\top=B_k g.$$

The QP uses tracking weights $Q=\mathrm{diag}(5000I_4,20I_2,100000I_2,5000I_2)$,
$R=10^{-4}I_2$, $\lambda_g=0.05$, $\lambda_{\rm ini}=10^5$, and
$\lambda_{\Delta u}=5$. Coefficients retain their empirical bounds; applied
inputs are bounded by $[-1,1]$.

At each step, localize the noisy measured posture, choose among the N=10 terminal
candidates using graph-route cost, and track the next chart medoid until the
7 cm terminal-reference latch. DeePC always uses the **current chart's** basis.
The candidate pool is found once per rollout; candidate selection can change
online. Only the graph-routed variant is evaluated in this notebook.
""")
atlas_intro = md(r"""## 2. Inspecting the Constructed Physical-Configuration Atlas

K-means uses the latest six periodic joint-configuration channels. It is refitted
on noisy data at every nonzero noise level, before computing chart bases and
the transition graph. Inputs, velocities, and fingertip position are not used
for clustering, but remain in the full behavioral matrices.

The following baseline-style views show the **5% noise atlas** by default.
Change `ATLAS_NOISE_RATIO` to any tested level to inspect another atlas.
Light arms are 32 sampled chart members; bold arms are medoids. These show
**noisy observed configurations**, so link lengths and trigonometric consistency
need not be exact. No clean-state correction is applied. Chart IDs are local
to each independently fitted atlas and are not matched across noise levels.
""")
atlas_setup = code('''ATLAS_NOISE_RATIO = 0.05
assert ATLAS_NOISE_RATIO in SD_RATIOS
H_clean = np.load(experiment.baseline.DATA_PATH, mmap_mode="r")
H, _, _ = experiment.noisy_library(H_clean, ATLAS_NOISE_RATIO)
atlas = experiment.routed.load_atlas(
    RESULT_ROOT / experiment.level_name(ATLAS_NOISE_RATIO) / "behavioral_atlas.npz")
config = experiment.experiment_config()
print(f"Inspecting {100*ATLAS_NOISE_RATIO:g}% SD-noise atlas: 48 charts, rank {config.svd_rank}")
''')
baseline_notebook = nbf.read(ROOT / "notebooks/baselines/baseline_graph_routed_atlas_deepc.ipynb", as_version=4)
chart_sources = []
for prefix in ("from matplotlib.collections import LineCollection", "Y_windows = H[", "from sklearn.decomposition import PCA"):
    source = next(c.source for c in baseline_notebook.cells if c.cell_type == "code" and c.source.startswith(prefix))
    source = source.replace("experiment.behavioral_features(", "experiment.routed.behavioral_features(")
    source = source.replace('RESULT_ROOT / "', 'RESULT_ROOT / experiment.level_name(ATLAS_NOISE_RATIO) / "')
    source = source.replace('fig.suptitle("Physical-configuration chart clouds with bold medoids", fontsize=16)',
        'fig.suptitle(f"{100*ATLAS_NOISE_RATIO:g}% noise: chart clouds with bold medoids", fontsize=16)')
    chart_sources.append(code(source))
atlas_note = md("""### How to Read These Views

The workspace panel shows noisy offline fingertip coverage; the dashed circle is
the nominal kinematic reach, not a constraint on noisy measurements. PCA is a
two-dimensional visualization only. The graph panel draws the two strongest
outgoing edges per chart, while control retains all edges meeting the count
threshold of three. Forward-kinematics inconsistency is an expected diagnostic
of independent output-channel noise, not a correction used by control.
""")
# Match the baseline sequence: formulation, atlas, audit, results/timing, videos.
cells = (cells[:2] + [formulation] + cells[2:4]
         + [atlas_intro, atlas_setup] + chart_sources + [atlas_note]
         + cells[8:10] + cells[4:8] + cells[10:])
from measurement_noise_rank62_notebook_section import build_cells as rank62_cells
cells[-1].source = cells[-1].source.replace("## 7. Reproduction", "## 8. Reproduction")
cells[-1].source += """\nRank 62 is an additional paired section, not a replacement for rank 59.
Its seven-level panel can be resumed with:

```bash
python src/measurement_noise_rank62_ablation.py --workers 4
```

Rank-62 outputs are stored in `measurement_noise_rank62_lamg005_100_v1`.
The six videos in Section 6 remain the rank-59 diagnostics.
"""
cells = cells[:-1] + rank62_cells() + cells[-1:]
for cell in cells:
    if cell.cell_type == "code":
        compile(cell.source, str(OUTPUT), "exec")
metadata = {"kernelspec": {"display_name": "Python (gym-robotics-mujoco)", "language": "python", "name": "gym-robotics-mujoco"}}
nbf.write(nbf.v4.new_notebook(cells=cells, metadata=metadata), OUTPUT)
print(OUTPUT)
