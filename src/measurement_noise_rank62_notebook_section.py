"""Notebook cells for the paired rank-59/rank-62 measurement-noise study."""

import nbformat as nbf


def build_cells():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    return [md(r"""## 7. Rank-62 Ablation on the Same Noisy Data

**Seven noise levels, 100 paired seeds per level (211--310).** This section changes
only the retained linear-subspace rank from **59 to 62**. Clarabel,
$\lambda_g=0.05$, all other QP settings, N=10 iterative terminal selection,
graph waypoints, 320-step rollouts, and evaluation thresholds remain unchanged.

For each noise level, reuse its already fitted noisy K-means atlas, medoids, and
transition graph. Reconstruct exactly the same noisy Hankel matrix and compute
rank-62 bases. The first 59 basis directions and their empirical bounds are
checked against the saved rank-59 cache. Online Gaussian draws and targets are
paired across ranks. Closed-loop states and chart sequences can still differ.

The chart clouds in Section 2 apply to both ranks: no second clustering is needed
when changing rank alone. Rank-62 traces and timings are saved separately; the
rank-59 results above are preserved. These are exploratory ablations on previously
inspected seeds, not a fresh held-out validation of rank selection.
"""), code('''import measurement_noise_rank62_ablation as rank62_experiment
RUN_RANK62_EXPERIMENTS = False
RANK62_ROOT = rank62_experiment.RESULT_ROOT
if RUN_RANK62_EXPERIMENTS:
    rank62_experiment.run_all(SD_RATIOS, SEEDS, workers=WORKERS)
rank62_payloads = {}
rank62_errors = {}
rank62_extra = []
for ratio in SD_RATIOS:
    folder = RANK62_ROOT / experiment.level_name(ratio)
    path = folder / "summary.json"
    assert path.exists(), f"Missing rank-62 panel {ratio}: enable RUN_RANK62_EXPERIMENTS."
    payload = json.loads(path.read_text())
    assert payload["version"] == rank62_experiment.VERSION and payload["seeds"] == SEEDS
    rows = payload["results"]
    assert len(rows) == 100 and {r["seed"] for r in rows} == set(SEEDS)
    assert all(r["svd_rank"] == 62 and r["lam_g"] == .05 and r["steps"] == 320 for r in rows)
    assert all(r["graph_waypoints_enabled"] and r["evaluation_uses_true_outputs"] for r in rows)
    assert all(r["candidate_count"] == 10 and r["clusters"] == 48 for r in rows)
    assert all(not r["fitted_dynamics_model"] and not r["feedback_fallback"] and not r["noise_on_inputs"] for r in rows)
    assert all(r["minimum_selected_goal_source_separation"] >= .06 for r in rows)
    artifact = payload["artifact"]
    assert artifact["same_chart_memberships_and_graph_verified"]
    assert artifact["nested_basis_prefix_verified"]
    old_folder = RESULT_ROOT / experiment.level_name(ratio)
    old_artifact = payloads[ratio]["artifact"]
    assert artifact["signature"]["rank59_artifact_manifest_sha256"] == file_digest(old_folder / "artifact_manifest.json")
    previous_config = dict(old_artifact["signature"]["config"])
    current_config = dict(artifact["signature"]["config"])
    assert previous_config.pop("svd_rank") == 59 and current_config.pop("svd_rank") == 62
    assert previous_config == current_config
    for name, digest in artifact["artifact_sha256"].items():
        assert file_digest(folder / name) == digest
    with np.load(folder / "behavioral_atlas.npz", allow_pickle=True) as current:
        with np.load(old_folder / "behavioral_atlas.npz", allow_pickle=True) as previous:
            for key in previous.files:
                if key != "config":
                    np.testing.assert_array_equal(current[key], previous[key])
    with np.load(folder / "chart_cache.npz") as current:
        with np.load(old_folder / "chart_cache.npz") as previous:
            assert current["basis"].shape == (48, 144, 62)
            np.testing.assert_allclose(current["basis"][:, :, :59], previous["basis"], rtol=1e-10, atol=1e-10)
            for name in ("g_min", "g_max"):
                np.testing.assert_allclose(current[name][:, :59], previous[name], rtol=1e-10, atol=1e-10)
    for row in rows:
        seed = row["seed"]
        filename = f"seed{seed}_control_results.npz"
        with np.load(folder / filename, allow_pickle=True) as current:
            with np.load(old_folder / filename, allow_pickle=True) as previous:
                old_row = json.loads(previous["summary"].item())
                np.testing.assert_array_equal(row["exact_target_xy"], old_row["exact_target_xy"])
                np.testing.assert_array_equal(current["sensor_true"][0], previous["sensor_true"][0])
                np.testing.assert_allclose(current["sensor_measured"]-current["sensor_true"],
                    previous["sensor_measured"]-previous["sensor_true"], rtol=0, atol=1e-13)
            distance = np.asarray(current["distance"])
            np.testing.assert_allclose(distance, np.linalg.norm(current["y"][:, 6:8]-row["exact_target_xy"], axis=1))
            for window in (30, 100):
                tail = distance[-window:]
                assert row[f"success_{window}"] == bool(tail.mean() < .01 and tail.max() < .02)
            error = 1000*distance
            rank62_errors[ratio, seed] = error.copy()
        tail = error[-100:]
        rank62_extra.append({"noise_sd_ratio": ratio, "seed": seed,
            "tail_rmse_mm": np.sqrt(np.mean(tail**2)), "tail_std_mm": tail.std(),
            "tail_p95_mm": np.quantile(tail, .95), "tail_within_10_pct": 100*np.mean(tail < 10),
            "tail_within_20_pct": 100*np.mean(tail < 20)})
    rank62_payloads[ratio] = payload
rank62_frame = pd.DataFrame([r for p in rank62_payloads.values() for r in p["results"]])
rank62_frame["strict_success"] = rank62_frame.success_30 & rank62_frame.success_100
rank62_frame = rank62_frame.merge(pd.DataFrame(rank62_extra), on=["noise_sd_ratio", "seed"], validate="one_to_one")
print(f"Verified {len(rank62_frame)} rank-62 rollouts: same atlases, nested bases, targets, initial states, and noise streams.")
'''), md("""### 7.1. Paired Success and True Fingertip Error

Every seed contributes equally. Success still requires mean error below 10 mm and
maximum error below 20 mm in both final-30 and final-100 windows. Repaired and
regressed counts compare the same seed and noise realization, not unpaired totals.
"""), code('''rank_rows = []
rank_pair_rows = []
for ratio in SD_RATIOS:
    for rank, data in [(59, frame), (62, rank62_frame)]:
        rows = data[data.noise_sd_ratio == ratio]
        rank_rows.append({"Rank": rank, "Noise SD (%)": 100*ratio,
            "Successes": int(rows.strict_success.sum()), "Trials": len(rows),
            "Success rate (%)": 100*rows.strict_success.mean(),
            "Median final-100 mean error (mm)": 1000*rows.mean_last_100.median(),
            "P90 final-100 mean error (mm)": 1000*rows.mean_last_100.quantile(.9),
            "Median tail RMSE (mm)": rows.tail_rmse_mm.median(),
            "Median within-seed tail SD (mm)": rows.tail_std_mm.median(),
            "Mean tail time below 10 mm (%)": rows.tail_within_10_pct.mean(),
            "Unsolved QP steps": int(rows.unsolved_steps.sum())})
    first = frame[frame.noise_sd_ratio == ratio].set_index("seed").loc[SEEDS]
    second = rank62_frame[rank62_frame.noise_sd_ratio == ratio].set_index("seed").loc[SEEDS]
    repaired = second.index[~first.strict_success & second.strict_success].tolist()
    regressed = second.index[first.strict_success & ~second.strict_success].tolist()
    rank_pair_rows.append({"Noise SD (%)": 100*ratio, "Repaired by rank 62": len(repaired),
        "Regressed at rank 62": len(regressed), "Net repaired": len(repaired)-len(regressed),
        "Median paired mean-error change (mm)": 1000*(second.mean_last_100-first.mean_last_100).median(),
        "Repaired seeds": repaired, "Regressed seeds": regressed})
rank_summary = pd.DataFrame(rank_rows)
rank_pairs = pd.DataFrame(rank_pair_rows).set_index("Noise SD (%)")
display(rank_summary.pivot(index="Noise SD (%)", columns="Rank",
    values=["Successes", "Median final-100 mean error (mm)", "P90 final-100 mean error (mm)"]).round(3))
display(rank_pairs.drop(columns=["Repaired seeds", "Regressed seeds"]).round(3))
display(HTML("<details><summary>Full rank-62 summary and paired seed lists</summary>"
    + rank_summary.set_index(["Rank", "Noise SD (%)"]).round(3).to_html()
    + rank_pairs.to_html() + "</details>"))
(RANK62_ROOT / "rank_comparison_report.json").write_text(rank_summary.to_json(orient="records", indent=2))
(RANK62_ROOT / "paired_rank_report.json").write_text(rank_pairs.reset_index().to_json(orient="records", indent=2))
fig, axes = plt.subplots(1, 3, figsize=(16, 4.7), constrained_layout=True)
for rank, color in [(59, "#0072B2"), (62, "#D55E00")]:
    rows = rank_summary[rank_summary.Rank == rank]
    axes[0].plot(rows["Noise SD (%)"], rows["Success rate (%)"], "o-", color=color, label=f"Rank {rank}")
    axes[1].plot(rows["Noise SD (%)"], rows["Median final-100 mean error (mm)"], "o-", color=color,
                 label=f"Rank {rank}: median")
    axes[1].plot(rows["Noise SD (%)"], rows["P90 final-100 mean error (mm)"], "s--", color=color,
                 label=f"Rank {rank}: P90")
    axes[2].plot(rows["Noise SD (%)"], rows["Mean tail time below 10 mm (%)"], "o-", color=color, label=f"Rank {rank}")
axes[0].set(title="Strict success", ylabel="success rate (%)", ylim=(0, 105))
axes[1].set(title="Final-100 mean true error", ylabel="fingertip error (mm)", ylim=(0, None))
axes[1].axhline(10, color="#666666", linestyle=":", label="mean-error criterion")
axes[2].set(title="Time near the target", ylabel="mean final-100 steps below 10 mm (%)", ylim=(0, 105))
for ax in axes:
    ax.set_xlabel("noise SD / signal SD (%)")
    ax.set_xticks(percent)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.suptitle("Paired noise ablation: rank 59 vs rank 62, lambda_g = 0.05")
fig.savefig(RANK62_ROOT / "rank_comparison_success_error.png", dpi=170)
plt.show()
fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
for ax, ratio in zip(axes.flat, SD_RATIOS):
    for rank, data, color in [(59, frame, "#0072B2"), (62, rank62_frame, "#D55E00")]:
        values = np.sort(1000*data.loc[data.noise_sd_ratio == ratio, "mean_last_100"].to_numpy())
        ax.step(np.r_[0, values], np.r_[0, np.arange(1, len(values)+1)/len(values)],
                where="post", color=color, label=f"Rank {rank}")
    ax.axvline(10, color="#666666", linestyle=":")
    ax.set(xlim=(0, 30), ylim=(0, 1.02), title=f"{100*ratio:g}% SD noise",
           xlabel="final-100 mean true error (mm)", ylabel="fraction of seeds")
    ax.grid(alpha=.2)
axes.flat[0].legend()
for ax in list(axes.flat)[len(SD_RATIOS):]:
    ax.set_visible(False)
fig.suptitle("Paired error distributions near the target; errors above 30 mm remain in the statistics")
fig.savefig(RANK62_ROOT / "rank_comparison_error_cdf.png", dpi=170)
plt.show()
'''), md("""### 7.2. Rank-62 Trajectories Near the Criterion

The same twelve fixed seeds and seven noise levels as the rank-59 section above.
The zoom retains a 0--30 mm vertical range; triangles flag samples outside it.
The second figure retains the full error range. All errors are unsmoothed simulator
truth and all 320 control steps contribute to the saved results.
"""), code('''for zoom in (True, False):
    fig, axes = plt.subplots(4, 3, figsize=(15, 13), constrained_layout=True)
    for ax, seed in zip(axes.flat, EXAMPLE_SEEDS):
        for ratio, color in zip(SD_RATIOS, COLORS):
            error = rank62_errors[ratio, seed]
            ax.plot(error, color=color, linewidth=1, label=f"{100*ratio:g}% SD")
            if zoom:
                outside = np.flatnonzero(error > 30)
                ax.scatter(outside, np.full(len(outside), 29.6), marker="^", s=8, color=color, alpha=.25, linewidths=0)
        ax.axvspan(220, 319, color="#666666", alpha=.06)
        ax.axhline(10, color="#666666", linestyle="--", linewidth=1.3)
        ax.set(ylim=(0, 30) if zoom else (0, None), xlim=(0, 319), title=f"Seed {seed}",
               xlabel="control step", ylabel="true fingertip error (mm)")
        ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(SD_RATIOS),
               title="Rank 62, lambda_g = 0.05; noise SD / signal SD")
    suffix = "zoom" if zoom else "full"
    fig.savefig(RANK62_ROOT / f"rank62_noise_example_traces_{suffix}.png", dpi=150)
    plt.show()
'''), md("""### 7.3. Computation Time by Module

Same timing definitions as rank 59: four rollout workers, one numerical-library
thread per worker, first five steps excluded, no dedicated warm-up. QP time
includes construction and solve; simulator stepping and sensor-noise generation
are excluded. Rank panels were measured in separate batches on a shared computer,
so these timings do not isolate the causal effect of rank on runtime.

Means are stacked because they add. Median and p99 module values are not stacked.
Offline clustering is reused from rank 59, while rank-62 basis construction is
performed separately and is not charged to each online control step.
"""), code('''rank_timing_rows = []
for rank, collection in [(59, payloads), (62, rank62_payloads)]:
    for ratio in SD_RATIOS:
        assert collection[ratio]["timing"]["timed_steps"] == 31500
        rank_timing_rows.extend({"Rank": rank, "Noise SD (%)": 100*ratio, **r}
                               for r in collection[ratio]["timing"]["modules"])
rank_timing = pd.DataFrame(rank_timing_rows)
display(rank_timing[rank_timing.module == "Total online control"].pivot(
    index="Noise SD (%)", columns="Rank", values=["mean_ms", "median_ms", "p99_ms"]).round(3))
display(HTML("<details><summary>All timing modules for both ranks</summary>"
    + rank_timing.set_index(["Rank", "Noise SD (%)", "module"])[["mean_ms", "median_ms", "p95_ms", "p99_ms"]]
      .round(4).to_html() + "</details>"))
(RANK62_ROOT / "rank_comparison_timing.json").write_text(rank_timing.to_json(orient="records", indent=2))
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
for ax, metric, title in zip(axes, ["median_ms", "p99_ms"], ["Median", "P99"]):
    for rank, color in [(59, "#0072B2"), (62, "#D55E00")]:
        rows = rank_timing[(rank_timing.Rank == rank) & (rank_timing.module == "Total online control")]
        ax.plot(rows["Noise SD (%)"], rows[metric], "o-", color=color, label=f"Rank {rank}")
    ax.set(ylim=(0, None), title=title, ylabel="total online ms per step", xlabel="noise SD / signal SD (%)")
    ax.set_xticks(percent)
    ax.grid(alpha=.2)
    ax.legend()
fig.savefig(RANK62_ROOT / "rank_comparison_timing.png", dpi=170)
plt.show()
rank62_timing = rank_timing[rank_timing.Rank == 62]
rank_modules = [m for m in rank62_timing.module.unique() if m != "Total online control"]
fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
bottom = np.zeros(len(SD_RATIOS))
for module, color in zip(rank_modules, module_colors):
    values = rank62_timing[rank62_timing.module == module].set_index("Noise SD (%)").loc[percent, "mean_ms"].to_numpy()
    ax.bar(np.arange(len(SD_RATIOS)), values, bottom=bottom, color=color, label=module)
    bottom += values
ax.set(ylim=(0, None), ylabel="mean ms per step", xlabel="noise SD / signal SD (%)", title="Rank 62: mean computation by module")
ax.set_xticks(np.arange(len(SD_RATIOS)), [f"{x:g}" for x in percent])
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
fig.savefig(RANK62_ROOT / "rank62_mean_timing_modules.png", dpi=170)
plt.show()
fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
for ax, module in zip(axes.flat, rank_modules):
    rows = rank62_timing[rank62_timing.module == module].set_index("Noise SD (%)").loc[percent]
    ax.bar(np.arange(len(SD_RATIOS))-.18, rows.median_ms, width=.36, color="#0072B2", label="Median")
    ax.bar(np.arange(len(SD_RATIOS))+.18, rows.p99_ms, width=.36, color="#D55E00", label="P99")
    ax.set(ylim=(0, None), title=module, ylabel="ms per step", xlabel="noise SD / signal SD (%)")
    ax.set_xticks(np.arange(len(SD_RATIOS)), [f"{x:g}" for x in percent])
    ax.grid(axis="y", alpha=.2)
axes.flat[0].legend()
fig.suptitle("Rank 62: computation-time percentiles by module")
fig.savefig(RANK62_ROOT / "rank62_timing_module_percentiles.png", dpi=170)
plt.show()
display(pd.DataFrame([{"Noise SD (%)": 100*r,
                      "Rank-62 offline bases/routes (s)": rank62_payloads[r]["artifact"]["offline_preparation_s"]["svd_cache_and_routes"],
                      "One-time goal setup median (ms)": rank62_payloads[r]["timing"]["one_time_goal_setup_ms"]["median"]}
                     for r in SD_RATIOS]).set_index("Noise SD (%)").round(3))
''')]
