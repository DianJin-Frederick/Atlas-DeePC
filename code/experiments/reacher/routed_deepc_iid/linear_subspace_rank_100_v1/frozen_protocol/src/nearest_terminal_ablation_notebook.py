"""Read-only notebook section for the paired nearest-terminal-reference ablation."""

import nbformat as nbf


def notebook_cells():
    return [nbf.v4.new_markdown_cell("""### 4.3. Small ablation: nearest-fingertip terminal reference

**Question:** Does graph-route-cost selection of the terminal configuration help
compared with simply choosing the configuration whose fingertip is closest to the target?

**Paired setup:** 100 seeds, **211--310**, with 320 control steps per seed. The current
baseline's matching saved rollouts are reused; only the nearest-reference variant
was newly simulated. This is a subset of the 1,000-seed panel, not a new held-out test.

| Component | Current baseline | Nearest-fingertip ablation |
|---|---|---|
| Candidate search at initialization | 10 closest eligible configurations; deduplicate by chart | Single closest eligible configuration |
| Terminal configuration selection | Minimum route cost from the current chart; repeated each step | Nearest fingertip in Euclidean distance; fixed throughout rollout |
| Graph-based waypoint routing | Updated each step | Updated each step, unchanged |

**Everything else is held fixed:** the same offline data, 48 physical-configuration
charts, rank-60 linear-subspace bases, initial conditions and targets, output channels,
DeePC objective and constraints, and 7 cm terminal-mode latch. In both methods,
candidate eligibility requires the source episode's original target to be at least
6 cm from the online target. The selected reference uses the sampled joint posture,
zero joint velocities, and exact online target fingertip coordinates.

**Strict success:** both final-30 and final-100 windows must have mean fingertip
error below 1 cm and maximum error below 2 cm. No early termination is used.
"""), nbf.v4.new_code_cell('''ABLATION_DIR = ROOT / "experiments/routed_deepc_iid/ablation_nearest_terminal_linear_v1"
ABLATION_BASELINE_DIR = RESULT_ROOT / "evaluation_1000"
ABLATION_SEEDS = list(range(211, 311))
ablation_payload = json.loads((ABLATION_DIR / "summary.json").read_text())
ablation_baseline_payload = json.loads((ABLATION_BASELINE_DIR / "summary.json").read_text())
assert ablation_payload["method"] == "linear_fixed_nearest_eligible_terminal_v1"
for key in ("config", "data_sha256", "atlas_sha256", "source_sha256"):
    assert ablation_payload["signature"][key] == ablation_baseline_payload["signature"][key]
ablation_baseline_rows, missing_ablation = validate_panel(ablation_baseline_payload, ABLATION_SEEDS, config)
assert not missing_ablation
ablation_nearest_rows = ablation_payload["results"]
assert len(ablation_nearest_rows) == 100
assert {r["seed"] for r in ablation_nearest_rows} == set(ABLATION_SEEDS)
assert all(r["candidate_count"] == 1 and r["matrix_geometry"] == "linear"
           and r["steps"] == config.steps and not r["dynamic_terminal_reference"]
           and r["minimum_selected_goal_source_separation"] >= config.goal_source_target_min_distance
           for r in ablation_nearest_rows)
ablation_old = {r["seed"]: r for r in ablation_baseline_rows}
paired_records = []
for r in ablation_nearest_rows:
    old = ablation_old[r["seed"]]
    with np.load(ABLATION_BASELINE_DIR / f"seed{r['seed']}_control_results.npz", allow_pickle=True) as old_trace:
        with np.load(ABLATION_DIR / f"seed{r['seed']}_control_results.npz", allow_pickle=True) as new_trace:
            assert np.unique(new_trace["goal_index"]).size == 1
            same_reference = bool(np.all(old_trace["goal_index"] == new_trace["goal_index"][0]))
    paired_records.append({
        "seed": r["seed"],
        "baseline success": bool(old["success_30"] and old["success_100"]),
        "nearest success": bool(r["success_30"] and r["success_100"]),
        "baseline final-100 (mm)": 1000*old["mean_last_100"],
        "nearest final-100 (mm)": 1000*r["mean_last_100"],
        "same reference throughout": same_reference,
    })
ablation_pairs = pd.DataFrame(paired_records).sort_values("seed")
baseline_count = int(ablation_pairs["baseline success"].sum())
nearest_count = int(ablation_pairs["nearest success"].sum())
repaired = ablation_pairs.loc[~ablation_pairs["baseline success"] & ablation_pairs["nearest success"], "seed"].tolist()
regressed = ablation_pairs.loc[ablation_pairs["baseline success"] & ~ablation_pairs["nearest success"], "seed"].tolist()
display(pd.DataFrame({
    "Terminal-reference policy": ["N=10, minimum route cost (current baseline)", "Single nearest eligible fingertip (fixed)"],
    "Successes / trials": [f"{baseline_count}/100", f"{nearest_count}/100"],
    "Strict success rate": [f"{baseline_count:.1f}%", f"{nearest_count:.1f}%"],
}))
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
bars = ax.bar(["N=10: minimum route cost", "Single nearest fingertip"],
              [baseline_count, nearest_count], color=["#0072B2", "#D55E00"], width=0.55)
for bar, count in zip(bars, [baseline_count, nearest_count]):
    ax.text(bar.get_x()+bar.get_width()/2, count+1, f"{count}/100 ({count:.0f}%)", ha="center")
ax.set(ylim=(0, 105), ylabel="strict success rate (%)",
       title="Terminal-reference selection: paired 100-seed ablation (211--310)")
ax.grid(axis="y", alpha=0.2)
fig.savefig(ABLATION_DIR / "success_rate_comparison.png", dpi=160)
plt.show()
changed = ablation_pairs.loc[ablation_pairs["baseline success"] != ablation_pairs["nearest success"]].copy()
changed["Outcome with nearest reference"] = np.where(changed["nearest success"], "Repaired baseline failure", "Lost baseline success")
display(changed[["seed", "Outcome with nearest reference", "baseline final-100 (mm)", "nearest final-100 (mm)"]]
        .style.hide(axis="index").format({"baseline final-100 (mm)": "{:.3f}", "nearest final-100 (mm)": "{:.3f}"}))
same_count = int(ablation_pairs["same reference throughout"].sum())
display(HTML(
    f"<p><strong>Takeaway:</strong> nearest-fingertip selection repaired {len(repaired)} baseline failure(s) "
    f"and lost {len(regressed)} baseline success(es), changing the success rate by "
    f"{nearest_count-baseline_count:+d} percentage points on this panel. "
    f"Repaired seeds: {repaired}; regressed seeds: {regressed}.</p>"
    f"<p>The selected reference was identical throughout the rollout in {same_count}/100 cases. "
    "Thus nearest fingertip and route-cost selection are not equivalent. "
    "This small exploratory panel favors route-cost selection, but does not establish "
    "a general advantage or separate its initial selection benefit from online reselection.</p>"
))
print("Unsolved QP steps (nearest-reference variant):", sum(r["unsolved_steps"] for r in ablation_nearest_rows))
'''), nbf.v4.new_markdown_cell("""**Scope and reproducibility.** These seeds were inspected in prior experiments.
The new runs used four concurrent workers, so their wall-clock timings are **not**
a controlled speed comparison with the serial baseline benchmark in Section 4.2.
No baseline settings or existing videos were changed by this ablation.

Results: `experiments/routed_deepc_iid/ablation_nearest_terminal_linear_v1/`.
The section above reads saved results only. To run/resume the experiment separately:

```bash
python src/ablation_nearest_terminal_reference.py --seed-start 211 --seed-stop 310 --workers 4
```
""")]
