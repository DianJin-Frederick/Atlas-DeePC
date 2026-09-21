"""Build the lambda_g=0 repeat, preserving the earlier executed notebook."""

from copy import deepcopy
import build_linear_subspace_rank_notebook as template

OUTPUT = template.ROOT / "notebooks/ablations/ablation_linear_subspace_rank_lamg0.ipynb"
OLD_VERSION = "linear_subspace_rank_100_v1"
VERSION = "linear_subspace_rank_lamg0_100_v1"
cells = []
for original in template.cells:
    if original.cell_type == "markdown" and original.source.startswith("## Follow-up: why ranks 59 and 60 differ"):
        continue
    cell = deepcopy(original)
    cell.source = cell.source.replace(OLD_VERSION, VERSION)
    cell.source = cell.source.replace(r"$\lambda_g=.05$", r"$\lambda_g=0$")
    cell.source = cell.source.replace("datasets. The live baseline in `src/` now belongs to a newer protocol.",
                                      "datasets. Frozen sources protect this repeat from later workspace edits.")
    if cell.cell_type == "markdown" and cell.source.startswith("# Atlas-DeePC:"):
        start = cell.source.index("The shared workspace's baseline changed")
        end = cell.source.index("## Fixed setup")
        cell.source = cell.source[:start] + r"""This is a fresh repeat with **$\lambda_g=0$**, removing coefficient-norm
regularization. Relative to the previous $\lambda_g=0.05$ sweep, all other
controller parameters, chart geometry, goal policy, offline data, and evaluation
seeds are unchanged. The isolated source snapshot differs only in this default
and experiment/version identifiers. The prior results are retained separately
and compared below; none are relabeled as new runs.

""" + cell.source[end:]
        cell.source = cell.source.replace("# Atlas-DeePC: Linear-Subspace Rank", "# Atlas-DeePC: Linear-Subspace Rank, λg = 0", 1)
    if cell.cell_type == "code":
        cell.source = cell.source.replace('assert experiment.prepare() == payload["signature"], "Experiment sources/data changed."',
            'assert experiment.prepare() == payload["signature"], "Experiment sources/data changed."\nassert payload["signature"]["base_config"]["lam_g"] == 0.0')
        cell.source = cell.source.replace('report = {"complete": True,', 'report = {"lam_g": 0.0, "complete": True,')
    cells.append(cell)

comparison_md = template.md(r"""## Paired comparison: λg = 0 versus 0.05

Both sweeps use the same rank/seed grid, targets, frozen atlas and control setup.
Only coefficient regularization changes. A repaired seed fails with λg=0.05 and
succeeds with λg=0 at that same rank; a regressed seed does the reverse. Both
methods retain the original full-duration strict-success test and QP-failure
handling. These are reused exploratory seeds.

The two sweeps occurred at different times on a shared computer. Their timings
cannot isolate the effect of coefficient regularization from changing system load.
""")
comparison_code = template.code('''OLD_ROOT = ROOT / "experiments/routed_deepc_iid/linear_subspace_rank_100_v1"
old_payload = json.loads((OLD_ROOT / "summary.json").read_text())
assert old_payload["complete"] and old_payload["ranks"] == RANKS and old_payload["seeds"] == SEEDS
old_signature = old_payload["signature"]
new_signature = payload["signature"]
assert old_signature["data_sha256"] == new_signature["data_sha256"]
assert old_signature["atlas_sha256"] == new_signature["atlas_sha256"]
config_changes = {key: [value, new_signature["base_config"][key]]
    for key, value in old_signature["base_config"].items()
    if value != new_signature["base_config"][key]}
assert config_changes == {"lam_g": [.05, 0.0]}
for name, digest in old_signature["source_sha256"].items():
    assert experiment.file_digest(OLD_ROOT / "frozen_protocol/src" / name) == digest
comparison_rows = []
new_rows = frame.set_index(["rank", "seed"])
for rank in RANKS:
    old_panel = json.loads((OLD_ROOT / f"rank{rank:03d}" / "summary.json").read_text())
    assert old_panel["seeds"] == SEEDS and [r["seed"] for r in old_panel["results"]] == SEEDS
    repaired, regressed = [], []
    for old_row in old_panel["results"]:
        seed = old_row["seed"]
        expected = {**old_signature["base_config"], "seed": seed, "svd_rank": rank}
        assert all(old_row[key] == value for key, value in expected.items())
        new_row = new_rows.loc[(rank, seed)]
        assert new_row.lam_g == 0.0
        np.testing.assert_array_equal(new_row.exact_target_xy, old_row["exact_target_xy"])
        old_success = experiment.strict_success(old_row)
        if new_row.strict and not old_success: repaired.append(seed)
        if old_success and not new_row.strict: regressed.append(seed)
    comparison_rows.append({"rank": rank, "success_lamg0": int(summary.loc[rank, "successes"]),
        "success_lamg005": old_panel["successes"],
        "median_mm_lamg0": float(summary.loc[rank, "median_tail100_mm"]),
        "median_mm_lamg005": old_panel["median_tail100_mm"],
        "p90_mm_lamg0": float(summary.loc[rank, "p90_tail100_mm"]),
        "p90_mm_lamg005": old_panel["p90_tail100_mm"],
        "unsolved_lamg0": int(summary.loc[rank, "unsolved_qps"]),
        "unsolved_lamg005": old_panel["unsolved_qps"],
        "net_success_change": len(repaired)-len(regressed),
        "repaired_seeds": repaired, "regressed_seeds": regressed})
comparison = pd.DataFrame(comparison_rows).set_index("rank")
highlights = sorted({40, 50, 56, 59, 60, 62, 64, 80, *best_ranks})
display(comparison.loc[highlights, ["success_lamg0", "success_lamg005", "median_mm_lamg0", "median_mm_lamg005", "net_success_change"]].round(3))
fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
for ax, prefix, label in zip(axes.flat[:3], ["success", "median_mm", "p90_mm"],
        ["Strict successes /100", "Median final-100 mean error (mm)", "P90 final-100 mean error (mm)"]):
    for suffix, color, legend in [("lamg0", BLUE, "λg = 0"), ("lamg005", ORANGE, "λg = 0.05")]:
        ax.plot(RANKS, comparison[f"{prefix}_{suffix}"], color=color, marker=".", label=legend)
    rank_axis(ax, label)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
axes.flat[0].set_ylim(0, 105)
axes.flat[3].plot(RANKS, comparison.net_success_change, color=BLUE, marker=".")
axes.flat[3].axhline(0, color=GRAY)
rank_axis(axes.flat[3], "Success change: λg = 0 minus 0.05")
save_show(fig, "rank_lambda_comparison.png")
display(HTML("<details><summary>All 61 paired regularization comparisons and seed lists</summary>" + comparison.round(3).to_html() + "</details>"))
experiment.write_json(RESULT_ROOT / "lambda_comparison_report.json", {
    "complete": True, "paired_rollouts": 6100, "configuration_differences": config_changes,
    "ranks": comparison.reset_index().to_dict(orient="records")})
print("Verified paired comparison: 6,100 matching rank/seed/target cases; only lambda_g changed.")
''')
cells[-1:-1] = [comparison_md, comparison_code]

if __name__ == "__main__":
    template.build(OUTPUT, cells)
