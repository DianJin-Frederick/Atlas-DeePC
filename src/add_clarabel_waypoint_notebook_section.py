"""Add the matched lambda_g=0.05 ablation without replacing historical outputs."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks/baselines/baseline_graph_routed_atlas_deepc.ipynb"


def update():
    notebook = nbf.read(PATH, as_version=4)
    notebook.cells = [c for c in notebook.cells if not c.metadata.get("clarabel_waypoint_ablation")]
    cells = [nbf.v4.new_markdown_cell(r'''### 4.4. With and Without Graph Waypoints: Clarabel, $\lambda_g=0.05$

Both methods use the same frozen offline data, 48 physical charts, rank-60 linear
subspaces, Q/R weights, 320-step rollout, and iterative N=10 terminal-reference
selection. **Only intermediate graph waypoints are removed** in the ablation:
DeePC tracks the selected terminal configuration from the first step. The terminal
configuration can still change through the common route-cost selection policy.
Thus "without graph waypoints" does not mean that all graph usage is removed.

Set `RUN_WAYPOINT_COMPARISON=True` in the next cell to run the missing no-waypoint
rollouts on the same 1,000 seeds. Matching completed rollouts resume automatically;
the historical graph results and videos are not overwritten. One rollout runs at
a time, with one numerical thread. No simulations start in saved-view mode.

For a fresh timing comparison of both methods, also set
`REMEASURE_BOTH=True`: a new output folder is created and both methods run in
alternating order per seed without reusing old timings. Otherwise graph timing is
historical and must not be interpreted as a controlled speed comparison against
the new no-waypoint run. All timings exclude the first five rollout steps.
'''), nbf.v4.new_code_cell('''import baseline_clarabel_waypoint_comparison as clarabel_comparison
from baseline_1000_benchmark import summarize_timing

WAYPOINT_SEEDS = list(EVALUATION_SEEDS)
RUN_WAYPOINT_COMPARISON = False
REMEASURE_BOTH = False
WAYPOINT_RESULT_DIR = clarabel_comparison.RESULT_ROOT
assert config.lam_g == 0.05 and SOLVER == "CLARABEL"
if RUN_WAYPOINT_COMPARISON:
    if REMEASURE_BOTH:
        from datetime import datetime, timezone
        from uuid import uuid4
        WAYPOINT_RESULT_DIR = WAYPOINT_RESULT_DIR / (
            "fresh_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    print("Comparison output:", WAYPOINT_RESULT_DIR, flush=True)
    clarabel_comparison.run_panel(
        WAYPOINT_SEEDS, WAYPOINT_RESULT_DIR,
        modes=("terminal", "graph") if REMEASURE_BOTH else ("terminal",), config=config)
else:
    print("Saved-result view. Set RUN_WAYPOINT_COMPARISON=True to run missing no-waypoint seeds.")

no_waypoint_rows = clarabel_comparison.load_rows(WAYPOINT_RESULT_DIR, "terminal", WAYPOINT_SEEDS)
fresh_graph_rows = clarabel_comparison.load_rows(WAYPOINT_RESULT_DIR, "graph", WAYPOINT_SEEDS)
historical_payload = json.loads((EVAL_DIR / "summary.json").read_text())
if no_waypoint_rows:
    comparison_manifest = json.loads((WAYPOINT_RESULT_DIR / "manifest.json").read_text())
    for key in ("data_sha256", "atlas_sha256"):
        assert comparison_manifest[key] == historical_payload["signature"][key]
historical_graph_rows, historical_missing = validate_panel(historical_payload, WAYPOINT_SEEDS, config)
assert not historical_missing, "The historical comparison panel does not cover all requested seeds."
graph_rows_for_comparison = fresh_graph_rows if fresh_graph_rows else historical_graph_rows
graph_is_fresh = bool(fresh_graph_rows)
comparison_rows = {"terminal": no_waypoint_rows, "graph": graph_rows_for_comparison}
completion = pd.DataFrame([{
    "Method": clarabel_comparison.METHODS[mode], "Completed": len(rows),
    "Requested": len(WAYPOINT_SEEDS),
    "Result source": "new comparison" if mode == "terminal" or graph_is_fresh else "historical saved graph benchmark",
    "Success rate on requested panel (%)":
        100*np.mean([r["success_30"] and r["success_100"] for r in rows])
        if len(rows) == len(WAYPOINT_SEEDS) else np.nan,
} for mode, rows in comparison_rows.items()])
display(completion)
paired_seeds = sorted(set(r["seed"] for r in no_waypoint_rows) & set(r["seed"] for r in graph_rows_for_comparison))
print(f"Completed matched subset: {len(paired_seeds)}/{len(WAYPOINT_SEEDS)} seeds.")
print("Graph timing source:", "new comparison" if graph_is_fresh else "historical; not a controlled speed comparison")
'''), nbf.v4.new_code_cell('''if not paired_seeds:
    print("No paired results yet. Existing graph results remain displayed in Sections 4.1 and 4.2.")
else:
    paired_rows = {mode: {r["seed"]: r for r in rows if r["seed"] in paired_seeds}
                   for mode, rows in comparison_rows.items()}
    timing_records, outcome_records = [], []
    for mode, rows in paired_rows.items():
        paths = [(EVAL_DIR / f"seed{s}_control_results.npz")
                 if mode == "graph" and not graph_is_fresh else
                 (WAYPOINT_RESULT_DIR / mode / f"seed{s}_control_results.npz") for s in paired_seeds]
        timing = summarize_timing(paths)
        timing_records.extend({"Method": clarabel_comparison.METHODS[mode], **row} for row in timing["modules"])
        outcome_records.append({"Method": clarabel_comparison.METHODS[mode],
            "Seeds": len(rows), "Successes": sum(r["success_30"] and r["success_100"] for r in rows.values()),
            "Median final-100 error (mm)": 1000*np.median([r["mean_last_100"] for r in rows.values()]),
            "Unsolved QPs": sum(r["unsolved_steps"] for r in rows.values())})
    outcomes = pd.DataFrame(outcome_records).set_index("Method")
    outcomes["Success rate (%)"] = 100*outcomes["Successes"]/outcomes["Seeds"]
    display(outcomes.round(3))
    graph_only = [s for s in paired_seeds
                  if paired_rows["graph"][s]["success_30"] and paired_rows["graph"][s]["success_100"]
                  and not (paired_rows["terminal"][s]["success_30"] and paired_rows["terminal"][s]["success_100"])]
    terminal_only = [s for s in paired_seeds
                     if paired_rows["terminal"][s]["success_30"] and paired_rows["terminal"][s]["success_100"]
                     and not (paired_rows["graph"][s]["success_30"] and paired_rows["graph"][s]["success_100"])]
    display(pd.Series({"With-waypoints-only successes": len(graph_only),
                       "Without-waypoints-only successes": len(terminal_only),
                       "Net successes added by waypoints": len(graph_only)-len(terminal_only)}))
    display(HTML("<details><summary>Seeds with different outcomes</summary>" +
        pd.Series({"With waypoints only": graph_only, "Without waypoints only": terminal_only})
        .to_frame().to_html() + "</details>"))
    timing_table = pd.DataFrame(timing_records)
    display(timing_table.pivot(index="module", columns="Method", values=["mean_ms", "median_ms", "p99_ms"]).round(4))
    fig, axes = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)
    labels = ["Without waypoints", "With waypoints"]
    colors = ["#D55E00", "#0072B2"]
    rates = outcomes["Success rate (%)"].to_numpy()
    bars = axes[0].bar(labels, rates, color=colors)
    axes[0].bar_label(bars, fmt="%.1f%%", padding=3)
    axes[0].set(ylim=(0, 105), ylabel="strict success rate (%)")
    totals = timing_table[timing_table.module == "Total online control"]
    for ax, metric, label in zip(axes[1:], ["mean_ms", "median_ms", "p99_ms"], ["Mean", "Median", "P99"]):
        values = totals[metric].to_numpy()
        bars = ax.bar(labels, values, color=colors)
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set(ylim=(0, values.max()*1.2), ylabel=f"{label} online time (ms)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=.2)
    note = "new paired timing run" if graph_is_fresh else "timings from separate sessions; descriptive only"
    fig.suptitle(f"Clarabel, lambda_g=0.05 | {len(paired_seeds)} matched seeds | {note}")
    fig.savefig(WAYPOINT_RESULT_DIR / "waypoint_comparison.png", dpi=160, bbox_inches="tight")
    plt.show()
    report = {"solver": "CLARABEL", "lambda_g": .05, "seeds": paired_seeds,
              "outcomes": outcomes.reset_index().to_dict(orient="records"),
              "module_timing": timing_records, "timing_caveat": note,
              "with_waypoints_only": graph_only, "without_waypoints_only": terminal_only}
    (WAYPOINT_RESULT_DIR / "paired_comparison_report.json").write_text(json.dumps(report, indent=2))
''')]
    for cell in cells:
        cell.metadata["clarabel_waypoint_ablation"] = True
    index = next(i for i, c in enumerate(notebook.cells) if c.cell_type == "markdown" and c.source.startswith("## 5."))
    notebook.cells[index:index] = cells
    if "A no-graph-waypoints comparison" not in notebook.cells[0].source:
        notebook.cells[0].source += "\nA no-graph-waypoints comparison using the same Clarabel and lambda_g=0.05 settings is available in Section 4.4.\n"
    nbf.validate(notebook)
    nbf.write(notebook, PATH)


if __name__ == "__main__":
    update()
