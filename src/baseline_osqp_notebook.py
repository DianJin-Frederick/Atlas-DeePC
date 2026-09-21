"""Additional notebook cells for the direct OSQP substitution experiment."""

import nbformat as nbf


def notebook_cells():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    return [md(r"""## 8. OSQP Solver Comparison

This experiment changes **only the QP solver** from Clarabel to OSQP for both
baseline variants. Lambda_g remains zero. The data, 48 charts, rank 60, N=10
terminal selection, costs, constraints, seeds and 320-step rollout length remain
the same. The Clarabel results above are preserved.

The existing controller creates a new CVXPY Problem at each step. Consequently,
this is a **direct solver substitution**, not an optimized persistent OSQP
workspace with cross-step warm starts or cached factorizations.

OSQP uses the existing 10,000-iteration cap and the installed CVXPY adapter's
defaults: absolute and relative tolerances 1e-5, with polishing enabled. The
manifest records CVXPY and OSQP versions. A non-solved QP retains the existing
zero-action behavior; there is no fallback to Clarabel. Such steps remain in
the success/error statistics and are counted explicitly.

Success outcomes use all requested seeds. The latency comparison excludes pilot
seeds 222 and 237 from **both** solvers: those OSQP compatibility tests used two
workers, while the main panel uses four. For the default range this gives 998
paired timing seeds, discarding the first five steps of each rollout. Measurements
remain shared-computer wall-clock times, not an isolated hardware benchmark.
"""), code('''import baseline_osqp_comparison as osqp_experiment
RUN_OSQP = False
OSQP_RESULT_ROOT = osqp_experiment.RESULT_ROOT
if RUN_OSQP:
    osqp_experiment.run_panel(EVALUATION_SEEDS, OSQP_RESULT_ROOT, WORKERS)
osqp_payload = json.loads((OSQP_RESULT_ROOT / "summary.json").read_text())
assert osqp_payload["version"] == osqp_experiment.VERSION
assert osqp_payload["signature"]["solver"] == "OSQP"
assert osqp_payload["signature"]["config"] == payload["signature"]["config"] == asdict(config)
for key in ("data_sha256", "atlas_sha256"):
    assert osqp_payload["signature"][key] == payload["signature"][key]
for name, digest in payload["signature"]["source_sha256"].items():
    assert osqp_payload["signature"]["source_sha256"][name] == digest
solver_rows = {"Clarabel": rows_by_mode, "OSQP": {}}
for mode in METHODS:
    rows = {int(r["seed"]): r for r in osqp_payload["panels"][mode]["results"]}
    missing = sorted(set(EVALUATION_SEEDS)-set(rows))
    assert not missing, f"OSQP {mode}: missing {len(missing)} requested seeds. Enable RUN_OSQP."
    solver_rows["OSQP"][mode] = {s: rows[s] for s in EVALUATION_SEEDS}
    for seed, row in solver_rows["OSQP"][mode].items():
        assert row["solver"] == "OSQP" and row["lam_g"] == 0
        assert row["solver_ablation_version"] == osqp_experiment.VERSION
        assert row["matrix_geometry"] == "linear" and row["candidate_count"] == 10
        assert row["steps"] == config.steps and row["reference_mode"] == mode
        assert row["minimum_selected_goal_source_separation"] >= .06
        np.testing.assert_array_equal(row["exact_target_xy"], solver_rows["Clarabel"][mode][seed]["exact_target_xy"])
solver_report = pd.DataFrame([{
    "Waypoints": "With" if mode == "graph" else "Without", "Solver": solver,
    "Successes": sum(bool(r["success_30"] and r["success_100"]) for r in rows.values()),
    "Seeds": len(rows),
    "Success rate (%)": 100*np.mean([r["success_30"] and r["success_100"] for r in rows.values()]),
    "Median final-100 mean (mm)": 1000*np.median([r["mean_last_100"] for r in rows.values()]),
    "P90 final-100 mean (mm)": 1000*np.percentile([r["mean_last_100"] for r in rows.values()],90),
    "Unsolved QPs": sum(r["unsolved_steps"] for r in rows.values()),
} for mode in METHODS for solver, modes in solver_rows.items() for rows in [modes[mode]]])
display(solver_report.set_index(["Waypoints","Solver"]).round(3))
paired_solver_changes = []
for mode in METHODS:
    a = solver_rows["Clarabel"][mode]
    b = solver_rows["OSQP"][mode]
    success = lambda row: bool(row["success_30"] and row["success_100"])
    osqp_only = [s for s in EVALUATION_SEEDS if success(b[s]) and not success(a[s])]
    clarabel_only = [s for s in EVALUATION_SEEDS if success(a[s]) and not success(b[s])]
    paired_solver_changes.append({"Waypoints": "With" if mode == "graph" else "Without",
        "OSQP-only successes": len(osqp_only), "Clarabel-only successes": len(clarabel_only),
        "Net change with OSQP": len(osqp_only)-len(clarabel_only)})
display(pd.DataFrame(paired_solver_changes).set_index("Waypoints"))
solver_report.to_json(OSQP_RESULT_ROOT/"solver_comparison_report.json",orient="records",indent=2)
SOLVER_COLORS = {"Clarabel":"#0072B2", "OSQP":"#D55E00"}
fig, axes = plt.subplots(1,3,figsize=(16,4.7),constrained_layout=True)
for i, solver in enumerate(solver_rows):
    values = [100*np.mean([r["success_30"] and r["success_100"] for r in solver_rows[solver][mode].values()]) for mode in METHODS]
    bars = axes[0].bar(np.arange(2)+(i-.5)*.34, values, width=.34, color=SOLVER_COLORS[solver],label=solver)
    axes[0].bar_label(bars,fmt="%.1f%%",padding=3)
    for ax,mode in zip(axes[1:],METHODS):
        values = np.sort([1000*r["mean_last_100"] for r in solver_rows[solver][mode].values()])
        ax.step(np.r_[0,values],np.r_[0,np.arange(1,len(values)+1)/len(values)],where="post",color=SOLVER_COLORS[solver],linestyle="--" if solver=="OSQP" else "-",linewidth=2,label=solver)
axes[0].set(xticks=[0,1],xticklabels=["Without waypoints","With waypoints"],ylim=(0,110),ylabel="strict success (%)")
axes[0].legend()
for ax,mode in zip(axes[1:],METHODS):
    ax.set(xlim=(0,30),ylim=(0,1.02),xlabel="final-100 mean fingertip error (mm)",ylabel="fraction of seeds",title=METHODS[mode])
    ax.axvline(10,color="#667085",linestyle="--")
for ax in axes: ax.grid(axis="y",alpha=.2)
fig.suptitle(f"Direct solver substitution | lambda_g=0 | {len(EVALUATION_SEEDS)} paired seeds")
fig.savefig(OSQP_RESULT_ROOT/"osqp_success_and_error.png",dpi=160)
plt.show()
'''), code('''TIMING_SEEDS = [s for s in EVALUATION_SEEDS if s not in {222,237}]
assert TIMING_SEEDS, "Timing needs at least one seed other than the two pilot seeds."
solver_timings = {}
for solver, directory in [("Clarabel",RESULT_ROOT),("OSQP",OSQP_RESULT_ROOT)]:
    for mode in METHODS:
        solver_timings[solver,mode] = summarize_timing([directory/mode/f"seed{s}_control_results.npz" for s in TIMING_SEEDS])
        assert solver_timings[solver,mode]["timed_steps"] == len(TIMING_SEEDS)*(config.steps-5)
solver_timing_table = pd.DataFrame([{
    "Solver":solver,"Waypoints":"With" if mode=="graph" else "Without",**row
} for (solver,mode), data in solver_timings.items() for row in data["modules"]])
display(pd.Series({"paired timing seeds":len(TIMING_SEEDS),"timed steps per method/solver":len(TIMING_SEEDS)*(config.steps-5),"main-panel workers":4}))
display(solver_timing_table.pivot(index="module",columns=["Waypoints","Solver"],values=["mean_ms","median_ms","p99_ms"]).round(4))
fig, axes = plt.subplots(1,3,figsize=(15,4.7),constrained_layout=True)
for ax,stat,label in zip(axes,["mean_ms","median_ms","p99_ms"],["Mean","Median","P99"]):
    for i,solver in enumerate(solver_rows):
        values = [solver_timing_table.loc[(solver_timing_table.Solver==solver)&(solver_timing_table.Waypoints==("With" if mode=="graph" else "Without"))&(solver_timing_table.module=="Total online control"),stat].item() for mode in METHODS]
        bars=ax.bar(np.arange(2)+(i-.5)*.34,values,width=.34,color=SOLVER_COLORS[solver],label=solver)
        ax.bar_label(bars,fmt="%.2f",padding=3)
    ax.set(xticks=[0,1],xticklabels=["Without waypoints","With waypoints"],ylabel=f"{label} online time per step (ms)")
    ax.margins(y=.2)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y",alpha=.2)
axes[0].legend()
fig.suptitle(f"Shared-computer timing | {len(TIMING_SEEDS)} paired seeds | first five steps excluded")
fig.savefig(OSQP_RESULT_ROOT/"osqp_online_timing.png",dpi=160)
plt.show()
solver_timing_table.to_json(OSQP_RESULT_ROOT/"solver_module_timing_report.json",orient="records",indent=2)
'''), md("""### OSQP Numerical Diagnostics

These diagnostic summaries include all 320 control steps and all requested seeds,
including the two pilot seeds. Iteration/residual statistics use steps for which
CVXPY returned solver statistics. Exceptions can lack those statistics and are
counted separately. `optimal_inaccurate` is accepted by the unchanged controller;
it is not silently treated as an exact solve. The reported OSQP backend time is
not the full controller latency shown above.
"""), code('''osqp_diagnostics = {}
diagnostic_rows = []
for mode in METHODS:
    records = []
    statuses = []
    for seed in EVALUATION_SEEDS:
        with np.load(OSQP_RESULT_ROOT/mode/f"seed{seed}_control_results.npz",allow_pickle=True) as trace:
            rec = json.loads(trace["solver_diagnostics"].item())
            assert len(rec)==config.steps and all(r["solver"]=="OSQP" for r in rec)
            np.testing.assert_allclose(trace["distance"],np.linalg.norm(trace["y"][:,6:8]-trace["final_goal"].reshape(-1)[6:8],axis=1))
            records.extend(rec)
            statuses.extend(trace["status"].astype(str).tolist())
    osqp_diagnostics[mode] = records
    iterations = [r["iterations"] for r in records if r["iterations"] is not None]
    backend_ms = [1000*r["solver_reported_time_s"] for r in records if r["solver_reported_time_s"] is not None]
    diagnostic_rows.append({"Waypoints":"With" if mode=="graph" else "Without",
        "Total QPs":len(records),"Optimal":statuses.count("optimal"),
        "Optimal inaccurate":statuses.count("optimal_inaccurate"),
        "Exceptions":sum(r["error"] is not None for r in records),
        "Steps with iteration stats":len(iterations),
        "Median iterations":np.median(iterations) if iterations else np.nan,
        "P99 iterations":np.percentile(iterations,99) if iterations else np.nan,
        "Median backend-reported time (ms)":np.median(backend_ms) if backend_ms else np.nan})
diagnostic_report=pd.DataFrame(diagnostic_rows).set_index("Waypoints")
display(diagnostic_report.round(3))
diagnostic_report.to_json(OSQP_RESULT_ROOT/"osqp_diagnostic_report.json",orient="index",indent=2)
''')]
