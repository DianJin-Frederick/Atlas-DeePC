"""Build the merged six-rank report from saved results; never run experiments."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from atlas_deepc.common.paths import project_root
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")
import nbformat as nbf

ROOT = project_root()
RESULT = ROOT / "experiments/reacher/routed_deepc_iid/selected_ranks_1000_clarabel_v1/merged_report"
NOTEBOOK = ROOT / "notebooks/reacher/ablations/ablation_selected_ranks_1000.ipynb"


def build():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md("""# Atlas-DeePC: selected ranks over 1,000 seeds

**Ranks 58, 59, 61, 62, 63 and 64 · seeds 311–1310 · 6,000 complete rollouts.**
This consolidated report merges the original ranks 59/62/64 with the additional ranks 58/61/63. It reads saved checkpoints only and never launches simulations. Rank 60 is not part of these two source studies.

All six ranks use **λg = 0.05, Clarabel, 48 fixed physical charts, uncentered linear bases, graph waypoints and iterative N=10 terminal references**, with 320 steps per rollout. Offline data, memberships, centers and routing graph are identical. Only the retained basis rank changes; coefficient bounds use the same rule in each rank's coordinates.

Strict success requires both final-30 and final-100 mean error <10 mm and maximum error <20 mm. Error quantiles summarize each seed's final-100 mean error, **including failures**. Confidence intervals are 95% Wilson intervals. These seeds have been inspected in earlier studies; this is an exploratory comparison, not an untouched validation panel.

**Timing provenance:** session A evaluated ranks 59/62/64; session B evaluated ranks 58/61/63. Each used one worker, one numerical thread and rotated rank order within each seed. They ran in separate sessions on a shared computer. Timing differences across sessions cannot be attributed to rank alone. Each rank's timing pools 315,000 control steps after excluding the first five per rollout; all 320 samples remain in the source files. QP time includes construction and solve. Setup and rollout wall times are reported separately.

The original source results and plots remain unchanged. The earlier three-rank notebook is archived at `archive/notebooks/ablation_selected_ranks_1000_before_merge_20260921.ipynb`; the adjacent-ranks notebook remains available as its original report. All merged exports go to `selected_ranks_1000_clarabel_v1/merged_report/`.
"""), code('''RUN_EXPERIMENTS = False
from pathlib import Path
from collections import Counter
from itertools import combinations
import hashlib, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from IPython.display import display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "src/atlas_deepc/notebook_tools/build_selected_rank_1000_notebook.py").is_file())
BASE = ROOT / "experiments/reacher/routed_deepc_iid"
RESULT = BASE / "selected_ranks_1000_clarabel_v1/merged_report"
FIGURES = RESULT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
RANKS = [58, 59, 61, 62, 63, 64]
SEEDS = list(range(311, 1311))
SOURCES = {"A": (BASE / "selected_ranks_1000_clarabel_v1", [59, 62, 64]),
           "B": (BASE / "adjacent_ranks_1000_clarabel_v1", [58, 61, 63])}
STAGES = {"Chart localization": "localization_time_s",
          "Terminal-reference selection": "goal_selection_time_s",
          "Graph routing": "routing_time_s",
          "Linear subspace retrieval": "matrix_preparation_time_s",
          "QP construction and solve": "solve_time_s"}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

summaries, reports, rows, times, sessions = {}, {}, {}, {}, {}
source_hashes, checkpoint_hashes = {}, {}
for session, (folder, ranks) in SOURCES.items():
    summary = json.loads((folder / "summary.json").read_text())
    sig = summary["signature"]
    assert summary["complete"] and summary["rollouts"] == 3000
    assert summary["ranks"] == ranks and summary["seeds"] == SEEDS
    assert json.loads((folder / "manifest.json").read_text()) == sig
    assert sig["base_config"]["lam_g"] == .05 and sig["base_config"]["clusters"] == 48
    assert sig["base_config"]["steps"] == 320 and sig["candidate_count"] == 10
    assert sig["matrix_geometry"] == "linear" and sig["solver"] == "CLARABEL"
    assert sig["workers"] == sig["numerical_threads"] == 1
    assert sig["timing_discard_initial_steps"] == 5
    frozen = folder / "frozen_protocol"
    checks = {frozen / "data/reacher/select_iid_offline_hankel.npy": sig["data_sha256"],
              frozen / "experiments/reacher/routed_deepc_iid/clustering_jointconfig48_fresh211_310/behavioral_atlas.npz": sig["atlas_sha256"],
              frozen / "src/selected_rank_1000_experiment.py": sig["runner_sha256"]}
    checks.update({frozen / "src" / name: sha for name, sha in
                   {**sig["source_sha256"], **sig["dependency_sha256"]}.items()})
    # Source drift is disclosed for reporting; data hashes remain mandatory.
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from atlas_deepc.common.provenance import validate_report_files
    validate_report_files(checks)
    invariance = json.loads((folder / "invariance_latest.json").read_text())
    assert invariance["passed"] and invariance["through_seed"] == 1310
    summaries[session] = summary
    source_hashes[str((folder / "summary.json").relative_to(ROOT))] = digest(folder / "summary.json")
    for rank in ranks:
        report_path = folder / f"rank{rank:03d}/summary.json"
        report = json.loads(report_path.read_text())
        source_hashes[str(report_path.relative_to(ROOT))] = digest(report_path)
        checked, collected = [], {k: [] for k in [*STAGES, "Other control-loop work", "Total online control"]}
        paths = [folder / f"rank{rank:03d}/seed{seed}.npz" for seed in SEEDS]
        assert set(paths) == set((folder / f"rank{rank:03d}").glob("seed*.npz"))
        for seed, path in zip(SEEDS, paths):
            checkpoint_hashes[str(path.relative_to(ROOT))] = digest(path)
            with np.load(path, allow_pickle=False) as trace:
                row = json.loads(trace["summary"].item())
                expected = dict(sig["base_config"], seed=seed, svd_rank=rank)
                assert all(row[k] == v for k, v in expected.items())
                assert row["rank_ablation_version"] == sig["version"]
                assert row["matrix_geometry"] == "linear" and row["candidate_count"] == 10
                assert row["reference_mode"] == "graph" and row["dynamic_terminal_reference"]
                assert row["terminal_reference_policy"] == "iterative_minimum_route_cost"
                assert sum(row["statuses"].values()) == 320
                assert row["unsolved_steps"] == sum(v for k,v in row["statuses"].items() if k not in ("optimal", "optimal_inaccurate"))
                for key in ("distance", *STAGES.values(), "control_compute_time_s"):
                    values = trace[key]
                    assert values.shape == (320,) and np.isfinite(values).all() and (values >= 0).all()
                for window in (30, 100):
                    error = trace["distance"][-window:]
                    np.testing.assert_allclose([row[f"mean_last_{window}"], row[f"max_last_{window}"]],
                                               [error.mean(), error.max()], rtol=1e-12, atol=1e-15)
                    assert row[f"success_{window}"] == bool(error.mean() < .01 and error.max() < .02)
                parts = {name: 1000*trace[key][5:] for name,key in STAGES.items()}
                total = 1000*trace["control_compute_time_s"][5:]
                other = total - sum(parts.values())
                assert other.min() >= -1e-6
                parts.update({"Other control-loop work": np.maximum(other, 0), "Total online control": total})
                for name, values in parts.items():
                    collected[name].append(values)
            checked.append(row)
        assert checked == report["results"] and report["timing"]["timed_steps"] == 315000
        frame = pd.DataFrame(checked).set_index("seed").sort_index()
        assert int((frame.success_30 & frame.success_100).sum()) == report["successes"]
        timing = pd.DataFrame(report["timing"]["modules"]).set_index("module")
        for name, arrays in collected.items():
            values = np.concatenate(arrays)
            stats = [values.mean(), *np.percentile(values, [50,95,99])]
            np.testing.assert_allclose(timing.loc[name, ["mean_ms", "median_ms", "p95_ms", "p99_ms"]], stats, rtol=1e-12, atol=1e-12)
        reports[rank], rows[rank], times[rank], sessions[rank] = report, frame, timing, session

# Compare all protocol fields other than the two intentional runner/rank identities.
ignore = {"version", "ranks", "runner_sha256"}
assert {k:v for k,v in summaries["A"]["signature"].items() if k not in ignore} == {k:v for k,v in summaries["B"]["signature"].items() if k not in ignore}
for rank in RANKS:
    np.testing.assert_array_equal(rows[rank].exact_target_xy.tolist(), rows[58].exact_target_xy.tolist())
rows, reports, times = ({r: mapping[r] for r in RANKS} for mapping in (rows, reports, times))
plt.rcParams.update({"figure.dpi": 125, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
colors = dict(zip(RANKS, ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]))
def save(name):
    plt.savefig(FIGURES / (name + ".png"), dpi=160, bbox_inches="tight")
    plt.savefig(FIGURES / (name + ".pdf"), bbox_inches="tight")
    plt.show()
    plt.close()
print("Verified 6,000 full rollouts; identical paired targets, offline files and numerical dependencies; final invariance audits; and raw timing summaries.")
display(pd.DataFrame([{"Session": s, "Ranks": str(rs), "Completed UTC": summaries[s]["completed_utc"],
                       "Source": str(folder.relative_to(ROOT))} for s,(folder,rs) in SOURCES.items()]))
'''), md("""## Success and tracking error

All six ranks use the same 1,000 seeds. Every trial, including unsuccessful trials, contributes to the error distribution. The dashed 10 mm line indicates only the mean-error threshold; strict success additionally requires both maximum-error tests.
"""), code('''def wilson(k, n=1000, z=1.959963984540054):
    p = k/n
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    return 100*np.array([center-half, center+half])
records = []
for rank, frame in rows.items():
    count = int((frame.success_30 & frame.success_100).sum())
    errors = 1000*frame.mean_last_100.to_numpy()
    record = {"rank": rank, "session": sessions[rank], "successes": count,
              "success_rate_pct": count/10, "95% Wilson interval": "%.2f–%.2f%%" % tuple(wilson(count)),
              "unsolved_qps": int(frame.unsolved_steps.sum())}
    record.update(dict(zip(["median_error_mm", "p90_error_mm", "p95_error_mm", "p99_error_mm"], np.percentile(errors,[50,90,95,99]))))
    records.append(record)
table = pd.DataFrame(records).set_index("rank")
display(table.round(3))
table.to_csv(RESULT / "rank_results.csv")
fig, ax = plt.subplots(figsize=(10,4.8), constrained_layout=True)
rates = table.success_rate_pct.to_numpy()
intervals = np.array([wilson(k) for k in table.successes])
ax.bar(range(len(RANKS)), rates, color=[colors[r] for r in RANKS], width=.62)
ax.errorbar(range(len(RANKS)), rates, yerr=np.vstack([rates-intervals[:,0],intervals[:,1]-rates]), fmt="none", color="#222222", capsize=4)
for i,(rate,count) in enumerate(zip(rates,table.successes)):
    ax.text(i,intervals[i,1]+1.1,f"{rate:.1f}%\\n{count}/1000",ha="center",fontsize=10)
ax.set(xticks=range(len(RANKS)),xticklabels=RANKS,xlabel="Retained rank",ylabel="Strict success (%)",ylim=(0,108),title="Success on seeds 311–1310 · 95% Wilson intervals")
ax.grid(axis="y",alpha=.2)
save("success")
fig, ax = plt.subplots(figsize=(9,5),constrained_layout=True)
for rank,frame in rows.items():
    ax.plot(np.sort(1000*frame.mean_last_100),np.arange(1,1001)/1000,label=f"Rank {rank}",color=colors[rank],lw=1.7)
ax.axvline(10,color="gray",linestyle="--",linewidth=1)
ax.set(xlabel="Mean final-100 target error (mm)",ylabel="Fraction of seeds",xscale="log",title="Tracking-error distributions · all 1,000 trials per rank")
ax.legend(ncol=2);ax.grid(alpha=.2)
save("tracking_error")
'''), md("""## Paired outcomes and solver statuses

All 15 rank pairs are compared seed by seed. These counts compare performance on identical targets, including pairs drawn from separate timing sessions.
"""), code('''paired = []
for a,b in combinations(RANKS,2):
    sa = rows[a].success_30 & rows[a].success_100
    sb = rows[b].success_30 & rows[b].success_100
    paired.append({"rank A":a,"rank B":b,"both succeed":int((sa & sb).sum()),
                   "A only":int((sa & ~sb).sum()),"B only":int((~sa & sb).sum()),
                   "both fail":int((~sa & ~sb).sum())})
paired_table = pd.DataFrame(paired)
assert (paired_table[["both succeed","A only","B only","both fail"]].sum(axis=1)==1000).all()
display(paired_table)
paired_table.to_csv(RESULT / "paired_outcomes.csv",index=False)
status_counts = {rank:dict(sum((Counter(x) for x in frame.statuses),Counter())) for rank,frame in rows.items()}
for rank,counts in status_counts.items():
    assert sum(counts.values())==320000
    assert counts==summaries[sessions[rank]]["statuses_by_rank"][str(rank)]
display(pd.DataFrame(status_counts).fillna(0).astype(int).rename_axis("Solver status"))
print("Total QPs:",sum(sum(v.values()) for v in status_counts.values()))
'''), md("""## Computation time

Session A: ranks 59/62/64. Session B: ranks 58/61/63. The plots label both sessions explicitly. Differences between them also reflect execution date and system load.

Tables report pooled per-step mean, median, P95 and P99. The stacked panel uses **means**, which add; the percentile panel is unstacked because module medians/percentiles do not add. Offline setup and one-time goal setup are separate from control-step timing.
"""), code('''timing_table = pd.concat(times,names=["Rank","Module"])
timing_table["Session"] = [sessions[r] for r,_ in timing_table.index]
display(timing_table[["Session","mean_ms","median_ms","p95_ms","p99_ms"]].round(4))
timing_table.to_csv(RESULT / "module_timings.csv")
labels = [f"{r}\\n{sessions[r]}" for r in RANKS]
x = np.arange(len(RANKS))
fig, axes = plt.subplots(1,2,figsize=(13,5.4),constrained_layout=True)
bottom = np.zeros(len(RANKS))
stage_names = times[58].index.drop("Total online control")
module_colors = ["#56B4E9","#E69F00","#009E73","#CC79A7","#0072B2","#777777"]
for module,color in zip(stage_names,module_colors):
    values = np.array([times[r].loc[module,"mean_ms"] for r in RANKS])
    axes[0].bar(x,values,bottom=bottom,label=module,color=color,width=.62)
    bottom += values
np.testing.assert_allclose(bottom,[times[r].loc["Total online control","mean_ms"] for r in RANKS])
axes[0].set(ylabel="Mean time per step (ms)",title="Online computation by module")
axes[0].legend(loc="upper left",fontsize=8)
axes[0].set_ylim(0,max(bottom)*1.65)
for i,total in enumerate(bottom):axes[0].text(i,total+.6,f"{total:.2f}",ha="center",fontsize=9)
for j,(stat,label,color) in enumerate([("median_ms","Median","#0072B2"),("p95_ms","P95","#E69F00"),("p99_ms","P99","#D55E00")]):
    axes[1].bar(x+(j-1)*.23,[times[r].loc["Total online control",stat] for r in RANKS],width=.22,label=label,color=color)
axes[1].set(ylabel="Total online time per step (ms)",title="Total-time percentiles")
axes[1].legend()
for ax in axes:
    ax.set_xticks(x,labels);ax.set_xlabel("Rank / measurement session");ax.grid(axis="y",alpha=.2)
fig.suptitle("Separate shared-computer sessions · cross-session differences are not rank effects alone",fontsize=11)
save("computation_time")
setup = pd.DataFrame({r:report["timing"]["one_time_goal_setup_ms"] for r,report in reports.items()}).T
setup.index.name = "Rank"
setup.insert(0,"Session",[sessions[r] for r in setup.index])
print("One-time goal setup per rollout (ms)");display(setup.round(3))
setup.to_csv(RESULT / "goal_setup.csv")
offline, wall = [], []
for session,(folder,_) in SOURCES.items():
    launch = json.loads((folder / "launch.json").read_text())
    for path in sorted(folder.glob("setup_*.json")):
        record = json.loads(path.read_text())
        offline.append({"Session":session,"Recorded UTC":record["utc"],
                        "Phase":"preflight" if record["utc"] < launch["started_utc"] else "evaluation",
                        **{f"Rank {r} setup (s)":v for r,v in record["matrix_and_route_setup_seconds_by_rank"].items()},
                        "Total load/setup (s)":record["total_load_and_setup_seconds"]})
    wall.extend({**json.loads(line),"Session":session} for line in (folder / "rollout_wall_times.jsonl").read_text().splitlines())
print("Offline matrix/route setup (seconds; preflight and evaluation distinguished)")
display(pd.DataFrame(offline).round(4))
wall = pd.DataFrame([r for r in wall if not r["reused_checkpoint"]])
print("End-to-end rollout wall time (seconds; includes trace writing and validation)")
display(wall.groupby(["rank","Session"]).seconds.agg(["count","mean","median","max"]).round(3))
report = {"complete":True,"rollouts":6000,"seeds":SEEDS,"ranks":table.reset_index().to_dict(orient="records"),
          "paired_outcomes":paired,"statuses_by_rank":status_counts,"sessions_by_rank":sessions,
          "timing":timing_table.reset_index().to_dict(orient="records"),"offline_setup":offline,
          "timing_note":"Two separate single-worker shared-computer sessions; cross-session timing differences cannot be attributed solely to rank.",
          "verified_offline_invariance":True,"verified_paired_targets":True,"timings_recomputed_from_raw_checkpoints":True,
          "source_summary_sha256":source_hashes,"figures":[str(p) for p in sorted(FIGURES.glob("*.png"))]}
(RESULT / "notebook_report.json").write_text(json.dumps(report,indent=2))
(RESULT / "checkpoint_audit.json").write_text(json.dumps({"count":6000,"sha256":checkpoint_hashes},indent=2))
''')]
    notebook = nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{
        "display_name":"Python (gym-robotics-mujoco)","language":"python","name":"gym-robotics-mujoco"}})
    nbf.validate(notebook)
    for cell in cells:
        if cell.cell_type == "code":
            compile(cell.source,"notebook","exec")
    return notebook


def main(execute=False):
    notebook=build()
    if execute:
        from nbclient import NotebookClient
        NotebookClient(notebook,timeout=900,kernel_name="gym-robotics-mujoco",
                       resources={"metadata":{"path":str(NOTEBOOK.parent)}}).execute()
        code=[c for c in notebook.cells if c.cell_type=="code"]
        assert len(code)==4 and all(c.execution_count is not None for c in code)
        assert not any(o.output_type=="error" for c in code for o in c.outputs)
        assert sum("image/png" in o.get("data",{}) for c in code for o in c.outputs)==3
    temp=NOTEBOOK.with_suffix(".ipynb.tmp")
    nbf.write(notebook,temp);temp.replace(NOTEBOOK)
    if execute:
        (RESULT / "notebook_execution.json").write_text(json.dumps({
            "executed_utc":datetime.now(timezone.utc).isoformat(),"notebook":str(NOTEBOOK),
            "sha256":hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),"executed_code_cells":4,
            "saved_inline_figures":3,"ranks":[58,59,61,62,63,64],"rollouts":6000,
            "run_experiments":False,"visual_inspection":"pending"},indent=2))
    print(NOTEBOOK)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    main(parser.parse_args().execute)
