"""Move the report to seeds 311--1310, keeping the previous notebook intact."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive/notebooks/baseline_clarabel_lamg005_seeds211_1210_before_shift.ipynb"
TARGET = ROOT / "notebooks/baselines/baseline_graph_routed_atlas_deepc.ipynb"


def update():
    n = nbf.read(ARCHIVE, as_version=4)
    n.cells[0].source = r'''# Baseline Atlas-DeePC: Seeds 311--1310

**Clarabel, $\lambda_g=0.05$, 1,000 paired seeds per method.**
The notebook evaluates Atlas-DeePC with and without intermediate graph waypoints.
All use the same frozen offline data, 48 physical charts, rank-60 linear bases,
controller weights and constraints, and full 320-step evaluation.

The original 211--1210 notebook and data are preserved. Overlapping verified
rollouts are reused without modifying their traces; missing seeds are newly run.
Timing statistics therefore combine measurement sessions and are descriptive,
not a controlled cross-method speed benchmark. This is an overlapping evaluation
panel, not 1,000 newly untouched seeds. Provenance is recorded per rollout.

The original videos remain as explicitly labeled historical diagnostics; the
success-rate and timing tables below use only seeds 311--1310.
'''
    s = n.cells[1].source
    s = s.replace('EVALUATION_SEEDS = list(range(211, 1211))', 'EVALUATION_SEEDS = list(range(311, 1311))')
    s = s.replace('EVAL_DIR = RESULT_ROOT / "evaluation_1000"', 'import baseline_clarabel_seed_panel as seed_panel\nPANEL_ROOT = seed_panel.RESULT_ROOT\nEVAL_DIR = PANEL_ROOT / "graph"')
    s = s.replace('    from baseline_1000_benchmark import run_panel as run_timed_baseline\n    raise RuntimeError("Historical saved-result notebook: use a separate fresh benchmark for new timings.")', '    seed_panel.run_panel(EVALUATION_SEEDS, PANEL_ROOT)')
    s = s.replace('print("Baseline version:", SAVED_BASELINE_VERSION)', 'print("Evaluation version:", seed_panel.VERSION)\nprint("Results:", PANEL_ROOT)')
    n.cells[1].source = s
    n.cells[9].source = '''## 3. Result and Fairness Audit

The displayed graph-waypoint panel uses seeds 311--1310. Trace provenance retains
the historical version for reused seeds and the original new-run version for added
seeds; these are not relabeled. Parameter and offline-data checks guard the merge.
'''
    n.cells[10].source = '''from dataclasses import asdict
from baseline_clarabel_seed_panel import validate_panel
result_summary = json.loads((EVAL_DIR / "summary.json").read_text())
runs, missing_seeds = validate_panel(result_summary, EVALUATION_SEEDS, config)
assert not missing_seeds, f"Incomplete graph panel: {missing_seeds}"
panel_name = f"Graph-waypoint baseline: seeds {min(EVALUATION_SEEDS)}--{max(EVALUATION_SEEDS)}"
audit = result_summary["fairness_audit"]
assert audit["offline_stabilized_episode_count"] == 0
from candidate_goal_routing_experiment import file_digest
assert result_summary["signature"]["data_sha256"] == file_digest(DATA_PATH)
assert result_summary["signature"]["atlas_sha256"] == file_digest(ATLAS_PATH)
display(pd.Series({
    "displayed panel": panel_name, "offline seed range": str(audit["data_seed_range"]),
    "stabilized offline episodes": audit["offline_stabilized_episode_count"],
    "minimum source-target separation (m)": min(r["minimum_selected_goal_source_separation"] for r in runs),
    "reused graph rollouts": sum(r["provenance"] == "historical_reuse" for r in runs),
    "new graph rollouts": sum(r["provenance"] == "new_panel_rollout" for r in runs),
    "per-step full-library retrieval": "none", "one-time target-conditioned state search": "top 10 eligible configurations",
}))
'''
    n.cells[11].source = '''## 4. Baseline Results and Timing

Strict success requires mean fingertip error below 1 cm and maximum error below
2 cm over both the final-30 and final-100 windows. All 320 steps are simulated.
Timing excludes the first five steps of each rollout and includes QP construction,
canonicalization, and solving. Reused and new timings come from different sessions.
'''
    s = n.cells[14].source
    start = s.index('TEST_SEEDS =')
    s = 'from baseline_success_rate import success_statistics\nfrom baseline_clarabel_seed_panel import validate_panel\n\n' + s[start:]
    s = s.replace('TEST_SEEDS = list(range(211, 1211))', 'TEST_SEEDS = list(EVALUATION_SEEDS)')
    s = s.replace('TEST_RESULT_DIR = RESULT_ROOT / "evaluation_1000"', 'TEST_RESULT_DIR = EVAL_DIR')
    s = s.replace('    from baseline_1000_benchmark import run_panel as run_timed_baseline\n    raise RuntimeError("Historical saved-result notebook: use a separate fresh benchmark for new timings.")', '    seed_panel.run_panel(TEST_SEEDS, PANEL_ROOT)')
    s = s.replace('"baseline": SAVED_BASELINE_VERSION', '"baseline": seed_panel.VERSION')
    s = s.replace('title="1,000-seed baseline test"', 'title=f"1,000-seed baseline test: {min(TEST_SEEDS)}--{max(TEST_SEEDS)}"')
    n.cells[14].source = s
    n.cells[13].source = '''### 4.1. 1,000-Seed Success-Rate Test

The graph-waypoint baseline is evaluated on seeds **311--1310**. Its data and
settings are frozen. This panel overlaps the previous test in 900 seeds; those
verified rollouts are reused and the new 100 are simulated separately.
`RUN_EVALUATION=True` runs/resumes both notebook variants without overwriting
historical results. Saved-view mode is the default.
'''
    s = n.cells[16].source
    s = s.replace('[TEST_RESULT_DIR / f"seed{s}_control_results.npz" for s in TEST_SEEDS]', '[r["result_path"] for r in test_rows]')
    start = s.index('    display(pd.DataFrame([{')
    end = s.index('    print("Timed control steps:', start)
    s = s[:start] + '    print("Timing provenance:", test_timing["timing_caveat"])\n' + s[end:]
    n.cells[16].source = s
    n.cells[15].source += '\n**Panel provenance:** the aggregates combine 900 historical and 100 added graph rollouts. All used one worker and one numerical thread, but ran in separate sessions.\n'
    n.cells[20].source = '''### 4.3. With and Without Graph Waypoints: Seeds 311--1310

Both methods use Clarabel, lambda_g=0.05, and iterative N=10 terminal-reference
selection. The no-waypoint method tracks that terminal reference from the first
step; it still uses graph-route cost to choose the terminal candidate.
Each panel reuses 900 matching rollouts and adds 100. Timing statistics retain
their original measurements and combine sessions; they are not freshly paired timing tests.
'''
    n.cells[21].source = '''import baseline_clarabel_waypoint_comparison as clarabel_comparison
from baseline_1000_benchmark import summarize_timing
WAYPOINT_SEEDS = list(EVALUATION_SEEDS)
WAYPOINT_RESULT_DIR = PANEL_ROOT
comparison_rows = {}
for mode in ("terminal", "graph"):
    panel_payload = json.loads((PANEL_ROOT / mode / "summary.json").read_text())
    rows, missing = validate_panel(panel_payload, WAYPOINT_SEEDS, config)
    assert not missing, f"Incomplete {mode} panel: {missing}"
    comparison_rows[mode] = rows
completion = pd.DataFrame([{
    "Method": clarabel_comparison.METHODS[mode], "Completed": len(rows), "Requested": len(WAYPOINT_SEEDS),
    "Historical reuse": sum(r["provenance"] == "historical_reuse" for r in rows),
    "Added rollouts": sum(r["provenance"] == "new_panel_rollout" for r in rows),
    "Success rate (%)": 100*np.mean([r["success_30"] and r["success_100"] for r in rows]),
} for mode, rows in comparison_rows.items()])
display(completion)
paired_seeds = list(WAYPOINT_SEEDS)
print("Timings combine historical and new measurement sessions; descriptive only.")
'''
    s = n.cells[22].source
    start = s.index('        paths =')
    end = s.index('        timing =', start)
    s = s[:start] + '        paths = [rows[s]["result_path"] for s in paired_seeds]\n' + s[end:]
    s = s.replace('note = "new paired timing run" if graph_is_fresh else "timings from separate sessions; descriptive only"', 'note = "historical + added measurements; descriptive only"')
    s = s.replace('{len(paired_seeds)} matched seeds |', '{len(paired_seeds)} matched seeds ({min(paired_seeds)}--{max(paired_seeds)}) |')
    n.cells[22].source = s
    for cell in n.cells:
        if cell.cell_type == 'code':
            cell.source = cell.source.replace('fig.savefig(RESULT_ROOT /', 'fig.savefig(PANEL_ROOT /')
        if cell.cell_type == 'markdown' and cell.source.startswith('## 5.'):
            cell.source = '## 5. Historical Diagnostic Replays\n\nThese original videos are retained as diagnostic examples. Their seeds are not presented as representatives sampled from the new 311--1310 panel.\n\n' + cell.source.split('\n', 1)[1]
        if cell.cell_type == 'markdown' and cell.source.startswith('## 7.'):
            cell.source = '''## 7. Reproduction and Preservation

`RUN_EVALUATION=True` runs or resumes both variants for the configured panel.
The result directory has a seed-list and source-hash manifest; use a new directory
for another seed panel. The original 211--1210 report remains at
`archive/notebooks/baseline_clarabel_lamg005_seeds211_1210_before_shift.ipynb`.
Offline data, original traces, and original diagnostic videos remain unchanged.
'''
    for i, cell in enumerate(n.cells):
        if cell.cell_type == 'code' and i in (1, 10, 12, 14, 16, 18, 21, 22):
            cell.outputs = []
            cell.execution_count = None
    del n.cells[17:20]
    nbf.validate(n)
    for c in n.cells:
        if c.cell_type == 'code':
            compile(c.source, '<notebook>', 'exec')
    nbf.write(n, TARGET)


if __name__ == '__main__':
    update()
