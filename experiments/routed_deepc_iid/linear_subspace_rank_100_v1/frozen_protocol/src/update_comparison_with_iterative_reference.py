"""Append iterative terminal augmentation to the existing five-method notebook."""

from pathlib import Path
import shutil

import nbformat


ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "notebooks/comparisons/comparison.ipynb"
backup = ROOT / "experiments/deepc_algorithm_comparison/five_method_with_subspace_211_1210/comparison_before_iterative_reference.ipynb"
if not backup.exists():
    shutil.copy2(path, backup)
notebook = nbformat.read(backup, as_version=4)


def replace(index, old, new):
    if old not in notebook.cells[index].source:
        raise ValueError(f"Expected text missing from cell {index}: {old}")
    notebook.cells[index].source = notebook.cells[index].source.replace(old, new)


replace(0, "Five-method", "Six-method")
replace(0, "**Subspace Atlas-DeePC**,", "**Subspace Atlas-DeePC**, **Iterative-reference Atlas-DeePC**,")
replace(0, "goal-construction rule,", "offline goal-candidate eligibility rule,")
replace(0, "The original four methods are preserved. Only **Subspace Atlas-DeePC** is newly evaluated:", "The first five methods' results are preserved. **Subspace Atlas-DeePC** uses")
notebook.cells[0].source += "\n\nThe added **Iterative-reference Atlas-DeePC** implements the user's candidate-route augmentation with **N=10**. It keeps the original affine charts, not the pure-subspace ablation. It reuses its completed 500-seed ablation and extends only this algorithm to seeds 711--1210."
replace(1, "| Contextual DPC |", "| Iterative-reference Atlas-DeePC | Same cached rank-60 affine basis as baseline Atlas | Repeatedly select the minimum-route-cost goal chart among the top-10 eligible near-target configurations, then follow graph waypoints | Cached shortest-path costs plus per-step candidate-chart selection |\n| Contextual DPC |")
notebook.cells[1].source += "\n\n**Iterative augmentation:** gather the top-10 eligible offline configurations by fingertip distance once per rollout. At each step, choose the candidate chart with minimum directed route cost from the current chart; ties prefer smaller fingertip error, then source index. Use that candidate's posture with zero joint velocity and the requested fingertip target as the full terminal reference. Chart-to-chart shortest-path costs are precomputed offline; online rerouting uses the same graph costs as the candidate ablation. The 7 cm switch latches use of the terminal reference, but the candidate itself can still change online."
replace(2, "five_method_with_subspace_211_1210'", "six_method_with_iterative_reference_211_1210'")
notebook.cells[2].source += """

ITERATIVE_METHOD = 'Iterative-reference Atlas-DeePC'
ITERATIVE_DIR = ROOT / 'experiments/deepc_algorithm_comparison/iterative_reference_n10_211_1210'
ITERATIVE_SUMMARY = ITERATIVE_DIR / 'summary.json'
RUN_ITERATIVE_METHOD = False
if RUN_ITERATIVE_METHOD:
    subprocess.run([
        sys.executable, 'src/iterative_reference_comparison.py',
        '--seed-start', str(SEEDS[0]), '--seed-stop', str(SEEDS[-1]),
        '--candidate-count', '10', '--output-dir', str(ITERATIVE_DIR),
    ], cwd=ROOT, check=True)
assert ITERATIVE_SUMMARY.exists(), 'Set RUN_ITERATIVE_METHOD=True to complete the new method.'
"""
notebook.cells[3].source += "\n\nIterative augmentation reuses the original N=10 rollout files for seeds 211--710 and adds new rollouts for 711--1210. **N=10 was chosen from the earlier 500-seed candidate-count ablation**, so the combined 1,000-seed result is not an untouched test. A separate 711--1210 result is reported below. No candidate count is tuned on that extension."
replace(4, "assert all(len(rows) == 1000", "iterative_payload = json.loads(ITERATIVE_SUMMARY.read_text())\niterative_by_seed = {int(row['seed']): row for row in iterative_payload['results']}\nresults[ITERATIVE_METHOD] = [iterative_by_seed[seed] for seed in SEEDS]\nassert all(len(rows) == 1000")
replace(4, "all_rows =", "assert {k: v for k, v in iterative_payload['config'].items() if k != 'seed'} == reference_config\nassert iterative_payload['candidate_count'] == 10\nassert iterative_payload['signature']['geometry'] == 'affine'\nassert all(row['dynamic_terminal_reference'] for row in results[ITERATIVE_METHOD])\nassert all(row['minimum_selected_goal_source_separation'] >= 0.06 for row in results[ITERATIVE_METHOD])\n\nall_rows =")
notebook.cells[5].source += "\n\nIterative-reference timings combine its archived serial ablation and a new single-threaded serial extension. Both are summarized after dropping the first five control steps; the extension also excludes a 10-step warm-up. The archived rollout timings were not replaced. Treat timings across these batches as descriptive, not a controlled causal speed comparison."
replace(6, "NEW_METHOD, 'Contextual DPC'", "NEW_METHOD, ITERATIVE_METHOD, 'Contextual DPC'")
replace(6, "controlled_records = pd.DataFrame(controlled_payload['per_seed'] + subspace_payload['per_seed'])", "controlled_aggregate.update({row['method']: row for row in iterative_payload['aggregate']})\ncontrolled_records = pd.DataFrame(controlled_payload['per_seed'] + subspace_payload['per_seed'] + iterative_payload['per_seed'])")
replace(7, "NEW_METHOD: '#6A3D9A',", "NEW_METHOD: '#6A3D9A',\n    ITERATIVE_METHOD: '#946600',")
replace(7, "figsize=(19, 5.4)", "figsize=(22, 5.7)")
replace(7, "Serial timing (new method: separate batch)", "Serial timing (separate measurement batches)")
replace(7, "'Subspace\\nAtlas-DeePC', 'Contextual\\nDPC'", "'Subspace\\nAtlas-DeePC', 'Iterative-ref.\\nAtlas-DeePC', 'Contextual\\nDPC'")
replace(7, "Five-method", "Six-method")
replace(7, "five_method_1000_seed_comparison.png", "six_method_1000_seed_comparison.png")
notebook.cells[8].source += "\n\nFor iterative augmentation, the **data / prototype selection** column includes terminal-candidate chart selection (`goal_selection_time_s`); graph routing covers route reconstruction. One-time target-neighborhood construction is outside the control-step timing boundary and is shown separately below, rather than hidden in an online per-step speed claim."
replace(9, "1,000-seed module timing (subspace: separate batch)", "1,000-seed module timing (separate serial batches)")
notebook.cells[9].source += """

iterative_timing = controlled_records[controlled_records['method'] == ITERATIVE_METHOD]
timing_by_batch = iterative_timing.groupby('timing_batch').agg(
    rollouts=('seed', 'size'),
    median_step_ms=('total online control', 'median'),
    median_QP_ms=('QP solve', 'median'),
    median_goal_selection_ms=('data / prototype selection', 'median'),
    median_route_ms=('graph routing', 'median'),
    median_goal_setup_ms=('goal setup (ms)', 'median'),
)
display(Markdown('### Iterative augmentation: timing batches and one-time goal setup'))
display(timing_by_batch.style.format({c: '{:.4f}' for c in timing_by_batch if c != 'rollouts'}))
"""
notebook.cells[13].source += """

iterative_success = np.asarray([strict_success(row) for row in results[ITERATIVE_METHOD]])
iterative_pairs = []
for panel, lo, hi in [('All comparison seeds (includes selection panel)', 211, 1210),
                      ('Prior candidate-count selection panel', 211, 710),
                      ('Extension after fixing N=10', 711, 1210)]:
    mask = (np.asarray(SEEDS) >= lo) & (np.asarray(SEEDS) <= hi)
    a, b = affine_success[mask], iterative_success[mask]
    gains, losses = int(np.sum(b & ~a)), int(np.sum(a & ~b))
    iterative_pairs.append({
        'panel': panel, 'seeds': int(mask.sum()),
        'affine successes': int(a.sum()), 'iterative successes': int(b.sum()),
        'repaired': gains, 'regressed': losses, 'net repaired': gains-losses,
        'exact McNemar p': exact_mcnemar_p(gains, losses),
    })
display(Markdown('### Iterative terminal augmentation versus fixed augmentation'))
display(pd.DataFrame(iterative_pairs).set_index('panel').style.format({'exact McNemar p': '{:.4g}'}))
display(pd.Series({
    'rollouts whose terminal sample changes online': sum(row['goal_sample_switches'] > 0 for row in results[ITERATIVE_METHOD]),
    'mean terminal sample switches per rollout': np.mean([row['goal_sample_switches'] for row in results[ITERATIVE_METHOD]]),
    'maximum terminal sample switches': max(row['goal_sample_switches'] for row in results[ITERATIVE_METHOD]),
}))
"""
notebook.cells[17].source += """

iterative_all, _, iterative_extension = iterative_pairs
display(Markdown(f'''
### User-proposed iterative terminal augmentation (N=10)
Iterative-reference Atlas-DeePC succeeds on **{iterative_all['iterative successes']}/1000**
seeds versus **{iterative_all['affine successes']}/1000** for the original fixed augmentation.
It repairs **{iterative_all['repaired']}** seeds and regresses **{iterative_all['regressed']}**
(exploratory paired p={iterative_all['exact McNemar p']:.4g}).

N=10 was selected using the earlier seeds 211--710. On the separate extension 711--1210,
the counts are **{iterative_extension['iterative successes']}/500** versus
**{iterative_extension['affine successes']}/500** (paired p={iterative_extension['exact McNemar p']:.4g}).
Only terminal augmentation and its route lookup change; this method retains the original affine
chart matrices, QP, offline data, and source-target separation rule. It does not use contextual rescue.
'''))
"""
replace(19, "'subspace_paired':", "'iterative_paired': iterative_pairs,\n    'iterative_selection_caveat': iterative_payload['protocol']['selection_caveat'],\n    'iterative_timing_caveat': iterative_payload['protocol']['timing_caveat'],\n    'subspace_paired':")
replace(19, "'subspace_atlas':", "'iterative_reference': str(ITERATIVE_SUMMARY),\n        'subspace_atlas':")
notebook.cells[20].source += "\n\nThe five-method notebook is also archived. `RUN_ITERATIVE_METHOD=True` reuses compatible archived candidate experiments and runs missing iterative-reference seeds only. It does not run any other method. The original candidate ablation files remain unchanged."
replace(21, "'source': str(SUBSPACE_SUMMARY), 'rerun in this update': True},", "'source': str(SUBSPACE_SUMMARY), 'rerun in this update': False},\n    {'methods': ITERATIVE_METHOD,\n     'timestamp UTC': iterative_payload['protocol']['timestamp_utc'],\n     'source': str(ITERATIVE_SUMMARY), 'rerun in this update': 'Only 500 missing seeds'},")
for cell in notebook.cells:
    if cell.cell_type == "code":
        compile(cell.source, str(path), "exec")
        cell.outputs = []
        cell.execution_count = None
nbformat.write(notebook, path)
print(path)
