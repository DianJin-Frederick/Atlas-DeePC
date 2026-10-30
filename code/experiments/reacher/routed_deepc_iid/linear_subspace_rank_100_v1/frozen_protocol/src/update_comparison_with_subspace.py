"""Extend the existing comparison without regenerating its historical results."""

from pathlib import Path
import shutil

import nbformat


ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "notebooks/comparisons/comparison.ipynb"
backup = ROOT / "experiments/deepc_algorithm_comparison/four_method_contextual_atlas_211_1210/comparison_before_subspace.ipynb"
if not backup.exists():
    shutil.copy2(path, backup)
notebook = nbformat.read(backup, as_version=4)


def replace(index, old, new):
    source = notebook.cells[index].source
    if old not in source:
        raise ValueError(f"Expected text missing from cell {index}: {old}")
    notebook.cells[index].source = source.replace(old, new)


replace(0, "Four-method", "Five-method")
replace(0, "**Atlas-DeePC**, **Contextual DPC**", "**Atlas-DeePC**, **Subspace Atlas-DeePC**, **Contextual DPC**")
replace(0, "The methods differ only in reference policy and in how", "The methods differ in chart centering, reference policy, and in how")
notebook.cells[0].source += "\n\nThe original four methods are preserved. Only **Subspace Atlas-DeePC** is newly evaluated: an uncentered SVD with zero affine offset, retaining graph routing and all other baseline settings."
replace(1, "| Contextual DPC |", "| Subspace Atlas-DeePC | Cached rank-60 **uncentered** SVD of the same measured physical chart, $b=U_{k,60}g$ | Same Atlas graph-routed waypoint | Same physical-chart localization and cached lookup |\n| Contextual DPC |")
replace(1, "Strict success requires both the final-30 and final-100 mean fingertip errors to be below 1 cm.", "Strict success uses the existing `success_30` and `success_100` flags: both final-30 and final-100 windows must have mean fingertip error below 1 cm and maximum error below 2 cm. No success labels were changed.")
notebook.cells[1].source += "\n\nFor the added method, each full-column chart matrix $H_k$ is factorized without mean subtraction. The first 60 left singular vectors are cached, and the affine offset is zero. The original empirical coefficient-bound rule and coefficient regularization are retained in the new coordinates. This is not the terminal-only ablation and not a contextual variant."
replace(2, "four_method_contextual_atlas_211_1210'", "five_method_with_subspace_211_1210'")
notebook.cells[2].source += """

NEW_METHOD = 'Subspace Atlas-DeePC'
SUBSPACE_DIR = ROOT / 'experiments/deepc_algorithm_comparison/subspace_atlas_serial_211_1210'
SUBSPACE_SUMMARY = SUBSPACE_DIR / 'summary.json'
RUN_NEW_METHOD = False

if RUN_NEW_METHOD:
    subprocess.run([
        sys.executable, 'src/subspace_comparison_experiment.py',
        '--seed-start', str(SEEDS[0]), '--seed-stop', str(SEEDS[-1]),
        '--output-dir', str(SUBSPACE_DIR),
    ], cwd=ROOT, check=True)
assert SUBSPACE_SUMMARY.exists(), 'Set RUN_NEW_METHOD=True to run only the new variant.'
"""
notebook.cells[3].source += "\n\nThe pure-subspace variant has its own 1,000-seed result file. `RUN_NEW_METHOD=True` runs only that method and resumes compatible per-seed checkpoints. None of the four historical result or timing files is rewritten."
replace(4, "assert all(len(rows) == 1000", "subspace_payload = json.loads(SUBSPACE_SUMMARY.read_text())\nsubspace_by_seed = {int(row['seed']): row for row in subspace_payload['results']}\nresults[NEW_METHOD] = [subspace_by_seed[seed] for seed in SEEDS]\nassert all(len(rows) == 1000")
replace(4, "all_rows =", "assert {k: v for k, v in subspace_payload['config'].items() if k != 'seed'} == reference_config\nassert subspace_payload['signature']['geometry'] == 'linear'\nassert subspace_payload['signature']['reference_mode'] == 'graph'\nassert all(a['data_goal_index'] == b['data_goal_index'] for a, b in zip(results['Atlas-DeePC'], results[NEW_METHOD]))\n\nall_rows =")
notebook.cells[5].source = """## 3. Primary 1,000-seed result

The table uses the same strict sustained criterion for all methods. Confidence intervals are Wilson intervals for the binomial success proportion.

The four original computation measurements are retained from the completed serial benchmark. The new subspace method is measured in a **separate serial batch**, with the same timing boundary: one process, one numerical thread, 10 warm-up steps on seed 210 excluded, and the first five steps of each evaluated rollout discarded. Each entry is the median across rollout-level median control-step times. A timing difference between the new method and old methods may reflect machine conditions as well as algorithm changes; this is not a fresh interleaved five-method speed benchmark.
"""
replace(6, "'Atlas-DeePC', 'Contextual DPC'", "'Atlas-DeePC', NEW_METHOD, 'Contextual DPC'")
replace(6, "controlled_records = pd.DataFrame(controlled_payload['per_seed'])", "controlled_aggregate.update({row['method']: row for row in subspace_payload['aggregate']})\ncontrolled_records = pd.DataFrame(controlled_payload['per_seed'] + subspace_payload['per_seed'])")
replace(6, "== 4 * len(SEEDS)", "== len(METHOD_ORDER) * len(SEEDS)")
replace(7, "'Atlas-DeePC': '#0072B2',", "'Atlas-DeePC': '#0072B2',\n    NEW_METHOD: '#6A3D9A',")
replace(7, "figsize=(17, 4.8)", "figsize=(19, 5.4)")
replace(7, "Fresh 1,000-seed local timing", "Serial timing (new method: separate batch)")
replace(7, "short = ['Select-DPC', 'Atlas-DeePC', 'Contextual DPC'", "short = ['Select-DPC', 'Atlas-DeePC', 'Subspace\\nAtlas-DeePC', 'Contextual DPC'")
replace(7, "'Subspace\\nAtlas-DeePC', 'Contextual DPC',", "'Subspace\\nAtlas-DeePC', 'Contextual\\nDPC',")
replace(7, "axis.set_xticks(x, short, rotation=12, ha='right')", "axis.set_xticks(x, short, rotation=0, ha='center', fontsize=9)")
replace(7, "Four-method", "Five-method")
replace(7, "four_method_1000_seed_comparison.png", "five_method_1000_seed_comparison.png")
notebook.cells[8].source = """### Serial computation by module

Each entry is the median across 1,000 rollout-level median stage times. Matrix preparation includes online SVD for Select-DPC and Contextual DPC; all three Atlas variants use cached bases. The QP timing wraps controller problem construction, canonicalization, and solution, not only the solver's internal numerical iterations.

Medians of separate stages need not add exactly to the median total. The stacked chart is therefore a descriptive decomposition; its nonnegative residual is labeled **other loop overhead**.

The original four methods were interleaved in the original benchmark. Only the new subspace method ran in the added batch, with matching single-thread, warm-up, and discarded-step settings. Original measured values have not been replaced.
"""
replace(9, "Fresh 1,000-seed computation-time decomposition", "1,000-seed module timing (subspace: separate batch)")
notebook.cells[12].source += "\n\nA second paired table directly compares pure-subspace and affine Atlas-DeePC. These are exploratory, unadjusted comparisons; the first 500 seeds were already used in the preceding ablation."
notebook.cells[13].source += """

affine_success = np.asarray([strict_success(row) for row in results['Atlas-DeePC']])
subspace_success = np.asarray([strict_success(row) for row in results[NEW_METHOD]])
repaired = [seed for seed, a, b in zip(SEEDS, affine_success, subspace_success) if b and not a]
regressed = [seed for seed, a, b in zip(SEEDS, affine_success, subspace_success) if a and not b]
subspace_paired = {
    'subspace-only successes': len(repaired),
    'affine-only successes': len(regressed),
    'net repaired seeds': len(repaired) - len(regressed),
    'exact McNemar p': exact_mcnemar_p(len(repaired), len(regressed)),
    'repaired seeds': repaired,
    'regressed seeds': regressed,
}
display(Markdown('### Pure subspace versus affine Atlas-DeePC'))
display(pd.Series(subspace_paired))
"""
replace(17, "In the fresh controlled benchmark", "In the retained four-method controlled benchmark")
replace(17, "statistically indistinguishable at this resolution", "numerically very close")
notebook.cells[17].source += """

display(Markdown(f'''
### Added pure-subspace variant
**Subspace Atlas-DeePC: {int(summary.loc[NEW_METHOD, 'successes'])}/{len(SEEDS)}**, versus
**{int(summary.loc['Atlas-DeePC', 'successes'])}/{len(SEEDS)}** for affine Atlas-DeePC.
It repairs {len(repaired)} seeds and regresses {len(regressed)}; the exploratory paired
McNemar p-value is **{subspace_paired['exact McNemar p']:.4g}**.

Its separately measured median online time is **{summary.loc[NEW_METHOD, 'median online compute (ms)']:.2f} ms**
per step, including **{controlled_aggregate[NEW_METHOD]['median QP solve (ms)']:.2f} ms**
inside the controller solve call. The four previous timing rows are unchanged.
The new-versus-old timing difference should not be attributed solely to removing the affine offset.
'''))
"""
replace(19, "mean_last_30 < 0.01 m and mean_last_100 < 0.01 m", "both final-30 and final-100: mean error < 0.01 m and max error < 0.02 m")
replace(19, "'three_comparators':", "'subspace_atlas': str(SUBSPACE_SUMMARY),\n        'original_controlled_timing': str(CONTROLLED_TIMING),\n        'three_comparators':")
replace(19, "'methods':", "'subspace_paired': subspace_paired,\n    'timing_caveat': subspace_payload['protocol']['timing_caveat'],\n    'methods':")
notebook.cells[20].source = """## 8. Result and timing provenance

The original four-method notebook is archived beside its original compact results. This notebook writes its five-method record and plots into a new directory. To run only the new variant, enable `RUN_NEW_METHOD` in the setup cell; compatible completed seeds are reused. For a fresh local timing measurement, choose a new `SUBSPACE_DIR` before running. None of these controls launches the four historical methods.
"""
notebook.cells[21].source = """display(pd.DataFrame([
    {'methods': 'Original four methods',
     'timestamp UTC': controlled_payload['protocol']['timestamp_utc'],
     'source': str(CONTROLLED_TIMING), 'rerun in this update': False},
    {'methods': NEW_METHOD,
     'timestamp UTC': subspace_payload['protocol']['timestamp_utc'],
     'source': str(SUBSPACE_SUMMARY), 'rerun in this update': True},
]))
print('Original four-method notebook:',
      ROOT / 'experiments/deepc_algorithm_comparison/four_method_contextual_atlas_211_1210/comparison_before_subspace.ipynb')
"""
for cell in notebook.cells:
    if cell.cell_type == "code":
        compile(cell.source, str(path), "exec")
        cell.outputs = []
        cell.execution_count = None
nbformat.write(notebook, path)
print(path)
