"""Build the executed report for query-distilled pure contextual Atlas-DeePC."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/baselines/query_distilled_contextual_atlas_deepc.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {
    "display_name": "Python (gym-robotics-mujoco)",
    "language": "python",
    "name": "gym-robotics-mujoco",
}
nb["metadata"]["language_info"] = {"name": "python", "version": "3.10"}
nb["cells"] = [
    markdown("""
# Query-distilled pure contextual Atlas-DeePC

This notebook continues `baseline_contextual_atlas_deepc.ipynb` and `pure_contextual_atlas_deepc.ipynb`. The controller is always contextual and trigger-free, but its atlas is distilled from the causal query geometry of contextual-aware DPC rather than clustered from offline trajectories.

Development uses seeds 111–210. The frozen paired result below uses disjoint seeds 711–1210.
"""),
    code("""
import json, math, os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/atlas_deepc_mpl')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src" / "baseline_atlas_deepc.py").is_file())
RESULT = ROOT/'experiments/routed_deepc_iid/query_distilled_teacher_k1024_medoid_heldout711_1210'
ARTIFACT = ROOT/'experiments/routed_deepc_iid/query_distilled_teacher_k1024_medoid_dev111_210/contextual_cluster_atlas.npz'
COMPARISON = ROOT/'experiments/deepc_algorithm_comparison/four_method_contextual_atlas_211_1210/summary.json'
summary = json.loads((RESULT/'summary.json').read_text())
comparison = json.loads(COMPARISON.read_text())
artifact = np.load(ARTIFACT, allow_pickle=True)
training = json.loads(artifact['atlas_training'].item())
print(training)
"""),
    markdown("""
## Construction

For each causal development query $q=[Y_p,r_f]$, use the same normalized past-output and Q-weighted final-reference geometry as contextual-aware DPC. Mini-batch K-means partitions the query distribution, and a real query nearest each center is retained as a medoid. Each of 1,024 medoids retrieves its top 200 complete Hankel columns **offline** and caches a rank-60 affine basis.

Online control evaluates only the balanced distance to the 1,024 medoids and solves one cached basis. It never scans Hankel columns or rebuilds an SVD.
"""),
    code("""
rows = {int(r['seed']): r for r in summary['results']}
assert set(rows) == set(range(711, 1211))
assert all(r['contextual_usage_fraction'] == 1.0 for r in rows.values())
assert all(not r['contextual_triggered'] for r in rows.values())
assert all(r['online_hankel_columns_scored'] == 0 for r in rows.values())
assert all(not r['online_svd_rebuild'] for r in rows.values())
assert all(r['contextual_cluster']['reference_mode'] == 'final_only' for r in rows.values())
pd.Series({
    'cached contextual bases': summary['artifact']['prototype_count'],
    'support columns per basis': summary['artifact']['support_columns'],
    'basis rank': summary['artifact']['basis_rank'],
    'development queries': training['queries'],
    'development seeds': f"{training['development_seed_start']}-{training['development_seed_stop']}",
    'evaluation seeds': '711-1210',
})
"""),
    code("""
def heldout(name):
    return {int(r['seed']): bool(r['success']) for r in comparison['per_seed'][name]
            if 711 <= int(r['seed']) <= 1210}

ours = {s: bool(r['success_30'] and r['success_100']) for s, r in rows.items()}
methods = {
    'Physical Atlas': heldout('Atlas-DeePC'),
    'Pure query-distilled contextual Atlas': ours,
    'Contextual-aware DPC': heldout('Contextual DPC'),
    'Triggered contextual Atlas': heldout('Contextual Atlas-DeePC'),
}

def wilson(x, n):
    z=1.959963984540054; d=1+z*z/n
    c=(x/n+z*z/(2*n))/d
    h=z*math.sqrt(x/n*(1-x/n)/n+z*z/(4*n*n))/d
    return c-h,c+h

table=[]
for name, outcomes in methods.items():
    x=sum(outcomes.values()); lo,hi=wilson(x,len(outcomes))
    table.append({'method':name,'successes':x,'trials':len(outcomes),
                  'success rate':x/len(outcomes),'95% Wilson CI':f'[{lo:.3f}, {hi:.3f}]'})
outcomes = pd.DataFrame(table).set_index('method')
display(outcomes.style.format({'success rate':'{:.1%}'}))
assert outcomes.loc['Pure query-distilled contextual Atlas','successes'] == 472
"""),
    code("""
fig, ax = plt.subplots(figsize=(9.5,4.6))
colors=['#6b7280','#087e8b','#d1495b','#7a5195']
bars=ax.bar(outcomes.index,100*outcomes['success rate'],color=colors)
ax.set(ylabel='strict success rate (%)',ylim=(84,99),title='Disjoint evaluation: seeds 711-1210')
ax.tick_params(axis='x',rotation=14)
for bar,x in zip(bars,outcomes['successes']):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.25,f'{x}/500',ha='center',fontweight='bold')
ax.grid(axis='y',alpha=.25); plt.tight_layout(); plt.show()
"""),
    markdown("""
## Paired evidence

The comparison is paired by MuJoCo seed. This distinguishes a genuine repair from two methods failing on unrelated seeds.
"""),
    code("""
paired=[]
for name in ['Physical Atlas','Contextual-aware DPC','Triggered contextual Atlas']:
    other=methods[name]
    ours_only=sum(ours[s] and not other[s] for s in ours)
    other_only=sum(other[s] and not ours[s] for s in ours)
    p=binomtest(min(ours_only,other_only),ours_only+other_only,.5).pvalue
    paired.append({'comparison':name,'pure only':ours_only,'other only':other_only,
                   'net pure gain':ours_only-other_only,'exact paired p':p})
display(pd.DataFrame(paired).set_index('comparison'))
"""),
    code("""
timing = pd.Series({
    'median total control compute (ms)': np.median([r['median_control_compute_ms'] for r in rows.values()]),
    'median prototype scoring (ms)': np.median([r['median_cluster_scoring_ms'] for r in rows.values()]),
    'median DeePC solve (ms)': np.median([r['median_solve_ms'] for r in rows.values()]),
    'contextual-aware DPC median compute (ms)': 76.83014575741254,
})
display(timing)
"""),
    markdown("""
## Conclusion

The pure, trigger-free contextual atlas reaches **472/500 (94.4%)**, up from **440/500 (88.0%)** for physical Atlas. It repairs 44 physical-Atlas failures while regressing on 12 (exact paired $p=2.09\times10^{-5}$).

Against contextual-aware DPC's **483/500 (96.6%)**, the paired discordances are 9 versus 20 and the exact two-sided $p=0.0614$. Thus the held-out difference is not significant at 0.05. It is also statistically indistinguishable from the triggered contextual Atlas (94.4% versus 95.2%, paired $p=0.557$), while remaining always contextual from the first control step and avoiding online Hankel retrieval and SVD.
"""),
]

nbf.write(nb, OUTPUT)
print(OUTPUT)
