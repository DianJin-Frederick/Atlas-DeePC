"""Build the executed report for pure chart-local contextual Atlas-DeePC."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/baselines/pure_contextual_atlas_deepc.ipynb"


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
    markdown(r"""
# Pure Contextual Atlas-DeePC: Always On, No Trigger

This notebook records the trigger-free contextual controller developed from the concerns raised by `baseline_contextual_atlas_deepc.ipynb`. The global 128-prototype contextual atlas is visually clean, but its 444/500 strict successes are statistically tied with the 443/500 physical atlas. The failure analysis below shows that global contextual quantization and mode switching discard too much physical locality.

The replacement is still a **pure contextual atlas controller**. A contextual basis is selected and used from control step 0 through step 319. There is no stall detector, activation event, physical-basis fallback, online Hankel-column scan, or online SVD.
"""),
    code(r"""
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/atlas_deepc_mpl')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src" / "baseline_atlas_deepc.py").is_file())
sys.path.insert(0, str(ROOT / 'src'))

import chart_local_contextual_atlas_experiment as pure

RESULT_ROOT = ROOT / 'experiments' / 'routed_deepc_iid'
BASELINE_DIR = RESULT_ROOT / 'clustering_jointconfig48_baseline_cache'
GLOBAL_CONTEXT_DIR = RESULT_ROOT / 'contextual_cluster_eval211_710_global128_always'
PURE_CONTEXT_DIR = RESULT_ROOT / 'always_on_contextual_eval211_310'
TRIGGERED_DIR = RESULT_ROOT / 'contextual_cluster_eval211_710_global128_triggered'
ARTIFACT_PATH = PURE_CONTEXT_DIR / 'always_on_contextual_atlas.npz'
EVAL_SEEDS = list(range(211, 711))

for path in [BASELINE_DIR/'summary.json', GLOBAL_CONTEXT_DIR/'summary.json',
             PURE_CONTEXT_DIR/'summary.json', TRIGGERED_DIR/'summary.json', ARTIFACT_PATH]:
    assert path.exists(), path
print('Paired evaluation:', f'{EVAL_SEEDS[0]}-{EVAL_SEEDS[-1]}', f'({len(EVAL_SEEDS)} seeds)')
"""),
    markdown(r"""
## 1. Context-local atlas construction

Let $k_t$ be the physical localization chart used by the existing graph router. Offline, each chart's complete Hankel columns are clustered into eight modes using normalized, controller-weighted future behavior,

$$
\phi_i^f = Q^{1/2}Y_{f,i}^{\mathrm{norm}}.
$$

For every pair $(k,m)$, the 350 columns in physical chart $k$ nearest future-mode prototype $m$ form an overlapping contextual support. Its centered rank-60 SVD basis is cached. Thus each physical region contains eight alternative behavioral charts, but DeePC always receives one contextual basis—not the ordinary physical-chart basis.

At every step, only the eight modes of the measured chart are scored:

$$
s_{k,m}(t)=z\!\left(\|\phi^f_{k,m}-r_t^f\|_2^2\right)
+0.5z\!\left(d_G(\operatorname{dest}(k,m),k_g)\right)
+0.1\,\mathbf 1[m\ne m_{t-1}],
$$

where $d_G$ is the frozen physical transition-graph distance and the small persistence term applies only while localization remains in the same physical chart. This factors context as **physical compatibility first, future intent second**. It eliminates the fast global prototype switching seen in the original pure contextual atlas without introducing a trigger.
"""),
    code(r"""
artifact = pure.load_chart_local_contextual_atlas(ARTIFACT_PATH)
config = pure.ChartLocalContextualConfig(**json.loads(artifact['contextual_config'].item()))
pd.Series({
    'physical localization charts': artifact['prototypes'].shape[0],
    'context modes per chart scored online': artifact['prototypes'].shape[1],
    'total cached contextual bases': int(np.prod(artifact['prototypes'].shape[:2])),
    'support columns per contextual basis': artifact['selected_indices'].shape[2],
    'cached basis rank': artifact['matrix_bases'].shape[-1],
    'graph weight': config.graph_weight,
    'within-chart switch penalty': config.switch_penalty,
})
"""),
    markdown(r"""
## 2. Frozen paired evaluation

Strict success requires both the final-30 and final-100 windows to have mean fingertip error below 1 cm and maximum error below 2 cm. All methods use the same offline IID-random library, 500 MuJoCo seeds, goals, graph-routed references, DeePC objective, constraints, horizon, and rank.
"""),
    code(r"""
def load_rows(directory):
    payload = json.loads((directory/'summary.json').read_text())
    rows = {int(row['seed']): row for row in payload['results'] if int(row['seed']) in EVAL_SEEDS}
    assert set(rows) == set(EVAL_SEEDS)
    return payload, rows

def strict(row):
    return bool(row['success_30'] and row['success_100'])

def wilson(successes, trials):
    z = 1.959963984540054
    p = successes/trials
    denominator = 1 + z*z/trials
    center = (p + z*z/(2*trials))/denominator
    half = z*math.sqrt((p*(1-p) + z*z/(4*trials))/trials)/denominator
    return center-half, center+half

baseline_payload, baseline = load_rows(BASELINE_DIR)
global_payload, global_context = load_rows(GLOBAL_CONTEXT_DIR)
pure_payload, pure_context = load_rows(PURE_CONTEXT_DIR)
triggered_payload, triggered = load_rows(TRIGGERED_DIR)
methods = {
    'Physical Atlas': baseline,
    'Global contextual (pure)': global_context,
    'Chart-local contextual (pure)': pure_context,
    'Triggered contextual (hybrid)': triggered,
}

table_rows = []
for method, rows in methods.items():
    successes = sum(strict(row) for row in rows.values())
    low, high = wilson(successes, len(rows))
    table_rows.append({
        'method': method, 'successes': successes, 'trials': len(rows),
        'success rate': successes/len(rows), '95% Wilson CI': f'[{low:.3f}, {high:.3f}]',
    })
outcomes = pd.DataFrame(table_rows).set_index('method')
display(outcomes.style.format({'success rate': '{:.1%}'}))

assert outcomes.loc['Chart-local contextual (pure)', 'successes'] == 463
assert outcomes.loc['Physical Atlas', 'successes'] == 443
assert all(row['online_hankel_columns_scored'] == 0 for row in pure_context.values())
assert all(row['online_svd_rebuild'] is False for row in pure_context.values())
assert all(row['fitted_dynamics_model'] is False for row in pure_context.values())
assert all(row['controller'] == 'always-on chart-local contextual Atlas-DeePC' for row in pure_context.values())
"""),
    code(r"""
fig, ax = plt.subplots(figsize=(9.5, 4.8))
colors = ['#6b7280', '#93a5b1', '#167d9a', '#7a5195']
bars = ax.bar(outcomes.index, 100*outcomes['success rate'], color=colors)
ax.set(ylabel='strict success rate (%)', ylim=(82, 99), title='Matched Reacher evaluation: seeds 211-710')
ax.tick_params(axis='x', rotation=16)
for bar, successes in zip(bars, outcomes['successes']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{successes}/500', ha='center', fontweight='bold')
ax.grid(axis='y', alpha=.25)
plt.tight_layout()
plt.show()
"""),
    markdown(r"""
## 3. Paired improvement over the physical atlas

Aggregate rates can conceal a controller that repairs and breaks equal numbers of seeds. The exact paired comparison below counts both directions.
"""),
    code(r"""
repairs = [seed for seed in EVAL_SEEDS if strict(pure_context[seed]) and not strict(baseline[seed])]
regressions = [seed for seed in EVAL_SEEDS if strict(baseline[seed]) and not strict(pure_context[seed])]
p_value = binomtest(min(len(repairs), len(regressions)), len(repairs)+len(regressions), 0.5).pvalue
display(pd.Series({
    'pure-contextual repairs': len(repairs),
    'pure-contextual regressions': len(regressions),
    'net paired gain': len(repairs)-len(regressions),
    'exact two-sided McNemar p': p_value,
}))
print('Repair seeds:', repairs)
print('Regression seeds:', regressions)
assert (len(repairs), len(regressions)) == (40, 20)
"""),
    markdown(r"""
## 4. Why the chart-local factorization works

The global contextual atlas selects one of 128 supports built across the full configuration space. Its failed rollouts switch prototypes frequently because a moving waypoint changes the future score enough to replace a physically useful basis. Merely doubling the codebook to 256 prototypes was worse (90/100 on development), and a global score-margin sweep also failed to beat the physical atlas.

Chart-local modes change the representation instead of only tuning the global selector. Every candidate basis is compatible with the currently measured physical region; the future score chooses the motion intent inside that region. The diagnostic below measures actual basis switching in the saved paired rollouts.
"""),
    code(r"""
def within_chart_switch_fractions(directory, field):
    values = []
    for seed in EVAL_SEEDS:
        with np.load(directory/f'seed{seed}_control_results.npz', allow_pickle=True) as data:
            chart = data['chart'].astype(int)
            selected = data[field].astype(int)
            values.append(np.mean(
                (chart[1:] == chart[:-1]) & (selected[1:] != selected[:-1])
            ))
    return np.asarray(values)

global_switch = within_chart_switch_fractions(GLOBAL_CONTEXT_DIR, 'contextual_cluster')
local_switch = within_chart_switch_fractions(PURE_CONTEXT_DIR, 'contextual_mode')
display(pd.DataFrame([
    {'method': 'Global contextual', 'mean within-chart switch fraction': global_switch.mean(),
     'median': np.median(global_switch), '95th percentile': np.percentile(global_switch, 95)},
    {'method': 'Chart-local contextual', 'mean within-chart switch fraction': local_switch.mean(),
     'median': np.median(local_switch), '95th percentile': np.percentile(local_switch, 95)},
]).set_index('method').style.format('{:.3%}'))
"""),
    code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
counts = artifact['mode_counts']
im = axes[0].imshow(np.log10(np.maximum(counts, 1)), aspect='auto', cmap='Blues')
axes[0].set(xlabel='context mode', ylabel='physical chart', title='Hard assignment support (log10 count)')
fig.colorbar(im, ax=axes[0], shrink=.85)

dest = artifact['mode_destinations']
im = axes[1].imshow(dest, aspect='auto', cmap='turbo', vmin=0, vmax=47)
axes[1].set(xlabel='context mode', ylabel='physical chart', title='Terminal graph destination represented by each mode')
fig.colorbar(im, ax=axes[1], shrink=.85, label='destination chart')
plt.tight_layout()
plt.show()
"""),
    markdown(r"""
## 5. Reproduce or extend

The saved 500-seed panel predates the restored runner. Seed 232 was replayed through the restored implementation and matches every saved mode, input, and output exactly. The command below reuses the frozen 384-basis artifact; it does not retrain or invoke a trigger.
"""),
    code(r"""
RERUN = False
if RERUN:
    pure.run_panel(
        range(711, 721),
        BASELINE_DIR,
        RESULT_ROOT/'pure_contextual_reproduction',
        config,
        data_path=ROOT/'data/select_iid_offline_hankel.npy',
        workers=8,
        source_artifact=ARTIFACT_PATH,
    )
else:
    print('Set RERUN=True to evaluate new seeds with the frozen pure contextual atlas.')
"""),
    markdown(r"""
## Conclusion

The trigger-free objective is achieved. Chart-local contextual Atlas-DeePC reaches **463/500 (92.6%)**, compared with **443/500 (88.6%)** for physical Atlas-DeePC and **444/500 (88.8%)** for the global pure-contextual atlas. The paired improvement over the physical baseline is 40 repairs versus 20 regressions (exact two-sided McNemar $p=0.0135$).

The essential change is not a larger contextual codebook. It is a factorized atlas: physical localization restricts the admissible behavioral region, while always-on $Y_f$ context selects one of eight cached motion modes inside that region. Every control action still comes from DeePC using a contextual cached basis, from the first step onward.
"""),
]

nbf.write(nb, OUTPUT)
print(OUTPUT)
