"""Build the paired eight/ten-output graph-cost notebook."""

import argparse
import base64
import json

import nbformat as nbf

from atlas_deepc.common.paths import project_root


def build():
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text.strip()))
    def code(text):
        cells.append(nbf.v4.new_code_cell(text.strip()))

    md(r"""
# Graph-Cost Ablation: Centroid versus Chordal Distance

We compare **graph-routed Atlas-DeePC** under two edge costs, separately for
$p=8$ and $p=10$ outputs. All four conditions use seeds **211--310**, the same
original offline episodes, raw (unstandardized) physical-configuration K-means,
48 charts, rank-60 **linear** full-trajectory bases, and a fixed nearest eligible
terminal configuration. There is no contextual rescue, coefficient box, or affine constraint.

This is a paired exploratory ablation on previously used seeds, not an untouched test set.
Only the graph edge cost changes within each output dimension. Saved baseline results
are preserved; these four conditions are evaluated together in a separate result folder.
""")
    code(r"""
import os, sys, json
from pathlib import Path
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('MUJOCO_GL', 'disable')
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'src/atlas_deepc').is_dir())
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from atlas_deepc.ablations import chordal_graph_cost as experiment
from atlas_deepc.common.metrics import strict_success
from atlas_deepc.atlas.candidate_goal_routing_experiment import shortest_path_tables
from atlas_deepc.atlas.routed_deepc_experiment import shortest_chart_path
%matplotlib inline
plt.rcParams.update({'font.size': 11, 'mathtext.fontset': 'cm'})
SEEDS = list(range(211, 311))
OUTPUT_ROOT = ROOT / 'experiments/reacher/chordal_graph_cost_v1'
RUN_EXPERIMENTS = False  # Set True to run new seeds; matching completed rollouts are reused.
WORKERS = 4
COLORS = {'centroid': '#0072B2', 'chordal': '#D55E00'}
config = experiment.experiment.default_config()
""")
    md(r"""
## 1. What Changes

Each chart uses an orthonormal basis $\Phi_k\in\mathbb R^{(m+p)L\times r}$ of the
**complete input-output Hankel columns**, not just the clustering features:
$m=2$, $L=T_{\rm ini}+T_f=2+10=12$, and $r=60$.
Thus the ambient dimensions are 120 and 144, respectively.

For equal-rank charts, the chordal distance is
$$d_{\rm ch}(k,k')=\frac{1}{\sqrt2}\|\Phi_k\Phi_k^\top-\Phi_{k'}\Phi_{k'}^\top\|_F
=\sqrt{r-\|\Phi_k^\top\Phi_{k'}\|_F^2}.$$
It measures subspace orientation, is invariant to basis rotations, and is not a
distance between representative arm poses. No additional row scaling, whitening,
singular-value weighting, or tuning is applied.

The observed directed edges remain exactly the same: $k\ne k'$ and $n_{kk'}\ge3$.
Their weights become $w_{kk'}=d_{\rm ch}(k,k')/\sqrt{n_{kk'}}$.
The centroid control retains the existing implementation's distance
$\|c_k-c_{k'}\|_2/\sqrt{d_{\rm cluster}}$, also divided by $\sqrt{n_{kk'}}$.
That constant normalization does not change shortest routes within an atlas.

**Reference construction stays fixed:** select the offline last-past pose with the
closest fingertip to the target, subject to the existing 6 cm source-episode target
separation rule. Copy its configuration channels, zero its velocities, and use the
exact target for its position channels. The candidate index and goal chart never
change during a rollout. Graph waypoints still change with the measured current chart;
the existing 7 cm terminal-reference latch is unchanged.

The eight-output case drops only $\cos(q_1+q_2),\sin(q_1+q_2)$.
Each dimension gets its own fitted atlas, but both costs share identical memberships,
medoids, bases, edges, controller weights, and terminal goals within that dimension.
""")
    code(r"""
display(pd.Series({
    'evaluation seeds': f'{SEEDS[0]}--{SEEDS[-1]} ({len(SEEDS)})',
    'outputs': '8 and 10', 'charts / rank': f'{config.clusters} / {config.svd_rank}',
    'steps / past / future': f'{config.steps} / 2 / 10',
    'Q, eight outputs': '[5000, 5000, 5000, 5000, 20, 20, 100000, 100000]',
    'Q, ten outputs': 'eight-output Q plus [5000, 5000]',
    'R': '0.0001 I_2', 'lambda_g': config.lam_g,
    'lambda_ini / lambda_delta_u': '100000 / 5', 'solver': 'Clarabel',
    'standardization / g boxes / affine constraint': 'none / none / none',
    'terminal policy': 'one fixed nearest eligible configuration',
    'graph routing': 'enabled in all four conditions',
    'workers / numerical threads per worker': f'{WORKERS} / 1',
}))
if RUN_EXPERIMENTS:
    experiment.run_all(SEEDS, OUTPUT_ROOT, WORKERS)
panels = experiment.load_results(SEEDS, OUTPUT_ROOT)
print('Verified: paired terminal references and goal chart/index histories are identical.')
""")
    md("""
## 2. Paired Success and Fingertip Error

Strict success requires the final-30 and final-100 mean true fingertip errors to be
below 10 mm, and the maximum error in each window below 20 mm. These are full
320-step rollouts; entering the target region does not terminate a run.
""")
    code(r"""
records, paired = [], []
for dimension in (8, 10):
    for method in experiment.METHODS:
        rows = panels[dimension, method]
        errors = 1000*np.array([r['mean_last_100'] for r in rows])
        rollout_errors=[]
        for row in rows:
            with np.load(row['result_path'],allow_pickle=True) as trace:
                rollout_errors.append(float(np.mean(trace['distance']))*1000)
        records.append({'outputs': dimension, 'cost': method,
                        'successes': sum(map(strict_success, rows)), 'seeds': len(rows),
                        'success (%)': 100*np.mean(list(map(strict_success, rows))),
                        'median final-100 (mm)': np.median(errors),
                        'p90 final-100 (mm)': np.percentile(errors, 90),
                        'mean final-100 (mm)': errors.mean(),
                        'mean whole-rollout error (mm)': np.mean(rollout_errors),
                        'unsolved QPs': sum(r['unsolved_steps'] for r in rows)})
    before, after = panels[dimension, 'centroid'], panels[dimension, 'chordal']
    repaired = [a['seed'] for a,b in zip(before,after) if not strict_success(a) and strict_success(b)]
    regressed = [a['seed'] for a,b in zip(before,after) if strict_success(a) and not strict_success(b)]
    paired.append({'outputs': dimension, 'repaired': repaired, 'regressed': regressed,
                   'net repaired': len(repaired)-len(regressed)})
summary = pd.DataFrame(records)
display(summary.round(3))
display(pd.DataFrame(paired))
for dimension, method in panels:
    print(f'R^{dimension}, {method} failures:', [r['seed'] for r in panels[dimension,method] if not strict_success(r)])

fig, axes = plt.subplots(1,3,figsize=(16,4.5))
for j,method in enumerate(experiment.METHODS):
    values = summary[summary.cost == method]['success (%)'].to_numpy()
    bars=axes[0].bar(np.arange(2)+(j-.5)*.34, values, .34, label=method, color=COLORS[method])
    axes[0].bar_label(bars, fmt='%.0f%%', padding=3)
axes[0].set(xticks=[0,1], xticklabels=['8 outputs','10 outputs'], ylim=(0,105), ylabel='Strict success (%)')
axes[0].legend()
for ax,dimension in zip(axes[1:],(8,10)):
    for method in experiment.METHODS:
        errors=np.sort(1000*np.array([r['mean_last_100'] for r in panels[dimension,method]]))
        ax.step(errors,np.arange(1,len(errors)+1)/len(errors),where='post',color=COLORS[method],label=method)
    ax.axvline(10,ls=':',color='#333333',label='10 mm mean criterion')
    ax.set(xlabel='Final-100 mean error (mm)',ylabel='Cumulative fraction',title=f'{dimension} outputs',xlim=(0,None),ylim=(0,1.02))
    ax.grid(alpha=.2); ax.legend()
fig.tight_layout(); plt.show()
""")
    md("""
## 3. Did the Routes Actually Change?

Compare offline shortest paths for every reachable ordered chart pair, then the
requested next charts in paired closed-loop runs. Closed-loop differences also
include subsequent state divergence; they are not a same-state policy comparison.
""")
    code(r"""
route_records=[]
for dimension in (8,10):
    folder=OUTPUT_ROOT/f'output{dimension}'
    atlas=experiment.experiment.routed.load_atlas(folder/'behavioral_atlas.npz')
    with np.load(folder/'graph_geometry.npz') as saved:
        geometry={key:saved[key] for key in saved.files}
    mask=np.isfinite(geometry['centroid_cost'])
    np.testing.assert_array_equal(mask,np.isfinite(geometry['chordal_cost']))
    variants={m:{**atlas,'routing_edge_costs':geometry[f'{m}_cost']} for m in experiment.METHODS}
    tables={m:shortest_path_tables(variants[m],config.edge_count_threshold)[0] for m in experiment.METHODS}
    changed=reachable=0
    for a in range(config.clusters):
        for b in range(config.clusters):
            if a==b: continue
            paths=[shortest_chart_path(variants[m],a,b,config.edge_count_threshold) for m in experiment.METHODS]
            assert bool(paths[0])==bool(paths[1])
            if paths[0]:
                reachable+=1; changed+=paths[0]!=paths[1]
    differing_steps=[]
    differing_references=[]
    for first,second in zip(panels[dimension,'centroid'],panels[dimension,'chordal']):
        with np.load(first['result_path'],allow_pickle=True) as a, np.load(second['result_path'],allow_pickle=True) as b:
            for method,trace in [('centroid',a),('chordal',b)]:
                np.testing.assert_allclose(trace['goal_route_cost'],tables[method][trace['chart'].astype(int),trace['goal_chart'].astype(int)])
            differing_steps.append(float(np.mean(a['next_chart']!=b['next_chart'])))
            differing_references.append(float(np.mean(np.any(np.abs(a['ref']-b['ref'])>1e-10,axis=1))))
    route_records.append({'outputs':dimension,'directed edges':int(mask.sum()),
                          'reachable ordered pairs':reachable,'changed shortest routes':changed,
                          'changed closed-loop next-chart steps (%)':100*np.mean(differing_steps),
                          'rollouts with different next charts':sum(v>0 for v in differing_steps),
                          'changed active-reference steps (%)':100*np.mean(differing_references),
                          'chordal distance min':geometry['chordal_distance'][mask].min(),
                          'chordal distance max':geometry['chordal_distance'][mask].max()})
display(pd.DataFrame(route_records).round(3))
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,dimension in zip(axes,(8,10)):
    with np.load(OUTPUT_ROOT/f'output{dimension}'/'graph_geometry.npz') as g:
        edge=np.isfinite(g['centroid_cost'])
        ax.scatter(g['centroid_cost'][edge],g['chordal_cost'][edge],s=16,alpha=.6,color='#0072B2')
    ax.set(title=f'{dimension} outputs',xlabel='Centroid edge cost',ylabel='Chordal edge cost')
    ax.grid(alpha=.2)
fig.tight_layout(); plt.show()
""")
    md("""
## 4. Representative Paired Error Traces

For each output dimension, show the largest improvement, largest regression, and
seed 277. Selection is diagnostic and made after evaluation; it is not independent validation.
The bottom row zooms to 0--25 mm. The top row retains the full error range.
""")
    code(r"""
for dimension in (8,10):
    a={r['seed']:r for r in panels[dimension,'centroid']}
    b={r['seed']:r for r in panels[dimension,'chordal']}
    delta={s:b[s]['mean_last_100']-a[s]['mean_last_100'] for s in SEEDS}
    chosen=list(dict.fromkeys([min(delta,key=delta.get),max(delta,key=delta.get),277 if 277 in SEEDS else SEEDS[0]]))
    fig,axes=plt.subplots(2,len(chosen),figsize=(5*len(chosen),7),squeeze=False)
    for j,seed in enumerate(chosen):
        for method,rows in [('centroid',a),('chordal',b)]:
            with np.load(rows[seed]['result_path'],allow_pickle=True) as trace:
                errors=1000*trace['distance']
            for ax in axes[:,j]:
                ax.plot(errors,color=COLORS[method],label=method,lw=1.4)
        for ax in axes[:,j]:
            ax.axhline(10,ls=':',color='#333333'); ax.grid(alpha=.2)
            ax.set(xlabel='Control step',ylabel='True fingertip error (mm)')
        axes[0,j].set_title(f'{dimension} outputs, seed {seed}')
        axes[0,j].legend(); axes[1,j].set_ylim(0,25)
    fig.tight_layout(); plt.show()
""")
    md("""
## 5. Computation Time

Subspace distances, SVD bases, and all-pairs shortest paths are computed **offline**.
The chordal score is not recomputed during control. Online graph routing reconstructs
the route from the measured chart to the fixed goal chart using cached predecessors.

Per-module timers exclude the first five steps of each rollout. The four panels run
concurrently with one numerical thread per worker. Timings therefore include shared
machine contention and should **not** be compared directly with isolated baseline
latencies or interpreted as an intrinsic speed difference between the edge costs.
""")
    code(r"""
timing_records=[]
offline=[]
for dimension in (8,10):
    folder=OUTPUT_ROOT/f'output{dimension}'
    manifest=json.loads((folder/'graph_manifest.json').read_text())
    offline.append({'outputs':dimension,**{k:v for k,v in manifest.items() if k.endswith('_seconds')}})
    for method in experiment.METHODS:
        result=json.loads((folder/method/'summary.json').read_text())
        for row in result['timing']['modules']:
            timing_records.append({'outputs':dimension,'cost':method,**row})
display(pd.DataFrame(offline).round(4))
timing=pd.DataFrame(timing_records)
display(timing[['outputs','cost','module','mean_ms','median_ms','p99_ms']].round(3))
fig,axes=plt.subplots(1,2,figsize=(13,4))
total=timing[timing.module=='Total online control']
labels=[f"R{r.outputs}\n{r.cost}" for r in total.itertuples()]
for ax,metric,title in zip(axes,('median_ms','p99_ms'),('Median online time','99th-percentile online time')):
    bars=ax.bar(labels,total[metric],color=[COLORS[m] for m in total.cost])
    ax.bar_label(bars,fmt='%.1f',padding=3)
    ax.set(title=title+' (shared machine)',ylabel='Time (ms)',ylim=(0,float(total[metric].max())*1.15))
fig.tight_layout(); plt.show()
""")
    md("""
## Saved Artifacts

`experiments/reacher/chordal_graph_cost_v1/output{8,10}/` contains the fitted atlas,
exact full-trajectory bases, labels, observed transition counts, both distance/cost
matrices, provenance manifests, and complete per-seed traces and module timings.
The paired loader checks that terminal references, selected sample indices, and goal
chart histories agree between costs. No original experiment files are overwritten.
""")
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    path = project_root() / 'notebooks/reacher/ablations/ablation_chordal_graph_cost.ipynb'
    nbf.write(notebook, path)
    return path


def execute_report(path):
    from nbclient import NotebookClient
    notebook = nbf.read(path, as_version=4)
    NotebookClient(notebook, timeout=600, kernel_name='gym-robotics-mujoco',
                   resources={'metadata': {'path': str(path.parent)}}).execute()
    nbf.validate(notebook)
    code_cells = [c for c in notebook.cells if c.cell_type == 'code']
    assert all(c.execution_count is not None for c in code_cells)
    assert not any(o.output_type == 'error' for c in code_cells for o in c.outputs)
    output = project_root() / 'experiments/reacher/chordal_graph_cost_v1/report'
    output.mkdir(parents=True, exist_ok=True)
    figures = 0
    for cell in code_cells:
        for item in cell.outputs:
            if 'image/png' in item.get('data', {}):
                figures += 1
                (output / f'figure{figures}.png').write_bytes(base64.b64decode(item.data['image/png']))
    assert figures == 5
    temporary = path.with_suffix('.ipynb.tmp')
    nbf.write(notebook, temporary)
    temporary.replace(path)
    (output / 'execution.json').write_text(json.dumps({
        'notebook': str(path), 'executed_code_cells': len(code_cells),
        'inline_figures': figures, 'rollouts': 400,
        'visual_inspection': 'pending'}, indent=2))
    print(f'Executed {len(code_cells)} code cells; saved {figures} figures.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    path = build()
    print(path)
    if args.execute:
        execute_report(path)
