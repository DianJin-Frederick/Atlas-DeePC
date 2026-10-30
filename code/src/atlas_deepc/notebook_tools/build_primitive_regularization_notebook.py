"""Build and optionally execute the primitive baseline regularization report."""

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
# Primitive Atlas-DeePC: Regularization on Failure Seed 252

Vary only the reduced-basis coefficient penalty $\lambda_g\|g\|_2^2$.
The reference experiment is the **eight-output primitive baseline**, not the
graph-routed or contextual controller. Its saved seed-252 rollout has final-100
mean fingertip error **231.159 mm** at $\lambda_g=0.05$.

The exact saved offline matrix and 48-chart atlas are reused, without refitting.
Keep rank 60, raw clustering/online coordinates, nearest-centroid selection,
one fixed nearest eligible terminal configuration, 320 steps, the same $Q,R$ and
initialization/input-increment penalties, Clarabel, and no coefficient boxes or affine constraint.
There are **no graph waypoints and no rescue**. Each run is serial with one numerical thread.

The expanded sweep tests **50 distinct values**: zero, the original positive values,
and 34 additional logarithmically spaced values. Positive weights span
$10^{-4}$ through $10^6$. Previous runs are reused only after validating their manifests.

This is a single-seed diagnostic, not evidence of population-level improvement.
""")
    code(r"""
import os, sys, json
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
os.environ.setdefault('MUJOCO_GL','disable')
ROOT=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'src/atlas_deepc').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from atlas_deepc.ablations import regularization_sweep as experiment
%matplotlib inline
SEEDS=[252]
OUTPUT_ROOT=experiment.RESULT_ROOT
RUN_EXPERIMENTS=False
LAMBDA_VALUES=list(experiment.LAMBDAS)
if RUN_EXPERIMENTS:
    experiment.run(SEEDS,LAMBDA_VALUES,OUTPUT_ROOT)
report=experiment.summarize(SEEDS,LAMBDA_VALUES,OUTPUT_ROOT)
table=pd.DataFrame(report['records']).sort_values('lambda_g').reset_index(drop=True)
assert len(table)==50 and table.lambda_g.nunique()==50
display(pd.DataFrame(report['baseline_reproduction']))
columns=['lambda_g','strict_success','final100_mean_mm','final100_max_mm',
         'minimum_mm','charts','chart_switches','unsolved_steps']
display(HTML('<details><summary>All 50 regularization results</summary>'+
             table[columns].to_html(index=False,float_format=lambda v:f'{v:.6g}')+'</details>'))
""")
    md("""
## Error versus Regularization

The positive $\lambda_g$ values use a logarithmic horizontal axis. Zero is shown
in a separate small panel, not placed at an arbitrary positive value.
The upper row starts the error axis at zero; the lower row is an explicitly
labeled zoom. Both final-100 mean tracking error and whole-rollout mean tracking
error are shown in millimeters. Every marker is an evaluated regularizer value.
Success requires both final-30 and final-100 mean errors below 10 mm and each
window's maximum error below 20 mm. All statistics use true simulated fingertip positions.
""")
    code(r"""
positive=table[table.lambda_g>0]
zero=table[table.lambda_g==0].iloc[0]
series=[('final100_mean_mm','Final-100 mean','#0072B2'),
        ('mean_rollout_mm','Whole-rollout mean','#D55E00')]
values=table[[s[0] for s in series]].to_numpy()
padding=max(.15,.12*np.ptp(values))
limits=[(0,1.06*values.max()),(values.min()-padding,values.max()+padding)]
fig,axes=plt.subplots(2,2,figsize=(12,8),sharey='row',
                      gridspec_kw={'width_ratios':[1,8],'wspace':.06,'hspace':.35})
for row in range(2):
    left,right=axes[row]
    for key,label,color in series:
        left.plot([0],[zero[key]],'o',color=color,ms=5)
        right.semilogx(positive.lambda_g,positive[key],'-o',color=color,
                       ms=3.5,lw=1.4,label=label)
    left.set(xticks=[0],xticklabels=['0'],xlim=(-.5,.5),ylim=limits[row],xlabel=r'$\lambda_g$')
    right.set(ylim=limits[row],xlabel=r'$\lambda_g>0$ (log scale)')
    right.axvline(.05,color='#888888',ls='--',lw=.9,label=r'Baseline $\lambda_g=0.05$')
    for ax in (left,right): ax.grid(alpha=.2)
    left.set_ylabel('Mean tracking error (mm)'+('\nZoomed scale' if row else '\nFull scale'))
    if row==0:
        for ax in (left,right): ax.axhline(10,color='#555555',ls=':',lw=1)
        right.legend(loc='lower left',bbox_to_anchor=(0,.10))
        right.text(.99,.055,'Dotted line: 10 mm final-window mean criterion',
                   transform=right.transAxes,ha='right',fontsize=10)
axes[0,0].set_title('Unregularized',fontsize=10)
axes[0,1].set_title('Seed 252: tracking error versus regularization (50 values)')
axes[1,1].set_title('Zoom of the same data')
fig.subplots_adjust(left=.10,right=.98,top=.94,bottom=.08)
plt.show()
""")
    md("""
## Representative Error Traces

These plots show representative values spanning the 50-value sweep, including the
baseline value, divided into small panels for readability. All 50 complete traces
are retained in the result folders.
The run is not stopped early when a fingertip first enters the target region.
""")
    code(r"""
selected=np.unique(np.r_[np.linspace(0,len(table)-1,min(12,len(table))).round().astype(int),
                         table.index[table.lambda_g==.05].to_numpy()])
groups=np.array_split(selected,int(np.ceil(len(selected)/4)))
fig,axes=plt.subplots(len(groups),1,figsize=(12,3.3*len(groups)),squeeze=False)
for ax,indices in zip(axes[:,0],groups):
    for j,index in enumerate(indices):
        row=table.iloc[index]
        with np.load(row.result_path,allow_pickle=True) as trace:
            ax.plot(1000*trace['distance'],lw=1.5,ls=['-','--','-.',':'][j],label=rf'$\lambda_g={row.lambda_g:.3g}$')
    ax.axhline(10,ls=':',color='#555555',lw=1)
    ax.set(xlabel='Control step',ylabel='Fingertip error (mm)',ylim=(0,None))
    ax.legend(ncol=4); ax.grid(alpha=.2)
fig.tight_layout(); plt.show()
""")
    md("""
## Computation and Prediction Diagnostics

Timings exclude the first five control steps. Coefficient norms and one-step
fingertip prediction errors help distinguish a changed optimization solution from
an unchanged closed-loop outcome. The full trajectories, control inputs, chart
histories, predictions, coefficients, and module timings are saved for each value.
""")
    code(r"""
columns=['lambda_g','median_g_norm','median_prediction_error_mm',
         'median_online_ms','median_solve_ms','unsolved_steps']
display(HTML('<details><summary>All 50 computation and prediction diagnostics</summary>'+
             table[columns].to_html(index=False,float_format=lambda v:f'{v:.6g}')+'</details>'))
successful=table[table.strict_success]
best=table.loc[table.final100_mean_mm.idxmin()]
print('Successful lambda values:',successful.lambda_g.tolist())
print(f'Lowest final-100 mean: lambda_g={best.lambda_g:g}, {best.final100_mean_mm:.4f} mm.')
print('Saved atlas hash:',report['atlas_sha256'])
print('Verified: the terminal reference is identical across all tested values.')
""")
    path = project_root() / 'notebooks/reacher/ablations/ablation_primitive_lambda_g_seed252.ipynb'
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'name':'gym-robotics-mujoco','display_name':'gym-robotics-mujoco','language':'python'}})
    nbf.write(notebook,path)
    return path


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    path=build()
    if args.execute:
        from nbclient import NotebookClient
        notebook=nbf.read(path,as_version=4)
        NotebookClient(notebook,timeout=600,kernel_name='gym-robotics-mujoco',
                       resources={'metadata':{'path':str(path.parent)}}).execute()
        nbf.validate(notebook)
        output=project_root()/'experiments/reacher/primitive_lambda_g_seed252_v1/report'
        output.mkdir(parents=True,exist_ok=True)
        figures=0
        for cell in notebook.cells:
            for item in cell.get('outputs',[]):
                assert item.output_type!='error'
                if 'image/png' in item.get('data',{}):
                    figures+=1
                    (output/f'figure{figures}.png').write_bytes(base64.b64decode(item.data['image/png']))
                    if figures==1:
                        (output/'tracking_error_vs_lambda_g.png').write_bytes(base64.b64decode(item.data['image/png']))
        nbf.write(notebook,path)
        print(f'Executed report with {figures} figures.')
    print(path)
