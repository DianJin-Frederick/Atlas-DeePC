"""Build the ranks 55--65 physical-noise report without launching rollouts."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from atlas_deepc.common.paths import project_root
from pathlib import Path
import argparse
import hashlib
import json
import shutil
from datetime import datetime,timezone
import nbformat as nb
ROOT=project_root()
NOTEBOOK=ROOT/'notebooks/reacher/ablations/ablation_measurement_noise.ipynb'
RESULT=ROOT/'experiments/reacher/routed_deepc_iid/physical_sensor_ranks55_65_100_v1'


def build():
    md,code=nb.v4.new_markdown_cell,nb.v4.new_code_cell
    cells=[md(r'''# Atlas-DeePC: physical sensor noise across ranks 55–65

**11 ranks × 6 noise levels × 100 seeds = 6,600 rollouts.**
Ranks 55–65 inclusive; multipliers **0×, 1×, 2×, 3×, 4×, 5×**; seeds 211–310.
**10× is excluded from every rank and all plots.** Its historical files are retained.
The completed rank-59 and rank-62 panels contribute 1,200 reused rollouts;
5,400 additional rollouts evaluate the other nine ranks.

At 1×, independent primitive Gaussian errors have SD **0.1° per joint angle,
1 mm per fingertip-position axis, and 0.01 rad/s per joint velocity**. Multiply
these fixed physical scales by the level; do not scale by signal magnitude.
Rebuild all six trigonometric features from the same two noisy angles. Position
and velocity represent independent sensor measurements. Apply the identical model
offline and online, keeping input commands and target coordinates exact.

For each level, all ranks use identical corrupted offline samples, chart memberships,
medoids and routing graph. Repeated Hankel entries share one corrupted sample.
Reuse the previously fitted rank-59 atlas, then compute nested rank-65 SVD bases
and take rank prefixes with the original coefficient-bound rule. The prefixes are
verified against completed ranks 59 and 62. Offline sources and numerical code are frozen.

All ranks retain λg=.05, Clarabel, 48 linear charts, graph waypoints, iterative N=10
terminal references and 320 steps. New trials run serially with one numerical thread;
new-rank order rotates within each seed. Standard-normal sensor draws and initial
targets are paired. Rank-59/62 timings come from earlier sessions, so cross-session
timing differences cannot be attributed solely to rank.

Strict success requires mean true fingertip error <10 mm and maximum <20 mm in
both final-30 and final-100 windows. Error distributions include unsuccessful trials.
This notebook is ready for **local Run All**. `RUN_EXPERIMENTS=True` resumes the
separate sweep first, then builds the report. Set it to False for report-only use.
The Codex-launched run was stopped at your request; its monitor is paused.
Incomplete panels display counts and are omitted from final success/error plots.
'''),code('''RUN_EXPERIMENTS = True  # Resume locally, then display the report.
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'src/atlas_deepc/noise/sensor_noise_rank_sweep.py').exists())
RESULT=ROOT/'experiments/reacher/routed_deepc_iid/physical_sensor_ranks55_65_100_v1'
if RUN_EXPERIMENTS:
    import os, sys, subprocess, signal
    from collections import deque
    from datetime import datetime
    RESULT.mkdir(parents=True,exist_ok=True)
    log_path=RESULT/('resume_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.log')
    command=[sys.executable,'-u',str(ROOT/'src/atlas_deepc/noise/sensor_noise_rank_sweep.py')]
    print('Resuming compatible checkpoints. Log:',log_path,flush=True)
    tail=deque(maxlen=40)
    with log_path.open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,bufsize=1,start_new_session=os.name!='nt')
        try:
            for line in process.stdout:
                log.write(line);log.flush();tail.append(line);print(line,end='',flush=True)
            if process.wait()!=0:
                raise RuntimeError('Sweep stopped; completed checkpoints retained. '+str(log_path)+'\\n'+''.join(tail))
        except KeyboardInterrupt:
            if os.name!='nt':os.killpg(process.pid,signal.SIGINT)
            else:process.terminate()
            process.wait()
            raise
FIGURES=RESULT/'figures'; FIGURES.mkdir(exist_ok=True)
summary=json.loads((RESULT/'summary.json').read_text())
RANKS=list(range(55,66)); LEVELS=[0,1,2,3,4,5]
assert summary['ranks']==RANKS and summary['levels']==LEVELS and summary['seeds']==list(range(211,311))
frame=pd.DataFrame([{k:v for k,v in p.items() if k not in ('results','timing')} for p in summary['panels']])
assert not frame.duplicated(['rank','multiplier']).any()
frame['success_rate_pct']=100*frame.successes/frame.n
print(f"{'COMPLETE' if summary['complete'] else 'PARTIAL'}: {summary['rollouts']}/6600 validated rollouts")
if (RESULT/'progress.json').exists():print('Live sweep progress:',json.loads((RESULT/'progress.json').read_text()))
display(frame.pivot(index='rank',columns='multiplier',values='n').reindex(index=RANKS,columns=LEVELS).fillna(0).astype(int).rename_axis(columns='Noise multiplier'))
complete=frame[frame.complete].copy()
if summary['complete']:
    assert len(complete)==66 and (complete.n==100).all()
    assert json.loads((RESULT/'invariance_final.json').read_text())['passed']
plt.rcParams.update({'figure.dpi':120,'font.size':10})
COLORS=plt.cm.viridis(np.linspace(.05,.95,len(RANKS)))
def save(name):
    plt.savefig(FIGURES/(name+'.png'),dpi=160,bbox_inches='tight')
    plt.savefig(FIGURES/(name+'.pdf'),bbox_inches='tight')
    plt.show();plt.close()
'''),md('''## Success and true tracking error

Each displayed result uses all 100 paired seeds. Blank cells are incomplete.
Error is each seed's mean true fingertip error over its final 100 steps.
'''),code('''display(complete[['rank','multiplier','successes','n','success_rate_pct','median_error_mm','p90_error_mm','p95_error_mm','p99_error_mm','unsolved_qps']].set_index(['rank','multiplier']).round(3))
complete.to_csv(RESULT/'rank_noise_results.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(13,7),constrained_layout=True)
for ax,metric,title,cmap,fmt in [(axes[0],'success_rate_pct','Strict success (%)','viridis','.0f'),
                               (axes[1],'median_error_mm','Median final-100 mean error (mm)','magma_r','.2f')]:
    grid=complete.pivot(index='rank',columns='multiplier',values=metric).reindex(index=RANKS,columns=LEVELS)
    v=grid.to_numpy(dtype=float)
    image=ax.imshow(v,aspect='auto',cmap=cmap,vmin=0,vmax=100 if metric=='success_rate_pct' else None)
    for i in range(len(RANKS)):
        for j in range(len(LEVELS)):
            if np.isfinite(v[i,j]):
                rgba=image.cmap(image.norm(v[i,j])); luminance=.299*rgba[0]+.587*rgba[1]+.114*rgba[2]
                ax.text(j,i,format(v[i,j],fmt),ha='center',va='center',color='black' if luminance>.55 else 'white',fontsize=9)
    ax.set_xticks(range(6),[f'{l}×' for l in LEVELS]);ax.set_yticks(range(11),RANKS)
    ax.set(xlabel='Noise multiplier',ylabel='Retained rank',title=title);fig.colorbar(image,ax=ax,shrink=.8)
save('rank_noise_heatmaps')
fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
for rank,color in zip(RANKS,COLORS):
    rows=complete[complete['rank']==rank].sort_values('multiplier')
    if len(rows):
        axes[0].plot(rows.multiplier,rows.success_rate_pct,'o-',color=color,label=str(rank))
        axes[1].plot(rows.multiplier,rows.p90_error_mm,'o-',color=color,label=str(rank))
for ax in axes:ax.set_xticks(LEVELS);ax.set_xlabel('Noise multiplier');ax.grid(alpha=.2)
axes[0].set(ylabel='Strict success (%)',ylim=(0,102));axes[1].set(ylabel='P90 final-100 mean error (mm)',ylim=(0,None))
axes[1].legend(title='Rank',ncol=2,fontsize=8)
save('rank_noise_profiles')
'''),md('''## Full error distributions

These CDFs include all failed trials. A CDF value at 10 mm checks only final-100
mean error; strict success additionally requires the other mean and maximum criteria.
'''),code('''fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
for level,ax in zip(LEVELS,axes.flat):
    for rank,color in zip(RANKS,COLORS):
        panel=next((p for p in summary['panels'] if p['rank']==rank and p['multiplier']==level and p['complete']),None)
        if panel:
            values=np.sort([r['mean_last_100']*1000 for r in panel['results']])
            ax.step(values,np.arange(1,101)/100,where='post',label=str(rank),color=color)
    ax.set(xscale='log',ylim=(0,1.02),xlabel='Final-100 mean true error (mm)',ylabel='Fraction of seeds',title=f'{level}× sensor noise')
    ax.axvline(10,color='gray',linestyle='--');ax.grid(alpha=.2)
handles,labels=axes.flat[0].get_legend_handles_labels()
fig.legend(handles,labels,title='Rank',loc='outside upper center',ncol=11)
save('rank_noise_error_cdfs')
'''),md('''## Timing and solver outcomes

Per-step timings pool 31,500 steps per panel after excluding the first five steps
of each rollout. QP time includes construction and solve; noise injection and
simulator stepping are excluded. No new dedicated warm-up is added. Ranks 59/62
are historical sessions; the other ranks share the expanded serial session.
Setup costs remain separate in each offline manifest and the original studies.
'''),code('''timing=[]
for panel in summary['panels']:
    if panel['complete']:
        assert panel['timing']['timed_steps']==31500
        timing.extend({'rank':panel['rank'],'multiplier':panel['multiplier'],'session':panel['session'],**t} for t in panel['timing']['modules'])
timing=pd.DataFrame(timing)
display(timing.set_index(['rank','multiplier','session','module'])[['mean_ms','median_ms','p95_ms','p99_ms']].round(4))
timing.to_csv(RESULT/'module_timing_results.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
for rank,color in zip(RANKS,COLORS):
    rows=timing[(timing['rank']==rank)&(timing.module=='Total online control')].sort_values('multiplier')
    for ax,stat in zip(axes,['median_ms','p99_ms']):
        ax.plot(rows.multiplier,rows[stat],'o--' if rank in (59,62) else 'o-',color=color,label=f'{rank}'+(' (earlier)' if rank in (59,62) else ''))
for ax,title in zip(axes,['Median','P99']):
    ax.set(xlabel='Noise multiplier',ylabel='Total online time (ms)',title=title,ylim=(0,None));ax.set_xticks(LEVELS);ax.grid(alpha=.2)
axes[1].legend(ncol=2,fontsize=8)
fig.suptitle('Descriptive timings: separate sessions for historical ranks 59/62')
save('rank_noise_timing')
print('Unsolved QPs in available panels:',int(frame.unsolved_qps.sum()))
setup=[json.loads(p.read_text()) for p in sorted((RESULT/'offline').glob('*/manifest.json'))]
display(pd.DataFrame(setup)[['level','setup_seconds','source_atlas_reused','nested_prefix_59_62_verified']])
''')]
    return nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (gym-robotics-mujoco)','language':'python','name':'gym-robotics-mujoco'}})


def main(execute=False):
    archive=ROOT/'archive/notebooks/ablation_measurement_noise_two_ranks_before_55_65.ipynb'
    if not archive.exists():shutil.copy2(NOTEBOOK,archive)
    n=build()
    for c in n.cells:
        if c.cell_type=='code':compile(c.source,str(NOTEBOOK),'exec')
    if execute:
        for c in n.cells:
            if c.cell_type=="code":c.source=c.source.replace("RUN_EXPERIMENTS = True", "RUN_EXPERIMENTS = False")
        from nbclient import NotebookClient
        NotebookClient(n,timeout=900,kernel_name='gym-robotics-mujoco',resources={'metadata':{'path':str(NOTEBOOK.parent)}}).execute()
    nb.validate(n);nb.write(n,NOTEBOOK)
    if execute:
        codes=[c for c in n.cells if c.cell_type=='code']
        assert len(codes)==5 and all(c.execution_count is not None for c in codes)
        assert not any(o.output_type=='error' for c in codes for o in c.outputs)
        assert sum('image/png' in o.get('data',{}) for c in codes for o in c.outputs)==4
        write={'executed_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),'code_cells':5,'figures':4,'visual_inspection':'pending'}
        (RESULT/'notebook_execution.json').write_text(json.dumps(write,indent=2)+'\n')
    print(NOTEBOOK)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');main(parser.parse_args().execute)
