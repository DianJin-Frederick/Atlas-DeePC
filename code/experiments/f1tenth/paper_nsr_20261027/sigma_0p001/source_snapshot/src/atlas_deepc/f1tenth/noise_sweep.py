"""Matched offline/online output-noise sweep, with one frozen signal scale."""
import argparse
import json
from pathlib import Path
import numpy as np
from . import task
from .runner import run, atlas_method
from .partial_experiment import SETTINGS, summarize
from .partial_selection import METHOD
from .config import NOISE_SIGMAS, PAPER_SEEDS
from .paths import NOISE_SWEEP_DIR, noise_tag

METHODS=tuple(atlas_method(k,d) for k,d in SETTINGS)+('contextual','contextual60','contextual90')
MAIN_METHODS=('atlas','atlas42','atlas_k12_d40','contextual','contextual60','contextual90')
HIGH_NOISE_METHODS=('atlas_k5_d40','atlas_k10_d28','atlas_k10_d42','atlas_k12_d40',
                    'contextual','contextual60','contextual90')


def methods_at(sigma):
    return HIGH_NOISE_METHODS if sigma>=.01 else METHODS


def aggregate(root):
    root=Path(root);manifest=json.loads((root/'sweep_protocol.json').read_text())
    rows=[];diagnostics=[]
    for sigma in manifest['sigmas']:
        folder=root/f'sigma_{noise_tag(sigma)}'
        if not (folder/'validation.json').exists():continue
        model=json.loads((folder/'protocol.json').read_text())['noise_model']
        rows.extend(dict(sigma=sigma,noise_std=model['std'],**r) for r in json.loads((folder/'report.json').read_text()))
        diagnostics.extend(dict(sigma=sigma,**r) for r in json.loads((folder/'selection_diagnostics.json').read_text()))
    task.write(root/'report.json',rows);task.write(root/'selection_diagnostics.json',diagnostics)
    lines=['# Consistent offline/online measurement noise','',
        'Noise model: y_meas,j = y_true,j + sigma * s_j * epsilon_j, with independent standard-normal epsilon. s_j is the population standard deviation of channel j in the clean offline recordings (global coordinates), fixed across levels and methods. Thus sigma is an amplitude ratio; noise/signal variance ratio is sigma² and SNR is -20 log10(sigma) dB. At zero sigma SNR is infinite.',
        '', 'Each level rebuilds the shared noisy offline dataset, speed clusters, chart bases and contextual normalization. Noise is added once per original sample before window extraction and frame alignment. Offline noise seed 3408 is independent of rollout seeds 511–530. The same standardized noise draws are scaled across levels. One offline-noise realization per level; 20 online rollouts per setting. Initial pose perturbations remain fixed even at sigma=0. No control weights, ranks, selector rule or regularization are tuned by noise level.',
        '', '| sigma | Method | Collision-free laps | Median successful RMSE (cm) | Median controller time (ms) |',
        '|---:|---|---:|---:|---:|']
    for r in rows:
        if r['method'] not in set(MAIN_METHODS+HIGH_NOISE_METHODS):continue
        error='—' if r['median_rmse_cm'] is None else f"{r['median_rmse_cm']:.2f}"
        timing=r['timing'];t='—' if timing is None else f"{timing['median_ms']:.2f}"
        lines.append(f"| {r['sigma']:.3g} | {r['label']} | {r['collision_free_laps']}/{r['evaluated']} | {error} | {t} |")
    lines+=['','The main comparison above keeps zero-success entries visible to avoid hiding noise-induced failures. Full K/rank results are retained in report.json; per-level K/rank tables omit zero-success rows. RMSE uses successful laps only. Timing pools all runs, including failed prefixes, and is not a matched-lap comparison when success counts differ.',
            '', 'At 1%, 2.5% and 5%, the scope is narrowed to four Atlas configurations that succeeded at some nonzero lower-noise level (K=5,d=40; K=10,d=28; K=10,d=42; K=12,d=40), plus all three contextual sizes. Settings not evaluated at a level are absent, not counted as failures. The 10% level was cancelled. Each retained level has its own noisy offline dataset, newly fitted clusters/bases and contextual normalization; lower-level performance only selects configurations to evaluate.',
            '', 'Past-image rank saturation can make partial residuals uninformative; selection_diagnostics.json records actual chart use and tie rates. These are exploratory seeds, not held-out validation. Previous clean-offline results are a different protocol and are not reused.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return rows


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);rows=json.loads((root/'report.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    levels=sorted({r['sigma'] for r in rows})
    for method in dict.fromkeys(MAIN_METHODS+HIGH_NOISE_METHODS):
        points=sorted((r for r in rows if r['method']==method),key=lambda r:r['sigma'])
        if not points:continue
        lookup={r['sigma']:r for r in points}
        x=[100*s for s in levels]
        label=points[0]['label']
        axes[0].plot(x,[100*lookup[s]['collision_free_laps']/lookup[s]['evaluated'] if s in lookup else np.nan for s in levels],'o-',label=label)
        axes[1].plot(x,[lookup[s]['median_rmse_cm'] if s in lookup and lookup[s]['median_rmse_cm'] is not None else np.nan for s in levels],'o-',label=label)
    axes[0].set(ylabel='Collision-free completed laps (%)',ylim=(-3,103))
    axes[1].set(ylabel='Median successful-lap RMSE (cm)')
    for ax in axes:
        ax.set_xscale('symlog',linthresh=.1)
        ticks=sorted({100*r['sigma'] for r in rows})
        ax.set_xticks(ticks,[f'{x:g}' for x in ticks])
        ax.set_xlabel('Noise/signal standard-deviation ratio (%)')
        ax.grid(alpha=.25);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle('Same output noise model offline and online; 20 rollouts per setting')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(root/f'noise_response.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def sweep(root=NOISE_SWEEP_DIR,sigmas=NOISE_SIGMAS):
    root=Path(root);sigmas=tuple(float(s) for s in sigmas)
    expected=dict(sigmas=list(sigmas),seeds=list(PAPER_SEEDS),
                  methods_by_sigma={noise_tag(s):list(methods_at(s)) for s in sigmas},
                  scope='Full retained grid below 1%; four previously successful Atlas settings plus contextual30/60/90 at 1%, 2.5%, 5%; 10% cancelled by user.',
                  noise_models=[task.noise_model(s) for s in sigmas])
    file=root/'sweep_protocol.json'
    if file.exists() and json.loads(file.read_text())!=expected:raise ValueError('Sweep protocol mismatch')
    task.write(file,expected)
    total=sum(len(methods_at(s))*len(PAPER_SEEDS) for s in sigmas)
    for index,sigma in enumerate(sigmas):
        task.write(root/'status.json',dict(status='running',sigma=sigma,completed_levels=index,
                   planned_levels=len(sigmas),planned_rollouts=total))
        folder=root/f'sigma_{noise_tag(sigma)}'
        status=folder/'status.json'
        if status.exists() and json.loads(status.read_text())['status']=='complete':
            saved=json.loads((folder/'protocol.json').read_text())
            if (saved['requested_methods']!=list(methods_at(sigma)) or saved['seeds']!=list(PAPER_SEEDS)
                    or saved['noise_model']!=task.noise_model(sigma)):
                raise ValueError('Completed level has a different experiment design')
        else:
            run(folder,PAPER_SEEDS,methods_at(sigma),METHOD,noise_sigma=sigma)
        summarize(folder)
        aggregate(root)
        print(f'Validated noise level sigma={sigma}',flush=True)
    plot(root)
    completed=sum(json.loads((root/f'sigma_{noise_tag(s)}'/'status.json').read_text())['completed'] for s in sigmas)
    task.write(root/'status.json',dict(status='complete',completed_levels=len(sigmas),
               planned_levels=len(sigmas),requested_rollouts=total,completed_rollouts=completed,
               skipped_rollouts=total-completed,cancelled_sigmas=[.1]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=NOISE_SWEEP_DIR)
    parser.add_argument('--sigmas',type=float,nargs='+',default=NOISE_SIGMAS)
    args=parser.parse_args();sweep(args.output,args.sigmas)
