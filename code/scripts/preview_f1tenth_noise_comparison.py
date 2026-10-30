"""Quick, reproducible preview of the retained F1TENTH comparison."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT / 'experiments/f1tenth/paper_nsr_20261027/report.json').read_text())
out = ROOT / 'output/figures/f1tenth_noise_comparison'
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 11,
    'mathtext.fontset': 'cm', 'axes.titlesize': 12, 'axes.labelsize': 11,
    'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#888888', 'axes.linewidth': .7,
})
levels = sorted({r['sigma'] for r in rows})
x = np.arange(len(levels))
fig, axes = plt.subplots(1, 3, figsize=(12, 3.75))
styles = [
    ('Atlas-DeePC', None, '#005099', 'o', '-'),
    ('Contextual-DPC (30)', 30, '#e68122', 's', '--'),
    ('Contextual-DPC (60)', 60, '#48958a', '^', '-.'),
    ('Contextual-DPC (90)', 90, '#aa6688', 'D', ':'),
]
for name, columns, color, marker, linestyle in styles:
    selected = [r for r in rows if
                (r['method'].startswith('atlas') if columns is None
                 else r['config'].get('columns') == columns)]
    lookup = {r['sigma']: r for r in selected}
    ordered = [lookup[s] for s in levels]
    values = [
        [100*r['collision_free_laps']/r['evaluated'] for r in ordered],
        [np.nan if r['median_rmse_cm'] is None else r['median_rmse_cm'] for r in ordered],
        [r['timing']['median_ms'] for r in ordered],
    ]
    for ax, y in zip(axes, values):
        ax.plot(x, y, label=name, color=color, marker=marker, linestyle=linestyle,
                linewidth=1.7, markersize=6, markerfacecolor='white', markeredgewidth=1.3)

for ax, title, ylabel in zip(axes,
        ['(a) Collision-free lap success', '(b) Tracking accuracy', '(c) Online computation'],
        ['Success rate (%)', 'Median cross-track RMSE (cm)', 'Median controller time (ms/step)']):
    ax.set_title(title, loc='left', pad=10)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('NSR (%)')
    ax.set_xticks(x, [f'{100*s:g}' for s in levels])
    ax.set_xlim(-.2, len(levels)-.8)
    ax.grid(True, color='#e2e6eb', linewidth=.65)
    ax.set_axisbelow(True)
axes[0].set_ylim(-4, 105)
axes[0].set_yticks([0, 25, 50, 75, 100])
axes[1].set_ylim(0, 12)
axes[1].set_yticks(np.arange(0, 13, 2))
axes[2].set_ylim(0, 16)
axes[2].set_yticks([0, 4, 8, 12, 16])
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
           bbox_to_anchor=(.5, 1.01), ncol=4, frameon=False, handlelength=2.8,
           columnspacing=1.8)
fig.subplots_adjust(left=.065, right=.99, bottom=.24, top=.78, wspace=.33)
fig.text(.5, .035,
         '20 test rollouts per setting; Atlas (K, d) selected separately at each noise level.\n'
         'RMSE: successful rollouts only; missing points indicate no successes. Time includes failed rollouts.',
         ha='center', va='bottom', fontsize=9, color='#444444')
fig.savefig(out / 'f1tenth_noise_comparison_preview.png', dpi=180, facecolor='white')
fig.savefig(out / 'f1tenth_noise_comparison_preview.svg', facecolor='white')
print(out / 'f1tenth_noise_comparison_preview.png')
