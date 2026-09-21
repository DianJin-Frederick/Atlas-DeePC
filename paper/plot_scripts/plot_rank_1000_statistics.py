"""Paper-sized figures from saved matched seed panels; no experiments are run."""
from pathlib import Path
import hashlib
import json
import os
import argparse

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/atlas_deepc_paper_mpl')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / 'experiments/routed_deepc_iid'
SELECTED = EXPERIMENTS / 'selected_ranks_1000_clarabel_v1'
BASELINE = EXPERIMENTS / 'clarabel_lamg005_panel_311_1310_v1'
OUT = ROOT / 'paper/figures/reacher_rank_1000'
RANKS = [59, 60, 62, 64]
SEEDS = list(range(311, 1311))
COLORS = {59: '#DF7B20', 60: '#28343D', 62: '#0077BB', 64: '#009988'}
MARKERS = {59: 'o', 60: 's', 62: 'D', 64: '^'}
LINES = {59: '-', 60: '--', 62: '-.', 64: ':'}
SOURCES = {}


def read(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def wilson(k, n=1000):
    z = 1.959963984540054
    p = k / n
    c = (p + z*z/(2*n)) / (1 + z*z/n)
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return 100*np.array([c-h, c+h])


def style(ax, axis='both'):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis=axis, color='#e9e9e9', lw=.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.65)


def save(fig, name):
    # No tight bounding box: retain exact physical dimensions and font sizes.
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=320, facecolor='white')
    plt.close(fig)


def main(timing_only=False):
    selected_manifest = read(SELECTED / 'manifest.json')
    baseline_manifest = read(BASELINE / 'manifest.json')
    for key in ('data_sha256', 'atlas_sha256', 'solver', 'matrix_geometry'):
        assert selected_manifest[key] == baseline_manifest[key], key
    for key, value in baseline_manifest['config'].items():
        assert selected_manifest['base_config'][key] == value, key
    rows, timing, errors, statistics = {}, {}, {}, {}
    for rank in RANKS:
        path = BASELINE / 'graph/summary.json' if rank == 60 else SELECTED / f'rank{rank:03}/summary.json'
        report = read(path)
        rows[rank] = sorted(report['results'], key=lambda r: r['seed'])
        assert [r['seed'] for r in rows[rank]] == SEEDS
        for row in rows[rank]:
            assert row['steps'] == 320 and row['svd_rank'] == rank
            assert row['lam_g'] == .05 and row['reference_mode'] == 'graph'
            assert row['matrix_geometry'] == 'linear' and row['candidate_count'] == 10
            assert np.isfinite(row['mean_last_100']) and row['mean_last_100'] > 0
            for window in (30, 100):
                assert bool(row[f'success_{window}']) == (
                    row[f'mean_last_{window}'] < .01 and row[f'max_last_{window}'] < .02)
        timing[rank] = read(BASELINE / 'graph/timing_summary.json') if rank == 60 else report['timing']
        assert timing[rank]['timed_steps'] == 315000
        assert timing[rank]['discard_initial_steps_per_rollout'] == 5
        modules = {r['module']: r for r in timing[rank]['modules']}
        np.testing.assert_allclose(sum(r['mean_ms'] for k, r in modules.items()
                                      if k != 'Total online control'), modules['Total online control']['mean_ms'])
        errors[rank] = 1000*np.array([r['mean_last_100'] for r in rows[rank]])
        count = sum(r['success_30'] and r['success_100'] for r in rows[rank])
        statistics[rank] = dict(successes=count, success_rate_pct=count/10,
            wilson_95_pct=wilson(count).tolist(), median_error_mm=float(np.median(errors[rank])),
            p90_error_mm=float(np.quantile(errors[rank], .9)),
            max_error_mm=float(errors[rank].max()),
            unsolved_qps=sum(r['unsolved_steps'] for r in rows[rank]),
            timing_modules=timing[rank]['modules'],
            one_time_goal_setup_ms=timing[rank]['one_time_goal_setup_ms'])
    for rank in RANKS[1:]:
        np.testing.assert_array_equal([r['exact_target_xy'] for r in rows[59]],
                                      [r['exact_target_xy'] for r in rows[rank]])

    font = font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix', 'font.size': 10, 'axes.labelsize': 10,
        'axes.titlesize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 9, 'axes.edgecolor': '#606060', 'axes.linewidth': .65,
        'text.color': '#28343D', 'axes.labelcolor': '#28343D',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'lines.solid_capstyle': 'round'})
    OUT.mkdir(parents=True, exist_ok=True)

    # Half-column point-and-interval plot: no truncated bars.
    fig, ax = plt.subplots(figsize=(3, 2.35))
    fig.subplots_adjust(left=.18, right=.97, bottom=.21, top=.94)
    style(ax, 'y')
    for i, rank in enumerate(RANKS):
        s = statistics[rank]
        rate = s['success_rate_pct']
        lo, hi = s['wilson_95_pct']
        ax.errorbar(i, rate, yerr=[[rate-lo], [hi-rate]], color=COLORS[rank],
                    marker=MARKERS[rank], ms=5, lw=1.25, capsize=4, capthick=1)
        ax.text(i, hi+.45, f'{rate:.1f}%', ha='center', va='bottom', fontsize=9,
                color=COLORS[rank])
    ax.set(xlim=(-.48, 3.48), ylim=(89, 100), xticks=range(4),
           xticklabels=RANKS, yticks=[90, 92, 94, 96, 98, 100],
           xlabel='Representation rank', ylabel='Success rate (%)')
    ax.text(.97, .04, '95% Wilson intervals', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, color='#606060')
    if not timing_only:
        save(fig, 'rank_success_rate')
    else:
        plt.close(fig)

    # Empirical CDF: every rollout retained, including failures, with no smoothing.
    fig, ax = plt.subplots(figsize=(3, 2.35))
    fig.subplots_adjust(left=.18, right=.97, bottom=.21, top=.94)
    style(ax)
    for rank in RANKS:
        x = np.sort(errors[rank])
        ax.step(np.r_[.004, x, 100], np.r_[0, np.arange(1, 1001)/10, 100],
                where='post', color=COLORS[rank], ls=LINES[rank], lw=1.2,
                label=f'Rank {rank}', zorder=4 if rank == 59 else 3)
    ax.axvline(10, color='#989898', lw=.8, ls=(0, (3, 3)), zorder=1)
    ax.text(11.5, 37, '10 mm', fontsize=9, color='#666666', rotation=90, va='center')
    ax.set(xscale='log', xlim=(.004, 100), ylim=(0, 102), yticks=[0,25,50,75,100],
           xlabel='Final-100 mean error (mm)', ylabel='Cumulative rollouts (%)')
    ax.set_xticks([.01, .1, 1, 10, 100], labels=['0.01', '0.1', '1', '10', '100'])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.legend(loc='upper left', frameon=False, handlelength=2.2, labelspacing=.25,
              borderpad=.1, handletextpad=.5)
    if not timing_only:
        save(fig, 'rank_tracking_error')
    else:
        plt.close(fig)

    # Two linear scales separate the dominant QP cost from the small modules.
    modules = {r: {v['module']: v['mean_ms'] for v in timing[r]['modules']} for r in RANKS}
    stages = [
        ('QP construction and solve', 'QP construction + solve', '#DF7B20'),
        ('Chart localization', 'Chart localization', '#0077BB'),
        ('Terminal-reference selection', 'Terminal-reference selection', '#009988'),
        ('Graph routing', 'Graph routing', '#66CCEE'),
        ('Linear subspace retrieval', 'Subspace retrieval (<1 '+r'$\mu$s'+')', '#28343D'),
        ('Other control-loop work', 'Other online work', '#B6BCC2')]
    fig, (total_ax, detail_ax) = plt.subplots(1, 2, figsize=(6, 3.05))
    fig.subplots_adjust(left=.09, right=.98, bottom=.245, top=.755, wspace=.37)
    for ax in (total_ax, detail_ax):
        style(ax, 'y')
        ax.set(xticks=range(4), xticklabels=RANKS, xlabel='Representation rank',
               xlim=(-.6, 3.6))
    total_bottom = np.zeros(4)
    detail_bottom = np.zeros(4)
    for key, label, color in stages:
        vals = np.array([modules[r][key] for r in RANKS])
        total_ax.bar(range(4), vals, bottom=total_bottom, color=color, width=.57,
                     linewidth=0, zorder=3)
        total_bottom += vals
        if key != 'QP construction and solve':
            detail_ax.bar(range(4), 1000*vals, bottom=detail_bottom, color=color,
                          width=.57, linewidth=0, zorder=3)
            detail_bottom += 1000*vals
    for i, (total, detail) in enumerate(zip(total_bottom, detail_bottom)):
        total_ax.text(i, total+.65, f'{total:.2f}', ha='center', va='bottom', fontsize=9)
        detail_ax.text(i, detail+2.4, f'{detail:.1f}', ha='center', va='bottom', fontsize=9)
    total_ax.set(ylim=(0, 41), yticks=[0,10,20,30,40], ylabel='Mean time (ms)')
    detail_ax.set(ylim=(0, 155), yticks=[0,50,100,150], ylabel=r'Mean time ($\mu$s)')
    total_ax.set_title('(a) Total online computation', loc='left', pad=8)
    detail_ax.set_title('(b) Non-QP module breakdown', loc='left', pad=8)
    handles = [Patch(facecolor=color, label=label) for _, label, color in stages]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(.51, 1.005), columnspacing=1.2, handletextpad=.45,
               handlelength=1.1, labelspacing=.5)
    fig.text(.5, .065, 'QP construction and solving account for 99.6-99.7% of total online time.',
             ha='center', va='bottom', fontsize=9, color='#606060')
    fig.text(.5, .012, 'Timings span different execution sessions; rank effects are not isolated.',
             ha='center', va='bottom', fontsize=9, color='#606060')
    save(fig, 'rank_module_timing')

    metadata = dict(seeds=[311,1310], rollouts_per_rank=1000, ranks=RANKS,
        font='Times New Roman', font_path=font, label_font_pt=10, tick_legend_font_pt=9,
        figure_inches={'rank_success_rate':[3,2.35], 'rank_tracking_error':[3,2.35],
                       'rank_module_timing':[6,3.05]},
        timing_visualization='Stacked module means: total in ms and non-QP detail in microseconds, both linear zero-based axes; colors encode modules',
        success_definition='Both final-30 and final-100 mean error <10 mm and maximum error <20 mm',
        tracking_error_definition='Empirical distribution of each rollout final-100 mean fingertip error; all rollouts included',
        timing_definition='Module means over 315000 steps per rank, discarding first 5 steps per rollout; QP includes construction and solve; offline and one-time setup excluded',
        timing_caveat='Historical rank-60 timings and selected-rank timings span sessions on a shared computer; differences cannot be attributed solely to rank',
        sources_sha256=SOURCES, statistics=statistics)
    (OUT/'plot_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    captions = r'''% The first two PDFs have 10-pt labels and 9-pt ticks at 3-inch width.
% Include at approximately half of the 6-inch text width; do not shrink further.
\begin{figure}[t]
    \centering
    \includegraphics[width=0.49\linewidth]{figures/reacher_rank_1000/rank_success_rate.pdf}\hfill
    \includegraphics[width=0.49\linewidth]{figures/reacher_rank_1000/rank_tracking_error.pdf}
    \caption{Rank sensitivity over the same 1,000 seeds (311--1310). Left: success rates with pointwise 95\% Wilson intervals. Success requires mean error below 10\,mm and maximum error below 20\,mm in both the final-30 and final-100 windows. Right: empirical distributions of final-100 mean fingertip error, including failed rollouts. The 10\,mm line marks only the mean-error component of the success criterion.}
    \label{fig:rank-performance}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/reacher_rank_1000/rank_module_timing.pdf}
    \caption{Mean online computation time: (a) total module costs in milliseconds and (b) the non-QP components on an expanded microsecond scale. Each rank includes 315,000 timed steps after excluding the first five steps of each rollout. QP time includes construction and solving; offline computation and one-time reference setup are excluded. Measurements span different sessions on a shared computer and do not isolate the effect of rank on runtime.}
    \label{fig:rank-timing}
\end{figure}
'''
    (OUT/'figure_inclusion.tex').write_text(captions)
    print(json.dumps({r: {k:v for k,v in s.items() if k not in ('timing_modules','one_time_goal_setup_ms')}
                      for r,s in statistics.items()}, indent=2))
    print('Figures:', OUT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timing-only', action='store_true')
    main(timing_only=parser.parse_args().timing_only)
