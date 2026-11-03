"""Paper-style 4x2 rank-sweep comparison from verified saved results only."""
import hashlib
import json
import os
from pathlib import Path
from atlas_deepc.common.paths import experiment_path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/atlas_deepc_paper_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
SOURCES = {
    48: ROOT/'experiments/reacher/projection_chordal_rank_lamg0_lamdu0_clarabel_v1',
    64: ROOT/'experiments/reacher/four_baselines_k64_rank_projection_chordal_v1',
    80: ROOT/'experiments/reacher/large_k_rank_compact_v1/k080',
    96: ROOT/'experiments/reacher/large_k_rank_compact_v1/k096',
}
OUTPUT = ROOT/'output/pdf/reacher_rank_sweep_k48_k64_k80_k96'
STEM = 'figure5_rank_success_k48_k64_k80_k96'
COLORS = {'terminal': '#377eb4', 'graph': '#e67e17'}


def load_saved():
    panels, provenance = {}, {}
    for k, directory in SOURCES.items():
        manifest = json.loads((directory/'manifest.json').read_text())
        assert manifest['seeds'] == list(range(211, 311))
        assert manifest['ranks'] == list(range(28, 71))
        assert manifest['charts'] == k
        summary = json.loads((directory/'summary.json').read_text())
        assert summary['complete'] and not summary['missing_panels']
        record_name = 'rank_results.json' if (directory/'rank_results.json').exists() else 'summary.json'
        rows = json.loads((directory/record_name).read_text()) if record_name == 'rank_results.json' else summary['records']
        assert len(rows) == 172
        selected = json.loads((directory/'best_rank_table.json').read_text())
        provenance[k] = dict(source=str(directory.relative_to(ROOT)),
            selection_rule=selected['selection_rule'],
            sha256={name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                    for name in ('manifest.json',record_name,'best_rank_table.json')})
        for p in (8, 10):
            for mode in ('terminal', 'graph'):
                curve = sorted([r for r in rows if r['output_dimension']==p
                                and r['reference_mode']==mode], key=lambda r:r['rank'])
                assert [r['rank'] for r in curve] == list(range(28, 71))
                assert all(r['seeds']==100 and abs(r['success_rate']-r['successes'])<1e-8 for r in curve)
                for record in curve:
                    path = experiment_path(record['source_directory']) / 'summary.json'
                    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['summary_sha256']
                    panel = json.loads(path.read_text())
                    results = panel['results']
                    assert sorted(r['seed'] for r in results) == list(range(211, 311))
                    signature = panel['signature']
                    assert signature['solver'] == 'CLARABEL'
                    assert signature['chart_selection_method'] == 'full_window_projection'
                    failed = []
                    for r in results:
                        assert (r['clusters'], r['output_dimension'], r['svd_rank']) == (k, p, record['rank'])
                        assert r['lam_g'] == r['lam_du'] == 0 and r['lam_ini'] == 1e5
                        assert r['reference_mode'] == mode
                        assert r['graph_waypoints_enabled'] == (mode == 'graph')
                        ok = all(r[f'mean_last_{n}'] < .01 and r[f'max_last_{n}'] < .02 for n in (30, 100))
                        assert ok == bool(r['success_30'] and r['success_100'])
                        if not ok:
                            failed.append(r['seed'])
                    assert 100-len(failed) == record['successes']
                    assert sorted(failed) == sorted(record['failed_seeds'])
                    compact_path = path.parent / 'seed_metrics.json'
                    if compact_path.exists():
                        compact = json.loads(compact_path.read_text())
                        assert compact['summary_sha256'] == record['summary_sha256']
                        assert sorted(r['seed'] for r in compact['seeds']) == list(range(211, 311))
                        assert sorted(r['seed'] for r in compact['seeds'] if not r['success']) == sorted(failed)
                chosen = next(r for r in selected['columns']
                              if r['output_dimension']==p and r['reference_mode']==mode)
                point = next(r for r in curve if r['rank']==chosen['rank'])
                assert abs(point['success_rate']-chosen['success_rate'])<1e-8
                assert point['successes']==max(r['successes'] for r in curve)
                winner = min(curve, key=lambda r: (-r['successes'],r['median_rollout_rmse_mm'],r['rank']))
                assert winner['rank'] == chosen['rank']
                panels[k,p,mode] = curve, point
        print(f'Verified K={k}: 172 rank/configuration panels, 17,200 seed outcomes.', flush=True)
    return panels, provenance


def load_chart_counts(panels):
    """Combine the p=10 guided count search with all matching rank sweeps."""
    directory = ROOT/'experiments/reacher/chart_count_rank_projection_chordal_v1'
    manifest = json.loads((directory/'manifest.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    assert summary['complete'] and manifest['output_dimension'] == 10
    assert manifest['seeds'] == list(range(211, 311))
    rows_by_k = {}
    for record in summary['records']:
        path = experiment_path(record['source_directory'])/'summary.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['summary_sha256']
        panel = json.loads(path.read_text())
        results = panel['results']
        assert sorted(r['seed'] for r in results) == manifest['seeds']
        successes = 0
        for r in results:
            assert (r['clusters'], r['output_dimension'], r['svd_rank']) == (record['clusters'], 10, record['rank'])
            assert r['reference_mode'] == 'graph' and r['graph_waypoints_enabled']
            assert r['lam_g'] == r['lam_du'] == 0 and r['lam_ini'] == 1e5
            successes += all(r[f'mean_last_{n}'] < .01 and r[f'max_last_{n}'] < .02 for n in (30, 100))
        assert successes == record['successes'] == record['success_rate']
        rows_by_k.setdefault(record['clusters'], []).append(record)
    assert sorted(rows_by_k) == manifest['chart_counts']
    for k in SOURCES:
        rows_by_k.setdefault(k, []).extend(panels[k, 10, 'graph'][0])
    selected = []
    for k, rows in sorted(rows_by_k.items()):
        winner = min(rows, key=lambda r: (-r['successes'], r['median_rollout_rmse_mm'], r['rank']))
        selected.append(dict(charts=k, output_dimension=10, reference_mode='graph',
                             rank=winner['rank'], success_rate=winner['success_rate'],
                             tested_ranks=sorted({r['rank'] for r in rows})))
    for p, mode in ((8, 'terminal'), (8, 'graph'), (10, 'terminal')):
        for k in SOURCES:
            curve, winner = panels[k, p, mode]
            selected.append(dict(charts=k, output_dimension=p, reference_mode=mode,
                                 rank=winner['rank'], success_rate=winner['success_rate'],
                                 tested_ranks=[r['rank'] for r in curve]))
    provenance = dict(source=str(directory.relative_to(ROOT)),
                      sha256={name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                              for name in ('manifest.json', 'summary.json')})
    for point in selected:
        point['controlled_steps'] = 320
    completion = ROOT/'experiments/reacher/low_k_missing_batches_t300_v3/summary.json'
    if completion.exists():
        added = json.loads(completion.read_text())
        if added['complete']:
            protocol = json.loads((completion.parent/'manifest.json').read_text())
            assert protocol['seeds'] == list(range(211, 311))
            assert protocol['controlled_steps'] == 300
            for winner in added['selected']:
                k, p, mode = winner['clusters'], winner['output_dimension'], winner['reference_mode']
                selection_path = completion.parent/f'cases/k{k:03d}/output{p}/{mode}/selection.json'
                selection = json.loads(selection_path.read_text())
                assert selection['complete'] and selection['winner'] == winner
                assert selection['tested_ranks'] == list(range(28, 71))
                panel_path = selection_path.parent/f"rank{winner['rank']:03d}.json"
                panel = json.loads(panel_path.read_text())
                assert panel['complete'] and panel['processed_seeds'] == len(protocol['seeds'])
                assert panel['successes'] == winner['successes']
                selected = [r for r in selected if (r['charts'], r['output_dimension'], r['reference_mode']) != (k,p,mode)]
                selected.append(dict(charts=k, output_dimension=p, reference_mode=mode,
                    rank=winner['rank'], success_rate=winner['success_rate'],
                    tested_ranks=selection['tested_ranks'], controlled_steps=300))
            provenance['completion'] = dict(source=str(completion.relative_to(ROOT)),
                sha256=hashlib.sha256(completion.read_bytes()).hexdigest())
    print('Verified chart-count curves:', [(r['output_dimension'], r['reference_mode'], r['charts'],
                                           r['rank'], r['success_rate']) for r in selected], flush=True)
    return selected, provenance


def main(short=False, with_chart_counts=False):
    panels, provenance = load_saved()
    count_points, count_provenance = load_chart_counts(panels) if with_chart_counts else ([], {})
    output = OUTPUT / 'short_preview' if short else OUTPUT
    stem = STEM + '_short' if short else STEM
    if with_chart_counts:
        output = OUTPUT / 'combined_chart_counts'
        stem = 'figure6_rank_and_chart_count_success'
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix',
        'font.size':10,'font.weight':'normal',
        'axes.titlesize':11 if short else 12,'axes.titleweight':'normal',
        'axes.labelsize':11,'axes.labelweight':'normal','legend.fontsize':11,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    if with_chart_counts:
        fig = plt.figure(figsize=(7.2, 7.5))
        grid = fig.add_gridspec(6, 2, left=.10, right=.98, bottom=.075, top=.91,
                               wspace=.16, hspace=.40, height_ratios=[1, 1, 1, 1, .25, 1.30])
        import numpy as np
        axes = np.array([[fig.add_subplot(grid[i,j]) for j in range(2)] for i in range(4)])
        count_ax = fig.add_subplot(grid[5, :])
        fig.text(.022, .62, 'Success rate (%)', rotation=90, va='center', fontsize=11)
    else:
        fig, axes = plt.subplots(4,2,figsize=(7.2,5.6 if short else 8.4),sharex=True,sharey=True)
    if short and not with_chart_counts:
        fig.subplots_adjust(left=.10,right=.98,bottom=.08,top=.895,wspace=.16,hspace=.32)
        fig.supylabel('Success rate (%)', x=.022, fontsize=11)
    elif not with_chart_counts:
        fig.subplots_adjust(left=.10,right=.98,bottom=.065,top=.915,wspace=.16,hspace=.42)
    labels = [f'({c})' for c in 'abcdefgh']
    annotation_positions = {
        (48,8,'terminal'):(59,65), (48,8,'graph'):(52,101),
        (48,10,'terminal'):(61,59), (48,10,'graph'):(60,103),
        (64,8,'terminal'):(55,67), (64,8,'graph'):(53,104),
        (64,10,'terminal'):(61,58), (64,10,'graph'):(61,105),
        (80,8,'terminal'):(54,63), (80,8,'graph'):(49,105),
        (80,10,'terminal'):(61,62), (80,10,'graph'):(47,105),
        (96,8,'terminal'):(59,63), (96,8,'graph'):(55,105),
        (96,10,'terminal'):(57,60), (96,10,'graph'):(45,105),
    }
    if short:
        annotation_positions = {key: (value[0], 109 if key[2] == 'graph' else 43)
                                for key, value in annotation_positions.items()}
    for i,k in enumerate(SOURCES):
        for j,p in enumerate((8,10)):
            ax = axes[i,j]
            for mode in ('terminal','graph'):
                curve, best = panels[k,p,mode]
                color=COLORS[mode]
                ax.plot([r['rank'] for r in curve],[r['success_rate'] for r in curve],
                    color=color,ls='--' if mode=='terminal' else '-',lw=1.1,
                    marker='o',ms=2.4,zorder=3)
                ax.plot(best['rank'],best['success_rate'],'o',ms=6.4,mfc='white',
                    mec=color,mew=1.5,zorder=5)
                ax.annotate(f"{best['success_rate']:.0f}% (rank {best['rank']})",
                    xy=(best['rank'],best['success_rate']),
                    xytext=annotation_positions[k,p,mode],
                    ha='center',va='center',color=color,fontsize=9.5,fontweight='normal',
                    bbox=dict(facecolor='white',edgecolor='none',pad=.6,alpha=.94),
                    arrowprops=dict(arrowstyle='-',color=color,lw=.6,shrinkB=4),zorder=6)
            ax.set_title(f'{labels[2*i+j]} $K={k},\\ p={p}$',loc='left',pad=4 if short else 7)
            ax.set(xlim=(27.5,70.5),ylim=(-2,120 if short else 114),xticks=[28,35,42,49,56,63,70],
                   yticks=[0,50,100] if short else [0,20,40,60,80,100])
            ax.grid(color='#e4e8ed',lw=.5,zorder=0)
            ax.tick_params(length=3,width=.6)
            for spine in ('left','bottom'): ax.spines[spine].set_color('#777777')
            if j==0 and not short: ax.set_ylabel('Success rate (%)')
            if i==3: ax.set_xlabel('Chart rank')
            if with_chart_counts:
                ax.tick_params(labelbottom=i==3, labelleft=j==0)
    if with_chart_counts:
        counts = sorted({r['charts'] for r in count_points})
        count_styles = {
            (8, 'terminal'): ('#258774', '--', 's'),
            (8, 'graph'): ('#9467ad', '-', 's'),
            (10, 'terminal'): (COLORS['terminal'], '--', 'o'),
            (10, 'graph'): (COLORS['graph'], '-', 'o'),
        }
        for (p, mode), (color, linestyle, marker) in count_styles.items():
            points = sorted((r for r in count_points if r['output_dimension']==p
                             and r['reference_mode']==mode), key=lambda r:r['charts'])
            guidance = 'without guidance' if mode=='terminal' else 'with guidance'
            count_ax.plot([r['charts'] for r in points], [r['success_rate'] for r in points],
                          color=color, ls=linestyle, lw=1.25, marker=marker, ms=4,
                          mfc='white' if p==8 else color, mew=.9, zorder=4 if p==8 else 3,
                          label=rf'$p={p}$, {guidance}')
        count_ax.set_xscale('log', base=2)
        count_ax.set(xlim=(.88, 110), ylim=(-2, 105), yticks=[0, 50, 100],
                     ylabel='Success rate (%)', xlabel=r'Number of charts $K$ (log scale)')
        count_ax.set_xticks(counts, labels=[str(k) for k in counts])
        count_ax.minorticks_off()
        count_ax.set_title(r'(i) Success rate at the best tested rank for each $K$',
                           loc='left', pad=7, fontsize=11)
        count_ax.legend(loc='lower right', ncol=2, frameon=False, fontsize=8.5,
                        columnspacing=1.2, handlelength=2.3, handletextpad=.5,
                        borderaxespad=.7)
        horizon_note = ('Low-K additions: 300 steps; historical points: 320 steps.'
                        if 'completion' in count_provenance else
                        r'For $K<48$, only $p=10$ with guidance was tested.')
        count_ax.text(.025, .89, horizon_note,
                      transform=count_ax.transAxes, fontsize=8.5, color='#555555')
        count_ax.grid(color='#e4e8ed', lw=.5, zorder=0)
        count_ax.set_axisbelow(True)
        count_ax.tick_params(length=3, width=.6, labelsize=9)
        for spine in ('left', 'bottom'):
            count_ax.spines[spine].set_color('#777777')
    handles = [Line2D([],[],color=COLORS[m],ls='--' if m=='terminal' else '-',
                      marker='o',ms=3,lw=1.1) for m in ('terminal','graph')]
    fig.legend(handles,['Without guidance','With guidance'],ncol=2,
               loc='upper center',bbox_to_anchor=(.535,.985),frameon=False,columnspacing=2.8)
    fig.canvas.draw()
    for ax in fig.axes:
        bounds = ax.get_tightbbox(fig.canvas.get_renderer())
        assert bounds.x0 >= 0 and bounds.y0 >= 0
        assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height
    for suffix in ('png','svg','pdf'):
        fig.savefig(output/f'{stem}.{suffix}',dpi=250,facecolor='white')
    plt.close(fig)
    caption = ('Success rate versus chart rank for K=48, 64, 80, and 96 (top to bottom), '
        'with p=8 (left column) and p=10 (right column). Each point uses the same 100 '
        'development seeds (211–310). Highlighted markers indicate the selected ranks: '
        'highest strict success count, with ties broken by lowest median rollout RMSE, '
        'then lowest rank. The highest observed success rate is 96%, attained with p=10 '
        'and guidance at both K=64 and K=96. These are development-set rank-selection '
        'results, not held-out performance.')
    if with_chart_counts:
        caption = (r'Success rates on 100 development seeds (211--310). '
            r'(a)--(h) Rank sweeps for $K=48,64,80,96$ (top to bottom), '
            r'with $p=8$ (left) and $p=10$ (right); hollow markers identify selected ranks. '
            r'(i) Success rate at the best tested rank for each available chart count '
            r'and configuration; the $K$-axis is logarithmic. All four configurations '
            r'were tested at $K=48,64,80,96$; for smaller $K$, only $p=10$ with '
            r'graph guidance was tested. Curves connect available observations only. '
            r'Ranks are selected by success count, then median rollout RMSE, then rank. '
            r'The tested rank sets vary across chart counts.')
        if 'completion' in count_provenance:
            caption = (r'Success rates on 100 development seeds (211--310). '
                r'(a)--(h) Historical 320-step rank sweeps for $K=48,64,80,96$. '
                r'(i) Best tested ranks for each chart count and configuration. '
                r'New low-$K$ results use 300 controlled steps; unreplaced historical '
                r'points use 320 steps. These horizons must not be interpreted as identical. '
                r'New low-$K$ studies test ranks 28--70, retain only the winner success '
                r'rate, and break success ties by smaller rank. Historical studies use '
                r'their original rank searches and RMSE tie-breaks. All results are '
                r'exploratory, not held-out.')
        (output/'figure6_inclusion.tex').write_text(
            '\\begin{figure*}[t]\n    \\centering\n'
            f'    \\includegraphics[width=\\textwidth]{{{stem}.pdf}}\n'
            f'    \\caption{{{caption}}}\n'
            '    \\label{fig:rank-chart-count-ablation}\n\\end{figure*}\n')
    (output/'figure_metadata.json').write_text(json.dumps(dict(caption=caption,sources=provenance,
        figsize_inches=[7.2,7.5 if with_chart_counts else (5.6 if short else 8.4)],
        chart_count_source=count_provenance, chart_count_selected=count_points,
        selected=[dict(charts=k,p=p,guidance=m=='graph',rank=point['rank'],success_rate=point['success_rate'])
                  for (k,p,m),(_,point) in panels.items()]),indent=2))
    print(caption)
    print(output/f'{stem}.png')


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--short', action='store_true', help='Save a shorter 4x2 preview without replacing the original.')
    parser.add_argument('--with-chart-counts', action='store_true', help='Append the complete guided p=10 chart-count comparison.')
    args = parser.parse_args()
    main(short=args.short or args.with_chart_counts, with_chart_counts=args.with_chart_counts)
