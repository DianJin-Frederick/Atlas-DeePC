"""Four-panel overlay preview; preserves the existing eight-panel figure."""
import json
from rank_sweep_k48_k64 import ROOT, SOURCES, load_saved, plt, Line2D

OUTPUT = ROOT / 'output/pdf/reacher_rank_sweep_compact_preview'
STEM = 'figure5_rank_success_four_panels'
STYLES = {
    48: ('#377eb4', '-', 'o'),
    64: ('#e67e17', '--', 's'),
    80: ('#258774', '-.', '^'),
    96: ('#9467ad', ':', 'D'),
}


def main():
    panels, provenance = load_saved()
    plt.rcParams.update({
        'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
        'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 11,
        'legend.fontsize': 10.5, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .6,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.085, right=.985, bottom=.115, top=.84,
                        wspace=.16, hspace=.31)
    selected = []
    for i, mode in enumerate(('terminal', 'graph')):
        for j, p in enumerate((8, 10)):
            ax = axes[i, j]
            for k, (color, ls, marker) in STYLES.items():
                curve, best = panels[k, p, mode]
                ranks = [r['rank'] for r in curve]
                rates = [r['success_rate'] for r in curve]
                ax.plot(ranks, rates, color=color, ls=ls, lw=1.05,
                        marker=marker, markevery=4, ms=2.4, mew=.45, zorder=3)
                ax.plot(best['rank'], best['success_rate'], marker=marker,
                        color=color, ls='', ms=6, mfc='white', mew=1.25, zorder=5)
                selected.append(dict(charts=k, p=p, guidance=mode == 'graph',
                                     rank=best['rank'], success_rate=best['success_rate']))
            guidance = 'Without guidance' if mode == 'terminal' else 'With guidance'
            ax.set_title(f'({chr(97+2*i+j)}) $p={p}$, {guidance.lower()}', loc='left', pad=6)
            ax.set(xlim=(27.5, 70.5), ylim=(-2, 105),
                   xticks=[28, 35, 42, 49, 56, 63, 70], yticks=[0, 20, 40, 60, 80, 100])
            ax.grid(color='#e4e8ed', lw=.5)
            ax.set_axisbelow(True)
            ax.tick_params(length=3, width=.6)
            for spine in ('left', 'bottom'):
                ax.spines[spine].set_color('#777777')
            if j == 0:
                ax.set_ylabel('Success rate (%)')
            if i == 1:
                ax.set_xlabel('Chart rank')
    handles = [Line2D([], [], color=color, ls=ls, marker=marker,
                      lw=1.1, ms=3.2, label=rf'$K={k}$')
               for k, (color, ls, marker) in STYLES.items()]
    fig.legend(handles=handles, ncol=4, loc='upper center',
               bbox_to_anchor=(.535, .99), frameon=False, columnspacing=1.6,
               handlelength=2.3)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes.flat:
        b = ax.get_tightbbox(renderer)
        assert b.x0 >= 0 and b.y0 >= 0 and b.x1 <= fig.bbox.width and b.y1 <= fig.bbox.height
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(OUTPUT / f'{STEM}.{ext}', dpi=250, facecolor='white')
    plt.close(fig)
    (OUTPUT / 'figure_metadata.json').write_text(json.dumps(dict(
        sources=provenance, selected=selected, figsize_inches=[7.2, 4.4],
        notes='All tested ranks plotted without smoothing. Large hollow markers identify selected ranks. '
              'Small markers are spaced every four ranks for readability. '
              'Development seeds 211--310, not held-out evaluation.'), indent=2) + '\n')
    print(OUTPUT / f'{STEM}.pdf')


if __name__ == '__main__':
    main()
