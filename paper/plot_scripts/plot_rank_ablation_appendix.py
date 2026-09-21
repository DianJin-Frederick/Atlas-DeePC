"""Plot the frozen lambda_g=.05, Clarabel development rank sweep (read-only)."""
from pathlib import Path
import json
import hashlib
import os

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/atlas_deepc_paper_mpl')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'experiments/routed_deepc_iid/linear_subspace_rank_100_v1'
OUT = ROOT / 'paper/figures/reacher_rank_ablation'
RANKS = np.arange(20, 81)
SEEDS = list(range(211, 311))


def main():
    summary = json.loads((SOURCE/'summary.json').read_text())
    assert summary['complete'] and summary['ranks'] == RANKS.tolist()
    assert summary['seeds'] == SEEDS
    records, source_hashes = [], {}
    targets = None
    for rank in RANKS:
        path = SOURCE / f'rank{rank:03}/summary.json'
        source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        panel = json.loads(path.read_text())
        rows = sorted(panel['results'], key=lambda row: row['seed'])
        assert [r['seed'] for r in rows] == SEEDS
        exact_targets = np.array([r['exact_target_xy'] for r in rows])
        if targets is None:
            targets = exact_targets
        else:
            np.testing.assert_array_equal(targets, exact_targets)
        for r in rows:
            assert r['steps'] == 320 and r['svd_rank'] == rank
            assert r['lam_g'] == .05 and r['matrix_geometry'] == 'linear'
            assert r['reference_mode'] == 'graph' and r['candidate_count'] == 10
            assert sum(r['statuses'].values()) == 320
            assert r['unsolved_steps'] == sum(v for k, v in r['statuses'].items()
                                             if k not in ('optimal','optimal_inaccurate'))
            for w in (30, 100):
                assert r[f'success_{w}'] == (r[f'mean_last_{w}'] < .01 and r[f'max_last_{w}'] < .02)
        errors = 1000*np.array([r['mean_last_100'] for r in rows])
        count = sum(r['success_30'] and r['success_100'] for r in rows)
        unsolved = sum(r['unsolved_steps'] for r in rows)
        assert count == panel['successes'] and unsolved == panel['unsolved_qps']
        np.testing.assert_allclose(np.median(errors), panel['median_tail100_mm'])
        np.testing.assert_allclose(np.quantile(errors,.9), panel['p90_tail100_mm'])
        records.append(dict(rank=int(rank), success_rate_pct=count,
            median_error_mm=float(np.median(errors)), p90_error_mm=float(np.quantile(errors,.9)),
            unsolved_qps=unsolved, unsolved_qp_pct=100*unsolved/32000))

    font = font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],
        'mathtext.fontset':'stix','font.size':9,'axes.labelsize':9,'axes.titlesize':10,
        'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
        'text.color':'#28343D','axes.labelcolor':'#28343D','axes.edgecolor':'#666666',
        'axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42})
    fig, axes = plt.subplots(1, 3, figsize=(6, 1.5))
    fig.subplots_adjust(left=.08, right=.985, bottom=.28, top=.84, wspace=.55)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(color='#e9e9e9', lw=.5)
        ax.set_axisbelow(True)
        ax.set(xlim=(19,81), xticks=[20,40,60,80], xlabel='Rank')
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 3
        ax.tick_params(length=3, width=.6)
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))

    ax = axes[0]
    success = np.array([r['success_rate_pct'] for r in records])
    ax.plot(RANKS, success, color='#59636B', lw=1, marker='.', ms=2.6, zorder=2)
    ax.set(ylim=(-3,106), yticks=[0,50,100], ylabel='Success rate (%)')
    ax.set_title('(a) Control success', loc='left', pad=5)

    ax = axes[1]
    ax.plot(RANKS, [r['median_error_mm'] for r in records], color='#0077BB',
            lw=1.15, marker='.', ms=2, label='Median')
    ax.plot(RANKS, [r['p90_error_mm'] for r in records], color='#DF7B20',
            lw=1.15, ls='--', marker='.', ms=2, label='90th pct.')
    ax.axhline(10, color='#8f969c', lw=.8, ls=':', zorder=1)
    ax.set(yscale='log', ylim=(1,350), ylabel='Error (mm)')
    ax.set_yticks([1,10,100], labels=['1','10','100'])
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_title('(b) Tracking error', loc='left', pad=5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.04,.99), frameon=False,
              borderpad=.05, labelspacing=.3, handlelength=1.6, handletextpad=.4,
              facecolor='white')

    ax = axes[2]
    failure = np.array([r['unsolved_qp_pct'] for r in records])
    ax.fill_between(RANKS, failure, 0, color='#009988', alpha=.13)
    ax.plot(RANKS, failure, color='#009988', lw=1.2, marker='.', ms=2.6)
    ax.set(ylim=(-.7,25), yticks=[0,10,20], ylabel='Unsolved QPs (%)')
    ax.set_title('(c) QP failures', loc='left', pad=5)
    ax.text(.97,.70,'No unsolved QPs\nat ranks 34 to 80',transform=ax.transAxes,
            ha='right',va='top',fontsize=9,color='#52606A')
    assert all(r['unsolved_qps']==0 for r in records if r['rank']>=34)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'rank_ablation_20_80.{ext}',dpi=320,facecolor='white')
    plt.close(fig)
    (OUT/'plot_metadata.json').write_text(json.dumps(dict(
        source=str(SOURCE),sources_sha256=source_hashes, ranks=RANKS.tolist(),
        seed_range=[211,310],rollouts=6100,steps_per_rollout=320,
        font_path=font,figure_inches=[6,1.5],font_label_tick_pt=9,
        success_definition='Both final-30 and final-100 mean error <10mm and maximum error <20mm',
        error_definition='Across-rollout median and 90th percentile of final-100 mean error, including failures',
        failure_definition='Steps with no QP solution / all 32000 control steps per rank; optimal_inaccurate accepted; zero-input fallback retained',
        highlighted_ranks=[],summary=records),indent=2)+'\n')
    (OUT/'figure_inclusion.tex').write_text(r'''\begin{figure}[tbp]
    \centering
    \includegraphics[width=\linewidth]{figures/reacher_rank_ablation/rank_ablation_20_80.pdf}
    \caption{Rank ablation on the same 100 development seeds (211--310). (a) Success rate. (b) Median and 90th percentile of each rollout's final-100 mean fingertip error, including failed rollouts; the dotted line indicates 10\,mm. (c) Percentage of control steps with no QP solution, out of 32,000 steps per rank. Such steps apply zero input and remain included in the evaluation.}
    \label{fig:rank-ablation}
\end{figure}
''')
    print('Validated 61 ranks x 100 paired seeds = 6100 complete rollouts.')
    print(OUT/'rank_ablation_20_80.pdf')


if __name__ == '__main__':
    main()
