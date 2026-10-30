"""Verify and summarize paired fixed-atlas guidance runs using final-30 errors."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments/reacher"
CASES = {
    "without": EXPERIMENTS / "four_baselines_500_mixed_k_t300_v1/output8/terminal",
    "with": EXPERIMENTS / "matched_guidance_500_p8_k64_r53_t300_v1/output8/graph",
}
OUT = ROOT / "output/tables/reacher_matched_guidance_500"
SEEDS = list(range(10001, 10501))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifests = {name: json.loads((folder / "manifest.json").read_text())
                 for name, folder in CASES.items()}
    for field in ("config", "atlas_sha256", "data_sha256", "solver", "solver_options",
                  "package_versions", "warmup_steps", "chart_selection_method"):
        assert manifests['without'][field] == manifests['with'][field], field
    assert manifests['with']['paired_manifest_sha256'] == digest(CASES['without'] / 'manifest.json')
    for m in manifests.values():
        assert m['evaluation_seeds'] == SEEDS
        assert m['output_dimension'] == 8
        assert m['config']['clusters'] == 64 and m['config']['svd_rank'] == 53
        assert m['config']['steps'] == 300

    summaries, records, paired_inputs = {}, {}, {}
    for name, folder in CASES.items():
        files = sorted(folder.glob('seed*_control_results.npz'))
        assert [int(p.name.split('_')[0][4:]) for p in files] == SEEDS
        signature = digest(folder / 'manifest.json')
        rows, timings = [], []
        for seed, path in zip(SEEDS, files):
            with np.load(path, allow_pickle=True) as d:
                meta = json.loads(d['summary'].item())
                assert meta['evaluation_signature_sha256'] == signature
                assert (meta['seed'], meta['output_dimension'], meta['clusters'], meta['svd_rank'], meta['steps']) == (seed, 8, 64, 53, 300)
                assert meta['reference_mode'] == ('terminal' if name == 'without' else 'graph')
                error, timing = d['distance'], d['control_compute_time_s']
                assert error.shape == timing.shape == (300,)
                assert np.isfinite(error).all() and np.isfinite(timing).all()
                assert (error >= 0).all() and (timing >= 0).all()
                target = d['final_goal'].reshape(-1)[6:8]
                np.testing.assert_allclose(error, np.linalg.norm(d['y'][:, 6:8] - target, axis=1))
                initial = d['selection_windows'][0]
                if name == 'without':
                    paired_inputs[seed] = (initial.copy(), target.copy())
                else:
                    np.testing.assert_array_equal(initial, paired_inputs[seed][0])
                    np.testing.assert_array_equal(target, paired_inputs[seed][1])
                success = all(error[-n:].mean() < .01 and error[-n:].max() < .02 for n in (30, 100))
                rows.append(dict(seed=seed, success=bool(success), final30_mm=float(error[-30:].mean()*1000),
                                 rollout_rmse_mm=float(np.sqrt(np.mean(error**2))*1000), sha256=digest(path)))
                timings.extend(timing[5:] * 1000)
        errors = [row['final30_mm'] for row in rows]
        summaries[name] = dict(rollouts=len(rows), successes=sum(row['success'] for row in rows),
            success_percent=100*np.mean([row['success'] for row in rows]),
            error_quantiles_mm={str(q):float(np.percentile(errors,q)) for q in (50,90,95,99)},
            median_rollout_rmse_mm=float(np.median([row['rollout_rmse_mm'] for row in rows])),
            computation_quantiles_ms={str(q):float(np.percentile(timings,q)) for q in (50,95,99)},
            timed_steps=len(timings), manifest_sha256=signature, source=str(folder))
        records[name] = rows
    outcome_names={(True,True):'both_success', (False,True):'recovered',
                   (True,False):'regressed', (False,False):'both_failed'}
    groups={name:[] for name in outcome_names.values()}
    for left,right in zip(records['without'],records['with']):
        assert left['seed']==right['seed']
        groups[outcome_names[left['success'],right['success']]].append(left['seed'])
    counts={key:len(value) for key,value in groups.items()}
    saved=json.loads((CASES['with'].parents[1]/'paired_summary.json').read_text())
    assert saved['complete'] and saved['completed_pairs']==500
    assert counts == {('recovered' if k=='repaired' else k):v for k,v in saved['paired_counts'].items()}
    result=dict(parameters=dict(p=8,K=64,d=53,steps=300), seeds=SEEDS,
        error_definition='Per-rollout mean fingertip distance over the final 30 steps; quantiles across all 500 rollouts, including failures.',
        success_definition='Mean <10 mm and maximum <20 mm over both final-30 and final-100 windows.',
        timing_definition='Total online computation pooled across steps 6--300; separate sessions.',
        caveat=manifests['with']['caveat'], summaries=summaries, paired_counts=counts, paired_seeds=groups,
        exact_mcnemar_pvalue=float(binomtest(counts['recovered'],counts['recovered']+counts['regressed'],.5).pvalue),
        per_seed=records)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'statistics.json').write_text(json.dumps(result,indent=2)+'\n')

    def cells(a,b,maximize=False):
        values=[a,b];best=(max if maximize else min)(values)
        return ' & '.join(r'\textbf{'+v+'}' if x==best else v
                         for x,v in zip(values,[f'{a:.2f}',f'{b:.2f}']))
    a,b=summaries['without'],summaries['with']
    lines=[r'% Requires booktabs and xcolor.', r'\begin{table}[t]',r'    \centering',r'    \begingroup',
           r'    \footnotesize',r'    \setlength{\tabcolsep}{5pt}',r'    \setlength{\fboxsep}{5pt}',
           r'    \colorbox[gray]{0.96}{%',r'    \begin{tabular}{@{}lrr@{}}',r'        \toprule',
           r'        Graph guidance & No & Yes \\',r'        \midrule',
           f"        Successes / 500 & {a['successes']} & \\textbf{{{b['successes']}}} \\",
           f"        Success rate & {a['success_percent']:.1f}\\% & \\textbf{{{b['success_percent']:.1f}\\%}} \\"]
    for q,label in [(50,'Median error (mm)'),(90,'90th-percentile error (mm)'),(95,'95th-percentile error (mm)'),(99,'99th-percentile error (mm)')]:
        lines.append(f"        {label} & "+cells(a['error_quantiles_mm'][str(q)],b['error_quantiles_mm'][str(q)])+r' \\')
    lines.append('        Median rollout RMSE (mm) & '+cells(a['median_rollout_rmse_mm'],b['median_rollout_rmse_mm'])+r' \\')
    lines.extend([r'        \midrule',r'        \multicolumn{3}{@{}l}{\textit{Total online computation (ms per step)}} \\'])
    for q,label in [(50,'Median'),(95,'95th percentile'),(99,'99th percentile')]:
        lines.append(f"        {label} & "+cells(a['computation_quantiles_ms'][str(q)],b['computation_quantiles_ms'][str(q)])+r' \\')
    lines.extend([r'        \bottomrule',r'    \end{tabular}%',r'    }',r'    \endgroup',
        r'    \caption{Effect of graph guidance on 500 paired Reacher rollouts',
        r'    (seeds 10001--10500), with $p=8$, $K=64$, $d=53$, and $T_{\mathrm{sim}}=300$.',
        r'    Both configurations use the same atlas, controller parameters, initial states, and targets.',
        r'    Error statistics summarize final-30 mean fingertip errors across all rollouts;',
        r'    rollout RMSE uses all 300 steps. Timings pool steps 6--300 and were measured in separate sessions.',
        r'    Guidance includes terminal-reference selection and graph routing.}',
        r'    \label{tab:reacher-matched-guidance}',r'\end{table}'])
    # f-strings above end with a single escaped slash; enforce TeX row terminators.
    lines=[line+'\\' if line.endswith('\\') and not line.endswith('\\\\') else line for line in lines]
    (OUT/'matched_guidance_table.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(summaries=summaries,paired_counts=counts,exact_mcnemar_pvalue=result['exact_mcnemar_pvalue']),indent=2))
    print(OUT/'matched_guidance_table.tex')


if __name__=='__main__':
    main()
