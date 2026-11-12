"""Matched 20-seed trial of past-history projection, with fixed atlas/QP."""
import json
import argparse
from pathlib import Path
import numpy as np
from . import task
from .runner import run, atlas_method, CONFIGS
from .reporting import report
from .validation import validate
from .partial_selection import METHOD, PastWindowProjectionSelector
from .config import PAPER_SEEDS
from .paths import RESULTS_DIR, PAPER_DIR

SETTINGS = [(k,d) for k in (2,5,10,12) for d in (24,26,28)] + [
    (2,40),(2,42),(5,40),(10,42),(12,40),(1,40)]


def summarize(output):
    output=Path(output)
    audit=validate(output)
    panel=report(output)
    seeds=json.loads((output/'protocol.json').read_text())['seeds']
    diagnostics=[]
    for s in panel:
        if not s['method'].startswith('atlas'):continue
        method=s['method'];cfg=CONFIGS[method]
        bases=np.load(output/f'atlas_k{cfg.clusters}.npz')['bases'][:,:,:cfg.rank]
        selection=PastWindowProjectionSelector(bases)
        counts=np.zeros(cfg.clusters,dtype=int);tied=0;calls=0;switches=0;transitions=0;maximum=0.
        for seed in seeds:
            folder=output/method/f'seed{seed}'
            trace=np.load(folder/'rollout.npz')
            saved=json.loads((folder/'calls.json').read_text())
            charts=np.array([c['chart'] for c in saved])
            counts+=np.bincount(charts,minlength=cfg.clusters)
            tied+=int(np.sum(trace['selection_tie_count']>1));calls+=len(charts)
            switches+=int(np.sum(charts[1:]!=charts[:-1]));transitions+=max(0,len(charts)-1)
            maximum=max(maximum,float(np.max(trace['selection_residuals'])))
            assert np.isfinite(trace['X']).all() and np.isfinite(trace['U']).all()
            if trace['U'].size:assert np.max(np.abs(trace['U'])-task.LIMIT)<1e-9
            for c in saved:
                if c['accepted']:assert c['constraint_violation']<1e-5
        diagnostics.append(dict(method=method,config=s['config'],past_ranks=selection.ranks.tolist(),
            chart_counts=counts.tolist(),tie_fraction=tied/calls,switch_fraction=switches/transitions if transitions else 0,
            maximum_residual=maximum,
            partial_history_success=s['collision_free_laps'],median_rmse_cm=s['median_rmse_cm']))
    task.write(output/'selection_diagnostics.json',diagnostics)
    audit.update(physical_constraints_checked=True)
    task.write(output/'validation.json',audit)
    lines=['# Retained partial-history selection experiment','',
           'Latest five observed input/output pairs, aligned to their final pose, projected onto the past rows of each full chart basis. Fixed 15-step warmup, data, noise, reference, QP and weights across configurations.','',
           '| K | d | Collision-free laps | Median RMSE (cm) |',
           '|---:|---:|---:|---:|']
    for s in diagnostics:
        if not s['partial_history_success']:continue
        error='—' if s['median_rmse_cm'] is None else f"{s['median_rmse_cm']:.2f}"
        lines.append(f"| {s['config']['clusters']} | {s['config']['rank']} | {s['partial_history_success']}/{len(seeds)} | {error} |")
    lines+=['','Rows with zero successes are omitted here and retained in selection_diagnostics.json/report.json. RMSE includes only successful laps; these reused seeds are exploratory.',
            '', 'Numerics: exact zero anchor x/y/yaw rows are removed before decomposition and restored as exact zeros in the bases. Restricted-basis singular values use a fixed relative cutoff of 1e-10. The QP objective and constraints are unchanged.',
            '',f'All {len(panel)*len(seeds)} rollouts verified. Selection diagnostics include numerical past ranks, chart use and tie rates.',
            '', 'The aligned five-sample query has 30 entries, with final x, y and yaw fixed to zero. A past-image rank of 27 therefore spans the entire admissible query space: an essentially zero residual is not evidence of predictive accuracy. Ties are resolved by smallest chart ID, without a centroid or controller fallback.']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    return diagnostics


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PAPER_DIR)
    parser.add_argument('--include-contextual',action='store_true')
    args=parser.parse_args()
    methods=tuple(atlas_method(k,d) for k,d in SETTINGS)
    if args.include_contextual:methods+=('contextual','contextual60','contextual90')
    run(args.output,PAPER_SEEDS,methods,METHOD)
    print(json.dumps(summarize(args.output),indent=2))
