"""Tables and track plots for the two retained fixed configurations."""
import json
from pathlib import Path
import numpy as np
from . import task
from .paths import PAPER_DIR
from .runner import load_run, verify_inventory, CONFIGS
from .config import ATLAS_RANKS

LABELS={'atlas':'Atlas-DeePC (K=2, d=40)','contextual':'Contextual-DPC (30 columns)',
        'contextual60':'Contextual-DPC (60 columns)','contextual90':'Contextual-DPC (90 columns)'}
LABELS.update({f'atlas{d}':f'Atlas-DeePC (K=2, d={d})' for d in ATLAS_RANKS if d!=40})
LABELS.update({m:f'Atlas-DeePC (K={c.clusters}, d={c.rank})' for m,c in CONFIGS.items() if m.startswith('atlas')})


def report(directory=PAPER_DIR):
    directory=Path(directory);verify_inventory(directory)
    protocol=json.loads((directory/'protocol.json').read_text())
    panel=[]
    for method in protocol['methods']:
        runs=[load_run(directory,method,s) for s in protocol['seeds']]
        good=[s for s in runs if s['verified_success']]
        calls=[c for s in protocol['seeds'] for c in json.loads((directory/method/f'seed{s}'/'calls.json').read_text())[5:]]
        timing=task.timing([s['total_s'] for s in calls])
        panel.append(dict(method=method,label=LABELS[method],evaluated=len(runs),
            collision_free_laps=len(good),geometric_laps=sum(s['lap_completed'] for s in runs),
            median_rmse_cm=float(np.median([s['rmse_cm'] for s in good])) if good else None,
            geometric_median_rmse_cm=float(np.median([s['rmse_cm'] for s in runs if s['lap_completed']])) if any(s['lap_completed'] for s in runs) else None,
            median_speed_rmse_mps=float(np.median([s['speed_rmse_mps'] for s in good])) if good else None,
            median_lap_fraction=float(np.median([s['lap_fraction'] for s in runs])),
            terminations={reason:sum(s['termination']==reason for s in runs) for reason in sorted({s['termination'] for s in runs})},
            timing=timing,config=runs[0]['config']))
    task.write(directory/'report.json',panel)
    lines=['# F1TENTH retained paper experiment','',
        '| Method | Collision-free laps | Geometric laps | Median RMSE (cm) | Median time (ms) | P99 time (ms) |',
        '|:---|---:|---:|---:|---:|---:|']
    for s in panel:
        if not s['collision_free_laps']:continue
        error='—' if s['median_rmse_cm'] is None else f"{s['median_rmse_cm']:.2f}"
        timing=s['timing'] or {'median_ms':float('nan'),'p99_ms':float('nan')}
        lines.append(f"| {s['label']} | {s['collision_free_laps']}/{s['evaluated']} | {s['geometric_laps']}/{s['evaluated']} | {error} | {timing['median_ms']:.2f} | {timing['p99_ms']:.2f} |")
    lines += ['',
        'RMSE is the median of per-rollout cross-track RMSE over collision-free completed laps. The methods share offline data, official varying-speed reference, evaluation seeds, noise and physical tracking/input weights. Their additional objective terms differ. Contextual DPC is our local reconstruction, not the authors’ released data or reported paper results.', '',
        'Only configurations with at least one collision-free completed lap are displayed. All evaluated settings, including zero-success settings, remain in report.json and the raw records. Ranks exceeding the smallest chart membership are skipped and recorded in protocol.json.', '',
        'Atlas comparisons keep training windows, Q/R, lambda_ini=100, solver tolerances and every task setting fixed. Chart memberships and centroids change with K; each rank within a K takes a prefix of the same uncentered SVD bases. No K/rank-specific tuning is performed. More rank changes model flexibility, not the amount of collected data. Timing pools all runs, including failed prefixes, so partial-success settings are not matched full-lap timing comparisons. These common seeds have been used in earlier exploratory comparisons.', '',
        'T_ini=5, T_f=10, dt=0.1 s. Physical Q=diag(100,100,1,1), R=diag(0.1,10). Atlas uses lambda_ini=100, offline raw speed-based K-means charts and uncentered SVD, online full-window projection-residual selection, with no g regularization or coefficient bounds. Contextual uses the displayed column count, fixed effective lambda_g=10 with the unsquared L2 norm, lambda_sigma=1 with L1 history slack, affine and incremental-input constraints, and Clarabel. Larger column counts are not separately retuned.', '',
        'Timing includes reference construction, selection, assembly, setup and solve, with the first five calls omitted. Offline preparation, simulation, logging and collision replay are excluded. A collision-free full lap requires state-verified simulator collision replay at every 0.01 s substep.', '',
        'All results use projection-residual Atlas selection. Both controllers receive identical 15-step reference-following warmup histories; the initializer is never used after handoff. Warmup is excluded from error/timing metrics but included in collision replay. Source/data hashes, selection windows and all residuals are saved for verification.']
    if protocol['chart_selection_method']=='past_window_projection':
        lines[0]='# F1TENTH experimental past-history projection selection'
        lines=[s.replace('online full-window projection-residual selection','online past-window projection-residual selection') for s in lines]
        lines+=['',protocol['selection_coordinates'],
                'Atlas uses past-history projection. At high ranks, restricted charts can span the entire aligned history space and cannot be distinguished by this residual; numerical ties select the lowest chart ID. Consult selection_diagnostics.json for measured chart use.']
    if all(m.startswith('contextual') for m in protocol['methods']):
        lines=['# Retained contextual-DPC comparison','']+[s for s in lines if s.startswith('|')]+['',
            'Unchanged contextual-DPC rollouts with 30/60/90 selected columns, 20 common seeds each. Median RMSE covers collision-free completed laps. This is our local reconstruction, not the authors’ reported results.',
            '', 'T_ini=5, T_f=10, dt=0.1 s; official varying-speed reference and noisy measurements. Physical Q=diag(100,100,1,1), R=diag(0.1,10). Fixed lambda_g=10 unsquared L2 coefficient penalty, lambda_sigma=1 L1 history slack, affine and incremental-input constraints, standardized coordinates and Clarabel.',
            '', 'Identical 15-step initialization and seeds as the retained Atlas trial. Collision replay includes warmup and every 0.01 s substep. Timings include selection, assembly, setup and solve, omitting the first five calls. Source snapshots and raw-record checksums were preserved when these runs were moved into their own folder.']
    if 'noise_model' in protocol:
        model=protocol['noise_model']
        lines+=['',f"Noise: sigma={model['sigma']} is the per-channel noise/signal standard-deviation ratio. Identical physical measurement standard deviations offline and online: {model['std']} for x/y/speed/yaw. Scales are fixed from clean offline global outputs; offline noise seed {model['offline_seed']}. Noise is added before alignment, clustering and window extraction."]
    (directory/'report.md').write_text('\n'.join(lines)+'\n')
    return panel


