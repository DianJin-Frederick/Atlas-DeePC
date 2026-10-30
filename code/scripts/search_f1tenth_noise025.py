"""Explore only K/rank at sigma=.0025; retain full, auditable staged results."""
import argparse
import json
from pathlib import Path
import shutil

from atlas_deepc.f1tenth import task
from atlas_deepc.f1tenth.config import PAPER_SEEDS
from atlas_deepc.f1tenth.paths import RESULTS_DIR
from atlas_deepc.f1tenth.runner import run, atlas_method
from atlas_deepc.f1tenth.partial_selection import METHOD
from atlas_deepc.f1tenth.partial_experiment import summarize

ROOT = RESULTS_DIR / 'noise025_kr_search_20261027'
SIGMA = .0025
DEFAULT_CLUSTERS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20)
DEFAULT_RANKS = (24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 48, 56, 64)


def stage(name, methods, seeds):
    task.write(ROOT / 'status.json', dict(status='running', stage=name))
    output = ROOT / name
    run(output, seeds, methods, METHOD, SIGMA)
    summarize(output)
    return json.loads((output / 'report.json').read_text())


def ranked(rows):
    return sorted(rows, key=lambda row: (
        -row['collision_free_laps'] / row['evaluated'],
        row['median_rmse_cm'] if row['median_rmse_cm'] is not None else float('inf'),
        row['config']['rank'], row['config']['clusters']))


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--clusters', type=int, nargs='+', default=DEFAULT_CLUSTERS)
    parser.add_argument('--ranks', type=int, nargs='+', default=DEFAULT_RANKS)
    args = parser.parse_args()
    ROOT = args.output
    ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, ROOT / Path(__file__).name)
    task.write(ROOT / 'search_plan.json', dict(
        sigma=SIGMA, clusters=list(args.clusters), ranks=list(args.ranks),
        screen_seeds=[511], refinement_seeds=list(range(512, 516)),
        final_seeds=list(PAPER_SEEDS), refinement_candidates=12, final_candidates=3,
        ranking='Collision-free success fraction first, then median successful-lap RMSE.',
        controls='Only K and rank vary. Same frozen noisy dataset, QP, weights, projection rule, noise and reference.',
        caveat='Exploratory selection on reused rollout seeds; final 20-seed results are not a held-out test.'))
    screen = stage('screen', [atlas_method(k, d) for k in args.clusters
                             for d in args.ranks], [511])
    shortlist = [row['method'] for row in ranked(screen) if row['collision_free_laps']][:12]
    task.write(ROOT / 'shortlist.json', shortlist)
    if not shortlist:
        task.write(ROOT / 'status.json', dict(status='complete', finding='No successful screening candidate'))
        return
    refined = stage('refinement', shortlist, range(512, 516))
    finalists = [row['method'] for row in ranked(refined) if row['collision_free_laps']][:3]
    task.write(ROOT / 'finalists.json', finalists)
    if not finalists:
        task.write(ROOT / 'status.json', dict(status='complete', finding='No successful refinement candidate'))
        return
    final = stage('final20', finalists, PAPER_SEEDS)
    reference = RESULTS_DIR / 'noise_ratio_sweep_20261027' / 'sigma_0p0025'
    baseline = json.loads((reference / 'report.json').read_text())
    baseline = [row for row in baseline if row['method'].startswith('contextual')]
    p1 = json.loads((ROOT / 'final20' / 'protocol.json').read_text())
    p2 = json.loads((reference / 'protocol.json').read_text())
    assert p1['offline_data_file'] == p2['offline_data_file']
    assert p1['hashes'][p1['offline_data_file']] == p2['hashes'][p2['offline_data_file']]
    assert p1['noise_model'] == p2['noise_model']
    rows = final + baseline
    task.write(ROOT / 'comparison.json', rows)
    lines = ['# Atlas K/rank search at 0.25% measurement noise', '',
             '| Method | Collision-free laps | Median RMSE (cm) | Median controller time (ms) |',
             '|:--|--:|--:|--:|']
    for row in rows:
        error = '—' if row['median_rmse_cm'] is None else f"{row['median_rmse_cm']:.2f}"
        lines.append(f"| {row['label']} | {row['collision_free_laps']}/{row['evaluated']} | {error} | {row['timing']['median_ms']:.2f} |")
    lines += ['', 'Both methods use the identical noisy offline file (hash checked), online noise, reference and 20 seeds. Only Atlas K/rank changed. Additional objective terms differ between methods.', '',
              'RMSE covers collision-free completed laps only. Screening: one seed across the supported grid; refinement: four more seeds for the top 12; final: 20 reused exploratory seeds for the top three. This is parameter search, not an independent generalization test.', '',
              'All failures and unsupported configurations remain in the stage reports and protocols. See final20/selection_diagnostics.json for projection ranks, chart occupancy and ties.']
    (ROOT / 'comparison.md').write_text('\n'.join(lines) + '\n')
    task.write(ROOT / 'status.json', dict(status='complete', stage='final20', finalists=finalists,
                                        identical_contextual_offline_data_verified=True))


if __name__ == '__main__':
    main()
