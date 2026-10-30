"""Select Atlas K/rank on development seeds, then freeze a fresh NSR table."""
import argparse
import json
from pathlib import Path
import shutil
from dataclasses import asdict
import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.f1tenth import task
from atlas_deepc.f1tenth.atlas import build_library, fit_kmeans
from atlas_deepc.f1tenth.config import NOISE_SIGMAS, PAPER_SEEDS
from atlas_deepc.f1tenth.paths import RESULTS_DIR, noise_tag
from atlas_deepc.f1tenth.runner import run, atlas_method, CONFIGS
from atlas_deepc.f1tenth.partial_selection import METHOD
from atlas_deepc.f1tenth.partial_experiment import summarize

ROOT = RESULTS_DIR / 'paper_nsr_20261027'
DEV = tuple(PAPER_SEEDS)
TEST = tuple(range(531, 551))
CONTEXTUAL = ('contextual', 'contextual60', 'contextual90')
COARSE_K = tuple(range(1, 25))
COARSE_D = tuple(range(16, 85, 4)) + (86,)
OLD = RESULTS_DIR / 'noise_ratio_sweep_20261027'


def read(p):
    return json.loads(Path(p).read_text())


def ranking(r):
    return (-r['collision_free_laps']/r['evaluated'],
            r['median_rmse_cm'] if r['median_rmse_cm'] is not None else 1e9,
            -r['geometric_laps']/r['evaluated'], -r['median_lap_fraction'],
            r['geometric_median_rmse_cm'] if r['geometric_median_rmse_cm'] is not None else 1e9,
            r['config'].get('rank', 0))


def comparable(atlas, reference):
    return (reference['collision_free_laps'] > 0
            and atlas['collision_free_laps'] >= reference['collision_free_laps']
            and atlas['median_rmse_cm'] is not None
            and atlas['median_rmse_cm'] <= 1.2*reference['median_rmse_cm'])


def stage(path, sigma, methods, seeds):
    task.write(ROOT/'status.json', dict(status='running', sigma=sigma, stage=str(path.relative_to(ROOT))))
    run(path, seeds, methods, METHOD, sigma)
    summarize(path)
    return read(path/'report.json')


def known_candidates(sigma):
    folders = [OLD/f'sigma_{noise_tag(sigma)}']
    if sigma == .0025:
        folders += [RESULTS_DIR/name/'final20' for name in
                    ('noise025_kr_search_20261027', 'noise025_kr_fine_20261027')]
    rows = {}
    for folder in folders:
        for r in read(folder/'report.json'):
            if r['method'].startswith('atlas') and r['evaluated'] == len(DEV):
                rows[r['method']] = dict(r, development_source=str(folder.relative_to(RESULTS_DIR)))
    return rows


def search_level(sigma):
    folder = ROOT/'development'/f'sigma_{noise_tag(sigma)}'
    locked = ROOT/'selection'/f'sigma_{noise_tag(sigma)}.json'
    if locked.exists():
        return read(locked)
    baseline = read(OLD/f'sigma_{noise_tag(sigma)}'/'report.json')
    reference = next(r for r in baseline if r['method']=='contextual90')
    candidates = known_candidates(sigma)
    best = min(candidates.values(), key=ranking)
    ledger = dict(sigma=sigma, target=reference, initial_candidates=list(candidates.values()),
                  stages=[], seed_roles=dict(development=list(DEV), paper=list(TEST)))
    if not comparable(best, reference):
        with threadpool_limits(limits=1):
            library = build_library(task.collect(sigma=sigma))
            memberships = {k:int(np.bincount(fit_kmeans(library[1], k, 11)[0], minlength=k).min())
                           for k in COARSE_K}
        ledger['minimum_chart_members'] = memberships
        # Use a seed whose common initializer is collision-free, so a warmup
        # collision cannot eliminate every candidate at the screening stage.
        eligible=[];handoff_distances={}
        for seed in DEV:
            previous=OLD/f'sigma_{noise_tag(sigma)}'/'contextual90'/f'seed{seed}'
            state=np.load(previous/'rollout.npz')['initial_state']
            index=task.PATH.localize(state[:2],0)%task.PATH.n
            distance=float(np.linalg.norm(state[:2]-task.PATH.xy[index]))
            handoff_distances[seed]=distance
            if (read(previous/'collision_replay.json').get('collision_phase')!='warmup'
                    and distance<=1.):
                eligible.append(seed)
        ledger['screening_handoff_distances_m']=handoff_distances
        if len(eligible)<5:
            raise ValueError('Fewer than five valid initialization seeds for development screening.')
        screen_seed = eligible[0]
        refinement_seeds = eligible[1:5]
        ledger['screen_seed'] = screen_seed
        ledger['refinement_seeds'] = refinement_seeds
        tested = set(candidates)
        for round_name in ('coarse', 'fine'):
            if round_name == 'coarse':
                pairs = [(k,d) for k in COARSE_K for d in COARSE_D]
            else:
                # Refine around the best fully evaluated models; no controller
                # equations, weights, data or online selector are changed.
                centers = sorted(candidates.values(), key=ranking)[:4]
                pairs = sorted({(k,d) for r in centers
                                for k in range(max(1,r['config']['clusters']-1), min(24,r['config']['clusters']+1)+1)
                                for d in range(max(16,r['config']['rank']-3), min(86,r['config']['rank']+3)+1)})
            methods = [atlas_method(k,d) for k,d in pairs
                       if d <= memberships[k] and atlas_method(k,d) not in tested]
            ledger.setdefault('infeasible', {})[round_name] = [dict(K=k,d=d,minimum_chart_members=memberships[k])
                                                             for k,d in pairs if d>memberships[k]]
            if not methods:
                continue
            screen = stage(folder/round_name/'screen', sigma, methods, [screen_seed])
            tested.update(methods)
            short = [r['method'] for r in sorted(screen,key=ranking)[:12]]
            refined = stage(folder/round_name/'refinement', sigma, short, refinement_seeds)
            finalists = [r['method'] for r in sorted(refined,key=ranking)[:3]]
            final = stage(folder/round_name/'final20', sigma, finalists, DEV)
            for r in final:
                candidates[r['method']] = dict(r, development_source=str((folder/round_name/'final20').relative_to(RESULTS_DIR)))
            ledger['stages'].append(dict(round=round_name, screening=screen, refinement=refined, final20=final))
            best = min(candidates.values(), key=ranking)
            task.write(folder/'search_ledger.json', ledger)
            if comparable(best, reference):
                break
    ledger.update(selected=best, development_target_reached=comparable(best,reference),
                  target_interpretable=reference['collision_free_laps']>0,
                  selection_rule='Maximize collision-free count, then minimize successful-lap RMSE; if no successes, use geometric laps and progress as tie breakers.',
                  search_limit='Coarse K=1..24, d=16,20,...,84,86; rank limited by chart sample count. Then local integer ranks/K around the four best development settings. No guarantee that K/rank alone reaches the benchmark.')
    task.write(folder/'search_ledger.json', ledger)
    result = dict(sigma=sigma, method=best['method'], config=asdict(CONFIGS[best['method']]),
                  development_result=best, development_reference=reference,
                  development_target_reached=comparable(best,reference),
                  target_interpretable=reference['collision_free_laps']>0,
                  frozen_before_paper_seeds=True)
    task.write(locked,result)
    print('LOCKED',sigma,result['method'],'target reached:',result['development_target_reached'],flush=True)
    return result


def aggregate():
    rows=[]; diagnostics=[]
    for sigma in NOISE_SIGMAS:
        folder=ROOT/f'sigma_{noise_tag(sigma)}'
        if not (folder/'validation.json').exists():
            continue
        rows.extend(dict(sigma=sigma,**r) for r in read(folder/'report.json'))
        diagnostics.extend(dict(sigma=sigma,**r) for r in read(folder/'selection_diagnostics.json'))
    task.write(ROOT/'report.json',rows)
    task.write(ROOT/'selection_diagnostics.json',diagnostics)
    lines=['# F1TENTH paper comparison across measurement NSR', '',
           'Twenty fresh, paired online seeds (531–550) per setting. One Atlas (K,d) is selected per noise level using development seeds 511–530 and frozen before these evaluations. The offline realization is shared between methods at each level; noise is added before clustering. NSR denotes a standard-deviation ratio, not a power ratio.', '',
           '| NSR (%) | Method | Collision-free laps | Geometric laps | Median successful RMSE (cm) | Median controller time (ms) |',
           '|--:|:--|--:|--:|--:|--:|']
    latex=[r'\begin{tabular}{rlrrrr}',r'\toprule',r'NSR (\%) & Method & Success & Laps & RMSE (cm) & Time (ms) \\',r'\midrule']
    for r in rows:
        err='—' if r['median_rmse_cm'] is None else f"{r['median_rmse_cm']:.2f}"
        timing='—' if not r['timing'] else f"{r['timing']['median_ms']:.2f}"
        lines.append(f"| {100*r['sigma']:g} | {r['label']} | {r['collision_free_laps']}/20 | {r['geometric_laps']}/20 | {err} | {timing} |")
        label=(f"Atlas ($K={r['config']['clusters']},d={r['config']['rank']}$)" if r['method'].startswith('atlas')
               else f"Contextual ({r['config']['columns']})")
        latex.append(f"{100*r['sigma']:g} & {label} & {r['collision_free_laps']}/20 & {r['geometric_laps']}/20 & {err.replace('—','--')} & {timing.replace('—','--')} "+r'\\')
    latex += [r'\bottomrule',r'\end{tabular}']
    lines += ['', 'RMSE uses collision-free completed laps only; a dash means none succeeded. Timing includes all controller calls after the first five, including failed prefixes. Geometric lap completion does not imply collision-free driving. Warmup is included in collision checks.', '',
              'Only K and d are tuned for Atlas; its QP, physical weights, partial-history projection rule and tie handling remain fixed. Contextual-DPC retains 30/60/90 columns and its own objective terms. The methods use identical noisy offline observations, physical noise standard deviations and paired online seeds; their objectives are not identical.', '',
              'The development target is Contextual-90 success count and RMSE within 20%; it is not imposed on the fresh paper results. Failed levels remain visible. When the contextual baseline has zero successes, successful-lap error comparability is undefined.', '',
              'Projection residuals may tie because chart past images span the aligned history space. Consult selection_diagnostics.json before interpreting successes as an adaptive switching benefit. There is one offline noise realization per level, not multiple independent training datasets.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'table.tex').write_text('\n'.join(latex)+'\n')
    return rows


def main():
    global ROOT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT)
    p.add_argument('--phase',choices=('search','evaluate','all'),default='all')
    args=p.parse_args();ROOT=args.output
    ROOT.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(__file__,ROOT/Path(__file__).name)
    plan=dict(sigmas=list(NOISE_SIGMAS),development_seeds=list(DEV),paper_seeds=list(TEST),
              contextual_columns=[30,60,90],selection_target='Contextual-90 success count and successful-lap median RMSE within 20%',
              coarse_clusters=list(COARSE_K),coarse_ranks=list(COARSE_D),
              screening_handoff_limit_m=1.,
              selector=METHOD,algorithm_unchanged=True,
              noise='One sigma per level, channel noise std=sigma*clean offline channel std; same noisy offline file for both methods.',
              code_sha256=task.digest(Path(__file__)))
    manifest=ROOT/'study_protocol.json'
    if manifest.exists() and read(manifest)!=plan:
        raise ValueError('Paper study protocol changed; use a new output folder')
    task.write(manifest,plan)
    if args.phase in ('search','all'):
        for sigma in NOISE_SIGMAS:
            search_level(sigma)
        task.write(ROOT/'status.json',dict(status='selection_complete',levels=len(NOISE_SIGMAS)))
    if args.phase in ('evaluate','all'):
        # Require all choices to be locked before exposing any paper result.
        choices={s:read(ROOT/'selection'/f'sigma_{noise_tag(s)}.json') for s in NOISE_SIGMAS}
        for sigma in NOISE_SIGMAS:
            stage(ROOT/f'sigma_{noise_tag(sigma)}',sigma,[choices[sigma]['method'],*CONTEXTUAL],TEST)
            aggregate()
        task.write(ROOT/'status.json',dict(status='complete',levels=len(NOISE_SIGMAS),rollouts=20*4*len(NOISE_SIGMAS),
                                          configurations_frozen_before_evaluation=True))


if __name__=='__main__':
    main()
