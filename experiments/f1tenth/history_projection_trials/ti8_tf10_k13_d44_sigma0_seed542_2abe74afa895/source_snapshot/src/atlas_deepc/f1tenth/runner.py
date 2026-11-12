"""Run projection-residual Atlas and matched contextual F1TENTH experiments."""
from dataclasses import asdict
import json
import shutil
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from . import task
from .atlas import Atlas, fit, build_library, fit_kmeans
from comparison_methods.contextual_dpc.f1tenth import Library, Config as ContextualConfig
from .config import AtlasConfig, ATLAS_RANKS, ATLAS_SEARCH_RANKS, ATLAS_CLUSTER_COUNTS, PAPER_SEEDS, TI, TF, DT, Q_PHYSICAL, R_PHYSICAL, WARMUP_STEPS, SELECTION_METHOD
from .paths import ROOT, OFFLINE_DATA_FILE, CLEAN_OFFLINE_DATA_FILE, WAYPOINT_FILE, MAP_BASE, RESULTS_DIR, noisy_data_file
from .config import NOISE, OFFLINE_NOISE_SEED, DEFAULT_NOISE_SIGMA
from .replay import verify
from .partial_selection import METHOD as PARTIAL_METHOD, RANK_RTOL

CONFIGS = {'atlas': AtlasConfig(), 'contextual': ContextualConfig(),
           'contextual60': ContextualConfig(columns=60),
           'contextual90': ContextualConfig(columns=90)}
ATLAS_METHODS=tuple('atlas' if d==40 else f'atlas{d}' for d in ATLAS_RANKS)
CONFIGS.update({f'atlas{d}':AtlasConfig(rank=d) for d in ATLAS_RANKS if d!=40})
def atlas_method(k,d):
    return ('atlas' if d==40 else f'atlas{d}') if k==2 else f'atlas_k{k}_d{d}'

K_RANK_METHODS=tuple(atlas_method(k,d) for k in ATLAS_CLUSTER_COUNTS for d in ATLAS_SEARCH_RANKS)
CONFIGS.update({atlas_method(k,d):AtlasConfig(clusters=k,rank=d)
                for k in ATLAS_CLUSTER_COUNTS for d in ATLAS_SEARCH_RANKS})
GROUPS={'both':('atlas','contextual'),
        'contextual-columns':('contextual','contextual60','contextual90'),
        'comparison':('atlas','contextual','contextual60','contextual90'),
        'atlas-ranks':ATLAS_METHODS,
        'rank-comparison':ATLAS_METHODS+('contextual','contextual60','contextual90'),
        'k-rank-comparison':K_RANK_METHODS+('contextual','contextual60','contextual90')}


def fingerprint(noise_sigma=DEFAULT_NOISE_SIGMA):
    sources = list(Path(__file__).parent.glob('*.py')) + [
        ROOT/'src/atlas_deepc/atlas/kmeans.py',
        ROOT/'src/atlas_deepc/atlas/chart_selection.py',
        ROOT/'src/atlas_deepc/control/expert_pipeline_search.py',
        ROOT/'src/comparison_methods/contextual_dpc/reacher.py',
        ROOT/'src/comparison_methods/contextual_dpc/f1tenth.py',
        ROOT/'src/comparison_methods/common/reacher.py']
    return {str(p.relative_to(ROOT)): task.digest(p) for p in sources + [
        noisy_data_file(noise_sigma), CLEAN_OFFLINE_DATA_FILE, WAYPOINT_FILE, MAP_BASE.with_suffix('.png'), MAP_BASE.with_suffix('.yaml')]}


def protocol(seeds, methods, selection_method=SELECTION_METHOD,noise_sigma=DEFAULT_NOISE_SIGMA):
    model=task.noise_model(noise_sigma)
    result=dict(seeds=list(seeds), methods=list(methods), configurations={m: asdict(CONFIGS[m]) for m in methods},
        input_history_convention=task.INPUT_HISTORY_CONVENTION,
        t_ini=TI, t_f=TF, dt=DT, Q_physical=Q_PHYSICAL.tolist(), R_physical=R_PHYSICAL.tolist(),
        chart_selection_method=SELECTION_METHOD,selection_window=TI+TF,warmup_steps=WARMUP_STEPS,
        warmup='Shared 1.5 m lookahead reference follower for L observed history pairs only; identical noisy measurements/actions for each seed across methods. Excluded from controller metrics; included in collision replay. No fallback after handoff.',
        selection_coordinates='Concatenate L observed inputs and L observed outputs, aligned to output TI-1 inside this past L-window, matching offline columns. Unweighted full-chart projection residual; smallest chart ID breaks numerical ties.',
        hashes=fingerprint(noise_sigma),noise_model=model,
        offline_data_file=str(noisy_data_file(noise_sigma).relative_to(ROOT)),
        reference='Official unscaled varying-speed Sao Paulo raceline; example_waypoints.csv.',
        data='Local contextual-DPC reconstruction: 300 s, 20 random-excitation episodes. Output measurements receive the online Gaussian noise model before alignment, clustering and window extraction; both methods share the same noisy observations. Not the authors recorded dataset.',
        offline_measurement_noise=dict(seed=OFFLINE_NOISE_SEED,std=model['std'],distribution='independent zero-mean Gaussian',application='Once per original sample in global coordinates, before alignment and clustering; overlapping windows reuse the same noisy sample; applied inputs shared offline and online; commands saved separately'),
        online_measurement_noise=dict(std=model['std'],distribution='independent zero-mean Gaussian'),
        atlas='K and rank from configurations; shared raw speed clustering and nested uncentered SVD bases within each K. Fixed lambda_ini=100 and physical Q/R; squared tracking/input/history-slack QP, no g regularization or coefficient constraints. No K/rank-specific retuning.',
        contextual='Selected column count from configurations; fixed effective lambda_g=10, lambda_sigma=1, literal unsquared g norm, L1 augmented-output history slack, affine constraint, input-difference penalty and standardized coordinates; local reconstruction. No count-specific retuning.',
        plant='Single-track dynamics; ideal steering-angle actuator assumption; RK4 at .01 s, controls at .1 s.',
        timing='One numerical thread; reference, selection, assembly, solver setup and solve; first five calls excluded. Offline preprocessing, plant, logging and replay excluded.',
        collision='Simulator collision flag at every .01 s substep with replay-state agreement <1e-9; a full geometric lap is also required.')
    if selection_method==PARTIAL_METHOD:
        result.update(chart_selection_method=PARTIAL_METHOD,selection_window=TI,
            partial_rank_relative_tolerance=RANK_RTOL,
            selection_coordinates='Latest T_ini observed inputs/outputs, aligned to latest output pose; Euclidean projection onto the past rows [Up; Yp] of each unchanged full chart basis. Remove exact zero anchor x/y/yaw rows before SVD and embed the orthonormal basis back into 30 rows with exact zero anchor rows, for selection only. Fixed relative rank tolerance 1e-10. Explicit residual; smallest chart ID breaks numerical ties. Same 15-step initializer as full-window baseline. Full bases and QP unchanged.')
    elif selection_method!=SELECTION_METHOD:raise ValueError('Unknown selection method')
    return result


def load_run(directory, method, seed):
    folder = Path(directory)/method/f'seed{seed}'
    summary = json.loads((folder/'summary.json').read_text())
    if summary['config'] != asdict(CONFIGS[method]) or summary['seed'] != seed:
        raise ValueError(f'Configuration mismatch: {folder}')
    manifest=json.loads((Path(directory)/'protocol.json').read_text())
    selection_method=manifest['chart_selection_method']
    if 'noise_model' in manifest and (summary.get('noise_sigma')!=manifest['noise_model']['sigma'] or summary.get('measurement_noise_std')!=manifest['noise_model']['std']):
        raise ValueError(f'Noise model mismatch: {folder}')
    if summary.get('warmup_steps')!=WARMUP_STEPS or (method.startswith('atlas') and summary.get('chart_selection_method')!=selection_method):
        raise ValueError(f'Obsolete selector/history protocol: {folder}')
    replay = json.loads((folder/'collision_replay.json').read_text())
    expected = bool(summary['lap_completed'] and replay['verified'])
    if summary['verified_success'] != expected or replay['maximum_state_mismatch'] >= 1e-9:
        raise ValueError(f'Invalid collision verification: {folder}')
    return dict(method=method, id=method, **summary)


def save_inventory(directory, folder=None, inventory=None):
    directory=Path(directory)
    files=([f for f in Path(folder).rglob('*') if f.is_file()] if folder is not None else
           [f for m in CONFIGS for f in (directory/m).rglob('*') if f.is_file()])
    inventory={} if inventory is None else inventory
    inventory.update({str(f.relative_to(directory)):task.digest(f) for f in files})
    task.write(directory/'checksums.json',inventory)
    return inventory


def verify_inventory(directory):
    directory=Path(directory)
    for name, checksum in json.loads((directory/'checksums.json').read_text()).items():
        if task.digest(directory/name) != checksum:
            raise ValueError(f'Result checksum mismatch: {directory/name}')


def run(output=RESULTS_DIR/'paper_rerun', seeds=PAPER_SEEDS, methods=('atlas','contextual'), selection_method=PARTIAL_METHOD,noise_sigma=DEFAULT_NOISE_SIGMA):
    output=Path(output);seeds=tuple(seeds);methods=tuple(methods)
    if not seeds or len(set(seeds))!=len(seeds) or not methods or any(m not in CONFIGS for m in methods):
        raise ValueError('Choose unique seeds and atlas/contextual methods.')
    with threadpool_limits(limits=1):
        record=task.collect(sigma=noise_sigma);requested_methods=methods;skipped=[];atlases={}
        atlas_methods=[m for m in methods if m.startswith('atlas')]
        if atlas_methods:
            library=build_library(record)
            for k in sorted({CONFIGS[m].clusters for m in atlas_methods}):
                labels,_=fit_kmeans(library[1],k,11)
                smallest=int(np.bincount(labels,minlength=k).min())
                candidates=[m for m in atlas_methods if CONFIGS[m].clusters==k]
                feasible=[m for m in candidates if CONFIGS[m].rank<=smallest]
                skipped.extend(dict(method=m,config=asdict(CONFIGS[m]),minimum_chart_members=smallest,
                                    reason='Requested rank exceeds smallest chart membership')
                               for m in candidates if m not in feasible)
                if feasible:atlases[k]=fit(max(CONFIGS[m].rank for m in feasible),k,library)
            excluded={s['method'] for s in skipped}
            methods=tuple(m for m in methods if m not in excluded)
            if not methods:raise ValueError('No requested configuration has enough chart members.')
        expected=protocol(seeds,methods,selection_method,noise_sigma)
        expected.update(requested_methods=list(requested_methods),skipped_configurations=skipped)
        manifest=output/'protocol.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != expected:
                raise ValueError(f'Existing output has a different protocol; choose a fresh --output: {output}')
            if (output/'checksums.json').exists():verify_inventory(output)
        elif output.exists() and any(output.iterdir()):
            raise ValueError(f'Output directory is not empty: {output}')
        task.write(manifest,expected)
        for name,checksum in expected['hashes'].items():
            if name.endswith('.py'):
                target=output/'source_snapshot'/name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/name,target)
                if task.digest(target)!=checksum:raise ValueError('Source changed while preparing run')
        task.write(output/'status.json',dict(status='running',planned=len(methods)*len(seeds),completed=0))
        inventory=json.loads((output/'checksums.json').read_text()) if (output/'checksums.json').exists() else {}
        policies={}
        atlas_methods=[m for m in methods if m.startswith('atlas')]
        for k,atlas in atlases.items():
            np.savez_compressed(output/f'atlas_k{k}.npz',**atlas)
        policies.update({m:Atlas(atlases[CONFIGS[m].clusters],CONFIGS[m],selection_method) for m in atlas_methods})
        if any(m.startswith('contextual') for m in methods):
            lib=Library(record)
            policies.update({m:lib for m in methods if m.startswith('contextual')})
            np.savez_compressed(output/'normalization.npz',**{k:getattr(lib,k) for k in ['z_mean','z_scale','du_mean','du_scale']})
        rows=[]
        for seed in seeds:
            for method in methods:
                folder=output/method/f'seed{seed}'
                if not (folder/'collision_replay.json').exists():
                    task.rollout(policies[method],CONFIGS[method],seed,save=folder,noise_sigma=noise_sigma)
                    check=verify(folder)
                    # The empty-rollout case is an explicit failed experiment.
                    if check.get('reason')=='empty rollout':
                        s=json.loads((folder/'summary.json').read_text());s['verified_success']=False
                        task.write(folder/'summary.json',s)
                rows.append(load_run(output,method,seed))
                task.write(output/'evaluation.json',rows)
                save_inventory(output,folder,inventory)
                task.write(output/'status.json',dict(status='running',planned=len(methods)*len(seeds),completed=len(rows)))
                print(f"{method}, seed {seed}: collision-free lap={rows[-1]['verified_success']}, RMSE={rows[-1]['rmse_cm']} cm",flush=True)
        task.write(output/'status.json',dict(status='complete',planned=len(methods)*len(seeds),completed=len(rows)))
        return rows


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=list(CONFIGS)+list(GROUPS), default='both')
    p.add_argument('--seeds', type=int, nargs='+', default=list(PAPER_SEEDS))
    p.add_argument('--output', type=Path, default=RESULTS_DIR/'paper_rerun')
    p.add_argument('--selection',choices=(SELECTION_METHOD,PARTIAL_METHOD),default=PARTIAL_METHOD)
    p.add_argument('--sigma',type=float,default=DEFAULT_NOISE_SIGMA)
    args=p.parse_args()
    run(args.output,args.seeds,GROUPS.get(args.method,(args.method,)),args.selection,args.sigma)
    from .reporting import report
    print(json.dumps(report(args.output),indent=2))
