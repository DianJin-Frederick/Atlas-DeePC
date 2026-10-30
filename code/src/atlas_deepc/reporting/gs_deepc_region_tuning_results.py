"""Audit and summarize parameter-only GS-DeePC trials without rerunning them."""
import json
import numpy as np

from atlas_deepc.benchmarks import gs_deepc_region_tuning as tuning

gs=tuning.gs


def export():
    original=gs.ROOT/'experiments/reacher/gs_deepc_control_tuning_tf10_v1'
    scheduler=tuning.pilot.ROOT
    directories=sorted(p.parent for p in tuning.ROOT.glob('*/manifest.json'))+[
        original/'q1000_r5_g10_two_seed_320',original/'validation_q1000_r5_g10_extra6',
        scheduler/'link2',scheduler/'validation_extra6/link2']
    groups={}; cases=[]; model_banks={}
    for directory in directories:
        manifest=json.loads((directory/'manifest.json').read_text())
        for key,path in [('controller_sha256',gs.__file__),('utilities_sha256',gs.utils.__file__),('data_sha256',gs.DATA)]:
            assert manifest[key]==gs.utils.digest(path)
        c=manifest['config']
        assert (c['tini'],c['horizon'],c['columns_per_region'])==(2,10,200)
        for key,value in tuning.asdict(gs.Config()).items():
            if key not in ('regions','q_xy','r','lambda_g','lambda_ini'):
                assert c[key]==value
        assert manifest['steps']==320
        key=json.dumps(dict(config=c,scheduler=manifest.get('scheduler','q2')),sort_keys=True)
        group=groups.setdefault(key,dict(config=c,scheduler=manifest.get('scheduler','q2'),rows={},banks=None))
        bank_path=directory/'offline_library.npz'
        if not bank_path.exists():
            continue
        with np.load(bank_path) as a:
            bank={k:a[k].copy() for k in ('hu','hy','provenance')}
        model_key=(group['scheduler'],c['regions'])
        if model_key in model_banks:
            for k in bank: np.testing.assert_array_equal(bank[k],model_banks[model_key][k])
        else:model_banks[model_key]=bank
        if group['banks'] is not None:
            for k in bank: np.testing.assert_array_equal(bank[k],group['banks'][k])
        else:group['banks']=bank
        rows=[]
        for seed in manifest['seeds']:
            path=directory/f'seed{seed}_control_results.npz'
            if not path.exists():continue
            with np.load(path) as a:row=json.loads(a['summary'].item())
            if seed in group['rows']:raise ValueError('Duplicate seed/configuration')
            group['rows'][seed]=row;rows.append(row)
        cases.append(dict(directory=str(directory.relative_to(gs.ROOT)),config=c,
            scheduler=group['scheduler'],seeds=manifest['seeds'],completed=len(rows),
            successes=sum(r['strict_success'] for r in rows),reused=directory.parent!=tuning.ROOT))
    panels={}
    for name,seeds in [('two_seed_screen',[211,222]),('eight_seed_panel',[211,219,222,237,258,281,294,305])]:
        rows=[]
        for group in groups.values():
            if not all(s in group['rows'] for s in seeds):continue
            results=[group['rows'][s] for s in seeds]
            c=group['config']
            rows.append(dict(scheduler=group['scheduler'],regions=c['regions'],composite_regions=c['regions']-1,
                q=c['q_xy'],r=c['r'],lambda_g=c['lambda_g'],lambda_ini=c['lambda_ini'],
                successes=sum(r['strict_success'] for r in results),total=len(seeds),
                mean_final100_mm=float(1000*np.mean([r['mean_last_100'] for r in results])),
                fallback_steps=sum(r['fallback_steps'] for r in results),results=results))
        panels[name]=sorted(rows,key=lambda r:(-r['successes'],r['mean_final100_mm']))
    result=dict(**panels,cases=cases,new_completed_rollouts=sum(c['completed'] for c in cases if not c['reused']),
        method='Unmodified GS-DeePC controller and switching code; tune uniform region count and existing quadratic weights only.',
        provenance=dict(paper=gs.PAPER,upstream_commit=gs.UPSTREAM_COMMIT),
        caveat='Development tuning, not held-out evaluation. Fixed baseline offline data; Reacher scheduler/XY interface and Clarabel port remain adaptations.',
        raw_banks_identical_across_weight_changes=True,promoted_candidate=None)
    gs.utils.write_json(tuning.ROOT/'region_tuning_report.json',result)
    return result


if __name__=='__main__':
    print(json.dumps(export(),indent=2))
