"""Read-only tuning and paired saved-result reporting for supplied Atlas goals."""
import json
import numpy as np
from atlas_deepc.benchmarks import contextual_dpc_atlas_terminal as case


def tuning_report():
    groups={}
    for path in sorted(case.ROOT.glob('pilot_*/manifest.json')):
        m=json.loads(path.read_text())
        assert m['source_sha256']==case.base.digest(case.__file__)
        assert m['base_sha256']==case.base.digest(case.base.__file__)
        key=json.dumps(m['config'],sort_keys=True)
        group=groups.setdefault(key,dict(config=m['config'],results={},directories=[]))
        group['directories'].append(path.parent.name)
        for seed in m['seeds']:
            saved=path.parent/f'seed{seed}_control_results.npz'
            if not saved.exists():continue
            with np.load(saved) as a:row=json.loads(a['summary'].item())
            assert seed not in group['results']
            group['results'][seed]=row
    panels={}
    for name,seeds in [('screen',[211,222]),('confirmation',[211,222,258,281])]:
        rows=[]
        for g in groups.values():
            if not all(s in g['results'] for s in seeds):continue
            r=[g['results'][s] for s in seeds]
            rows.append(dict(config=g['config'],seeds=seeds,successes=sum(v['strict_success'] for v in r),
                total=len(r),mean_final100_mm=float(1000*np.mean([v['mean_last_100'] for v in r])),
                fallback_steps=sum(v['fallback_steps'] for v in r),directories=g['directories']))
        panels[name]=sorted(rows,key=lambda r:(-r['successes'],r['mean_final100_mm']))
    result=dict(**panels,completed_rollouts=sum(len(g['results']) for g in groups.values()),
        selection_rule='Strict successes, then mean final-100 fingertip error on identical seed panels. Never rank different-weight native costs.',
        caveat='Small development tuning, not exhaustive optimization or held-out evaluation.')
    case.write_json(case.ROOT/'tuning_report.json',result)
    return result


def paired_report(new_directory):
    old=case.base.ROOT/'experiments/reacher/contextual_dpc_raw_final_211_310_q100_r01_serial_v1'
    seeds=[s for s in range(211,311) if all((d/f'seed{s}_control_results.npz').exists() for d in (old,new_directory))]
    rows=[]; outcomes=[]
    for label,directory in [('Contextual: XY reference',old),('Contextual: supplied Atlas 8-output reference',new_directory)]:
        errors,rmse,successes=[],[],[]
        timing={k:[] for k in ('control_compute_time_s','selection_time_s','matrix_preparation_time_s','solve_time_s')}
        for s in seeds:
            with np.load(directory/f'seed{s}_control_results.npz') as a:
                error=a['distance']; assert len(error)==320
                success=all(error[-n:].mean()<.01 and error[-n:].max()<.02 for n in (30,100))
                successes.append(success);errors.append(1000*error[-100:].mean())
                rmse.append(1000*np.sqrt(np.mean(error**2)))
                for k in timing:timing[k].extend(a[k][5:].tolist())
                if label.endswith('8-output reference'):
                    with np.load(old/f'seed{s}_control_results.npz') as b:
                        np.testing.assert_array_equal(a['target_xy'],b['target_xy'])
                        np.testing.assert_allclose(a['warmup_y'],b['warmup_y'],rtol=0,atol=1e-12)
        if seeds:
            outcomes.append(np.asarray(successes,dtype=bool))
            rows.append(dict(method=label,successes=int(sum(successes)),total=len(seeds),
                success_rate_pct=100*np.mean(successes),median_final100_mm=float(np.median(errors)),
                p90_final100_mm=float(np.percentile(errors,90)),p95_final100_mm=float(np.percentile(errors,95)),
                p99_final100_mm=float(np.percentile(errors,99)),median_rollout_rmse_mm=float(np.median(rmse)),
                modular_timing={k:case.timing_stats(v) for k,v in timing.items()}))
    report=dict(paired_seeds=seeds,rows=rows,note='Paired completed development seeds only. Reference information and tuned weights differ; native costs are not ranked. Supplied reference setup excluded from online step time.')
    if len(outcomes)==2:
        old_ok,new_ok=outcomes
        report['paired_outcomes']=dict(both_success=int(np.sum(old_ok&new_ok)),
            both_failure=int(np.sum(~old_ok&~new_ok)),
            new_only_success=[s for s,yes in zip(seeds,~old_ok&new_ok) if yes],
            original_only_success=[s for s,yes in zip(seeds,old_ok&~new_ok) if yes])
    if new_directory.exists():case.write_json(new_directory/'paired_reference_comparison.json',report)
    return report
