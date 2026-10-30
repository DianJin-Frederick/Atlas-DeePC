"""Summarize separate history experiments without rerunning any controllers."""
import json
import numpy as np

from atlas_deepc.benchmarks import gs_deepc_history_pilot as pilot

gs=pilot.gs


def summarize(directories,seeds):
    rows={}
    for directory in directories:
        for seed in seeds:
            path=directory/f'seed{seed}_control_results.npz'
            if path.exists():
                if seed in rows:
                    raise ValueError('Duplicate rollout')
                with np.load(path) as a:
                    rows[seed]=json.loads(a['summary'].item())
    return dict(seeds=seeds,completed=len(rows),successes=sum(r['strict_success'] for r in rows.values()),
        mean_final100_mm=float(1000*np.mean([r['mean_last_100'] for r in rows.values()])) if rows else None,
        fallback_steps=sum(r['fallback_steps'] for r in rows.values()),
        results=[rows[s] for s in sorted(rows)])


def export():
    baseline=gs.ROOT/'experiments/reacher/gs_deepc_control_tuning_tf10_v1'
    sched=pilot.scheduler_pilot.ROOT
    locations={('q2',2):[baseline/'q1000_r5_g10_two_seed_320',baseline/'validation_q1000_r5_g10_extra6'],
        ('link2',2):[sched/'link2',sched/'validation_extra6/link2']}
    for key in [('q2',4),('link2',3)]:
        locations[key]=sorted(pilot.ROOT.glob(f'{key[0]}_tini{key[1]}_seeds*'))
    for (scheduler,tini),directories in locations.items():
        for directory in directories:
            m=json.loads((directory/'manifest.json').read_text())
            assert m['config']==pilot.asdict(pilot.config(tini))
            assert m['steps']==320
            for k in ('controller_sha256','utilities_sha256','data_sha256'):
                assert m[k]==pilot.fingerprints()[k]
    rows=[]
    for (scheduler,tini),directories in locations.items():
        row=summarize(directories,[211,222])
        rows.append(dict(scheduler=scheduler,tini=tini,**row))
    result=dict(two_seed_screen=rows,prediction_audit=json.loads((pilot.ROOT/'prediction_audit.json').read_text()),
        note='Separate history-length adaptation, not the matched T_ini=2 benchmark. Same QP family, weights, horizon 10, raw columns; retained windows change with history length.')
    seeds=[211,219,222,237,258,281,294,305]
    extended=[dict(scheduler=s,tini=t,**summarize(d,seeds)) for (s,t),d in locations.items()]
    result['eight_seed_panel']=extended
    gs.utils.write_json(pilot.ROOT/'history_report.json',result)
    return result


if __name__=='__main__':
    print(json.dumps(export(),indent=2))
