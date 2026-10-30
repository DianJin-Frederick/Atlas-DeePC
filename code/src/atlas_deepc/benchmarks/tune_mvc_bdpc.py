"""Small, explicit control-weight pilots with one frozen MVC offline model."""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.benchmarks import mvc_bdpc_reacher as m

ROOT=m.ROOT/'experiments/reacher/mvc_bdpc_control_tuning_v1'
OFFLINE=m.ROOT/'experiments/reacher/mvc_bdpc_implementation_check_final_v2'
CANDIDATES={
    'original': (400.,1.,1400.),
    'effort100': (400.,100.,1400.),
    'balanced': (400.,100.,20.),
    'slack20': (400.,1.,20.),
    'slack1': (400.,1.,1.),
    'effort10000': (400.,10000.,1400.),
    'effort001_slack20': (400.,.01,20.),
    'effort001_slack1': (400.,.01,1.),
    'effort100_slack1': (400.,100.,1.),
    'effort4': (400.,4.,1400.),
    'effort10': (400.,10.,1400.),
    'effort25': (400.,25.,1400.),
    'slack200': (400.,1.,200.),
    'slack700': (400.,1.,700.),
    'slack700_effort4': (400.,4.,700.),
    'effort15': (400.,1.5,1400.),
    'effort2': (400.,2.,1400.),
    'slack1000': (400.,1.,1000.),
    'tracking100x': (40000.,1.,1400.),
    'slack10000': (400.,1.,10000.),
}


def report(directory,seeds):
    result=m.summarize(directory,seeds)
    rows=result['results']; errors=[]; predictions=[]; clips=[]; movement=[]
    for row in rows:
        with np.load(directory/f"seed{row['seed']}_control_results.npz") as a:
            if not row['complete']: continue
            errors.append(1000*np.mean(a['distance'][-100:]))
            predictions.append(1000*np.mean(a['prediction_error_xy']))
            clips.append(np.mean(a['action_clip']>1e-8))
            movement.append(float(np.mean(np.linalg.norm(a['u'],axis=1))))
    compact={'finished':len(rows),'full_rollouts':len(errors),'successes':sum(r['strict_success'] for r in rows),
             'mean_final100_mm':float(np.mean(errors)) if errors else None,
             'mean_prediction_mm':float(np.mean(predictions)) if predictions else None,
             'saturation_fraction':float(np.mean(clips)) if clips else None,
             'mean_action_norm':float(np.mean(movement)) if movement else None,
             'failed_seeds':[r['seed'] for r in rows if not r['strict_success']]}
    m.utils.write_json(directory/'tuning_summary.json',compact)
    return compact


def run(names,seeds,steps,phase):
    import fcntl
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'tuning.lock').open('a+') as lock,threadpool_limits(limits=1):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=json.loads((OFFLINE/'manifest.json').read_text())
        assert previous['source_sha256']==m.utils.digest(m.__file__)
        assert previous['data_sha256']==m.utils.digest(m.DATA)
        original=m.Config(**previous['config'])
        all_results={}
        for name in names:
            q,r,epsilon=CANDIDATES[name]
            config=replace(original,q_xy=q,r=r,lambda_epsilon=epsilon)
            lib=m.Library.load(OFFLINE/'offline_model.npz',config)
            directory=ROOT/phase/name;directory.mkdir(parents=True,exist_ok=True)
            manifest={'config':asdict(config),'seeds':seeds,'steps':steps,'controller_sha256':previous['source_sha256'],
                      'data_sha256':previous['data_sha256'],'offline_cache_sha256':m.utils.digest(OFFLINE/'offline_model.npz'),
                      'offline_source':str(OFFLINE),'tuner_sha256':m.utils.digest(__file__),
                      'note':'Only q_xy, r, lambda_epsilon changed; frozen clustering/models; serial one-thread pilot'}
            if (directory/'manifest.json').exists():
                if json.loads((directory/'manifest.json').read_text())!=manifest:raise ValueError('Incompatible cache; use a new phase')
            else:m.utils.write_json(directory/'manifest.json',manifest)
            start=time.perf_counter()
            for seed in seeds:
                if not (directory/f'seed{seed}_control_results.npz').exists():
                    with (directory/f'seed{seed}.log').open('w') as log,redirect_stdout(log):
                        m.run_seed(lib,seed,steps,directory)
                result=report(directory,seeds)
                print(json.dumps({'candidate':name,'seed':seed,**result}),flush=True)
            all_results[name]={'config':asdict(config),'elapsed_seconds':time.perf_counter()-start,**result}
        m.utils.write_json(ROOT/phase/'pilot_report.json',all_results)
        return all_results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates',nargs='+',choices=list(CANDIDATES),required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[211])
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--phase',required=True)
    a=parser.parse_args()
    if not all(211<=s<=310 for s in a.seeds) or a.steps<100:raise ValueError('Use development seeds and >=100 steps')
    run(a.candidates,a.seeds,a.steps,a.phase)
