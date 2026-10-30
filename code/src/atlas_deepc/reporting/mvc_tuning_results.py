"""Read-only experiment audit and reproducible weight-tuning diagnostic report."""
import json
from dataclasses import asdict

import numpy as np

from atlas_deepc.benchmarks import mvc_bdpc_reacher as m
from atlas_deepc.benchmarks import tune_mvc_bdpc as tuning


def export():
    cache=tuning.OFFLINE/'offline_model.npz'
    original=json.loads((tuning.OFFLINE/'manifest.json').read_text())
    config=m.Config(**original['config'])
    library=m.Library.load(cache,config)
    records=[]
    phases=['first_seed','intermediate','transition','full_length_crosscheck']
    for phase in phases:
        for manifest_path in sorted((tuning.ROOT/phase).glob('*/manifest.json')):
            directory=manifest_path.parent
            manifest=json.loads(manifest_path.read_text())
            for key,value in asdict(config).items():
                if key not in ['q_xy','r','lambda_epsilon']:
                    assert manifest['config'][key]==value, (directory,key)
            assert manifest['controller_sha256']==original['source_sha256']==m.utils.digest(m.__file__)
            assert manifest['offline_cache_sha256']==m.utils.digest(cache)
            for seed in manifest['seeds']:
                path=directory/f'seed{seed}_control_results.npz'
                if not path.exists():continue
                with np.load(path) as a:
                    row=json.loads(a['summary'].item())
                    records.append({'phase':phase,'candidate':directory.name,'seed':seed,
                        **{k:manifest['config'][k] for k in ['q_xy','r','lambda_epsilon']},
                        'requested_steps':row['requested_steps'],'completed_steps':row['completed_steps'],
                        'strict_success':row['strict_success'],'failure':row['failure'],
                        'mean_final100_mm':1000*row['mean_last_100'] if row['complete'] else None,
                        'saturation_fraction':row['saturated_steps']/row['completed_steps'] if row['completed_steps'] else None,
                        'mean_action_norm':float(np.mean(np.linalg.norm(a['u'],axis=1))) if row['completed_steps'] else None,
                        'mean_prediction_mm':float(np.mean(a['prediction_error_xy'])*1000) if row['completed_steps'] else None,
                        'source':str(path)})
    geometry=[]
    for j,model in enumerate(library.models):
        blocks=model.reshape(config.past+1,4,-1)
        past=blocks[:-1].reshape(4*config.past,-1)
        future_u=blocks[-1,:2]
        _,s,vt=np.linalg.svd(past,full_matrices=False)
        rank=int(np.sum(s>1e-8))
        row_basis=vt[:rank]
        unrepresented=future_u-(future_u@row_basis.T)@row_basis
        geometry.append({'cluster':j,'representative':int(library.representatives[j]),
                         'past_rank':rank,'past_plus_future_u_rank':int(np.linalg.matrix_rank(np.vstack([past,future_u]),tol=1e-8)),
                         'future_input_nullspace_relative_norm':float(np.linalg.norm(unrepresented)/np.linalg.norm(future_u)),
                         'rank_absolute_tolerance':1e-8})
    payload={'selected_candidate':None,'working_parameter_set_found':False,
             'unique_weight_sets':len(set(r['candidate'] for r in records)),
             'finished_rollouts':len(records),'successful_rollouts':sum(r['strict_success'] for r in records),
             'aborted_rollouts':sum(r['failure'] is not None for r in records),
             'records':records,'model_geometry':geometry,
             'scope':'Only q_xy, r and lambda_epsilon were changed; offline models, selection, horizon, solver and formulation stayed fixed.',
             'evaluation':'Seed 211 pilots: 200 steps. Cross-check seeds 222/258: 320 steps. Not a 100-seed success-rate estimate.',
             'diagnosis':'In all five current representative models, future input rows lie in the row space of past input/output rows. Fixing the past therefore fixes the future input; output slack is the route for changing commands. Weight changes do not restore these missing independent input directions.',
             'limitations':'This diagnoses the current equation-based Reacher port with its disclosed SVD/model conventions, not an impossibility theorem or a failure claim about the authors method. No candidate was promoted. Do not replace the controller with a modified formulation without reviewing those conventions.',
             'next_step':'Audit the center-relative model construction and rank/structure preservation against the paper before further control-weight search.'}
    m.utils.write_json(tuning.ROOT/'tuning_report.json',payload)
    return payload


if __name__=='__main__':
    result=export()
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','model_geometry']},indent=2))
