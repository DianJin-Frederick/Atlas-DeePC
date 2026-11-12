"""Independently audit every saved projection query, chart decision and replay."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from . import task
from .runner import load_run,verify_inventory
from .config import TI,TF,WARMUP_STEPS,SELECTION_METHOD
from .paths import ROOT,PAPER_DIR
from .partial_selection import METHOD as PARTIAL_METHOD, past_matrix, RANK_RTOL


def validate(directory=PAPER_DIR):
    directory=Path(directory);protocol=json.loads((directory/'protocol.json').read_text())
    selection_method=protocol['chart_selection_method']
    assert selection_method in (SELECTION_METHOD,PARTIAL_METHOD)
    partial=selection_method==PARTIAL_METHOD
    length=TI if partial else TI+TF
    verify_inventory(directory)
    for name,digest in protocol['hashes'].items():
        if name.endswith('.py'):
            assert task.digest(directory/'source_snapshot'/name)==digest,name
        else:assert task.digest(ROOT/name)==digest,name
    geometry={};checked=0;decisions=0;worst=0.;paired=0
    noisy_record=None
    if 'noise_model' in protocol:
        model=protocol['noise_model']
        clean=(task.collect_clean() if protocol.get('input_history_convention')==task.INPUT_HISTORY_CONVENTION
               else task.collect_commanded_clean())
        noisy_record=dict(np.load(ROOT/protocol['offline_data_file']))
        scale=clean['Y'].reshape(-1,4).std(axis=0)
        np.testing.assert_array_equal(scale,model['signal_scale'])
        std=model['sigma']*scale
        np.testing.assert_array_equal(std,model['std'])
        expected=clean['Y']+np.random.default_rng(model['offline_seed']).normal(size=clean['Y'].shape)*std
        np.testing.assert_array_equal(noisy_record['Y'],expected)
        for key in ('U','X'):np.testing.assert_array_equal(noisy_record[key],clean[key])
        np.testing.assert_array_equal(protocol['offline_measurement_noise']['std'],protocol['online_measurement_noise']['std'])
    with threadpool_limits(limits=1):
        for seed in protocol['seeds']:
            common=None
            for method in protocol['methods']:
                summary=load_run(directory,method,seed);folder=directory/method/f'seed{seed}'
                replay=json.loads((folder/'collision_replay.json').read_text())
                worst=max(worst,replay['maximum_state_mismatch'])
                with np.load(folder/'rollout.npz') as trace:
                    if noisy_record is not None:
                        rng=np.random.default_rng(seed);rng.normal(0,.05,2);rng.normal(0,np.deg2rad(.6))
                        history=trace['measured_history']
                        states=np.vstack([trace['warmup_states'],trace['X'].reshape(-1,7)])
                        expected=states[:len(history),[0,1,3,4]]+rng.normal(size=history.shape)*std
                        np.testing.assert_allclose(history,expected,atol=1e-13,rtol=0)
                    values=[trace[key] for key in ('warmup_states','warmup_inputs','initial_state')]
                    values.append(trace['measured_history'][:WARMUP_STEPS])
                    if common is None:common=[x.copy() for x in values]
                    else:
                        for x,y in zip(common,values):np.testing.assert_array_equal(x,y)
                        paired+=1
                    if method.startswith('atlas'):
                        cfg=summary['config'];k,d=cfg['clusters'],cfg['rank']
                        if k not in geometry:
                            saved=np.load(directory/f'atlas_k{k}.npz');geometry[k]=saved['bases']
                            if noisy_record is not None:
                                from .atlas import fit_kmeans
                                metadata=saved['metadata']
                                features=np.array([[noisy_record['Y'][e,start+TI-1,2]] for e,start in metadata])
                                labels,centers=fit_kmeans(features,k,11)
                                np.testing.assert_array_equal(saved['labels'],labels)
                                np.testing.assert_array_equal(saved['centers'],centers)
                        bases=geometry[k][:,:,:d]
                        windows=trace['selection_windows'];residuals=trace['selection_residuals']
                        calls=json.loads((folder/'calls.json').read_text())
                        assert windows.shape==(len(calls),6*length)
                        assert residuals.shape==(len(calls),k)
                        history=trace['measured_history'];inputs=trace['input_history']
                        for i,w in enumerate(windows):
                            start=i+WARMUP_STEPS-length
                            y=history[start:start+length]
                            expected=np.r_[inputs[start+1:start+1+length].ravel(),task.frame(y,y[TI-1,[0,1,3]],**({'anchor_index':TI-1} if protocol.get('input_history_convention')==task.INPUT_HISTORY_CONVENTION else {})).ravel()]
                            np.testing.assert_array_equal(w,expected)
                        independent=[]
                        for b in bases:
                            np.testing.assert_allclose(b.T@b,np.eye(d),atol=1e-10)
                            if partial:
                                restricted=past_matrix(b)
                                coefficients=np.linalg.lstsq(restricted,windows.T,rcond=RANK_RTOL)[0]
                                independent.append(np.linalg.norm(windows.T-restricted@coefficients,axis=0))
                            else:independent.append(np.linalg.norm(windows-(windows@b)@b.T,axis=1))
                        independent=np.array(independent).T
                        np.testing.assert_allclose(residuals,independent,atol=1e-7,rtol=1e-6)
                        tolerance=100*np.finfo(float).eps*np.maximum(1.,np.linalg.norm(windows,axis=1))
                        tied=residuals<=residuals.min(axis=1)[:,None]+tolerance[:,None]
                        selected=np.argmax(tied,axis=1)
                        np.testing.assert_array_equal(selected,[c['chart'] for c in calls])
                        np.testing.assert_array_equal(tied.sum(axis=1),trace['selection_tie_count'])
                        decisions+=len(calls)
                checked+=1
    result=dict(complete=True,rollouts=checked,configurations=len(protocol['methods']),
        unsupported_configurations=len(protocol['skipped_configurations']),projection_decisions_checked=decisions,
        identical_warmup_pairings=paired,maximum_replay_state_mismatch=worst,
        chart_selection_method=(selection_method if any(m.startswith('atlas') for m in protocol['methods'])
                                else 'contextual_trajectory_selection'),warmup_steps=WARMUP_STEPS)
    task.write(directory/'validation.json',result)
    if noisy_record is not None:
        result.update(offline_and_online_noise_verified=True,clustering_uses_noisy_speed_verified=True,
                      noise_model=protocol['noise_model'])
        task.write(directory/'validation.json',result)
    return result

if __name__=='__main__':
    import sys
    print(json.dumps(validate(sys.argv[1] if len(sys.argv)>1 else PAPER_DIR),indent=2))
