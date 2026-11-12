"""Shared fixed vehicle benchmark, collector, reference and rollout timing."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from . import data, outputs
from .raceline import OriginalWaypointRaceline
from .paths import WAYPOINT_FILE, MAP_BASE, OFFLINE_DATA_FILE, CLEAN_OFFLINE_DATA_FILE, COMMANDED_OFFLINE_DATA_FILE, noisy_data_file
from .config import TI, TF, DT, LIMIT, NOISE, CLEAN_DATA_SHA256, OFFLINE_NOISE_SEED, DEFAULT_NOISE_SIGMA, WARMUP_STEPS, SELECTION_METHOD
PATH=OriginalWaypointRaceline(WAYPOINT_FILE,.05)
INPUT_HISTORY_CONVENTION = "applied_rk4_average_v1"

def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def step(state,action,*,return_applied=False):
    """Angle/acceleration command held .1 s; integrate retained ST plant in .01 s.

    Ideal steering-angle actuation is an explicit reconstruction assumption.
    Plant retains hidden yaw-rate and slip states; measured output is x,y,v,yaw.
    """
    state=state.copy();state[2]=action[1]
    if return_applied:
        applied_substeps = []
        for _ in range(10):
            state, acceleration = data.rk4_step(state,np.array([0.,action[0]]),.01,return_applied=True)
            applied_substeps.append([acceleration, state[2]])
        applied_substeps = np.asarray(applied_substeps)
        return state, applied_substeps.mean(axis=0), applied_substeps
    for _ in range(10): state=data.rk4_step(state,np.array([0.,action[0]]),.01)
    return state

def measured(state,rng,noise=True,noise_std=None):
    y=state[[0,1,3,4]].copy()
    if noise:
        std=np.asarray(noise_model()['std'] if noise_std is None else noise_std)
        y+=rng.normal(size=4)*std
    return y

def frame(y,pose,*,anchor_index=None):
    y=np.array(y,copy=True)
    dx=y[...,0]-pose[0];dy=y[...,1]-pose[1]
    c,s=np.cos(pose[2]),np.sin(pose[2])
    y[...,0]=c*dx+s*dy;y[...,1]=-s*dx+c*dy
    y[...,3]=np.unwrap(np.r_[pose[2],y[...,3].ravel()])[1:].reshape(y[...,3].shape)-pose[2]
    if anchor_index is not None:
        # Long histories can wrap through 2*pi before reaching their anchor.
        # Choose the equivalent continuous yaw branch with anchor yaw = 0.
        y[...,3] -= y[anchor_index,3]
    return y

def collect_commanded_clean(file=COMMANDED_OFFLINE_DATA_FILE):
    file=Path(file)
    if file.exists():
        if file.resolve()==COMMANDED_OFFLINE_DATA_FILE.resolve() and digest(file)!=CLEAN_DATA_SHA256:
            raise ValueError('The frozen offline dataset checksum changed.')
        return dict(np.load(file))
    file.parent.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(3407);xs=[];ys=[];us=[]
    # Independent free-space episodes; windows never cross reset boundaries.
    for episode in range(20):
        state=np.zeros(7);state[3]=rng.uniform(4,8)
        xx=[];yy=[];uu=[];steer=0.
        for t in range(150):
            a=rng.uniform(-3.,3.)
            steer=.8*steer+np.sqrt(1-.8**2)*rng.normal(0,.12)
            u=np.array([a,np.clip(steer,-.3,.3)])
            xx.append(state.copy());yy.append(measured(state,rng,noise=False));uu.append(u)
            state=step(state,u)
            if not np.isfinite(state).all():raise ValueError('nonfinite offline state')
        xs.append(xx);ys.append(yy);us.append(uu)
    result=dict(X=np.array(xs),Y=np.array(ys),U=np.array(us),dt=np.array(DT))
    np.savez_compressed(file,**result)
    write(file.with_name(file.stem+'_manifest.json'),{'sha256':digest(file),'seconds':300,'samples':3000,'episodes':20,'episode_steps':150,'seed':3407,'collection':'free-space random uniform acceleration [-3,3], OU steering std .12 rad clipped +/-.3; initial speed uniform [4,8]; clean training data','speed_range':[float(result['X'][:,:,3].min()),float(result['X'][:,:,3].max())]})
    return result


def collect_clean(file=CLEAN_OFFLINE_DATA_FILE):
    """Same offline states/outputs, with applied inputs reconstructed by plant replay.

    U contains the average applied acceleration and held steering angle over each
    control interval. Raw commands and all internal-step applied inputs are kept.
    Averaging captures the exact longitudinal increment, but does not claim that
    holding the average would reproduce every nonlinear state transition.
    """
    file=Path(file)
    if file.resolve()==COMMANDED_OFFLINE_DATA_FILE.resolve():
        raise ValueError('Applied-input data must not overwrite the original commands.')
    source=collect_commanded_clean()
    signature=dict(input_history_convention=INPUT_HISTORY_CONVENTION,
        source_sha256=digest(COMMANDED_OFFLINE_DATA_FILE),
        integrator_sha256=digest(Path(data.__file__)),
        dynamics_sha256=digest(Path(data.vehicle_dynamics_st.py_func.__code__.co_filename)))
    manifest=file.with_name(file.stem+'_manifest.json')
    if file.exists():
        if not manifest.exists():raise ValueError('Applied-input dataset manifest missing')
        saved_manifest=json.loads(manifest.read_text())
        if saved_manifest.get('signature')!=signature or digest(file)!=saved_manifest['sha256']:
            raise ValueError('Applied-input dataset provenance changed; use a new version/path.')
        return dict(np.load(file))
    applied=np.empty_like(source['U']);substeps=np.empty((*applied.shape[:2],10,2))
    next_states=np.empty_like(source['X']);worst=0.
    for episode in range(len(source['X'])):
        for i,(state,command) in enumerate(zip(source['X'][episode],source['U'][episode])):
            nxt,u,internal=step(state,command,return_applied=True)
            applied[episode,i]=u;substeps[episode,i]=internal;next_states[episode,i]=nxt
            if i+1<len(source['X'][episode]):
                mismatch=nxt-source['X'][episode,i+1]
                mismatch[4]=outputs.wrap_angle(mismatch[4])
                worst=max(worst,float(np.max(abs(mismatch))))
    if worst>=1e-9:raise ValueError('Offline command replay does not reproduce recorded states')
    np.testing.assert_allclose(next_states[...,3]-source['X'][...,3],DT*applied[...,0],atol=1e-12,rtol=0)
    result=dict(source,U=applied,U_applied=applied,U_commanded=source['U'],
                applied_substeps=substeps,X_next=next_states,
                input_history_convention=np.array(INPUT_HISTORY_CONVENTION))
    file.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(file,**result)
    write(manifest,dict(signature=signature,sha256=digest(file),maximum_state_mismatch=worst,
        changed_acceleration_samples=int(np.count_nonzero(abs(applied[...,0]-source['U'][...,0])>1e-10)),
        samples=int(np.prod(applied.shape[:2])),
        description='Applied acceleration is RK4-weighted at each .01 s substep, averaged over .1 s; steering is the actual held angle. Clean outputs and states unchanged.'))
    return result


def noise_model(sigma=DEFAULT_NOISE_SIGMA):
    sigma=float(sigma)
    if not np.isfinite(sigma) or sigma<0:raise ValueError('sigma must be finite and nonnegative')
    scale=collect_clean()['Y'].reshape(-1,4).std(axis=0,ddof=0)
    if np.any(scale<=0):raise ValueError('Output signal scale must be positive')
    return dict(sigma=sigma,signal_scale=scale.tolist(),std=(sigma*scale).tolist(),
        definition='sigma = noise standard deviation / clean offline channel standard deviation (population std, global x/y/speed/unwrapped yaw); same fixed physical std offline and online',
        noise_to_signal_power_ratio=sigma**2,snr_db=None if sigma==0 else float(-20*np.log10(sigma)),
        offline_seed=OFFLINE_NOISE_SEED,clean_source_sha256=digest(CLEAN_OFFLINE_DATA_FILE))


def collect(file=None,sigma=DEFAULT_NOISE_SIGMA):
    """One fixed noisy observation per physical sample, before window extraction.

    Clean states, commands and applied inputs remain available for provenance; both
    controllers and clustering consume Y, never the clean output array.
    """
    file=noisy_data_file(sigma) if file is None else Path(file)
    if file.resolve()==CLEAN_OFFLINE_DATA_FILE.resolve():
        raise ValueError('The clean source must not be overwritten by noisy observations')
    clean=collect_clean()
    model=noise_model(sigma);std=np.array(model['std'])
    noise=np.random.default_rng(OFFLINE_NOISE_SEED).normal(size=clean['Y'].shape)*std
    expected=dict(clean,Y=clean['Y']+noise,
                  noise_std=std,noise_seed=np.array(OFFLINE_NOISE_SEED),
                  noise_sigma=np.array(sigma),signal_scale=np.array(model['signal_scale']))
    if file.exists():
        saved=dict(np.load(file))
        if set(saved)!=set(expected) or any(not np.array_equal(saved[k],v) for k,v in expected.items()):
            raise ValueError('Offline observations do not match the frozen noise model/seed')
        return saved
    file.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(file,**expected)
    write(file.with_name(file.stem+'_manifest.json'),dict(sha256=digest(file),
        clean_source=str(CLEAN_OFFLINE_DATA_FILE),clean_source_sha256=digest(CLEAN_OFFLINE_DATA_FILE),
        seconds=300,samples=3000,episodes=20,episode_steps=150,collection_seed=3407,
        measurement_noise_seed=OFFLINE_NOISE_SEED,measurement_noise_std=std.tolist(),noise_model=model,
        noise_description='Independent additive zero-mean Gaussian output noise, identical standard deviations to online measurements. Added once per sample in global x/y/speed/yaw coordinates before alignment, clustering and overlapping window extraction. Commands retained separately; U contains applied inputs consistently offline and online. True states unchanged.'))
    return expected

def reference(progress):
    cursor=progress*PATH.sample_spacing;ys=[]
    for _ in range(TF):
        i=round(cursor/PATH.sample_spacing)%PATH.n
        ys.append([*PATH.xy[i],PATH.source_speed[i],PATH.psi[i]])
        cursor+=PATH.source_speed[i]*DT
    return np.array(ys)


def timing(t):
    t=np.array(t)*1000
    if not len(t):return None
    return dict(median_ms=float(np.median(t)),p99_ms=float(np.percentile(t,99)),max_ms=float(t.max()),calls=len(t))


def warmup_action(observation,progress):
    """Common reference follower ONLY for collecting L measured history pairs."""
    index=progress%PATH.n
    target=PATH.xy[(progress+round(1.5/PATH.sample_spacing))%PATH.n]
    delta=target-observation[:2]
    lateral=-np.sin(observation[3])*delta[0]+np.cos(observation[3])*delta[1]
    steering=np.arctan2(2*PATH.wheelbase*lateral,max(float(delta@delta),1e-8))
    acceleration=2*(PATH.source_speed[index]-observation[2])
    return np.clip([acceleration,steering],-LIMIT,LIMIT)


def rollout(lib,cfg,seed,steps=600,noise=True,save=None,noise_sigma=DEFAULT_NOISE_SIGMA):
    selector=getattr(lib,'selector',None)
    if selector is not None:selector.reset()
    rng=np.random.default_rng(seed)
    model=noise_model(noise_sigma);noise_std=np.array(model['std'])
    state=PATH.initial_state(0,float(PATH.source_speed[0]))
    state[:2]+=rng.normal(0,.05,2);state[4]+=rng.normal(0,np.deg2rad(.6))
    uh=[np.zeros(2)];command_history=[np.zeros(2)];yh=[];warmup_states=[];warmup_inputs=[];warmup_commands=[];warmup_substeps=[];warmup_progress=0
    # Identical initialization for both methods; never used after handoff.
    for _ in range(WARMUP_STEPS):
        obs=measured(state,rng,noise,noise_std)
        warmup_progress=PATH.localize(obs[:2],warmup_progress)
        u=warmup_action(obs,warmup_progress)
        warmup_states.append(state.copy());warmup_commands.append(u.copy())
        state,applied,internal=step(state,u,return_applied=True)
        warmup_inputs.append(applied.copy());warmup_substeps.append(internal)
        yh.append(obs);uh.append(applied.copy());command_history.append(u.copy())
    initial=progress=PATH.localize(state[:2],0);rows=[];xx=[];uu=[];commands=[];substeps=[];refs=[];error=[];spe=[];failure=None
    initial_state=state.copy();windows=[]
    for t in range(steps):
        start=perf_counter();obs=measured(state,rng,noise,noise_std)
        measured_progress=PATH.localize(obs[:2],progress)
        ref=reference(measured_progress)
        action,info=lib.action(np.array(uh[-WARMUP_STEPS-1:]),np.array(yh[-WARMUP_STEPS:]),ref,cfg)
        info['total_s']+=perf_counter()-start-info['total_s']
        rows.append(info)
        if selector is not None:windows.append(lib.selection_window.copy())
        if action is None:failure=info['status'];break
        xy=state[:2]-PATH.xy[progress%PATH.n];normal=np.array([-np.sin(PATH.psi[progress%PATH.n]),np.cos(PATH.psi[progress%PATH.n])])
        error.append(float(normal@xy));spe.append(float(state[3]-ref[0,2]));xx.append(state.copy());commands.append(action.copy());refs.append(ref)
        state,applied,internal=step(state,action,return_applied=True)
        uu.append(applied.copy());substeps.append(internal)
        yh.append(obs);uh.append(applied.copy());command_history.append(action.copy())
        info.update(commanded_input=action.tolist(),applied_input=applied.tolist())
        if not np.isfinite(state).all():failure='nonfinite_plant';break
        progress=PATH.localize(state[:2],progress)
        if np.linalg.norm(state[:2]-PATH.xy[progress%PATH.n])>1.:failure='track_corridor_exit';break
        if progress-initial>=PATH.n:break
    completed=bool(progress-initial>=PATH.n and failure is None)
    result=dict(config=asdict(cfg),seed=seed,noise=noise,steps=len(uu),lap_completed=completed,termination=failure or ('lap_completed' if completed else 'step_budget'),lap_fraction=float((progress-initial)/PATH.n),rmse_cm=100*float(np.sqrt(np.mean(np.array(error)**2))) if error else None,speed_rmse_mps=float(np.sqrt(np.mean(np.array(spe)**2))) if spe else None,timing=timing([r['total_s'] for r in rows[5:]]),all_calls_timing=timing([r['total_s'] for r in rows]),solve_failures=sum(not r['accepted'] for r in rows),collision_verified=False)
    result.update(warmup_steps=WARMUP_STEPS,chart_selection_method=getattr(lib,'selection_method',SELECTION_METHOD) if selector is not None else 'contextual_trajectory_selection')
    result.update(noise_sigma=noise_sigma,measurement_noise_std=noise_std.tolist(),
                  input_history_convention=INPUT_HISTORY_CONVENTION)
    if save:
        save=Path(save);save.mkdir(parents=True,exist_ok=True)
        write(save/'summary.json',result);write(save/'calls.json',rows)
        extra=dict(selection_windows=np.array(windows),selection_residuals=np.array(selector.residual_history),
                   selection_tie_count=np.array(selector.tie_history)) if selector is not None else {}
        np.savez_compressed(save/'rollout.npz',X=np.array(xx),U=np.array(uu),
                            U_applied=np.array(uu),U_commanded=np.array(commands),applied_substeps=np.array(substeps),
                            input_history_convention=np.array(INPUT_HISTORY_CONVENTION),reference=np.array(refs),cross_track_error=np.array(error),final_state=state,
                            initial_state=initial_state,measured_history=np.array(yh),input_history=np.array(uh),command_history=np.array(command_history),
                            warmup_commanded_inputs=np.array(warmup_commands),warmup_applied_substeps=np.array(warmup_substeps),
                            warmup_states=np.array(warmup_states),warmup_inputs=np.array(warmup_inputs),**extra)
    return result


