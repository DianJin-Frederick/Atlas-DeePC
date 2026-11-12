"""Replay the angle-command plant at every internal .01 s step with Gym map checks."""
import argparse
import json
from types import MethodType
import numpy as np
from . import task as r
from f110_gym.envs.base_classes import Integrator
from f110_gym.envs.f110_env import F110Env


def verify(directory):
    directory=r.Path(directory)
    trace=dict(np.load(directory/'rollout.npz'))
    warmup=len(trace['warmup_states'])
    states=np.concatenate([trace['warmup_states'],trace['X'].reshape(-1,7)])
    # New histories store applied inputs, but exact plant replay needs commands.
    warmup_commands=trace.get('warmup_commanded_inputs',trace['warmup_inputs'])
    control_commands=trace.get('U_commanded',trace['U'])
    commands=np.concatenate([warmup_commands,control_commands.reshape(-1,2)])
    if not len(states):
        result=dict(verified=False,collision_free=False,reason='empty rollout',maximum_state_mismatch=0.)
        r.write(directory/'collision_replay.json',result)
        return result
    summary=json.loads((directory/'summary.json').read_text())
    env=F110Env(map=str(r.MAP_BASE),map_ext='.png',num_agents=1,timestep=.01,integrator=Integrator.RK4,seed=summary['seed'])
    params=dict(env.sim.agents[0].params);params.update(r.data.VEHICLE_PARAMS);env.update_params(params)
    agent=env.sim.agents[0]
    def advance(car,acceleration,steering_angle):
        car.state[2]=steering_angle
        car.state=r.data.rk4_step(car.state,np.array([0.,acceleration]),.01)
        pose=np.array([car.state[0]+car.lidar_dist*np.cos(car.state[4]),car.state[1]+car.lidar_dist*np.sin(car.state[4]),car.state[4]])
        return car.scan_simulator.scan(pose,car.scan_rng)
    agent.update_pose=MethodType(advance,agent)
    env.reset(np.array([[states[0,0],states[0,1],states[0,4]]]))
    agent.state[:]=states[0];agent.in_collision=False;env.collisions[:]=0;env.sim.collisions[:]=0
    mismatch=0.;hit=None;count=0
    try:
        for i,u in enumerate(commands):
            difference=agent.state-states[i];difference[4]=r.outputs.wrap_angle(difference[4])
            mismatch=max(mismatch,float(np.max(abs(difference))))
            for j in range(10):
                obs,*_=env.step(u.reshape(1,2));count+=1
                if obs['collisions'][0]:hit=[i,j];break
            if hit is not None:break
        if hit is None:
            difference=agent.state-trace['final_state'];difference[4]=r.outputs.wrap_angle(difference[4])
            mismatch=max(mismatch,float(np.max(abs(difference))))
    finally:env.close()
    result=dict(verified=hit is None and mismatch<1e-9,collision_free=hit is None,
        collision_control_and_substep=None if hit is None else [hit[0]-warmup,hit[1]],
        collision_phase=None if hit is None else ('warmup' if hit[0]<warmup else 'controller'),
        warmup_steps_replayed=min(count,warmup*10),internal_steps_replayed=count,maximum_state_mismatch=mismatch)
    r.write(directory/'collision_replay.json',result)
    summary['collision_verified']=result['verified'];summary['verified_success']=bool(summary['lap_completed'] and result['verified'])
    r.write(directory/'summary.json',summary)
    return result

