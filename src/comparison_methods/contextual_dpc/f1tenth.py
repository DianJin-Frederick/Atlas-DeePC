"""Contextual-DPC local reconstruction with 30, 60 or 90 selected columns."""
from time import perf_counter
import numpy as np
import scipy.sparse as sp
import clarabel
from comparison_methods.contextual_dpc.reacher import Library as ReacherLibrary
from dataclasses import dataclass
from atlas_deepc.f1tenth.config import TI, TF, LIMIT, Q_PHYSICAL, R_PHYSICAL
from atlas_deepc.f1tenth.task import frame

@dataclass(frozen=True)
class Config:
    columns: int = 30
    lambda_g: float = 10.
    lambda_sigma: float = 1.
    mode: str = 'clarabel_written_g'
    weights: str = 'position'

class Library:
    select=ReacherLibrary.select
    def __init__(self,record):
        self.config=Config()
        zz=[];dd=[];meta=[]
        for e,(y,u) in enumerate(zip(record['Y'],record['U'])):
            for start in range(1,len(y)-TI-TF+1):
                anchor=y[start+TI-1,[0,1,3]]
                aligned=frame(y[start:start+TI+TF],anchor)
                zz.append(np.c_[aligned,u[start:start+TI+TF]])
                dd.append(u[start:start+TI+TF]-u[start-1:start+TI+TF-1]);meta.append([e,start])
        z=np.array(zz);du=np.array(dd)
        self.z_mean=z.mean(axis=(0,1));self.z_scale=np.maximum(z.std(axis=(0,1)),1e-8)
        self.du_mean=np.diff(record['U'],axis=1).mean(axis=(0,1))
        self.du_scale=np.maximum(np.diff(record['U'],axis=1).std(axis=(0,1)),1e-8)
        self.u_mean=self.z_mean[4:];self.u_scale=self.z_scale[4:]
        zn=(z-self.z_mean)/self.z_scale;dn=(du-self.du_mean)/self.du_scale
        self.zp=zn[:,:TI].reshape(len(z),-1).T.copy();self.zf=zn[:,TI:].reshape(len(z),-1).T.copy()
        self.dp=dn[:,:TI].reshape(len(z),-1).T.copy()
        self.duf=(du[:,TI:]/self.u_scale).reshape(len(z),-1).T.copy()
        self.uf=z[:,TI:,4:].reshape(len(z),-1).T.copy()
        self.un=zn[:,TI:,4:].reshape(len(z),-1).T.copy()
        self.up_last=zn[:,TI-1,4:].T.copy()
        self.Q=np.r_[Q_PHYSICAL*self.z_scale[:4]**2,0.,0.]
        self.R=R_PHYSICAL*self.u_scale**2
        self.future_weight=np.tile(self.Q,TF)
        self.past_norm2=np.sum(self.zp**2,axis=0)
        self.future_norm2=np.sum(self.zf**2*self.future_weight[:,None],axis=0)
        self.meta=np.array(meta)

    def problem_arrays(self,uh,yh,ref,cfg):
        self.config=cfg
        pose=yh[-1,[0,1,3]]
        past=(np.c_[frame(yh[-TI:],pose),uh[-TI:]]-self.z_mean)/self.z_scale
        target=(np.c_[frame(ref,pose),np.zeros((TF,2))]-self.z_mean)/self.z_scale
        dp=((np.diff(uh[-TI-1:],axis=0)-self.du_mean)/self.du_scale).ravel()
        indices,_=self.select(past.ravel(),target.ravel())
        F=self.zf[:,indices];Z=self.zp[:,indices];U=self.un[:,indices];D=self.duf[:,indices]
        W=self.future_weight;wr=np.tile(self.R,TF)
        hessian=2*(F.T@(W[:,None]*F)+U.T@(wr[:,None]*U)+D.T@(wr[:,None]*D))
        linear=-2*F.T@(W*target.ravel())
        # Enforce past last input exactly. Combined with sum(g)=1 and the
        # offline difference identity this enforces all future cumulative inputs.
        equality=np.vstack([self.dp[:,indices],np.ones((1,cfg.columns)),self.up_last[:,indices]])
        rhs=np.r_[dp,1.,(uh[-1]-self.u_mean)/self.u_scale]
        return indices,hessian,linear,Z,past.ravel(),self.uf[:,indices],equality,rhs

    def action(self,uh,yh,ref,cfg):
        if cfg.columns not in (30,60,90) or cfg != Config(columns=cfg.columns):
            raise ValueError('Use 30, 60 or 90 columns with the fixed contextual weights and penalties.')
        start=perf_counter()
        ix,P,q,Z,z,U,E,b=self.problem_arrays(uh,yh,ref,cfg)
        selection_assembly=perf_counter()-start
        n=cfg.columns;ns=TI*6
        bound=np.tile(LIMIT,TF)
        setup=perf_counter()
        # Variables g and epigraph s for ||Zg-z||_1; optional norm epigraph t.
        eq=np.c_[E,np.zeros((len(E),ns))]
        inequalities=np.vstack([np.c_[U,np.zeros((len(U),ns))],np.c_[-U,np.zeros((len(U),ns))],np.c_[Z,-np.eye(ns)],np.c_[-Z,-np.eye(ns)]])
        limits=np.r_[bound,bound,z,-z]
        h=sp.block_diag([P,sp.csc_matrix((ns+1,ns+1))],format='csc')
        linear=np.r_[q,np.full(ns,cfg.lambda_sigma),cfg.lambda_g]
        a0=np.c_[eq,np.zeros(len(eq))];a1=np.c_[inequalities,np.zeros(len(inequalities))]
        soc=np.zeros((n+1,n+ns+1));soc[0,-1]=-1;soc[1:,:n]=-np.eye(n)
        A=sp.csc_matrix(np.vstack([a0,a1,soc]));bb=np.r_[b,limits,np.zeros(n+1)]
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1;settings.tol_gap_abs=1e-6;settings.tol_gap_rel=1e-6;settings.tol_feas=1e-6
        solver=clarabel.DefaultSolver(sp.triu(h,format='csc'),linear,A,bb,[clarabel.ZeroConeT(len(eq)),clarabel.NonnegativeConeT(len(limits)),clarabel.SecondOrderConeT(n+1)],settings)
        setup_s=perf_counter()-setup;t=perf_counter();sol=solver.solve();solve_s=perf_counter()-t
        status=str(sol.status);accepted=status=='Solved';value=np.asarray(sol.x)
        iterations=sol.iterations;solver_reported=sol.solve_time
        info=dict(status=status,accepted=accepted,selection_assembly_s=selection_assembly,solver_setup_s=setup_s,solve_s=solve_s,solver_reported_s=solver_reported,iterations=int(iterations))
        if accepted:
            g=value[:n];residual=max(float(np.max(np.abs(E@g-b))),float(np.max(np.abs(U@g)-bound)),0.)
            info['constraint_violation']=residual
            if not np.isfinite(g).all() or residual>1e-3:info['accepted']=False;info['status']='rejected_residual'
            action=np.clip((U@g)[:2],-LIMIT,LIMIT)
        else:action=None;info['constraint_violation']=None
        info['total_s']=perf_counter()-start
        return action if info['accepted'] else None,info


