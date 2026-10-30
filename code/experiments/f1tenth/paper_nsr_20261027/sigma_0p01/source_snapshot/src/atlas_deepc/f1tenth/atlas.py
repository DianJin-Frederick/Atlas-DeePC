"""Atlas-DeePC K/rank comparison using shared raw K-means/SVD and exact QP."""
from time import perf_counter
import numpy as np
import scipy.sparse as sp
import clarabel
from atlas_deepc.atlas.kmeans import fit_kmeans
from atlas_deepc.atlas.chart_selection import FullWindowProjectionSelector
from atlas_deepc.control.expert_pipeline_search import fixed_rank_left_svd_basis
from . import task
from .config import AtlasConfig as Config, ATLAS_SEARCH_RANKS, ATLAS_CLUSTER_COUNTS, TI, TF, DT, Q_PHYSICAL as Q, R_PHYSICAL as R
from .paths import OFFLINE_DATA_FILE as DATA
from .config import SELECTION_METHOD
from .partial_selection import PastWindowProjectionSelector, METHOD as PARTIAL_METHOD, PAST_ROWS, ANCHOR_ROWS
M,P=2,4

def build_library(record=None):
    record=task.collect() if record is None else record
    columns=[];features=[];meta=[]
    assert float(record['dt'])==DT
    # Same 2,700 candidate window starts as contextual (sample zero omitted).
    for e,(y,u) in enumerate(zip(record['Y'],record['U'])):
        for start in range(1,len(y)-TI-TF+1):
            aligned=task.frame(y[start:start+TI+TF],y[start+TI-1,[0,1,3]])
            columns.append(np.r_[u[start:start+TI+TF].ravel(),aligned.ravel()])
            features.append([aligned[TI-1,2]])
            meta.append([e,start])
    H=np.array(columns).T;features=np.array(features);meta=np.array(meta)
    assert H.shape==(90,2700)
    return H,features,meta


def fit(max_rank=40, clusters=2, library=None):
    if max_rank not in ATLAS_SEARCH_RANKS or clusters not in ATLAS_CLUSTER_COUNTS:
        raise ValueError('Choose supported Atlas K and rank values.')
    H, features, meta=build_library() if library is None else library
    labels,centers=fit_kmeans(features,clusters,11)
    counts=np.bincount(labels,minlength=clusters)
    if counts.min()<max_rank:raise ValueError('Insufficient chart members for requested rank')
    # These coordinates are identically zero after pose alignment. Removing
    # them during eigendecomposition preserves that identity at every rank.
    anchor=PAST_ROWS[ANCHOR_ROWS]
    if np.max(np.abs(H[anchor]))>1e-12:raise ValueError('Invalid offline pose alignment')
    active=np.setdiff1d(np.arange(H.shape[0]),anchor)
    bases=np.zeros((clusters,H.shape[0],max_rank))
    for k in range(clusters):
        bases[k,active]=fixed_rank_left_svd_basis(H[active][:,labels==k],max_rank)
    return dict(bases=bases,centers=centers,labels=labels,member_count=counts,metadata=meta)


class Atlas:
    def __init__(self,atlas,cfg=Config(),selection_method=SELECTION_METHOD):
        if (cfg.rank not in ATLAS_SEARCH_RANKS or cfg.clusters not in ATLAS_CLUSTER_COUNTS
                or cfg != Config(clusters=cfg.clusters,rank=cfg.rank)):
            raise ValueError('Vary only supported K/rank values; keep the fixed Atlas weights.')
        if atlas['bases'].shape[0]!=cfg.clusters or atlas['bases'].shape[-1]<cfg.rank:
            raise ValueError('Atlas does not contain the requested K bases/rank.')
        self.atlas=atlas;self.cfg=cfg;self.parts=[]
        cache={(k,):({'basis':b[:,:cfg.rank], 'offset':np.zeros((b.shape[0],1))},[])
               for k,b in enumerate(atlas['bases'])}
        if selection_method not in (SELECTION_METHOD, PARTIAL_METHOD):
            raise ValueError('Unknown projection selection method')
        self.selection_method=selection_method
        self.selector=(PastWindowProjectionSelector(atlas['bases'][:,:,:cfg.rank])
                       if selection_method==PARTIAL_METHOD else
                       FullWindowProjectionSelector(cache,M,P,TI+TF))
        self.selection_window=None
        for basis in atlas['bases'][:,:,:cfg.rank]:
            up=basis[:M*TI];uf=basis[M*TI:M*(TI+TF)]
            yp=basis[M*(TI+TF):M*(TI+TF)+P*TI];yf=basis[M*(TI+TF)+P*TI:]
            qw=np.tile(Q,TF);rw=np.tile(R,TF)
            hessian=2*(yf.T@(qw[:,None]*yf)+uf.T@(rw[:,None]*uf)+cfg.lam_ini*yp.T@yp)
            hessian=(hessian+hessian.T)/2
            constraints=sp.csc_matrix(np.vstack([up,uf,-uf]))
            self.parts.append((basis,up,uf,yp,yf,sp.triu(sp.csc_matrix(hessian),format='csc'),constraints))
        self.last=None

    def action(self,uh,yh,ref,_):
        started=perf_counter()
        cfg=self.cfg;pose=yh[-1,[0,1,3]]
        y=task.frame(yh[-TI:],pose).ravel();u=uh[-TI:].ravel()
        target=task.frame(ref,pose).ravel()
        # Use only observed history, with the SAME coordinate convention as
        # each offline L-window (anchor at its TI-th output, not its last).
        length=TI if self.selection_method==PARTIAL_METHOD else TI+TF
        history=np.asarray(yh[-length:])
        if len(history)!=length or len(uh)<length:
            raise ValueError('Projection selection requires L observed input/output pairs.' if length==TI+TF
                             else 'Projection selection requires T_ini observed input/output pairs.')
        aligned=task.frame(history,history[TI-1,[0,1,3]])
        window=self.selector.pack_history(uh[-length:],aligned)
        chart=self.selector.select(window)
        self.selection_window=window
        B,up,uf,yp,yf,hessian,A=self.parts[chart]
        q=-2*(yf.T@(np.tile(Q,TF)*target)+cfg.lam_ini*yp.T@y)
        bounds=np.r_[u,np.tile(task.LIMIT,TF),np.tile(task.LIMIT,TF)]
        assembly=perf_counter()-started
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
        settings.max_iter=10000;settings.tol_feas=1e-10;settings.tol_gap_abs=1e-10;settings.tol_gap_rel=1e-12
        setup=perf_counter()
        solver=clarabel.DefaultSolver(hessian,q,A,bounds,[clarabel.ZeroConeT(M*TI),clarabel.NonnegativeConeT(2*M*TF)],settings)
        setup_s=perf_counter()-setup;start=perf_counter();solution=solver.solve();solve_s=perf_counter()-start
        accepted=str(solution.status)=='Solved';g=np.array(solution.x)
        violation=max(float(np.max(abs(up@g-u))),float(np.max(abs(uf@g)-np.tile(task.LIMIT,TF))),0.) if accepted else None
        accepted=bool(accepted and np.isfinite(g).all() and violation<1e-5)
        action=np.clip((uf@g)[:2],-task.LIMIT,task.LIMIT) if accepted else None
        info=dict(status=str(solution.status),accepted=accepted,constraint_violation=violation,chart=chart,selection_assembly_s=assembly,solver_setup_s=setup_s,solve_s=solve_s,solver_reported_s=float(solution.solve_time),total_s=perf_counter()-started)
        self.last=(B,u,y,target,g)
        return action,info


