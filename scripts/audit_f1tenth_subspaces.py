"""Read-only geometry audit of saved F1TENTH charts; no controller rollouts."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/f1tenth/subspace_audit_20261111'
CASES=[dict(name='paper_tini5',label='Paper: Ti=5, Tf=10, K=12, d=40, σ=0.001',ti=5,tf=10,
    atlas='experiments/f1tenth/paper_nsr_20261027/sigma_0p001/atlas_k12.npz',
    traces='experiments/f1tenth/paper_nsr_20261027/sigma_0p001/atlas_k12_d40'),
 dict(name='trial_tini8',label='Trial: Ti=8, Tf=10, K=13, d=44, σ=0',ti=8,tf=10,
    atlas='experiments/f1tenth/history_projection_trials/ti8_tf10_k13_d44_sigma0_seed542_2ffb750ed9ec/atlas.npz',
    traces='experiments/f1tenth/history_projection_trials/ti8_tf10_k13_d44_sigma0_seed542_2ffb750ed9ec/rollout')]

def distances(bases):
    projectors=[q@q.T for q in bases]
    return np.array([[np.linalg.norm(a-b,'fro')/np.sqrt(2) for b in projectors] for a in projectors])

def stats(values):
    v=np.asarray(values)
    return dict(min=float(v.min()),median=float(np.median(v)),max=float(v.max()),count=int(v.size))

def audit(c):
    ti,tf=c['ti'],c['tf']
    with np.load(ROOT/c['atlas']) as saved:b=saved['bases'].copy()
    k,ambient,d=b.shape
    for basis in b:np.testing.assert_allclose(basis.T@basis,np.eye(d),atol=1e-12)
    rows=np.r_[0:2*ti,2*(ti+tf):2*(ti+tf)+4*ti]
    anchor=np.array([6*ti-4,6*ti-3,6*ti-1]);active=np.setdiff1d(np.arange(6*ti),anchor)
    bases=[];ranks=[]
    for basis in b:
        assert np.max(abs(basis[rows][anchor]))<1e-12
        u,s,_=np.linalg.svd(basis[rows][active],full_matrices=False)
        rank=int(np.count_nonzero(s>1e-10*s[0]));q=np.zeros((6*ti,rank));q[active]=u[:,:rank]
        bases.append(q);ranks.append(rank)
    full=distances(b);past=distances(bases);ii,jj=np.triu_indices(k,1)
    result=dict(label=c['label'],input_history_convention='commanded',full_basis_shape=[ambient,d],
        past_basis_shape=[6*ti,d],effective_history_dimension=6*ti-3,history_ranks=ranks,
        full_pairwise_distance=stats(full[ii,jj]),past_pairwise_distance=stats(past[ii,jj]),
        definition='||Qi Qi^T - Qj Qj^T||_F / sqrt(2), orthonormal bases; history rank cutoff 1e-10 relative to largest singular value.')
    groups=[]
    for rank1 in sorted(set(ranks)):
        for rank2 in sorted(set(ranks)):
            if rank2<rank1:continue
            pairs=[(i,j) for i,j in zip(ii,jj) if sorted([ranks[i],ranks[j]])==[rank1,rank2]]
            if pairs:groups.append(dict(rank_pair=[rank1,rank2],**stats([past[i,j] for i,j in pairs])))
    result['history_distance_by_rank_pair']=groups
    if ti==8:
        C=np.zeros((ti-1,6*ti))
        for j in range(ti-1):C[j,2*j]=-.1;C[j,2*ti+4*j+2]=-1;C[j,2*ti+4*(j+1)+2]=1
        A=np.vstack((C,np.eye(6*ti)[anchor]));_,_,vh=np.linalg.svd(A,full_matrices=True);q0=vh[len(A):].T
        result['common_speed_relation_subspace']=dict(dimension=q0.shape[1],
            containment_residual_per_chart=[float(np.linalg.norm(q0-q@(q.T@q0))) for q in bases],
            explanation='Anchor x/y/yaw = 0 and seven relations v[j+1]-v[j]=0.1*a_command[j]. These hold when commanded acceleration is not modified by the plant.')
    trace_root=ROOT/c['traces']
    paths=sorted(trace_root.glob('seed*/rollout.npz')) if ti==5 else [trace_root/'rollout.npz']
    entries=[]
    for path in paths:
        with np.load(path) as saved:
            w=saved['selection_windows'];reported=saved['selection_residuals']
            recalculated=np.column_stack([np.linalg.norm(w-w@q@q.T,axis=1) for q in bases])
            np.testing.assert_allclose(recalculated,reported,atol=1e-10,rtol=1e-6)
            item=dict(trace=str(path.relative_to(ROOT)),decisions=len(w),
                histories_fitting_every_chart=int(np.sum(np.max(reported,axis=1)<1e-8)),
                maximum_residual=float(np.max(reported)))
            if ti==8:
                group=np.flatnonzero(np.array(ranks)==38)
                defect=np.max(abs(w@C.T),axis=1)
                item.update(histories_satisfying_speed_relations=int(np.sum(defect<1e-9)),
                    histories_fitting_all_rank38_charts=int(np.sum(np.max(reported[:,group],axis=1)<1e-8)),
                    rank38_chart_ids_one_based=(group+1).tolist(),
                    histories_fitting_at_least_11_charts=int(np.sum(np.sum(reported<1e-8,axis=1)>=11)))
            entries.append(item)
    result['query_checks']=entries
    result['total_decisions']=sum(r['decisions'] for r in entries)
    result['total_histories_fitting_every_chart']=sum(r['histories_fitting_every_chart'] for r in entries)
    result['source_hashes']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [ROOT/c['atlas'],*paths]}
    return result,full,past


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    results=[];matrices={}
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    for row,c in enumerate(CASES):
        result,full,past=audit(c);results.append(result)
        matrices[c['name']+'_full']=full;matrices[c['name']+'_history']=past
        for col,(matrix,title,upper) in enumerate([(full,'Full trajectory subspaces',4),(past,'History subspaces used for selection',2)]):
            ax=axes[row,col];k=len(matrix)
            im=ax.imshow(matrix,origin='lower',vmin=0,vmax=upper,cmap='viridis',extent=(.5,k+.5,.5,k+.5))
            ax.set(xlabel='Chart (1-based)',ylabel='Chart (1-based)',xticks=np.arange(1,k+1),yticks=np.arange(1,k+1))
            ax.set_title(c['label']+'\n'+title,fontsize=10)
            fig.colorbar(im,ax=ax,label='Projector distance',shrink=.82)
    fig.savefig(OUT/'subspace_distances.png',dpi=180)
    fig.savefig(OUT/'subspace_distances.pdf')
    plt.close(fig)
    np.savez_compressed(OUT/'pairwise_distances.npz',**matrices)
    (OUT/'audit.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='source_hashes'} for r in results],indent=2))

if __name__=='__main__':main()
