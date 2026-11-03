"""Minimum-mean SIMPLE directed path, without exhaustive path enumeration.

Bounded-length walk DP gives a lower bound. If its optimum is simple, it
certifies the answer immediately. Otherwise Dinkelbach fractional optimization
uses globally solved mixed-integer path subproblems with MTZ cycle elimination.
No search-node/time cutoff, no best-so-far fallback, no change to total cost.
All optimality claims are subject to the explicit floating-point tolerance.
"""
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def solve_minimum_mean_route(edges, start, target, tolerance=1e-7):
    started=time.perf_counter()
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Positive finite tolerance required')
    if any(not np.isfinite(w) or w < 0 for w in edges.values()):
        raise ValueError('Finite nonnegative edge weights required')
    if start==target:
        return [start],0.,dict(method='trivial',absolute_gap_bound=0.,seconds=0.)
    # Discard edges that cannot occur on a simple s-t path.
    usable={(a,b):float(w) for (a,b),w in edges.items() if a!=b and a!=target and b!=start}
    def reachable(root, reverse=False):
        reached={root};stack=[root]
        adjacency={}
        for a,b in usable:
            if reverse:a,b=b,a
            adjacency.setdefault(a,[]).append(b)
        while stack:
            for nxt in adjacency.get(stack.pop(),[]):
                if nxt not in reached:reached.add(nxt);stack.append(nxt)
        return reached
    active=reachable(start)&reachable(target,True)
    if target not in active:
        raise ValueError('No directed path to target')
    pairs=sorted((a,b) for a,b in usable if a in active and b in active)
    nodes=sorted(active);index={v:i for i,v in enumerate(nodes)};n=len(nodes);e=len(pairs)
    weights=np.array([usable[p] for p in pairs])
    aa=np.array([index[a] for a,b in pairs]);bb=np.array([index[b] for a,b in pairs])
    # Exact DP over walks of 1..n-1 edges: a relaxation of simple paths.
    distance=np.full((n,n),np.inf);previous=np.full((n,n),-1,int)
    distance[0,index[start]]=0.
    for length in range(1,n):
        for j in range(e):
            value=distance[length-1,aa[j]]+weights[j]
            if value < distance[length,bb[j]]:
                distance[length,bb[j]]=value;previous[length,bb[j]]=aa[j]
    means=distance[1:,index[target]]/np.arange(1,n)
    length=int(np.argmin(means))+1
    lower=float(means[length-1]);route=[target];cur=index[target]
    for k in range(length,0,-1):
        cur=int(previous[k,cur]);route.append(nodes[cur])
    route.reverse()
    if len(set(route))==len(route):
        return route,lower,dict(method='walk_dp_certificate',absolute_gap_bound=0.,
                               seconds=time.perf_counter()-started,milp_solves=0)
    # Binary edge decisions, continuous topological positions. Flow plus degree
    # bounds yields a path plus possible cycles; MTZ removes ALL cycles.
    ri=[];ci=[];vv=[];lbs=[];ubs=[]
    def constraint(coefficients,lo,hi):
        row=len(lbs);lbs.append(lo);ubs.append(hi)
        for col,val in coefficients:ri.append(row);ci.append(col);vv.append(val)
    for node in range(n):
        outgoing=np.flatnonzero(aa==node);incoming=np.flatnonzero(bb==node)
        rhs=1. if nodes[node]==start else (-1. if nodes[node]==target else 0.)
        constraint([(int(j),1.) for j in outgoing]+[(int(j),-1.) for j in incoming],rhs,rhs)
        constraint([(int(j),1.) for j in outgoing],0.,1.)
        constraint([(int(j),1.) for j in incoming],0.,1.)
    for j in range(e):
        constraint([(e+int(aa[j]),1.),(e+int(bb[j]),-1.),(j,float(n))],-np.inf,n-1.)
    pair_index={p:j for j,p in enumerate(pairs)}
    for j,(a,b) in enumerate(pairs):
        other=pair_index.get((b,a))
        if other is not None and j<other:
            constraint([(j,1.),(other,1.)],0.,1.)
    matrix=coo_matrix((vv,(ri,ci)),shape=(len(lbs),e+n)).tocsc()
    upper=np.r_[np.ones(e),np.full(n,n-1.)];upper[e+index[start]]=0.
    bounds=Bounds(np.zeros(e+n),upper)
    constraints=LinearConstraint(matrix,np.array(lbs),np.array(ubs))
    integrality=np.r_[np.ones(e,int),np.zeros(n,int)]
    lam=lower;iteration=0;certificates=[]
    scale=max(1.,float(weights.max()))
    while True:
        iteration+=1
        objective=np.r_[(weights-lam)/scale,np.zeros(n)]
        result=milp(objective,integrality=integrality,bounds=bounds,constraints=constraints,
                    options={'mip_rel_gap':0.,'presolve':True})
        if result.status!=0 or result.x is None:
            raise RuntimeError(f'Route MILP not solved to optimality: {result.message}')
        chosen=np.flatnonzero(result.x[:e]>.5)
        successor={pairs[j][0]:pairs[j][1] for j in chosen}
        route=[start]
        while route[-1]!=target:
            nxt=successor.get(route[-1])
            if nxt is None or nxt in route:
                raise RuntimeError('Invalid MILP path reconstruction')
            route.append(nxt)
        if len(route)-1!=len(chosen):
            raise RuntimeError('MILP returned disconnected edges')
        value=float(sum(usable[a,b] for a,b in zip(route,route[1:]))/(len(route)-1))
        # For any path of length l>=1: W/l >= lam + dual_bound/l.
        # A nonpositive dual bound is safely divided by 1; positive by n-1.
        dual=float(result.mip_dual_bound)*scale
        bound=max(lower,lam+(dual if dual<0 else dual/(n-1)))
        gap=max(0.,value-bound)
        certificates.append(dict(parameter=lam,mean=value,lower_bound=bound,
                                 absolute_gap_bound=gap,branch_nodes=int(result.mip_node_count)))
        if gap <= tolerance:
            return route,value,dict(method='fractional_milp',absolute_gap_bound=gap,
                lower_bound=bound,milp_solves=iteration,iterations=certificates,
                seconds=time.perf_counter()-started)
        if abs(lam-value)<=np.finfo(float).eps*max(1.,abs(value)):
            raise RuntimeError('MILP numerical certificate insufficient for requested tolerance')
        lam=value


def minimum_mean_path(edges,start,target,max_paths=None):
    """Compatibility interface; legacy max_paths is intentionally not a cutoff."""
    route,value,_=solve_minimum_mean_route(edges,start,target)
    return route,value
