"""Joint-NMF clustering and trajectory windows required by KDE-BDPC."""
import numpy as np


def hankel(samples, length):
    """Eq. (4), one whole recorded trajectory, time-major/channel-minor rows."""
    a = np.asarray(samples, dtype=float)
    if a.ndim != 2 or not 1 <= length <= len(a):
        raise ValueError("Invalid trajectory or Hankel depth")
    return np.stack([a[t:t+length].reshape(-1) for t in range(len(a)-length+1)], axis=1)



def nmf_objective(views, us, vs, consensus, weights):
    return float(sum(np.sum((x-u@v.T)**2)+lam*np.sum((v-consensus)**2)
                     for x,u,v,lam in zip(views,us,vs,weights)))

def joint_nmf(raw_views, config):
    """Algorithm 1: coupled multiplicative updates and consensus, not K-means."""
    c = config
    weights = np.array([c.lambda_view1, c.lambda_view2])
    if len(raw_views) != 2:
        raise ValueError("This protocol has two views")
    views = []
    for x in raw_views:
        if not np.isfinite(x).all() or np.min(x) < 0 or np.sum(x) <= 1e-12:
            raise ValueError("A view is invalid/degenerate; inspect singular spectra/cutoff, do not silently drop a view")
        # Explicit convention: entrywise L1 norm, consistent with nonnegative NMF data.
        views.append(x / x.sum())
    n = views[0].shape[1]
    if any(x.shape[1] != n for x in views) or c.clusters > n:
        raise ValueError("Incompatible views or cluster count")
    rng = np.random.default_rng(c.nmf_seed)
    us = [rng.uniform(.5, 1.5, (x.shape[0], c.clusters)) for x in views]
    us = [u/u.sum(axis=0) for u in us]
    vs = [rng.uniform(.5, 1.5, (n, c.clusters))/(n*c.clusters) for _ in views]
    consensus = np.average(np.stack(vs), axis=0, weights=weights)
    history = [nmf_objective(views, us, vs, consensus, weights)]
    converged = False
    inner_counts = []
    floor = np.finfo(float).tiny
    for outer in range(c.nmf_outer):
        counts = []
        inner_converged = []
        for view, (x, lam) in enumerate(zip(views, weights)):
            u, v = us[view], vs[view]
            previous = np.sum((x-u@v.T)**2)+lam*np.sum((v-consensus)**2)
            inner_done = False
            for inner in range(c.nmf_inner):
                # Algorithm 1 step 3, with Q=diag(column sums of U).
                numerator = x @ v + lam*np.sum(v*consensus, axis=0)[None,:]
                denominator = u @ (v.T@v) + lam*u.sum(axis=0)[None,:]*np.sum(v*v, axis=0)[None,:]
                u *= numerator/np.maximum(denominator, floor)
                q = u.sum(axis=0)
                if np.any(q <= floor):
                    raise ValueError("NMF collapsed a factor; choose a different explicit initialization")
                u /= q
                v *= q
                # Algorithm 1 step 5, after step 4 has normalized U.
                v *= (x.T@u + lam*consensus)/np.maximum(v@(u.T@u) + lam*v, floor)
                current = np.sum((x-u@v.T)**2)+lam*np.sum((v-consensus)**2)
                if abs(current-previous) <= c.nmf_tol*max(previous, 1e-15):
                    inner_done = True
                    break
                previous = current
            counts.append(inner+1)
            inner_converged.append(inner_done)
            us[view], vs[view] = u, v
        consensus = np.average(np.stack(vs), axis=0, weights=weights)
        current = nmf_objective(views, us, vs, consensus, weights)
        if current > history[-1]+1e-10*max(history[-1], 1e-12):
            raise RuntimeError("NMF objective increased; stop rather than accept an unchecked clustering")
        history.append(current)
        inner_counts.append(counts)
        if all(inner_converged) and abs(history[-1]-history[-2]) <= c.nmf_tol*max(history[-2], 1e-15):
            converged = True
            break
    return {"views": np.stack(views), "U": np.stack(us), "V": np.stack(vs),
            "consensus": consensus, "labels": np.argmax(consensus, axis=1),
            # Paper III-A: maximum consensus loading per column, NOT an arithmetic centroid.
            "representatives": np.argmax(consensus, axis=0), "objective": np.array(history),
            "inner_iterations": np.array(inner_counts), "converged": converged}
