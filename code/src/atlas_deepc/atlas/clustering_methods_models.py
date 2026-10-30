"""Offline chart gates; embeddings affect membership only, never DeePC outputs."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import numpy as np
from scipy import sparse
from scipy.special import logsumexp
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors

METHODS = {
    "kmeans": {"label": "Baseline K-means", "assignment": "nearest baseline centroid", "dimension": 6},
    "kflats": {"label": "Affine K-subspaces", "assignment": "minimum affine-plane residual", "dimension": 2, "max_iter": 50},
    "mppca": {"label": "Mixture of PPCA", "assignment": "maximum posterior PPCA component", "dimension": 2, "max_iter": 100, "variance_floor": 1e-5, "tol": 1e-5},
    "spectral": {"label": "Spectral clustering", "assignment": "sparse-kernel Nystrom extension, row normalization, embedding K-means", "dimension": 48, "landmarks": 1500, "neighbors": 20},
    "diffusion": {"label": "Diffusion maps + K-means", "assignment": "density-normalized Nystrom diffusion coordinates, embedding K-means", "dimension": 4, "landmarks": 1500, "neighbors": 20, "alpha": 1, "diffusion_time": 1},
    "isomap": {"label": "Isomap + K-means", "assignment": "sklearn Isomap.transform, embedding K-means", "dimension": 4, "landmarks": 1500, "neighbors": 20},
    "umap": {"label": "UMAP + K-means", "assignment": "native UMAP.transform per point with fixed seed, embedding K-means", "dimension": 4, "landmarks": 1500, "neighbors": 20, "min_dist": .1, "epochs": 200},
}
SEED = 11
LANDMARK_SEED = 1701
K = 48


def minibatch(X):
    return MiniBatchKMeans(n_clusters=K, random_state=SEED, n_init=5,
        batch_size=2048, max_iter=300, reassignment_ratio=.002).fit(X)


def plane_scores(X, centers, bases):
    delta = X[:, None, :] - centers[None, :, :]
    return np.maximum(np.sum(delta**2, axis=2) - np.sum(np.einsum("nkd,kdr->nkr", delta, bases)**2, axis=2), 0)


class ChartGate:
    def __init__(self, method):
        self.method = method
        self.spec = METHODS[method]
        self.diagnostics = {}
        self.arrays = {}

    def fit(self, X, landmark_indices):
        self.landmark_indices = np.asarray(landmark_indices)
        if self.method == "kmeans":
            self.kmeans = minibatch(X)
            labels = self.kmeans.predict(X)
            self.arrays = {"centroids": self.kmeans.cluster_centers_}
        elif self.method == "kflats":
            labels = minibatch(X).predict(X)
            history = []
            for iteration in range(self.spec["max_iter"]):
                centers, bases = [], []
                for k in range(K):
                    members = X[labels == k]
                    if len(members) <= 2:
                        raise ValueError(f"K-flats empty/degenerate component {k}: {len(members)} members")
                    center = members.mean(axis=0)
                    _, _, vt = np.linalg.svd(members-center, full_matrices=False)
                    centers.append(center); bases.append(vt[:2].T)
                self.centers, self.bases = np.array(centers), np.array(bases)
                scores = plane_scores(X, self.centers, self.bases)
                new = scores.argmin(axis=1)
                history.append(float(scores[np.arange(len(X)), new].sum()))
                converged = np.array_equal(new, labels)
                labels = new
                if converged:
                    break
            self.diagnostics = {"iterations": iteration+1, "converged": converged, "objective": history}
            self.arrays = {"centers": self.centers, "local_bases": self.bases}
        elif self.method == "mppca":
            self._fit_ppca(X)
            labels = self.predict(X)
        else:
            L = X[self.landmark_indices]
            self.landmarks = L.copy()
            self.neighbors = NearestNeighbors(n_neighbors=20, algorithm="kd_tree", n_jobs=1).fit(L)
            if self.method in ("spectral", "diffusion"):
                embedding = self._fit_graph(L)
            elif self.method == "isomap":
                graph = self.neighbors.kneighbors_graph(mode="connectivity")
                components = connected_components(graph, directed=False)[0]
                self.diagnostics["landmark_graph_components"] = int(components)
                if components != 1:
                    raise ValueError(f"Isomap landmark graph has {components} components; no artificial bridge added")
                self.reducer = Isomap(n_neighbors=20, n_components=4, eigen_solver="arpack", path_method="D", n_jobs=1)
                embedding = self.reducer.fit_transform(L)
            else:
                import umap
                self.reducer = umap.UMAP(n_neighbors=20, n_components=4, min_dist=.1,
                    n_epochs=200, random_state=SEED, transform_seed=SEED, n_jobs=1,
                    metric="euclidean", low_memory=True).fit(L.astype(np.float32))
                embedding = self.reducer.embedding_.copy()
            self.landmark_embedding = embedding
            self.kmeans = KMeans(n_clusters=K, random_state=SEED, n_init=10, max_iter=300).fit(embedding)
            self.arrays.update(landmark_indices=self.landmark_indices, landmark_features=L,
                landmark_embedding=embedding, embedding_centroids=self.kmeans.cluster_centers_)
            # All original columns retain original order. UMAP uses singleton
            # calls both offline and online to avoid batch-dependent transforms.
            full = self.transform(X)
            labels = self.kmeans.predict(full)
            self.arrays["embedding"] = full
        self.labels = labels.astype(np.int64)
        return self

    def _fit_ppca(self, X):
        labels = minibatch(X).predict(X)
        resp = np.eye(K)[labels]
        history = []
        floor = self.spec["variance_floor"]
        for iteration in range(self.spec["max_iter"]):
            mass = resp.sum(axis=0)
            if mass.min() <= 3:
                raise ValueError("MPPCA collapsed component; no reseeding against evaluation results")
            self.weights = mass / len(X)
            self.centers = resp.T @ X / mass[:,None]
            covs, loadings, noise = [], [], []
            for k in range(K):
                delta = X-self.centers[k]
                cov = (delta.T * resp[:,k]) @ delta / mass[k]
                values, vectors = np.linalg.eigh(cov)
                order = np.argsort(values)[::-1]
                values, vectors = values[order], vectors[:,order]
                variance = max(float(values[2:].mean()), floor)
                W = vectors[:,:2] * np.sqrt(np.maximum(values[:2]-variance, 0))
                covs.append(W @ W.T + variance*np.eye(X.shape[1]))
                loadings.append(W); noise.append(variance)
            self.covariances = np.array(covs)
            self.precisions = np.linalg.inv(self.covariances)
            self.logdet = np.linalg.slogdet(self.covariances)[1]
            logp = self.log_posterior(X)
            total = logsumexp(logp, axis=1)
            objective = float(total.mean())
            history.append(objective)
            if len(history)>1 and objective < history[-2]-1e-7:
                raise ValueError("MPPCA likelihood decreased beyond numerical tolerance")
            converged = len(history)>1 and abs(history[-1]-history[-2]) < self.spec["tol"]
            resp = np.exp(logp-total[:,None])
            if converged:
                break
        self.diagnostics = {"iterations": iteration+1, "converged": converged, "mean_log_likelihood": history}
        self.arrays = {"means": self.centers, "loadings": np.array(loadings), "noise_variances": np.array(noise),
            "weights": self.weights, "covariances": self.covariances, "precisions": self.precisions}

    def log_posterior(self, X):
        delta = X[:,None,:]-self.centers[None,:,:]
        quad = np.einsum("nkd,kde,nke->nk", delta, self.precisions, delta)
        return np.log(self.weights)[None,:]-.5*(quad+self.logdet[None,:]+X.shape[1]*np.log(2*np.pi))

    def _fit_graph(self, L):
        dist, idx = self.neighbors.kneighbors(L, n_neighbors=21)
        dist, idx = dist[:,1:], idx[:,1:]
        self.bandwidth = max(float(np.median(dist[:,-1])), 1e-8)
        W = sparse.csr_matrix((np.exp(-dist.ravel()**2/self.bandwidth**2),
            (np.repeat(np.arange(len(L)),20),idx.ravel())), shape=(len(L),len(L)))
        W = W.maximum(W.T)
        components = connected_components(W, directed=False)[0]
        self.diagnostics["landmark_graph_components"] = int(components)
        if components != 1:
            raise ValueError(f"Landmark kernel graph has {components} components")
        self.density = np.asarray(W.sum(axis=1)).ravel()
        self.alpha = 1 if self.method == "diffusion" else 0
        A = sparse.diags(self.density**(-self.alpha)) @ W @ sparse.diags(self.density**(-self.alpha))
        degree = np.asarray(A.sum(axis=1)).ravel()
        S = sparse.diags(degree**-.5) @ A @ sparse.diags(degree**-.5)
        count = 5 if self.method == "diffusion" else 48
        vals, vecs = eigsh(S, k=count, which="LA", tol=1e-9,
            v0=np.random.default_rng(SEED).normal(size=len(L)))
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:,order]
        signs = np.sign(vecs[np.argmax(np.abs(vecs),axis=0),np.arange(count)])
        vecs *= signs
        psi = vecs/np.sqrt(degree[:,None])
        if self.method == "diffusion":
            vals, psi = vals[1:], psi[:,1:]
        self.eigenvalues, self.psi = vals, psi
        if np.min(np.abs(vals)) < 1e-8:
            raise ValueError("Unstable near-zero eigenvalue in Nystrom extension")
        self.graph = W
        self.arrays.update(eigenvalues=vals, eigenfunctions=psi, kernel_density=self.density,
            kernel_bandwidth=np.array(self.bandwidth), kernel_data=W.data, kernel_indices=W.indices,
            kernel_indptr=W.indptr, kernel_shape=np.array(W.shape))
        return self._coordinates(psi)

    def _coordinates(self, psi):
        if self.method == "spectral":
            return psi/np.maximum(np.linalg.norm(psi,axis=1,keepdims=True),1e-15)
        return psi*self.eigenvalues

    def transform(self, X):
        if self.method in ("spectral", "diffusion"):
            dist, idx = self.neighbors.kneighbors(X)
            weights = np.exp(-dist**2/self.bandwidth**2)
            # query-density factor cancels under row normalization.
            weights /= self.density[idx]**self.alpha
            weights /= np.maximum(weights.sum(axis=1,keepdims=True),1e-300)
            psi = np.einsum("nk,nkd->nd", weights, self.psi[idx])/self.eigenvalues
            result = self._coordinates(psi)
        elif self.method == "isomap":
            result = self.reducer.transform(X)
        else:
            result = np.vstack([self.reducer.transform(x[None,:].astype(np.float32))[0] for x in X])
        # At an exact fitted landmark, use its saved in-sample coordinates.
        dist, idx = self.neighbors.kneighbors(X, n_neighbors=1)
        exact = dist[:,0] < 1e-12
        result[exact] = self.landmark_embedding[idx[exact,0]]
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite embedding extension")
        return result

    def predict(self, X):
        X = np.asarray(X).reshape(-1,6)
        if self.method == "kflats":
            return plane_scores(X,self.centers,self.bases).argmin(axis=1)
        if self.method == "mppca":
            return self.log_posterior(X).argmax(axis=1)
        if self.method == "kmeans":
            return self.kmeans.predict(X)
        return self.kmeans.predict(self.transform(X))
