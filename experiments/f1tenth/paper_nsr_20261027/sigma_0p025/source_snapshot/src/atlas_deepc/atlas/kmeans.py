"""The baseline's shared MiniBatchKMeans fit and nearest-centroid assignment."""

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def fit_kmeans(features, clusters, seed):
    model = MiniBatchKMeans(n_clusters=clusters, random_state=seed, n_init=5,
                            batch_size=2048, max_iter=300, reassignment_ratio=0.002)
    labels = model.fit_predict(features)
    return labels, model.cluster_centers_


def nearest_centroid(centers, features):
    features = np.atleast_2d(features)
    return np.argmin(np.mean((features[:, None] - centers[None]) ** 2, axis=2), axis=1)
