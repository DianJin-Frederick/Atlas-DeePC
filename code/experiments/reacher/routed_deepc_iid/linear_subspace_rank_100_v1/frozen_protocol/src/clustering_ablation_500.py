"""Resumable fixed-feature clustering ablation for graph-routed Atlas-DeePC.

All methods use the same offline Hankel library, six-dimensional periodic joint-
configuration gate, 48 charts, controller, and evaluation seeds.  Methods that
do not provide an out-of-sample ``predict`` operation use the adapter documented
in ``METHODS``; the adapter name is also saved in every summary.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
from scipy import sparse
from sklearn.cluster import (
    AgglomerativeClustering,
    HDBSCAN,
    MiniBatchKMeans,
    SpectralClustering,
)
from sklearn.linear_model import orthogonal_mp
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

import expert_pipeline_search as pipe
import routed_deepc_experiment as routed
from local_affine_atlas_experiment import audit_iid_library


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "experiments" / "routed_deepc_iid" / "clustering_methods_500"
DATA_PATH = ROOT / "data" / "select_iid_offline_hankel.npy"
EVAL_SEEDS = tuple(range(111, 611))
CLUSTERS = 48
MIN_CHART_SIZE = 61  # rank-60 chart matrices require at least rank + 1 columns

# ``kind`` makes approximations explicit in the machine-readable outputs.
METHODS = {
    "kmeans": {
        "label": "MiniBatch K-means",
        "kind": "exact_inductive",
        "online_assignment": "nearest centroid (Euclidean)",
    },
    "spherical_kmeans": {
        "label": "Spherical K-means",
        "kind": "exact_inductive",
        "online_assignment": "nearest unit centroid (cosine)",
    },
    "gmm_diag": {
        "label": "Diagonal Gaussian mixture",
        "kind": "exact_inductive",
        "online_assignment": "maximum posterior component",
    },
    "agglomerative_ward": {
        "label": "Agglomerative (Ward + k-NN connectivity)",
        "kind": "exact_offline_with_inductive_adapter",
        "online_assignment": "nearest offline-cluster centroid",
    },
    "spectral_knn": {
        "label": "Spectral clustering (k-NN graph)",
        "kind": "exact_offline_with_inductive_adapter",
        "online_assignment": "nearest offline-cluster centroid",
    },
    "k_subspaces": {
        "label": "Affine K-subspaces (rank 2)",
        "kind": "exact_inductive",
        "online_assignment": "minimum affine-subspace residual",
    },
    "ssc_landmark": {
        "label": "Sparse subspace clustering (landmark)",
        "kind": "landmark_approximation",
        "online_assignment": "5-nearest landmark majority vote",
    },
    "lrr_landmark": {
        "label": "Low-rank representation (landmark, Frobenius noise)",
        "kind": "landmark_approximation",
        "online_assignment": "5-nearest landmark majority vote",
    },
    "kmedoids_clara": {
        "label": "K-medoids (CLARA)",
        "kind": "sampling_approximation",
        "online_assignment": "nearest medoid (Euclidean)",
    },
    "hdbscan_fixed_k": {
        "label": "HDBSCAN projected to 48 charts",
        "kind": "fixed_k_adapter",
        "online_assignment": "nearest adapted-cluster centroid",
    },
}


def _base_config(method: str, steps: int = 320) -> routed.RoutedDeePCConfig:
    clustering_method = {
        "kmeans": "minibatch_kmeans",
        "gmm_diag": "gaussian_mixture_diag",
        "k_subspaces": "affine_k_subspaces",
    }.get(method, "minibatch_kmeans")
    return routed.RoutedDeePCConfig(
        clusters=CLUSTERS,
        cluster_seed=11,
        clustering_input_weight=0.0,
        clustering_method=clustering_method,
        clustering_feature_mode="plant_joint_configuration",
        clustering_subspace_rank=2,
        clustering_max_iter=20,
        formulation="svd_current",
        svd_rank=60,
        steps=steps,
    )


def _cluster_centers(features: np.ndarray, labels: np.ndarray, clusters: int) -> np.ndarray:
    centers = []
    for cluster in range(clusters):
        members = features[labels == cluster]
        if not members.size:
            raise RuntimeError(f"Clustering produced empty chart {cluster}.")
        centers.append(members.mean(axis=0))
    return np.asarray(centers)


def _base_atlas(
    H: np.ndarray,
    config: routed.RoutedDeePCConfig,
    labels: np.ndarray,
    centers: np.ndarray,
    assignment_method: str = "minibatch_kmeans",
    method_data: dict | None = None,
) -> dict:
    features, feature_mean, feature_scale = routed.behavioral_features(
        H,
        config.clustering_input_weight,
        config.clustering_feature_mode,
        config.clustering_configuration_weight,
        config.clustering_velocity_weight,
        config.clustering_fingertip_weight,
    )
    labels = np.asarray(labels, dtype=int)
    centers = np.asarray(centers, dtype=float)
    clusters = centers.shape[0]
    counts = np.zeros((clusters, clusters), dtype=int)
    for episode in range(H.shape[1] // routed.WINDOWS_PER_EPISODE):
        start = episode * routed.WINDOWS_PER_EPISODE
        stop = start + routed.WINDOWS_PER_EPISODE
        np.add.at(counts, (labels[start : stop - 1], labels[start + 1 : stop]), 1)

    Y = H[pipe.M * pipe.L :].reshape(pipe.L, pipe.P, H.shape[1])
    medoid_states = []
    for cluster in range(clusters):
        indices = np.flatnonzero(labels == cluster)
        if assignment_method == "spherical_kmeans":
            members = features[indices]
            members = members / np.maximum(np.linalg.norm(members, axis=1, keepdims=True), 1e-12)
            error = 1.0 - members @ centers[cluster]
        else:
            delta = features[indices] - centers[cluster]
            error = np.mean(delta * delta, axis=1)
        medoid_index = int(indices[np.argmin(error)])
        state = Y[pipe.T_INI - 1, :, medoid_index].copy()
        state[4:6] = 0.0
        medoid_states.append(state)

    center_delta = centers[:, None, :] - centers[None, :, :]
    atlas = {
        "labels": labels,
        "centers": centers,
        "clustering_method": np.array(assignment_method),
        "clustering_feature_mode": np.array(config.clustering_feature_mode),
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "transition_counts": counts,
        "center_distance": np.sqrt(np.mean(center_delta * center_delta, axis=2)),
        "medoid_states": np.asarray(medoid_states),
    }
    if method_data:
        atlas.update(method_data)
    return atlas


def _spherical_kmeans(features: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    fit = MiniBatchKMeans(
        n_clusters=CLUSTERS,
        random_state=seed,
        n_init=5,
        batch_size=2048,
        max_iter=300,
        reassignment_ratio=0.002,
    ).fit(normalized)
    centers = fit.cluster_centers_
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    return fit.labels_, centers


def _clara_kmedoids(features: np.ndarray, seed: int, sample_size: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """CLARA-style K-medoids with three deterministic subsamples."""
    rng = np.random.default_rng(seed)
    best_medoids = None
    best_loss = np.inf
    for _ in range(3):
        sample_indices = rng.choice(features.shape[0], min(sample_size, features.shape[0]), replace=False)
        sample = features[sample_indices]
        init = MiniBatchKMeans(n_clusters=CLUSTERS, random_state=seed, n_init=3).fit(sample)
        distance_to_centers = pairwise_distances(sample, init.cluster_centers_)
        medoid_local = np.array([
            np.argmin(distance_to_centers[:, cluster]) for cluster in range(CLUSTERS)
        ])
        medoids = sample[medoid_local]
        for _ in range(5):
            labels = np.argmin(pairwise_distances(sample, medoids), axis=1)
            new_medoids = []
            for cluster in range(CLUSTERS):
                members = np.flatnonzero(labels == cluster)
                if not members.size:
                    new_medoids.append(rng.integers(sample.shape[0]))
                    continue
                candidates = members if members.size <= 256 else rng.choice(members, 256, replace=False)
                candidate_loss = pairwise_distances(sample[members], sample[candidates]).sum(axis=0)
                new_medoids.append(candidates[np.argmin(candidate_loss)])
            medoid_local = np.asarray(new_medoids)
            medoids = sample[medoid_local]
        probe = rng.choice(features.shape[0], min(5000, features.shape[0]), replace=False)
        loss = np.min(pairwise_distances(features[probe], medoids), axis=1).mean()
        if loss < best_loss:
            best_loss, best_medoids = loss, medoids.copy()
    labels = np.argmin(pairwise_distances(features, best_medoids), axis=1)
    return labels, best_medoids


def _force_fixed_k(features: np.ndarray, labels: np.ndarray, clusters: int, seed: int) -> np.ndarray:
    """Convert density labels (including -1 noise) into exactly ``clusters`` groups."""
    labels = np.asarray(labels, dtype=int).copy()
    non_noise = sorted(set(labels) - {-1})
    if not non_noise:
        return MiniBatchKMeans(n_clusters=clusters, random_state=seed, n_init=5).fit_predict(features)
    remap = {old: new for new, old in enumerate(non_noise)}
    mapped = np.array([remap.get(value, -1) for value in labels])
    centers = _cluster_centers(features[mapped >= 0], mapped[mapped >= 0], len(non_noise))
    noise = mapped < 0
    if noise.any():
        mapped[noise] = np.argmin(pairwise_distances(features[noise], centers), axis=1)
    current = len(non_noise)
    if current > clusters:
        merge = AgglomerativeClustering(n_clusters=clusters, linkage="ward").fit_predict(centers)
        return merge[mapped]
    if current == clusters:
        return mapped

    # Split the largest groups until there are exactly K, preserving density groups.
    next_label = current
    rng = np.random.default_rng(seed)
    while next_label < clusters:
        sizes = np.bincount(mapped, minlength=next_label)
        parent = int(np.argmax(sizes))
        members = np.flatnonzero(mapped == parent)
        if members.size < 2:
            raise RuntimeError("Cannot split HDBSCAN groups into the requested chart count.")
        split = MiniBatchKMeans(
            n_clusters=2, random_state=int(rng.integers(2**31 - 1)), n_init=3
        ).fit_predict(features[members])
        mapped[members[split == 1]] = next_label
        next_label += 1
    return mapped


def _landmarks(features: np.ndarray, seed: int, count: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(features.shape[0], min(count, features.shape[0]), replace=False))
    return indices, features[indices]


def _ssc_landmark_labels(landmarks: np.ndarray, seed: int) -> np.ndarray:
    neighbor_count = min(64, landmarks.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=neighbor_count + 1).fit(landmarks)
    neighbor_indices = nn.kneighbors(return_distance=False)[:, 1:]
    rows, cols, values = [], [], []
    sparsity = min(5, landmarks.shape[1] - 1)
    for index, candidates in enumerate(neighbor_indices):
        dictionary = landmarks[candidates].T
        coefficients = orthogonal_mp(dictionary, landmarks[index], n_nonzero_coefs=sparsity)
        nonzero = np.flatnonzero(np.abs(coefficients) > 1e-10)
        rows.extend([index] * nonzero.size)
        cols.extend(candidates[nonzero])
        values.extend(np.abs(coefficients[nonzero]))
    affinity = sparse.csr_matrix((values, (rows, cols)), shape=(landmarks.shape[0],) * 2)
    affinity = affinity + affinity.T
    return SpectralClustering(
        n_clusters=CLUSTERS,
        affinity="precomputed",
        assign_labels="cluster_qr",
        random_state=seed,
    ).fit_predict(affinity)


def _lrr_landmark_labels(landmarks: np.ndarray, seed: int, regularization: float = 1.0) -> np.ndarray:
    # Closed-form nuclear-norm LRR with Frobenius reconstruction noise:
    # min_Z ||Z||_* + lambda/2 ||X-XZ||_F^2.
    _, singular_values, vt = np.linalg.svd(landmarks.T, full_matrices=False)
    shrink = np.maximum(1.0 - 1.0 / np.maximum(regularization * singular_values**2, 1e-12), 0.0)
    representation = (vt.T * shrink) @ vt
    affinity = np.abs(representation) + np.abs(representation.T)
    np.fill_diagonal(affinity, 0.0)
    return SpectralClustering(
        n_clusters=CLUSTERS,
        affinity="precomputed",
        assign_labels="cluster_qr",
        random_state=seed,
    ).fit_predict(affinity)


def _expand_landmark_labels(features: np.ndarray, landmarks: np.ndarray, labels: np.ndarray) -> np.ndarray:
    nearest = NearestNeighbors(n_neighbors=1).fit(landmarks)
    return labels[nearest.kneighbors(features, return_distance=False)[:, 0]]


def _repair_small_charts(
    features: np.ndarray, labels: np.ndarray, minimum: int, seed: int
) -> tuple[np.ndarray, int]:
    """Dissolve undersized charts and bisect large charts to preserve fixed K."""
    labels = np.asarray(labels, dtype=int).copy()
    rng = np.random.default_rng(seed)
    repairs = 0
    while np.bincount(labels, minlength=CLUSTERS).min() < minimum:
        sizes = np.bincount(labels, minlength=CLUSTERS)
        small = int(np.argmin(sizes))
        valid = np.flatnonzero(np.arange(CLUSTERS) != small)
        centers = _cluster_centers(
            features[labels != small],
            np.searchsorted(valid, labels[labels != small]),
            valid.size,
        )
        members = np.flatnonzero(labels == small)
        labels[members] = valid[
            np.argmin(pairwise_distances(features[members], centers), axis=1)
        ]
        sizes = np.bincount(labels, minlength=CLUSTERS)
        donor = int(np.argmax(np.where(np.arange(CLUSTERS) == small, -1, sizes)))
        donor_members = np.flatnonzero(labels == donor)
        split = MiniBatchKMeans(
            n_clusters=2,
            random_state=int(rng.integers(2**31 - 1)),
            n_init=5,
        ).fit_predict(features[donor_members])
        smaller_half = int(np.argmin(np.bincount(split, minlength=2)))
        labels[donor_members[split == smaller_half]] = small
        repairs += 1
        if repairs > CLUSTERS:
            raise RuntimeError("Could not enforce the minimum chart size.")
    return labels, repairs


def fit_method_atlas(H: np.ndarray, method: str, config: routed.RoutedDeePCConfig) -> tuple[dict, dict]:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {tuple(METHODS)}")
    started = time.perf_counter()
    if method in {"kmeans", "gmm_diag", "k_subspaces"}:
        atlas = routed.fit_behavioral_atlas(H, config)
    else:
        features, _, _ = routed.behavioral_features(
            H, config.clustering_input_weight, config.clustering_feature_mode
        )
        method_data = {}
        assignment_method = "minibatch_kmeans"
        if method == "spherical_kmeans":
            labels, centers = _spherical_kmeans(features, config.cluster_seed)
            assignment_method = "spherical_kmeans"
        elif method == "agglomerative_ward":
            connectivity = kneighbors_graph(features, n_neighbors=15, include_self=False)
            labels = AgglomerativeClustering(
                n_clusters=CLUSTERS, linkage="ward", connectivity=connectivity
            ).fit_predict(features)
            centers = _cluster_centers(features, labels, CLUSTERS)
        elif method == "spectral_knn":
            labels = SpectralClustering(
                n_clusters=CLUSTERS,
                affinity="nearest_neighbors",
                n_neighbors=15,
                assign_labels="cluster_qr",
                random_state=config.cluster_seed,
                n_jobs=1,
            ).fit_predict(features)
            centers = _cluster_centers(features, labels, CLUSTERS)
        elif method == "kmedoids_clara":
            labels, centers = _clara_kmedoids(features, config.cluster_seed)
        elif method == "hdbscan_fixed_k":
            density_labels = HDBSCAN(
                min_cluster_size=max(50, features.shape[0] // (CLUSTERS * 3)),
                min_samples=15,
                cluster_selection_method="leaf",
                n_jobs=1,
            ).fit_predict(features)
            labels = _force_fixed_k(features, density_labels, CLUSTERS, config.cluster_seed)
            centers = _cluster_centers(features, labels, CLUSTERS)
            method_data["native_density_cluster_count"] = np.array(
                len(set(density_labels) - {-1}), dtype=int
            )
            method_data["native_noise_fraction"] = np.array(np.mean(density_labels < 0))
        elif method in {"ssc_landmark", "lrr_landmark"}:
            landmark_indices, landmark_features = _landmarks(features, config.cluster_seed)
            if method == "ssc_landmark":
                landmark_labels = _ssc_landmark_labels(landmark_features, config.cluster_seed)
            else:
                landmark_labels = _lrr_landmark_labels(landmark_features, config.cluster_seed)
            labels = _expand_landmark_labels(features, landmark_features, landmark_labels)
            labels, repair_count = _repair_small_charts(
                features, labels, MIN_CHART_SIZE, config.cluster_seed
            )
            centers = _cluster_centers(features, labels, CLUSTERS)
            assignment_method = "nearest_landmark"
            method_data.update({
                "assignment_features": landmark_features,
                "assignment_labels": labels[landmark_indices],
                "assignment_neighbors": np.array(5, dtype=int),
                "landmark_indices": landmark_indices,
                "small_chart_repair_count": np.array(repair_count, dtype=int),
            })
        else:  # pragma: no cover - guarded by METHODS
            raise AssertionError(method)
        atlas = _base_atlas(
            H, config, labels, centers, assignment_method=assignment_method, method_data=method_data
        )

    metadata = {
        "method_key": method,
        **METHODS[method],
        "fit_time_s": time.perf_counter() - started,
        "offline_windows": int(H.shape[1]),
        "features": "six-dimensional standardized periodic joint configuration",
        "clusters": int(atlas["centers"].shape[0]),
        "minimum_chart_size_required": MIN_CHART_SIZE,
        "small_chart_repair_count": int(np.asarray(atlas.get("small_chart_repair_count", 0)).item()),
    }
    return atlas, metadata


def _load_existing(path: Path, config: routed.RoutedDeePCConfig) -> dict[int, dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    expected = asdict(config)
    observed = payload.get("config", {})
    mismatches = {
        key: (observed.get(key), value)
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Cannot resume {path} with changed configuration: {mismatches}. "
            "Use a different --result-root or explicitly pass --force."
        )
    return {int(row["seed"]): row for row in payload.get("results", [])}


def _write_summary(
    path: Path, config, audit, atlas, metadata, results: dict[int, dict], requested_seeds
) -> None:
    sizes = np.bincount(atlas["labels"].astype(int), minlength=atlas["centers"].shape[0])
    payload = {
        "experiment": "paired_clustering_method_ablation_500",
        "method": metadata,
        "config": asdict(config),
        "fairness_audit": audit,
        "chart_sizes": sizes.tolist(),
        "requested_seeds": list(requested_seeds),
        "completed_seeds": sorted(results),
        "results": [results[seed] for seed in sorted(results)],
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def run_method(
    method: str,
    seeds=EVAL_SEEDS,
    steps: int = 320,
    force: bool = False,
    result_root: Path = RESULT_ROOT,
) -> Path:
    config = _base_config(method, steps=steps)
    output_dir = Path(result_root) / method
    output_dir.mkdir(parents=True, exist_ok=True)
    H = np.load(DATA_PATH)
    audit = audit_iid_library(H, eval_seeds=seeds)
    atlas_path = output_dir / "behavioral_atlas.npz"
    metadata_path = output_dir / "method_metadata.json"
    if atlas_path.exists() and metadata_path.exists() and not force:
        atlas = routed.load_atlas(atlas_path)
        metadata = json.loads(metadata_path.read_text())
    else:
        atlas, metadata = fit_method_atlas(H, method, config)
        routed.save_atlas(atlas_path, atlas, config)
        metadata_path.write_text(json.dumps(metadata, indent=2))
    sizes = np.bincount(atlas["labels"].astype(int), minlength=atlas["centers"].shape[0])
    print(method, "chart sizes", int(sizes.min()), float(np.median(sizes)), int(sizes.max()), flush=True)

    summary_path = output_dir / "summary.json"
    results = {} if force else _load_existing(summary_path, config)
    matrix_cache = routed.precompute_chart_matrix_cache(H, atlas, config)
    for seed in seeds:
        if seed in results and not force:
            continue
        run_config = routed.RoutedDeePCConfig(**{**asdict(config), "seed": int(seed)})
        result_path = output_dir / f"seed{seed}_control_results.npz"
        summary = routed.run_routed_deepc(
            H,
            atlas,
            run_config,
            save_path=result_path,
            summary_metadata={"clustering_ablation": metadata},
            matrix_cache=matrix_cache,
        )
        summary["result_path"] = str(result_path)
        results[int(seed)] = summary
        _write_summary(summary_path, config, audit, atlas, metadata, results, seeds)
        print(
            method,
            "seed", seed,
            "mean30", f"{summary['mean_last_30']:.6f}",
            "success", bool(summary["success_30"] and summary["success_100"]),
            flush=True,
        )
    return summary_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS), default=tuple(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=EVAL_SEEDS)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    return parser.parse_args()


def main(args):
    for method in args.methods:
        run_method(
            method,
            seeds=tuple(args.seeds),
            steps=args.steps,
            force=args.force,
            result_root=args.result_root,
        )


if __name__ == "__main__":
    main(parse_args())
