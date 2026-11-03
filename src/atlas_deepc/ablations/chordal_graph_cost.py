"""Chordal subspace distances and transition-count-weighted graph costs."""

import numpy as np


def chordal_distances(bases):
    """Projection chordal distance; equals sqrt(r - ||U.T V||_F^2) at equal rank."""
    bases = [np.asarray(b, dtype=float) for b in bases]
    if not bases or any(b.ndim != 2 or b.shape[0] != bases[0].shape[0] for b in bases):
        raise ValueError("Bases must have a common ambient dimension")
    for b in bases:
        if not np.all(np.isfinite(b)) or not np.allclose(b.T @ b, np.eye(b.shape[1]), atol=1e-8):
            raise ValueError("Chart bases must have orthonormal columns")
    distance = np.zeros((len(bases), len(bases)))
    for k, u in enumerate(bases):
        for j in range(k):
            v = bases[j]
            squared = (u.shape[1] + v.shape[1]) / 2 - np.sum((u.T @ v)**2)
            distance[k, j] = distance[j, k] = np.sqrt(max(0., squared))
    return distance


def graph_costs(distance, counts, threshold):
    distance, counts = np.asarray(distance), np.asarray(counts)
    if distance.shape != counts.shape or distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("Distance/count matrices must be matching square matrices")
    edge = (counts >= threshold) & ~np.eye(len(counts), dtype=bool)
    if np.any(~np.isfinite(distance[edge])) or np.any(distance[edge] < 0):
        raise ValueError("Edge distances must be finite and nonnegative")
    cost = np.full(counts.shape, np.inf)
    cost[edge] = distance[edge] / np.sqrt(counts[edge])
    return cost
