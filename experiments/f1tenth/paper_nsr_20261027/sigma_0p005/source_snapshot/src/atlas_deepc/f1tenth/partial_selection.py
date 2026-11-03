"""Experimental selection using the past rows of the existing full chart bases.

This changes selection only: no retraining, coefficient penalty or QP change.
Restricted bases are orthonormalized separately and may have different ranks.
"""
import numpy as np
from .config import TI, TF

METHOD = 'past_window_projection'
PAST_ROWS = np.r_[0:2*TI, 2*(TI+TF):2*(TI+TF)+4*TI]
RANK_RTOL = 1e-10
ANCHOR_ROWS = np.array([6*TI-4, 6*TI-3, 6*TI-1])
ACTIVE_ROWS = np.setdiff1d(np.arange(6*TI), ANCHOR_ROWS)


def past_matrix(full):
    restricted = full[PAST_ROWS].copy()
    # x, y and yaw of the anchor pose are identically zero by construction.
    # Covariance eigenvectors can carry roundoff in these exact zero rows.
    if np.max(np.abs(restricted[ANCHOR_ROWS])) > 1e-12:
        raise ValueError('Atlas basis violates the expected pose-anchor convention')
    restricted[ANCHOR_ROWS] = 0.
    return restricted


class PastWindowProjectionSelector:
    def __init__(self, full_bases):
        self.bases = []
        self.ranks = []
        for full in full_bases:
            # Remove identically zero coordinates before SVD; retaining them
            # lets ill-conditioned small modes leak into anchor coordinates.
            restricted = past_matrix(full)[ACTIVE_ROWS]
            left, singular, _ = np.linalg.svd(restricted, full_matrices=False)
            # Fixed numerical rank threshold, independent of rollout results.
            tolerance = RANK_RTOL * singular[0]
            rank = int(np.sum(singular > tolerance))
            basis = np.zeros((6*TI, rank))
            basis[ACTIVE_ROWS] = left[:, :rank]
            self.bases.append(basis)
            self.ranks.append(rank)
        self.ranks = np.asarray(self.ranks)
        self.reset()

    def reset(self):
        self.residual_history = []
        self.tie_history = []

    def pack_history(self, u_history, y_history):
        u = np.asarray(u_history).reshape(-1)
        y = np.asarray(y_history).reshape(-1)
        if u.size != 2*TI or y.size != 4*TI:
            raise ValueError('Expected T_ini observed input/output pairs')
        return np.r_[u, y]

    def residuals(self, window):
        window = np.asarray(window, dtype=float).reshape(-1)
        if window.shape != (6*TI,) or not np.isfinite(window).all():
            raise ValueError('Expected a finite past-window input/output vector')
        # Explicit residual avoids catastrophic cancellation near exact fits.
        return np.array([np.linalg.norm(window - q @ (q.T @ window)) for q in self.bases])

    def select(self, window):
        residuals = self.residuals(window)
        tolerance = 100 * np.finfo(float).eps * max(1., np.linalg.norm(window))
        ties = np.flatnonzero(residuals <= residuals.min() + tolerance)
        self.residual_history.append(residuals)
        self.tie_history.append(int(ties.size))
        return int(ties[0])
