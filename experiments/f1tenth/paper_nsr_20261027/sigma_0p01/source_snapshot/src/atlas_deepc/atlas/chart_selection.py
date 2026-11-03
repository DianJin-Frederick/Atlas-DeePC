"""Chart selection by full-window input/output projection residual."""

import numpy as np


class FullWindowProjectionSelector:
    """Project the latest L measured input/output samples onto full linear charts."""

    def __init__(self, matrix_cache, input_dimension, output_dimension, window_length):
        self.input_rows = input_dimension * window_length
        self.output_rows = output_dimension * window_length
        self.window_dimension = self.input_rows + self.output_rows
        self.chart_ids = np.array(sorted(key[0] for key in matrix_cache if len(key) == 1), dtype=int)
        if not self.chart_ids.size:
            raise ValueError("No single-chart matrices supplied")
        bases = []
        ranks = []
        for chart in self.chart_ids:
            matrix, _ = matrix_cache[(int(chart),)]
            if np.any(matrix["offset"]):
                raise ValueError("Full-window projection requires linear charts")
            basis = np.asarray(matrix["basis"], dtype=float)
            if basis.shape[0] != self.window_dimension:
                raise ValueError("Chart basis has the wrong full-window row count")
            left, singular, _ = np.linalg.svd(basis, full_matrices=False)
            rank = int(np.sum(singular > max(basis.shape) * np.finfo(float).eps * singular[0]))
            bases.append(left[:, :rank])
            ranks.append(rank)
        if len(set(ranks)) != 1:
            raise ValueError("Chart bases must have the same rank for batch projection")
        self.bases = np.stack(bases)
        self.ranks = np.asarray(ranks)
        self.reset()

    def reset(self):
        self.residual_history = []
        self.tie_history = []

    def pack_history(self, u_history, y_history):
        u = np.asarray(u_history, dtype=float).reshape(-1)
        y = np.asarray(y_history, dtype=float).reshape(-1)
        if u.size != self.input_rows or y.size != self.output_rows:
            raise ValueError("Expected L observed input and output samples")
        return np.r_[u, y]

    def residuals(self, window):
        window = np.asarray(window, dtype=float).reshape(-1)
        if window.shape != (self.window_dimension,) or not np.all(np.isfinite(window)):
            raise ValueError("Expected a finite full-window input/output vector")
        coefficients = np.einsum("knr,n->kr", self.bases, window)
        squared = np.dot(window, window) - np.einsum("kr,kr->k", coefficients, coefficients)
        return np.sqrt(np.maximum(squared, 0.))

    def select(self, window):
        residuals = self.residuals(window)
        tolerance = 100 * np.finfo(float).eps * max(1., np.linalg.norm(window))
        ties = np.flatnonzero(residuals <= residuals.min() + tolerance)
        self.residual_history.append(residuals)
        self.tie_history.append(int(ties.size))
        return int(self.chart_ids[ties[0]])
