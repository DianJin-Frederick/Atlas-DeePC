"""Shared DeePC controller, Reacher coordinates, and linear-chart helpers.

The historical module name is retained for notebook imports and source manifests;
this module no longer contains an experiment search or a standalone rollout runner.
"""

import cvxpy as cp
import numpy as np


REACHER_ENV_ID = "Reacher-v5"
M = 2
P = 10
T_INI = 2
T_F = 10
L = T_INI + T_F


def reacher_output_from_obs(obs):
    obs = np.asarray(obs, dtype=float).reshape(-1)
    x_targ = obs[4]
    y_targ = obs[5]
    x_finger = obs[8] + x_targ
    y_finger = obs[9] + y_targ
    cos_sum = obs[0] * obs[1] - obs[2] * obs[3]
    sin_sum = obs[2] * obs[1] + obs[0] * obs[3]
    native_output = np.concatenate([obs[0:4], obs[6:8], [x_finger, y_finger]])
    if P == 8:
        return native_output
    if P == 10:
        return np.concatenate([native_output, [cos_sum, sin_sum]])
    raise ValueError(f"Unsupported Reacher output dimension P={P}; expected 8 or 10.")


def reacher_joint_configuration_rows():
    """Rows used to compare periodic arm configurations for the active output map."""
    if P == 8:
        return np.arange(4)
    if P == 10:
        return np.r_[0:4, 8:10]
    raise ValueError(f"Unsupported Reacher output dimension P={P}; expected 8 or 10.")


def partition_hankel_rows(H):
    expected_rows = (M + P) * L
    if H.shape[0] != expected_rows:
        raise ValueError(
            f"Hankel row mismatch: got {H.shape[0]} rows, expected {expected_rows} "
            f"for M={M}, P={P}, T_INI={T_INI}, T_F={T_F}, L={L}."
        )
    U_all = H[:M * L, :]
    Y_all = H[M * L:, :]
    U_blocks = U_all.reshape(L, M, H.shape[1])
    Y_blocks = Y_all.reshape(L, P, H.shape[1])
    U_p = U_blocks[:T_INI].reshape(M * T_INI, H.shape[1])
    U_f = U_blocks[T_INI:].reshape(M * T_F, H.shape[1])
    Y_p = Y_blocks[:T_INI].reshape(P * T_INI, H.shape[1])
    Y_f = Y_blocks[T_INI:].reshape(P * T_F, H.shape[1])
    return U_p, U_f, Y_p, Y_f


def fixed_rank_left_svd_basis(X, rank):
    """Return an exact fixed-rank left-SVD subspace through its covariance matrix."""
    X = np.asarray(X, dtype=float)
    rank = min(int(rank), X.shape[0])
    covariance = X @ X.T
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:rank]
    return eigenvectors[:, order]


def unpack_basis(A):
    if isinstance(A, dict):
        return A["offset"], A["basis"], A.get("g_min"), A.get("g_max")
    if isinstance(A, tuple):
        return A[0], A[1], None, None
    return 0.0, A, None, None


def closed_loop_control_cost(
    outputs,
    actions,
    reference,
    Q,
    R,
    reference_description="active control reference at each step",
):
    """Evaluate realized stage cost against a constant or stepwise reference.

    For graph-routed control, ``reference[t]`` must be the waypoint active while
    ``actions[t]`` produced ``outputs[t]``. A single reference vector is
    broadcast across the rollout for final-reference controllers and auxiliary
    final-goal diagnostics.
    """
    outputs = np.asarray(outputs, dtype=float)
    actions = np.asarray(actions, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if reference.ndim == 1:
        reference = reference.reshape(1, -1)
    if reference.shape[0] == 1:
        reference = np.repeat(reference, outputs.shape[0], axis=0)
    if reference.shape != outputs.shape:
        raise ValueError(
            f"reference shape {reference.shape} must equal output shape "
            f"{outputs.shape}, or contain one broadcastable row"
        )
    output_error = outputs - reference
    tracking_per_step = np.einsum("ti,ij,tj->t", output_error, Q, output_error)
    input_per_step = np.einsum("ti,ij,tj->t", actions, R, actions)
    tracking_cost = float(np.sum(tracking_per_step))
    input_cost = float(np.sum(input_per_step))
    total_cost = tracking_cost + input_cost
    return {
        "closed_loop_control_cost": total_cost,
        "closed_loop_tracking_cost": tracking_cost,
        "closed_loop_input_cost": input_cost,
        "mean_closed_loop_stage_cost": float(total_cost / max(outputs.shape[0], 1)),
        "closed_loop_cost_definition": (
            "sum_t (y_t-r_t)^T Q (y_t-r_t) + u_t^T R u_t; "
            f"r_t is the {reference_description}"
        ),
    }


def validate_objective_options(lam_g=0., lam_du=0., Q_terminal=None):
    """Reject retired penalties retained in historical configuration metadata."""
    if lam_g != 0 or lam_du != 0 or (Q_terminal is not None and np.any(Q_terminal)):
        raise ValueError("The paper objective requires lam_g=0, lam_du=0 and no terminal cost; "
                         "use frozen historical source to reproduce older objectives")


class DeePCController:
    """Tracking, input effort and initial-output slack cost."""

    def __init__(self, Q, R, lam_ini, u_limit, solver,
                 *, dimensions=None):
        self.Q = Q
        self.R = R
        self.lam_ini = lam_ini
        self.u_limit = u_limit
        self.solver = solver
        self.dimensions = dimensions

    def tracking_cost(self, u, y, sigma, y_ref):
        """Paper objective: ||y-r||_Q^2 + ||u||_R^2 + lam_ini ||sigma||^2."""
        m, p, _, tf = self.dimensions or (M, P, T_INI, T_F)
        cost = self.lam_ini * cp.sum_squares(sigma)
        for j in range(tf):
            yj = y[j * p:(j + 1) * p]
            rj = y_ref[j * p:(j + 1) * p]
            uj = u[j * m:(j + 1) * m]
            cost += cp.quad_form(yj - rj, self.Q) + cp.quad_form(uj, self.R)
        return cost

    def solve(self, A, u_ini, y_ini, y_ref):
        m, p, ti, tf = self.dimensions or (M, P, T_INI, T_F)
        if self.Q.shape != (p, p) or self.R.shape != (m, m):
            raise ValueError("Cost dimensions do not match the DeePC task")
        affine_offset, A_matrix, g_min, g_max = unpack_basis(A)
        g_sum = A.get("g_sum") if isinstance(A, dict) else None
        g_nonnegative = bool(A.get("g_nonnegative", False)) if isinstance(A, dict) else False
        g_negative_weight = float(A.get("g_negative_weight", 0.0)) if isinstance(A, dict) else 0.0
        u = cp.Variable((m * tf, 1))
        y = cp.Variable((p * tf, 1))
        g = cp.Variable((A_matrix.shape[1], 1))
        sigma = cp.Variable((p * ti, 1))
        b = cp.vstack([u_ini, u, y_ini + sigma, y])
        cost = self.tracking_cost(u, y, sigma, y_ref)
        if g_negative_weight > 0.0:
            cost += g_negative_weight * cp.sum(cp.pos(-g))
        limit = np.asarray(self.u_limit, dtype=float)
        if limit.ndim:
            limit = np.tile(limit.reshape(m), tf).reshape(-1, 1)
        constraints = [b == affine_offset + A_matrix @ g, u <= limit, u >= -limit]
        # Empirical SVD-coordinate boxes (mean +/- 3 SD) are disabled.
        # if g_min is not None:
        #     constraints.append(g >= g_min)
        # if g_max is not None:
        #     constraints.append(g <= g_max)
        # Preserve the separate one-sided floor used by raw-column methods.
        if g_min is not None and g_max is None:
            constraints.append(g >= g_min)
        if g_sum is not None:
            constraints.append(cp.sum(g) == float(g_sum))
        if g_nonnegative:
            constraints.append(g >= 0.0)
        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=self.solver, verbose=False, max_iter=10000)
        except Exception as exc:
            return None, None, None, f"crashed: {exc}", np.inf
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return None, None, None, prob.status, np.inf
        residual = np.linalg.norm(affine_offset + A_matrix @ g.value - np.vstack([u_ini, u.value, y_ini, y.value]))
        return u.value, y.value, g.value, prob.status, float(residual)
