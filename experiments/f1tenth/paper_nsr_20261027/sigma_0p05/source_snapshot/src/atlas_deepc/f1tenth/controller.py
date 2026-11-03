"""F1TENTH QP: tracking + input effort + output-history slack only."""

import cvxpy as cp
import numpy as np

from atlas_deepc.control.expert_pipeline_search import DeePCController as SharedController

OBJECTIVE = "tracking_input_history_slack_v2"
OBJECTIVE_FORMULA = "||y_f-y_ref||_Q^2 + ||u_f||_R^2 + lambda_ini*||sigma||_2^2"


class DeePCController(SharedController):
    """Shared three-term objective with explicit F1TENTH dimensions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.dimensions is None:
            raise ValueError("F1TENTH requires explicit task dimensions")

    def build_problem(self, A, u_ini, y_ini, y_ref):
        if not isinstance(A, np.ndarray):
            raise TypeError("F1TENTH accepts only unbounded linear array bases")
        m, p, ti, tf = self.dimensions
        if self.Q.shape != (p, p) or self.R.shape != (m, m):
            raise ValueError("Cost dimensions do not match the DeePC task")
        if A.ndim != 2 or A.shape[0] != (m+p)*(ti+tf):
            raise ValueError("Basis dimensions do not match the DeePC task")
        u = cp.Variable((m*tf, 1), name="u_future")
        y = cp.Variable((p*tf, 1), name="y_future")
        g = cp.Variable((A.shape[1], 1), name="g")
        sigma = cp.Variable((p*ti, 1), name="sigma")
        cost = self.tracking_cost(u, y, sigma, y_ref)
        b = cp.vstack([u_ini, u, y_ini+sigma, y])
        limit = np.asarray(self.u_limit, dtype=float)
        if limit.ndim:
            limit = np.tile(limit.reshape(m), tf).reshape(-1, 1)
        problem = cp.Problem(cp.Minimize(cost), [b == A@g, u <= limit, u >= -limit])
        return problem, u, y, g

    def solve(self, A, u_ini, y_ini, y_ref):
        problem, u, y, g = self.build_problem(A, u_ini, y_ini, y_ref)
        try:
            problem.solve(solver=self.solver, verbose=False, max_iter=10000, warm_start=False)
        except Exception as exc:
            return None, None, None, f"crashed: {exc}", np.inf
        if problem.status not in ("optimal", "optimal_inaccurate"):
            return None, None, None, problem.status, np.inf
        residual = np.linalg.norm(A@g.value-np.vstack([u_ini, u.value, y_ini, y.value]))
        return u.value, y.value, g.value, problem.status, float(residual)
