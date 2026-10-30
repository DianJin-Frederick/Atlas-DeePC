"""Reacher ports of the original trajectory-coefficient DPC formulations.

No trajectory SVD, affine centering, or Atlas coefficient bounds are used here.
Select-DPC is a small dependency-light port of the supplied Choose-Wisely code;
the original objective and five-pass refinement are retained. Contextual DPC
defaults to arXiv:2503.23890v2 Eq. (4), not our former Atlas-QP wrapper.
See docs/THREE_METHOD_COMPARISON.md for source and task-adaptation details.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from dataclasses import dataclass
import time

import cvxpy as cp
import numpy as np


# Native Choose-Wisely Reacher order: periodic joints, fingertip xy, velocity.
OUTPUT_INDICES = np.array([0, 1, 2, 3, 6, 7, 4, 5])


@dataclass(frozen=True)
class SelectConfig:
    past: int = 2
    horizon: int = 15
    columns: int = 200
    iterations: int = 5
    q_xy: float = 40000.0
    q_velocity: float = 10.0
    r: float = 10.0
    lambda_g1: float = 10.0
    lambda_projection: float = 10000.0
    lambda_slack: float = 5000000.0
    input_limit: float = 1.0


@dataclass(frozen=True)
class ContextualConfig:
    past: int = 2
    horizon: int = 10
    columns: int = 200
    q_xy: float = 1.0
    q_velocity: float = 0.0
    r: float = 0.1
    lambda_g_bar: float = 0.001
    lambda_slack: float = 1.0
    input_limit: float = 1.0
    # Paper Eq. (4): unsquared L2(g), L1(slack). Public code: squared L2.
    formulation: str = "paper_eq4"


def block_hankel(values, length):
    """Preserve complete time-ordered windows; never cross episode boundaries."""
    values = np.asarray(values, dtype=float)
    if length > len(values) or length <= 0:
        raise ValueError("Invalid Hankel window length")
    return np.stack([values[i:i + length].reshape(-1)
                     for i in range(len(values) - length + 1)], axis=1)


def reconstruct_offline_episodes(H, m=2, p=10, length=12, episode_steps=200):
    windows = episode_steps - length + 1
    if H.shape[0] != (m + p) * length or H.shape[1] % windows:
        raise ValueError("Offline library does not match the frozen episode layout")
    U = H[:m * length].reshape(length, m, -1)
    Y = H[m * length:].reshape(length, p, -1)
    inputs, outputs = [], []
    for start in range(0, H.shape[1], windows):
        uw, yw = U[:, :, start:start + windows], Y[:, :, start:start + windows]
        if not (np.allclose(uw[1:, :, :-1], uw[:-1, :, 1:], atol=1e-12, rtol=0)
                and np.allclose(yw[1:, :, :-1], yw[:-1, :, 1:], atol=1e-12, rtol=0)):
            raise ValueError("Offline windows overlap inconsistently")
        inputs.append(np.vstack([uw[:, :, 0], uw[-1, :, 1:].T]))
        outputs.append(np.vstack([yw[:, :, 0], yw[-1, :, 1:].T]))
    return inputs, outputs


def native_library(inputs, outputs, past, horizon):
    return (np.hstack([block_hankel(u, past + horizon) for u in inputs]),
            np.hstack([block_hankel(y[:, OUTPUT_INDICES], past + horizon)
                       for y in outputs]))


def native_reference(target):
    # Unregulated posture entries have zero Q weight. No Atlas goal is supplied.
    reference = np.zeros(8)
    reference[4:6] = target
    return reference


def select_l1(Hu, Hy, query_u, query_y, count):
    distances = (np.linalg.norm(Hu - np.asarray(query_u).reshape(-1, 1), ord=1, axis=0)
                 + np.linalg.norm(Hy - np.asarray(query_y).reshape(-1, 1), ord=1, axis=0))
    return np.argsort(distances)[:count]


class SelectSubproblem:
    """Choose-Wisely raw QP, including projection regularization and L1 slack."""
    def __init__(self, Hu, Hy, config):
        c = self.config = config
        self.g = cp.Variable(Hu.shape[1])
        self.u = cp.Variable(2 * c.horizon)
        self.y = cp.Variable(8 * c.horizon)
        self.slack = cp.Variable(8 * c.past)
        self.up = cp.Parameter(2 * c.past)
        self.yp = cp.Parameter(8 * c.past)
        self.ref = cp.Parameter(8 * c.horizon)
        Up, Uf = Hu[:2 * c.past], Hu[2 * c.past:]
        Yp, Yf = Hy[:8 * c.past], Hy[8 * c.past:]
        self.Q = np.diag([0., 0., 0., 0., c.q_xy, c.q_xy, c.q_velocity, c.q_velocity])
        objective = (cp.quad_form(self.y - self.ref, np.kron(np.eye(c.horizon), self.Q))
                     + c.r * cp.sum_squares(self.u) + c.lambda_slack * cp.norm1(self.slack))
        if c.lambda_g1:
            objective += c.lambda_g1 * cp.norm1(self.g)
        # This pinv is the ORIGINAL projection regularizer, not data compression.
        Z = np.vstack([Up, Uf, Yp, np.ones((1, Hu.shape[1]))])
        self.projection = np.linalg.pinv(Z) @ Z
        if c.lambda_projection:
            objective += c.lambda_projection * cp.sum_squares(
                (np.eye(Hu.shape[1]) - self.projection) @ self.g)
        self.problem = cp.Problem(cp.Minimize(objective), [
            Up @ self.g == self.up, Uf @ self.g == self.u,
            Yp @ self.g == self.yp + self.slack, Yf @ self.g == self.y,
            cp.sum(self.g) == 1, self.u <= c.input_limit, self.u >= -c.input_limit,
        ])
        self.previous_u = None
        self.open_loop_index = 1

    def solve(self, up, yp, reference):
        c = self.config
        self.up.value, self.yp.value = np.asarray(up).reshape(-1), np.asarray(yp).reshape(-1)
        self.ref.value = np.tile(reference, c.horizon)
        try:
            self.problem.solve(solver="OSQP", warm_start=False, max_iter=6000,
                               eps_abs=1e-5, eps_rel=1e-4, verbose=False)
            status = self.problem.status
        except cp.SolverError:
            status = "failed"
        if status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            action = self.u.value[:2].copy()
            self.previous_u = self.u.value.copy()
            self.open_loop_index = 1
        else:
            action = np.zeros(2)
            if self.previous_u is not None:
                candidate = self.previous_u[2 * self.open_loop_index:2 * (self.open_loop_index + 1)]
                self.open_loop_index += 1
                if candidate.size == 2:
                    action = candidate.copy()
        # Preserve the reference implementation's history/query update order.
        self.up.value = np.r_[self.up.value[2:], action]
        return action, status

    def planned(self):
        c = self.config
        u = (np.zeros(2 * (c.past + c.horizon)) if self.u.value is None
             else np.r_[self.up.value, self.u.value])
        y = (np.zeros(8 * (c.past + c.horizon)) if self.y.value is None
             else np.r_[self.yp.value, self.y.value])
        return u, y


class SelectDPC:
    def __init__(self, Hu, Hy, config=SelectConfig()):
        if not 1 <= config.columns <= min(400, Hu.shape[1]):
            raise ValueError("Use 1..400 raw columns; the authors' >400 branch factorizes data")
        if config.iterations < 1:
            raise ValueError("Select-DPC must have at least one refinement")
        self.Hu, self.Hy, self.config = Hu, Hy, config
        self.subproblem = None

    def step(self, state, reference, **unused):
        c = self.config
        timings = {"selection_s": 0., "matrix_s": 0., "solve_s": 0.}
        statuses = []
        for iteration in range(c.iterations):
            if self.subproblem is None:
                self.yp = np.tile(state, c.past)
                self.up = np.zeros(2 * c.past)
                query_y = np.tile(state, c.past + c.horizon)
                query_u = np.zeros(2 * (c.past + c.horizon))
            else:
                if iteration == 0:
                    self.yp = np.r_[self.yp[8:], state]
                query_u, query_y = self.subproblem.planned()
            stage = time.perf_counter()
            selected = select_l1(self.Hu, self.Hy, query_u, query_y, c.columns)
            timings["selection_s"] += time.perf_counter() - stage
            stage = time.perf_counter()
            new = SelectSubproblem(self.Hu[:, selected], self.Hy[:, selected], c)
            if self.subproblem is not None:
                new.previous_u = self.subproblem.previous_u
                new.open_loop_index = self.subproblem.open_loop_index
            self.subproblem = new
            timings["matrix_s"] += time.perf_counter() - stage
            stage = time.perf_counter()
            action, status = new.solve(self.up, self.yp, reference)
            timings["solve_s"] += time.perf_counter() - stage
            statuses.append(status)
        self.up = np.r_[self.up[2:], action]
        prediction = (np.full(8, np.nan) if new.y.value is None else new.y.value[:8].copy())
        return action, prediction, {**timings, "statuses": statuses,
                                    "selected": selected, "coefficient_count": c.columns}


class ContextualDPC:
    """Eq. (4)--(8) on raw, z-score-normalized Reacher trajectory columns.

The paper core uses actual torque inputs here. Vehicle/quadrotor spatial
alignment and incremental-input augmentation are not silently transplanted to
the fixed-base arm; this task adaptation is explicit in the benchmark contract.
"""
    def __init__(self, Hu, Hy, input_samples, output_samples, config=ContextualConfig()):
        c = self.config = config
        if not 1 <= c.columns <= Hu.shape[1]:
            raise ValueError("Invalid selected column count")
        if c.formulation not in ("paper_eq4", "public_code_qp"):
            raise ValueError("Unknown contextual objective version")
        self.u_mean = np.mean(input_samples, axis=0)
        self.u_scale = np.maximum(np.std(input_samples, axis=0), 1e-8)
        self.y_mean = np.mean(output_samples, axis=0)
        self.y_scale = np.maximum(np.std(output_samples, axis=0), 1e-8)
        L = c.past + c.horizon
        self.Hu = (Hu - np.tile(self.u_mean, L)[:, None]) / np.tile(self.u_scale, L)[:, None]
        self.Hy = (Hy - np.tile(self.y_mean, L)[:, None]) / np.tile(self.y_scale, L)[:, None]
        self.q = np.array([0., 0., 0., 0., c.q_xy, c.q_xy, c.q_velocity, c.q_velocity])
        self.g = cp.Variable(c.columns)
        self.u = cp.Variable(2 * c.horizon)
        self.y = cp.Variable(8 * c.horizon)
        self.slack = cp.Variable(8 * c.past)
        self.Up = cp.Parameter((2 * c.past, c.columns))
        self.Uf = cp.Parameter((2 * c.horizon, c.columns))
        self.Yp = cp.Parameter((8 * c.past, c.columns))
        self.Yf = cp.Parameter((8 * c.horizon, c.columns))
        self.up, self.yp = cp.Parameter(2 * c.past), cp.Parameter(8 * c.past)
        self.ref = cp.Parameter(8 * c.horizon)
        du = self.u[2:] - self.u[:-2]
        objective = (cp.sum(cp.multiply(np.tile(self.q, c.horizon), cp.square(self.y - self.ref)))
                     + c.r * (cp.sum_squares(self.u) + cp.sum_squares(du)))
        self.lambda_g = c.lambda_g_bar * c.columns
        if c.formulation == "paper_eq4":
            objective += self.lambda_g * cp.norm(self.g, 2) + c.lambda_slack * cp.norm1(self.slack)
        else:
            objective += self.lambda_g * cp.sum_squares(self.g) + c.lambda_slack * cp.sum_squares(self.slack)
        lower = np.tile((-c.input_limit - self.u_mean) / self.u_scale, c.horizon)
        upper = np.tile((c.input_limit - self.u_mean) / self.u_scale, c.horizon)
        self.problem = cp.Problem(cp.Minimize(objective), [
            self.Up @ self.g == self.up, self.Yp @ self.g == self.yp + self.slack,
            self.Uf @ self.g == self.u, self.Yf @ self.g == self.y,
            cp.sum(self.g) == 1, self.u >= lower, self.u <= upper,
        ])

    def indices(self, upast, ypast, reference):
        c = self.config
        yp = ((ypast - self.y_mean) / self.y_scale).reshape(-1)
        ref = np.tile((reference - self.y_mean) / self.y_scale, c.horizon)
        past = np.linalg.norm(self.Hy[:8 * c.past] - yp[:, None], axis=0)
        delta = self.Hy[8 * c.past:] - ref[:, None]
        future = np.sqrt(np.sum(np.tile(self.q, c.horizon)[:, None] * delta**2, axis=0))
        zscore = lambda x: (x - x.mean()) / max(float(x.std()), 1e-12)
        return np.argsort(zscore(past) + zscore(future))[:c.columns], yp, ref

    def step(self, state, reference, *, upast, ypast):
        c = self.config
        stage = time.perf_counter()
        selected, yp, ref = self.indices(upast, ypast, reference)
        selection_time = time.perf_counter() - stage
        stage = time.perf_counter()
        self.Up.value, self.Uf.value = self.Hu[:2 * c.past, selected], self.Hu[2 * c.past:, selected]
        self.Yp.value, self.Yf.value = self.Hy[:8 * c.past, selected], self.Hy[8 * c.past:, selected]
        self.up.value = ((upast - self.u_mean) / self.u_scale).reshape(-1)
        self.yp.value, self.ref.value = yp, ref
        matrix_time = time.perf_counter() - stage
        stage = time.perf_counter()
        try:
            # Eq. (4)'s unsquared L2 norm requires a conic solver, not OSQP.
            if c.formulation == "paper_eq4":
                self.problem.solve(solver="CLARABEL", warm_start=True, ignore_dpp=True, max_iter=10000)
            else:
                self.problem.solve(solver="OSQP", warm_start=True, ignore_dpp=True,
                                   max_iter=10000, eps_abs=1e-5, eps_rel=1e-4)
            status = self.problem.status
        except cp.SolverError:
            status = "failed"
        solve_time = time.perf_counter() - stage
        if status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            action = self.u.value[:2] * self.u_scale + self.u_mean
            prediction = self.y.value[:8] * self.y_scale + self.y_mean
        else:
            action, prediction = np.zeros(2), np.full(8, np.nan)
        return action, prediction, {"selection_s": selection_time, "matrix_s": matrix_time,
                                    "solve_s": solve_time, "statuses": [status],
                                    "selected": selected, "coefficient_count": c.columns}
