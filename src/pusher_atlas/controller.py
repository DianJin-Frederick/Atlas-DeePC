"""Cached local affine DeePC controllers for Pusher."""

from __future__ import annotations

from time import perf_counter

import cvxpy as cp
import numpy as np

from .config import PusherAtlasConfig
from .environment import (
    ACTION_DIM,
    OBJECT_VXY,
    OBJECT_XY,
    PHASE_APPROACH,
    PHASE_HOLD,
    QPOS,
    QVEL,
    STATE_DIM,
    TIP,
)


class LocalAffineDeePC:
    """Cached current-chart DeePC QP with no identified state-space model."""

    def __init__(self, chart, cfg: PusherAtlasConfig):
        self.chart = chart
        self.cfg = cfg
        m, p = ACTION_DIM, STATE_DIM
        self.u_ini = cp.Parameter((m * cfg.t_ini, 1))
        self.y_ini = cp.Parameter((p * cfg.t_ini, 1))
        self.previous_u = cp.Parameter((m, 1))
        self.u_prior = cp.Parameter((m, 1))
        self.tip_reference = cp.Parameter((cfg.t_f, 3))
        self.object_reference = cp.Parameter((cfg.t_f, 2))
        self.u = cp.Variable((m * cfg.t_f, 1))
        self.y = cp.Variable((p * cfg.t_f, 1))
        self.g = cp.Variable((chart["rank"], 1))
        self.g_anchor = cp.Parameter((chart["rank"], 1))
        self.sigma_u = cp.Variable((m * cfg.t_ini, 1))
        self.sigma_y = cp.Variable((p * cfg.t_ini, 1))

        scale = chart["behavior_scale"]
        past_u_scale = scale[: m * cfg.t_ini].reshape(-1, 1)
        past_y_start = m * (cfg.t_ini + cfg.t_f)
        past_y_scale = scale[
            past_y_start : past_y_start + p * cfg.t_ini
        ].reshape(-1, 1)

        behavior = chart["h_bar"].reshape(-1, 1) + chart["basis"] @ self.g
        trajectory = cp.vstack(
            [self.u_ini + self.sigma_u, self.u, self.y_ini + self.sigma_y, self.y]
        )
        bounds = chart["coefficient_bound"].reshape(-1, 1)
        constraints = [trajectory == behavior, self.g >= -bounds, self.g <= bounds]
        constraints += [self.u >= -cfg.torque_limit, self.u <= cfg.torque_limit]

        phase = int(chart["phase"])
        tip_weight = cfg.tip_weight_approach if phase == PHASE_APPROACH else cfg.tip_weight_push
        object_weight = cfg.object_weight_approach if phase == PHASE_APPROACH else cfg.object_weight_push
        representative_q = chart["representative_y"][QPOS]
        objective = 0
        previous = self.previous_u
        for step in range(cfg.t_f):
            us = self.u[step * m : (step + 1) * m]
            ys = self.y[step * p : (step + 1) * p]
            multiplier = cfg.terminal_multiplier if step == cfg.t_f - 1 else 1.0
            objective += multiplier * tip_weight * cp.sum_squares(
                ys[TIP] - cp.reshape(self.tip_reference[step], (3, 1), order="C")
            )
            objective += multiplier * object_weight * cp.sum_squares(
                ys[OBJECT_XY]
                - cp.reshape(self.object_reference[step], (2, 1), order="C")
            )
            if phase in (PHASE_APPROACH, PHASE_HOLD):
                objective += cfg.object_velocity_weight * cp.sum_squares(ys[OBJECT_VXY])
            objective += cfg.q_velocity_weight * cp.sum_squares(ys[QVEL])
            objective += cfg.q_posture_weight * cp.sum_squares(
                ys[QPOS] - representative_q.reshape(7, 1)
            )
            objective += cfg.input_weight * cp.sum_squares(us)
            objective += cfg.input_rate_weight * cp.sum_squares(us - previous)
            previous = us
        objective += cfg.lam_ini * (
            cp.sum_squares(cp.multiply(1.0 / past_u_scale, self.sigma_u))
            + cp.sum_squares(cp.multiply(1.0 / past_y_scale, self.sigma_y))
        )
        objective += cfg.lam_g * cp.sum_squares(self.g)
        objective += cfg.lam_behavior_anchor * cp.sum_squares(self.g - self.g_anchor)
        objective += cfg.policy_prior_weight * cp.sum_squares(
            self.u[:m] - self.u_prior
        )
        self.problem = cp.Problem(cp.Minimize(objective), constraints)

    def solve(
        self,
        u_ini,
        y_ini,
        previous_u,
        tip_reference,
        object_reference,
        g_anchor,
        u_prior,
    ):
        self.u_ini.value = np.asarray(u_ini, dtype=float).reshape(-1, 1)
        self.y_ini.value = np.asarray(y_ini, dtype=float).reshape(-1, 1)
        self.previous_u.value = np.asarray(previous_u, dtype=float).reshape(-1, 1)
        self.tip_reference.value = np.asarray(tip_reference, dtype=float)
        self.object_reference.value = np.asarray(object_reference, dtype=float)
        self.g_anchor.value = np.asarray(g_anchor, dtype=float).reshape(-1, 1)
        self.u_prior.value = np.asarray(u_prior, dtype=float).reshape(-1, 1)
        started = perf_counter()
        osqp_error = None
        try:
            self.problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False,
                max_iter=10000,
                eps_abs=5e-4,
                eps_rel=5e-4,
                polish=False,
            )
        except Exception as exc:
            osqp_error = exc
        if osqp_error is not None or self.problem.status not in (
            cp.OPTIMAL,
            cp.OPTIMAL_INACCURATE,
        ):
            try:
                self.problem.solve(
                    solver=cp.CLARABEL,
                    warm_start=True,
                    verbose=False,
                    max_iter=400,
                )
                suffix = ":clarabel_retry"
            except Exception as exc:
                return None, {
                    "status": f"both_failed:{type(osqp_error).__name__}/{type(exc).__name__}",
                    "solve_time": perf_counter() - started,
                }
        else:
            suffix = ":osqp"
        elapsed = perf_counter() - started
        if self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or self.u.value is None:
            return None, {"status": str(self.problem.status), "solve_time": elapsed}
        return self.u.value[:ACTION_DIM, 0].copy(), {
            "status": str(self.problem.status) + suffix,
            "solve_time": elapsed,
            "objective": float(self.problem.value),
            "slack_norm": float(np.linalg.norm(self.sigma_u.value) + np.linalg.norm(self.sigma_y.value)),
            "coefficient_norm": float(np.linalg.norm(self.g.value)),
            "predicted_y": self.y.value.reshape(self.cfg.t_f, STATE_DIM),
        }


def geometric_references(state, goal_xy, phase, cfg: PusherAtlasConfig):
    obj = np.asarray(state[OBJECT_XY], dtype=float)
    delta = np.asarray(goal_xy, dtype=float) - obj
    distance = np.linalg.norm(delta)
    direction = delta / max(distance, 1e-9)
    if phase == PHASE_APPROACH:
        # Controller outputs are task-relative: object-minus-goal and
        # fingertip-minus-object (xy), with raw fingertip z.
        object_reference = np.repeat((obj - goal_xy)[None, :], cfg.t_f, axis=0)
        tip_xy = -cfg.precontact_offset * direction
        tip_reference = np.repeat(np.r_[tip_xy, -0.275][None, :], cfg.t_f, axis=0)
    else:
        # An aggressive finite-horizon path prevents the regularizers from
        # preferring a stationary contact. Feasibility is still determined only
        # by the local behavioral subspace.
        progress = distance * (np.arange(cfg.t_f) + 1) / cfg.t_f
        object_world_reference = obj[None, :] + progress[:, None] * direction[None, :]
        object_reference = object_world_reference - np.asarray(goal_xy)[None, :]
        if phase == PHASE_HOLD:
            object_reference[:] = 0.0
            offset = cfg.precontact_offset
        else:
            offset = cfg.push_contact_offset
        tip_relative = np.repeat((-offset * direction)[None, :], cfg.t_f, axis=0)
        tip_reference = np.column_stack([tip_relative, np.full(cfg.t_f, -0.275)])
    return tip_reference, object_reference, direction


def update_phase(state, goal_xy, phase, step, cfg: PusherAtlasConfig):
    obj = state[OBJECT_XY]
    tip = state[TIP]
    delta = np.asarray(goal_xy) - obj
    distance = np.linalg.norm(delta)
    direction = delta / max(distance, 1e-9)
    precontact = np.r_[obj - cfg.precontact_offset * direction, -0.275]
    if distance <= cfg.success_radius:
        return PHASE_HOLD
    if phase == PHASE_APPROACH:
        if np.linalg.norm(tip - precontact) <= cfg.approach_tolerance or step >= cfg.online_approach_steps:
            return 1
        return PHASE_APPROACH
    if phase == PHASE_HOLD:
        return PHASE_HOLD
    if np.linalg.norm(tip[:2] - obj) > cfg.contact_lost_distance:
        return PHASE_APPROACH
    return phase
