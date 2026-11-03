"""Retained single-track plant, unchanged RK4 equations and vehicle parameters."""
import numpy as np
from . import paths  # makes the licensed simulator importable
from .outputs import wrap_angle
from f110_gym.envs.dynamic_models import vehicle_dynamics_st

VEHICLE_PARAMS = {
    "mu": 1.0489, "C_Sf": 4.718, "C_Sr": 5.4562, "lf": 0.15875,
    "lr": 0.17145, "h": 0.074, "m": 3.74, "I": 0.04712,
    "s_min": -0.4189, "s_max": 0.4189, "sv_min": -2.0, "sv_max": 2.0,
    "v_switch": 7.319, "a_max": 3.0, "v_min": 0.0, "v_max": 8.0,
}


def dynamics_derivative(state, action):
    p = VEHICLE_PARAMS
    return vehicle_dynamics_st(
        state, action, p["mu"], p["C_Sf"], p["C_Sr"], p["lf"], p["lr"],
        p["h"], p["m"], p["I"], p["s_min"], p["s_max"], p["sv_min"],
        p["sv_max"], p["v_switch"], p["a_max"], p["v_min"], p["v_max"]
    )


def rk4_step(state: np.ndarray, action: np.ndarray, dt: float):
    action = np.clip(np.asarray(action, dtype=float), [-2.0, -3.0], [2.0, 3.0])
    k1 = dynamics_derivative(state, action)
    k2 = dynamics_derivative(state + 0.5 * dt * k1, action)
    k3 = dynamics_derivative(state + 0.5 * dt * k2, action)
    k4 = dynamics_derivative(state + dt * k3, action)
    nxt = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    nxt[4] = float(wrap_angle(nxt[4]))
    return nxt


