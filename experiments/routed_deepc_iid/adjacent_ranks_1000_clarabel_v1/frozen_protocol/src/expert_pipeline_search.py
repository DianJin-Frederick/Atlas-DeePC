import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cvxpy as cp
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from sklearn.cluster import KMeans


REACHER_ENV_ID = "Reacher-v5"
M = 2
P = 10
T_INI = 2
T_F = 10
L = T_INI + T_F
K = 8
DATA_PATH = Path("expert_offline_hankel.npy")
LABEL_PATH = Path("expert_prediction_error_labels.npz")
RESULTS_PATH = Path("expert_search_results.json")


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


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def ik_target(target_xy, elbow="down", link_lengths=(0.1, 0.1)):
    x, y = np.asarray(target_xy, dtype=float).reshape(2)
    l1, l2 = link_lengths
    radius = np.hypot(x, y)
    max_radius = l1 + l2 - 1e-6
    if radius > max_radius:
        x *= max_radius / radius
        y *= max_radius / radius
    c2 = np.clip((x * x + y * y - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -1.0, 1.0)
    q2 = np.arccos(c2)
    if elbow == "down":
        q2 = -q2
    q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    return np.array([wrap_to_pi(q1), wrap_to_pi(q2)])


def ik_output_features(target_xy, elbow="down"):
    q1, q2 = ik_target(target_xy, elbow=elbow)
    qsum = q1 + q2
    return np.array([
        np.cos(q1),
        np.cos(q2),
        np.sin(q1),
        np.sin(q2),
        np.cos(qsum),
        np.sin(qsum),
    ])


def ik_action(obs, kp=5.0, kd=1.0, limit=0.5, elbow="down"):
    obs = np.asarray(obs, dtype=float).reshape(-1)
    q = np.array([np.arctan2(obs[2], obs[0]), np.arctan2(obs[3], obs[1])])
    dq = obs[6:8]
    q_des = ik_target(obs[4:6], elbow=elbow)
    return np.clip(kp * wrap_to_pi(q_des - q) - kd * dq, -limit, limit)


def collect_expert_hankel(seed=123, episodes=140, episode_len=180, limit=0.5, noise_std=0.035):
    rng = np.random.default_rng(seed)
    columns = []
    for ep in range(episodes):
        env = gym.make(REACHER_ENV_ID, max_episode_steps=episode_len + 5)
        obs, _ = env.reset(seed=seed + ep)
        elbow = "down" if ep % 2 == 0 else "up"
        U = []
        Y = []
        for t in range(episode_len):
            base = ik_action(obs, kp=5.0, kd=1.0, limit=limit, elbow=elbow)
            # Add decaying exploration so the dataset contains both approach and local regulation.
            noise_scale = noise_std * (1.0 if t < 80 else 0.35)
            action = np.clip(base + rng.normal(0.0, noise_scale, size=M), -limit, limit)
            obs, reward, terminated, truncated, info = env.step(action)
            U.append(action.copy())
            Y.append(reacher_output_from_obs(obs))
            if terminated or truncated:
                break
        env.close()
        U = np.asarray(U, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if U.shape[0] < L:
            continue
        for start in range(U.shape[0] - L + 1):
            u_window = U[start:start + L].reshape(L * M)
            y_window = Y[start:start + L].reshape(L * P)
            columns.append(np.concatenate([u_window, y_window]))
    H = np.asarray(columns, dtype=float).T
    np.save(DATA_PATH, H)
    return H


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


def build_W_Y(H):
    U_p, U_f, Y_p, Y_f = partition_hankel_rows(H)
    return np.vstack([U_p, U_f, Y_p]), Y_f


def standardize(X, idx, eps=1e-8):
    mu = X[:, idx].mean(axis=1, keepdims=True)
    sd = np.maximum(X[:, idx].std(axis=1, keepdims=True), eps)
    return (X - mu) / sd


def fit_ridge(W, Y, lam):
    A = W @ W.T + lam * np.eye(W.shape[0])
    return (Y @ W.T) @ np.linalg.solve(A, np.eye(A.shape[0]))


def fit_predictors(W, Y, labels, train_idx, lam=1e-6, min_size=100):
    predictors = []
    valid = []
    for k in range(K):
        idx = train_idx[labels[train_idx] == k]
        if idx.size < min_size:
            predictors.append(None)
            valid.append(False)
        else:
            predictors.append(fit_ridge(W[:, idx], Y[:, idx], lam))
            valid.append(True)
    return predictors, np.asarray(valid)


def refine_labels(W, Y, labels, train_idx, lam=1e-6, min_size=100, margin=1.0):
    predictors, valid = fit_predictors(W, Y, labels, train_idx, lam=lam, min_size=min_size)
    valid_ids = np.where(valid)[0]
    moved = 0
    new_labels = labels.copy()
    for idx in train_idx:
        errs = np.full(K, np.inf)
        for k in valid_ids:
            e = predictors[k] @ W[:, idx] - Y[:, idx]
            errs[k] = float(np.mean(e * e))
        best = int(np.argmin(errs))
        old = int(labels[idx])
        if best != old and errs[best] * margin < errs[old]:
            new_labels[idx] = best
            moved += 1
    return new_labels, moved


def learn_prediction_error_atlas(H, seed=0, label_path=LABEL_PATH):
    W, Y = build_W_Y(H)
    N = H.shape[1]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    train_idx = np.sort(perm[: int(0.8 * N)])
    val_idx = np.sort(perm[int(0.8 * N):])
    Z = standardize(W, train_idx)
    labels = KMeans(n_clusters=K, n_init=10, random_state=seed).fit_predict(Z.T)
    labels_initial = labels.copy()
    for _ in range(2):
        labels, moved = refine_labels(W, Y, labels, train_idx, lam=1e-6, min_size=100, margin=1.0)
        if moved == 0:
            break
    np.savez(
        label_path,
        labels=labels,
        labels_initial=labels_initial,
        train_idx=train_idx,
        val_idx=val_idx,
        t_ini=np.array(T_INI),
        t_f=np.array(T_F),
        l=np.array(L),
        m=np.array(M),
        p=np.array(P),
    )
    return labels


def local_svd_basis(X, rank=None, energy=None):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    if energy is not None:
        spectral_energy = np.cumsum(s * s) / max(float(np.sum(s * s)), 1e-30)
        r = int(np.searchsorted(spectral_energy, energy) + 1)
    elif rank is not None:
        r = int(rank)
    else:
        r = U.shape[1]
    return U[:, :min(r, U.shape[1])]


def fixed_rank_left_svd_basis(X, rank):
    """Return an exact fixed-rank left-SVD subspace through its covariance matrix."""
    X = np.asarray(X, dtype=float)
    rank = min(int(rank), X.shape[0])
    covariance = X @ X.T
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:rank]
    return eigenvectors[:, order]


def compute_bases(H, labels, rank=None, energy=None):
    bases = []
    n_charts = max(K, int(np.max(labels)) + 1) if labels.size else K
    for k in range(n_charts):
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            bases.append(None)
        else:
            bases.append(local_svd_basis(H[:, idx], rank=rank, energy=energy))
    return bases


def compute_affine_bases(H, labels, rank=None, energy=None):
    bases = []
    n_charts = max(K, int(np.max(labels)) + 1) if labels.size else K
    for k in range(n_charts):
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            bases.append(None)
            continue
        X = H[:, idx]
        mu = X.mean(axis=1, keepdims=True)
        B = local_svd_basis(X - mu, rank=rank, energy=energy)
        bases.append((mu, B))
    return bases


def fit_affine_k_subspaces(
    H,
    n_clusters=K,
    rank=12,
    seed=0,
    max_iter=20,
    change_tolerance=1e-3,
    standardize_rows=True,
):
    """Cluster trajectory columns by distance to affine local subspaces.

    KMeans supplies a deterministic initialization. Each iteration alternates
    between fitting a centered SVD basis per cluster and assigning every column
    to the affine subspace with the smallest squared reconstruction residual.
    """
    H = np.asarray(H, dtype=float)
    if H.ndim != 2:
        raise ValueError(f"H must be a matrix, got shape {H.shape}.")
    if not 1 <= n_clusters <= H.shape[1]:
        raise ValueError("n_clusters must be between 1 and the number of columns.")
    if rank < 1 or rank >= H.shape[0]:
        raise ValueError("rank must be positive and smaller than the ambient dimension.")

    row_mean = H.mean(axis=1, keepdims=True) if standardize_rows else np.zeros((H.shape[0], 1))
    row_scale = np.maximum(H.std(axis=1, keepdims=True), 1e-8) if standardize_rows else np.ones((H.shape[0], 1))
    Z = (H - row_mean) / row_scale
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(Z.T)
    objective_history = []
    change_history = []

    for _ in range(max_iter):
        centers = []
        bases = []
        distances = np.empty((n_clusters, H.shape[1]), dtype=float)
        for cluster_id in range(n_clusters):
            idx = np.where(labels == cluster_id)[0]
            if idx.size == 0:
                raise RuntimeError(f"Affine K-subspaces produced empty cluster {cluster_id}.")
            X = Z[:, idx]
            center = X.mean(axis=1, keepdims=True)
            basis = fixed_rank_left_svd_basis(X - center, rank=rank)
            centered_all = Z - center
            total_energy = np.sum(centered_all * centered_all, axis=0)
            projected_energy = np.sum((basis.T @ centered_all) ** 2, axis=0)
            distances[cluster_id] = np.maximum(total_energy - projected_energy, 0.0)
            centers.append(center)
            bases.append(basis)

        new_labels = np.argmin(distances, axis=0)
        minimum_distances = distances[new_labels, np.arange(H.shape[1])]
        missing = sorted(set(range(n_clusters)) - set(np.unique(new_labels)))
        if missing:
            farthest = np.argsort(minimum_distances)[::-1]
            used = set()
            for cluster_id, point_idx in zip(missing, farthest):
                while point_idx in used:
                    point_idx = farthest[len(used)]
                new_labels[point_idx] = cluster_id
                used.add(int(point_idx))

        changed_fraction = float(np.mean(new_labels != labels))
        objective_history.append(float(np.mean(minimum_distances)))
        change_history.append(changed_fraction)
        labels = new_labels
        if changed_fraction <= change_tolerance:
            break

    physical_bases = compute_affine_bases(H, labels, rank=rank)[:n_clusters]
    return {
        "labels": labels,
        "affine_bases": physical_bases,
        "row_mean": row_mean,
        "row_scale": row_scale,
        "objective_history": np.asarray(objective_history, dtype=float),
        "change_history": np.asarray(change_history, dtype=float),
    }


def concatenate_affine_bases(affine_bases, global_offset=None):
    """Represent several affine charts as one offset plus concatenated directions.

    Each chart center contributes a displacement from the global offset, followed
    by that chart's SVD directions. The result uses the existing ``(offset, basis)``
    interface understood by the DeePC solver.
    """
    valid = [A for A in affine_bases if A is not None]
    if not valid:
        raise ValueError("At least one affine basis is required.")
    centers = [np.asarray(center, dtype=float) for center, _ in valid]
    if global_offset is None:
        global_offset = np.mean(np.stack(centers, axis=0), axis=0)
    else:
        global_offset = np.asarray(global_offset, dtype=float).reshape(centers[0].shape)
    pieces = []
    component_slices = []
    column_start = 0
    for center, basis in valid:
        chart_matrix = np.hstack([center - global_offset, np.asarray(basis, dtype=float)])
        pieces.append(chart_matrix)
        column_stop = column_start + chart_matrix.shape[1]
        component_slices.append((column_start, column_stop))
        column_start = column_stop
    return (global_offset, np.hstack(pieces)), np.asarray(component_slices, dtype=int)


def compute_bounded_bases(H, labels, rank=None, energy=None, affine=False, coef_bound_scale=3.0):
    bases = []
    n_charts = max(K, int(np.max(labels)) + 1) if labels.size else K
    for k in range(n_charts):
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            bases.append(None)
            continue
        X = H[:, idx]
        offset = X.mean(axis=1, keepdims=True) if affine else np.zeros((H.shape[0], 1))
        B = local_svd_basis(X - offset, rank=rank, energy=energy)
        coords = B.T @ (X - offset)
        coord_mean = coords.mean(axis=1, keepdims=True)
        coord_std = np.maximum(coords.std(axis=1, keepdims=True), 1e-8)
        bases.append({
            "offset": offset,
            "basis": B,
            "g_min": coord_mean - coef_bound_scale * coord_std,
            "g_max": coord_mean + coef_bound_scale * coord_std,
        })
    return bases


def unpack_basis(A):
    if isinstance(A, dict):
        return A["offset"], A["basis"], A.get("g_min"), A.get("g_max")
    if isinstance(A, tuple):
        return A[0], A[1], None, None
    return 0.0, A, None, None


def combine_chart_matrices(matrices):
    """Build one DeePC matrix from several chart bases or raw column blocks."""
    valid = [A for A in matrices if A is not None]
    if not valid:
        return None
    if all(isinstance(A, dict) for A in valid):
        offsets = [np.asarray(A["offset"], dtype=float) for A in valid]
        if any(np.linalg.norm(offset - offsets[0]) > 1e-8 for offset in offsets[1:]):
            raise ValueError("Mixture charts currently require a shared affine offset.")
        g_min = [A.get("g_min") for A in valid]
        g_max = [A.get("g_max") for A in valid]
        return {
            "offset": offsets[0],
            "basis": np.hstack([A["basis"] for A in valid]),
            "g_min": np.vstack(g_min) if all(g is not None for g in g_min) else None,
            "g_max": np.vstack(g_max) if all(g is not None for g in g_max) else None,
        }
    if all(isinstance(A, tuple) for A in valid):
        offsets = [np.asarray(A[0], dtype=float) for A in valid]
        if any(np.linalg.norm(offset - offsets[0]) > 1e-8 for offset in offsets[1:]):
            raise ValueError("Mixture charts currently require a shared affine offset.")
        return offsets[0], np.hstack([A[1] for A in valid])
    if any(isinstance(A, (tuple, dict)) for A in valid):
        raise ValueError("Cannot mix raw matrices and basis dictionaries in one DeePC solve.")
    return np.hstack(valid)


def past_residual(A, u_ini, y_ini):
    if isinstance(A, (tuple, dict)):
        mu, B, _, _ = unpack_basis(A)
        U_p, U_f, Y_p, Y_f = partition_hankel_rows(B)
        mu_U_p, mu_U_f, mu_Y_p, mu_Y_f = partition_hankel_rows(mu)
        A_past = np.vstack([U_p, Y_p])
        mu_past = np.vstack([mu_U_p, mu_Y_p])
    else:
        U_p, U_f, Y_p, Y_f = partition_hankel_rows(A)
        A_past = np.vstack([U_p, Y_p])
        mu_past = 0.0
    b = np.vstack([u_ini, y_ini])
    coef = np.linalg.lstsq(A_past, b - mu_past, rcond=None)[0]
    return float(np.linalg.norm(mu_past + A_past @ coef - b) / (np.linalg.norm(b) + 1e-8))


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


class DeePCController:
    def __init__(self, Q, R, Q_terminal, lam_g, lam_ini, lam_du, u_limit, solver):
        self.Q = Q
        self.R = R
        self.Q_terminal = Q_terminal
        self.lam_g = lam_g
        self.lam_ini = lam_ini
        self.lam_du = lam_du
        self.u_limit = u_limit
        self.solver = solver

    def solve(self, A, u_ini, y_ini, y_ref):
        affine_offset, A_matrix, g_min, g_max = unpack_basis(A)
        g_sum = A.get("g_sum") if isinstance(A, dict) else None
        g_nonnegative = bool(A.get("g_nonnegative", False)) if isinstance(A, dict) else False
        g_negative_weight = float(A.get("g_negative_weight", 0.0)) if isinstance(A, dict) else 0.0
        u = cp.Variable((M * T_F, 1))
        y = cp.Variable((P * T_F, 1))
        g = cp.Variable((A_matrix.shape[1], 1))
        sigma = cp.Variable((P * T_INI, 1))
        b = cp.vstack([u_ini, u, y_ini + sigma, y])
        cost = self.lam_ini * cp.sum_squares(sigma) + self.lam_g * cp.sum_squares(g)
        if g_negative_weight > 0.0:
            cost += g_negative_weight * cp.sum(cp.pos(-g))
        for j in range(T_F):
            yj = y[j * P:(j + 1) * P]
            rj = y_ref[j * P:(j + 1) * P]
            uj = u[j * M:(j + 1) * M]
            cost += cp.quad_form(yj - rj, self.Q) + cp.quad_form(uj, self.R)
            if j == T_F - 1:
                cost += cp.quad_form(yj - rj, self.Q_terminal)
        for j in range(T_F - 1):
            cost += self.lam_du * cp.sum_squares(u[(j + 1) * M:(j + 2) * M] - u[j * M:(j + 1) * M])
        constraints = [b == affine_offset + A_matrix @ g, u <= self.u_limit, u >= -self.u_limit]
        if g_min is not None:
            constraints.append(g >= g_min)
        if g_max is not None:
            constraints.append(g <= g_max)
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


@dataclass
class Variant:
    name: str
    mode: str = "basis"
    rank: int = 80
    energy: float | None = None
    max_cols: int = 1200
    try_top: int = 4
    u_limit: float = 0.5
    q_xy: float = 1000.0
    q_dq: float = 1.0
    q_joint: float = 0.0
    q_sum: float = 0.0
    q_terminal_xy: float = 0.0
    q_terminal_dq: float = 0.0
    q_terminal_joint: float = 0.0
    q_terminal_sum: float = 0.0
    r: float = 0.01
    lam_g: float = 1.0
    lam_ini: float = 100000.0
    lam_du: float = 1.0
    solver: str = "CLARABEL"
    steps: int = 120
    warmup_scale: float = 0.05
    seed: int = 42
    chart_selection: str = "first_feasible"
    selection_residual_weight: float = 0.0
    selection_g_weight: float = 0.0
    selection_past_weight: float = 0.0
    selection_zero_action_weight: float = 0.0
    selection_min_action_norm: float = 0.0
    selection_relevance_weight: float = 0.0
    relevance_past_weight: float = 0.0
    chart_switch_cost: float = 0.0
    coef_bound_scale: float | None = None
    coef_bound_affine: bool = False
    mixture_top: int = 3
    fixed_chart: int | None = None
    use_ik_ref: bool = False
    ik_elbow: str = "down"
    feedback_blend: float = 0.0
    feedback_kp: float = 5.0
    feedback_kd: float = 1.0
    feedback_fallback: bool = False


def predicted_tracking_score(
    y_sol,
    u_sol,
    y_ref,
    q_xy,
    q_dq,
    r,
    lam_du,
    g_sol=None,
    residual=0.0,
    residual_weight=0.0,
    g_weight=0.0,
    q_joint=0.0,
    q_sum=0.0,
    q_terminal_xy=0.0,
    q_terminal_dq=0.0,
    q_terminal_joint=0.0,
    q_terminal_sum=0.0,
):
    y_arr = np.asarray(y_sol, dtype=float).reshape(T_F, P)
    u_arr = np.asarray(u_sol, dtype=float).reshape(T_F, M)
    ref_arr = np.asarray(y_ref, dtype=float).reshape(T_F, P)
    xy_err = y_arr[:, 6:8] - ref_arr[:, 6:8]
    dq_err = y_arr[:, 4:6] - ref_arr[:, 4:6]
    joint_err = y_arr[:, 0:4] - ref_arr[:, 0:4]
    sum_err = y_arr[:, 8:10] - ref_arr[:, 8:10]
    score = q_xy * np.sum(xy_err * xy_err) + q_dq * np.sum(dq_err * dq_err)
    score += q_joint * np.sum(joint_err * joint_err) + q_sum * np.sum(sum_err * sum_err)
    score += q_terminal_xy * np.sum(xy_err[-1] * xy_err[-1])
    score += q_terminal_dq * np.sum(dq_err[-1] * dq_err[-1])
    score += q_terminal_joint * np.sum(joint_err[-1] * joint_err[-1])
    score += q_terminal_sum * np.sum(sum_err[-1] * sum_err[-1])
    score += r * np.sum(u_arr * u_arr)
    if T_F > 1:
        du = np.diff(u_arr, axis=0)
        score += lam_du * np.sum(du * du)
    score += residual_weight * float(residual) ** 2
    if g_sol is not None:
        score += g_weight * float(np.linalg.norm(g_sol)) ** 2
    return float(score)


def make_ref(obs, use_ik_ref=False, ik_elbow="down"):
    obs = np.asarray(obs, dtype=float)
    single = np.zeros(P)
    single[6:8] = obs[4:6]
    if use_ik_ref:
        features = ik_output_features(obs[4:6], elbow=ik_elbow)
        single[0:4] = features[0:4]
        single[8:10] = features[4:6]
    return np.tile(single.reshape(P, 1), (T_F, 1))


def run_atlas_deepc(H, labels, variant: Variant, save_path=None, video_path=None, controller_matrices=None):
    rng = np.random.default_rng(variant.seed)
    if controller_matrices is not None:
        bases = list(controller_matrices)
        if not bases:
            raise ValueError("controller_matrices must contain at least one matrix.")
    elif variant.coef_bound_scale is not None and variant.mode in ("basis", "subspace", "affine_subspace", "mixture_subspace"):
        bases = compute_bounded_bases(
            H,
            labels,
            rank=variant.rank,
            energy=variant.energy,
            affine=variant.coef_bound_affine or variant.mode == "affine_subspace",
            coef_bound_scale=variant.coef_bound_scale,
        )
    elif variant.mode in ("basis", "subspace", "mixture_subspace"):
        bases = compute_bases(H, labels, rank=variant.rank, energy=variant.energy)
    elif variant.mode == "affine_subspace":
        bases = compute_affine_bases(H, labels, rank=variant.rank, energy=variant.energy)
    else:
        n_charts = max(K, int(np.max(labels)) + 1) if labels.size else K
        bases = [None] * n_charts
    Q = np.diag([variant.q_joint, variant.q_joint, variant.q_joint, variant.q_joint, variant.q_dq, variant.q_dq, variant.q_xy, variant.q_xy, variant.q_sum, variant.q_sum])
    Q_terminal = np.diag([
        variant.q_terminal_joint,
        variant.q_terminal_joint,
        variant.q_terminal_joint,
        variant.q_terminal_joint,
        variant.q_terminal_dq,
        variant.q_terminal_dq,
        variant.q_terminal_xy,
        variant.q_terminal_xy,
        variant.q_terminal_sum,
        variant.q_terminal_sum,
    ])
    R = variant.r * np.eye(M)
    H_U_p, H_U_f, H_Y_p, H_Y_f = partition_hankel_rows(H)
    H_Y_f_blocks = H_Y_f.reshape(T_F, P, H.shape[1])
    H_U_p_sd = np.maximum(H_U_p.std(axis=1), 1e-6)
    H_Y_p_sd = np.maximum(H_Y_p.std(axis=1), 1e-6)
    solver = cp.CLARABEL if variant.solver == "CLARABEL" else cp.OSQP
    deepc = DeePCController(Q, R, Q_terminal, variant.lam_g, variant.lam_ini, variant.lam_du, variant.u_limit, solver)
    render_mode = "rgb_array" if video_path is not None else None
    env = gym.make(REACHER_ENV_ID, render_mode=render_mode, max_episode_steps=variant.steps + T_INI + 5)
    obs, _ = env.reset(seed=variant.seed)
    frames = []
    if video_path is not None:
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame, dtype=np.uint8))
    u_hist = []
    y_hist = []
    for _ in range(T_INI):
        action = rng.uniform(-variant.warmup_scale, variant.warmup_scale, size=M)
        obs, *_ = env.step(action)
        if video_path is not None:
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame, dtype=np.uint8))
        u_hist.append(action.reshape(M))
        y_hist.append(reacher_output_from_obs(obs))
    records = {"y": [], "u": [], "ref": [], "chart": [], "chart_set": [], "status": [], "pred": [], "g_norm": [], "residual": [], "candidate_score": []}
    previous_chart = None
    for t in range(variant.steps):
        u_ini = np.asarray(u_hist[-T_INI:], dtype=float).reshape(M * T_INI, 1)
        y_ini = np.asarray(y_hist[-T_INI:], dtype=float).reshape(P * T_INI, 1)
        y_ref = make_ref(obs, use_ik_ref=variant.use_ik_ref, ik_elbow=variant.ik_elbow)
        relevance_scores = np.full(len(bases), np.inf)
        n_charts = len(bases)
        scores = np.full(n_charts, np.inf)
        matrices = []
        for k in range(n_charts):
            if variant.mode in ("basis", "subspace", "affine_subspace", "mixture_subspace"):
                A = bases[k]
            else:
                idx = np.where(labels == k)[0]
                if idx.size > variant.max_cols:
                    idx = rng.choice(idx, size=variant.max_cols, replace=False)
                A = H[:, idx]
            matrices.append(A)
            if A is not None:
                try:
                    scores[k] = past_residual(A, u_ini, y_ini)
                except Exception:
                    pass
            idx = np.where(labels == k)[0]
            if idx.size:
                target_xy = y_ref[:P, 0][6:8]
                target_dist = np.full(idx.size, np.inf)
                for j in range(T_F):
                    xy = H_Y_f_blocks[j, 6:8, :][:, idx].T
                    target_dist = np.minimum(target_dist, np.linalg.norm(xy - target_xy, axis=1))
                if variant.relevance_past_weight:
                    u_flat = u_ini[:, 0]
                    y_flat = y_ini[:, 0]
                    past_dist = np.linalg.norm((H_Y_p[:, idx].T - y_flat) / H_Y_p_sd, axis=1)
                    past_dist += 0.1 * np.linalg.norm((H_U_p[:, idx].T - u_flat) / H_U_p_sd, axis=1)
                    target_dist = target_dist + variant.relevance_past_weight * past_dist / np.sqrt(max(y_flat.size, 1))
                relevance_scores[k] = float(np.min(target_dist))
        action = np.zeros(M)
        solved_deepc = False
        chart = -1
        chart_set = ""
        status = "not_solved"
        y_pred = np.full(P, np.nan)
        best_candidate = None
        relevance_modes = ("best_relevance", "relevance_then_predicted", "mixture_relevance")
        ranking_scores = relevance_scores.copy() if variant.chart_selection in relevance_modes else scores.copy()
        if previous_chart is not None and variant.chart_switch_cost > 0.0 and 0 <= previous_chart < ranking_scores.size:
            ranking_scores[previous_chart] -= variant.chart_switch_cost
        if variant.fixed_chart is None:
            if variant.chart_selection in relevance_modes:
                candidate_charts = np.argsort(ranking_scores)[:variant.try_top]
            else:
                candidate_charts = np.argsort(ranking_scores)[:variant.try_top]
        else:
            candidate_charts = np.asarray([variant.fixed_chart], dtype=int)

        if variant.chart_selection == "mixture_relevance" or variant.mode == "mixture_subspace":
            finite = [int(k) for k in candidate_charts if matrices[k] is not None and np.isfinite(ranking_scores[k])]
            finite = finite[:max(1, int(variant.mixture_top))]
            if finite:
                try:
                    A_mix = combine_chart_matrices([matrices[k] for k in finite])
                    mix_score = float(np.mean([ranking_scores[k] for k in finite]))
                    u_sol, y_sol, g_sol, status, residual = deepc.solve(A_mix, u_ini, y_ini, y_ref)
                except Exception as exc:
                    u_sol, y_sol, g_sol, status, residual = None, None, None, f"crashed: {exc}", np.inf
                if u_sol is not None:
                    candidate_score = predicted_tracking_score(
                        y_sol,
                        u_sol,
                        y_ref,
                        variant.q_xy,
                        variant.q_dq,
                        variant.r,
                        variant.lam_du,
                        g_sol=g_sol,
                        residual=residual,
                        residual_weight=variant.selection_residual_weight,
                        g_weight=variant.selection_g_weight,
                        q_joint=variant.q_joint,
                        q_sum=variant.q_sum,
                        q_terminal_xy=variant.q_terminal_xy,
                        q_terminal_dq=variant.q_terminal_dq,
                        q_terminal_joint=variant.q_terminal_joint,
                        q_terminal_sum=variant.q_terminal_sum,
                    )
                    if variant.selection_relevance_weight:
                        candidate_score += variant.selection_relevance_weight * mix_score
                    best_candidate = (candidate_score, int(finite[0]), status, residual, u_sol, y_sol, g_sol, ",".join(map(str, finite)))

        for k in ([] if best_candidate is not None and (variant.chart_selection == "mixture_relevance" or variant.mode == "mixture_subspace") else candidate_charts):
            if not np.isfinite(scores[k]) or matrices[k] is None:
                continue
            u_sol, y_sol, g_sol, status, residual = deepc.solve(matrices[k], u_ini, y_ini, y_ref)
            if u_sol is not None:
                candidate_score = predicted_tracking_score(
                    y_sol,
                    u_sol,
                    y_ref,
                    variant.q_xy,
                    variant.q_dq,
                    variant.r,
                    variant.lam_du,
                    g_sol=g_sol,
                    residual=residual,
                    residual_weight=variant.selection_residual_weight,
                    g_weight=variant.selection_g_weight,
                    q_joint=variant.q_joint,
                    q_sum=variant.q_sum,
                    q_terminal_xy=variant.q_terminal_xy,
                    q_terminal_dq=variant.q_terminal_dq,
                    q_terminal_joint=variant.q_terminal_joint,
                    q_terminal_sum=variant.q_terminal_sum,
                )
                if variant.selection_past_weight:
                    candidate_score += variant.selection_past_weight * float(scores[k]) ** 2
                if variant.selection_relevance_weight:
                    candidate_score += variant.selection_relevance_weight * float(relevance_scores[k])
                if variant.selection_zero_action_weight and variant.selection_min_action_norm > 0.0:
                    first_action_norm = float(np.linalg.norm(u_sol[:M, 0]))
                    action_gap = max(0.0, variant.selection_min_action_norm - first_action_norm)
                    candidate_score += variant.selection_zero_action_weight * action_gap ** 2
                candidate = (candidate_score, int(k), status, residual, u_sol, y_sol, g_sol, str(int(k)))
                if variant.chart_selection in ("best_predicted", "relevance_then_predicted"):
                    if best_candidate is None or candidate_score < best_candidate[0]:
                        best_candidate = candidate
                else:
                    best_candidate = candidate
                    break
        g_norm = np.nan
        residual = np.nan
        candidate_score = np.nan
        if best_candidate is not None:
            candidate_score, chart, status, residual, u_sol, y_sol, g_sol, chart_set = best_candidate
            action = np.asarray(u_sol[:M, 0]).reshape(M)
            action = np.clip(action, -variant.u_limit, variant.u_limit)
            y_pred = np.asarray(y_sol[:P, 0]).reshape(P)
            g_norm = float(np.linalg.norm(g_sol))
            previous_chart = chart
            solved_deepc = True
        if variant.feedback_blend > 0.0 and (solved_deepc or variant.feedback_fallback):
            feedback_action = ik_action(
                obs,
                kp=variant.feedback_kp,
                kd=variant.feedback_kd,
                limit=variant.u_limit,
                elbow=variant.ik_elbow,
            )
            blend = float(np.clip(variant.feedback_blend, 0.0, 1.0))
            action = np.clip((1.0 - blend) * action + blend * feedback_action, -variant.u_limit, variant.u_limit)
        obs, reward, terminated, truncated, info = env.step(action)
        if video_path is not None:
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame, dtype=np.uint8))
        y_now = reacher_output_from_obs(obs)
        u_hist.append(action.copy())
        y_hist.append(y_now.copy())
        records["y"].append(y_now)
        records["u"].append(action)
        records["ref"].append(y_ref[:P, 0].copy())
        records["chart"].append(chart)
        records["chart_set"].append(chart_set)
        records["status"].append(status)
        records["pred"].append(y_pred)
        records["g_norm"].append(g_norm)
        records["residual"].append(residual)
        records["candidate_score"].append(candidate_score)
        if terminated or truncated:
            break
    env.close()
    if video_path is not None and frames:
        imageio.mimsave(video_path, frames, fps=20, macro_block_size=None)
    for key in ["y", "u", "ref", "chart", "pred", "g_norm", "residual", "candidate_score"]:
        records[key] = np.asarray(records[key])
    records["chart_set"] = np.asarray(records["chart_set"], dtype=object)
    records["status"] = np.asarray(records["status"], dtype=object)
    dist = np.linalg.norm(records["y"][:, 6:8] - records["ref"][:, 6:8], axis=1)
    summary = {
        **asdict(variant),
        "n_steps": int(dist.size),
        "first": float(dist[0]),
        "min": float(dist.min()),
        "last": float(dist[-1]),
        "mean_last_30": float(dist[-30:].mean()),
        "max_last_30": float(dist[-30:].max()),
        "mean": float(dist.mean()),
        "success": bool(dist[-30:].mean() < 0.01 and dist[-30:].max() < 0.02),
        "unique_charts": np.unique(records["chart"]).astype(int).tolist(),
        "unique_chart_sets": sorted({str(s) for s in records["chart_set"]}),
        "statuses": {str(s): int((records["status"] == s).sum()) for s in np.unique(records["status"])},
    }
    if save_path is not None:
        np.savez(save_path, **records, distance=dist, summary=np.array(json.dumps(summary), dtype=object))
    return summary


def main():
    started = time.time()
    if DATA_PATH.exists():
        H = np.load(DATA_PATH)
    else:
        H = collect_expert_hankel()
    print("H", H.shape, "u range", H[:M * L].min(), H[:M * L].max())
    if LABEL_PATH.exists():
        labels = np.load(LABEL_PATH)["labels"]
    else:
        labels = learn_prediction_error_atlas(H)
    print("cluster sizes", np.bincount(labels, minlength=K).tolist())

    variants = []
    for mode in ["basis", "raw"]:
        for u_limit in [0.2, 0.35, 0.5]:
            for q_xy in [500.0, 1500.0, 5000.0]:
                variants.append(Variant(
                    name=f"{mode}_u{u_limit}_q{int(q_xy)}",
                    mode=mode,
                    rank=90,
                    max_cols=1600,
                    try_top=6,
                    u_limit=u_limit,
                    q_xy=q_xy,
                    q_dq=0.1,
                    r=0.001,
                    lam_g=0.1 if mode == "basis" else 10.0,
                    lam_ini=100000.0,
                    lam_du=0.5,
                    solver="CLARABEL",
                    steps=120,
                ))
    results = []
    for variant in variants:
        print("RUN", variant)
        summary = run_atlas_deepc(H, labels, variant, save_path=Path(f"expert_{variant.name}.npz"))
        results.append(summary)
        print(" ->", summary)
        RESULTS_PATH.write_text(json.dumps({"elapsed": time.time() - started, "results": results}, indent=2))
        if summary["success"]:
            print("SUCCESS", summary)
            break
    results.sort(key=lambda r: (r["mean_last_30"], r["max_last_30"], r["last"]))
    RESULTS_PATH.write_text(json.dumps({"elapsed": time.time() - started, "best": results[:10], "results": results}, indent=2))
    print("BEST")
    for row in results[:5]:
        print(row)


if __name__ == "__main__":
    main()
