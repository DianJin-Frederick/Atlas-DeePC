"""Exact isometric visualization of complete length-two input-output windows."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_motivation_mpl")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
import json
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
T_INI, T_F, L = 1, 1, 2
SEED = 29
# Orthonormal columns span S = {w in R^4 : u0 + u1 = 0}.
# Window order is (u0, u1, y0, y1); each ROW in stored arrays is one window.
Q = np.column_stack([
    np.array([1., -1., 0., 1.])/np.sqrt(3),
    np.array([0., 0., 1., 0.]),
    np.array([-1., 1., 0., 2.])/np.sqrt(6),
])


def block_hankel(signal, depth):
    """H_depth(z), with z.shape=(N, dimension); columns contain time blocks."""
    signal = np.asarray(signal)
    if signal.ndim == 1:
        signal = signal[:, None]
    if not 1 <= depth <= len(signal):
        raise ValueError("Hankel depth must be between 1 and the signal length.")
    return np.stack([signal[j:j+depth].reshape(-1)
                     for j in range(len(signal)-depth+1)], axis=1)


def trajectory_windows(initial_outputs, first_inputs, length=L):
    """Mosaic of actual H_2 columns for y[k+1]=y[k]^2+u[k], with u1=-u0."""
    if length != 2:
        raise ValueError("This illustrative construction specifically uses L=2.")
    x, a = np.broadcast_arrays(initial_outputs, first_inputs)
    windows = []
    for initial, first in zip(x.ravel(), a.ravel()):
        u = np.array([first, -first])
        y = [float(initial)]
        for value in u:
            y.append(y[-1]**2 + value)
        # Both inputs are applied. y2 validates the second transition, but lies
        # beyond the length-two output block (y0,y1), as in the Hankel convention.
        assert np.isclose(y[2], (initial**2 + first)**2 - first)
        column = np.vstack([block_hankel(u, 2), block_hankel(np.array(y[:2]), 2)])
        windows.append(column[:, 0])
    return np.asarray(windows)


def embedding(w):
    """Orthonormal coordinates of complete R^4 windows, z=Q^T w."""
    return np.asarray(w) @ Q


def decode(z):
    """Exact LINEAR reconstruction w=Q z on S."""
    return np.asarray(z) @ Q.T


def fit_diagnostics():
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(SEED)
    train = trajectory_windows(rng.uniform(-1, 1, 1200), rng.uniform(-0.5, 0.5, 1200))
    test = trajectory_windows(rng.uniform(-1, 1, 500), rng.uniform(-0.5, 0.5, 500))
    groups = KMeans(n_clusters=16, n_init=20, random_state=SEED).fit(train)
    labels = groups.predict(test)
    def plane(points):
        center = points.mean(axis=0)
        _, _, vt = np.linalg.svd(points-center, full_matrices=False)
        return center, vt[:2].T
    def project(points, model):
        center, basis = model
        return center + (points-center) @ basis @ basis.T
    global_fit = project(test, plane(train))
    local_fit = np.empty_like(test)
    for j in range(16):
        mask = labels == j
        local_fit[mask] = project(test[mask], plane(train[groups.labels_ == j]))
    z = embedding(test)
    dw = test[1:]-test[:-1]
    dz = z[1:]-z[:-1]
    # Same initial values and pulse pairs for a linear-system control.
    linear_w = test.copy()
    linear_w[:, 3] = 0.7*test[:, 2]+test[:, 0]
    metrics = {
        "T_ini": T_INI, "T_f": T_F, "L": L,
        "inputs_per_window": 2, "outputs_per_window": 2,
        "ambient_dimension": 4, "selected_linear_subspace_dimension": 3,
        "intrinsic_dimension_of_selected_family": 2,
        "empirical_affine_span_rank": int(np.linalg.matrix_rank(train-train.mean(axis=0))),
        "train_windows": len(train), "test_windows": len(test), "local_charts": 16,
        "global_rank_two_window_rms": float(np.sqrt(np.mean(np.sum((test-global_fit)**2, axis=1)))),
        "local_rank_two_window_rms": float(np.sqrt(np.mean(np.sum((test-local_fit)**2, axis=1)))),
        "max_full_window_reconstruction_error": float(np.max(np.abs(decode(z)-test))),
        "max_pairwise_squared_distance_error": float(np.max(np.abs(np.sum(dw**2, axis=1)-np.sum(dz**2, axis=1)))),
        "max_surface_residual": float(np.max(np.abs(z[:, 2]-np.sqrt(2/3)*z[:, 1]**2))),
        "max_dynamics_residual": float(np.max(np.abs(test[:, 3]-test[:, 2]**2-test[:, 0]))),
        "linear_system_control_affine_rank": int(np.linalg.matrix_rank(linear_w-linear_w.mean(axis=0))),
    }
    assert np.allclose(Q.T @ Q, np.eye(3))
    assert np.allclose(test[:, 0]+test[:, 1], 0)
    for name in ["max_full_window_reconstruction_error", "max_pairwise_squared_distance_error",
                 "max_surface_residual", "max_dynamics_residual"]:
        assert metrics[name] < 1e-12, (name, metrics[name])
    assert metrics["empirical_affine_span_rank"] == 3
    assert metrics["linear_system_control_affine_rank"] == 2
    assert metrics["local_rank_two_window_rms"] < metrics["global_rank_two_window_rms"]
    return train, test, metrics


def paper_figure():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.transforms import Bbox
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    style = {
        "font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 8,
        "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
        "axes.labelsize": 9, "axes.linewidth": 0.6, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "path",
    }
    a, x = np.meshgrid(np.linspace(-0.5, 0.5, 23), np.linspace(-1, 1, 23))
    windows = trajectory_windows(x, a)
    points = embedding(windows)
    grid = points.reshape(*a.shape, 3)
    # A finer mesh shades the exact analytical surface; the 529 measured
    # trajectory points above remain unchanged. Lighting is purely stylistic.
    fine_a, fine_x = np.meshgrid(np.linspace(-0.5, 0.5, 61), np.linspace(-1, 1, 61))
    fine = embedding(trajectory_windows(fine_x, fine_a)).reshape(61, 61, 3)
    normals = np.stack([np.zeros_like(fine_x), -2*np.sqrt(2/3)*fine_x,
                        np.ones_like(fine_x)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    light = np.array([-0.25, -0.6, 0.75])
    light /= np.linalg.norm(light)
    illumination = np.clip((normals @ light + 0.2)/1.2, 0, 1)
    shade = np.array([0.57, 0.82, 0.94])
    highlight = np.array([0.90, 0.98, 1.0])
    facecolors = shade + illumination[..., None]*(highlight-shade)
    # Preblend against white: opaque tessellation avoids transparency seams
    # between adjacent polygons in vector and high-resolution raster exports.
    facecolors = 0.72*facecolors + 0.28
    with plt.rc_context(style):
        fig = plt.figure(figsize=(3.5, 3.4), dpi=200, facecolor="white")
        ax = fig.add_axes([0.015, 0.08, 0.88, 0.88], projection="3d", computed_zorder=False)
        surface = ax.plot_surface(fine[..., 0], fine[..., 1], fine[..., 2], facecolors=facecolors,
                                  rcount=61, ccount=61, alpha=1.0, linewidth=0,
                                  antialiased=False, shade=False, zorder=0)
        surface.set_edgecolor("none")
        # Both diagonal parameter families are genuine curved sections of the
        # same manifold. Every mesh sample is constructed from a full H_2 column.
        for direction in [-1, 1]:
            for offset in np.linspace(-1.5, 1.5, 9):
                lo = max(-1, -1-direction*offset)
                hi = min(1, 1-direction*offset)
                if lo >= hi:
                    continue
                t = np.linspace(lo, hi, 181)
                curve_x, curve_a = direction*t+offset, 0.5*t
                curve = embedding(trajectory_windows(curve_x, curve_a))
                assert np.allclose(curve[:, 2], np.sqrt(2/3)*curve[:, 1]**2)
                ax.plot(*curve.T, color="#28A1C5", alpha=0.42, lw=0.35, zorder=1)
        for edge in [fine[0], fine[-1], fine[:, 0], fine[:, -1]]:
            ax.plot(*edge.T, color="#168EBC", alpha=0.8, lw=0.65, zorder=1)
        ax.scatter(*points.T, s=1.9, color="#079DD5", alpha=0.90,
                   edgecolors="white", linewidths=0.12, depthshade=True, zorder=2)
        ax.set(xlim=(-1.05, 1.6), ylim=(-1.1, 1.1), zlim=(-0.06, 0.9),
               xlabel=r"$z_1$", ylabel=r"$z_2$", zlabel="")
        ax.text2D(1.06, 0.58, r"$z_3$", transform=ax.transAxes,
                  rotation=90, ha="center", va="center", fontsize=9)
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_zticks([0, 0.4, 0.8])
        ax.tick_params(pad=-0.5, colors="#43515B")
        ax.xaxis.labelpad = 0; ax.yaxis.labelpad = 0; ax.zaxis.labelpad = 0
        ax.set_box_aspect((1.2, 1, 0.7))
        ax.view_init(elev=27, azim=-63)
        ax.set_proj_type("ortho")
        ax.grid(False)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.fill = False
            axis.pane.set_edgecolor((1, 1, 1, 0))
            axis.line.set_color("#81919A")
            axis.line.set_linewidth(0.4)
        OUT.mkdir(parents=True, exist_ok=True)
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        rows = np.flatnonzero(np.any(rgb < 245, axis=(1, 2)))
        bottom = (rgb.shape[0]-1-rows[-1])/fig.dpi - 0.035
        height = (rows[-1]-rows[0]+1)/fig.dpi + 0.07
        bounds = Bbox.from_bounds(0, bottom, 3.5, height)
        for extension in ["pdf", "svg", "png"]:
            fig.savefig(OUT / f"io_trajectory_paper.{extension}", dpi=600,
                        facecolor="white", bbox_inches=bounds, pad_inches=0)
        plt.close(fig)
    return windows


if __name__ == "__main__":
    plotted = paper_figure()
    train, test, metrics = fit_diagnostics()
    (OUT / "io_trajectory_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    np.savez_compressed(OUT / "io_trajectory_windows.npz", train=train, test=test,
                        plotted=plotted, basis=Q, T_ini=T_INI, T_f=T_F)
    print(json.dumps(metrics, indent=2))
