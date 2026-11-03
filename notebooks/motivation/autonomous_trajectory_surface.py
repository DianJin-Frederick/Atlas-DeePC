"""Autonomous second-order system: full length-three trajectories form a surface."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_motivation_mpl")
import json
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
L = 3
# Show a shorter patch of the same surface, with the original camera unchanged.
PLOT_A_HALF_WIDTH = 0.60


def simulate(y0, y1, n_samples=3):
    """Autonomous recurrence y[k+2] = (y[k] + y[k+1]^2)/2."""
    if n_samples < 2:
        raise ValueError("Provide at least two samples for the second-order recurrence.")
    y = np.empty(n_samples)
    y[:2] = [y0, y1]
    for k in range(n_samples-2):
        y[k+2] = 0.5*y[k] + 0.5*y[k+1]**2
    return y


def hankel(y, depth=3):
    """One column is one COMPLETE depth-sample output trajectory."""
    y = np.asarray(y)
    if y.ndim != 1 or not 1 <= depth <= len(y):
        raise ValueError("Expected a scalar output record and valid Hankel depth.")
    return np.stack([y[j:j+depth] for j in range(len(y)-depth+1)], axis=1)


def trajectory_windows(y0, y1):
    a, b = np.broadcast_arrays(y0, y1)
    return np.stack([hankel(simulate(x, y), L)[:, 0]
                     for x, y in zip(a.ravel(), b.ravel())])


def validate():
    rng = np.random.default_rng(41)
    records = [simulate(a, b, n_samples=40) for a, b in rng.uniform(-1, 1, (100, 2))]
    # Build each Hankel matrix separately; no columns cross experiment boundaries.
    H = np.concatenate([hankel(y, L) for y in records], axis=1)
    residual = H[2] - 0.5*H[0] - 0.5*H[1]**2
    assert np.array_equal(hankel([1, 2, 3, 4]), [[1, 2], [2, 3], [3, 4]])
    assert np.max(np.abs(residual)) < 1e-12
    assert np.max(np.abs(H)) <= 1+1e-12
    a,b = np.meshgrid(np.linspace(-PLOT_A_HALF_WIDTH, PLOT_A_HALF_WIDTH, 23), np.linspace(-1, 1, 23))
    plotted = trajectory_windows(a,b)
    expected = np.column_stack([a.ravel(), b.ravel(), 0.5*a.ravel()+0.5*b.ravel()**2])
    assert np.array_equal(plotted, expected)
    # The Jacobian's first two rows are the identity, so its rank is two everywhere.
    jacobian = np.stack([np.tile([1., 0.], (len(plotted), 1)),
                         np.tile([0., 1.], (len(plotted), 1)),
                         np.column_stack([np.full(len(plotted), 0.5), plotted[:, 1]])], axis=1)
    assert np.all(np.linalg.matrix_rank(jacobian) == 2)
    metrics = {"input_dimension": 0, "output_dimension": 1, "state_dimension": 2,
               "L": L, "trajectory_ambient_dimension": 3, "intrinsic_dimension": 2,
               "plotted_complete_windows": len(plotted), "validation_rollouts": len(records),
               "plotted_y0_range": [-PLOT_A_HALF_WIDTH, PLOT_A_HALF_WIDTH],
               "plotted_y1_range": [-1, 1],
               "samples_per_rollout": 40, "validated_Hankel_columns": H.shape[1],
               "max_dynamics_residual": float(np.max(np.abs(residual))),
               "max_absolute_output": float(np.max(np.abs(H))),
               "affine_rank_of_plotted_family": int(np.linalg.matrix_rank(plotted-plotted.mean(axis=0)))}
    assert metrics["affine_rank_of_plotted_family"] == 3
    return plotted, H, metrics


def paper_figure():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.transforms import Bbox
    from mpl_toolkits.mplot3d import proj3d
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    style = {
        "font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 8,
        "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
        "axes.labelsize": 9, "axes.linewidth": 0.6, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "path",
    }
    a, x = np.meshgrid(np.linspace(-PLOT_A_HALF_WIDTH, PLOT_A_HALF_WIDTH, 23), np.linspace(-1, 1, 23))
    windows = trajectory_windows(a, x)
    points = windows
    grid = points.reshape(*a.shape, 3)
    # A finer mesh shades the exact analytical surface; the 529 measured
    # trajectory points above remain unchanged. Lighting is purely stylistic.
    fine_a, fine_x = np.meshgrid(np.linspace(-PLOT_A_HALF_WIDTH, PLOT_A_HALF_WIDTH, 61), np.linspace(-1, 1, 61))
    fine = trajectory_windows(fine_a, fine_x).reshape(61, 61, 3)
    normals = np.stack([-0.5*np.ones_like(fine_x), -fine_x,
                        np.ones_like(fine_x)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    light = np.array([-0.25, -0.6, 0.75])
    light /= np.linalg.norm(light)
    illumination = np.clip((normals @ light + 0.2)/1.2, 0, 1)
    # A continuous height gradient emphasizes the true surface geometry.
    # Translucent faces keep overlapping portions of the surface visible.
    cmap = LinearSegmentedColormap.from_list(
        "trajectory_height", ["#4968AD", "#398FAF", "#66C2C4", "#C8EBE4"])
    facecolors = cmap(Normalize(-0.5, 1.0)(fine[..., 2]))[..., :3]
    facecolors *= 0.88 + 0.12*illumination[..., None]
    facecolors = 0.82*facecolors + 0.18
    with plt.rc_context(style):
        fig = plt.figure(figsize=(3.5, 3.4), dpi=200, facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1], projection="3d", computed_zorder=False)
        surface = ax.plot_surface(fine[..., 0], fine[..., 1], fine[..., 2], facecolors=facecolors,
                                  rcount=61, ccount=61, alpha=0.50, linewidth=0,
                                  antialiased=True, shade=False, zorder=0)
        surface.set_edgecolor("none")
        # Both diagonal parameter families are genuine curved sections of the
        # same manifold. Every mesh sample is a full depth-three Hankel column.
        for direction in [-1, 1]:
            for offset in np.linspace(-1.5, 1.5, 9):
                lo = max(-PLOT_A_HALF_WIDTH, -1-direction*offset)
                hi = min(PLOT_A_HALF_WIDTH, 1-direction*offset)
                if lo >= hi:
                    continue
                t = np.linspace(lo, hi, 181)
                curve_x, curve_a = direction*t+offset, t
                curve = trajectory_windows(curve_a, curve_x)
                assert np.allclose(curve[:, 2], 0.5*curve[:, 0]+0.5*curve[:, 1]**2)
                ax.plot(*curve.T, color="#386E89", alpha=0.32, lw=0.35, zorder=1)
        for edge in [fine[0], fine[-1], fine[:, 0], fine[:, -1]]:
            ax.plot(*edge.T, color="#337F9B", alpha=0.8, lw=0.65, zorder=1)
        ax.scatter(*points.T, s=1.7, color="#286A96", alpha=0.75,
                   edgecolors="white", linewidths=0.12, depthshade=True, zorder=2)
        ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), zlim=(-0.56, 1.06))
        # Keep the original view and axis scaling: only the patch extent changes.
        ax.set_box_aspect((1, 1, 0.78))
        ax.view_init(elev=27, azim=-63)
        ax.set_proj_type("ortho")
        ax.set_axis_off()
        # The zero equilibrium is a complete trajectory on the surface.
        # Faint, unlabeled coordinate axes intersect at the actual origin.
        origin = simulate(0.0, 0.0, 3)
        assert np.array_equal(origin, np.zeros(3))
        for direction in np.eye(3):
            tail = -0.30*direction
            ax.plot(*np.stack([tail, origin]).T, color="#596E79",
                    alpha=0.50, lw=0.6, zorder=4)
            ax.quiver(*origin, *(0.55*direction), color="#596E79",
                      alpha=0.60, linewidth=0.6, arrow_length_ratio=0.10,
                      normalize=False, zorder=4)
        ax.scatter(*origin, s=11, color="#455861", edgecolors="white",
                   linewidths=0.45, depthshade=False, zorder=5)
        origin_xy = proj3d.proj_transform(*origin, ax.get_proj())[:2]
        ax.annotate(r"$(0,\,0,\,0)$", xy=origin_xy, xytext=(4, -10), textcoords="offset points",
                    fontsize=9, color="#455861", ha="left", va="top", zorder=6)
        OUT.mkdir(parents=True, exist_ok=True)
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        ink = np.any(rgb < 245, axis=2)
        rows = np.flatnonzero(np.any(ink, axis=1))
        columns = np.flatnonzero(np.any(ink, axis=0))
        padding = 0.01
        left = columns[0]/fig.dpi - padding
        width = (columns[-1]-columns[0]+1)/fig.dpi + 2*padding
        bottom = (rgb.shape[0]-1-rows[-1])/fig.dpi - padding
        height = (rows[-1]-rows[0]+1)/fig.dpi + 2*padding
        bounds = Bbox.from_bounds(left, bottom, width, height)
        for extension in ["pdf", "svg", "png"]:
            fig.savefig(OUT / f"autonomous_trajectory_paper.{extension}", dpi=600,
                        facecolor="white", bbox_inches=bounds, pad_inches=0)
        plt.close(fig)
    return windows


if __name__ == "__main__":
    plotted, H, metrics = validate()
    pictured = paper_figure()
    assert np.array_equal(pictured, plotted)
    np.savez_compressed(OUT / "autonomous_trajectory_windows.npz", plotted=plotted, H=H)
    (OUT / "autonomous_trajectory_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    print(json.dumps(metrics, indent=2))
