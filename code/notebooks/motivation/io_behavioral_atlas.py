"""K-means + local affine charts for complete R^4 input-output windows."""
from pathlib import Path
import importlib.util
import json
import numpy as np

FOLDER = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("io_surface", FOLDER / "io_trajectory_surface.py")
io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(io)
OUT = io.OUT
K_DEFAULT = 8
COLORS = ["#089ED1", "#F08A24", "#28AD87", "#A46DCC",
          "#E65C80", "#537AD5", "#9AAE35", "#C28D36"]


def collect():
    # Same experiment family, sample sizes, and seed as the surface notebook.
    rng = np.random.default_rng(io.SEED)
    train = io.trajectory_windows(rng.uniform(-1, 1, 1200), rng.uniform(-0.5, 0.5, 1200))
    test = io.trajectory_windows(rng.uniform(-1, 1, 500), rng.uniform(-0.5, 0.5, 500))
    return train, test


def learn_atlas(train, n_charts=K_DEFAULT):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_charts, n_init=20, random_state=io.SEED).fit(train)
    # Stable display numbering; no change to the fitted partition.
    order = np.lexsort((km.cluster_centers_[:, 0], km.cluster_centers_[:, 2]))
    inverse = np.empty(n_charts, dtype=int)
    inverse[order] = np.arange(n_charts)
    labels = inverse[km.labels_]
    centers, bases, singular_values = [], [], []
    for j in range(n_charts):
        members = train[labels == j]
        center = members.mean(axis=0)
        _, s, vt = np.linalg.svd(members-center, full_matrices=False)
        centers.append(center)
        bases.append(vt[:2].T)
        singular_values.append(s)
    return {"labels": labels, "centers": np.asarray(centers),
            "bases": np.asarray(bases), "singular_values": np.asarray(singular_values)}


def assign(windows, atlas):
    distance2 = np.sum((windows[:, None, :]-atlas["centers"][None, :, :])**2, axis=2)
    return distance2.argmin(axis=1)


def reconstruct(windows, atlas):
    labels = assign(windows, atlas)
    reconstructed = np.empty_like(windows)
    for j, (center, basis) in enumerate(zip(atlas["centers"], atlas["bases"])):
        mask = labels == j
        reconstructed[mask] = center + (windows[mask]-center) @ basis @ basis.T
    return reconstructed


def diagnostics(train, test, atlas):
    from scipy.spatial.distance import cdist
    # Q is isometric on this family: the K-means distance objective and local
    # affine residual norms can be visualized faithfully in these coordinates.
    train_z = io.embedding(train)
    centers_z = io.embedding(atlas["centers"])
    d4 = cdist(train, atlas["centers"], metric="sqeuclidean")
    d3 = cdist(train_z, centers_z, metric="sqeuclidean")
    assert np.allclose(d4, d3, atol=1e-12)
    assert np.array_equal(assign(train, atlas), atlas["labels"])
    residuals = []
    for k in [1, 4, 8, 16]:
        fitted = atlas if k == K_DEFAULT else learn_atlas(train, k)
        pred = reconstruct(test, fitted)
        residuals.append({"K": k, "test_window_rms": float(np.sqrt(np.mean(np.sum((test-pred)**2, axis=1))))})
    pred = reconstruct(test, atlas)
    assert np.allclose(io.decode(io.embedding(pred)), pred, atol=1e-12)
    assert np.allclose(np.linalg.norm(test-pred, axis=1),
                       np.linalg.norm(io.embedding(test)-io.embedding(pred), axis=1))
    chart_stats = []
    for j in range(K_DEFAULT):
        members = train[atlas["labels"] == j]
        singular = atlas["singular_values"][j]
        chart_stats.append({
            "chart": j+1, "training_windows": len(members),
            "center": atlas["centers"][j].tolist(),
            "centered_full_rank": int(np.linalg.matrix_rank(members-members.mean(axis=0))),
            "uncentered_full_rank": int(np.linalg.matrix_rank(members)),
            "rank_two_discarded_variance_fraction": float(np.sum(singular[2:]**2)/np.sum(singular**2)),
        })
    return {"system": "y[k+1] = y[k]^2 + u[k]", "input_family": "u1 = -u0",
            "T_ini": 1, "T_f": 1, "full_window_dimension": 4,
            "training_windows": len(train), "test_windows": len(test), "K": K_DEFAULT,
            "chart_type": "centered rank-two affine SVD",
            "max_KMeans_distance_preservation_error": float(np.max(np.abs(d4-d3))),
            "sweep": residuals, "charts": chart_stats}


def style_axes(ax):
    ax.set(xlim=(-1.05, 1.6), ylim=(-1.1, 1.1), zlim=(-0.06, 0.9),
           xlabel=r"$z_1$", ylabel=r"$z_2$", zlabel="")
    ax.text2D(1.045, 0.58, r"$z_3$", transform=ax.transAxes,
              rotation=90, ha="center", va="center", fontsize=9)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_zticks([0, 0.4, 0.8])
    ax.tick_params(pad=-0.5, colors="#34434A")
    ax.xaxis.labelpad = 0; ax.yaxis.labelpad = 0
    ax.set_box_aspect((1.2, 1, 0.7))
    ax.view_init(elev=24, azim=-58)
    ax.set_proj_type("ortho")
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.line.set_color("#607078")
        axis.line.set_linewidth(0.45)


def background(ax):
    # True curved mesh on the known surface, only a faint geometric guide.
    for direction in [-1, 1]:
        for offset in np.linspace(-1.5, 1.5, 9):
            lo, hi = max(-1, -1-direction*offset), min(1, 1-direction*offset)
            if lo < hi:
                t = np.linspace(lo, hi, 80)
                curve = io.embedding(io.trajectory_windows(direction*t+offset, 0.5*t))
                ax.plot(*curve.T, color="#9DB8C4", alpha=0.22, lw=0.35, zorder=0)


def draw_clusters(ax, train, atlas):
    background(ax)
    z = io.embedding(train)
    for j, color in enumerate(COLORS):
        points = z[atlas["labels"] == j]
        ax.scatter(*points.T, s=3.0, color=color, alpha=0.83,
                   edgecolors="none", depthshade=True, zorder=2)


def draw_charts(ax, train, atlas):
    from scipy.spatial import ConvexHull
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    background(ax)
    z = io.embedding(train)
    ax.scatter(*z.T, s=1.5, color="#6B8491", alpha=0.14,
               edgecolors="none", depthshade=False, zorder=1)
    for j, (center, basis, color) in enumerate(zip(atlas["centers"], atlas["bases"], COLORS)):
        members = train[atlas["labels"] == j]
        scores = (members-center) @ basis
        # Display only the observed footprint of each affine plane, not an
        # arbitrary rectangle. This polygon lies exactly on the fitted chart.
        hull = ConvexHull(scores)
        corners4 = center + scores[hull.vertices] @ basis.T
        corners3 = io.embedding(corners4)
        assert np.allclose((corners4-center) @ (np.eye(4)-basis@basis.T), 0, atol=1e-12)
        poly = Poly3DCollection([corners3], facecolors=color, edgecolors=color,
                                alpha=0.43, linewidths=0.65, zorder=2)
        ax.add_collection3d(poly)
        center3 = io.embedding(center[None])[0]
        ax.scatter(*center3, s=12, color=color, edgecolors="white",
                   linewidths=0.45, depthshade=False, zorder=3)


def make_figures(train, atlas):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.transforms import Bbox
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    style = {"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 8,
             "axes.labelsize": 9, "axes.titlesize": 10,
             "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
             "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
             "pdf.fonttype": 42, "svg.fonttype": "path"}
    OUT.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(style):
        fig = plt.figure(figsize=(7.2, 3.5), dpi=180)
        ax1 = fig.add_axes([0.01, 0.19, 0.43, 0.77], projection="3d", computed_zorder=False)
        ax2 = fig.add_axes([0.51, 0.19, 0.43, 0.77], projection="3d", computed_zorder=False)
        for ax in [ax1, ax2]:
            style_axes(ax)
        draw_clusters(ax1, train, atlas)
        draw_charts(ax2, train, atlas)
        fig.text(0.245, 0.955, "(a) K-means groups of complete windows", ha="center", fontsize=10)
        fig.text(0.745, 0.955, "(b) Local rank-two affine charts", ha="center", fontsize=10)
        handles = [Line2D([], [], color=c, marker="o", markersize=4, lw=0, label=f"Chart {j+1}")
                   for j, c in enumerate(COLORS)]
        fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False,
                   bbox_to_anchor=(0.5, 0.025), columnspacing=1.2, handletextpad=0.4)
        for ext in ["pdf", "svg", "png"]:
            fig.savefig(OUT / f"io_behavioral_atlas.{ext}", dpi=450, facecolor="white")
        plt.close(fig)
        # Title-free single-column variants for paper insertion.
        for name, draw in [("clusters", draw_clusters), ("charts", draw_charts)]:
            fig = plt.figure(figsize=(3.5, 3.4), dpi=200)
            ax = fig.add_axes([0.015, 0.08, 0.88, 0.88], projection="3d", computed_zorder=False)
            style_axes(ax)
            draw(ax, train, atlas)
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            rows = np.flatnonzero(np.any(rgb < 245, axis=(1, 2)))
            bottom = (len(rgb)-1-rows[-1])/fig.dpi-0.035
            height = (rows[-1]-rows[0]+1)/fig.dpi+0.07
            bounds = Bbox.from_bounds(0, bottom, 3.5, height)
            for ext in ["pdf", "svg", "png"]:
                fig.savefig(OUT / f"io_atlas_{name}.{ext}", dpi=600, facecolor="white",
                            bbox_inches=bounds, pad_inches=0)
            plt.close(fig)


def run():
    train, test = collect()
    atlas = learn_atlas(train)
    metrics = diagnostics(train, test, atlas)
    make_figures(train, atlas)
    (OUT / "io_behavioral_atlas_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    np.savez_compressed(OUT / "io_behavioral_atlas.npz", train=train, test=test,
                        labels=atlas["labels"], centers=atlas["centers"], bases=atlas["bases"],
                        singular_values=atlas["singular_values"], Q=io.Q)
    return metrics


if __name__ == "__main__":
    metrics = run()
    print(json.dumps({k: v for k, v in metrics.items() if k != "charts"}, indent=2))
    print("Per-chart populations:", [c["training_windows"] for c in metrics["charts"]])
