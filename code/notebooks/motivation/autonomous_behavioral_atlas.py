"""K-means and uncentered rank-two SVD on complete autonomous trajectories."""
from pathlib import Path
import json
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_motivation_mpl")
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from autonomous_trajectory_surface import trajectory_windows, PLOT_A_HALF_WIDTH

OUT = Path(__file__).resolve().parent / "figures"
SEED = 20260929
K = 6
COLORS = ["#4388C7", "#E5A33B", "#42AB99", "#9473BC", "#E17A69", "#C46E9E"]


def collect(n, rng):
    """Independent three-sample rollouts; each row is a full H_3 column."""
    a = rng.uniform(-PLOT_A_HALF_WIDTH, PLOT_A_HALF_WIDTH, n)
    b = rng.uniform(-1, 1, n)
    return trajectory_windows(a, b)


def fit(train, k=K):
    km = KMeans(n_clusters=k, n_init=30, random_state=SEED).fit(train)
    order = np.lexsort((km.cluster_centers_[:, 0], km.cluster_centers_[:, 1]))
    inverse = np.empty(k, dtype=int)
    inverse[order] = np.arange(k)
    labels = inverse[km.labels_]
    bases, spectra, normals = [], [], []
    for j in range(k):
        # Do NOT subtract the centroid: these are linear, not affine, subspaces.
        _, s, vt = np.linalg.svd(train[labels == j], full_matrices=False)
        basis = vt[:2].T.copy()
        for col in range(2):
            basis[:, col] *= np.sign(basis[np.argmax(abs(basis[:, col])), col])
        normal = np.cross(basis[:, 0], basis[:, 1])
        normal *= np.sign(normal[2])
        bases.append(basis)
        spectra.append(s)
        normals.append(normal)
    bases = np.array(bases)
    return dict(centers=km.cluster_centers_[order], labels=labels,
                bases=bases, singular_values=np.array(spectra),
                normals=np.array(normals), projectors=bases @ bases.transpose(0, 2, 1),
                inertia=float(km.inertia_))


def assign(w, atlas):
    return np.argmin(np.sum((w[:, None]-atlas["centers"][None])**2, axis=2), axis=1)


def project(w, atlas, labels=None):
    labels = assign(w, atlas) if labels is None else labels
    return np.einsum("nij,nj->ni", atlas["projectors"][labels], w)


def rms(w, fitted):
    return float(np.sqrt(np.mean(np.sum((w-fitted)**2, axis=1))))


def evaluate(train, test, atlas):
    k = len(atlas["centers"])
    test_labels = assign(test, atlas)
    train_fit = project(train, atlas, atlas["labels"])
    test_fit = project(test, atlas, test_labels)
    global_basis = np.linalg.svd(train, full_matrices=False)[2][:2].T
    charts = []
    for j in range(k):
        members = train[atlas["labels"] == j]
        selected_test = test_labels == j
        b, p, normal = atlas["bases"][j], atlas["projectors"][j], atlas["normals"][j]
        assert np.allclose(b.T @ b, np.eye(2), atol=1e-12)
        assert np.allclose(p @ p, p, atol=1e-12)
        assert np.allclose(normal @ b, 0, atol=1e-12)
        charts.append(dict(cluster=j+1, train_count=len(members), test_count=int(selected_test.sum()),
                           exact_linear_rank=int(np.linalg.matrix_rank(members)),
                           centroid=atlas["centers"][j].tolist(),
                           basis=b.tolist(), normal=normal.tolist(), projector=p.tolist(),
                           singular_values=atlas["singular_values"][j].tolist(),
                           retained_energy=float(np.sum(atlas["singular_values"][j, :2]**2)/
                                                 np.sum(atlas["singular_values"][j]**2)),
                           train_rms=rms(members, members @ p),
                           test_rms=rms(test[selected_test], test_fit[selected_test])))
    assert np.array_equal(assign(train, atlas), atlas["labels"])
    for w in (train, test):
        assert np.max(abs(w[:, 2]-(w[:, 0]+w[:, 1]**2)/2)) < 1e-12
    return dict(system="y[k+2] = (y[k] + y[k+1]^2)/2", L=3, m=0, p=1,
                seed=SEED, K=k, rank=2, train_count=len(train), test_count=len(test),
                sampling="Independent length-three rollouts with uniform initial pairs; no input",
                initial_y0_range=[-PLOT_A_HALF_WIDTH, PLOT_A_HALF_WIDTH], initial_y1_range=[-1, 1],
                clustering="Euclidean K-means on full (y0,y1,y2), no standardization",
                fitting="Uncentered rank-two SVD per cluster: S_j = im(U_j); origin fixed",
                test_assignment="Nearest training K-means centroid, not nearest subspace",
                train_rms=rms(train, train_fit), test_rms=rms(test, test_fit),
                global_rank_two_test_rms=rms(test, test @ global_basis @ global_basis.T),
                kmeans_inertia=atlas["inertia"], charts=charts)


def plane_polygon(members, basis):
    """Finite representative patch of an infinite plane, always including origin."""
    coordinates = np.vstack([members @ basis, np.zeros((1, 2))])
    polygon = coordinates[ConvexHull(coordinates).vertices] @ basis.T
    assert np.allclose(polygon, polygon @ basis @ basis.T, atol=1e-12)
    return polygon


def figures(train, atlas, metrics):
    k = len(atlas["centers"])
    prefix = "autonomous_kmeans" if k == K else f"autonomous_kmeans_k{k}"
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.transforms import Bbox
    from mpl_toolkits.mplot3d import proj3d
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    style = {"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 9,
             "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
             "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
             "pdf.fonttype": 42, "svg.fonttype": "path"}
    a, b = np.meshgrid(np.linspace(-PLOT_A_HALF_WIDTH, PLOT_A_HALF_WIDTH, 37), np.linspace(-1, 1, 45))
    grid = trajectory_windows(a, b).reshape(*a.shape, 3)

    def axes(ax):
        ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), zlim=(-0.56, 1.06))
        ax.set_box_aspect((1, 1, .78), zoom=1.25 if k == 3 else 1.6)
        # View all three fitted planes more face-on, avoiding the former
        # near-edge-on blue plane. Keep the six-group reference view unchanged.
        ax.view_init(elev=35 if k == 3 else 27, azim=160 if k == 3 else -63)
        ax.set_proj_type("ortho")
        ax.set_axis_off()

    def surface(ax):
        ax.plot_surface(*grid.transpose(2, 0, 1), color="#A9BEC9", alpha=.12,
                        linewidth=0, antialiased=True, shade=False, zorder=0)
        for edge in [grid[0], grid[-1], grid[:, 0], grid[:, -1]]:
            ax.plot(*edge.T, color="#81949D", alpha=.6, lw=.5, zorder=1)
        for curve in grid[::7]:
            ax.plot(*curve.T, color="#9BAEB6", alpha=.35, lw=.35, zorder=1)
        for curve in grid[:, ::8].transpose(1, 0, 2):
            ax.plot(*curve.T, color="#9BAEB6", alpha=.35, lw=.35, zorder=1)

    def group(ax, j, planes=False, residuals=False, marker_size=2.4):
        members = train[atlas["labels"] == j]
        # A deterministic subset keeps the figure readable; all data enter the fit.
        shown = members[np.linspace(0, len(members)-1, min(160, len(members))).astype(int)]
        if planes:
            polygon = plane_polygon(members, atlas["bases"][j])
            patch = Poly3DCollection([polygon], facecolors=COLORS[j], edgecolors=COLORS[j],
                                     alpha=.24, linewidths=.65, zorder=2)
            ax.add_collection3d(patch)
        ax.scatter(*shown.T, s=marker_size, color=COLORS[j], alpha=.75, depthshade=False,
                   edgecolors="none", zorder=3)
        if residuals:
            candidates = shown[np.linspace(0, len(shown)-1, 9).astype(int)]
            projected = candidates @ atlas["projectors"][j]
            for w, wp in zip(candidates, projected):
                ax.plot(*np.stack([w, wp]).T, color="#495460", lw=.5, alpha=.65, zorder=4)

    def coordinate_axes(ax):
        """Subtle coordinate arrows intersecting at the true zero trajectory."""
        origin = np.zeros(3)
        for direction in np.eye(3):
            ax.plot(*np.stack([-.18*direction, origin]).T, color="#596E79",
                    alpha=.50, lw=.4, zorder=4)
            ax.quiver(*origin, *(.50*direction), color="#596E79", alpha=.65,
                      linewidth=.45, arrow_length_ratio=.12, normalize=False,
                      zorder=4)
        ax.scatter(*origin, s=3, color="#455861", edgecolors="white",
                   linewidths=.25, depthshade=False, zorder=5)
        origin_xy = proj3d.proj_transform(*origin, ax.get_proj())[:2]
        ax.annotate(r"$(0,\,0,\,0)$", xy=origin_xy, xytext=(3, -15),
                    textcoords="offset points", fontsize=5.5, color="#455861",
                    ha="left", va="top", zorder=6)

    def save(fig, name, fixed_size=False):
        bounds = "tight"
        if fixed_size:
            # Trim all unused margins, leaving 0.02 inches around visible content.
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            ink = np.any(rgb < 245, axis=2)
            ink_rows = np.flatnonzero(np.any(ink, axis=1))
            ink_columns = np.flatnonzero(np.any(ink, axis=0))
            bottom = max(0, (rgb.shape[0]-1-ink_rows[-1])/fig.dpi - .02)
            width, height = fig.get_size_inches()
            top = min(height, (rgb.shape[0]-ink_rows[0])/fig.dpi + .02)
            left = max(0, ink_columns[0]/fig.dpi - .02)
            right = min(width, (ink_columns[-1]+1)/fig.dpi + .02)
            bounds = Bbox.from_bounds(left, bottom, right-left, top-bottom)
        for extension in ("pdf", "svg", "png"):
            fig.savefig(OUT / f"{name}.{extension}", dpi=600,
                        bbox_inches=bounds, pad_inches=0 if fixed_size else .035)
        plt.close(fig)

    with plt.rc_context(style):
        paper_size = k == 3
        fig = plt.figure(figsize=(2, 1.15) if paper_size else (7, 3.05))
        headings = (["(a) Clusters", "(b) Subspaces"] if paper_size else
                    ["(a) K-means trajectory groups", "(b) Rank-two linear subspaces"])
        for i, label in enumerate(headings):
            horizontal_offset = (.045 if i == 0 else -.045) if paper_size else 0
            bounds = [i*.5 + horizontal_offset, .13, .5, .76] if paper_size else [i*.5, .04, .5, .92]
            ax = fig.add_axes(bounds, projection="3d", computed_zorder=False)
            axes(ax)
            surface(ax)
            for j in range(k):
                group(ax, j, planes=i == 1, marker_size=1.15 if paper_size else 2.4)
            if paper_size:
                coordinate_axes(ax)
            if i == 1 and not paper_size:
                ax.scatter(0, 0, 0, color="#263541", s=14, depthshade=False, zorder=5)
                ax.text(0, 0, -.12, "0", fontsize=9, color="#263541", zorder=6)
            if not paper_size:
                fig.text(.25+i*.5 + horizontal_offset, .93,
                         label, ha="center", fontsize=10)
        handles = [Line2D([], [], marker="o", ls="", color=COLORS[j], markersize=3 if paper_size else 4,
                          label=f"{j+1}") for j in range(k)]
        fig.legend(handles=handles, loc="lower center", ncol=k, frameon=False,
                   handletextpad=.3, columnspacing=1.3 if paper_size else 1.6,
                   fontsize=8 if paper_size else 9, bbox_to_anchor=(.5, .015 if paper_size else .01))
        save(fig, f"{prefix}_overview", fixed_size=paper_size)

        ncols, nrows = min(3, k), (k+2)//3
        fig = plt.figure(figsize=(2.8*ncols, 2.65*nrows))
        for j in range(k):
            row, col = divmod(j, ncols)
            panel_height = .94/nrows
            bottom = .03 + (nrows-1-row)*panel_height
            ax = fig.add_axes([col/ncols, bottom, 1/ncols, panel_height], projection="3d", computed_zorder=False)
            axes(ax)
            surface(ax)
            group(ax, j, planes=True, residuals=True)
            ax.scatter(0, 0, 0, color="#263541", s=13, depthshade=False, zorder=5)
            ax.text(0, 0, -.12, "0", fontsize=8, color="#263541", zorder=6)
            fig.text((col+.5)/ncols, bottom + panel_height - .03,
                     rf"$\mathcal{{S}}_{j+1}$   (test RMS {metrics['charts'][j]['test_rms']:.3f})",
                     ha="center", color=COLORS[j], fontsize=10)
        fig.text(.5, .015, "Colored dots: trajectories   •   Shaded polygons: linear-plane patches   •   Black dot: origin",
                 ha="center", fontsize=9)
        save(fig, f"{prefix}_subspaces")


def main(k=K):
    if not 1 <= k <= len(COLORS):
        raise ValueError(f"Choose between 1 and {len(COLORS)} clusters.")
    prefix = "autonomous_kmeans" if k == K else f"autonomous_kmeans_k{k}"
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    train, test = collect(12000, rng), collect(3000, rng)
    atlas = fit(train, k=k)
    metrics = evaluate(train, test, atlas)
    np.savez_compressed(OUT / f"{prefix}_atlas.npz", train=train, test=test,
                        test_labels=assign(test, atlas), **atlas)
    (OUT / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    with (OUT / f"{prefix}_subspaces.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster", "train_count", "exact_linear_rank", "n0", "n1", "n2", "train_rms", "test_rms"])
        for chart in metrics["charts"]:
            writer.writerow([chart["cluster"], chart["train_count"], chart["exact_linear_rank"],
                             *chart["normal"], chart["train_rms"], chart["test_rms"]])
    figures(train, atlas, metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k != "charts"}, indent=2))
    return train, test, atlas, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", type=int, default=K)
    main(k=parser.parse_args().clusters)
