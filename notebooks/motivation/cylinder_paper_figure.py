"""Reproduce the introduction figure: python notebooks/motivation/cylinder_paper_figure.py."""
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_motivation_mpl")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path(__file__).resolve().parent / "figures"
H, A = 0.12, 0.35


def simulate():
    """The same plant, sweep inputs, and initial angles as the notebook."""
    n_steps = 180
    omega = 2 * np.pi * 1.25 / (n_steps * H)
    vertical = 2 / (n_steps * H)
    paths = []
    for theta in np.linspace(-np.pi, np.pi, 12, endpoint=False):
        y = np.array([np.cos(theta), np.sin(theta), -1.0])
        samples = [y.copy()]
        for _ in range(n_steps):
            v = vertical - A * y[1]
            assert abs(omega) <= 1.2 and abs(v) <= 0.8
            c, s = np.cos(H * omega), np.sin(H * omega)
            y = np.array([c*y[0] - s*y[1], s*y[0] + c*y[1],
                          y[2] + H*(v + A*y[1])])
            samples.append(y)
        paths.append(np.asarray(samples))
    points = np.concatenate(paths)
    assert np.max(np.abs(np.sum(points[:, :2]**2, axis=1) - 1)) < 1e-12
    return paths


def make_figure():
    # Fail explicitly rather than silently substituting another font.
    font_path = font_manager.findfont("Times New Roman", fallback_to_default=False)
    for style in ["normal", "italic"]:
        font_manager.findfont(font_manager.FontProperties(family="Times New Roman", style=style),
                              fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
        "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "path", "savefig.facecolor": "white",
    })
    paths = simulate()
    # Designed for a single-column width of 3.5 inches, with legible 8–9 pt labels.
    fig = plt.figure(figsize=(3.5, 3.7), facecolor="white")
    ax = fig.add_axes([0.015, 0.075, 0.90, 0.89], projection="3d", computed_zorder=False)
    angle = np.linspace(-np.pi, np.pi, 180)
    aa, zz = np.meshgrid(angle, np.linspace(-1, 1, 20))
    ax.plot_surface(np.cos(aa), np.sin(aa), zz, color="#ADC4D1",
                    alpha=0.045, linewidth=0, shade=False, zorder=0)
    for height in [-1, -0.5, 0, 0.5, 1]:
        ax.plot(np.cos(angle), np.sin(angle), np.full_like(angle, height),
                color="#899CA8", alpha=0.45, lw=0.45, zorder=1)
    for theta in np.linspace(-np.pi, np.pi, 8, endpoint=False):
        ax.plot([np.cos(theta)]*2, [np.sin(theta)]*2, [-1, 1],
                color="#899CA8", alpha=0.24, lw=0.4, zorder=1)
    for j, path in enumerate(paths):
        if j != 3:
            ax.plot(*path.T, color="#377EA0", alpha=0.68, lw=0.58, zorder=2)
    path = paths[3]
    ax.plot(*path.T, color="#D46727", lw=1.65, zorder=3)
    ax.scatter(*path[0], color="#D46727", marker="o", s=11, edgecolors="none", zorder=4)
    ax.scatter(*path[-1], color="#D46727", marker="^", s=19, edgecolors="none", zorder=4)
    ax.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), zlim=(-1.06, 1.06),
           xlabel=r"$y_1=\cos\theta$", ylabel=r"$y_2=\sin\theta$", zlabel=r"$y_3=z$")
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_zticks([-1, 0, 1])
    ax.tick_params(pad=0, width=0.6)
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2
    ax.zaxis.labelpad = 1
    ax.set_box_aspect((1, 1, 1.08))
    ax.view_init(elev=27, azim=-57)
    ax.set_proj_type("ortho")
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ["pdf", "svg", "png"]:
        # Keep physical single-column dimensions; do not crop using 3D tight-bbox estimates.
        fig.savefig(OUT / f"cylinder_paper.{extension}", dpi=600)
    plt.close(fig)
    print(f"Times New Roman: {font_path}")
    print(f"Publication figure: {OUT / 'cylinder_paper.pdf'}")
    return paths


if __name__ == "__main__":
    make_figure()
