"""Offline directed transition graph and count-labeled seed-573 induced subgraph."""
from pathlib import Path
import json
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_paper_mpl")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
ATLAS = ROOT / "experiments/routed_deepc_iid/clustering_jointconfig48_fresh211_310/behavioral_atlas.npz"
ROLLOUT = ROOT / "experiments/routed_deepc_iid/baseline_linear_iterative_n10_v1/evaluation_1000/seed573_control_results.npz"
OUT = ROOT / "paper/figures/offline_chart_transition_graph"
BLUE, INK = "#0077BB", "#273441"


def layout(adjacency):
    """Deterministic force layout, for display only; no added graph edges."""
    n = len(adjacency)
    pos = np.random.default_rng(11).normal(size=(n, 2))
    k = 1 / np.sqrt(n)
    for t in range(700):
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.maximum(np.linalg.norm(diff, axis=2), .015)
        force = (diff * (k*k/dist**2 - adjacency*dist/k)[:, :, None]).sum(axis=1)
        length = np.maximum(np.linalg.norm(force, axis=1), 1e-8)
        pos += force / length[:, None] * np.minimum(length, .10*(1-t/700))[:, None]
        pos -= pos.mean(axis=0)
    pos /= np.max(np.abs(pos))
    # Resolve close node disks without changing graph topology.
    for _ in range(100):
        for i in range(n):
            for j in range(i):
                delta = pos[i] - pos[j]
                distance = np.linalg.norm(delta)
                if distance < .15:
                    shift = .5*(.15-distance)*delta/max(distance,1e-8)
                    pos[i] += shift
                    pos[j] -= shift
    return pos


def arrow(ax, start, end, color, width, rad, shrink, alpha=1, size=7):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>",
        connectionstyle=f"arc3,rad={rad}", shrinkA=shrink, shrinkB=shrink,
        mutation_scale=size, linewidth=width, color=color, alpha=alpha, zorder=1)
    ax.add_patch(patch)


def main():
    with np.load(ATLAS) as a:
        counts, labels = a["transition_counts"].copy(), a["labels"].copy()
    # Independently reconstruct every directed count without crossing episode boundaries.
    reconstructed = np.zeros_like(counts)
    episodes = labels.reshape(200, 189)
    np.add.at(reconstructed, (episodes[:, :-1].ravel(), episodes[:, 1:].ravel()), 1)
    np.testing.assert_array_equal(counts, reconstructed)
    with np.load(ROLLOUT) as a:
        chart = a["chart"].astype(int) + 1
    change_steps = np.r_[0, np.flatnonzero(np.diff(chart)) + 1]
    sequence = chart[change_steps].tolist()
    used_edges = set(zip(sequence[:-1], sequence[1:]))
    selected = np.unique(chart)
    retained = (counts >= 3) & ~np.eye(len(counts), dtype=bool)
    edges = np.argwhere(retained)
    assert len(edges) == 298
    assert all(retained[i-1, j-1] for i, j in used_edges)
    detail_nodes = [43, 16, 44, 5, 26, 48, 30]
    assert sorted(detail_nodes) == selected.tolist()
    detail_edges = [(i, j, int(counts[i-1, j-1])) for i in detail_nodes
                    for j in detail_nodes if retained[i-1, j-1]]
    plt.rcParams.update({"font.family":"serif", "font.serif":["Times New Roman"],
        "font.size":9, "mathtext.fontset":"stix", "text.color":INK,
        "pdf.fonttype":42, "ps.fonttype":42})
    fig = plt.figure(figsize=(6.3, 2.9))
    left = fig.add_axes([.015, .12, .425, .76])
    right = fig.add_axes([.50, .12, .485, .76])
    pos = layout((retained | retained.T).astype(float))
    for i, j in sorted(edges.tolist(), key=lambda e: (e[0]+1, e[1]+1) in used_edges):
        highlighted = (i+1, j+1) in used_edges
        arrow(left, pos[i], pos[j], BLUE if highlighted else "#AEB6BD",
              .65 if highlighted else .35, .08, 4.4,
              alpha=.8 if highlighted else .32, size=4)
    for i in range(48):
        active = i+1 in selected
        left.scatter(*pos[i], s=88, c=BLUE if active else "#F3F5F7",
                     edgecolors=BLUE if active else "#89949E", linewidths=.5, zorder=3)
        left.text(*pos[i], str(i+1), ha="center", va="center", fontsize=5.5,
                  color="white" if active else INK, zorder=4)
    left.set(xlim=(-1.16, 1.16), ylim=(-1.12, 1.12), aspect="equal")
    # Spread the seven charts in a schematic arc, not Cartesian arm coordinates.
    detail_pos = {43:(1.9,-.55), 16:(2.0,.5), 44:(1.5,1.45),
                  5:(.25,1.85), 26:(-1.0,1.45), 48:(-1.5,.5), 30:(-1.0,-.55)}
    right.set(xlim=(-1.95, 2.55), ylim=(-1.0, 2.35), aspect="equal")
    # Layout needs a realized equal-aspect transform for precise Bezier label positions.
    fig.canvas.draw()
    for i, j, count in detail_edges:
        p, q = np.array(detail_pos[i]), np.array(detail_pos[j])
        rad = .19
        if {i,j} == {43,44}:
            # Bypass node 16 on the inside, with separately spaced opposite arrows.
            rad = -.38 if i == 43 else .65
        width = .65 + .60*np.log1p(count)/np.log1p(counts[retained].max())
        arrow(right, p, q, BLUE, width, rad, 10, size=7)
        a, b = right.transData.transform(np.vstack([p,q]))
        delta = b-a
        midpoint = (a+b)/2 + .5*rad*np.array([delta[1],-delta[0]])
        label_pos = right.transData.inverted().transform(midpoint)
        right.text(*label_pos, str(count), ha="center", va="center", fontsize=8,
                   bbox=dict(facecolor="white", edgecolor="none", pad=.6), zorder=3)
    for node, point in detail_pos.items():
        right.scatter(*point, s=330, color="#E7F0F8", edgecolor=BLUE, linewidth=.9, zorder=4)
        right.text(*point, str(node), ha="center", va="center", color="#004488", fontsize=9, zorder=5)
    for ax in (left,right):
        ax.set_axis_off()
    fig.text(.04,.945,"(a) Offline chart graph",ha="left",fontsize=10)
    fig.text(.515,.945,"(b) Seed 573 chart subgraph",ha="left",fontsize=10)
    fig.text(.23,.055,"48 charts; 298 directed edges",ha="center",fontsize=8)
    fig.text(.745,.055,"Arrow labels: observed offline transitions",ha="center",fontsize=8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"),dpi=300)
    plt.close(fig)
    OUT.with_suffix(".json").write_text(json.dumps({
        "atlas":str(ATLAS),"rollout":str(ROLLOUT),"minimum_count":3,
        "self_loops_shown":False,"total_charts":48,"total_retained_edges":len(edges),
        "detail_chart_ids_one_based":detail_nodes,"detail_edges":detail_edges,
        "highlighted_controller_sequence":sequence,
        "controller_chart_change_steps":change_steps.tolist(),
        "highlighted_directed_edges":sorted(used_edges),
        "all_edges":[[int(i+1),int(j+1),int(counts[i,j])] for i,j in edges],
        "count_validation":"Exact match to within-episode adjacent labels, 200 episodes x 189 columns",
        "layout":"Schematic; positions do not encode physical configuration or routing costs"
    },indent=2)+"\n")
    print("Validated",len(edges),"offline directed edges; detail:",detail_edges)
    print(OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
