"""Standalone offline atlas graph for the reference-generation section."""
import json
import argparse
import numpy as np
from plot_offline_chart_graph import ATLAS, ROOT, INK, BLUE, ROLLOUT, layout, arrow
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--highlight-seed573", action="store_true")
    args = parser.parse_args()
    with np.load(ATLAS) as a:
        counts, labels = a["transition_counts"], a["labels"]
    checked = np.zeros_like(counts)
    episodes = labels.reshape(200,189)
    np.add.at(checked,(episodes[:,:-1].ravel(),episodes[:,1:].ravel()),1)
    np.testing.assert_array_equal(counts,checked)
    retained = (counts >= 3) & ~np.eye(len(counts),dtype=bool)
    edges = np.argwhere(retained)
    assert len(counts) == 48 and len(edges) == 298
    sequence, change_steps, used_edges = [], [], set()
    if args.highlight_seed573:
        with np.load(ROLLOUT) as a:
            chart = a["chart"].astype(int) + 1
        change_steps = np.r_[0, np.flatnonzero(np.diff(chart)) + 1].tolist()
        sequence = chart[change_steps].tolist()
        used_edges = set(zip(sequence[:-1], sequence[1:]))
        assert sequence == [43, 16, 44, 5, 26, 48, 30]
        assert all(retained[i-1,j-1] for i,j in used_edges)
    pos = layout((retained | retained.T).astype(float))
    plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman"],
        "font.size":8,"mathtext.fontset":"stix","text.color":INK,
        "pdf.fonttype":42,"ps.fonttype":42})
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_axes([.025,.025,.95,.95])
    for i,j in sorted(edges.tolist(), key=lambda e: (e[0]+1,e[1]+1) in used_edges):
        active = (i+1,j+1) in used_edges
        arrow(ax,pos[i],pos[j],BLUE if active else "#AEB6BD",
              1.05 if active else .4,.08,6.0,
              alpha=1 if active else .32,size=7 if active else 5)
    ax.scatter(pos[:,0],pos[:,1],s=140,
        c=[BLUE if i+1 in sequence else "#F3F5F7" for i in range(len(pos))],
        edgecolors=[BLUE if i+1 in sequence else "#89949E" for i in range(len(pos))],
        linewidths=.65,zorder=3)
    for i,point in enumerate(pos):
        ax.text(*point,str(i+1),ha="center",va="center",fontsize=7.5,
            color="white" if i+1 in sequence else INK,zorder=4)
    ax.set(xlim=(-1.12,1.12),ylim=(-1.12,1.12),aspect="equal")
    ax.set_axis_off()
    stem = ROOT/"paper/figures/offline_chart_graph_overview"
    if args.highlight_seed573:
        stem = stem.with_name("offline_chart_graph_seed573_closed_loop")
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"),dpi=300)
    plt.close(fig)
    stem.with_suffix(".json").write_text(json.dumps({
        "atlas":str(ATLAS),"charts":48,"directed_edges":298,"minimum_transition_count":3,
        "self_loops_shown":False,"seed_specific_highlighting":args.highlight_seed573,"figure_inches":[3,3],
        "rollout":str(ROLLOUT) if args.highlight_seed573 else None,
        "controller_chart_sequence":sequence,"controller_chart_change_steps":change_steps,
        "highlighted_directed_edges":sorted(used_edges),
        "node_positions":pos.tolist(),"layout":"Same schematic force layout as combined graph",
        "edges":[[int(i+1),int(j+1),int(counts[i,j])] for i,j in edges],
        "count_validation":"Exact reconstruction from within-episode adjacent chart labels"
    },indent=2)+"\n")
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
