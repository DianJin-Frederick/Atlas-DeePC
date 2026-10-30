"""Saved-result figures and verified media for the promoted Reacher baseline."""

import json
from pathlib import Path

import numpy as np

from atlas_deepc.ablations import four_baseline_projection_chordal as study
from atlas_deepc.common.files import file_digest
from atlas_deepc.common.metrics import strict_success

RESULT_ROOT = study.RESULT_ROOT
REPORT_ROOT = RESULT_ROOT / "baseline_report"


def load_panels(root=RESULT_ROOT, seeds=study.SEEDS):
    root = Path(root)
    panels = {}
    for p, mode in study.four.CASES:
        directory = root / f"output{p}/{mode}"
        panel = json.loads((directory / "summary.json").read_text())
        signature = json.loads((directory / "manifest.json").read_text())
        assert panel["signature"] == signature
        assert signature["chart_selection_method"] == "full_window_projection"
        assert signature["config"]["lam_g"] == 0 and signature["solver"] == "CLARABEL"
        assert signature["config"]["svd_rank"] == 60 and signature["warmup_steps"] == 12
        assert not signature["standardize"] and not signature["coefficient_bounds_enabled"]
        assert signature["terminal_reference_policy"] == study.four.experiment.terminal_policy(mode)
        assert signature["graph_cost_method"] == ("chordal" if mode == "graph" else "not_used")
        assert file_digest(root / f"output{p}/behavioral_atlas.npz") == signature["atlas_sha256"]
        assert file_digest(study.four.DATA_PATH) == signature["data_sha256"]
        assert panel["seeds"] == list(seeds)
        rows = {r["seed"]: r for r in panel["results"]}
        assert set(rows) == set(seeds)
        for seed in seeds:
            assert (directory / f"seed{seed}_control_results.npz").exists()
        panels[p, mode] = panel
    return panels


def chart_clouds(p, root=RESULT_ROOT, report_root=REPORT_ROOT, *, atlas_file=None):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    report_root = Path(report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    experiment = study.four.experiment
    H = experiment.load_hankel(study.four.DATA_PATH, p)
    atlas = experiment.routed.load_atlas(
        Path(atlas_file) if atlas_file is not None else Path(root) / f"output{p}/behavioral_atlas.npz")
    Y = H[24:].reshape(12, p, -1)
    past_color, future_color, representative_color = "#6699CC", "#DF7B20", "#004488"

    def poses(values):
        return np.stack([np.zeros((len(values), 2)), .10*values[:, [0, 2]], values[:, 6:8]], axis=1)

    chart_count = len(atlas["medoid_states"])
    rows = (chart_count + 7) // 8
    fig, axes = plt.subplots(rows, 8, figsize=(19, 2.5*rows), squeeze=False)
    rng, sampled_columns = np.random.default_rng(2026), {}
    for k, ax in enumerate(axes.flat):
        if k >= chart_count:
            ax.set_visible(False)
            continue
        members = np.flatnonzero(atlas["labels"] == k)
        sampled = np.sort(rng.choice(members, min(8, len(members)), replace=False))
        sampled_columns[k] = sampled.tolist()
        block = Y[:, :, sampled].transpose(2, 0, 1)
        for arms, color, alpha in [(poses(block[:, 2:].reshape(-1, p)), future_color, .22),
                                   (poses(block[:, :2].reshape(-1, p)), past_color, .5)]:
            ax.add_collection(LineCollection(arms, colors=color, linewidths=.8, alpha=alpha))
            ax.scatter(arms[:, 2, 0], arms[:, 2, 1], s=3, color=color, alpha=alpha, linewidths=0)
        representative = poses(atlas["medoid_states"][k][None])[0]
        ax.plot(representative[:, 0], representative[:, 1], color="white", lw=4.7)
        ax.plot(representative[:, 0], representative[:, 1], "-o", color=representative_color,
                lw=2.4, markersize=3.5, markerfacecolor="white", zorder=5)
        ax.scatter([0], [0], s=12, color="#333333", zorder=6)
        ax.axhline(0, color="#DDDDDD", lw=.6, zorder=0)
        ax.axvline(0, color="#DDDDDD", lw=.6, zorder=0)
        ax.set(xlim=(-.22, .22), ylim=(-.22, .22), aspect="equal", xticks=[], yticks=[])
        ax.set_title(rf"Chart {k} | $|\mathcal{{I}}_k|={len(members)}$", fontsize=10)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#AAAAAA")
    handles = [Line2D([], [], color=past_color, lw=2, label="Past poses (2 per trajectory)"),
               Line2D([], [], color=future_color, lw=2, label="Future poses (10 per trajectory)"),
               Line2D([], [], color=representative_color, lw=2.5, marker="o",
                      markerfacecolor="white", label="Nearest-centroid last-past pose")]
    fig.suptitle(f"Complete-trajectory chart configurations: {p} outputs, K={chart_count}", y=.997, fontsize=18)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .976), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .945), h_pad=1.1, w_pad=.7)
    for suffix in ("png", "pdf"):
        fig.savefig(report_root / f"physical_chart_clouds_output{p}.{suffix}", dpi=160, bbox_inches="tight")
    (report_root / f"cloud_samples_output{p}.json").write_text(json.dumps(sampled_columns, indent=2))
    return fig


def media(root=RESULT_ROOT, report_root=REPORT_ROOT, seeds=(212, 277), render=False):
    from atlas_deepc.reporting import render_saved_rollout as replay
    from atlas_deepc.reporting import render_routed_deepc_dashboard as dashboard
    panels = load_panels(root)
    entries = []
    for (p, mode), panel in panels.items():
        directory = Path(root) / f"output{p}/{mode}"
        atlas_path = directory.parent / "behavioral_atlas.npz"
        media_dir = Path(report_root) / "videos" / f"output{p}/{mode}"
        media_dir.mkdir(parents=True, exist_ok=True)
        by_seed = {r["seed"]: r for r in panel["results"]}
        for seed in seeds:
            row = by_seed[seed]
            source = directory / f"seed{seed}_control_results.npz"
            digest = file_digest(source)
            with np.load(source, allow_pickle=True) as trace:
                stop = len(trace["y"])
                if strict_success(row):
                    for t in range(1, stop-10):
                        if (np.all(trace["distance"][t:] < .01)
                            and np.max(np.linalg.norm(trace["y"][t:, 6:8]-trace["y"][t, 6:8], axis=1)) < .001
                            and np.max(np.linalg.norm(trace["y"][t:, :4]-trace["y"][t, :4], axis=1)) < .01):
                            stop = t+1
                            break
            video = media_dir / f"seed{seed}_simulator.mp4"
            movie = media_dir / f"seed{seed}_dashboard_025x.mp4"
            graph_digest = file_digest(directory / "graph_geometry.npz") if mode == "graph" else None
            for path, kind in ((video, "simulator"), (movie, "dashboard")):
                meta_path = path.with_suffix(".json")
                meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                valid = path.exists() and meta.get("source_result_sha256") == digest
                if kind == "simulator":
                    valid = valid and meta.get("warmup_zeros") == 12 and meta.get("fps") == 50
                else:
                    valid = (valid and meta.get("renderer_sha256") == file_digest(dashboard.__file__)
                             and meta.get("source_graph_geometry_sha256") == graph_digest
                             and meta.get("source_atlas_sha256") == file_digest(atlas_path)
                             and meta.get("displayed_control_steps") == stop)
                if not valid and render:
                    print(f"Rendering {kind}: p={p}, {mode}, seed={seed}", flush=True)
                    if kind == "simulator":
                        replay.render_saved_rollout(source, seed, path, fps=50, frame_stride=1,
                            warmup_zeros=12, verification_path=meta_path,
                            output_from_obs=lambda obs, p=p: replay.pipe.reacher_output_from_obs(obs)[:p])
                        meta = json.loads(meta_path.read_text())
                        meta.update(source_result_sha256=digest, physical_play_speed=1.)
                        meta_path.write_text(json.dumps(meta, indent=2))
                    else:
                        dashboard.render_dashboard(source, atlas_path, path, fps=25, frame_stride=2,
                            play_speed=.25, end_step=stop, hold_seconds=1., poster_step=min(70, stop-1),
                            evaluation_label=f"Projection selection | p={p}")
                    valid = True
                if not valid:
                    raise ValueError(f"Missing or outdated media: {path}. Enable RENDER_MEDIA.")
            entries.append({"output_dimension": p, "mode": mode, "seed": seed,
                            "success": strict_success(row), "final100_mm": 1000*row["mean_last_100"],
                            "simulator": str(video), "dashboard": str(movie), "displayed_steps": stop})
    (Path(report_root) / "media_manifest.json").write_text(json.dumps(entries, indent=2))
    return entries


def gallery(entries, kind):
    from IPython.display import HTML, Video, display
    cards = []
    for entry in entries:
        outcome = "SUCCESS" if entry["success"] else "FAILURE"
        color = "#007A5E" if entry["success"] else "#B33B24"
        mode = "graph" if entry["mode"] == "graph" else "no graph"
        player = Video(entry[kind], embed=True, width=620)._repr_html_()
        player = player.replace("<video ", '<video style="width:100%;height:auto" ')
        cards.append(f'<div style="min-width:0;background:white;color:#182238;border:1px solid #cbd0d6;padding:12px">'
                     f'<strong>p={entry["output_dimension"]}, {mode}; seed {entry["seed"]}: '
                     f'<span style="color:{color}">{outcome}</span></strong>'
                     f'<p>Final-100 mean: {entry["final100_mm"]:.3f} mm</p>{player}</div>')
    display(HTML('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:12px">'
                 + "".join(cards) + "</div>"))
