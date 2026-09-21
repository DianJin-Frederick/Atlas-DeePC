"""Restore the baseline's full replay/dashboard layout with current-version media."""

import nbformat as nbf


def restore_media_sections(cells):
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    start = next(i for i, c in enumerate(cells) if c.source.startswith("## 5. Current-baseline routing videos"))
    closing = cells[start + 2:]
    closing[0].source = closing[0].source.replace("## 6. Reproduction", "## 7. Reproduction")
    media = [md("""## 5. Atlas routing replays

The original gallery format and seed IDs are restored below: ten representative
simulator replays followed by three diagnostic replays. Every video replays the
**linear-subspace + iterative N=10** baseline, and its measured outputs are checked
against the saved numerical rollout. Labels are recomputed from the current results,
not inherited from the old affine/fixed-reference method.

These are selected diagnostic examples, not a random success-rate panel. Simulator
replays retain the previous compact playback format; the dashboard section below
provides slowed 0.25x views. All videos are embedded for reliable notebook playback.
"""), code('''import base64
from baseline_notebook_media import (
    GALLERY_SEEDS, DIAGNOSTIC_VIDEO_SEEDS, ADDITIONAL_DASHBOARD_SEEDS,
    TRACE_DIR, VIDEO_DIR, prepare_media,
)
from render_routed_deepc_dashboard import file_sha256

PREPARE_MEDIA = False
if PREPARE_MEDIA:
    prepare_media()
video_summary = json.loads((TRACE_DIR / "summary.json").read_text())
assert video_summary["method"] == baseline.BASELINE_VERSION
video_by_seed = {int(row["seed"]): row for row in video_summary["results"]}

def embedded_media(path, mime):
    return "data:" + mime + ";base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")

def media_card(seed, dashboard=False):
    row = video_by_seed[seed]
    assert row["baseline_version"] == baseline.BASELINE_VERSION
    assert row["matrix_geometry"] == "linear" and row["candidate_count"] == 10
    trace = TRACE_DIR / f"seed{seed}_control_results.npz"
    name = f"seed{seed}_025x" if dashboard else f"seed{seed}_control"
    video = VIDEO_DIR / (name + ".mp4")
    metadata_path = VIDEO_DIR / (name + (".json" if dashboard else "_verification.json"))
    assert video.exists() and metadata_path.exists(), f"Prepare media first: {video}"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["source_result_sha256"] == file_sha256(trace)
    if dashboard:
        assert metadata["play_speed"] == 0.25
    else:
        assert metadata["maximum_replay_output_error"] <= 1e-8
    success = bool(row["success_30"] and row["success_100"])
    status, color = ("SUCCESS", "#166534") if success else ("FAILURE", "#b91c1c")
    target = video_summary["fairness_audit"]["held_out_targets"][str(seed)]["target"]
    detail = f"Final-30 mean: {1000*row['mean_last_30']:.3f} mm | Target: ({target[0]:+.3f}, {target[1]:+.3f})"
    poster_attribute = ""
    if dashboard:
        poster = VIDEO_DIR / (name + "_poster.png")
        poster_attribute = f' poster="{embedded_media(poster, "image/png")}"'
        detail += f" | 0.25x | {metadata['displayed_control_steps']} of {metadata['saved_control_steps']} steps"
    return f"""<div style="border:1px solid #d1d5db;padding:10px;background:#fff;color:#111827;min-width:0">
      <div style="font:600 14px sans-serif;margin-bottom:6px">Seed {seed} &nbsp; <span style="color:{color}">{status}</span></div>
      <div style="font:12px sans-serif;margin-bottom:8px;color:#4b5563">{detail}</div>
      <video src="{embedded_media(video, 'video/mp4')}"{poster_attribute} controls preload="metadata"
             style="display:block;width:100%;background:#f5f7fa"></video></div>"""

def show_gallery(seeds, title, dashboard=False):
    width = 480 if dashboard else 280
    cards = "".join(media_card(seed, dashboard) for seed in seeds)
    display(HTML(f'<h3>{title}</h3><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min({width}px,100%),1fr));gap:14px">{cards}</div>'))

all_gallery_success = all(video_by_seed[s]["success_30"] and video_by_seed[s]["success_100"] for s in GALLERY_SEEDS)
show_gallery(GALLERY_SEEDS, "Ten representative baseline successes" if all_gallery_success else "Ten representative baseline rollouts (current outcomes)")
show_gallery(DIAGNOSTIC_VIDEO_SEEDS, "Baseline success and failure diagnostics")
print("All 13 simulator replays verified against the current baseline's numerical outputs.")
'''), md("""## 6. Slowed Atlas-routing dashboard

The main dashboard uses **seed 222**, followed by the original five additional
dashboard seeds. Each shows the measured arm, one-step DeePC prediction, active
waypoint posture, graph route, changing goal chart, target error, and applied inputs.
Playback is **0.25x**. The caption reports the actual displayed duration: new overview
dashboards use 60 steps; already available full-length current-baseline videos are
retained. Full 320-step results determine every success label.
"""), code('''show_gallery([222], "Main slowed baseline dashboard", dashboard=True)
'''), code('''show_gallery(ADDITIONAL_DASHBOARD_SEEDS, "Five additional baseline dashboards", dashboard=True)
'''), md("""### Terminal stabilization diagnostic: seed 232

The full rollout is shown so late drift remains visible. The error and outcome below
come from the current baseline; no historical failure numbers are reused.
"""), code('''row = video_by_seed[232]
print(f"Seed 232: closest {1000*row['minimum']:.3f} mm; final-100 mean {1000*row['mean_last_100']:.3f} mm; strict success = {row['success_30'] and row['success_100']}")
show_gallery([232], "Seed 232: full terminal-stabilization diagnostic", dashboard=True)
'''), md("""### Former mechanical-failure case: seed 277

This is the same seed that failed in the old method. The new combined baseline's
saved rollout succeeds, so it is now labeled as a repaired example, not a current
mechanical failure. The complete rollout is retained for inspection.
"""), code('''row = video_by_seed[277]
print(f"Seed 277: closest {1000*row['minimum']:.3f} mm; final-100 mean {1000*row['mean_last_100']:.3f} mm; strict success = {row['success_30'] and row['success_100']}")
show_gallery([277], "Seed 277: current-baseline outcome", dashboard=True)
''')]
    return cells[:start] + media + closing
