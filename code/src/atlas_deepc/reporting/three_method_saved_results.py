"""Compare saved serial runs only; never launch or modify controller experiments."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/reacher"
OUTPUT = BASE / "three_method_p8_no_graph_saved_comparison"
SEEDS = list(range(211, 311))
SOURCES = {
    "Atlas-DeePC": BASE / "projection_chordal_rank_lamg0_lamdu0_clarabel_v1/selected_serial/rank053/output8/terminal",
    "Select-DPC": BASE / "select_dpc_upstream_final_clarabel_211_310_v1",
    "Contextual DPC": BASE / "contextual_dpc_raw_final_211_310_q100_r01_serial_v1",
}
NOTES = [
    "100 development seeds 211–310, 320 post-action controlled samples per seed; failures included. No simulations are launched.",
    "Success requires BOTH final-30 and final-100 windows to have mean XY error <10 mm and maximum <20 mm. Error quantiles summarize per-seed final-100 mean errors. Rollout RMSE uses all 320 steps.",
    "Main timing statistics pool steps 6–320 (31,500 samples per method), matching the Atlas table. All-320-step statistics are also saved. Runs are serial with one numerical thread, at different times on the same computer; background load was not controlled.",
    "Selection is chart localization for Atlas and online raw-column selection for the other methods. Select-DPC times sum all five refinement iterations per applied control step, not one QP.",
    "Construction + solve includes matrix retrieval/preparation and the recorded solve interval. Atlas's solve interval already includes QP construction, so its pure solve-call time cannot be recovered. CVXPY solve-call timers for the other methods include canonicalization/setup, not just solver iterations.",
    "Other online work is total minus selection minus construction/solve. It includes reference bookkeeping, preprocessing, extraction and measured diagnostics as applicable. Means add to the total; medians and percentiles generally do not. Offline preparation, simulation stepping, file logging, and one-time controller/goal setup are excluded from per-step totals.",
    "No method uses graph guidance. Atlas uses a fixed offline-derived terminal posture in addition to the same Cartesian target; the other methods track only the final Cartesian target. Contextual uses 8 physical outputs plus 2 appended inputs internally (10 augmented outputs).",
    "This is an adapted-method development comparison, not an otherwise-identical ablation or held-out test: Atlas rank 53 and contextual Q=100/R=0.1 were selected on these seeds. Select-DPC retains authors' data, Reacher-v4, horizon 15, no physical warmup, input bounds ±0.5 and fingertip y≤0.1. Atlas/contextual share offline data, use Reacher-v5, horizon 10, 12 zero-input warmup steps and input bounds ±1. Solver tolerances/objectives differ. All use Clarabel.",
    "Native weighted control costs are intentionally not ranked: their weights, normalizations and tracked channels differ. Unweighted fingertip errors/RMSE are comparable task metrics.",
]


def stats_ms(seconds):
    x = np.asarray(seconds, dtype=float).reshape(-1) * 1000
    assert x.size and np.isfinite(x).all() and x.min() >= -1e-5
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            **{f"p{q}": float(np.percentile(x, q)) for q in (95, 99)},
            "max": float(x.max()), "samples": int(x.size)}


def collect():
    records, shared_targets = [], {}
    for label, directory in SOURCES.items():
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["workers"] == 1
        assert manifest.get("threads", manifest.get("numerical_threads")) == 1
        assert manifest["solver"] == "CLARABEL"
        atlas = label == "Atlas-DeePC"
        if atlas:
            assert manifest["config"]["svd_rank"] == 53
            assert manifest["output_dimension"] == 8 and manifest["reference_mode"] == "terminal"
            assert manifest["timing_workers"] == 1
        elif label == "Contextual DPC":
            assert manifest["config"]["q_xy"] == 100 and manifest["config"]["r"] == .1
            assert manifest["svd"] is False
        else:
            assert manifest["selected_columns"] == 200 and manifest["refinement_iterations"] == 5
        per_seed, modules = [], {}
        for seed in SEEDS:
            with np.load(directory / f"seed{seed}_control_results.npz", allow_pickle=False) as trace:
                row = json.loads(trace["summary"].item())
                assert row["seed"] == seed
                target = trace["final_goal"][6:8] if atlas else trace["target_xy"]
                if seed in shared_targets:
                    np.testing.assert_allclose(target, shared_targets[seed], rtol=0, atol=1e-12)
                else:
                    shared_targets[seed] = target.copy()
                distance = trace["distance"]
                assert distance.shape == (320,) and np.isfinite(distance).all()
                np.testing.assert_allclose(distance, np.linalg.norm(trace["y"][:, 6:8]-target, axis=1))
                success = all(distance[-n:].mean() < .01 and distance[-n:].max() < .02 for n in (30, 100))
                assert success == bool(row["success_30"] and row["success_100"])
                np.testing.assert_allclose(distance[-100:].mean(), row["mean_last_100"])
                per_seed.append({"seed": seed, "success": bool(success),
                                 "final100_error_mm": float(1000*distance[-100:].mean()),
                                 "rollout_rmse_mm": float(1000*np.sqrt(np.mean(distance**2)))})
                selection = trace["localization_time_s" if atlas else "selection_time_s"]
                preparation, solve, total = (trace[k] for k in
                    ("matrix_preparation_time_s", "solve_time_s", "control_compute_time_s"))
                if label == "Select-DPC":
                    for key, expected in [("inner_selection_time_s", selection),
                                          ("inner_construction_time_s", preparation), ("inner_solve_time_s", solve)]:
                        assert trace[key].shape == (320, 5)
                        np.testing.assert_allclose(trace[key].sum(axis=1), expected)
                values = {"selection": selection, "preparation_and_solve": preparation+solve,
                          "other_online": total-selection-preparation-solve, "total": total,
                          "recorded_preparation": preparation, "recorded_solve_interval": solve}
                for key, value in values.items():
                    assert value.shape == (320,) and np.isfinite(value).all()
                    modules.setdefault(key, []).append(value.copy())
        errors = np.array([r["final100_error_mm"] for r in per_seed])
        records.append({"method": label, "source_directory": str(directory),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "successes": sum(r["success"] for r in per_seed), "total": len(SEEDS),
            "median_error_mm": float(np.median(errors)),
            **{f"p{q}_error_mm": float(np.percentile(errors, q)) for q in (90, 95, 99)},
            "median_rollout_rmse_mm": float(np.median([r["rollout_rmse_mm"] for r in per_seed])),
            "timing_discard_first5": {k: stats_ms(np.asarray(v)[:, 5:]) for k, v in modules.items()},
            "timing_all_steps": {k: stats_ms(v) for k, v in modules.items()},
            "recorded_preparation_definition": "Offline basis retrieval only" if atlas else "Problem construction",
            "recorded_solve_definition": "QP construction + solve" if atlas else "CVXPY solve call including canonicalization/setup",
            "per_seed": per_seed})
    # Reproduce the exact Atlas column supplied by the user, without reading notebook outputs.
    saved = json.loads((BASE / "projection_chordal_rank_lamg0_lamdu0_clarabel_v1/best_rank_table.json").read_text())
    reference = next(c for c in saved["columns"] if c["output_dimension"] == 8 and not c["graph_guidance"])
    for key in ("successes", "median_error_mm", "p90_error_mm", "p95_error_mm", "p99_error_mm", "median_rollout_rmse_mm"):
        np.testing.assert_allclose(records[0][key], reference[key])
    for stat in ("median", "p95", "p99"):
        np.testing.assert_allclose(records[0]["timing_discard_first5"]["total"][stat], reference[f"{stat}_time_ms"])
    return {"seeds": SEEDS, "steps": 320, "notes": NOTES, "records": records,
            "validation": "All 300 traces validated; matched Cartesian targets, strict success, error definitions, serial manifests, Select refinement sums and source Atlas table."}


def table_rows(report):
    records = report["records"]
    rows = [("Physical output dimension", ["8", "8", "8 (+2 input augmentation)"]),
            ("Graph guidance", ["No"]*3),
            ("Trajectory representation", ["Rank 53 (offline)", "200 raw columns", "200 raw columns"]),
            ("Solves per control step", ["1", "5", "1"]),
            ("Success rate ↑", [f"{r['successes']}%" for r in records])]
    for label, key, decimals in [("Median error (mm) ↓", "median_error_mm", 2),
                                 *[(f"{q}th-percentile error (mm) ↓", f"p{q}_error_mm", 3) for q in (90, 95, 99)],
                                 ("Median rollout RMSE (mm) ↓", "median_rollout_rmse_mm", 2)]:
        rows.append((label, [f"{r[key]:.{decimals}f}" for r in records]))
    rows.append(("Total online computation (ms/step)", [""]*3))
    for label, key in [("Mean", "mean"), ("Median", "median"), ("95th percentile", "p95"),
                       ("99th percentile", "p99"), ("Maximum", "max")]:
        rows.append((label, [f"{r['timing_discard_first5']['total'][key]:.2f}" for r in records]))
    rows.append(("Modular mean computation (ms/step)", [""]*3))
    for label, key in [("Data/chart selection", "selection"), ("Problem preparation + solve", "preparation_and_solve"),
                       ("Other online work", "other_online")]:
        rows.append((label, [f"{r['timing_discard_first5'][key]['mean']:.3f}" for r in records]))
    return rows


def export(output=OUTPUT):
    """Write a reproducible report, LaTeX table fragment and scientific table image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = collect()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2)+"\n")
    rows = table_rows(report)
    headers = ["Metric", *SOURCES]
    # HTML is a portable, fully self-contained table including the metric definitions.
    from html import escape
    html = '<!doctype html><meta charset="utf-8"><title>Three-method Reacher comparison</title>'
    html += '<style>body{font:16px Georgia;margin:40px;max-width:1200px}table{border-collapse:collapse;width:100%;margin:24px 0}th,td{padding:8px;text-align:right;border-bottom:1px solid #ddd}th:first-child,td:first-child{text-align:left}thead{border-top:2px solid;border-bottom:2px solid}p{line-height:1.45}</style>'
    html += '<h1>Three-method Reacher comparison — no graph guidance</h1><table><thead><tr>'
    html += ''.join(f'<th>{escape(h)}</th>' for h in headers)+'</tr></thead><tbody>'
    for label, values in rows:
        html += '<tr>'+''.join(f'<td>{escape(v)}</td>' for v in [label, *values])+'</tr>'
    html += '</tbody></table><h2>Detailed modular timing (ms/control step)</h2><table><thead><tr>'
    html += ''.join(f'<th>{h}</th>' for h in ['Method', 'Recorded module', 'Mean', 'Median', 'p95', 'p99', 'Max'])+'</tr></thead><tbody>'
    for r in report['records']:
        for key, label in [("selection", "Data/chart selection"), ("recorded_preparation", r['recorded_preparation_definition']),
                           ("recorded_solve_interval", r['recorded_solve_definition']), ("other_online", "Other online work")]:
            values = [r['method'], label, *[f"{r['timing_discard_first5'][key][s]:.3f}" for s in ['mean','median','p95','p99','max']]]
            html += '<tr>'+''.join(f'<td>{escape(v)}</td>' for v in values)+'</tr>'
    html += '</tbody></table>'+''.join(f'<p>{escape(n)}</p>' for n in NOTES)
    (output / "comparison.html").write_text(html)

    def tex(s):
        return s.replace('↑', r'$\uparrow$').replace('↓', r'$\downarrow$').replace('%', r'\%').replace('&', r'\&')
    lines = [r'% Requires booktabs. See comparison.json/HTML for definitions and protocol caveats.',
             r'\begin{tabular}{lrrr}', r'\toprule', ' & '.join(map(tex, headers))+r' \\', r'\midrule']
    for label, values in rows:
        if not any(values):
            lines += [r'\midrule', r'\multicolumn{4}{l}{\textit{'+tex(label)+r'}} \\']
        else:
            lines.append(' & '.join(map(tex, [label, *values]))+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (output / "comparison_table.tex").write_text('\n'.join(lines)+'\n')

    fig, ax = plt.subplots(figsize=(12.8, 9.4))
    ax.axis('off')
    table = ax.table(cellText=[[label, *values] for label, values in rows], colLabels=headers,
                     colWidths=[.42,.18,.18,.22], loc='upper center', bbox=[0,.08,1,.86])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (i,j), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_text_props(ha='left' if j == 0 else 'center')
        if i == 0:
            cell.set_text_props(weight='bold')
            cell.visible_edges = 'TB'; cell.set_linewidth(1.3)
        elif not any(rows[i-1][1]):
            cell.set_facecolor('#eef1f4'); cell.set_text_props(weight='bold', fontsize=10)
        elif i == len(rows):
            cell.visible_edges = 'B'; cell.set_linewidth(1.3)
    ax.set_title('Reacher: 100 development seeds (211–310), 320 control steps', fontsize=15, pad=12)
    fig.text(.04,.035,'Timing: serial, one numerical thread; first 5 steps omitted per rollout. Select-DPC includes all 5 refinements.\n'
             'Construction + solve includes modeling/setup overhead. Different method protocols; see accompanying notes.', fontsize=10)
    fig.tight_layout()
    fig.savefig(output / 'comparison_table.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    return report


if __name__ == '__main__':
    result = export()
    for name, values in table_rows(result):
        print(name+' | '+' | '.join(values))
    print('Saved report:', OUTPUT)
