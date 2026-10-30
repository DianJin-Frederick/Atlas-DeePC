"""Add full-controlled-rollout fingertip RMSE to a saved four-case report."""

import argparse
import json
from pathlib import Path

import numpy as np

from atlas_deepc.ablations import basic_four_cases_osqp as four


def full_rollout_rmse(distances):
    errors = np.asarray(distances, dtype=float)
    if errors.ndim != 2 or not errors.size or not np.all(np.isfinite(errors)):
        raise ValueError("Expected a finite, nonempty seeds-by-control-steps error array")
    per_seed = 1000 * np.sqrt(np.mean(errors ** 2, axis=1))
    pooled = float(1000 * np.sqrt(np.mean(errors ** 2)))
    return pooled, per_seed


def update(root):
    root = Path(root)
    report = json.loads((root / "four_case_table.json").read_text())
    seeds = report["seeds"]
    records = []
    for column in report["columns"]:
        p = column["output_dimension"]
        mode = "graph" if column["graph_guidance"] else "terminal"
        directory = root / f"output{p}/{mode}"
        manifest = json.loads((directory / "manifest.json").read_text())
        errors = []
        for seed in seeds:
            with np.load(directory / f"seed{seed}_control_results.npz") as trace:
                target = trace["final_goal"].reshape(-1)[6:8]
                distance = trace["distance"]
                assert distance.shape == (manifest["config"]["steps"],)
                np.testing.assert_allclose(distance, np.linalg.norm(trace["y"][:, 6:8]-target, axis=1))
                errors.append(distance.copy())
        pooled, per_seed = full_rollout_rmse(errors)
        final_errors = 1000 * np.mean(np.asarray(errors)[:, -100:], axis=1)
        np.testing.assert_allclose(np.median(final_errors), column["median_error_mm"])
        column["full_rollout_rmse_mm"] = pooled
        column["median_rollout_rmse_mm"] = float(np.median(per_seed))
        records.append({"output_dimension": p, "graph_guidance": mode == "graph",
                        "full_rollout_rmse_mm": pooled, "controlled_steps_per_seed": len(errors[0]),
                        "median_rollout_rmse_mm": float(np.median(per_seed)),
                        "pooled_sample_count": int(np.asarray(errors).size),
                        "per_seed_rmse_mm": {str(s): float(e) for s, e in zip(seeds, per_seed)}})
    report["pooled_rmse_definition"] = (
        "1000*sqrt(mean_{seed,t}(||true_fingertip(seed,t)-final_target(seed)||_2^2)); "
        "all saved post-action controlled samples, including transients; warmup excluded; "
        "not an average of per-seed RMSEs and not per-coordinate RMSE")
    report["rmse_definition"] = (
        "median_{seed}(1000*sqrt(mean_t(||true_fingertip(seed,t)-final_target(seed)||_2^2))); "
        "all saved post-action controlled samples per rollout, including transients; warmup excluded")
    four.save_json(root / "full_rollout_rmse.json", {"definition": report["rmse_definition"],
                   "pooled_definition": report["pooled_rmse_definition"], "cases": records})
    four.save_json(root / "four_case_table.json", report)
    assert report["lambda_du"] == report["lambda_g"] == 0 and report["solver"] == "CLARABEL"
    assert seeds == list(range(211, 311))
    caption = r"""Four Atlas-DeePC baselines on 100 seeds (211--310), with
    $\lambda_g=\lambda_{\Delta u}=0$, $\lambda_{\mathrm{ini}}=10^5$, rank 60,
    and Clarabel. Charts are selected by full-window projection residual.
    Unguided cases use one fixed nearest eligible terminal configuration;
    guided cases use chordal graph costs and iterative $N=10$ terminal selection.
    Median and percentile errors summarize the final-100 mean fingertip error
    across seeds. Median rollout RMSE is the median across 100 seeds of
    each rollout's fingertip-distance RMSE relative to the final target,
    computed over all 320 controlled steps, excluding warmup.
    Timings pool online steps after discarding the first
    five per rollout. Bold indicates the best observed control-performance value."""
    table = format_table(report["columns"], caption, "tab:four-baselines-no-input-smoothing")
    (root / "table_clarabel_lamg0.tex").write_text(table)
    (root / "table_clarabel_lamg0_preview.tex").write_text(standalone(table))
    print(table)
    return report


def format_table(columns, caption, label, include_rank=False):
    table_label = label
    lines = [r"\begin{table}[htbp]", r"    \centering", r"    \small",
             r"    \begin{tabular}{lrrrr}", r"        \toprule",
             r"        Output dimension & \multicolumn{2}{c}{$p=8$} & \multicolumn{2}{c}{$p=10$} \\",
             r"        \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"        Graph guidance & No & Yes & No & Yes \\", r"        \midrule"]
    if include_rank:
        lines.append("        Selected rank & " + " & ".join(str(c["rank"]) for c in columns) + r" \\")
    rows = [(r"Success rate $\uparrow$", "success_rate", 0, True),
            (r"Median error (mm) $\downarrow$", "median_error_mm", 2, False),
            *[(rf"{q}th-percentile error (mm) $\downarrow$", f"p{q}_error_mm", 3, False)
              for q in (90, 95, 99)],
            (r"Median rollout RMSE (mm) $\downarrow$", "median_rollout_rmse_mm", 2, False)]
    for label, key, decimals, maximize in rows:
        values = [c[key] for c in columns]
        best = max(values) if maximize else min(values)
        cells = []
        for value in values:
            cell = f"{value:.{decimals}f}" + (r"\%" if key == "success_rate" else "")
            cells.append(r"\textbf{"+cell+"}" if np.isclose(value, best, rtol=0, atol=1e-12) else cell)
        lines.append("        " + label + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"        \midrule", r"        \multicolumn{5}{l}{\textit{Total online computation time (ms)}} \\"])
    for label, key in (("Median", "median_time_ms"), ("95th percentile", "p95_time_ms"),
                       ("99th percentile", "p99_time_ms")):
        lines.append("        " + label + " & " + " & ".join(f"{c[key]:.2f}" for c in columns) + r" \\")
    lines.extend([r"        \bottomrule", r"    \end{tabular}",
        r"    \caption{"+caption+"}", r"    \label{"+table_label+"}", r"\end{table}"])
    return "\n".join(lines)+"\n"


def standalone(table):
    return "\n".join([r"\documentclass[10pt]{article}",
        r"\usepackage[landscape,margin=1in]{geometry}", r"\usepackage{booktabs}",
        r"\usepackage{times}", r"\pagestyle{empty}", r"\begin{document}", table, r"\end{document}"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    update(parser.parse_args().root)
