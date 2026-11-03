"""LaTeX formatting for four-method Reacher performance tables."""

import numpy as np


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

