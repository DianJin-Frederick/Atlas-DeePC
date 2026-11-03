"""K=64 table helper and compatibility entry point for the combined notebook."""

from pathlib import Path

from atlas_deepc.ablations import k64_rank as study


def full_table(records):
    lookup = {(r["rank"], r["output_dimension"], r["reference_mode"]): r for r in records}
    complete = len(lookup) == len(study.RANKS)*len(study.four.CASES)
    maxima = {case: max((r["successes"] for r in records
                        if (r["output_dimension"], r["reference_mode"]) == case), default=-1)
              for case in study.four.CASES}
    lines = [r"% Requires booktabs, diagbox, adjustbox.", r"\begin{table*}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{3pt}", r"\begin{adjustbox}{max width=\linewidth}",
        r"\begin{tabular}{@{}c|cccc||c|cccc||c|cccc@{}}", r"\toprule",
        r"\multicolumn{5}{c}{Ranks 28--42} & \multicolumn{5}{c}{Ranks 43--56} & \multicolumn{5}{c}{Ranks 57--70} \\",
        r"\cmidrule(lr){1-5}\cmidrule(lr){6-10}\cmidrule(lr){11-15}",
        r"& \multicolumn{2}{c}{$p=8$} & \multicolumn{2}{c}{$p=10$} & & \multicolumn{2}{c}{$p=8$} & \multicolumn{2}{c}{$p=10$} & & \multicolumn{2}{c}{$p=8$} & \multicolumn{2}{c}{$p=10$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){12-13}\cmidrule(lr){14-15}",
        " & ".join([r"\diagbox[width=7em,height=2.5em]{Rank}{Guidance} & No & Yes & No & Yes"]*3) + r" \\",
        r"\midrule"]
    blocks = [list(range(28, 43)), list(range(43, 57)), list(range(57, 71))]
    for i in range(15):
        cells = []
        for block in blocks:
            if i >= len(block):
                cells.extend([""]*5)
                continue
            rank = block[i]
            cells.append(str(rank))
            for p, mode in study.four.CASES:
                record = lookup.get((rank, p, mode))
                value = "--" if record is None else str(record["successes"]) + r"\%"
                if record and complete and record["successes"] == maxima[(p, mode)]:
                    value = r"\textbf{" + value + "}"
                cells.append(value)
        lines.append(" & ".join(cells) + r" \\")
    status = "Complete" if complete else "Partial (dashes denote unfinished panels)"
    caption = (status + r" rank ablation with $K=64$ charts: success rates on seeds 211--310. "
        r"No/Yes indicate graph guidance. All cases use unstandardized linear charts, full-window "
        r"projection-residual selection, $\lambda_{\mathrm{ini}}=10^5$, and Clarabel. "
        r"Guided cases use chordal graph costs and iterative $N=10$ terminal selection; "
        r"unguided cases use a fixed nearest eligible terminal configuration. "
        r"In the complete table, bold marks the highest success rate per configuration, including ties. "
        r"Ranks are selected by success, then lower median rollout RMSE, then lower rank. "
        r"Selection and reporting use the same exploratory seeds, not a held-out test set.")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
                  r"\caption{" + caption + "}", r"\label{tab:reacher-rank-k64}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def build(root=study.RESULT_ROOT):
    """Compatibility entry point for the consolidated saved-results report."""
    if Path(root).resolve() != study.RESULT_ROOT.resolve():
        raise ValueError("The consolidated report uses the saved canonical result roots")
    from atlas_deepc.notebook_tools.build_combined_rank_notebook import build as combined
    return combined()
