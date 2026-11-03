"""Present the current four-method rank sweep without overwriting old results."""

from pathlib import Path

from atlas_deepc.ablations import projection_chordal_rank as study
from atlas_deepc.reporting.four_case_tracking_table import format_table, standalone


def write_table(root, result):
    root = Path(root)
    ranks = result["ranks_tested"]
    caption = (rf"Best observed ranks among {len(ranks)} tested ranks ({min(ranks)}--{max(ranks)}) "
        r"for four Atlas-DeePC baselines on seeds 211--310. Each method's rank maximizes "
        r"strict success; ties are broken by lower median rollout RMSE, then lower rank. "
        r"Rank selection and performance reporting use the same exploratory 100 seeds, "
        r"not an independent test set. All cases use $\lambda_g=\lambda_{\Delta u}=0$, "
        r"$\lambda_{\mathrm{ini}}=10^5$, Clarabel, 48 unstandardized linear charts, and "
        r"full-window projection-residual chart selection. Unguided cases use a fixed nearest "
        r"eligible terminal configuration; guided cases use rank-dependent chordal graph costs "
        r"and iterative $N=10$ terminal selection. Median and percentile errors summarize "
        r"the final-100 mean fingertip error across seeds. Median rollout RMSE is the median "
        r"of individual rollout RMSEs over all 320 controlled steps, excluding warmup. "
        r"Timings are from separate serial, single-thread reruns of the selected ranks on "
        r"all 100 seeds, excluding the first five controlled steps per rollout. "
        r"Bold indicates the best observed control-performance value.")
    table = format_table(result["columns"], caption, "tab:four-baselines-selected-ranks", include_rank=True)
    (root / "table_best_ranks.tex").write_text(table)
    (root / "table_best_ranks_preview.tex").write_text(standalone(table))


def build(root=study.RESULT_ROOT):
    """Compatibility entry point for the consolidated saved-results report."""
    if Path(root).resolve() != study.RESULT_ROOT.resolve():
        raise ValueError("The consolidated report uses the saved canonical result roots")
    from atlas_deepc.notebook_tools.build_combined_rank_notebook import build as combined
    return combined()


if __name__ == "__main__":
    build()
