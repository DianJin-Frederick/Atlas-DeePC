"""Build the lambda_g=0 OSQP report, preserving both Clarabel studies."""

from copy import deepcopy
import build_linear_subspace_rank_lamg0_notebook as previous

template = previous.template
VERSION = "linear_subspace_rank_lamg0_osqp_100_v1"
OUTPUT = template.ROOT / "notebooks/ablations/ablation_linear_subspace_rank_lamg0_osqp.ipynb"
cells = deepcopy(previous.cells)
for cell in cells:
    cell.source = cell.source.replace(previous.VERSION, VERSION)
    if cell.cell_type == "markdown" and cell.source.startswith("# Atlas-DeePC:"):
        cell.source = cell.source.replace("Rank, λg = 0", "Rank, λg = 0, OSQP", 1)
        start = cell.source.index("This is a fresh repeat")
        end = cell.source.index("## Fixed setup")
        cell.source = cell.source[:start] + r"""This fresh sweep uses **$\lambda_g=0$ and OSQP**. Within this sweep only
rank varies. The previous complete study used $\lambda_g=0.05$ and Clarabel,
so the comparison below changes **both regularization and solver** and cannot
attribute differences to either alone. The interrupted $\lambda_g=0$ Clarabel
sweep remains separate; its partial checkpoints are not used as OSQP results.

OSQP 1.0.4 through CVXPY 1.6.5 uses `max_iter=10000`, `eps_abs=1e-5`,
`eps_rel=1e-5`, and polishing enabled (the installed CVXPY defaults made explicit).
Other OSQP settings use package defaults. There is no fallback solver. Frozen
sources and hashes record the solver selection and options for every rollout.
All controller weights other than the requested coefficient penalty, chart geometry,
goal policy, offline data and evaluation seeds match the completed study.

""" + cell.source[end:]
    if cell.cell_type == "markdown" and cell.source.startswith("## Paired comparison:"):
        cell.source = """## Paired comparison: λg = 0 / OSQP versus λg = 0.05 / Clarabel

Both sweeps use the same rank/seed grid, targets, frozen atlas and control setup.
Regularization and solver both change. A repaired seed fails under the old setup
and succeeds under the new setup at the same rank; a regressed seed does the
reverse. Both methods retain the full-duration strict-success test and zero-input
QP-failure handling. These are reused exploratory seeds.

This comparison cannot isolate the effect of lambda_g from the solver. Timings
also reflect runs at different times on a shared computer.
"""
    if cell.cell_type == "markdown":
        cell.source = cell.source.replace(
            f"python experiments/routed_deepc_iid/{VERSION}/frozen_protocol/src/linear_subspace_rank_ablation.py --workers 6",
            "python src/linear_subspace_rank_lamg0_osqp_ablation.py --workers 6")
    if cell.cell_type == "code":
        cell.source = cell.source.replace(
            'assert payload["signature"]["base_config"]["lam_g"] == 0.0',
            'assert payload["signature"]["base_config"]["lam_g"] == 0.0\nassert payload["signature"]["qp_solver"] == "OSQP"')
        cell.source = cell.source.replace('report = {"lam_g": 0.0,', 'report = {"qp_solver": "OSQP", "lam_g": 0.0,')
        if 'report = {"qp_solver":' in cell.source:
            cell.source = cell.source.replace(
                'experiment.write_json(RESULT_ROOT / "notebook_report.json", report)',
                '''status_by_rank = pd.DataFrame([
    {"rank": rank, **pd.DataFrame([r["statuses"] for r in panels[rank]["results"]]).fillna(0).sum().astype(int).to_dict()}
    for rank in RANKS]).set_index("rank").fillna(0).astype(int)
assert (status_by_rank.sum(axis=1) == 32000).all()
status_totals = status_by_rank.sum().astype(int).to_dict()
assert sum(status_totals.values()) == 6100 * 320
assert sum(v for k, v in status_totals.items() if k not in ("optimal", "optimal_inaccurate")) == int(summary.unsolved_qps.sum())
report["solver_status_totals"] = status_totals
report["solver_status_by_rank"] = status_by_rank.reset_index().to_dict(orient="records")
display(pd.Series(status_totals, name="QP steps by returned status"))
display(HTML("<details><summary>Solver-status counts for all 61 ranks</summary>" + status_by_rank.to_html() + "</details>"))
experiment.write_json(RESULT_ROOT / "notebook_report.json", report)''')
        if cell.source.startswith('OLD_ROOT ='):
            cell.source = cell.source.replace('assert new_row.lam_g == 0.0',
                'assert new_row.lam_g == 0.0 and new_row.qp_solver == "OSQP"')
            cell.source = cell.source.replace('comparison_rows = []',
                'assert "cp.CLARABEL," in (OLD_ROOT / "frozen_protocol/src/routed_deepc_experiment.py").read_text()\ncomparison_rows = []')
            cell.source = cell.source.replace('"λg = 0"', '"λg = 0 / OSQP"')
            cell.source = cell.source.replace('"λg = 0.05"', '"λg = 0.05 / Clarabel"')
            cell.source = cell.source.replace('Success change: λg = 0 minus 0.05', 'Success change: new minus old setup')
            cell.source = cell.source.replace('rank_lambda_comparison.png', 'rank_setup_comparison.png')
            cell.source = cell.source.replace('paired regularization comparisons', 'paired setup comparisons')
            cell.source = cell.source.replace('lambda_comparison_report.json', 'setup_comparison_report.json')
            cell.source = cell.source.replace('"configuration_differences": config_changes,',
                '"configuration_differences": config_changes, "solver_change": ["CLARABEL", "OSQP"],')
            cell.source = cell.source.replace('only lambda_g changed.', 'lambda_g and solver changed.')

if __name__ == "__main__":
    template.build(OUTPUT, cells)
