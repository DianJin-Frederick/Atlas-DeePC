# Code Map

The `atlas_deepc` package contains Atlas-DeePC and shared Reacher/F1TENTH tasks.
Other controllers live in the sibling `comparison_methods` package.
Notebooks add `src/` to their import path. Experiment directories contain saved
artifacts, not importable controller code.

## Package Layout

| Directory | Responsibility |
|---|---|
| `control/` | DeePC optimization and dataset/simulator helpers |
| `atlas/` | Features, clustering, subspaces, chart selection and routing |
| `baselines/` | Atlas-DeePC controller and eight/ten-output rollout utilities |
| `benchmarks/` | Atlas-DeePC evaluation runners |
| `ablations/` | Rank, chart-count and matched-guidance studies |
| `common/` | Metrics, timing, data, paths, hashes and locking |
| `f1tenth/` | Vehicle data, references and evaluation |
| `reporting/` | Saved-result readers, tables, figures and replay |
| `notebook_tools/` | Rank-table and notebook generation |

## Reacher

The main notebook is
`notebooks/reacher/baselines/baseline_graph_routed_atlas_deepc.ipynb`.
It uses `benchmarks/reacher_mixed_k.py`, with shared execution in
`benchmarks/reacher_heldout.py`. The four frozen configurations are:

| Outputs | Guidance | Charts | Rank |
|---|---|---|---|
| 8 | No | 64 | 53 |
| 8 | Yes | 96 | 57 |
| 10 | No | 80 | 64 |
| 10 | Yes | 64 | 62 |

Evaluation uses seeds 10001--10500 and 300 controlled steps. The objective
contains tracking, input effort, and initial-output slack. Charts are selected
by projection residual; graph guidance uses chordal costs and iterative
terminal-reference selection.

Shared stage timing lives in `common/timing.py`; success, tracking error, and
rollout RMSE live in `common/metrics.py`. These helpers do not depend on a
historical experiment launcher. `reporting/four_case_tracking_table.py` only
formats tables.

`benchmarks/reacher_rank_panel.py` executes rank panels with explicit atlas,
configuration, and source-manifest arguments. `ablations/projection_chordal_rank.py`
and `ablations/k64_rank.py` retain the paper rank-study entry points. The merged
rank notebook reads existing results through `reporting/rank_sweep_k48_k64.py`.
`ablations/low_k_completion.py` runs the compact chart-count batches without
importing historical rank launchers.

The K=48 rank-60 artifacts now live inside that study's `rank060/` directory.
`common/paths.py::experiment_path` resolves their old recorded paths without
rewriting measured results or original provenance. `module_map.json` resolves
current source filenames when freezing new runs.

The matched-guidance notebook uses `ablations/matched_guidance_500.py`.
The comparison notebooks import `comparison_methods`, not Atlas controller
modules. Select-DPC remains an external dependency.

## Comparison Methods

| Directory under `comparison_methods/` | Responsibility |
|---|---|
| `contextual_dpc/` | Reacher and F1TENTH Contextual-DPC controllers and evaluations |
| `select_dpc/` | External Select-DPC installation checks and comparison adapters |
| `kde_bdpc/` | One-step KDE-BDPC paper formulation, routing, batches and result audit |
| `common/` | Shared episode decoding, reference recovery and multi-view clustering |

KDE-Atlas hybrids, standalone MVC-BDPC controllers, and the retired multistep
KDE launcher are removed. Joint-NMF helpers remain only because faithful
KDE-BDPC uses them; they are not a separate MVC-BDPC comparison.
The KDE multistep audit notebook only reads historical results; it does not
provide another runnable KDE controller. The faithful KDE-BDPC entry point is
`comparison_methods.kde_bdpc.paper`, with batch execution in
`comparison_methods.kde_bdpc.batches`.

Notebook source imports were updated without executing cells or changing saved
outputs. After this package move, restart notebook kernels. Existing source-hash
guards deliberately reject resuming old runs under modified code; use a fresh
result directory for a new run. Original summaries and SSD data are unchanged.

## Saved Results

See `experiments/README.md` for the retained study inventory. Many local artifact
paths are symlinks to the portable SSD. Do not edit or delete their targets as
part of a local-only cleanup.

Saved source hashes describe the original experiments. Cleanup does not change
those measurements or hashes. New runs with changed source must use a fresh
result directory; do not disable compatibility checks to resume an old cache.

## F1TENTH

The notebook is `notebooks/f1tenth/baselines/baseline_f1tenth_atlas_deepc.ipynb`.
The study uses matching offline/online noise ratios and rebuilds the atlas and
contextual library at each level. See `docs/f1tenth/README.md` for the protocol.
