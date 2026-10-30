# Reacher Experiments

- [Full-data Atlas comparison](comparisons/atlas_deepc.ipynb): p=8 without graph
  guidance, K=64, rank 53, 300 controlled steps, seeds 211--310. Saves full
  trajectories, predicted horizons, coefficients, slack, selection residuals,
  geometry, realized costs, and timings. Checks initial-state/target pairing and
  fixed-reference agreement with the saved Select-DPC/contextual-DPC studies.
  This is a development comparison, not held-out performance. Timing can be
  affected by the concurrent 500-seed evaluation.
- [Main baseline](baselines/baseline_graph_routed_atlas_deepc.ipynb): development-selected
  evaluation on 500 seeds (10001--10500) for each of four methods, 2,000 rollouts.
  Uses K=64/96/80/64 unstandardized charts, projection selection, chordal guidance,
  Clarabel, and the simplified objective. Ranks 53/57/64/62 were selected on development
  seeds 211--310 for p=8 no graph/graph and p=10 no graph/graph, respectively.
  Runs 300 controlled steps after 12 warmup steps, in a separate mixed-K result
  folder. These 500 seeds were previously evaluated with K=48, so this is a
  repeated evaluation rather than a new untouched test set.
  Set RUN_EXPERIMENTS=True to run/resume serially; wait for other heavy sweeps
  to finish for comparable timings. Includes optional chart clouds and replays,
  failed-seed lists, median rollout RMSE, and a final LaTeX table.
- [Historical primitive baseline](baselines/baseline_reacher_atlas_deepc.ipynb): eight outputs,
  fixed nearest terminal posture, no graph waypoints.
- [Fixed terminal + graph, ten outputs](baselines/baseline_reacher_fixed_terminal_graph_output10.ipynb).
- [Historical 1,000-seed baseline](../../archive/notebooks/baseline_graph_routed_atlas_deepc_before_projection_chordal_20261012.ipynb).
- [Output/reference ablations](baselines/baseline_atlas_deepc_no_standardization.ipynb).
- [Centroid versus chordal graph costs](ablations/ablation_chordal_graph_cost.ipynb):
  paired graph-routing experiments with eight and ten outputs and fixed terminal goals.
- [Input-smoothing ablation](ablations/ablation_input_smoothing_four_baselines.ipynb):
  the current four baselines with only lambda_du changed from 5 to 0; 400 rollouts
  on seeds 211--310, saved separately with per-module timing and paired outcomes.
- [Effect of graph guidance](ablations/effect_of_graph_guidance.ipynb):
  paired 500-seed comparison at p=8, K=64, rank=53 and T_sim=300. Reuses the
  existing unguided baseline and runs only the guided policy on seeds
  10001--10500. Set RUN_MATCHED_GUIDANCE_500=True to run or resume; reports
  success rates, repaired/regressed seeds, and the exact paired McNemar test.
- [Combined rank ablation](ablations/ablation_linear_subspace_rank.ipynb):
  K=48, 64, 80, 96; every rank 28--70 for p=8/10 without/with guidance,
  on development seeds 211--310. One saved-results-only report with combined
  plots, complete success tables, selected ranks, tracking metrics, serial timings,
  and provenance. Replaces the separate K=64 and K=80/96 notebooks; Run All does
  not launch experiments or modify saved results.
- [Chart-count and rank ablation](ablations/ablation_chart_count_and_rank.ipynb):
  ten outputs with graph guidance; K=1, 2, 4, 8, 16, 24, 32, 48, 64, with a
  coarse rank search and local refinement on seeds 211--310. Uses the current
  projection/chordal baseline with lambda_g=lambda_du=0. K=1 is the global-chart
  control without inter-chart waypoints. The notebook explicitly marks partial
  results until both the search and selected-configuration serial timing finish.
- [Primitive baseline regularization, seed 252](ablations/ablation_primitive_lambda_g_seed252.ipynb):
  coefficient-penalty sweep with the saved eight-output atlas and no graph waypoints.
- [Contextual DPC, raw trajectories](comparisons/contextual_dpc_raw_final_reference.ipynb):
  equation-based arXiv v2 adaptation; 200 columns, no SVD, T_ini=2/T_f=10,
  Clarabel. The top section prepares seeds 10001--10500 with the exact saved
  p=8 unguided Atlas terminal references and frozen development-tuned weights.
  Set RUN_EVALUATION_500=True to launch locally; runs are serial and resumable.
  Historical 211--310 XY-only and eight-output results remain separately available
  below, with their launch switches disabled. No SVD or Atlas simulation is added.
- [Original Select-DPC](comparisons/select_dpc_original_final_reference.ipynb):
  seeds 10001--10500, 320 steps, original 200-column/five-refinement controller,
  original data and final reference, Clarabel. Set RUN_EXPERIMENT=True to launch.
  A fresh evaluation directory preserves the 211--310 development results.
  Run the two methods sequentially for comparable timing measurements.
- [Saved three-method comparison, p=8 without graph guidance](comparisons/comparison_saved_p8_no_graph.ipynb):
  rank-53 Atlas, original Select-DPC, and tuned contextual DPC on all 100 seeds.
  Read-only result aggregation with a paper-style table and modular timing;
  Run All regenerates reports without launching simulations. Uses serial timings,
  verifies matched Cartesian targets, and discloses method-specific protocol differences.
- [Multi-view-clustering BDPC](comparisons/mvc_bdpc_reacher.ipynb):
  Han et al. ACC 2024 equation-based adaptation, using the baseline offline data
  and user-selected fingertip x/y outputs. Joint NMF of Euclidean/angle views,
  representative-trajectory models, and one-step center-relative control.
  Numerical conventions and untuned Reacher parameters are explicit; all runs
  are disabled by default. This is not a p=8 controlled ablation or author code.
- [Gain-scheduled DeePC](comparisons/gain_scheduled_deepc_reacher.ipynb):
  Zieglmeier et al. arXiv:2512.02797v2 composite-region implementation, checked
  against the public MATLAB code. Same baseline offline data, raw XY trajectories,
  second-joint-angle scheduling, 200 columns per region / 400 per composite,
  matched T_ini=2/N=10 (paper N=5) and untuned weights, including the upstream terminal equality.
  Clarabel, modular timings and resumable seeds 211--310; all runs disabled by default.
  This is a disclosed Reacher port, not an authors' Reacher benchmark.
- [Other ablations](ablations/) and [algorithm comparisons](comparisons/).

## Related Files

- Original offline data: [data/reacher/](../../data/reacher/).
- Saved results, atlases, plots and videos: [experiments/reacher/](../../experiments/reacher/).
- Shared implementation: [code map](../../src/README.md).

The main baseline separates development rank selection from repeated evaluation.
No test rollouts execute by default. The previous main notebook is preserved in
`archive/notebooks/baseline_before_heldout500.ipynb`. Historical notebooks retain
their original protocols and must not be relabeled as held-out evaluations.
The previous K=48 500-seed notebook is also archived as
`archive/notebooks/baseline_heldout500_k48_before_mixed_k.ipynb`.

## Storage Cleanup

On October 13, 2026, superseded Reacher raw rollout arrays, caches, and generated
videos were removed, along with the October 6 handoff ZIP. Historical notebooks,
source snapshots, summaries, tables, and static figures remain, but their old
replay and raw-data analysis sections may require regenerating the removed data.

The current four-baseline studies, the complete rank 28--70 sweep and its serial
timing reruns, regularization ablations, frozen offline data, required source
atlases, website assets, and F1Tenth and pendulum results were preserved.
The exact deletion inventory and preservation checks are in
[`archive/cleanup_20261013`](../../archive/cleanup_20261013/).

On October 17, notebook-orphaned result arrays and videos were removed after
checking active notebook imports and saved-result dependencies. Summaries,
tables, static figures, and frozen source remain. This includes raw traces from
the standalone fingertip-reference and zero-initial-history-penalty studies;
their reports remain available, but raw-trace analyses require rerunning them.
The K=48 and K=64 rank sweeps, the 500-seed held-out baseline, notebook-linked
comparison/ablation data, and all original offline datasets were preserved.
The file-level deletion inventory is in
[`archive/cleanup_20261017`](../../archive/cleanup_20261017/).

On October 18, the superseded K=48 500-seed raw rollout arrays, derived geometry,
and videos were removed at the user's request. Its compact metrics, summaries,
tables, atlases, and frozen source remain. Development sweeps and offline data
were not removed. The cleanup inventory is stored alongside the retained K=48
results as `raw_data_cleanup.json`.

Later on October 18, the user requested removal of the older experiment data
instead of converting it to 300 steps. This removed 48,533 inactive 320-step
Reacher rollout files (9.85 GB), including historical rank-sweep traces.
All 300-step traces, the running mixed-K evaluation, offline datasets, atlases,
configurations, frozen source, compact metrics, and summary tables were retained.
The historical summaries still describe 320 steps; they were not relabeled or
recomputed at 300 steps. Historical raw-trace analysis and replay cells now
require regeneration of those traces. The exact deletion plan and verification
are in `archive/cleanup_20261018/old_320_rollouts_plan.json` and
`archive/cleanup_20261018/old_320_rollouts_outcome.json`.
