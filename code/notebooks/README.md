# Notebook Index

These are the active notebooks retained with their code and saved-result
dependencies. Notebooks locate the project root from their folder. Each study
keeps its own settings; check its setup before enabling experiment switches.

## Reacher

All Reacher notebooks are in [reacher/](reacher/README.md), grouped into baselines,
comparisons and ablations. Saved results are under `experiments/reacher/`.

### Baselines

- [Main Reacher Atlas-DeePC](reacher/baselines/baseline_graph_routed_atlas_deepc.ipynb):
  projection-residual selection on the full 12-sample measured input/output window;
  chordal graph costs; rank 60, 48 raw-configuration charts, lambda_g=0, Clarabel.
  Four 100-seed panels (211-310), with 77%/83% success for 8 outputs without/with
  graph guidance and 75%/92% for 10 outputs. No-graph uses one fixed nearest goal;
  graph uses iterative N=10 goal selection and waypoints. Twelve zero-input warmup
  steps, no coefficient bounds or affine constraint. Current chart clouds, verified
  simulator videos, slowed dashboards and per-module timings. Run/render switches
  are off by default. Earlier results below remain historical studies.

- [Reacher fixed terminal with graph routing](reacher/baselines/baseline_reacher_fixed_terminal_graph_output10.ipynb):
  10 outputs, one fixed nearest eligible terminal configuration, and graph waypoints.
  The route can change but the goal sample/chart cannot. Raw K-means, 48 charts,
  rank 60, Clarabel, lambda_g=0.05, no coefficient bounds. Separate 100-seed
  panel (211-310), complete-trajectory clouds, module timings and 0.25x dashboards.

- [Primitive Reacher Atlas-DeePC](reacher/baselines/baseline_reacher_atlas_deepc.ipynb):
  standalone 8-output baseline, without standardization, graph routing, waypoints,
  projection selection, or coefficient boxes. Uses nearest-centroid selection,
  48 rank-60 linear-subspace charts and a fixed nearest eligible terminal reference.
  Reports the completed 100-seed panel (211-310), audits saved chart choices,
  displays full past/future trajectory configuration clouds with a consistent
  blue/orange palette, module timings, simulator videos and 0.25x dashboards.
  Original v6 results remain unchanged; a separate switch runs current-code results.

- [Reacher output channels and reference generation](reacher/baselines/baseline_atlas_deepc_no_standardization.ipynb):
  merged unstandardized study: 8/10 outputs, graph-guided versus fixed-nearest control.
  Same offline episodes and seeds 211-310; 48 charts, rank 60, Clarabel,
  lambda_g=0.05, no empirical coefficient box. Graph-guided uses iterative N=10
  route-cost selection and waypoints; no-graph uses one fixed nearest eligible
  configuration, without route-cost selection or waypoints. The v7 result roots
  preserve earlier experiments. Section 6 compares nearest-centroid localization
  with full 12-step input/output projection on a separately matched-warmup panel.
  Both reference modes share one atlas per output layout. Includes both chart
  clouds, paired success/error comparisons, median/p99 module timings.
  The user's experiment execution switch remains enabled. Predecessor notebooks are archived.
- [Historical 1,000-seed Reacher Atlas-DeePC](../archive/notebooks/baseline_graph_routed_atlas_deepc_before_projection_chordal_20261012.ipynb):
  Clarabel, lambda_g=0.05, rank-60 linear-subspace charts, 48 physical charts,
  iterative N=10 terminal references. Saved paired with/without intermediate
  waypoint results on seeds 311-1310, chart clouds, videos, and module timings.
  Both variants retain graph-route-cost terminal selection.
### Comparisons

- [Full-data Atlas-DeePC](reacher/comparisons/atlas_deepc.ipynb): p=8, no graph
  guidance, K=64 and rank 53; 100 development seeds (211--310), 300 steps.
  Retains measured trajectories, full predictions, coefficients, slack, chart
  geometry and timings. Includes paired physical metrics and protocol caveats
  for the saved Select-DPC and contextual-DPC studies.
- [Three-method comparison](reacher/comparisons/comparison_three_methods.ipynb):
  independently tuned Select-DPC, contextual DPC, and Atlas-DeePC; shows available
  saved results and completeness, with separate tuning and final-test switches.
- [Earlier multi-method comparison](reacher/comparisons/comparison.ipynb):
  retained contextual, physical-chart, linear-subspace, and iterative-reference
  comparisons, including original controlled timing measurements.

### Ablations

- [Position-only tracking cost](reacher/ablations/ablation_position_only_tracking_cost.ipynb):
  historical rank-60, lambda_g=0 paired study at five noise levels; retains the
  corresponding full-configuration-Q data as its comparison.
- [Joint-sum output channels](reacher/ablations/ablation_joint_sum_output_channels.ipynb):
  10-output versus 8-output comparison on 100 noise-free seeds.
- [Component ablations](reacher/ablations/ablation_atlas_deepc_components.ipynb):
  100-seed geometry, coefficient-sum, nearest-reference, and waypoint comparisons,
  including paired success and failure videos.
- [Dense linear-rank sweep](reacher/ablations/ablation_linear_subspace_rank.ipynb):
  every integer rank 20-80 on 100 seeds; lambda_g=0.05 and Clarabel.
- [Merged 1,000-seed rank study](reacher/ablations/ablation_selected_ranks_1000.ipynb):
  ranks 58/59/61/62/63/64 on seeds 311-1310, with paired results and original
  frozen protocols. Timing sessions are identified separately.
- [Adjacent-rank source report](reacher/ablations/ablation_adjacent_ranks_1000.ipynb):
  ranks 58/61/63; also included in the merged report above.
- [Control parameters](reacher/ablations/ablation_control_parameter.ipynb):
  Clarabel, rank-60, iterative-N=10 lambda-g sweep on 100 seeds; displays saved
  completeness and has a resumable experiment switch.

## F1TENTH

The [F1TENTH notebook](f1tenth/baselines/baseline_f1tenth_atlas_deepc.ipynb)
retains Atlas-DeePC K=2, rank 40 and the contextual-DPC 30-column comparison.
Both use the same 300-second offline dataset and 20 evaluation seeds. Saved
results and figures are in `experiments/f1tenth/paper/`. Run All displays them;
set `RUN_ROLLOUTS=True` to rerun the fixed configurations separately. Old
baselines, datasets and tuning studies have been removed. See the
[protocol](../docs/f1tenth/README.md) for details.

## Motivation

- [Autonomous K-means and linear subspaces](motivation/autonomous_behavioral_atlas.ipynb):
  clustering complete length-three trajectories, saved rank-two linear bases,
  and visualizations of the fitted planes against the nonlinear surface.
- [Autonomous trajectory surface](motivation/autonomous_trajectory_surface.ipynb):
  scalar-output autonomous dynamics whose complete length-three trajectories
  form an exact curved surface in three dimensions.
- [Input-output trajectory surface](motivation/input_output_trajectory_surface.ipynb):
  full length-two input-output windows, local affine diagnostics, and paper figures.
- [Nonlinear cylinder](motivation/nonlinear_cylinder.ipynb):
  an earlier output-space illustration of charts, routing, and contextual selection.

## Archives and Maintenance

`archive/notebooks/` contains historical snapshots with saved outputs. They are
not part of the supported rerun set after removal of obsolete experiment data.
The September 24 cleanup removed the other robot tasks and the root-level
Pusher notebook. F1TENTH was subsequently added through the shared-controller
adapter; the other removed task implementations remain excluded.

Obsolete cleanup and merged-rank reports have been removed. The historical
[migration map](../docs/task_organization_2026-10-03/moves.json) remains for
regression-test compatibility, not as a list of currently retained files.
Compatibility links are retained only where needed; deleted studies are not
restored to keep archived paths valid. Current notebooks use the task folders.
