# Notebook Index

Notebooks are grouped by purpose. Saved results, figures, and videos are retained.
Open a notebook from its folder or from the project root: its setup locates the
project root automatically. Data and videos remain in `data/` and `experiments/`.
Moving a notebook does not change its controller, seeds, or run switches.

## Paper Motivation

- [Input–output trajectory surface](motivation/input_output_trajectory_surface.ipynb):
  current paper example; each point is a full length-two input–output window
  in R⁴, visualized isometrically in its containing three-dimensional subspace.
  Includes a curved surface, full-space local affine diagnostics,
  Times New Roman publication figures, and introduction text.
- [Nonlinear cylinder](motivation/nonlinear_cylinder.ipynb): an exact curved output
  surface in three dimensions, learned local affine fits, observed-transition
  routing, and reference-conditioned trajectory selection. Includes executed
  results and PNG/PDF figures; this is an earlier output-space illustration,
  not a visualization of full input–output windows or a DeePC benchmark.
- Keep future small motivation examples in [motivation/](motivation/).

## Baselines

- [Current Reacher Atlas-DeePC](baselines/baseline_graph_routed_atlas_deepc.ipynb):
  OSQP, lambda_g=0, linear-subspace charts, iterative N=10 terminal references;
  paired 1,000-seed comparison with/without intermediate graph waypoints,
  per-module timing, physical chart plots, and paired videos. Both variants
  retain graph-route-cost terminal-candidate selection. Set RUN_EVALUATION=True
  for fresh single-worker local timings with warm-up and a new output folder;
  otherwise it displays the saved OSQP experiment, not new local measurements.
- [Contextual Atlas-DeePC](baselines/baseline_contextual_atlas_deepc.ipynb):
  contextual-atlas deployment, diagnostics, and videos.

## Comparisons

- [Three-method comparison](comparisons/comparison_three_methods.ipynb):
  independently tuned Select-DPC, contextual DPC, and current Atlas-DeePC.
- [Earlier multi-method comparison](comparisons/comparison.ipynb):
  retained comparison results, including subspace and iterative-reference variants.

## Ablations

- [Gaussian measurement noise](ablations/ablation_measurement_noise.ipynb):
  baseline with lambda_g=0 and noisy offline/online outputs at five levels,
  100 seeds each; error distributions, near-target traces, paired noisy-rollout
  videos, and computation times.
- [Position-only tracking cost](ablations/ablation_position_only_tracking_cost.ipynb):
  zero non-position Q weights, retaining augmented outputs and graph routing;
  paired comparison over 100 seeds at each of five noise levels, with lambda_g=0.
- [Joint-sum output channels](ablations/ablation_joint_sum_output_channels.ipynb):
  current-baseline 10-output versus 8-output comparison; removes the two derived
  joint-sum channels throughout the pipeline, 100 paired noise-free seeds.
- [Component ablations](ablations/ablation_atlas_deepc_components.ipynb):
  consolidated 100-seed results and paired videos with/without waypoints.
- [Chart geometry and graph routing](ablations/ablation_affine_subspace_and_graph_routing.ipynb):
  current-baseline 100-seed geometry/waypoint study.
- [Terminal-reference candidates](ablations/ablation_candidate_goal_routing.ipynb).
- [Clustering methods](ablations/ablation_clustering_methods_500.ipynb):
  selected spectral clustering and UMAP + K-means comparison on 100 exploratory seeds211–310, using
  waypoint references, lambda_g=0.05, Clarabel, 48 charts and rank60.
  Reuses 200 completed rollouts from the original serial study: spectral90%, UMAP92%.
  All64,000 selected-method QPs optimal. Includes timing, paired errors, chart health,
  and separate 48-chart physical arm-pose grids with bold medoids (PNG/PDF).
  Both the historical 500-seed notebook and complete seven-method report are archived;
  see [protocol and status](../docs/clustering_methods_100_status_2026-09-16.md).
- [Contextual clustering](ablations/ablation_contextual_cluster_atlas.ipynb).
- [Linear-subspace rank — lambda_g=0.05, Clarabel](ablations/ablation_linear_subspace_rank.ipynb):
  primary rank-ablation notebook, restored as the selected version on September 15.
  Completed 6,100-rollout sweep: every integer rank 20–80 on the same 100 seeds,
  with executed success, error, paired-outcome, and timing figures.
- [Additional ranks 58, 61, 63 — 1,000 seeds](ablations/ablation_adjacent_ranks_1000.ipynb):
  completed 3,000 rollouts on seeds 311–1310, single worker, lambda_g=.05 and Clarabel.
  Success rates 95.3%/94.2%/95.7%; all 960,000 QPs optimal. Fixed offline data/charts,
  targets paired with ranks 59/62/64, four executed cells and three inspected figures.
  Timings come from a new shared-computer session; [results and audit](../docs/adjacent_rank_1000_status_2026-09-17.md).
- [Selected ranks over 1,000 seeds](ablations/ablation_selected_ranks_1000.ipynb):
  completed 3,000 rollouts with lambda_g=0.05 and Clarabel on paired seeds 311–1310.
  Ranks 59/62/64 reach 96.5%/95.9%/95.2% success, with mean online time
  30.68/31.45/32.30 ms; all 960,000 QPs optimal. Fixed offline data/charts,
  serial rotating rank order, executed report and three inspected figures.
  See [results and protocol](../docs/selected_rank_1000_status_2026-09-15.md).
- [Linear-subspace rank with lambda_g=0](ablations/ablation_linear_subspace_rank_lamg0.ipynb):
  interrupted Clarabel repeat; partial results preserved after the OSQP update.
  This notebook is unexecuted and must not be presented as a completed study.
- [Linear-subspace rank with lambda_g=0 and OSQP](ablations/ablation_linear_subspace_rank_lamg0_osqp.ipynb):
  preserved alternative study; completed 6,100-rollout repeat of every rank 20–80 on seeds 211–310, with six
  executed code cells and seven checked figures. Ranks 59/62 reach 97% success;
  rank 64 reaches 96%. Comparison with lambda_g=0.05 / Clarabel changes both
  penalty and solver; solver-status counts and paired outcomes are included.
- [Atlas control parameters](ablations/ablation_control_parameter.ipynb): local-run-ready
  lambda-g sweep using the Clarabel graph-waypoint baseline, fixed rank-60 linear
  charts and iterative N=10 references. Defaults to 100 seeds (211–310), 52 values
  including zero and 0.05; experiments disabled until explicitly enabled.
  Portable setup uses `requirements-control-parameter.txt`; old results are not reused.
- [Contextual control parameters](ablations/ablation_control_parameter_contextual.ipynb).

Older ablations retain their original experimental settings. Consult each notebook's
setup rather than assuming every study uses the current baseline.

## Other Robot Tasks

- [Simple robot tasks](robot_tasks/simple_robot_atlas_deepc.ipynb).
- [Pusher](robot_tasks/pusher_atlas_deepc.ipynb).

## Archives

Historical notebooks remain in [archive/notebooks](../archive/notebooks/).
The earlier [Clarabel/OSQP comparison](../archive/notebooks/baseline_clarabel_osqp_solver_comparison.ipynb)
retains both solvers' original results; the active baseline notebook uses OSQP only.
The superseded [contextual affine-rank study](../archive/notebooks/ablation_affine_rank_contextual.ipynb)
retains its original results and controller settings.
Comparison snapshots stored with experiment results remain there for provenance.
Notebook builders in `src/` now write to the corresponding folders above.
