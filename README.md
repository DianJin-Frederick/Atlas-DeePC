# Atlas DeePC Reacher

This project explores Atlas / charted subspace DeePC controllers on Gymnasium
MuJoCo tasks. The primary `Reacher-v5` result is a graph-routed, fixed-chart
DeePC controller; the repository also contains integrated `Pusher-v5` and
`Swimmer-v5` studies. Each controller predicts with combinations of offline
behavioral columns and fits no state-space dynamics model.

## Start Here

Use the [notebook index](notebooks/README.md) to browse the notebooks:

| Folder | Contents |
|---|---|
| [motivation](notebooks/motivation/) | Small illustrative systems and figures for the paper introduction |
| [baselines](notebooks/baselines/) | Reacher Atlas-DeePC and contextual Atlas-DeePC |
| [comparisons](notebooks/comparisons/) | Cross-method comparisons |
| [ablations](notebooks/ablations/) | Component, routing, clustering, rank, and parameter studies |
| [robot_tasks](notebooks/robot_tasks/) | Pusher and simple robot tasks |

The main entry points are the
[current baseline](notebooks/baselines/baseline_graph_routed_atlas_deepc.ipynb)
and the [component ablations with paired videos](notebooks/ablations/ablation_atlas_deepc_components.ipynb).
Notebooks locate the project root automatically; all existing outputs and videos
are preserved. No experiment settings were changed during this reorganization.

## Research Guide

The notes below also describe historical experiments; some older notebook entry
points are no longer in the active collection. Use the index above for files
currently present.

- `notebooks/baselines/baseline_graph_routed_atlas_deepc.ipynb` is the current Reacher baseline:
  48 cached rank-60 **uncentered linear-subspace** charts plus iterative
  minimum-route-cost terminal-reference augmentation with **N=10** candidates.
  Use `src/baseline_atlas_deepc.py` (`BaselineAtlasDeePC` or `run_panel`) for all
  new Reacher baseline runs. Results are versioned under
  `experiments/routed_deepc_iid/baseline_linear_iterative_n10_v1/`.
  Its diagnostic panel is not a large-scale success-rate estimate.
- The low-level `src/routed_deepc_experiment.py` retains historical defaults for
  existing ablations and contextual experiments. Calling it directly without
  the new baseline wrapper does **not** select the current combined baseline.
  Comparisons, contextual controllers, and other robotic tasks are not silently
  migrated. The previous baseline notebook and videos remain archived in
  `archive/notebooks/baseline_affine_fixed_reference_before_linear_iterative.ipynb`.
- `notebooks/robot_tasks/simple_robot_atlas_deepc.ipynb` is the executed cross-task transfer study on
  dm-control PointMass, Cartpole Balance, Pendulum Swingup, Finger Turn,
  Ball-in-Cup, and Cartpole Swingup. It compares Atlas-DeePC with contextual
  Atlas-DeePC under task-specific random-data profiles and reports both solved
  transfers and unresolved contact/energy-acquisition stress tests.
- `pure_contextual_atlas_deepc.ipynb` is the executed, trigger-free contextual
  Atlas result. Its chart-local future-mode factorization uses a contextual
  basis at every control step and reaches 463/500 strict successes, versus
  443/500 for the matched physical-chart Atlas baseline. It performs no online
  Hankel-column scan or SVD rebuild.
- `deepc_four_algorithm_comparison.ipynb` is the clean, runnable five-method
  comparison of Atlas, a Select-DPC-style selector, Contextual Sampling with
  graph or final reference, and the causal Atlas/contextual rescue hybrid from
  `ablation_graph_routing.ipynb`. Every selected local dataset uses a centered
  rank-60 SVD representation; online selector timings include rebuilding it,
  while Atlas chart bases are offline.
- `deepc_algorithm_comparison.ipynb` is the earlier executed three-way comparison of
  Atlas chart selection, a rank-matched Select-DPC trajectory selector, and
  Contextual Sampling under the same graph-routed waypoint reference and DeePC
  formulation, including their paired success rates on 500 held-out seeds
  (111–610).
- `edge_conditioned_atlas_ablation.ipynb` is the executed 100-seed paired
  ablation of cached graph-edge / terminal-reference-conditioned Atlas bases
  against the frozen current-chart Atlas baseline.
- `ablation_graph_routing.ipynb` is the paired development/evaluation study of
  causal contextual rescue: the frozen atlas remains active unless terminal
  tracking stalls, after which controller-Q-aware online selection supplies the
  rank-60 DeePC matrix.
- `notebooks/ablations/ablation_clustering_methods_500.ipynb` is the resumable paired clustering
  ablation on seeds 111–610. It compares K-means with spherical K-means, a
  diagonal GMM, Ward agglomeration, k-NN spectral clustering, affine
  K-subspaces, landmark SSC/LRR, CLARA K-medoids, and fixed-K-adapted HDBSCAN.
- `notebooks/robot_tasks/pusher_atlas_deepc.ipynb` is the executed, coverage-validated `Pusher-v5`
  study. It loads the retained 16-seed results and coverage audit from
  `experiments/pusher_atlas/coverage_300_routed/`. A subsequent disjoint
  100-seed audit of the current controller records 90/100 sustained successes
  under `experiments/pusher_atlas/heldout201_300/`. The paired strict fixed-atlas
  ablation, with all teacher-derived online guidance removed, records 0/100 under
  `experiments/pusher_atlas/heldout201_300_pure_atlas/`.
- `src/routed_deepc_experiment.py` is the shared historical rollout engine.
- `src/edge_conditioned_atlas_experiment.py` runs the edge-conditioned Atlas
  ablation on held-out seeds 111 through 210.
- `src/q_weighted_routing_experiment.py` assigns controller-aware graph costs
  from transition-supported future-output blocks and runs the paired ablation.
- `src/contextual_rescue_atlas_experiment.py` runs the causal stagnation-rescue
  experiment and records selection, basis-construction, solve, and total timing.
- `src/chart_local_contextual_atlas_experiment.py` runs the pure always-on
  chart-local contextual atlas. Each physical localization region owns eight
  cached Q-weighted future modes; online control scores only those eight cached
  prototypes and always supplies the selected contextual rank-60 basis to DeePC.
- `src/query_amortized_contextual_atlas.py` distills causal contextual-aware DPC
  queries into a trigger-free cached atlas. The frozen 1,024-medoid version scores
  472/500 (94.4%) on disjoint seeds 711–1210, versus 440/500 physical Atlas and
  483/500 contextual-aware DPC. The paired gap to contextual-aware DPC is not
  significant at 0.05 (exact p=0.0614), while median online compute falls from
  76.8 ms to 51.8 ms because no Hankel scan or SVD occurs online.
- `query_distilled_contextual_atlas_deepc.ipynb` is the executed construction,
  purity audit, held-out comparison, timing report, and paired analysis.
- `src/deepc_algorithm_comparison_heldout100.py` runs the resumable paired
  five-method Atlas / Select-DPC / two-reference Contextual Sampling / causal
  Atlas-contextual rescue panel.
- `src/pusher_atlas/` contains the integrated Pusher data collection, coverage
  audit, behavioral atlas, controller, evaluation, plotting, and video modules.
- `src/pusher_coverage_study.py` reproduces the full Pusher study;
  `src/pusher_atlas_experiment.py` provides its smaller general runner.
- `src/pusher_heldout_evaluation.py` runs and checkpoints the disjoint Pusher
  seed panel (201 through 300).
- `docs/PUSHER_ATLAS_DEEPC.md` documents the Pusher design and reproduction
  commands. The earlier lightweight shared-helper port remains in
  `src/pusher_routed_deepc_experiment.py` for comparison.
- `src/` contains all Python modules and utilities.
- `src/simple_robot_atlas_benchmark.py` contains the shared cross-task DeePC
  pipeline; `src/run_simple_robot_profiles.py` freezes the task-specific
  profiles used by `notebooks/robot_tasks/simple_robot_atlas_deepc.ipynb`.
- `experiments/routed_deepc_iid/clustering_jointconfig48_fresh211_310/`
  contains the untouched 100-seed evaluation of the selected physical-chart atlas.
- `archive/notebooks/` contains superseded exploratory notebooks.

## Historical Results

The rates below describe earlier affine/fixed-reference variants, not the current
linear-chart plus iterative-N=10 combination. Consult the current baseline
notebook for version-matched results; its larger comparison is maintained separately.

The earlier controller clusters each offline window by its latest six-dimensional
periodic joint configuration, builds 48 fixed charts, and retains a rank-60
behavioral basis inside every chart. After selecting this formulation on earlier
experiments, it succeeds on 90 of 100 untouched seeds (211 through 310). The 95%
Wilson confidence interval is 82.6% to 94.5%. All 32,000 DeePC optimizations solve
optimally. Goal construction enforces at least 6 cm separation between the
offline source episode's target and the online target.

The contextual-rescue extension in `ablation_graph_routing.ipynb` raises the
same untouched panel from 90/100 to 96/100 (95% Wilson interval 90.2% to 98.4%).
It repairs six failures with no regressions; exact paired McNemar p = 0.03125.
This extension performs online behavioral-column selection only after a causal
terminal-stagnation trigger and should be described as a hybrid Atlas/contextual
DeePC controller, not as a retrieval-free fixed-atlas result.

## Reproducible Experiments

- `src/affine_k_subspaces_random_walk.py` runs the random-walk offline-data, affine
  K-subspaces, per-cluster SVD, concatenated-atlas DeePC experiment. Its inputs,
  configuration, and outputs live under `experiments/affine_k_subspaces_random_walk/`.
- `src/local_affine_atlas_experiment.py` runs the current pure-DeePC candidate. It
  builds an adaptive affine chart from nearby IID-random trajectory windows and
  succeeds on seven of eight recorded seeds without expert data or feedback.
- `src/weighted_affine_atlas_experiment.py` tests convex simplex weighting of fixed
  affine K-subspace charts. It is retrieval-free, but its whole-trajectory chart
  blends have poor online prediction.
- `src/gated_chart_atlas_experiment.py` tests a fixed Voronoi atlas with one SVD
  trajectory model per chart. Hard chart confinement causes local equilibria.
- `src/stitched_atlas_mpc_experiment.py` composes fixed learned one-step affine
  charts across the control horizon. This is an MPC comparison baseline because
  it first fits local affine dynamics.
- `src/routed_atlas_experiment.py` is the graph-routed MPC comparison baseline. Its
  strong results motivated graph routing, but they are not DeePC results.
- `src/routed_deepc_experiment.py` keeps the graph router while replacing every
  model-based control solve with fixed-chart DeePC.
- `src/select_dpc_comparison.py` preserves Select-DPC's raw-L1 full-trajectory
  selection and iterative refinement while matching the Atlas Reacher data,
  horizons, goals, costs, constraints, regularization, solver, and success test.
- `src/atlas_select_shared_waypoint_comparison.py` performs the primary native-
  formulation comparison: both methods use the same reconstructed trajectories
  and physical waypoints, while retaining their own representations, horizons,
  objectives, constraints, and solvers.
- `src/select_dpc_native_data_graph_waypoints.py` replays the identical Atlas
  waypoint schedule using Select-DPC's supplied IID Reacher trajectories and
  original controller settings.
- `src/online_selected_deepc_comparison.py` is the controlled primary ablation:
  cached Atlas chart columns versus online top-200 L1 trajectory selection, with
  the rank-60 controller and all other graph-routed DeePC settings held fixed.

## Data

### Repository storage

The repository includes research code, notebooks with their saved outputs,
documentation, original offline inputs under `data/`, and experiment reports,
figures, and videos. Generated experiment arrays (`.npz` and `.npy`), fitted
`.joblib` caches, vendored experiment dependencies, and temporary files remain
local and are excluded by `.gitignore`. Existing notebooks may require those
local artifacts or regeneration using the relevant experiment scripts; cloning
the repository alone does not restore every experiment cache.

The website will be previewed locally during development. GitHub Pages publishing
is not enabled.

- `data/select_iid_offline_hankel.npy` is the random-IID offline behavioral
  library used by the authoritative experiment.
- Generated labels and future collected datasets also belong under `data/`.

## Organized Artifacts

- `artifacts/fair_iid_suspect/`: old fair-IID output files; currently show failure, not success.
- `artifacts/pure_deepc/`: finite-horizon pure-DeePC result artifacts.
- `artifacts/stabilizing_behavior/`: sanity-check artifacts showing reuse of stabilizing/expert-offline behavior.
- `artifacts/overlap_hybrid/`: overlap-hybrid result artifacts.
- `scratch/label_experiments/`: temporary label-search artifacts.
