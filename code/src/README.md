# Code Map

Reacher and F1TENTH are supported. Active code is in
the `atlas_deepc` package. Notebooks add `src/` to their Python path and import
named subpackages; no package installation or new environment is required.

## Folder Layout

| Folder under `atlas_deepc/` | Responsibility |
|---|---|
| `control/` | DeePC QP and comparison controllers |
| `atlas/` | Feature maps, clustering, bases, graph routing and goal policies |
| `baselines/` | Baseline rollouts and the shared eight/ten-output unstandardized study |
| `benchmarks/` | Paired algorithm panels and computation-time summaries |
| `ablations/` | Rank, clustering, reference and controller-component studies |
| `noise/` | Distinct measurement-noise models and their study settings |
| `f1tenth/` | Vehicle task/data/reference adapters using the shared baseline controller |
| `reporting/` | Figures, saved-result statistics and video renderers |
| `notebook_tools/` | Notebook generation and maintenance, not controller code |
| `common/` | Shared Hankel reconstruction, metrics, hashes, locks and paths |
| `tests/` | Unit and short integration tests |

For example: `from atlas_deepc.baselines import baseline_atlas_deepc`.

The merged `baseline_atlas_deepc_no_standardization.ipynb` uses
`baselines/unstandardized_atlas_deepc.py`: `output_dimension=10` or `8` selects
the layout; `reference_mode="graph"` uses iterative N=10 route-cost terminal
selection and intermediate waypoints. `"terminal"` now uses a single fixed nearest
eligible fingertip configuration, without graph routing or online reselection.
Each output layout has one shared atlas and separate mode-specific rollout folders.
Three noise-video entry points share `reporting/noise_videos.py`; they contain
only study-specific settings. Noise models share Hankel reconstruction and
true-error metrics, but their different noise laws are deliberately not merged.

`atlas_deepc/module_map.json` maps previous filenames to current source paths.
`common/paths.py::source_path` resolves source files for new-run signatures.
Existing results and frozen historical code retain their original bytes.
Task-folder organization moves their directories while preserving old paths as
compatibility links; controller modules and their hashes are unchanged.
Read-only reports disclose source changes; experiment resumption still requires
an exact signature match. Use a fresh result directory for a changed protocol.

## Reacher

The main Reacher notebook now evaluates the development-selected ranks
53/53/57/60 on 500 new seeds (10001--10500), with 48 charts. Its serial,
resumable runner is `benchmarks/reacher_heldout.py`; rebuild the notebook with
`notebook_tools/build_heldout_reacher_notebook.py`. This runner reuses the shared
controller, projection selector, chordal metric, and tracking/timing helpers.
Historical reproduction entry points remain unchanged to preserve their saved
source signatures, including `ablations/four_baseline_projection_chordal.py`
and `ablations/basic_four_cases_osqp.py`. The old promotion builder reproduces
the historical presentation, not the new held-out evaluation.
`reporting/current_reacher_baseline.py` still provides shared chart-cloud and
gallery rendering; its old fixed-rank panel loader is not used for held-out data.
`render_routed_deepc_dashboard.py` loads the saved chordal geometry when displaying
chordal runs; it must not substitute centroid-cost paths. Historical controller
defaults and other task protocols are not silently changed by this promotion.

| Purpose | Modules |
|---|---|
| Current baseline protocol / shared rollout | `benchmarks/reacher_heldout.py` / `baseline_atlas_deepc.py` |
| Physical charts, graph, basis construction | `routed_deepc_experiment.py` |
| Terminal candidate selection | `candidate_goal_routing_experiment.py` |
| Shared offline targets and terminal references | `atlas/goal_reference.py` |
| Reacher dataset metadata and independence audit | `common/reacher_data.py` |
| Frozen Clarabel seed panels and waypoint comparison | `baseline_clarabel_seed_panel.py`, `baseline_clarabel_waypoint_comparison.py` |
| Algorithm comparisons | `three_method_benchmark.py`, `faithful_dpc_controllers.py`, `online_selected_deepc_comparison.py` |
| Rank studies | `linear_subspace_rank_ablation.py`, `adjacent_rank_1000_clarabel.py`, `selected_rank_1000_clarabel.py` |
| Compact K=80/96 rank study | `ablations/compact_large_k_rank.py`; reuses the K=64 rollout runner, discards validated sweep traces, retains selected-rank serial reruns; launcher `scripts/run_reacher_large_k_compact.py` |
| Current physical-sensor noise | `measurement_sensor_noise_core.py`, `measurement_sensor_noise_rank59.py`, `measurement_sensor_noise_rank62.py` |
| Other retained ablations | `*_ablation.py`, `small_atlas_ablations.py`, `clustering_methods_*.py` |
| Media | `baseline_notebook_media.py`, `render_*.py` |

`control/expert_pipeline_search.py` retains its legacy filename for traceability.
Its name does not describe the active baseline algorithm. The retired local-affine
controller has been removed; its model-independent dataset helpers now live in
`common/reacher_data.py`.

## Maintenance

F1TENTH uses `PYTHONPATH=src python -m atlas_deepc.f1tenth.noise_sweep` and the
notebook in `notebooks/f1tenth/baselines/`. Its current study uses identical
noise/signal ratios offline and online, rebuilding the atlas and contextual
library at each level. Results are in `experiments/f1tenth/noise_ratio_sweep_20261027/`.
Use the CLI `--sigma` option for an individual comparison. Clean source data and
per-level noisy datasets remain in `data/f1tenth/`. See `../docs/f1tenth/README.md`.
