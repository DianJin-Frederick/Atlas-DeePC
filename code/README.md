# Atlas-DeePC: Reacher and F1TENTH

This is the research-code directory of the `atlas_deepc` project. Run the commands
below from `atlas_deepc/code/`. The project website is a sibling at
[`../website/`](../website/), and Git is maintained at the project root.

Data-driven predictive control using local behavioral subspaces, terminal-reference
augmentation, and graph-guided waypoint references. The main Reacher controller
uses offline input-output data, not a fitted state-space dynamics model.

## Start Here

- [Notebook index](notebooks/README.md): every active notebook, grouped by robot task.
- Task folders: [Reacher](notebooks/reacher/README.md),
  and [F1TENTH](notebooks/f1tenth/README.md).
- [Main Reacher baseline](notebooks/reacher/baselines/baseline_graph_routed_atlas_deepc.ipynb):
  held-out evaluation on seeds 10001-10500 for each of four methods (2,000 rollouts).
  Ranks 53/53/57/60 were selected using development seeds 211-310 for 8 outputs
  without/with graph guidance and 10 outputs without/with guidance, respectively.
  Retains 48 raw-configuration charts, full-window projection selection, chordal
  graph costs, the simplified objective, and Clarabel. Runs serially and resumably;
  default execution prepares/checks the frozen protocol without launching rollouts.
  Includes optional chart clouds and replays, failed seeds, median rollout RMSE,
  and a LaTeX table generated after all four evaluation panels finish.
- [Historical primitive Reacher baseline](notebooks/reacher/baselines/baseline_reacher_atlas_deepc.ipynb):
  8 outputs, raw configuration K-means, nearest-centroid selection and a fixed
  terminal reference, without graph routing or coefficient bounds. Completed
  100-seed report with past/future chart clouds, timings and success/failure videos.
- [F1TENTH paper experiment](notebooks/f1tenth/baselines/baseline_f1tenth_atlas_deepc.ipynb):
  Past-history projection Atlas-DeePC and contextual-DPC with consistent offline
  and online output noise. Seven noise/signal ratios, one Atlas configuration
  selected per level and contextual sizes 30/60/90, evaluated on 20 fresh paired
  seeds per setting (560 paper rollouts). Development tuning is kept separate.
  [Protocol and commands](docs/f1tenth/README.md).
- [Historical 1,000-seed Reacher baseline](archive/notebooks/baseline_graph_routed_atlas_deepc_before_projection_chordal_20261012.ipynb):
  saved Clarabel results with lambda_g=0.05, rank-60 linear-subspace charts,
  48 physical charts, and iterative N=10 terminal-reference selection.
  Includes the paired with/without intermediate-waypoint comparison on seeds
  311-1310, chart visualizations, videos, and per-module timings.
- [Merged rank study](notebooks/reacher/ablations/ablation_selected_ranks_1000.ipynb):
  ranks 58, 59, 61, 62, 63, and 64 on seeds 311-1310.
- [Component ablations](notebooks/reacher/ablations/ablation_atlas_deepc_components.ipynb):
  saved 100-seed comparisons and paired rollout videos.

## Project Layout

| Folder | Purpose |
|---|---|
| `notebooks/` | Task baselines, comparisons, ablations, and supporting illustrations |
| `src/` | Shared controllers, experiment runners, renderers, notebook generators, and tests used by retained studies |
| `data/` | Original Reacher Hankel matrix and imported F1TENTH data/map assets |
| `configs/` | Explicit task experiment settings |
| `third_party/` | Licensed simulator dependencies, not alternative Atlas-DeePC implementations |
| `experiments/` | Retained reports, numerical rollouts, chart caches, frozen protocols, figures, and videos |
| `../website/` | Project website and video-source manifest; protected from experiment cleanup |
| `archive/` | Historical notebook snapshots and saved outputs, not the supported rerun collection |
| `docs/` | Research notes and dependency/cleanup audit |
| `scripts/` | Maintenance and saved-report verification utilities |

Task-specific notebooks, data, and results are organized as follows:

```text
notebooks/                     experiments/                  data/
  reacher/                       reacher/                      reacher/
    baselines/                   f1tenth/                      f1tenth/
    ablations/
    comparisons/
  f1tenth/baselines/
  motivation/
```

Shared controllers and utilities remain in `src/atlas_deepc/`; they are not
copied into each task. Old data/result locations are small compatibility links
to the canonical task folders. They preserve archived paths and cache signatures
without duplicating data. Website files and saved numerical results are unchanged.
The historical [migration map](docs/task_organization_2026-10-03/moves.json)
is retained for regression-test compatibility; it is not a current file inventory.

Reacher and F1TENTH are supported robot tasks. Other task
notebooks, datasets, and implementations remain removed. See the
[code map](src/README.md) for module ownership and entry points.

## Running Notebooks

### External Experiment Storage

Bulky saved experiment arrays, videos, and archived notebook snapshots are stored
under `/Volumes/Crucial X10/atlas_deepc_data/`, preserving their project-relative
`experiments/` and `archive/` paths. Local file links pass through
`.external_experiment_data`, so existing notebook paths and frozen experiment
signatures are unchanged. Do not delete these links as if they were unused data.

The original `data/` inputs (including the Reacher Hankel matrix and F1TENTH
datasets/maps), current notebooks, code, JSON result summaries, and current
figure exports remain on the computer. Saved notebook outputs and local summaries
can be viewed without the SSD; cells loading trajectories or atlas arrays require
it. Connect the SSD before running experiments or recomputing detailed reports.
New files are still created locally unless their existing path links to the SSD;
this migration does not silently redirect future experiments.

Use `python scripts/migrate_experiment_storage.py status` to check the connection
and migration count, or `python scripts/migrate_experiment_storage.py verify` to
recheck every migrated file and the unchanged offline inputs. The migration plan
and SHA-256 journal are retained in `docs/storage_migration_plan.json` and
`docs/storage_migration_checksums.jsonl`, with copies on the SSD. An interrupted
transfer can be resumed using the script's `migrate` action. Keep a second backup
of irreplaceable results; the portable SSD is not itself a redundant backup.

### Execution

Use your existing scientific Python environment. This project does not include
or require a project-local virtual environment. Notebooks find the project root
from their location. Leave run/render switches disabled to inspect saved results;
enable them deliberately to launch experiments. Some ablations currently have
their run switch enabled, so inspect their setup before using Run All.

Experiment settings are explicit in each notebook. The current Reacher baseline
uses lambda_g=0 and a complete 12-sample measured window for projection selection,
following 12 zero-input warmup steps. Its DeePC initial-condition window remains
two samples. Graph-guided cases use iterative N=10 terminal selection and chordal
waypoint routes; no-graph cases use one fixed nearest eligible terminal posture
and no graph information. No standardization, affine constraint or coefficient
box is enabled. Completed data live in
`experiments/reacher/four_baselines_projection_chordal_lamg0_clarabel_v1/`.
The reproduction entry point is `atlas_deepc.ablations.four_baseline_projection_chordal`;
its historical module name is retained to preserve source signatures.
Do not apply these settings to historical ablations, whose protocols remain frozen.
Earlier centroid-selection four-baselines used a two-step warmup, so comparisons
with the new panel are not isolated selection-only ablations.

Several results verify hashes of their source code, offline data, and chart
caches. Keep `frozen_protocol/` folders and their dependencies. Do not edit
controller modules merely to change labels, or move trace files without updating
their consumers. Shared offline-target and terminal-reference helpers live in
`src/atlas_deepc/atlas/goal_reference.py`. The retired stitched model-based
controller and its standalone results have been removed; frozen experiment
sources and provenance elsewhere remain unchanged.
Reacher dataset constants, target lookup, and independence checks live in
`src/atlas_deepc/common/reacher_data.py`; they do not impose an affine chart model.
The retired local-affine controller and its standalone results have also been removed.

## Retained Data

The original Reacher input is `data/reacher/select_iid_offline_hankel.npy`. Older result
folders are retained when current notebooks depend on them. Consult each current
notebook's explicit result paths rather than obsolete study instructions.

Generated arrays, fitted caches, and temporary files may be excluded from Git.
A clone alone therefore does not necessarily contain all local results.

Outdated cleanup reports and source backups have been removed. Obsolete
exploratory data are not kept solely to make archived notebooks runnable.
Their saved notebook outputs remain historical snapshots.
