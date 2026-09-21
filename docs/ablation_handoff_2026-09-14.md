# Reacher Atlas-DeePC Ablation Handoff

Updated: 2026-09-14. Read this before continuing the ablation work.

## Immediate Request and Current Status

The user is moving ablations to a new conversation. The latest unfinished task is:

> Replace `ablation_affine_rank.ipynb` with a rank ablation of the current
> **linear-subspace baseline**, tested on the same 100 seeds.

The user explicitly chose **every integer rank from 20 through 80**, not a coarse
seven-rank grid. Use **seeds 211--310 inclusive at each rank**: 61 x 100 = **6,100
rollouts**. The user wants the experiment run and results visible in the notebook.

**The new rank experiment has NOT been tested or launched. There are no new rank
results.** The previous conversation was interrupted during implementation. No
rank-sweep process was started by that conversation; completed earlier ablation
processes had already exited. Other conversations may have unrelated processes.

### Changes Already Made for the Rank Task

1. Moved the old notebook, preserving it, to:
   `archive/notebooks/ablation_affine_rank_contextual.ipynb`.
2. Changed `src/build_affine_rank_ablation_notebook.py` to generate that archived
   path rather than recreate the old active notebook.
3. Added **untested draft** `src/linear_subspace_rank_ablation.py` (183 lines at
   handoff). Read/review it before running.
4. Have **not** created the new notebook or its builder. Intended name:
   `notebooks/ablations/ablation_linear_subspace_rank.ipynb`.
5. Have **not** updated `notebooks/README.md` for this rename. Its old affine-rank
   link is currently stale and needs replacing.
6. Expected new result directory (does not exist yet):
   `experiments/routed_deepc_iid/linear_subspace_rank_100_v1/`.

The archived notebook studied **triggered contextual affine charts**, with a
dense 20--80 sweep on 1,000 seeds. It is not the desired baseline. Do not reuse its
results as linear-baseline results.

## Workspace and Runtime

- Repository/workspace: `/Users/kumiko/Projects/atlas_deepc_reacher`
- Existing Python:
  `/Users/kumiko/miniconda3/envs/gym-robotics-mujoco/bin/python`
- Existing Jupyter kernel: `gym-robotics-mujoco`
- Do not install another environment or duplicate large datasets. The user has
  previously run low on disk space.
- The machine reported 8 logical CPUs. The draft defaults to 6 workers, with one
  numerical-library thread per worker. This is a substantial sweep; monitor it
  honestly and keep it resumable. Other jobs can affect timing.
- Use `apply_patch` for manual source changes. Notebook builders use `nbformat`.
- Other conversations share the workspace. Do not revert unrelated changes or
  overwrite the comparison/contextual experiments.
- Local Jupyter execution previously required escalation to allow its local
  socket. It does not need new packages or internet access.

## Meaning of "Baseline Atlas-DeePC"

The authoritative entry point is **`src/baseline_atlas_deepc.py`**, matching:
`notebooks/baselines/baseline_graph_routed_atlas_deepc.ipynb`.

Do not call the lower-level `routed.run_routed_deepc` with historical defaults and
assume that is the current baseline. Its default matrix construction is affine,
and omitting the current goal policy changes terminal-reference construction.

Current baseline:

- 48 physical-configuration charts, MiniBatch K-means, clustering seed 11.
- Localization/clustering uses the latest configuration in the two-step Y_p
  history. Six trigonometric configuration channels are used; input feature rows
  are present but have zero weight. No online nearest-trajectory retrieval.
- Every chart is lifted to its **full input-output Hankel columns** for DeePC.
- **Linear**, uncentered rank-60 chart bases by default. Zero affine offset and
  no sum(g)=1 constraint.
- Current-chart matrix only. No contextual rescue, chart union, or fitted model.
- Iterative N=10 terminal-reference policy plus graph waypoint routing.
- All rollout actions come from the DeePC QP, a linear combination of the selected
  behavioral basis columns. No IK, hand-designed feedback, or expert-action replay.

### Baseline Files

- `src/baseline_atlas_deepc.py`: current controller wrapper and linear cache.
- `src/routed_deepc_experiment.py`: configuration, clustering, routing/control loop.
- `src/candidate_goal_routing_experiment.py`: `CandidateGoalPolicy`, path tables,
  file hashes, timing key names.
- `src/expert_pipeline_search.py`: output map, Hankel partition, SVD, DeePC QP.
- `src/baseline_1000_benchmark.py`: `summarize_timing` and stage names.
- Offline data: `data/select_iid_offline_hankel.npy`.
- Frozen baseline atlas:
  `experiments/routed_deepc_iid/clustering_jointconfig48_fresh211_310/behavioral_atlas.npz`.
- Existing full baseline results:
  `experiments/routed_deepc_iid/baseline_linear_iterative_n10_v1/evaluation_1000/`.

### Outputs and Parameters

Output order, P=10:

```
[cos(q1), cos(q2), sin(q1), sin(q2), dq1, dq2,
 fingertip_x, fingertip_y, cos(q1+q2), sin(q1+q2)]
```

The fingertip coordinates are absolute, reconstructed from the environment's
relative fingertip/target observation. The final two channels are computed
algebraically from the first four, not measured by additional sensors.

```
M=2, P=10, T_ini=2, T_f=10, L=12
H shape = (144, 37800)
clusters=48, svd_rank=60, svd_bound_scale=3
Q = diag(5000,5000,5000,5000,20,20,100000,100000,5000,5000)
R = 0.0001 I
lambda_g = 0.05       # Current nominal baseline, NOT the noisy lambda_g=0 study.
lambda_ini = 100000
lambda_du = 5
additional terminal Q = 0
input limits = [-1, 1]
steps = 320
edge_count_threshold = 3
target_switch_distance = 0.07 m
goal_source_target_min_distance = 0.06 m
```

The coefficient bounds are computed from projected offline coordinates as mean
plus/minus 3 standard deviations. Changing rank changes the number of coordinates,
but must not change that bound rule or other weights.

`pipe.fixed_rank_left_svd_basis` uses the eigendecomposition of X X^T. With linear
geometry X is uncentered. `routed.chart_matrix` returns dictionaries with keys
`basis`, `offset`, `g_min`, `g_max`; the key is **not** `A`.

### Terminal Reference and Graph

At rollout setup, filter source episodes whose original target is less than
6 cm from the online target. Among eligible offline states, select the ten
fingertip positions closest to the online target in Euclidean distance, with
deterministic source-index tie breaking. Candidate states use the latest Y_p
configuration. Within each candidate chart retain the nearest candidate.

At each online step select the candidate chart with minimum directed route cost
from the current chart. The full reference retains that candidate's joint
configuration, sets joint velocities to zero, and sets fingertip position to the
exact requested target. Thus the candidate pool is fixed per rollout but the
chosen terminal configuration can change online.

Graph edges use observed offline transitions; baseline costs are chart-center
distance divided by the square root of transition count. Before the 7 cm latch,
the next chart's representative provides a waypoint reference. After the latch,
the selected terminal reference is used. The graph supplies references, not actions.

### Data Fairness and Evaluation

- Same random-exploration offline episodes, seeds 2026--2225, 200 episodes.
- No stabilized offline episodes; best offline final-tail mean about 29.331 mm.
- No new expert or stabilized trajectories may be added to improve results.
- Seeds 211--310 are reused exploratory ablation seeds, not untouched test seeds.
- Strict success: BOTH final-30 and final-100 windows must have mean fingertip
  error <10 mm and maximum fingertip error <20 mm.
- Do not truncate rollouts when the target is first reached. Reaching once is
  different from sustained success.

## Finishing the Rank Sweep

### What the Draft Implements

`src/linear_subspace_rank_ablation.py` currently:

- Defaults to ranks `range(20,81)` and seeds `range(211,311)`.
- Runs rank 60 first, then all other ranks, using the exact baseline wrapper.
- Changes only `svd_rank` via `dataclasses.replace`.
- Uses the same frozen 48-chart atlas, graph, output map, and goal algorithm.
- Constructs each rank's linear cache through the existing baseline API.
- Keeps only the current rank's controller per worker to limit memory.
- Checks basis shape `(144, rank)`, zero offsets, and absence of `g_sum`.
- Saves compact per-seed `.npz` checkpoints: distance history, six timing arrays,
  and summary metadata. Full trajectories exist only in temporary files, then
  are discarded. No video/rendering overhead is included.
- Writes a completed summary per rank and incrementally updates a root summary.
- Hashes data, atlas, and controller sources to reject incompatible caches.

### Required Review/Tests Before Launch

1. Review the draft. **It has not been compiled, smoke-tested, or run.**
2. Add fail-fast cancellation around failed futures. Currently the process-pool
   context manager can wait for other queued jobs after a future raises. Do not
   let a bookkeeping error consume all 6,100 rollouts.
3. Test low/default/high ranks (20,60,80): actual QP dimensions, unchanged Q,
   zero affine offset, same chart memberships, correct retained ranks, full
   terminal-reference policy. Test compact checkpoint loading and timing summary.
4. Confirm rank 60 reproduces the baseline reasonably. A fresh current-baseline
   100-seed run in the preceding ablation obtained 93/100.
5. Finish source edits before creating long-run manifests; edits to hashed files
   invalidate resumability. Do not bypass hash checks for different algorithms.
6. Build `ablation_linear_subspace_rank.ipynb` with a root-finding setup, dense
   RANKS and 100-seed configuration, and `RUN_EXPERIMENTS=False` after completion.
7. Update the notebook index; preserve the archived contextual-affine study.

Suggested report: success vs rank; median/P90 final-100 mean error vs rank; a
100-seed error or success heatmap; per-module timing vs rank; median and P99 total
online times; repaired/regressed seeds relative to rank 60. Mark rank 60 as the
baseline. Collapse large per-seed tables. Do not display Wilson intervals unless
the user asks for them.

After review, intended run command:

```bash
/Users/kumiko/miniconda3/envs/gym-robotics-mujoco/bin/python -u \
  src/linear_subspace_rank_ablation.py --workers 6
```

Execute and save the notebook when the full sweep is complete. Inspect generated
figures and validate all code cells executed without errors. Do not claim the
sweep is complete based on partial cached summaries.

## Timing Conventions

`baseline_1000_benchmark.summarize_timing(paths)` pools per-step timings after
discarding the first five steps of each rollout: **31,500 timed steps per rank**.

Modules:

- Chart localization
- Terminal-reference selection
- Graph routing
- Linear subspace retrieval
- QP construction and solve
- Other control-loop work
- Total online control

QP timing includes construction and solve, not just the solver's internal time.
Offline SVD and simulator stepping are excluded. One-time target candidate setup
is reported separately. Module means add to the mean total; medians and P99s do
not generally add. Multiple workers and other tasks influence wall-clock timing;
do not describe this as an isolated hardware benchmark.

## Completed Ablation: Remove Joint-Sum Outputs

Notebook: `notebooks/ablations/ablation_joint_sum_output_channels.ipynb`

Runner: `src/baseline_output_channels_ablation.py`

Builder: `src/build_output_channels_ablation_notebook.py`

Tests: `src/test_baseline_output_channels_ablation.py` (3 passed).

Results: `experiments/routed_deepc_iid/baseline_output_channels_linear_n10_v1/`.

Completed fresh 100-seed comparison, noise-free, nominal lambda_g=0.05, current
linear N=10 baseline. Eight-output version removes the two derived channels
throughout the pipeline and refits clustering, graph, representatives, and SVD.
Remaining posture/velocity costs stay active. Same raw offline trajectories.

| Metric | 10 outputs | 8 outputs |
| --- | ---: | ---: |
| Strict success | 93/100 | 83/100 |
| Median final-100 mean error | 2.0623 mm | 2.3467 mm |
| P90 final-100 mean error | 5.4260 mm | 18.4311 mm |
| Ever reached <10 mm | 94/100 | 90/100 |
| Unsolved QPs | 0 | 0 |
| Median online time, shared computer | 41.274 ms | 40.028 ms |

Repaired: `[232,247,252,288,299]`.

Regressed: `[231,237,241,242,243,251,253,255,262,274,283,284,287,294,298]`.

All four notebook code cells executed, with four saved inline figures. The
earlier `src/reduced_output_ablation.py` is a historical experiment with old
algorithm defaults; only its Hankel-row-removal helper was reused.

One metadata-only wrapper bug was corrected during this run. `final_goal` in the
saved trace contains the true target; the baseline summary does not originally
contain `exact_target_xy`. The wrapper now adds that summary field from
`final_goal`. Valid saved numerical runs were reused, not altered. Migration
provenance is saved in the result directory.

### Explanation Given to the User

The two derived channels describe distal-link absolute orientation, q1+q2. They
are deterministic functions of observed joint sines/cosines and add no sensors,
privileged state, or independently measured information. They change clustering
geometry, orientation tracking cost, and finite-rank SVD/bound geometry.

With ideal two-link kinematics and absolute fingertip position already present,
they are even linearly recoverable from existing outputs. Do not claim that they
necessarily increase independent information or linear representational capacity.
The ablation supports practical benefit under the tested settings, but does not
isolate clustering vs prediction vs cost. Present as task-specific feature design,
not the core Atlas-DeePC contribution.

## Completed Ablation: Position-Only Q Under Noise

Notebook: `notebooks/ablations/ablation_position_only_tracking_cost.ipynb`.

Runner: `src/position_only_q_ablation.py`; builder:
`src/build_position_only_q_notebook.py`.

Result root: `experiments/routed_deepc_iid/position_only_tracking_q_noise_100_v1/`.

500 new runs: same 100 seeds at five noise levels. lambda_g=0. Only
`q_data_goal_joint=0` and `q_dq=0`; all ten outputs, references, routing, and
past-output penalties retained. Exact same noisy atlases/bases and Gaussian
sensor draws as the original full-Q noise study were reused.

| Noise SD / signal SD | Full Q successes | Position-only Q successes |
| --- | ---: | ---: |
| 0% | 93/100 | 0/100 |
| 1% | 90/100 | 0/100 |
| 2% | 86/100 | 0/100 |
| 5% | 33/100 | 0/100 |
| 10% | 2/100 | 0/100 |

Position-only median final-100 mean errors: 87.068, 70.559, 75.699, 70.221,
73.829 mm. All QPs solved. Without noise, 58/100 briefly reached within 10 mm but
none stabilized under the strict test. This jointly removes posture tracking and
velocity damping; it does not isolate their contributions.

Notebook fully executed: six code cells, 14 figures, matched-noise integrity
checks, errors, time-near-target diagnostics, and module timings.

## Completed Noise Study

Notebook: `notebooks/ablations/ablation_measurement_noise.ipynb`.

Runner: `src/measurement_noise_ablation.py`.

Result root: `experiments/routed_deepc_iid/measurement_noise_lam_g0_100_v1/`.

Current version uses lambda_g=0, full Q, five noise levels and 100 seeds each.
Its success counts are the full-Q column above. Median final-100 mean errors:
2.0616, 2.9402, 4.5064, 10.8490, 17.3069 mm.

Noise convention: independent Gaussian noise on all ten output channels,
offline and online, SD ratio rho relative to each clean offline channel's SD.
Power NSR is rho^2. Offline noise is injected once per unique time sample before
forming overlapping Hankel windows. Inputs and targets are exact. Simulator truth
is used for evaluation, not supplied to the controller. No filtering. Noise on
lifted channels is independent, so noisy channels can be kinematically inconsistent.

Six diagnostic noisy videos are embedded, paired for seeds 212,222,237. They live
under that result root's `videos/` folder. Renderer:
`src/render_measurement_noise_videos.py`, reusing
`src/render_routed_deepc_dashboard.py`. Do not regenerate videos for the rank sweep
unless requested. Keep compact storage.

## User Preferences

- Reuse existing functions and the exact current baseline; no silent switch to
  historical affine/contextual methods.
- No stabilized/expert offline data or hidden control fallbacks to manufacture
  success. Keep failures and all requested seeds.
- Save runnable notebooks with executed results and clear plots.
- Use readable colors, zero-based linear axes for computation-time bars, error
  distributions and near-10-mm detail, not just success rates.
- Collapse large tables. No Wilson-interval text requested currently.
- Distinguish full-duration success evaluation from edited presentation videos.
- Preserve prior results, archive superseded notebooks, keep the project organized.
