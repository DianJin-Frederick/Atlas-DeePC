# Claims And Artifacts

## Current Honest Status

The old notebook claim remains stale, but a new scripted fair-IID Atlas-DeePC
experiment is verified. Its frozen formulation succeeds on all eight development
seeds and 17 of 20 untouched seeds without fitting a dynamics model.

## Fair IID Selected-Chart/Subspace DeePC

Notebook:

- `success_full_pipeline.ipynb`

Artifacts:

- `artifacts/fair_iid_suspect/select_iid_subspace_control_results.npz`
- `artifacts/fair_iid_suspect/select_iid_subspace_control.mp4`
- `artifacts/fair_iid_suspect/select_iid_subspace_xy_trajectory.mp4`
- `artifacts/fair_iid_suspect/select_iid_subspace_tracking.png`

Status:

- Suspect / stale.
- The notebook text claims success, but the saved result file reports failure:
  - `success = false`
  - `mean_last_30 = 0.2017047712478196`
  - `last = 0.189006098850046`

Interpretation:

- Do not cite this as a verified fair-IID success.
- The video correctly looks far from the target.

## Pure DeePC Finite-Horizon Case

Notebook:

- `pure_deepc_case.ipynb`

Artifacts:

- `artifacts/pure_deepc/seed2_h160_control_results.npz`
- `artifacts/pure_deepc/seed2_h160_summary.json`
- `artifacts/pure_deepc/seed2_h160_tracking.png`

Status:

- Verified numerically as a limited finite-horizon case.
- Saved metrics:
  - `success = true`
  - `mean_last_30 = 0.009209033985593208`
  - `max_last_30 = 0.016993216670666122`
  - `last = 0.016993216670666122`

Limitations:

- Favorable seed/configuration.
- Not robust across longer horizons or multiple targets.
- No saved MuJoCo video was found for this exact `h160` run.

## Stabilizing / Expert-Behavior Offline Data Sanity Check

Notebook:

- `stabilizing_offline_multitarget.ipynb`

Artifacts:

- `artifacts/stabilizing_behavior/seed42_control_results.npz`
- `artifacts/stabilizing_behavior/seed42_control.mp4`
- `artifacts/stabilizing_behavior/seed42_tracking.png`

Status:

- Verified saved success for seed 42.
- This is a sanity check for the mechanism: if the offline library already contains successful reaching behavior for the relevant target region, DeePC can select/recombine that behavior and reach.
- Saved metrics:
  - `success = true`
  - `mean_last_30 = 0.005343914380203674`
  - `max_last_30 = 0.01337696219260556`
  - `last = 0.005655749407056843`

Interpretation:

- This is a valid positive control, not a cheat, as long as it is described as such.
- It should not be used as the clean fair-IID random-data claim.

## Overlap-Hybrid Atlas DeePC

Notebook:

- `overlap_hybrid_pipeline.ipynb`

Artifacts:

- `artifacts/overlap_hybrid/summary.json`
- `artifacts/overlap_hybrid/multi_seed_summary.png`

Status:

- Numerically strong across several seeds.
- Saved strict successes: seeds `1`, `2`, `3`, `8`, and `42`.
- Near misses: seeds `4`, `5`, and `7` are slightly above the 1 cm mean threshold.

Limitations:

- Uses overlap/mixture charting plus IK reference and a small stabilizing feedback blend.
- Shows practical behavior reuse, not pure-DeePC-only robustness.

## Random-Walk Affine K-Subspaces Concatenation

Runner:

- `src/affine_k_subspaces_random_walk.py`

Artifacts:

- `experiments/affine_k_subspaces_random_walk/full/`

Status:

- Complete reproducible baseline using 200 random-walk episodes of 200 steps.
- Eight affine K-subspaces are fitted, each final cluster receives a rank-12
  centered SVD basis, and all chart components are concatenated into one affine
  matrix for pure DeePC.
- The QP remained optimal at every step, but none of eight evaluation seeds
  stabilized; final-30 mean distances ranged from 0.066 m to 0.184 m.

Interpretation:

- The requested pipeline is operational, but unrestricted concatenation loses
  the local-chart restriction that should make the atlas useful on a nonlinear
  plant. This result should be treated as a baseline for chart selection or
  chart-weighting experiments, not as a successful controller.

## Adaptive Local-Affine Pure DeePC

Runner:

- `src/local_affine_atlas_experiment.py`

Artifacts:

- `experiments/local_affine_iid/candidate/`
- `experiments/local_affine_iid/data_goal_q5000_gneg002_dev/`
- `experiments/local_affine_iid/data_goal_q5000_gneg002_eval/`

Status:

- Seven of eight recorded seeds pass both the final-30 and final-100 dwell tests.
- Seed 3 is the sole failure, stabilizing at approximately 0.0135 m.
- Every recorded optimization step is optimal.

Fairness:

- Uses the frozen 200-by-200 IID random-action library, not expert data.
- The automated audit finds zero stabilized offline episodes.
- Evaluation seeds are outside the collection seed range.
- No IK controller, feedback blend, or fallback is used.
- Online chart retrieval uses only measured past inputs and outputs, not target labels.
- A static joint reference is retrieved from an isolated random state near the
  requested fingertip coordinate. No actions or trajectory around that state
  are reused, and the source episode has a different offline target.

Interpretation:

- Local affine constraints are essential: `sum(g)=1` removes free-motion
  cancellation, while `g >= -0.002` permits limited extrapolation beyond the
  convex hull without returning to the inaccurate global subspace.

## Fixed-Chart And Stitched Atlas Experiments

Runners:

- `src/weighted_affine_atlas_experiment.py`
- `src/gated_chart_atlas_experiment.py`
- `src/stitched_atlas_mpc_experiment.py`

Primary artifacts:

- `experiments/stitched_atlas_iid/candidate_v2/`

Status:

- Convex weighting of full-trajectory affine K-subspaces remained inaccurate:
  chart blending produced optimistic futures with large one-step error.
- Hard gating into fixed SVD trajectory charts also stalled because a
  fixed-length chart could not represent transitions into neighboring charts.
- Stitched one-step affine charts solve that compositional problem. The frozen
  64-chart, multiscale candidate passes both dwell tests on seeds `1`, `2`, `4`,
  `5`, `8`, and `42`; seeds `3` and `7` fail.

Fairness:

- Uses the same frozen 200-by-200 IID random-action library.
- Automated audit finds zero stabilized offline episodes.
- Performs no online trajectory or nearest-neighbor retrieval.
- Uses no IK, feedback blend, expert action, or fallback controller.
- Two static random-state goal charts represent elbow branches. Their source
  episodes have offline targets 6.1 cm to 27.6 cm from the evaluation targets;
  no source actions or trajectories are reused.

Interpretation:

- This is Atlas predictive control with fixed learned affine behavioral charts,
  not classical single-Hankel DeePC. It is an honest test of the proposed Atlas
  intuition and is essentially different from Select-DPC's online nearest-100
  trajectory selection.
- Six-of-eight success is promising but not yet robust. The remaining failures
  stall before entering the 3 cm local-regulation region.

## Graph-Routed Atlas MPC Baseline

Runner:

- `src/routed_atlas_experiment.py`

Primary artifacts:

- `experiments/routed_atlas_iid/candidate_waypoint_switch7cm/`
- `experiments/routed_atlas_iid/robust28_coarse_hybrid/`
- `experiments/routed_atlas_iid/probe_seed29_graph256_thr1/`

Status:

- The 64-chart waypoint graph stabilizes all original eight seeds. All 2,560
  QPs are optimal, and every seed passes both final-30 and final-100 dwell tests.
- A coarse hybrid route, combining waypoint MPC with aggregate random
  chart-transition actions, stabilizes 27 of 28 combined seeds. This includes
  the original eight and 19 of 20 previously untouched seeds. All 8,960 QPs are
  optimal.
- Seed 29 is the sole broad-controller failure. A separately declared 256-chart
  waypoint graph resolves its folded-arm topology and holds at 2.31 mm.
- The fine graph is not automatically selected by the broad controller. Do not
  report this as one 28/28 controller.

Fairness:

- The transition graphs and edge actions are aggregate statistics learned from
  IID random transitions. They are not successful trajectories.
- The automated audit finds zero stabilized offline episodes.
- No online nearest-trajectory retrieval, IK, hand-coded PD, expert action, or
  fallback controller is used.
- Goal configurations are isolated random states from episodes with different
  environment targets; no neighboring actions are reused.

Interpretation:

- This is a graph-routed **MPC baseline**, not a DeePC result: the controller
  uses fitted local affine dynamics.
- The result supports the Atlas hypothesis more strongly than fixed chart
  concatenation or Select-DPC-like nearest-100 retrieval: local charts need an
  explicit transition topology and a short local regulation scale.
- The remaining automatic graph-resolution choice is now the clearest research
  problem. Coarse charts are more reliable globally; fine charts resolve some
  folded configurations but are brittle on other seeds.

## Graph-Routed Fixed-Chart Atlas-DeePC

Runner:

- `src/routed_deepc_experiment.py`

Notebook:

- `routed_deepc_experiment.ipynb`

Primary artifacts:

- `experiments/routed_deepc_iid/candidate_rank60_compatible/`
- `experiments/routed_deepc_iid/heldout20_rank60_compatible/`
- `experiments/routed_deepc_iid/candidate_rank60_compatible/seed1_control.mp4`
- `experiments/routed_deepc_iid/candidate_rank60_compatible/seed1_control_video_verification.json`

Status:

- The frozen rank-60 affine-SVD formulation passes both dwell tests on all 8
  development seeds and 17 of 20 untouched seeds: 25/28 combined.
- All 8,960 saved optimization steps are optimal.
- All 28 primary MP4s replay their 320 saved actions in MuJoCo. Their replay
  audits find zero numerical mismatch against every saved controller output.
- Final-30 mean error among successful seeds ranges from 0.41 mm to 4.68 mm.
- Untouched seeds 14, 24, and 27 fail under the frozen formulation. Separate
  post-hoc rank/branch diagnostics under
  `experiments/routed_deepc_iid/diagnostics/` solve each, but are not counted as
  a single automatic 28/28 controller.

DeePC formulation:

- Each current behavioral chart selects fixed columns from the offline Hankel
  matrix. A centered rank-60 left-SVD basis is computed for that chart.
- At every control step, the DeePC QP constrains the complete past/future
  trajectory to `h_bar + V_60 g`. Thus prediction is an affine linear
  combination of behavioral data columns, not a rollout of fitted dynamics.
- The offline chart-transition graph supplies waypoint references only. It does
  not supply control actions.

Fairness:

- Uses only the frozen IID random-action library; the audit finds zero stabilized
  offline episodes.
- Evaluation seeds are outside the offline collection range.
- Goal configurations are isolated random states whose original environment
  targets differ from the online target. No adjacent action or trajectory is
  reused.
- No online nearest-trajectory retrieval, fitted `A/B/c` model, IK, expert
  action, feedback blend, or fallback controller is used.

Interpretation:

- This is the strongest current evidence for the intended Atlas-DeePC idea.
  The manifold topology is learned once as charts and a transition graph, while
  every online action still comes from a behavioral-column DeePC optimization.
- The three held-out failures and their post-hoc specialists indicate that
  automatic elbow-branch and local-rank selection are the next robustness
  problem.
