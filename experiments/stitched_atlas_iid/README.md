# Stitched Fixed-Chart Atlas Control

This experiment tests a chart method that is structurally different from
Select-DPC. The offline IID random-action library is partitioned once into 64
fixed state charts. Each chart learns an affine one-step dynamics model. During
control, the optimizer composes a sequence of these fixed models across the
prediction horizon; it never retrieves nearest trajectories or rebuilds a local
Hankel matrix online.

The controller uses a 50-step horizon to move across charts and a three-step
local horizon inside 3 cm of the target. Two static goal-configuration charts
represent the two elbow branches. The compatible branch is chosen from the
measured initial joint state. These goal charts contain isolated random states
only; no actions or neighboring trajectory samples are reused.

This is best described as **Atlas predictive control with learned affine
behavioral charts**, not classical single-Hankel DeePC. It tests the Atlas
principle that local models should be learned offline and stitched during
control.

## Fairness

- Offline inputs are IID random actions from 200 episodes of 200 steps.
- The audit finds zero offline episodes that stabilize at their own target.
- Evaluation seeds are disjoint from collection seeds.
- Goal-state source episodes use different environment targets, separated from
  the evaluation targets by 6.1 cm to 27.6 cm.
- No IK controller, stabilizing feedback, expert action, or fallback controller
  is used.
- No online nearest-trajectory query is performed.

## Candidate V2

Artifacts are in `candidate_v2/`. Six of eight seeds pass both the final-30 and
final-100 dwell tests.

| Seed | Mean last 30 | Max last 30 | Minimum | Result |
| ---: | ---: | ---: | ---: | :--- |
| 1 | 0.00165 m | 0.00165 m | 0.00165 m | success |
| 2 | 0.00161 m | 0.00161 m | 0.00161 m | success |
| 3 | 0.06624 m | 0.06624 m | 0.04636 m | failure |
| 4 | 0.00286 m | 0.00286 m | 0.00286 m | success |
| 5 | 0.00128 m | 0.00128 m | 0.00128 m | success |
| 7 | 0.08702 m | 0.08702 m | 0.04536 m | failure |
| 8 | 0.00040 m | 0.00040 m | 0.00015 m | success |
| 42 | 0.00824 m | 0.00824 m | 0.00268 m | success |

The successful seed-5 run has one infeasible QP step. Seed 8 has ten failed QP
steps. Those steps apply zero action; the saved status counts are retained in
`summary.json` rather than hidden.

Run the fixed evaluation with:

```bash
conda run -n gym-robotics-mujoco python src/stitched_atlas_mpc_experiment.py \
  --reuse-atlas experiments/stitched_atlas_iid/candidate_v2/stitched_atlas.npz \
  --output-name candidate_v2_reproduction
```

Generate the saved-result plots with:

```bash
conda run -n gym-robotics-mujoco python src/plot_stitched_atlas_results.py \
  experiments/stitched_atlas_iid/candidate_v2
```
