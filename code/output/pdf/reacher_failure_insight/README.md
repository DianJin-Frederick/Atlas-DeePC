# Seed 573: failure insight for Figure 4

Recommended: `reacher_failure_two_panel.pdf` (3.8 x 1.35 inches).
It combines the arm configurations with horizon predictions and actual recorded
closed-loop measurements. Use its natural width to preserve 9-pt text.

Alternatives:
- `reacher_failure_prediction_companion.pdf`: the diagnostic alone, 2.15 x 1.65 inches.
- `reacher_failure_three_panel_insight.pdf`: a full-width 6 x 1.35 inch figure
  adding the planned input sequence. Its first input is essentially zero.

Each PDF has a matching 300-dpi PNG. Typography, colors and axes styling follow
`paper/plot_scripts/plot_no_waypoint_success.py`. No original paper figures,
controller parameters, or experiment traces were modified.

## Suggested two-panel caption

A failed rollout without intermediate waypoints (seed 573). (a) Plot conventions
follow Figure 3; numbers indicate chart IDs. (b) Predicted and measured
fingertip-to-target distances.

For the three-panel version, append: "(c) The planned inputs; only the first
input is applied in receding-horizon operation."

## Interpretation and provenance

The orange curve in panel (b) contains the actual recorded measurements from
the original closed-loop rollout, not a separate open-loop simulation.
The time axis uses the same elapsed-step convention as Figure 3: the measured
starting state is at step 100, and the measured samples cover steps 100-110
inclusive (saved output/distance indices 99-109). The ten predicted future
outputs from the step-100 plan are at steps 101-110. The ten planned inputs
in panel (c) are at steps 100-109;
the input at t advances the simulator to the state at t+1.
The next prediction is reconstructed independently at step 101 and extends
through step 111. Its first input and first predicted output exactly match the
saved closed-loop trace. The measured anchor on each forecast is only a display
connector; it is not the slack-adjusted past output represented by the chart.
The snapshot is stored index 100, before the 101st action. Reconstruction
matched the original first input and first output prediction; the current plot
builder verifies these matches again against the saved trace.

The forecasts predict progress while the actual closed-loop trajectory remains
near 240.7 mm from the target. Only the first planned action is applied before
replanning, so the displayed comparison is not a same-input open-loop
model-validation test. It does not isolate the causal effects of coefficient
bounds, rank truncation, history slack, or prediction horizon. No horizon or
coefficient-bound ablation was performed for these figures.

Build: run `paper/plot_scripts/plot_failure_prediction_insight.py` with the
`gym-robotics-mujoco` Python environment. The builder only reads saved data;
it does not rerun optimization or simulation. Hashes and numerical values are
recorded in `figure_metadata.json`.
The additional step-101 diagnostic is stored in `next_prediction_101.npz` and
`next_prediction_101.json`. It can be reconstructed with
`paper/plot_scripts/reconstruct_seed573_next_prediction.py`, which replays the
original inputs in a separate simulator and solves only this one saved-state QP.
