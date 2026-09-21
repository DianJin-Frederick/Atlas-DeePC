# Representative final rollouts

Selection was frozen before the final evaluation: lowest-seed success and
failure, when available, for A/C in nominal and disturbed conditions. Videos
are illustrative, not success-rate estimates. Every replay reproduces its
saved state trajectory exactly; see [verification index](index.json).

| Condition | Method | Success | Failure |
|---|---|---|---|
| Nominal | A: policy alone | [Seed 62002](nominal_A_policy_seed62002_success.mp4) | [Seed 62001](nominal_A_policy_seed62001_failure.mp4) |
| Nominal | C: combined | [Seed 62001](nominal_C_combined_seed62001_success.mp4) | [Seed 62002](nominal_C_combined_seed62002_failure.mp4) |
| Displacement | A: policy alone | [Seed 62001](disturbance_A_policy_seed62001_success.mp4) | [Seed 62002](disturbance_A_policy_seed62002_failure.mp4) |
| Displacement | C: combined | [Seed 62003](disturbance_C_combined_seed62003_success.mp4) | [Seed 62001](disturbance_C_combined_seed62001_failure.mp4) |

For a matched disturbance comparison, watch the two seed-62001 videos: policy
success and hybrid failure. Nominal seeds 62001 and 62002 show opposite winners,
illustrating why representative clips cannot replace the full paired panel.
Disturbance occurs 8 s after the common 0.1-s initialization; video runs at 20 fps.

[Full research report](../../FINAL_REPORT.md), [all 2,000 results](../per_trial.csv).
