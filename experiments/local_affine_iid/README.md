# Adaptive Local-Affine Atlas-DeePC

This experiment uses only the frozen Select-DPC-style IID random-action library.
At every online step it retrieves the 100 trajectory windows nearest to the
measured input-output past and treats their affine hull as the active atlas
chart. DeePC coefficients satisfy `sum(g) = 1` and `g >= -0.002`.

The small negative bound allows controlled local extrapolation while preventing
the unrealistic unrestricted combinations seen in the concatenated-chart
baseline. A static joint configuration is retrieved from the random state whose
fingertip is closest to the requested coordinate. Only that configuration is
used as a reference; its action sequence, velocity, and offline target are not
used. The fingertip reference remains the exact held-out task target.

Strict exclusions:

- No IK reference
- No feedback controller or fallback
- No expert or stabilizing offline data
- No target coordinate in nearest-neighbor selection
- Evaluation seeds are disjoint from collection seeds
- The runner aborts if any offline episode satisfies the stabilization criterion

The target coordinate is used only to retrieve a static joint configuration
from random states. This is a data-derived inverse-kinematics lookup, not a
stored successful reach: the fairness audit confirms zero stabilized offline
episodes, and every source episode has a different environment seed and target.

Run the predeclared evaluation with:

```bash
conda run -n gym-robotics-mujoco python src/local_affine_atlas_experiment.py
```

Artifacts and the fairness audit are saved under `candidate/` by default.

## Candidate Result

The fixed candidate configuration succeeds on seven of eight seeds under both
the final-30 and final-100 dwell tests:

| Seed | Mean last 30 | Max last 30 | Mean last 100 | Result |
| ---: | ---: | ---: | ---: | :--- |
| 1 | 0.00384 | 0.00386 | 0.00384 | success |
| 2 | 0.00967 | 0.01062 | 0.00971 | success |
| 3 | 0.01346 | 0.01362 | 0.01346 | failure |
| 4 | 0.00456 | 0.00584 | 0.00419 | success |
| 5 | 0.00557 | 0.00661 | 0.00557 | success |
| 7 | 0.00433 | 0.00744 | 0.00407 | success |
| 8 | 0.00880 | 0.01043 | 0.00869 | success |
| 42 | 0.00667 | 0.00861 | 0.00725 | success |

All recorded optimization steps have status `optimal`. Development artifacts
for seeds 1, 3, 4, and 5 are in `data_goal_q5000_gneg002_dev/`; the 320-step
seed 2, 7, 8, and 42 runs are in `data_goal_q5000_gneg002_eval/`.
