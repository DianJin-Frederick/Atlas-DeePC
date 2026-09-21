# Verified résumé wording — completed study

Recommendation: use **one modest supporting bullet** under the broader
Atlas-DeePC research project. These are alternatives, not three separate
accomplishments. The study does not support a standalone claim of superior,
safe, real-time or real-robot hybrid manipulation.

## 1. Recommended: experimental contribution

“Established a reproducible robotic-control benchmark, measured by 2,000
verified MuJoCo Pusher rollouts across 100 paired seeds and four conditions,
by implementing imitation-only, unguided predictive, hybrid, and two component
ablation baselines under frozen configurations.”

Evidence: [protocol and baseline definitions](README.md),
[frozen configuration](final_protocol.json),
[2,000-trial verification](final/independent_verification.json),
[complete per-trial results](final/per_trial.csv).
Scope: simulation on one host; 100 distinct paired seeds, not 2,000 independent
seeds. “Unguided” means no learned policy guidance, not no demonstration data.

## 2. Alternative: rigorous diagnosis

“Identified a hybrid controller's disturbance-recovery limitation, measured
by 13% versus 37% final task success against imitation-only on 100 paired
MuJoCo Pusher seeds, by applying a frozen 10-cm object displacement at 8 s
and paired statistical testing.”

Evidence: [research conclusion and inference](FINAL_REPORT.md),
[paired result](final/summary.json) (`disturbance:C_minus_A_policy`).
Difference: −24 percentage points, pointwise bootstrap 95% CI [−34, −14];
Holm-adjusted exact McNemar p = 5.06e−5. “Final task success” is the all-seed
endpoint, not the selected recovery-eligible subgroup. Do not shorten this
to “13% recovery rate.”

## 3. Alternative: prior contribution with its limitation

“Isolated learned-guidance benefits and limits, measured by 54% nominal
success versus 13% for unguided predictive control but 66% for imitation-only
on 100 paired MuJoCo Pusher seeds, by separating policy-guided retrieval
from action regularization in a five-variant ablation.”

Evidence: [all five nominal results](final/comparison.md),
[paired estimates](final/summary.json), [matched implementation audit](README.md).
C−B = +41 pp [29, 53], Holm p = 9.16e−8. C−A = −12 pp [−22, −2],
Holm p = 0.116; do not call the latter family-adjusted statistically significant.
The full study includes sliding resistance, mass and disturbance, with negative
results retained. C/B do not have equal supervised-information budgets: the
retained policy encodes an extra corpus and joint-target labels. This is the
benefit of the existing guidance package, not proof of architectural superiority.

## Claims to avoid

- “Achieved 90% held-out success”: 90/100 is historical saved-state evidence;
  the fresh current-engine hybrid nominal result is 54/100.
- “Outperformed imitation learning,” “improved robustness,” or “safe control”:
  not established by this study.
- “Real-time 20-Hz control”: hybrid p95 computation exceeds 50 ms and has
  larger cold/tail delays.
- “Reduced energy by 10%”: the measured reduction is in squared torque,
  not energy, and accompanies lower success.
- “Validated on a real robot” or “RTX-accelerated”: neither occurred.

All wording refers only to completed results; paper publication, robot
deployment and future algorithm improvements are not claimed.
