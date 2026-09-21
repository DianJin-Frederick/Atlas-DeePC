# Pusher contribution study — completed frozen evaluation

## Conclusion

**The current combined controller does not establish a task-performance or
robustness advantage over its imitation policy alone.** Adding learned guidance
to the predictive baseline substantially improves nominal and sliding-resistance
success, but adding predictive optimization to the policy does not. The hybrid
uses less squared torque and smaller action increments, with worse success and
much higher computation. This is an observed trade-off, not demonstrated
efficiency at matched task performance or a safety improvement.

Completed: **2,000 rollouts, 480,000 control steps**, five methods × four conditions
× the same 100 fresh seeds, 62001–62100. No controller settings changed after
freezing, no final trials were dropped, and no execution-error/incomplete trials
occurred. All source/artifact hashes passed verification. All 500 paired
nominal/disturbed trajectories match exactly for the 160 steps preceding the
intervention. [Independent verification](final/independent_verification.json).

## Success and paired evidence

Success means all final eight post-control distances are ≤8 cm, over a fixed
240-step, 12-second episode following 0.1 s of shared initialization. Every cell
below has denominator 100; these are **100 paired seeds**, not 2,000 independent
random seeds.

| Condition | A: policy alone | B: predictive, no learned guidance | C: combined | D: selection only | E: regularization only |
|---|---:|---:|---:|---:|---:|
| Nominal | 66 | 13 | 54 | 70 | 66 |
| Sliding resistance | 66 | 13 | 60 | 66 | 62 |
| Object mass 50 g | 0 | 7 | 0 | 6 | 1 |
| 10-cm displacement at 8 s | 37 | 4 | 13 | 15 | 17 |

The following intervals are pointwise paired seed-bootstrap 95% intervals,
10,000 resamples. The p-values apply Holm adjustment to the eight predeclared
exact McNemar tests. Differences are **percentage points**, not relative gains.

| Condition | C − A, pp [95% CI] | Holm p | C − B, pp [95% CI] | Holm p |
|---|---:|---:|---:|---:|
| Nominal | −12 [−22, −2] | 0.116 | +41 [29, 53] | 9.16e−8 |
| Sliding resistance | −6 [−16, 4] | 0.654 | +47 [34, 59] | 3.46e−9 |
| Mass | 0 [0, 0]* | 1.000 | −7 [−13, −2] | 0.0781 |
| Displacement | −24 [−34, −14] | 5.06e−5 | +9 [2, 17] | 0.116 |

The disturbance disadvantage versus policy alone survives the predeclared
multiplicity correction. The nominal disadvantage is descriptive with a
pointwise interval below zero, but **does not** meet the family-adjusted 0.05
criterion. These are different inferential procedures, not contradictory results.
There is no demonstrated success improvement over A in any condition. The
sliding-resistance result is not evidence of equivalence or non-inferiority.

*Both methods fail all mass trials. The empirical bootstrap cannot invent
unobserved discordances; [0, 0] is not population certainty. Each 0/100 success
rate has Wilson 95% interval approximately 0–3.7%. The predeclared practical
threshold was 5 pp, which this panel cannot resolve uniformly.

Exploratory separation of the two policy pathways gives nominal C−D = −16 pp
[−27, −5] and C−E = −12 pp [−23, −1]. This configuration does not show a benefit
from combining both pathways. D's 70/100 versus A's 66/100 is not a verified
superiority claim; it was not one of the primary paired contrasts.
[All rates, Wilson intervals and comparisons](final/secondary_metrics.md),
[machine-readable paired estimates](final/summary.json).

## Distance, settling and recovery

| Condition | Mean final distance A / C (cm) | Mean capped settling A / C (s) |
|---|---:|---:|
| Nominal | 8.88 / 10.24 | 8.067 / 8.894 |
| Sliding resistance | 9.25 / 9.92 | 8.072 / 8.437 |
| Mass | 34.70 / 34.38 | 12.000 / 12.000 |
| Displacement | 11.44 / 13.84 | 10.956 / 11.541 |

Capped settling assigns failures the 12-s horizon; it is not an uncensored
mean arrival time. For nominal trials, C−A = +0.828 s [0.283, 1.384]. For
displacement, final distance increases by 2.40 cm [0.86, 3.98] and capped
settling by 0.586 s [0.266, 0.944]. Secondary intervals are exploratory and
pointwise. Success-only nominal settling averages are 6.041 s for A (66 cases)
and 6.249 s for C (54 cases); those selected sets are not a matched comparison.

Recovery eligibility requires an eight-sample goal hold immediately before
displacement and an out-of-goal state immediately afterward. A recovered in
**21/65 eligible trials**, C in **3/65**. On the **58 pairs eligible under both**,
the counts are 21/58 versus 3/58: C−A = −31.0 pp [−44.8, −19.0], with a
4-second-capped recovery-time difference of +0.578 s [0.338, 0.828]. This is a
selected, descriptive subgroup; overall disturbed success remains primary.

Among their respective recovered cases, mean recovery times are 2.193 s for A
and 2.517 s for C. Only two pairs recover under both methods: C is 0.825 s
slower on average. That tiny selected subgroup cannot support a population
recovery-speed claim. [Recovery and timing details](final/detailed_metrics.json).

## Effort, physical constraints and solver outcomes

Nominal mean torque-square integral is **214.48 for A versus 192.11 for C**:
C−A = −22.37 [−26.51, −18.09], about 10.4% lower relative to A. This integral
is not mechanical energy. RMS per-component torque increment is 3.015 versus
2.898: difference −0.117 [−0.149, −0.085]. Effort and increments also decrease
in the other conditions, including mass where both methods always fail.
Less actuation with worse task completion is not evidence of better task
efficiency. No equal-success or energy-optimality claim is supported.

| Condition | A joint-excursion samples / 24,000 | C / 24,000 | C−A sample fraction, pp [95% CI] |
|---|---:|---:|---:|
| Nominal | 5,174 | 5,252 | +0.33 [−4.24, 5.04] |
| Sliding resistance | 5,123 | 5,331 | +0.87 [−3.88, 5.68] |
| Mass | 7,083 | 9,323 | +9.33 [3.11, 15.78] |
| Displacement | 6,322 | 5,435 | −3.70 [−8.15, 0.61] |

Joint excursions mean an actual arm position exceeds an XML joint range by
more than 1e−4 rad, sampled every 50 ms. Inter-sample excursions may be missed;
MuJoCo limits are compliant. The sample-fraction intervals are an exploratory
paired-seed analysis, not independent-step statistics. Nominal excursions
occur in 89/100 A episodes and 90/100 C episodes. Mean per-episode maximum
excess is 0.01029 versus 0.00916 rad; its paired difference interval includes
zero. In disturbance, the maximum-excess endpoint is modestly lower for C
(difference −0.00124 rad [−0.00247, −0.000073]), but episode incidence is
99/100 versus 100/100 and sample-fraction evidence is inconclusive. There is
**no consistent physical-constraint advantage**, and no safety claim.

Across all methods: zero applied-torque violations of ±2 at tolerance 1e−6;
**384,000 QP calls, 168 `optimal_inaccurate` statuses, four successful Clarabel
retries, zero unresolved solves and zero zero-action fallbacks**. Raw QP torques
exceed ±2+1e−6 on 135,578 calls before the common clipping stage; do not
confuse those with applied actuator violations. The QP does not constrain
joint position or contact force. Its slack variables also make solver
feasibility weak evidence of useful physical prediction.

## Online computation and actual resources

Serial measurements on the same shared **M1 Pro Mac**, one numerical thread,
rotating method/condition order. Controller computation includes measurement,
phase/reference generation, policy/retrieval/QP as applicable, and clipping;
it excludes simulation, feedback bookkeeping, logging and rendering. Steady
timing excludes the first 20 calls and each chart's first call.

| Condition | A median / p95 / p99 (ms) | C median / p95 / p99 (ms) | C steady calls >50 ms | C all-call p95 (ms) |
|---|---:|---:|---:|---:|
| Nominal | 0.19 / 0.31 / 0.55 | 15.28 / 61.04 / 121.75 | 7.32% | 97.10 |
| Sliding resistance | 0.19 / 0.33 / 0.64 | 15.21 / 62.83 / 125.87 | 7.46% | 96.92 |
| Mass | 0.19 / 0.31 / 0.55 | 14.68 / 83.27 / 157.64 | 12.35% | 114.09 |
| Displacement | 0.19 / 0.33 / 0.51 | 15.73 / 65.67 / 131.91 | 8.29% | 102.91 |

Nominal pooled median compute is about **80× higher** for C. Including cold
calls, 10.68% of nominal C decisions exceed 50 ms; A has none across all four
conditions. Nominal C's worst steady call is 435 ms, worst all-call 1.273 s;
across shifts, a C call reaches 3.451 s. These tails are not compatible with
a demonstrated hard 20-Hz guarantee in this implementation.

Nominal median simulator stepping is 0.100 ms (A) and 0.166 ms (C); feedback
is 0.010 and 0.016 ms. The per-call **sum** of measured compute + simulation +
feedback has median/p95 0.304/0.497 ms (A), 15.463/61.276 ms (C). This is not
a complete loop timer: instrumentation, logging, setup and intervention work
are excluded. C controller-object setup averages 0.141 s per nominal episode,
separate from first-chart compilation. [Full timing details](final/detailed_metrics.json).

Evaluation ran from 19:07:25 to 22:59:51 UTC: **3 h 52 min elapsed**. The
invocation's monotonic timer reports 3 h 15 min and summed trial timers
3 h 14 min. Approximately 38 minutes of epoch elapsed time is not represented
in summed trial timers, with several large checkpoint gaps. Their cause was
not isolated; host pauses/suspend/clock behavior and unmeasured work must not
be silently attributed to the algorithm or discarded. This shared-host timing
is not isolated hardware profiling or a real-robot scheduling test.
Peak process memory: **594 MB**; compressed rollout files: **225 MB**, plus
small summaries/videos. No paid compute, GPU, new dependencies or retraining.

## What was actually compared

The retained policy predicts seven joint targets. Its existing decoder is
`clip(15*(q_target-q) - 3*qvel, -2, 2)`. A executes that decoded torque; C uses
it in member/chart retrieval (weight 2) and a first-QP-action penalty (weight
1000), then executes the clipped first QP torque with **no second PD stage**.
B–E retain the demonstration anchor (weight 500), representative posture,
atlas/scalers/graph and common past-conditioned member search. B is not
teacher-free. D removes only the action penalty; E removes only learned
selection. The reference-blending weight is zero. All use the same measured
21D simulator state, goal, shared phase manager, reset-latched region,
actuation bounds and horizon, with no online IK/teacher call or context rescue.

A and C use the exact same saved policy. Policy training includes two
300-episode corpora; the atlas uses the routed corpus only. B therefore loses
both the prior and the extra supervised information encoded in it. C versus B
tests the retained guidance package, not equal-label-budget architectures.
The clean practical contribution question is C versus A. The old 90/100 versus
0/100 ablation also removed anchoring and changed routing, so did not isolate
the prior or establish improvement over imitation.

The historical **90/100 is verified from saved states only**, not reproduced
under the current engine. Two saved-action replays diverge on MuJoCo 3.12.0;
old environment history lists 3.7.0 without complete run provenance. Engine
drift is plausible, not proven. Do not present 90% as the current held-out
success or interpret 90→54 as a controlled regression. Final seeds avoid all
retained training/probe candidate pools and known tuning/evaluation seeds;
unavailable external history cannot be certified. [Audit](audit.json),
[full matched design and physics caveats](README.md).

The friction shift is **0.02 N dry sliding resistance per planar object joint**,
not a table-contact coefficient change. Contacts have `condim=1`; changing
geom friction alone would not test friction. The 50-g mass changes effective
planar inertia from about 0.040008 to 0.09 kg with armature fixed; the nominal
body mass is artificially tiny, so this is not a mild ±20% uncertainty test.
The displacement is kinematic, preserves velocity, and is not a force impulse.
These simplified, zero-gravity benchmark conditions do not establish real
robot robustness or camera-based observability.

## Interpretation and next research steps

Supported finding: learned guidance rescues a weak predictive baseline in two
conditions, but the **current hybrid is not justified as a superior replacement
for imitation alone**. Its measurable effort/smoothness reductions accompany
lower success, poor recovery and substantially greater computation. The pilot's
apparent nominal joint-excursion benefit did not persist on the final panel.

Hypotheses from development/source inspection, not isolated causal diagnoses:
the separately logged [8-second development check](recovery_pilot/summary.json)
already showed recovery on both seeds for A and neither for C. In particular,
nominal demonstration anchoring and the strong first-action penalty may limit
corrective actions; the shared `hold` phase never reacquires approach/push after
leaving the goal; after the injected displacement, the current-state policy
sees the displaced object while the QP's past-output buffer updates after the
next step. These mechanisms could limit recovery. The phase limitation is
shared by A–E and cannot by itself explain their difference. Nominal charts,
simplified physics and incomplete historical environment provenance may also
limit transfer to the mass shift.

Further work would need **new development experiments and a new held-out panel**:
phase reacquisition, controlled anchoring/penalty sweeps, more varied dynamics
data, genuine physical constraints, and explicit timing-budget engineering.
Real-contact physics, sensor estimation and robot deployment are future work,
not completed claims. No such changes were tuned on this final panel.

For the two-page résumé, prefer a modest benchmark/ablation contribution under
the broader Atlas-DeePC project, not a standalone claim of superior or safe
hybrid manipulation. [Three verified wording options](FINAL_RESUME_BULLETS.md).

## Reproduction and videos

[Frozen configuration](final_protocol.json), [integrity lock](frozen_protocol.json),
[per-trial CSV](final/per_trial.csv), [compact comparison](final/comparison.md),
[all secondary tables](final/secondary_metrics.md),
[representative videos including failures](final/videos/README.md).
All eight final videos use the predeclared lowest-seed success/failure selection
for A/C in nominal/disturbed conditions, with zero saved-state replay error.

From the project root, using the existing environment:

```bash
conda run -n gym-robotics-mujoco python src/pusher_contribution_study.py final
conda run -n gym-robotics-mujoco python src/verify_pusher_contribution_results.py
conda run -n gym-robotics-mujoco python src/summarize_pusher_contribution_details.py
conda run -n gym-robotics-mujoco python src/pusher_contribution_study.py videos --stage final
```

The first command skips existing saved trials and validates the frozen
provenance; it does not overwrite trajectories. Do not rerun `init` or `freeze`
over this study. The earlier 50 development trials and historical results
remain preserved separately.
