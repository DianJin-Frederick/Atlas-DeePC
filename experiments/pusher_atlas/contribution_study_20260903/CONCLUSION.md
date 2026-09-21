# Audit and development findings — final evaluation pending

> Archived pre-approval pilot report. The final evaluation is now complete;
> use [FINAL_REPORT.md](FINAL_REPORT.md) for the verified held-out findings.
> The text below preserves what was known before the final panel was run.

**Completed:** audit of 200 historical trajectories; exact decoder and controller
parity checks; 40 development trials; a separate 10-trial recovery check; six
metric-definition unit tests. **Not completed:** the fresh paired 100-seed final
evaluation. No held-out hybrid-versus-imitation result is claimed.

## What is established

The historical 90/100 guided and 0/100 strict-atlas counts are correct when
recomputed from all saved state trajectories. However, removing the learned
policy also removed the demonstration anchor and changed retrieval. That gap
does not isolate either the learned policy's contribution or the value of adding
DeePC to imitation. The new matched B baseline retains the demonstration anchor
and removes only learned guidance. [Audit](audit.json), [design](README.md).

The policy predicts joint targets, decoded by the existing bounded PD law.
There is no new policy-only decoder: 200 expert/state decoder checks have zero
numerical error. The new C rollout reproduces the original combined controller
exactly on development seed 51001, with zero maximum action/state error.
[Decoder checks](audit.json), [controller/dynamics checks](validation.json).

Historical replay is not reproducible under the current installed engine:
saved actions for seeds 201 and 202 diverge from their stored states and no
longer succeed. Current MuJoCo is 3.12.0; environment history records 3.7.0, but
old run manifests are incomplete. Engine drift is a hypothesis, not an isolated
cause. Historical saved-data verification is not current-engine reproduction.
All new comparisons use the same current engine and saved policies/atlas.

## Development results, not a research performance claim

Every cell below is only **two paired development seeds**, 51001 and 51002.
The recovery row is the separately logged 8-second check; the initial 6-second
pilot is preserved, not overwritten or excluded from the record.

| Condition | A: imitation only | B: predictive, no learned guidance | C: combined | D: selection only | E: regularization only |
|---|---:|---:|---:|---:|---:|
| Nominal | 2/2 | 0/2 | 2/2 | 2/2 | 2/2 |
| Sliding-resistance shift | 2/2 | 1/2 | 1/2 | 2/2 | 2/2 |
| 50-g object mass | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| Displacement at 8 s | 2/2 | 0/2 | 0/2 | 1/2 | 0/2 |

[Full initial-pilot table](pilot/comparison.md), [per-trial CSV](pilot/per_trial.csv),
[paired estimates and intervals](pilot/summary.json),
[later recovery check](recovery_pilot/summary.json).

For nominal control, the pooled warm-up-excluded median/p95 computation times
were 0.19/0.37 ms for imitation alone versus 15.16/39.12 ms for the combined
controller. Mean final distances were 1.52 versus 5.35 cm. The combined
controller used less torque-square integral (158.03 versus 201.49) and had fewer
sampled joint-limit excursions (102 versus 212 of 480 control samples).
These are descriptive development observations, not statistically established
general improvements. Timing excludes simulation/rendering and was measured
serially with one numerical thread on this shared M1 Pro Mac, not the RTX PC.

All 9,600 QP calls across the 50 development trials returned solutions, with
no zero-action solver fallbacks. Applied torque limits were respected, but
joint-limit excursions occurred. The controller is not a demonstrated safety
filter; its QP does not constrain joint positions or contact forces.

The original 6-second disturbance had zero goal-restoration-eligible cases:
the object had not yet established the required pre-disturbance hold. At 8 s,
A/C were both eligible on both seeds. A recovered in 1.95 and 2.70 s; C failed
to restore the final hold before the 12-s horizon. That is a negative development
observation for the hybrid, not a general recovery-rate estimate. Both timings
and all failures remain saved. No controller was retuned.

With n=2, success intervals are very broad: Wilson 95% intervals are 34.2–100%
for 2/2 and 0–65.8% for 0/2. A two-discordance exact paired McNemar test has
p=0.5. Non-significance does not establish equivalence. Degenerate pilot
bootstrap intervals must not be read as population certainty.

## Interpretation

There is **no defensible general performance advantage over imitation-only
yet**. The pilot motivates testing an effort/constraint-excursion trade-off,
but does not justify a claim of improved nominal success, dynamics robustness,
or recovery. In particular, the method costs much more per decision while
the simpler policy is already competitive in these development trials.

Hypotheses, not established mechanisms:

- Demonstration anchoring and nominal affine charts may bias corrections
  toward nominal trajectories after an object disturbance.
- Posture/effort regularization may reduce torque usage at the expense of
  tracking accuracy or recovery.
- The shared online phase manager latches `hold` after first goal entry and
  does not return to approach/push when the object later leaves. This is a
  plausible recovery limitation, but it is shared by A–E and has not been
  isolated experimentally.
- The mass shift exposes extrapolation limits of both the policy and atlas;
  the simplified benchmark physics and engine drift limit physical interpretation.

Testing phase reacquisition, softer anchoring, richer dynamics data, contact
physics, sensor noise, or a real robot requires further experiments. None was
implemented or tuned on the final panel.

## Frozen next step and resource checkpoint

The proposed final scope is five methods × four conditions × 100 paired seeds
(62001–62100): **2,000 rollouts**. These seeds have not been simulated or examined
in this study. Retained Pusher training/probe candidate pools and previous
development/evaluation blocks do not overlap them; unavailable external history
cannot be certified.

The final protocol keeps all controller settings, sliding resistance, and mass
unchanged, and uses the 8-second disturbance because the development check
established actual goal-restoration eligibility. The initial protocol and its
results remain available. `final_protocol.json` and `frozen_protocol.json` record
the final conditions, code/artifact/XML hashes, dependencies, and decision.

After replacing the original disturbance timing with the later one, the serial
projection is **10,444 s (2.9 h)**; with 30% contingency it is **13,577 s (3.8 h)**.
Observed peak process memory was 492,503,040 bytes (about 0.49 GB); expected
rollout storage is about 230 MB plus videos. No paid compute, GPU, retraining,
or new package installation is required. This is a substantial multi-hour use
of the current laptop, so the full evaluation is pending user confirmation.

The runner checkpoints trials and has a four-hour per-invocation cap; reaching
that cap means an incomplete study to resume, not permission to drop late seeds.
See [commands and protocol](README.md) and [resource estimate](resource_estimate.json).

## Résumé recommendation

Do not present this as a proven hybrid-controller superiority result or an
independent research contribution yet. Prefer one modest experimental-methods
supporting bullet. [Verified wording candidates](RESUME_BULLETS.md) deliberately
avoid extrapolating the two-seed pilot to a held-out performance claim.

## Representative development videos

Selection is explicit: lowest development seed, 51001; paired recovery success
and failure, plus nominal success and mass-shift failure. Each replay checks
state agreement with its corresponding saved trajectory.

- [Imitation-only recovery success](videos/recovery_policy_seed51001_success.mp4).
- [Combined-controller recovery failure](videos/recovery_combined_seed51001_failure.mp4).
- [Combined-controller nominal success](videos/nominal_combined_seed51001_success.mp4).
- [Combined-controller mass-shift failure](videos/mass_combined_seed51001_failure.mp4).
