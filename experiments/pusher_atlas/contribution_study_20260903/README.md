# Pusher contribution study — 3 September 2026

Status: **completed and independently verified** — 50 development trials and
2,000 final rollouts on 100 fresh paired seeds, five methods and four conditions.
The frozen evaluation completed without controller changes or dropped trials.
Start with the [final research report](FINAL_REPORT.md),
[verified résumé options](FINAL_RESUME_BULLETS.md),
[complete secondary tables](final/secondary_metrics.md), and
[eight verified videos](final/videos/README.md).
The [earlier pilot conclusion](CONCLUSION.md) remains archived for provenance.

This study tests whether the existing imitation-guided Atlas-DeePC contributes
more than the saved imitation policy alone. It does not train new models or tune
the controller for a résumé number. Existing artifacts remain untouched.

## Audit and scope

The policy is a pair of regional two-layer tanh joint-target regressors, not
a direct torque regressor. Each expert's existing `predict()` decodes its
seven joint targets using `clip(15*(q_target-q)-3*qvel, -2, 2)`. The initial
object-to-goal direction selects an expert once per reset (x component below
-0.65 selects the turned expert). The phase and orientation are supplied inputs,
not decisions learned by the network.

In the combined method, the decoded torque has two online roles:

1. A weight-2 standardized action mismatch is added to measured-past mismatch
   when selecting a demonstration member and its chart.
2. A weight-1000 squared deviation penalizes the first QP torque only.

The retrieved demonstration coordinates also anchor `g` with weight 500.
The applied action is the clipped first QP torque, **not** another application
of PD to the QP output. Thus policy-only uses the exact existing decoder; there
is no newly invented low-level controller and no double PD stage.

Other teacher-derived online ingredients remain: the measured demonstration
atlas, chart centers/scalers, graph transitions and direction filters,
demonstrated coefficient bounds, representative joint posture in the objective,
the retrieved member anchor, the handcrafted approach/push/hold logic and
geometric offsets, and the reset-latched expert branch. Demonstrated future
reference blending has weight zero in the current configuration. No online IK,
teacher query, dynamics identification, or context rescue is used. In particular,
the old "all guidance removed" wording was too broad: even that ablation retained
demonstration-derived charts and representative posture.

The saved regional policy uses two 300-episode corpora (108,000 transitions
total). Only the routed corpus defines the saved atlas. The policy excludes
exploration-only warm-up labels; the atlas includes their measured behavior.
The MLP's early stopping uses a random within-corpus validation split, not a
held-out episode study. Policy initialization was selected on seeds 131–160.
Seeds 101–116 and 201–300 were already inspected and are not fresh holdouts.
Seed disjointness does not prove absence of memorization or distributional
overlap. Exact training reconstruction/provenance cannot be certified beyond
the retained artifacts and source.

`audit.json` recomputes the old successes from every saved state trajectory,
checks stored flags/distances and solver statuses, records hashes, checks entire
training/probe reset-candidate pools, validates the decoder, and replays two
saved action sequences under the installed engine. This is a saved-result audit,
not a fresh reproduction of all 100 historical trials.

The installed engine is MuJoCo **3.12.0** (Gymnasium 1.2.3). Replaying the saved
torques for historical seeds 201 and 202 does not reproduce their stored states:
maximum state-component errors are 0.2433 and 1.8971, respectively, and both
current-engine action replays fail the final hold criterion. The old environment
history records MuJoCo 3.7.0, but the old rollouts have no complete dependency/XML
manifest. Engine/version drift is a plausible cause, not a proven diagnosis.
Consequently, the 90/100 result is verified as a historical saved-data result,
not as a current-engine execution result. New paired comparisons are labeled
with the current dependencies; their "nominal" condition means current installed
Pusher physics, not certified identical physics to offline data collection.

## Matched variants

| Method | Learned retrieval term | Learned first-action penalty | QP |
|---|---:|---:|---:|
| A: policy only | — | — | No; execute existing decoded policy torque |
| B: predictive only | 0 | 0 | Yes |
| C: current combined | 2 | 1000 | Yes |
| D: selection only | 2 | 0 | Yes |
| E: regularization only | 0 | 1000 | Yes |

B–E retain the weight-500 demonstration anchor and use the same member-search
operator, candidate filters, graph, and QP. Setting the retrieval action weight
to zero removes dependence on the learned policy but preserves past-conditioned
member retrieval. This deliberately differs from the built-in `policy=None`
branch, which switches to chart-center gating and hysteresis. That older branch
would confound a factorial test of the two policy pathways. B is **not** a
teacher-free/strict-atlas controller.

All methods use the same measured 21-dimensional simulator state, goal, phase
manager, reset-latched region, initial two zero-input steps, 240 control steps,
50-ms control interval, and ±2 torque limits. Object velocity is read from the
simulator by all methods; this is richer than an ordinary camera observation
and does not establish real-robot observability. B–E necessarily use measured
history and demonstration retrieval; A has no predictive optimization or atlas
lookup. No method is given the perturbation parameters or future disturbance.

## Conditions and physical limitations

The stock Pusher-v5 XML has zero gravity, planar constrained object joints,
frictionless contacts (`condim=1`), 0.5 N·s/m planar damping and 0.04 kg
armature per object axis. Its body mass is approximately 7.85e-6 kg. Therefore,
changing `geom_friction` alone is a no-op and multiplying body mass by two is
almost immaterial to effective inertia. We explicitly avoid those false tests.

The development pilot uses one factor at a time:

- Nominal: unchanged benchmark.
- Sliding resistance: set dry friction loss to 0.02 N on each object slide
  joint, leaving damping and contact dimensions unchanged. This is an
  axis-separable support-friction surrogate, **not** a table-contact coefficient
  shift; diagonal resistance is not isotropic Coulomb friction. At 0.16 m/s,
  nominal viscous resistance is 0.08 N per moving axis.
- Mass: set body mass to 0.05 kg and scale body inertia proportionally, keeping
  armature fixed. Effective planar inertia rises from about 0.040008 to 0.09 kg.
  A 50-g object is plausible, but the body-mass ratio is extreme because the
  benchmark's nominal body is artificially light. This is not a ±20% uncertainty
  test or a realistic full rigid-body object.
- Disturbance: immediately before control step 120 (6.0 s after initialization),
  relocate the object by 0.10 m perpendicular to its initial goal direction.
  Odd/even seeds fix the sign. Preserve velocity. This is a reproducible
  kinematic displacement, **not** an applied force/impulse. The same displacement
  occurs at the same time for all methods, regardless of contact or success.

Development revision: the initial 6-second timing produced zero trials eligible
for restoring an established goal hold. A separately predeclared **10-trial**
check used control step 160 (8.0 s). A/C then had two eligible pairs: A recovered
on both, C on neither. The final protocol uses 8 seconds, retaining the 12-second
horizon. The original 6-second pilot remains saved and reported. This timing
choice improves metric validity; it does not improve the hybrid's pilot result.
No mass/friction condition or controller setting was adjusted.

`validation.json` checks that the mass/friction changes alter passive motion.
The runner applies the same conditions again during video replay and checks
saved-state agreement. No controller setting is changed after pilot inspection.
If a condition is invalid, any correction must be recorded before freezing.

## Protocol

Initial pilot: seeds 51001–51002, five methods × four conditions = 40 trials.
Separate later-disturbance pilot: the same development seeds × five methods
= 10 trials. The original `protocol.json` is preserved; the frozen final scope
is in `final_protocol.json`, with integrity lock `frozen_protocol.json`.
Completed final: paired seeds 62001–62100, five methods × four conditions =
2,000 trials. These were chosen without inspecting their reset states or outcomes.
The complete forbidden seed set and overlap checks are in `audit.json`.
Unavailable external history is a provenance limitation.

Primary success: all **final eight post-control object distances ≤0.08 m**.
It is not merely touching the region or holding it earlier. The episode is
12 s plus 0.1 s of common zero-input initialization. All trials run to the
fixed horizon; failures and solver fallbacks remain in denominators. Unexpected
exceptions/termination are retained as failed/incomplete trials, not rerun
until they pass. A per-invocation four-hour cap checkpoints completed trials.

Secondary metrics include final distance; final-uninterrupted goal-entry time;
its horizon-capped version with failures assigned 12 s (a capped endpoint, not
an uncensored mean arrival time); post-disturbance settling; torque-square
integral; per-component RMS torque increment; controller/simulator times;
solver retries/inaccurate solves/zero-action fallbacks; and constraint monitoring.

Recovery eligibility means eight pre-disturbance samples in goal **and** outside
immediately after displacement. Report numerator and denominator. Compare
recovery descriptively on pairs eligible under both methods, alongside the
all-seed post-disturbance primary result; do not use a selected recovery subgroup
as evidence of unconditional robustness. Settling time is only observed for
final successes and otherwise censored at episode end.

Physical monitoring checks applied actuator controls against ±2 (1e-6 tolerance)
and actual arm joint positions against XML ranges (1e-4-rad tolerance) at every
50-ms sample. This can miss between-sample excursions. The QP does not impose
joint-position, collision-force, or other physical safety constraints. Report
joint-limit excursions, not "safety." Raw QP torque overshoots before clipping
are recorded separately. Solver feasibility is not constraint-satisfaction proof.

Timing is serial on the same host, all numerical libraries limited to one
thread, with rotating method and condition order. Controller time ends when
torque is ready and excludes simulator stepping, logging, rendering, and
post-step feedback. First 20 controller calls and each chart's first call are
excluded from steady-state timing; all-step timing including cold compilation
and separate setup duration remain available. This is not a real-time scheduling
test. Runtime estimates include observed episode setup/compute/physics.

Report Wilson 95% intervals for success proportions and paired seed-bootstrap
95% intervals for mean differences (10,000 resamples; fixed RNG). The eight
primary C-versus-A/B tests across four conditions use exact McNemar tests and
Holm adjustment. D/E and secondary endpoints are exploratory. Bootstrap intervals
can degenerate with no observed discordance; this is not proof of equivalence.
Five percentage points is the predeclared practical success threshold, not a
claim that 100 seeds can precisely resolve it. Do not equate percentage-point
differences with relative improvements.

## Commands

Run from the repository root using the existing `gym-robotics-mujoco` environment.
The runner controls numerical thread counts itself. It refuses to mix changed
source/configuration/dependencies on resume. No new dependencies are required.

```bash
conda run -n gym-robotics-mujoco python src/pusher_contribution_study.py init
conda run -n gym-robotics-mujoco python src/pusher_contribution_study.py validate
conda run -n gym-robotics-mujoco python src/pusher_contribution_study.py pilot
conda run -n gym-robotics-mujoco python src/pusher_contribution_recovery_pilot.py
conda run -n gym-robotics-mujoco python -m unittest discover -s src -p test_pusher_contribution_study.py -v
# Review pilot validity and resource estimate before expanding the run.
conda run -n gym-robotics-mujoco python src/pusher_contribution_study.py freeze
conda run -n gym-robotics-mujoco python src/pusher_contribution_study.py final
conda run -n gym-robotics-mujoco python src/pusher_contribution_study.py summarize --stage final
conda run -n gym-robotics-mujoco python src/verify_pusher_contribution_results.py
conda run -n gym-robotics-mujoco python src/summarize_pusher_contribution_details.py
conda run -n gym-robotics-mujoco python src/pusher_contribution_study.py videos --stage final
```

`init` and `freeze` are intentionally one-time operations. Reuse the saved
protocol/lock on resume. Pilot outputs must not be represented as held-out
results. Videos select the lowest-seed nominal/disturbance success/failure for
A/C, when available; no post-hoc aesthetic selection.

The initial completed pilot has an earlier source manifest; later changes added
the final-protocol selector, recovery-time summary statistics, and auxiliary
render/test scripts, without changing action generation. Do not resume or
overwrite its original records with changed source. The final lock captures the
current complete code, dependencies, and XML hash.

To render any individual saved trial (including the later recovery pilot):

```bash
conda run -n gym-robotics-mujoco python src/render_pusher_study_trial.py \
  experiments/pusher_atlas/contribution_study_20260903/recovery_pilot/disturbance/C_combined/seed51001.npz \
  --output experiments/pusher_atlas/contribution_study_20260903/videos/recovery_combined_seed51001_failure.mp4
```

On this Mac, the offscreen renderer required access to the macOS graphics
service outside Codex's sandbox; simulation and timing did not. All delivered
videos have adjacent JSON records checking exact state replay.
