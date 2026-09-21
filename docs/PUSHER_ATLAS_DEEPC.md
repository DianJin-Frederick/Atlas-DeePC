# Coverage-validated Atlas-DeePC for Pusher-v5

This experiment adapts the central behavioral idea of Atlas-DeePC
controller to Gymnasium MuJoCo `Pusher-v5`. At every control step, one fixed
local chart enforces

```text
[u_ini + sigma_u; u; y_ini + sigma_y; y] = h_bar + V g
```

inside a convex QP. There is no identified state-space model and no `A` or `B`
matrix. The applied action is always the first optimized `u` from this QP.

The full experiment is in
[`pusher_atlas_deepc.ipynb`](../notebooks/robot_tasks/pusher_atlas_deepc.ipynb); backend code lives in
[`src/pusher_atlas/`](../src/pusher_atlas/).

The integrated Pusher study retains a compact source-and-results bundle. Large
offline transition archives, fitted atlas arrays, and serialized policy caches
are omitted because they are reproducible using the commands below. The final
metrics, coverage figures, rollout logs, and all 16 MP4s are retained.

## Final result

The supplied validated study uses:

- 300 routed atlas episodes plus a 300-episode standard-orientation intent
  corpus (108,000 measured transitions collected in total);
- a disjoint 24-episode coverage probe;
- 24 affine behavioral charts;
- 16 disjoint evaluation seeds (101 through 116);
- a strict success criterion requiring the final eight steps to remain within 8 cm.

It achieved **16/16 sustained successes**, **16/16 goal reaches**, and a
**100% QP solve rate**. Mean closest and final object-goal distances were 3.97
cm and 5.03 cm. Training, probe, and evaluation seed sets have no overlap.

The coverage audit of the 300-episode routed atlas corpus passes every configured check: full seven-dimensional input
rank, 99.7% episode contact rate, 89.7% offline-teacher success, all phases and
phase transitions represented, every chart larger than its retained rank, and
99.66% of disjoint probe windows inside a chart radius. The held-out/train
95th-percentile novelty ratio is 1.27. Online, 89.5% of visited windows remain
inside the conservative chart radii and the 95th-percentile gate ratio is 1.17.

See `experiments/pusher_atlas/coverage_300_routed/summary.json` and its adjacent
`coverage_report.json` for the complete metrics.

### 100-seed held-out audit

A later locked evaluation of the current routed-atlas/regional-policy controller
used the disjoint seeds 201 through 300. It achieved **90/100 sustained
successes (90.0%)**, with a **95% Wilson interval of 82.6% to 94.5%**. The
object entered the 8 cm goal region in 94/100 rollouts, and all 24,000 QP calls
returned a solution. The evaluation seeds do not overlap the offline corpus,
coverage probe, original 16-seed panel, or policy-development seeds.

On this run, median complete online-step wall time was 18.1 ms and its 95th
percentile was 71.1 ms; median controller computation was 17.9 ms. These are
host-specific measurements with numerical-library thread counts fixed to one.
The per-seed records and aggregate are in
`experiments/pusher_atlas/heldout201_300/`, and the reproducible, resumable
runner is `src/pusher_heldout_evaluation.py`.

### Strict fixed-atlas ablation

To test whether the fixed atlas works without the distilled teacher, the same
100 seeds were rerun with policy inference, the first-action policy penalty,
online demonstrated-member retrieval, the coefficient anchor, and demonstrated
reference blending all disabled. The offline IK–PD trajectories remained only
as measured behavioral data. No controller weights were tuned after observing
the panel.

This strict variant achieved **0/100 sustained successes** (95% Wilson interval
0.0% to 3.7%), compared with 90/100 for the policy-guided controller. It never
entered the 10 cm region, although all 24,000 QPs returned a solution. Its
median rollout changed between approach and push 94.5 times, versus 2 times for
the guided controller, and no strict-atlas rollout entered the hold phase. Mean
fractional progress toward the goal was 3.6%, versus 70.6% with guidance.

The paired outcome table contains 90 guided-only successes and no strict-atlas-
only successes (exact two-sided McNemar p = 1.62e-27). Therefore, the current
Pusher result cannot be attributed to the fixed atlas alone. See
`experiments/pusher_atlas/heldout201_300_pure_atlas/summary.json` and
`comparison_to_policy_guided.json` for the complete record. Reproduce it with
`src/pusher_heldout_evaluation.py --pure-atlas`.

## Why the offline collector changed

Random torques almost never create useful Pusher contact. The supplied offline
teacher therefore uses IK plus joint PD to produce contact-rich data, with
stratified resets, five excitation/contact/recovery variants, and randomized arm
warm-ups. IK is optional and is never called online; any sufficiently rich set
of successful pushes can replace it.

Two label streams are retained deliberately:

- measured, often saturated torques and post-step outputs form the DeePC
  behavioral matrices;
- the smooth pre-saturation IK joint intent trains an offline first-input QP
  regularizer.

The U-shaped tool has a difficult far-right operating region. A reset-latched
90-degree tool orientation avoids a wrist singularity there. A regional pair of
offline intent regressors preserves the standard and rotated IK branches.
Their output selects an action-compatible demonstrated behavior and regularizes
the QP; it is never applied directly and does not replace the DeePC equality.

## Run the supplied full study

Use the existing environment:

```bash
conda run -n gym-robotics-mujoco python src/pusher_coverage_study.py \
  --output-dir experiments/pusher_atlas/coverage_300_routed \
  --episodes 300 \
  --probe-episodes 24 \
  --test-seeds 101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116
```

This reuses the saved offline data and regional prior, rebuilds the atlas, runs
the 16 QP rollouts, and refreshes diagnostics.

To reproduce both offline corpora and regional prior from scratch:

```bash
conda run -n gym-robotics-mujoco python src/pusher_coverage_study.py \
  --output-dir experiments/pusher_atlas/coverage_300_target \
  --episodes 300 --probe-episodes 24 --test-seeds '' \
  --orientation-turn-threshold -2 --no-reuse-data

conda run -n gym-robotics-mujoco python src/pusher_coverage_study.py \
  --output-dir experiments/pusher_atlas/coverage_300_routed \
  --episodes 300 --probe-episodes 24 --test-seeds '' --no-reuse-data

conda run -n gym-robotics-mujoco python src/build_pusher_regional_policy.py
```

Then run the supplied full-study command above. Use `--render-videos` to encode
MuJoCo MP4 replays after evaluation.

## Online timing instrumentation

Every new rollout now records policy inference, behavior selection, QP call,
state/reference construction, action post-processing, MuJoCo stepping, feedback
processing, total controller computation, and complete simulated-step wall time.
`control_compute_time_s` ends when the optimized action is ready to apply;
`online_step_wall_time_s` additionally includes `env.step` and post-step feedback
processing. Python artifact logging is excluded from both measurements.

A five-seed, 1,100-step timing-only smoke run is retained under
`experiments/pusher_atlas/timing_smoke/`. On the shared development host, while
its one-minute load average was approximately 79, it measured 46.7 ms median
controller computation and 47.6 ms median complete simulated-step wall time;
the respective 95th percentiles were 164.6 ms and 170.9 ms. These numbers are
contention-sensitive and come from the reduced five-chart smoke atlas, whose
coverage and control-success checks fail. They are not a performance claim for
the validated 24-chart study. A publication-quality number requires rerunning
the instrumented 300-episode study on an otherwise idle, documented machine.

## Package layout

- `src/pusher_atlas/config.py`: collection, coverage, atlas, QP, and evaluation configuration.
- `src/pusher_atlas/environment.py`: task state and reset-invariant behavioral outputs.
- `src/pusher_atlas/data.py`: stratified offline collection, IK intent labels, and routing.
- `src/pusher_atlas/coverage.py`: offline/probe coverage metrics and figures.
- `src/pusher_atlas/atlas.py`: reset-safe windows, centered SVD charts, graph transitions, and
  policy-aware behavioral retrieval.
- `src/pusher_atlas/policy.py`: offline joint-intent and regional priors.
- `src/pusher_atlas/controller.py`: cached local affine DeePC QPs and phase references.
- `src/pusher_atlas/experiment.py`: artifact saving and multi-seed evaluation.
- `src/pusher_atlas/video.py`: deterministic MuJoCo replay video generation.
- `src/pusher_heldout_evaluation.py`: resumable disjoint 100-seed evaluation.
