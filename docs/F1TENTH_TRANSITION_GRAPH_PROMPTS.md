# F1TENTH Atlas-DeePC Transition-Graph Prompts

Use these prompts sequentially in the F1TENTH project. Each prompt should be
given to a coding agent with access to that repository. Keep experiment outputs
separate from source code, preserve existing baselines, and reuse local
collection, Hankel, clustering, DeePC, plotting, and video utilities whenever
possible.

## Research Invariant

Every online prediction must be generated as a linear combination of a fixed
data-derived behavioral matrix, for example

```text
[u_ini; u_future; y_ini + sigma; y_future] = h_bar_k + V_k g
```

or with raw Hankel columns. Do not fit state-space dynamics and call the result
DeePC. A graph may choose charts, chart sequences, or references, but it must not
silently provide an expert control action.

## Prompt 1: Audit the Existing F1TENTH Pipeline

```text
Inspect the complete F1TENTH project before making changes. Trace the pipeline
from offline data collection through window/Hankel construction, clustering or
selection, DeePC optimization, simulator stepping, evaluation, and video
generation.

Report:
1. Simulator and environment version.
2. Action definition, limits, rate limits, and sampling period.
3. Observation and controlled-output definitions, including coordinate frames.
4. Past and future horizons and the exact row order of every Hankel column.
5. How reference trajectories, track boundaries, and progress are represented.
6. Whether any dynamics model, pure-pursuit action, MPC action, or expert
   trajectory enters the online controller.
7. Offline-data provenance: tracks, seeds, policies, crashes, completed laps,
   and overlap with evaluation scenarios.
8. Current runtime per control step, solver statuses, and failure behavior.

Identify reusable functions and correctness risks. Do not edit code yet. Create
a concise pipeline diagram and cite files and line numbers.
```

## Prompt 2: Define the Behavioral Signal

```text
Based on the audit, propose the behavioral input u and output y for a local
F1TENTH DeePC atlas. Do not assume the existing observation vector is suitable.

Consider, where available:
- steering and acceleration/throttle as inputs;
- sin(yaw), cos(yaw), longitudinal/lateral velocity, yaw rate;
- cross-track and heading error in a track-relative frame;
- local curvature and distance to left/right boundaries;
- progress represented cyclically rather than as a discontinuous scalar;
- compact lidar features only if they are required for collision constraints.

Separate measured plant outputs from exogenous reference/track descriptors.
Specify which variables may be used for chart localization without leaking
future reference information. Define T_ini, T_f, normalization, angular
encoding, and the exact behavioral-column row order. Add shape assertions and a
round-trip test for partitioning and reconstructing columns.
```

## Prompt 3: Audit and Build Offline Windows

```text
Build or verify a frozen offline behavioral-window library for F1TENTH. Preserve
episode and lap boundaries: no window may cross a reset, collision, teleport,
track change, or end-of-log boundary.

Record metadata for every column, including episode, track, seed, start time,
collision status, lap completion, controller provenance, and reference segment.
Save a manifest with configuration and hashes of the data files.

Report coverage in speed, steering, yaw rate, curvature, cross-track error, and
track progress. Explicitly identify successful racing demonstrations or
repeated evaluation trajectories. Do not hide them; make it possible to run
both a broad-data experiment and a stricter held-out demonstration audit.
```

## Prompt 4: Learn Fixed Behavioral Charts

```text
Using only offline past-behavior features [U_p; Y_p], learn a fixed atlas of
F1TENTH charts. Start with normalized MiniBatchKMeans as a transparent baseline,
then optionally compare affine K-subspaces.

Requirements:
- chart assignment must not use future outputs;
- angular and cyclic variables must be encoded continuously;
- report chart sizes and reject empty or severely undersupported charts;
- visualize each chart in track-relative state space and by speed/curvature;
- calculate assignment stability and boundary chattering;
- save centers, normalization, labels, and configuration in one artifact.

For every chart k, construct both:
1. raw Hankel columns H_k;
2. a centered rank-r basis h_bar_k + V_k g with coefficient bounds learned
   from offline coordinates.

Do not run control yet.
```

## Prompt 5: Build the Observed Transition Graph

```text
Build a directed chart-transition graph from consecutive offline windows.
Process every episode separately so no edge crosses a reset or lap boundary.

For each edge i -> j, store:
- observation count and number of distinct episodes;
- distributions of transition time, input, speed, and prediction error;
- source and destination chart-center distance;
- collision and track-boundary statistics;
- representative transition windows, for diagnostics only.

Begin with a transparent reliability filter such as minimum count and minimum
distinct episodes. Compare edge costs based on count, distance, uncertainty,
and safety. Visualize the graph over track progress and flag shortcuts caused by
cyclic wraparound, self-intersections, resets, or spatially close but
dynamically incompatible track segments.
```

## Prompt 6: Define Reference-to-Chart Matching

```text
Unlike the Reacher setpoint task, F1TENTH already has a future racing-line or
reference trajectory. Do not invent a nearest-state augmented goal.

Define a task-independent compatibility score between each chart and a future
reference segment. It may use chart-level distributions of track-relative
position, heading, speed, and curvature, but it must not retrieve an expert
action sequence.

Given the current chart k_t and reference horizon r_{t:t+T_f}, return one or
more candidate destination charts with:
- reference mismatch;
- graph reachability from k_t;
- estimated transition reliability;
- safety margin.

Handle cyclic tracks and changing reference horizons explicitly. Visualize the
candidate charts for several straights, corners, and recovery states.
```

## Prompt 7: Implement Baseline Graph-Routed Local DeePC

```text
Implement the simplest graph-routed Atlas-DeePC baseline.

At every real control step:
1. Form the latest [u_ini; y_ini] and assign the measured current chart k_t.
2. Select a destination chart from the upcoming reference segment.
3. Compute a shortest directed route from k_t to that destination.
4. Choose the first chart after k_t as k_next.
5. Construct a waypoint output from a chart-level representative of k_next.
6. Feed only the CURRENT chart matrix into DeePC:

   [u_ini; u; y_ini + sigma; y] = h_bar_{k_t} + V_{k_t} g.

7. Track the waypoint/reference subject to steering, acceleration, rate, speed,
   and track-boundary constraints.
8. Apply only the first optimized action, measure again, and reroute.

Do not feed V_{k_next} into the current solve unless testing a separately named
overlap formulation. Do not use graph-edge actions. Save current/next/goal
charts, route, reference, matrix identity, prediction, action, solver status,
timing, and actual next chart at every step.
```

## Prompt 8: Diagnose Whether Requested Edges Are Realized

```text
Evaluate the baseline transition by transition. For each requested edge
k_t -> k_next, record whether the next measured chart equals k_next, remains in
k_t, skips ahead on the route, or enters an unrelated chart.

Produce:
- a requested-versus-actual transition confusion matrix;
- success rate by edge and track region;
- time required to enter each requested chart;
- DeePC prediction error and coefficient/slack statistics;
- examples of frequently requested but rarely realized edges.

Do not tune yet. Determine whether failures come from graph topology, waypoint
choice, current-chart prediction, insufficient horizon, or chart assignment
chattering.
```

## Prompt 9: Add Explicit Destination-Chart Membership

```text
Strengthen the local DeePC problem so the predicted next behavioral feature is
encouraged to lie in the Voronoi region of k_next.

For fixed destination center c_j, convert

  ||z - c_j||^2 <= ||z - c_l||^2

for every competing chart l into affine inequalities in the predicted
input-output history. First implement soft nonnegative membership slack and
penalize it heavily. Preserve convexity.

Report:
- zero-slack transition rate;
- requested/actual edge realization;
- QP feasibility and solve time;
- control and steering smoothness;
- lap completion, tracking, and collision metrics.

Compare against medoid waypoint tracking using the same frozen charts, graph,
data, seeds, and solver settings.
```

## Prompt 10: Learn DeePC-Feasible Graph Edges

```text
Replace observed adjacency with a control-relevant transition graph. For every
candidate edge i -> j, solve or approximate a local DeePC transition problem
using the chart-i matrix and a terminal chart-j membership condition.

Define an edge as feasible only when supported across multiple source histories.
Set its cost from predicted control effort, transition time, membership slack,
prediction uncertainty, and safety margin. Keep observed transition count as
evidence, not as the sole feasibility definition.

Save the tested source histories and aggregate statistics. Compare the learned
feasible graph with the raw adjacency graph and identify removed, added, and
high-cost edges. Then rerun the same closed-loop suite without changing the
local DeePC objective.
```

## Prompt 11: Test Transition-Conditioned Matrices

```text
For unreliable but important edges i -> j, construct a transition-conditioned
behavioral matrix from offline windows that begin in chart i and enter chart j
within a declared number of steps.

Compare:
- current-chart basis V_i;
- current/next overlap basis [V_i, V_j];
- transition-conditioned basis V_{i->j};
- a regularized combination that falls back toward V_i when edge data are
  sparse.

Every prediction must remain a linear combination of the declared data-derived
matrix. Report rank, condition number, column provenance, solve time,
prediction error, edge realization, and closed-loop safety. Do not silently
choose a formulation per evaluation seed.
```

## Prompt 12: Reference-Trajectory Routing

```text
Generalize from one destination chart to a chart sequence compatible with the
entire future F1TENTH reference horizon. Formulate a graph dynamic program that
balances chart-reference mismatch, transition cost, switching cost, and safety.

Compare two controllers:
1. receding single-waypoint routing: track only the next chart representative;
2. horizon chart scheduling: associate predicted horizon blocks with a planned
   chart sequence while retaining a convex DeePC subproblem whenever possible.

Keep the current-chart-only matrix baseline clearly separated. Test straights,
sharp corners, cyclic lap wraparound, recovery from cross-track disturbances,
and reference speed changes.
```

## Prompt 13: Safety and Failure Handling

```text
Audit all F1TENTH safety behavior. Add explicit steering, steering-rate,
acceleration, speed, and track-boundary constraints where the output
representation supports them.

Declare what happens when:
- the current chart has no path to the reference-compatible chart;
- the DeePC QP is infeasible;
- membership slack is too large;
- localization oscillates between charts;
- lidar or boundary estimates indicate imminent collision.

Separate a safety supervisor from the claimed Atlas-DeePC controller and report
every intervention. Never count a supervised rollout as pure Atlas-DeePC
without labeling it.
```

## Prompt 14: Computational Comparison With Select-DPC

```text
Implement or reuse a faithful Select-DPC baseline on the same F1TENTH data,
reference trajectories, horizons, constraints, and solver. Do not compare
against a simplified nearest-neighbor method while calling it Select-DPC.

Measure per control step:
- current-chart localization or trajectory-selection time;
- graph-routing time;
- behavioral-matrix preparation time;
- number and size of QPs;
- QP solve time;
- total median, p95, and maximum latency;
- missed real-time deadlines and memory use.

Scale offline library size, number of charts, chart rank, and Select-DPC subset
cardinality. Report performance and timing together; do not claim an efficiency
advantage solely from asymptotic reasoning.
```

## Prompt 15: Frozen Evaluation and Ablations

```text
Freeze one controller before broad evaluation. Use development tracks/seeds for
tuning and untouched tracks, initial conditions, disturbances, and reference
speeds for evaluation.

Run ablations for:
- no graph, current chart only;
- nearest reference-compatible chart without routing;
- raw adjacency graph with medoid waypoints;
- soft destination membership;
- DeePC-feasible edge graph;
- raw columns versus fixed-rank bases;
- Select-DPC baseline;
- optional non-DeePC MPC baseline, labeled clearly.

Report lap completion, collision rate, cross-track and heading errors, speed
tracking, control smoothness, solver reliability, prediction error, edge
realization, and timing. Preserve failures and post-hoc specialists separately.
```

## Prompt 16: Visual Evidence and Experiment Record

```text
Create an executed notebook that records the frozen F1TENTH Atlas-DeePC
experiment. Include the exact behavioral equation and matrix used at every
stage, data provenance, chart visualization, transition graph, requested versus
actual chart sequence, reference and driven trajectory, control inputs, solver
statuses, timing, and ablations.

Render videos for every primary evaluation seed. Overlay current chart, next
chart, route, active matrix name, reference waypoint, cross-track error, and
solver status. Replays must use saved actions and verify saved observations or
simulator state before being accepted. Include failures rather than showing only
the best lap.
```

## Prompt 17: Final Contribution Audit

```text
Review the final F1TENTH pipeline as a skeptical paper reviewer. Distinguish:
- ordinary clustering and chart selection;
- Reacher- or track-specific heuristics;
- the general behavioral-atlas contribution;
- the transition-topology contribution;
- evidence that graph edges are controllably realizable;
- measured computational differences from Select-DPC.

List every claim supported by frozen artifacts and every claim that remains a
hypothesis. Recommend the minimum additional experiment needed for a defensible
statement that Atlas-DeePC performs global planning over local behavioral
models rather than merely quantizing Select-DPC data selection.
```

## Suggested Core Claim

Use this only if the corresponding experiments support it:

> Atlas-DeePC organizes offline input-output trajectories into fixed local
> behavioral charts and learns a directed topology of data-supported,
> DeePC-feasible transitions. Online control plans globally over this topology
> while each action is generated by a bounded-size local DeePC optimization,
> avoiding repeated full-library trajectory selection.
