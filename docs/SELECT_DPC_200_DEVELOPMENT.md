# Select-DPC: bounded 200-column development study

## Authorized 100-seed waypoint development panel

The user has now requested **one fixed Select-DPC configuration on seeds
211--310, 200 steps each**, with intermediate waypoints; Atlas retains its graph
routing. Entry point: `src/select_dpc_waypoint_dev100.py`. Results:
`experiments/select_dpc_waypoint_dev100_v1/`. This supersedes the earlier
single-trial-only execution limit below, but does not restart the stopped grid.

Control settings remain `q_xy=40000`, `lambda_projection=1000000`,
`lambda_slack=50000`, all remaining original pilot settings unchanged, and a
uniform objective scale of 0.01. Use 200 raw selected columns, five refinement
passes, original projection/L1 penalties, and the same offline recordings as
the current Atlas notebook. No trajectory SVD is introduced.

For each seed, fix two intermediate xy points at 1/3 and 2/3 of the line from the
initial fingertip after the two zero-input initialization steps to the original
environment target. Switch within 5 mm or after 40 steps at each intermediate
reference, then hold the final target. This heuristic is not guaranteed to be
physically feasible and is not described as the paper's own path planner.
The hand-picked seed-211 pilot is not reused as a result under this different,
fixed waypoint rule. No outcome-dependent waypoint or parameter tuning occurs
within the batch.

Two workers, one numerical thread each; frozen source, runtime and data hashes;
per-seed traces and checkpoints; completed results are reused on resume.
`summary.json` updates after each completed seed and reports strict success,
last-30 and last-100 success separately, common final-target cost,
active-reference cost, median/p99/max computation time, and unsolved QPs.
Incomplete seeds do not count as completed failures or successes. Solver
fallbacks in completed rollouts remain part of the result.

The comparison with graph-routed Atlas is an **end-to-end system comparison**,
not an isolated comparison of selection methods. Atlas is not rerun or modified
here. Its current notebook reports 320-step results on seeds 311--1310, so those
aggregates cannot be presented as a matched comparison with this 200-step
development panel. Later comparisons must align seeds, initial states, final
targets, evaluation duration/criteria, and timing conditions. Online reference
generation/switching is included for Select-DPC, and Atlas routing time must
likewise remain included. Active-reference costs with different waypoint
schedules describe different objectives; final-target success and a common
evaluation cost provide shared endpoints. Development success is not an
unbiased held-out performance estimate.

## Current execution: focused trials replace the broad search

On September 16, the six-candidate background study was stopped at the user's
request. Its frozen protocol and diagnostic files are preserved; it is not
running. The schedule below describes that **superseded** study, not a current
queue.

The active approach uses `src/select_dpc_focused.py`: one explicitly chosen
parameter set on one development seed, now with **200 control steps** by default
(320 remains an explicit option), then a stop for
inspection. There is no grid, automatic expansion, or automatic 100-seed run.
The first setting is `q_xy=40000`, `lambda_projection=1000000`,
`lambda_slack=50000`, chosen from the earlier development study. Other settings
and the controller implementation are unchanged. Trial 1 uses seed 211.

Each trial saves its parameters, rationale, source snapshot, data hash, trace,
and diagnostics under `experiments/select_dpc_200_focused/`. Progress and traces
are checkpointed every ten steps. Incomplete rollouts never count as successes.
Diagnostics include prediction error, tracking error, solver status, history
slack, and the QP objective components. Use these to distinguish numerical
failure, prediction mismatch, and weak tracking before changing one weight at a
time. Once a setting looks promising, check a small shared development panel
before committing to all seeds 211--310. Report the final full-panel result,
including failures, not only the favorable pilot seeds.

The first diagnostic must not be interpreted as a 100-seed success-rate estimate.

Trial 1 was stopped with a saved 40-step trace: 30.69 mm current error and
19/200 unsolved QPs. Trial 2 uses the same seed and relative weights, multiplying
**all six objective weights** by 0.01. This preserves the exact mathematical
optimizer but changes numerical conditioning and the practical meaning of the
solver's stopping tolerances. It is explicitly recorded as a numerical
adaptation, not a verbatim reproduction of the authors' solver execution.
The original controller source and solver settings remain unmodified.

### 200-step, waypoint-assisted pilot

The user subsequently shortened pilot runs to 200 steps and authorized manually
supplied intermediate references. Trial 2's first 200 steps are preserved in
`trace_200.npz` and `result_200.json`; its original frozen plan is not overwritten.
The direct-target result passes the last-30 criterion but not the last-100
criterion (last-100 mean error 12.775 mm), so it is not counted as a strict success.

Trial 3 retains the same seed, data, control parameters, objective scaling,
selection passes and QP. Only the external reference schedule changes. It uses
`experiments/select_dpc_200_focused/seed211_manual_waypoints.json`: intermediate
xy positions (0.199, 0.006), (0.186, 0.017) m, then the unchanged final target.
These are seed-specific, manually chosen development aids, not a general path
planner and not a paper-reproduction claim. They must not be hand-tuned on a
held-out test panel or silently pooled with final-target-only runs.

The supplied authors' Reacher scheduler switches within 5 mm; that tolerance is
retained. Trial 3 additionally caps each intermediate reference at 40 steps to
reserve at least 120 steps for the final target. This timeout is our explicit
pilot-reference adaptation. Timeout events are not labeled as reaching a
waypoint. Omit `--waypoint-max-steps` for proximity-only switching.

All strict success and common-task cost measurements remain relative to the
original final target. The trace additionally stores the exact active reference
for each applied action, waypoint transitions, and active-reference cost.
Online reference switching is included in control computation time. Manual
offline reference design is not timed. The existing 320-step comparison is not
redefined by these short diagnostic trials.

## Scope

Only Select-DPC runs. The development panel is exactly seeds **211--310**,
320 control steps per rollout, following two zero-input initialization steps.
Neither Atlas nor contextual DPC is rerun. This is parameter development, not a
held-out comparison. Seeds 311--1310 remain outside this tuning procedure, but
their existing Atlas results are not claimed to be previously unseen.

The current baseline notebook is
`notebooks/baselines/baseline_graph_routed_atlas_deepc.ipynb`. Its offline source
is `data/select_iid_offline_hankel.npy` via `baseline_atlas_deepc.DATA_PATH`.
The study freezes this file's SHA-256 and the baseline notebook's source hash.
It reconstructs the original recordings from overlapping windows with an exact
overlap-consistency check, then creates 2+15-step windows without crossing episode
boundaries. No data are collected, discarded based on target, centered, whitened,
or SVD-compressed. The native eight measured outputs are retained; Atlas's two
additional derived trigonometric outputs are not added to the baseline method.

## Algorithm audit

Sources: [Choose Wisely, v2](https://arxiv.org/html/2503.18845v2), Algorithms 1--2,
Appendices A and D; supplied `choose-wisely-paper-main/select_deepc/deepc_controller.py`,
`data_selectors.py`, and `reacher_simulation.ipynb`. The exact supplied files are
hashed. This is the paper's **L1-selection variant**, not Isomap.

At each control step, five selection/QP refinement passes follow the supplied
Reacher implementation, including its initialization, prediction/history update
order, and previous-open-loop/zero-input fallback on solver failure. Algorithm 1
allows an iterative convergence criterion; five is the supplied experiment's cap.
The distance is the sum of raw full-trajectory input and output L1 distances.
Each selection retains exactly 200 complete raw trajectory columns.

With `Z = [Up; Uf; Yp; ones]` and `Pi = pinv(Z) @ Z`, the QP is

```
min  sum_k (y_k-r)' Q (y_k-r) + u_k' R u_k
     + lambda_slack ||sigma||_1 + lambda_g1 ||g||_1
     + lambda_projection ||(I-Pi)g||_2^2
s.t. Up g = u_ini, Uf g = u, Yp g = y_ini + sigma, Yf g = y,
     ones' g = 1, -1 <= u <= 1.
```

The coefficient dimension remains 200. The pseudoinverse may internally use SVD;
this is the original projection computation, **not trajectory basis reduction**.
There is no added nonnegativity or Atlas coefficient bound. L1 penalties are
represented through the solver's epigraph variables, so this remains a QP.
OSQP settings remain `max_iter=6000`, `eps_abs=1e-5`, `eps_rel=1e-4`, no warm start.
QP construction, projection calculation, selection and all five solves are timed.

Explicit task adaptations: torque limits are +/-1 to match Atlas, whereas the
supplied Reacher notebook uses +/-0.5; the physical reset/initialization and
320-step success metric are those of our task. The reference is the environment's
requested final fingertip position with zero velocity, not graph waypoints or
an Atlas-generated posture. Thus this is a faithful algorithm on a shared task,
not a reproduction of every condition in the authors' experiment.

## Finite parameter shortlist

Fixed: `Ncols=200`, `Tpast=2`, `Tfut=15`, five refinements, `q_velocity=10`,
`R=10 I`, `lambda_g1=10`, input bound 1. No structural or solver search.

| Candidate | q_xy | lambda_projection | lambda_slack |
|---|---:|---:|---:|
| paper_weights | 40000 | 10000 | 5000000 |
| relaxed_slack | 40000 | 10000 | 50000 |
| prior_promising | 40000 | 1000000 | 50000 |
| middle_projection | 40000 | 100000 | 50000 |
| lower_slack | 40000 | 1000000 | 5000 |
| stronger_tracking | 400000 | 1000000 | 50000 |

The anchor comes from the stopped study's development results on seeds
91001--91032, not from this new panel or a test panel. Those numerical rollouts
are **not** reused as seeds 211--310. This uses previous work to narrow the search.

Default bounded schedule:

1. Six candidates on ten shared seeds: 211, 221, ..., 301. All are full 320-step
   rollouts; no shortened-horizon proxy for success.
2. Retain the paper setting plus the two best alternatives. Expand those three
   to every seed 211--310, reusing their ten completed screening rollouts.
3. Stop. Maximum **330 development rollouts**, plus two 10-step diagnostics on
   separate seed 210. No automatically enlarged search or held-out test.

Rank by strict success count, then lower mean common evaluation cost, then name.
Strict success requires both final-30 and final-100 mean xy error <1 cm and
maximum xy error <2 cm. Evaluation cost (not the controller's tuned objective)
is `sum(1e5*xy_error^2 + 1e-4*||u||^2 + 5*||delta_u||^2)`.
Solver failures remain in the denominator and are reported separately.
The winner is the **best fully evaluated candidate**, not a claimed global optimum.
The selected development success rate is not an unbiased generalization estimate.

## Execution and outputs

`src/select_dpc_development.py init` creates a new study directory and frozen
source copy. Use the printed frozen script for `smoke` and `run`. Initialization
refuses to overwrite an existing study. `--all-candidates` at initialization
instead evaluates all six on all 100 seeds (maximum 600 development rollouts).

Two independent workers, one numerical thread each. Libraries are built once per
worker, not every rollout. Controllers are fresh per seed. Numerical formulations
are unchanged. Tuning times on a shared computer are descriptive, not controlled
cross-method latency evidence. No assumption that Atlas's old timings are paired.

Every completed rollout has an atomic JSON record and NPZ trace. `live/` records
step progress outside the control timer. `summary.json` updates after each result;
`finalists.json` records the screening decision; `best_parameters.json` is written
only after all finalist panels finish. The process holds a single-run lock and
checks plan, source, runtime and data hashes before resuming. The original study,
baseline notebook, and their results are untouched.
