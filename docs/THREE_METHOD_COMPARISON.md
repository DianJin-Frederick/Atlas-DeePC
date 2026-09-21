# Three-method independently tuned Reacher comparison

Entry points: [comparison_three_methods.ipynb](../notebooks/comparisons/comparison_three_methods.ipynb),
`src/three_method_benchmark.py`, and `src/faithful_dpc_controllers.py`.
The historical [comparison.ipynb](../notebooks/comparisons/comparison.ipynb) is preserved. Its affine/SVD-adapted
comparators and previously selected seed panels are not reused as new test results.

## Frozen algorithm identities

| Method | Controller representation | Reference | Selection |
|---|---|---|---|
| Select-DPC | Original raw-column QP, L1 output slack, L1 coefficient penalty, projection penalty, sum(g)=1 | Requested fingertip target; zero velocity; no posture objective | Raw full-trajectory L1; five refinement passes per step by default |
| Contextual DPC | Paper Eq. (4), normalized raw columns, L2 coefficient norm, L1 output slack, sum(g)=1 | Requested fingertip target; no Atlas routing or goal-posture retrieval | Standardized past-output plus Q-weighted future-reference distance; one solve |
| Atlas-DeePC | Current `BaselineAtlasDeePC`: 48 uncentered rank-60 linear bases, zero offset, existing empirical coordinate bounds, no sum-to-one | Graph waypoints plus iterative minimum-route-cost choice among top-10 eligible terminal samples, including after the 7 cm latch | Current physical chart; all bases and shortest-path tables cached offline |

The Atlas adapter calls `src/baseline_atlas_deepc.py` directly. It does not copy
the old affine baseline or the old N=10 experiment that retained affine charts.
There is no contextual rescue in this Atlas method.

## Sources and fidelity boundaries

Select-DPC source: [Choose Wisely](https://arxiv.org/abs/2503.18845) and the
user-supplied `/Users/kumiko/Projects/choose-wisely-paper-main/select_deepc/`
implementation. The dependency-light Reacher port retains `SelectDeePC`'s
query/history update order, refinement loop, OSQP tolerances, and existing
open-loop/zero-action solver-failure handling. The regression suite executes the
unmodified original class bodies as an independent oracle and compares multiple
actions, selected subsets, and query histories. It does not import unrelated
Torch/manifold components.

The projection regularizer computes a pseudoinverse. Its internal numerical
factorization is part of the original method; it does **not** reduce the number
of raw trajectory coefficients. Selected column counts are limited to 400 because
the supplied code switches to a Cholesky representation above 400. The search
uses 100, 200, and 400, all on the original raw branch.

Contextual source: [Less is More, arXiv:2503.23890v2](https://arxiv.org/html/2503.23890v2),
Eq. (4)--(8), and Section II-D. The paper and
[public repository](https://github.com/bassamlab/Contextual-DPC/tree/a14ffc2e73e29a8d137d431b6832fcbb566369e5)
are not identical:

- Paper Eq. (4) uses an **unsquared L2 norm** of g and L1 output slack.
- `controllers/base/dpc.py` at commit
  `a14ffc2e73e29a8d137d431b6832fcbb566369e5` uses squared-L2 penalties.
- The default `formulation="paper_eq4"` deliberately implements the displayed
  paper objective; CLARABEL solves the resulting conic problem. OSQP cannot
  solve that objective as a QP.
- `formulation="public_code_qp"` explicitly selects the squared penalties and
  OSQP instead. This is a separately declared objective variant, **not a claim
  of reproducing every option in the current public repository**. Do not choose
  between these versions after looking at test outcomes.

The contextual sampler uses deterministic top-N selection as specified in
Eq. (7) and Algorithm 1. The paper also mentions stochastic softmin sampling in
its prose; that is not silently mixed into the top-N baseline. The effective
coefficient regularizer is `lambda_g_bar * number_of_selected_columns`.

### Reacher task adaptations (explicit, not original robot reproductions)

1. All controllers run Reacher-v5 on identical reset seeds, requested xy targets,
   sampling period, 320 control steps, and physical torque bounds [-1, 1]. Each
   gets the same two zero-torque initialization steps. Select preserves its
   original repeated-state initial history; contextual and Atlas use measured
   histories. Controllers may use different horizons and objectives.
2. The supplied Select Reacher example's half-strength torque limit and extra
   fingertip-y <= 0.1 constraint are application-specific. They are replaced by
   the common task's torque bounds and unrestricted workspace. Keeping the y
   constraint would make many benchmark targets inadmissible for Select alone.
3. Select uses its original eight-output order: periodic joints, fingertip xy,
   and joint velocities. Contextual uses those same available measurements.
   Atlas retains its ten-output map, including two deterministic joint features.
   External methods receive no Atlas-generated goal posture or waypoint schedule.
4. The single stored Hankel library is inverted back into 200 complete,
   200-sample episodes, with overlap equality checked. Each method constructs
   windows for its own horizon without crossing episode boundaries. The
   collection budget and measurements are shared; no test rollouts enter data
   matrices or normalization statistics.
5. Contextual normalizes all input/output channels from offline episode samples,
   and applies those fixed statistics to histories, references, and torque
   bounds. It retains raw coefficients over these normalized trajectories.
6. This Reacher port implements the paper's **core DPC and sampling equations
   using actual torque inputs**. It does not apply the vehicle/drone-specific
   spatial alignment or incremental-input/output augmentation from Section II-D.
   The fixed-base arm does not have their translation symmetry. An incremental
   input adaptation, if desired, needs its own documented alignment and cost
   semantics; it is not enabled by a misleading “raw data” switch. Therefore
   describe this row as a **Reacher adaptation of the paper formulation**, not an
   exact reproduction of the vehicle/quadrotor experiments.
7. Select retains its published fallback behavior. Contextual applies zero
   torque on an unsolved optimization; Atlas retains its existing zero-input
   behavior. All unsuccessful subproblems and bound violations are recorded.
   No controller is silently replaced by Atlas or a tracking feedback law.

## Tuning and evaluation

The default plan is editable **before the first run** in
`experiments/three_method_faithful_v1/plan.json`:

- Smoke: seed 90001, 12 steps. Warm-up: seed 90000, 10 steps per method.
- Development: 91001--91032. Each grid is screened on the first 8 development
  seeds; the top five candidates are evaluated on all 32. One of the five slots
  retains the incumbent so a later stage cannot force a worse configuration.
- Test: 100001--101000, never used by tuning or the smoke test.
- Two sequential coarse grid-search stages per method, 27 combinations each
  under the default plan. The second starts from the first stage's selected
  configuration. This is a budgeted staged grid search, not a claim of finding
  the global optimum. An off-grid incumbent is also evaluated if grids are edited.
- Stage 1 tunes the method's own regularizers (and contextual sample count).
  Stage 2 tunes tracking/effort tradeoffs, plus Select sample count or contextual
  horizon. Original Select regularization/cost defaults and current Atlas
  defaults are included. Structural Atlas choices remain 48 charts, rank 60,
  and N=10; Select's five refinements remain fixed by default.
- The search objective is highest sustained success count, then smallest mean
  **common external task cost over all seeds**, then config hash for deterministic
  ties. It never drops failed trials or picks a configuration per test seed.
- `freeze` records exactly one configuration per method, the complete plan,
  tuning result hash, source/data/runtime manifest hash, and test seeds. After
  freezing, tuning in that output directory is refused. `test` requires this
  freeze record. Changing a controller, data, dependencies, or a parameter after
  work starts requires a new result directory.
- The initialization audit checks test-seed collisions with filenames in this
  project's prior experiments. It cannot certify usage in other projects or
  unsaved runs; the disjoint ranges are reserved for this new study.

Early screening and the finite grid are limitations to report. Grids may be
refined on development data in a new version before committing to the final
test. Do not change them in response to final test failures.

The runner is serial and checkpointed per configuration/seed, with a single
writer lock. Tests rotate method order by seed. Numerical libraries are limited
to one thread. Resume skips valid complete records; interrupted records are
rerun. No original 211--1210 outcome or old timing batch is bootstrapped.

The raw five-refinement Select controller can be much slower than the old
SVD-adapted comparator. Review smoke-test solver failures and the notebook's
rough runtime estimate before enabling a full grid: this study can take days.
The original Select weights are an incumbent, not an assumption that the
untuned solver is reliable on this dataset. The grid includes lower penalty
scales; numerical failures remain visible rather than being reported as
evidence that the method itself cannot control the task.

## Common metrics

Sustained success requires **both** final-30 and final-100 errors to have mean
below 1 cm and maximum below 2 cm. The smoke test cannot count as a success trial.
The headline evaluation cost is independent of each controller's tuned objective:

`sum_t(100000 * ||xy_after_action[t] - requested_target||^2
       + 0.0001 * ||action[t]||^2
       + 5 * ||action[t] - action[t-1]||^2)`

The previous action at the first step is zero. Dynamic Atlas waypoint/posture
references never replace the requested xy target in this external score. Costs
include failed trials. The fixed coefficients are recorded in the plan and
cannot be tuned to make one method's reported cost smaller.

The output table contains success/Wilson intervals, common costs, final error,
pooled median and p99 computation time, max computation time, unsolved
subproblems, and bound violations. Median and p99 pool control steps after
discarding the first five of each rollout; max includes all steps. These are not
medians of per-rollout p99s. Online timing includes selection, raw matrix/QP
preparation, every Select refinement, solving, and Atlas routing/reference
selection. Cached Atlas SVD/path-table construction is outside the control loop.
One-time setup is recorded separately and must not be confused with per-step
cost. Avoid benchmarking alongside the currently running heavy experiment.

The summary and figures use only seeds with all three completed records and
label partial reports explicitly. Pairwise exact McNemar tests are diagnostic,
unadjusted for the three comparisons. Plots have zero-based linear cost axes.

## Commands

Use the `gym-robotics-mujoco` Python environment:

```sh
python src/three_method_benchmark.py init
python src/three_method_benchmark.py smoke
python src/three_method_benchmark.py tune
python src/three_method_benchmark.py freeze
python src/three_method_benchmark.py test
python src/three_method_benchmark.py report
```

The new notebook exposes these stages separately; expensive stages default to
off. `report` regenerates PNG/PDF and tables from saved outcomes and never runs
controllers. The full grid and test panel have not been launched by code setup.
