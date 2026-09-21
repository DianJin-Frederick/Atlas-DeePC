# Linear rank sweep with lambda_g=0 and OSQP

## Completed and validated

All **6,100** rollouts finished at **2026-09-15 12:26:42 UTC**, about 9 hours
24 minutes after launch. The parent and all workers exited, the run lock is free,
and the complete summary contains exactly all 61 ranks and 100 seeds per rank.
No recovery, solver tuning, or dropped seeds were needed.

The notebook executed successfully at **12:34:45 UTC** with all **six code cells**,
**seven inline figures**, no errors, and `RUN_EXPERIMENTS=False`. All seven figures
were visually inspected; the final notebook and figure hashes are recorded in
`notebook_execution.json`. Solver-status totals and per-rank counts were added to
the report. Every checkpoint, source hash, data hash, and paired target validated.

| Rank | Strict successes /100 | Median final-100 mean error (mm) |
|---:|---:|---:|
| 35 | 74 | 5.497 |
| 40 | 80 | 5.594 |
| 50 | 86 | 4.031 |
| 56 | 92 | 2.251 |
| 58 | 96 | 2.099 |
| 59 | 97 | 2.043 |
| 60 | 93 | 2.062 |
| 62 | 97 | 1.906 |
| 64 | 96 | 2.135 |
| 80 | 83 | 2.336 |

Ranks 59 and 62 tie for highest observed success; rank 64 remains the conventional
power-of-two option. The rank-59/60 gap persists with zero unsolved or
optimal_inaccurate QPs at either rank. This rerun does not re-establish the old
mechanistic diagnosis for the new solver; it establishes the observed gap.

Success counts are unchanged versus lambda_g=0.05/Clarabel at 51 of 61 ranks.
Changed counts (new minus old): 25:+2, 28:+2, 30:+4, 31:+4, 32:+3, 33:+9,
34:-5, 37:-1, 68:-1, 69:+1. There are 38 flipped paired success outcomes overall.
Both regularization and solver change; their separate effects are not isolated.

Across 1,952,000 QPs: **1,763,456 optimal**, **11,897 optimal_inaccurate** accepted,
and **176,647 unsolved** (172,351 solver exceptions, 3,926 infeasible,
370 infeasible_inaccurate). Original complete Clarabel sweep: 55,577 unsolved.
All unsolved steps retain the baseline's zero-action handling. Ranks 36–68 had
zero unsolved QPs; low ranks dominate failures, with some failures at higher ranks.
Returned solver statuses do not independently certify physical infeasibility.

Median online computation per step was 71.95 ms at rank 56, 78.57 ms at 59,
74.92 ms at 60, 82.39 ms at 62, 87.96 ms at 64, and 238.24 ms at 80. These are
shared-computer wall-clock measurements including QP construction and solve;
they do not isolate solver speed from changing system load.

The completion heartbeat is now PAUSED. Preserve all three
studies and their notebooks. No further simulation is required for this request.

## Request and launch

Latest user instruction: **use lambda_g=0 and OSQP**. Repeat every integer rank
20–80 on seeds 211–310, with 320 steps each: 6,100 fresh rollouts. Six workers,
one numerical-library thread each; rank 60 first, then other ranks ascending.

Launched **2026-09-15 03:02:11 UTC** (September 14 evening in America/Chicago).
Parent PID **36532**, process session **60643**. The lock was verified held.
The following launch details are historical; the completed state is recorded above.

Initial checkpoint validation: rank 60 seeds 211–240 (30 full rollouts) passed
all checkpoint checks, each with `lam_g=0`, solver OSQP, and 320 optimal QPs.
All six worker processes were verified active under the correct parent. The new
report's six code cells compiled before launch; final execution is recorded above.

- Result root: `experiments/routed_deepc_iid/linear_subspace_rank_lamg0_osqp_100_v1/`.
- Log: `/tmp/atlas_linear_rank_lamg0_osqp_sweep.log`.
- `launch.json` records PID, start time and runner. The launcher holds an
  exclusive `run.lock` through exec and rejects a duplicate launch.
- Launch/resume only if the process has stopped and after diagnosis:

  ```bash
  /Users/kumiko/miniconda3/envs/gym-robotics-mujoco/bin/python -u src/linear_subspace_rank_lamg0_osqp_ablation.py --workers 6
  ```

The previous lambda_g=0 Clarabel run (PID 35233, session 64744) was stopped by
SIGINT and exited 130. Its lock is released; 937 rollout checkpoints are preserved
in `linear_subspace_rank_lamg0_100_v1/`. Do not resume it. The original complete
lambda_g=0.05 Clarabel grid remains in `linear_subspace_rank_100_v1/`.

## Protocol and validation

The new frozen sources were copied from the verified lambda_g=0 Clarabel snapshot.
The live baseline and other studies remain untouched. Data and experiment paths
are symlinks; no data duplication. `paired_protocol_audit.json` verifies identical
configuration/data/atlas versus the interrupted lambda_g=0 study. The changes
in `source_changes.diff` are solver selection/options, solver metadata validation,
and the experiment version. The manifest hashes all seven numerical sources;
do not edit frozen sources during the run or bypass hash checks.

- Solver OSQP 1.0.4 through CVXPY 1.6.5.
- Explicit options: `max_iter=10000`, `eps_abs=1e-5`, `eps_rel=1e-5`,
  `polishing=True`. These match the installed CVXPY OSQP defaults. Other solver
  options use package defaults; the mathematical QP is unchanged.
- Every checkpoint records and validates `qp_solver=OSQP`, solver options,
  `lam_g=0`, all baseline config values, and rank/seed/version.
- No fallback solver. Preserve every rollout and the baseline's zero-action
  response to unsolved QPs. Accepted `optimal_inaccurate` steps are excluded from
  unsolved counts; aggregate solver-status counts remain in each summary.
- The rank 20/60/80 integration test passed, checking actual OSQP calls, unchanged
  Q/R and penalties, coefficient bounds, chart membership, zero offsets, no sum(g)
  constraint, shapes, compact checkpoints, failure handling and timing accounting.
- Eight-step smoke tests at seed 211: rank 20 had six optimal and two unsolved
  QPs; verbose diagnostics confirmed both hit 10,000 iterations. Ranks 60 and 80
  each had eight optimal QPs. Smoke tests are temporary and not sweep results.
- Failure-cancellation test passed. Do not retune low-rank failures away or omit
  failed seeds. Solver iteration limits do not prove mathematical infeasibility.

## Report and completion

- Notebook: `notebooks/ablations/ablation_linear_subspace_rank_lamg0_osqp.ipynb`.
- Builder: `src/build_linear_subspace_rank_lamg0_osqp_notebook.py`.
- Executor: `src/execute_linear_subspace_rank_lamg0_osqp_notebook.py`.
- Six code cells and seven figures, all executed and verified; the notebook
  refuses to report completion until all 6,100 checkpoints validate.
- Compare with the completed lambda_g=0.05 **Clarabel** sweep. Both lambda_g
  and solver change, so do not label this a lambda-only comparison. The incomplete
  lambda_g=0 Clarabel study is not substituted for OSQP results.

After the sweep exits successfully and summary.complete=true, run:

```bash
/Users/kumiko/miniconda3/envs/gym-robotics-mujoco/bin/python src/execute_linear_subspace_rank_lamg0_osqp_notebook.py
```

Jupyter local sockets may require sandbox escalation; use the existing environment
and kernel, without package installation. Check all six code cells executed with
no errors and seven inline PNG outputs, while `RUN_EXPERIMENTS=False` remains set.
Validate `notebook_report.json`, `setup_comparison_report.json`, and
`notebook_execution.json`, including exact paired rank/seed/target grids, source
and data hashes, and both the parameter and solver changes.

Visually inspect all seven output figures: `rank_success_errors.png`,
`rank_seed_heatmaps.png`, `rank_error_distributions.png`, `rank_paired_changes.png`,
`rank_module_timings.png`, `rank_total_timings.png`, `rank_setup_comparison.png`.
Record successful inspection and hashes in notebook_execution.json. Report-only
fixes are allowed; do not alter frozen simulation sources. Report high-success
and lower-rank findings, rank 60, unsolved and optimal_inaccurate counts, preserved
zero-action failure handling, and shared-computer timing limitations. Link the new
executed notebook, update this status, and pause the monitor after final delivery.

Reuse heartbeat `finish-atlas-deepc-rank-ablation`, target thread
`01a09e5c-3162-7983-873d-24b6232f8962`, every ten minutes. Stay quiet during normal
progress; notify only on completion, failure, a meaningful change or required
user action. Do not create a duplicate monitor or archive this task.
