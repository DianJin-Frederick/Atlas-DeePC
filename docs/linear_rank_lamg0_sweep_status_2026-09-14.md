# Linear rank sweep repeated with lambda_g=0

## Superseded by the OSQP update

The user subsequently requested **lambda_g=0 and OSQP**. This Clarabel sweep
was stopped with SIGINT, confirmed exit code 130; its duplicate-run lock is
released. All **937** completed rollout checkpoints remain preserved, including
complete ranks 20–27 and 60, plus 37 seeds of rank 28. Do not resume this sweep.
The historical active-state notes below describe its launch, not its current state.
The reused heartbeat now follows the separate OSQP sweep described in
`docs/linear_rank_lamg0_osqp_sweep_status_2026-09-14.md`.

## Active request and state

The user explicitly requested: **redo the rank experiment using lambda_g=0**.
Run all integer ranks 20–80, each on seeds 211–310, for 6,100 new full 320-step
rollouts. Change only lambda_g from 0.05 to zero. Keep the old experiment and
executed notebook intact, and provide the new executed notebook with a paired
comparison. Do not reuse old lambda_g=0.05 checkpoints as new results.

The new sweep launched successfully at **2026-09-15 02:21 UTC** (September 14
evening in America/Chicago). Six workers, one numerical-library thread each,
rank 60 first and then the other ranks ascending. Rank 60 has completed:
**93/100 strict successes**, median final-100 mean error **2.061569 mm**, P90
**5.415151 mm**, and **zero unsolved QPs**. Rank 20 has started. The remaining
results and notebook execution are still pending.
The preceding full sweep took about 6 hours 22 minutes; do not promise the same
runtime on a shared computer.

## Process and paths

- New result root: `experiments/routed_deepc_iid/linear_subspace_rank_lamg0_100_v1/`.
- Process session: **64744**; parent PID at launch: **35233**. Verify current
  liveness and command identity; do not infer it from an old log.
- Log: `/tmp/atlas_linear_rank_lamg0_sweep.log`.
- `launch.json` records PID, source runner and start time. The launcher holds an
  exclusive `run.lock` across execution to reject a duplicate launcher.
- Launch/resume command from the project root, using the existing environment:

  ```bash
  /Users/kumiko/miniconda3/envs/gym-robotics-mujoco/bin/python -u src/linear_subspace_rank_lamg0_ablation.py --workers 6
  ```

- The launcher executes the new experiment's isolated
  `frozen_protocol/src/linear_subspace_rank_ablation.py`. Do not run the live
  historical rank runner or change any frozen source during the run.
- The previous monitoring automation `finish-atlas-deepc-rank-ablation` is now
  **active** for this new sweep in the same task. Do not create a duplicate monitor.
  Stay quiet during normal progress; notify only on completion, failure,
  meaningful changes, or required user action. Pause it after final delivery.

## Verified protocol

`paired_protocol_audit.json` verifies the only config difference is
`lam_g: [0.05, 0.0]`. Data and atlas hashes match the preceding study. Within the
seven hashed simulation sources, only that baseline default and version/result
identifiers differ. The source snapshot was copied from the verified original
snapshot before modification; data and experiment directories are links, so
datasets are not duplicated. The live baseline and all other studies are untouched.

The real QP/checkpoint integration checks at ranks 20, 60 and 80 passed with
lambda_g=0. They cover unchanged Q/R and remaining penalties, coefficient
dimensions, chart memberships, mean ±3 SD coefficient bounds, full goal policy,
zero affine offsets, no sum(g)=1 constraint, and timing summaries. The failure
cancellation test also passed. Smoke traces were temporary eight-step runs and
are not sweep results.

The manifest was created after all frozen-source changes. Preserve hash checks
and resumability. If the process stops unexpectedly, diagnose first; resume
validated checkpoints via the launcher only after verifying no sweep is running.
Do not retune failures away or remove seeds. The baseline's zero-input behavior
when a QP returns no solution is unchanged. `summary['statuses']` retains aggregate
status counts, including accepted `optimal_inaccurate` solutions; these are
excluded from unsolved-step totals.

## Report and completion

- New notebook:
  `notebooks/ablations/ablation_linear_subspace_rank_lamg0.ipynb`.
- Builder: `src/build_linear_subspace_rank_lamg0_notebook.py`, reusing the original
  report template without writing the previous notebook on import.
- Executor: `src/execute_linear_subspace_rank_lamg0_notebook.py`.
- The report has **six code cells and seven figures**: the same six rank/error/
  timing figures as before, plus `rank_lambda_comparison.png` comparing lambda_g
  zero against 0.05. It contains no old rank-59/60 diagnosis presented as new data.
- All report code compiles. The notebook is not yet executed; it refuses to
  report completion unless all 61 ranks and 6,100 checkpoints are present and valid.
- After the sweep exits successfully, confirm complete=true and the exact grids,
  then run:

  ```bash
  /Users/kumiko/miniconda3/envs/gym-robotics-mujoco/bin/python src/execute_linear_subspace_rank_lamg0_notebook.py
  ```

- Jupyter's local kernel socket may require sandbox escalation. Use the existing
  gym-robotics-mujoco kernel; no package installation or internet is needed.
- Check `notebook_report.json`, `lambda_comparison_report.json` and
  `notebook_execution.json`. All six code cells must execute without errors,
  all seven inline PNGs must be saved, and the notebook must retain
  `RUN_EXPERIMENTS=False`.
- Visually inspect all seven PNGs in the new result root. Report-only fixes are
  allowed; do not modify the frozen simulation sources. Record successful visual
  inspection and file hashes in `notebook_execution.json` after actually checking.
- Report high-success ranks, lower-rank results, rank 60, and the paired change
  from lambda_g=0.05. Keep all seeds and disclose solver-failure handling and
  shared-computer timing limitations. Link the new executed notebook, update this
  status document, and pause the reused heartbeat. Do not archive the task.

The old results remain in `linear_subspace_rank_100_v1/` and the old executed
notebook remains `ablation_linear_subspace_rank.ipynb`. The old source snapshot
and manifest must not be overwritten or repurposed.
