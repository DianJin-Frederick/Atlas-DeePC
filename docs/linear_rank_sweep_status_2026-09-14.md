# Linear-subspace rank sweep: live status and completion instructions

Updated 2026-09-14. This supersedes the rank-task status in
`ablation_handoff_2026-09-14.md` and the pasted conversation handoff.

## Current state

**Complete and verified.** All 61 ranks (20–80), each on seeds 211–310, finished:
6,100 full 320-step rollouts. The six-worker process exited with code 0 after
about 6 hours 22 minutes. At the 11:55 UTC heartbeat, the report was executed
successfully with the existing Jupyter kernel, all 6,100 checkpoints were
validated, and all six generated PNG figures were visually inspected.

The delivered notebook is
`notebooks/ablations/ablation_linear_subspace_rank.ipynb`: five executed code
cells, six saved inline figures, no cell errors, `RUN_EXPERIMENTS=False`.
`notebook_report.json` contains the complete rank summaries, paired seed changes,
and total online timing table. `notebook_execution.json` records execution,
successful visual inspection, and notebook/figure hashes. No new runs are needed.

| Rank | Strict successes /100 | Median final-100 mean error (mm) | P90 (mm) |
| --- | ---: | ---: | ---: |
| 59 | 97 | 2.045 | 4.431 |
| 60 (baseline) | 93 | 2.062 | 5.426 |
| 62 | 97 | 1.911 | 4.358 |
| 80 | 83 | 2.327 | 19.126 |

Ranks 59 and 62 tied for the best observed success count. Every rank 20–33 had
zero strict successes; rank 34 had 34/100. All QPs were solved at ranks 34–80.
Low-rank unsolved-QP totals and the unchanged zero-input failure handling are
explicitly disclosed in the notebook. These are reused exploratory seeds, so
the selected-rank differences are not a held-out generalization claim.

Median/P99 total online times were 39.540/80.842 ms at rank 59,
50.366/157.383 ms at rank 60, and 39.987/83.025 ms at rank 62. Each rank pools
31,500 steps after discarding five steps per rollout. Shared system load and the
fixed rank order confound timing comparisons; do not claim an isolated speedup.
Compact checkpoints occupy about 73.9 MiB in total.

- Rank 60: **93/100 strict successes**, median final-100 mean error 2.062307 mm,
  P90 5.426007 mm, zero unsolved QPs.
- All 100 rank-60 final-100 mean errors and final-30/final-100 success flags match
  the preceding fresh ten-output baseline exactly. Saved comparison:
  `experiments/routed_deepc_iid/linear_subspace_rank_100_v1/baseline_reproduction.json`.
- Initial observed throughput was roughly 100 rollouts in five minutes, suggesting
  about five hours for the full run, subject to system load and rank dependence.
- Low-rank runs have emitted CVXPY `optimal_inaccurate` warnings. Preserve results
  and report recorded solver outcomes honestly; do not retune or discard seeds.
- Ranks 20 and 21 each have 0/100 strict successes and at least one unsolved QP
  in every rollout: 6,822/32,000 and 6,697/32,000 unsolved steps respectively.
  Median final-100 mean errors are 148.714 and 141.086 mm. This is a result of the
  requested ablation, not a reason to stop the sweep or change parameters.
- All ranks 20–33 have 0/100 strict successes. Rank 33 has only nine unsolved QPs
  in 32,000 steps. Rank 34 is the first successful rank in ascending order:
  34/100 strict successes, with all 32,000 QPs solved. Its 100 compact checkpoints
  and aggregate metrics were independently revalidated at the 07:28 UTC heartbeat.
- Confirmed the unchanged lower-level baseline applies zero input when no QP
  solution is returned (`routed_deepc_experiment.py`, around line 614). These are
  not QP-generated actions. The report now explicitly discloses this and plots
  unsolved-step rates and rollouts with any unsolved QP. Accepted
  `optimal_inaccurate` solutions are not counted as unsolved. Compact checkpoints
  retain aggregate status counts in `summary['statuses']`, but discard individual
  per-step status arrays. Those aggregate counts include accepted inaccurate
  solutions; they do not by themselves identify the numerical failure mechanism.

## Process and monitor

**Shared baseline changed after launch.** At the 07:28 UTC heartbeat, validation
against live `src/` detected a changed `baseline_atlas_deepc.py`: the new protocol
is `linear_iterative_n10_lamg0_v2`, with lambda_g=0 and a reference-mode option.
The running workers and parent retain their originally imported lambda_g=0.05
code. Do not stop or duplicate the run, change its algorithm, or revert the newer
live baseline. This sweep remains the original lambda_g=0.05 experiment.

The original baseline source was reconstructed in an isolated source snapshot
and verified byte-for-byte against its launch SHA256. **All seven manifest source
hashes match exactly**, and `prepare()` passes without bypassing any hash checks.
Snapshot: `experiments/routed_deepc_iid/linear_subspace_rank_100_v1/frozen_protocol/src/`.
All local Python sources were copied (small files); data and experiment paths are
links to existing directories, so no datasets were duplicated. Provenance is in
`source_snapshot_provenance.json`. The notebook now imports the frozen sources.

If a restart is actually required after confirming the original process stopped,
use the existing Python with
`experiments/routed_deepc_iid/linear_subspace_rank_100_v1/frozen_protocol/src/linear_subspace_rank_ablation.py --workers 6`
from the workspace root. The original live-src command below is launch history,
**not the valid resume command now**. Preserve the original manifest unchanged.

- Parent PID at launch: **90720**. Verify command identity before relying on a PID.
- Tool process session: **71850**.
- Log: `/tmp/atlas_linear_rank_sweep.log`.
- Exact launch command, from repository root:
  `/Users/kumiko/miniconda3/envs/gym-robotics-mujoco/bin/python -u src/linear_subspace_rank_ablation.py --workers 6`
- Heartbeat automation: **finish-atlas-deepc-rank-ablation**, attached to this task;
  **paused** after completed execution and visual QA. No further monitoring
  is required for this finished sweep.
- The independent `deepc-benchmark-hourly-progress` automation belongs to another
  study and task. Do not change it.
- Process-list reads needed sandbox escalation; this was allowed by automatic
  review. A current tool-session poll or fresh read-only process check can verify
  liveness. Do not restart just because an old log is quiet.

## Completed preparation

- Reviewed the runner and added bounded job submission and fail-fast cancellation.
  A failed future cannot leave all 6,100 jobs queued; at most active worker jobs
  finish. Checkpoints and summaries are written via atomic replacement.
- Every checkpoint validates baseline configuration, full terminal policy,
  dimensions, finite timing/error arrays, and tail metrics against stored errors.
- Two tests passed in `src/test_linear_subspace_rank_ablation.py`: actual low,
  default and high-rank QPs/checkpoints and a simulated worker failure. The QP
  checks cover ranks 20/60/80, actual coefficient and prediction dimensions,
  unchanged Q/R/penalties, zero offset/no sum constraint, identical chart membership,
  coordinate mean ±3 SD bounds, and timing summaries. Smoke trajectories are
  temporary eight-step runs; they are not counted as sweep results.
- Manifest hashes were created after runner edits. Do not edit any hashed runner,
  baseline, or timing source while the sweep is running. Never bypass a mismatch.
- New notebook and builder exist, syntax and notebook format validated:
  `notebooks/ablations/ablation_linear_subspace_rank.ipynb` and
  `src/build_linear_subspace_rank_notebook.py`. **The notebook is now fully executed.**
- Notebook index fixed. The old contextual-affine notebook remains archived.
- Compact checkpoint size is about 12 KB per rollout, without videos, trajectory
  duplication, or new environment/package installations.

## Completion procedure (successfully carried out)

1. Confirm root `summary.json` has `complete=true`, exactly all 61 ranks 20–80,
   and seeds 211–310. Do not infer completion from a partial summary.
2. Run the prepared notebook executor with the existing environment:

   ```bash
   /Users/kumiko/miniconda3/envs/gym-robotics-mujoco/bin/python src/execute_linear_subspace_rank_notebook.py
   ```

   Local Jupyter kernel sockets may need sandbox escalation. No internet,
   packages, or new environment are required. The executor refuses to launch
   simulations and only saves the notebook after all five code cells execute
   successfully and all six inline PNG figures and 6,100 records are verified.
   The notebook uses root finding and `RUN_EXPERIMENTS=False`.
3. Fix report-only problems if execution fails. Do not rebuild over successfully
   executed results unless needed for a report change. The notebook builder and
   executor are not part of the frozen simulation manifest.
4. Visually inspect all six PNGs in the result directory: `rank_success_errors.png`,
   `rank_seed_heatmaps.png`, `rank_error_distributions.png`, `rank_paired_changes.png`,
   `rank_module_timings.png`, and `rank_total_timings.png`. Repair layout as needed.
5. Update `notebook_execution.json` to record successful visual inspection after
   actually inspecting the figures. Read `notebook_report.json` for final findings.
6. Report results and link the executed notebook in this task. Distinguish reused
   exploratory seeds from held-out evaluation and shared-computer timings from an
   isolated benchmark. No Wilson intervals. Update this document and pause the
   `finish-atlas-deepc-rank-ablation` heartbeat; do not archive the task.

The original handoff contains the complete baseline specification and other
completed ablation results. Only retained linear-subspace rank varies in this run.
