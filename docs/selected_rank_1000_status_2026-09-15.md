# Selected ranks: 1,000-seed Clarabel evaluation

Status: **complete**. All 3,000 full rollouts finished September 16, 2026 at 11:25:57 UTC (06:25 CDT), approximately 8 hours 28 minutes after launch. The report notebook was executed at 11:37 UTC; four code cells and three figures passed execution and visual checks. Process 90783 exited, and its lock is free.

## Final results

| Rank | Strict success | 95% Wilson interval | Median final-100 error (mm) | P90 error (mm) |
|---|---:|---:|---:|---:|
| 59 | 965/1000 (96.5%) | 95.17–97.47% | 2.016 | 4.434 |
| 62 | 959/1000 (95.9%) | 94.49–96.96% | 1.967 | 4.316 |
| 64 | 952/1000 (95.2%) | 93.69–96.36% | 2.016 | 4.322 |

Rank 59 has the highest observed success and lowest observed computation time. Rank 62 has slightly lower median tracking error. The success differences are 6 and 13 cases out of 1,000 relative to rank 59; do not equate these observed differences with certainty about the population ordering.

Paired outcomes: 59 vs 62 has 947 both-success, 18 only-59, 12 only-62, 23 both-failure. 59 vs 64 has 943 both-success, 22 only-59, 9 only-64, 26 both-failure. 62 vs 64 has 933 both-success, 26 only-62, 19 only-64, 22 both-failure.

All **960,000 QPs were optimal**: 320,000 per rank, zero unsolved and zero optimal_inaccurate. Final invariance checks passed through seed1310. The report revalidated every checkpoint, identical paired targets, and all source/data hashes.

| Rank | Total mean / median / P95 / P99 (ms) | QP mean / median / P95 / P99 (ms) |
|---|---|---|
| 59 | 30.68 / 29.48 / 33.99 / 59.26 | 30.59 / 29.39 / 33.88 / 59.17 |
| 62 | 31.45 / 30.20 / 35.08 / 60.09 | 31.36 / 30.11 / 34.97 / 59.98 |
| 64 | 32.30 / 30.98 / 36.69 / 60.89 | 32.21 / 30.88 / 36.58 / 60.79 |

QP construction and solving account for approximately 99.7% of online computation. Mean one-time goal setup is 4.74, 4.74 and 4.73 ms. Offline matrix/route setup in the actual run is 0.123, 0.119 and 0.125 seconds respectively; full initial loading/setup is 0.444 seconds. Setup measurements from the separate preflight are also preserved, explicitly outside online timing. Timing uses 315,000 steps per rank after excluding the first five steps of each full rollout and remains shared-computer wall-clock timing.

Executed report: [selected-rank notebook](../notebooks/ablations/ablation_selected_ranks_1000.ipynb). `notebook_report.json` and `notebook_execution.json` contain validation results, paired outcomes, and inspected figure hashes. Previous studies remain preserved.

## Protocol

- Ranks **59, 62, 64**, each on the same **1,000 seeds 311–1310**: 3,000 full 320-step rollouts.
- **lambda_g = 0.05, Clarabel**, original max_iter=10000, no solver fallback. All other original baseline parameters unchanged.
- The seed panel is beyond the earlier rank sweep's 211–310 seeds; offline collection seeds are 2026–2225. Do not claim these seeds are unused by every other study in this workspace.
- Exact physical copies of original offline Hankel data and 48-chart atlas; all original seven numerical source hashes verified against the completed .05/Clarabel manifest. Also copied and hashed the transitive `local_affine_atlas_experiment.py` dependency from that snapshot.
- Data SHA256: `75c6e2e8376c00ac1fc032ecaaa31b9a3eb830b671f68a18e559b22f9263c45b`.
- Atlas SHA256: `b8641b1c376a2f4af5dcf9415a5d591a36e99bb223b1c0f7c17390de16803007`.
- Data and atlas arrays are read-only. All three offline matrix caches and routing tables are built once, frozen, and reused. Identical chart memberships and graph routes, exact nested basis prefixes, zero offsets, no sum-g constraint. Coefficient bounds follow the same original mean ± 3 SD rule, with prefixes checked numerically.
- Source/data/atlas/cache hashes checked at startup, every 50 seed blocks, and completion. No online fitting or data regeneration.
- One rollout at a time, one numerical thread. Rotate rank order per seed: 59/62/64, 62/64/59, 64/59/62. Each rank gets a discarded 10-step seed-210 warmup. Actual warmup solve calls checked to use Clarabel and .05.
- Strict success: BOTH last-30 and last-100 windows have mean error <10 mm and maximum <20 mm. Unsolved QPs use the unchanged zero-action behavior and remain in all outcome/timing summaries.
- Every checkpoint retains distance and six timing arrays for all 320 steps. Report stage and total mean/median/P95/P99 after excluding first five steps (315,000 timed steps/rank). QP time includes construction + solve. One-time goal setup, offline setup, and rollout wall times are separate. Shared-computer wall-clock measurements.

## Files and process

Result directory: `experiments/routed_deepc_iid/selected_ranks_1000_clarabel_v1/`.

- Launcher: `src/selected_rank_1000_clarabel.py` (exclusive `run.lock`, inherited across exec).
- Frozen runner: result directory's `frozen_protocol/src/selected_rank_1000_experiment.py`.
- Python: `/Users/kumiko/miniconda3/envs/gym-robotics-mujoco/bin/python`.
- Initial PID **90783**, terminal session **91531**. `launch.json` is authoritative after any resume. Check actual liveness and lock; never infer it from a log alone.
- Log: `run.log`; paired progress: `progress.json`; per-trial wall time: `rollout_wall_times.jsonl`.
- `manifest.json`, `offline_invariance.json`, `invariance_latest.json`, `fairness_audit.json`, `preflight.json`, `setup_*.json`, `session_*.json` record provenance/validation.
- Checkpoints: `rank059/seed311.npz` etc.; atomic validated writes. On resume, existing compatible checkpoints are validated and reused.
- Initial preflight passed; full seed-311 rollouts validated for all three ranks, mismatched checkpoint version rejected, duplicate-run lock verified held, original .05/Clarabel notebook hash unchanged.
- Initial throughput roughly 10–11 seconds per rollout, giving approximately 9 hours total; actual runtime varies.

## Completion follow-through

Reuse heartbeat `finish-atlas-deepc-rank-ablation`, every 10 minutes, quiet during normal progress. Preserve every older study and all other tasks. Never resume the superseded lambda_g=0/Clarabel run or rerun the completed OSQP sweep.

1. Verify actual process liveness, lock, and fresh checkpoint/progress changes. Diagnose a stopped process before resuming via the locked launcher. Do not bypass hashes or alter the frozen numerical protocol. Progress only updates after three paired rollouts; use the log/checkpoint timestamps too.
2. Wait for `summary.json` with complete=true, ranks [59,62,64], seeds311–1310, rollouts3000; all three rank summaries and 3,000 checkpoints. If interrupted after checkpoints but before summary, resume the launcher to validate cached trials and finish aggregation. Do not drop failures or retune.
3. Execute `/Users/kumiko/miniconda3/envs/gym-robotics-mujoco/bin/python src/build_selected_rank_1000_notebook.py --execute`. Jupyter sockets may require sandbox escalation. Builder is report-only and may be fixed if necessary; do not change frozen sources. It verifies all checkpoints and paired targets and writes `notebook_report.json`.
4. Validate `notebooks/ablations/ablation_selected_ranks_1000.ipynb`: four executed code cells, no errors, three inline PNG figures, RUN_EXPERIMENTS=False. Completed September 16, 2026.
5. Visually inspect all three PNGs under result `figures/`, fix report layout if needed, then record inspection completion and figure SHA256s in `notebook_execution.json`. Verify final invariance audit through seed1310 and source/data hashes.
6. Summarize success counts/rates, uncertainty and paired outcomes, tracking error, solver status totals, mean/median/P95/P99 total and QP computation time, separate setup time. Do not infer a causal solver comparison from this experiment; all three use the same solver/regularization.
7. Update this status document and `notebooks/README.md`, open/link the executed notebook, deliver results, and PAUSE the heartbeat. Preserve the existing primary .05/Clarabel rank-sweep notebook and alternate solver notebooks.
