# Ranks 58, 61, 63: 1,000-seed Clarabel extension

Status: **complete and verified**. Completed 2026-09-18 at 01:33:37 UTC; report executed and all three figures visually inspected. The user requested 1,000 full testing seeds **311–1310** for ranks **58, 61, 63**, extending the completed rank 59/62/64 study. Target: 3,000 rollouts, 320 steps each. Do not rerun any older experiment.

## Protocol and invariants

- λg=.05, Clarabel max_iter=10000, same graph-waypoint baseline, iterative N=10 terminal references, same Q/R/constraints and strict success definition (both last30 and last100 mean<10mm and max<20mm).
- One worker, one numerical thread; rotate 58/61/63 order per seed. Warmup seed210,10steps per rank is excluded. Pool timing after excluding first5 steps per rollout; preserve all320 timing/error samples. QP time includes construction and solve.
- Exact copies of the completed `selected_ranks_1000_clarabel_v1/frozen_protocol` data, atlas and eight numerical dependency sources. Only new runner version/rank tuple and maximum-rank reference checks differ. Original seven source hashes are validated against the rank100 manifest; transitive dependency is also hashed.
- Data SHA256 `75c6e2e8376c00ac1fc032ecaaa31b9a3eb830b671f68a18e559b22f9263c45b`; atlas SHA256 `b8641b1c376a2f4af5dcf9415a5d591a36e99bb223b1c0f7c17390de16803007`.
- Read-only data/atlas/cache arrays, fixed memberships and graph routes, nested original linear bases, zero offsets, no sum-g constraint. Bounds retain the original mean±3SD rule. Source/data/atlas/cache checks at startup, every50 paired seeds, and completion.
- Same1,000 seeds as prior selected-rank runs. This extension is not untouched validation. Shared-computer timings come from a new session and are not an isolated causal rank comparison against earlier sessions.

## Process and artifacts

Result root: `experiments/routed_deepc_iid/adjacent_ranks_1000_clarabel_v1`.

- Python: `/Users/kumiko/miniconda3/envs/gym-robotics-mujoco/bin/python`.
- Locked launcher: `src/adjacent_rank_1000_clarabel.py`; actual runner: result root's `frozen_protocol/src/selected_rank_1000_experiment.py`.
- Initial terminal session27715, PID61667, launched2026-09-17T14:21:54UTC. `launch.json` contains authoritative PID/start after any resume. `run.lock` is inherited across exec; require lock/liveness checks before any resume.
- Initial preflight terminal7854 completed successfully: frozen data/charts, nested bases,30 actual Clarabel calls with λg=.05.
- Initial process scan found no other Python rank/benchmark/clustering experiment running.
- `launch_verification.json` confirms the first full seed311 rollout at all three ranks, target equality with previous rank59, unchanged original atlas arrays/eight numerical dependencies, lock held, and unchanged previous notebook hash. The first two paired seeds completed normally with no unsolved QPs.
- Original log `run.log`; paired progress `progress.json`; wall times `rollout_wall_times.jsonl`; atomic checkpoints `rank058/seed311.npz`, etc.
- Report builder `src/build_adjacent_rank_1000_notebook.py`; notebook `notebooks/ablations/ablation_adjacent_ranks_1000.ipynb`. Notebook is now executed: four code cells, three inline figures, no errors, and `RUN_EXPERIMENTS=False`. It includes median/P90/P95/P99 final100 error and all timing modules, paired outcomes, solver totals and three plots.

## Completion follow-through

1. Monitor actual process liveness, exclusive lock, and fresh logs/checkpoints. Do not infer liveness from stale progress. Resume only after diagnosing termination, using the locked launcher and a new resume log; never overwrite original logs or bypass source hashes. Preserve failures and all seeds; no retuning.
2. Require summary complete=true, ranks[58,61,63], seeds311–1310,3,000 valid full rollouts and final invariance audit through1310. Validate all checkpoints and status totals. Compare identical target coordinates against previous ranks59/62/64 for every seed; do not reuse historical timing.
3. Execute `/Users/kumiko/miniconda3/envs/gym-robotics-mujoco/bin/python src/build_adjacent_rank_1000_notebook.py --execute` (Jupyter kernel socket may require escalation). Report-only fixes allowed. Verify four executed code cells, three inline figures, no errors and RUN_EXPERIMENTS=False.
4. Inspect all three result PNGs, record visual inspection and figure hashes in notebook_execution.json. Deliver ranks58/61/63 success counts/rates, median/P90/P95/P99 errors, module median timings and total median/P95/P99, plus solver failures, setup costs and shared-session caveat. Use the same error definition as the user's table: across-seed quantiles of each trial's final100 mean error, including failures.
5. Update this document and notebook index, open/link the completed notebook, then pause heartbeat `finish-atlas-deepc-rank-ablation`. Stay quiet during normal progress; notify only on failure, meaningful change, required user action or completion. Preserve older rank studies, the selected spectral/UMAP notebook, and all other automations.

## Verified final results

All 3,000 checkpoints contain all 320 steps; each rank has exactly seeds 311–1310. All 960,000 QPs are optimal, with zero unsolved QPs. Final offline invariance passed through seed1310. Every seed target agrees exactly across the three new ranks and prior ranks59/62/64. Module and total timing summaries were independently recomputed from the raw checkpoints. The original process exited and its exclusive lock is free; no resume was needed.

| Metric | Rank 58 | Rank 61 | Rank 63 |
|---|---:|---:|---:|
| Successes / 1,000 | 953 | 942 | 957 |
| Success rate | 95.3% | 94.2% | 95.7% |
| Median final100 mean error (mm) | 2.084 | 1.952 | 1.998 |
| P90 error (mm) | 4.519 | 4.895 | 4.455 |
| P95 error (mm) | 8.488 | 11.181 | 8.132 |
| P99 error (mm) | 47.427 | 45.080 | 47.682 |
| Median chart localization (ms) | 0.0599 | 0.0601 | 0.0604 |
| Median terminal-reference selection (ms) | 0.0140 | 0.0140 | 0.0140 |
| Median graph routing (ms) | 0.0055 | 0.0055 | 0.0055 |
| Median subspace retrieval (ms) | 0.0007 | 0.0007 | 0.0007 |
| Median other online work (ms) | 0.0295 | 0.0295 | 0.0297 |
| Median QP construction and solve (ms) | 31.18 | 31.95 | 32.93 |
| Median total online time (ms) | 31.29 | 32.06 | 33.05 |
| P95 total online time (ms) | 54.40 | 54.99 | 56.70 |
| P99 total online time (ms) | 76.52 | 76.88 | 78.89 |

Error quantiles include failures. Timing pools 315,000 steps per rank after excluding each rollout's first five steps. Module medians need not sum to the total median.

Separate setup costs for the actual evaluation session: matrix/route setup 0.135623/0.139054/0.133653 seconds for ranks58/61/63; total data loading plus setup 0.495158 seconds. Median one-time goal setup per rollout 4.948/4.952/4.941 ms, excluded from online step timings. The earlier preflight setup is recorded separately in its original setup JSON.

This was a new session on a shared computer; timing differences from older rank studies cannot be attributed solely to rank. Older results and notebooks remain unchanged.

Audit artifacts under the result root: `completion_audit.json` (all 3,000 checkpoint hashes, raw timing verification, target pairing, solver totals); `notebook_execution.json` (notebook and figure hashes, four executed cells, three inspected figures); `notebook_report.json` (results, paired outcomes, timing summaries).
