# Seven chart-construction methods, 100-seed Clarabel experiment

## Selected-method notebook revision

At the user's request, the active `ablation_clustering_methods_500.ipynb` now presents only spectral clustering and UMAP + K-means, using the existing 200 rollouts. Its report and execution audit live in `spectral_umap_report/` under the result root. Each method has a separate 6×8 physical robot-arm pose grid following the baseline notebook: 32 deterministic member samples per chart, bold saved medoids, common spatial limits, and full support counts. PNG/PDF figures and sample indices are saved with the selected report. The previous executed seven-method notebook, builder and report audits are preserved in `seven_method_report_archive/`; original numerical artifacts and seven-method figures are unchanged. The original completion notes below describe that archived full comparison.

Status: **complete**. All seven methods completed all100 seeds (700 full rollouts) at **2026-09-16 19:53:51 UTC**, approximately2h46m after launch. Process14639 exited and the duplicate-run lock is free. User requested replacement of `notebooks/ablations/ablation_clustering_methods_500.ipynb` according to the supplied specification. Despite the historical filename, this is a **100-seed** experiment, 211–310. The old notebook was first copied byte-for-byte to `archive/notebooks/ablation_clustering_methods_500_before_20260916.ipynb`. Older experiments and baseline notebooks are preserved.

## Final comparison

User-selected alternatives for the subsequent ablation study: **UMAP + K-means** and **spectral clustering**, alongside the baseline. Preserve their fitted models, original-column memberships, atlases and controller caches for reuse without refitting. Spectral clustering is graph-based clustering; UMAP + K-means combines nonlinear dimensionality reduction with clustering. Neither should be described as affine subspace clustering. The full seven-method exploratory results remain preserved.

| Method | Success | Median final-100 error (mm) | Mean online time (ms) | Median (ms) | P99 (ms) |
|---|---:|---:|---:|---:|---:|
| K-means | 93/100 | 2.062 | 36.49 | 33.12 | 104.83 |
| Affine K-subspaces | 87/100 | 2.140 | 37.35 | 33.64 | 117.65 |
| Mixture PPCA | 88/100 | 2.432 | 37.98 | 33.74 | 118.23 |
| Spectral | 90/100 | 2.063 | 37.47 | 34.56 | 100.63 |
| Diffusion maps | 77/100 | 2.568 | 38.45 | 34.86 | 103.97 |
| Isomap | 67/100 | 2.826 | 39.35 | 36.25 | 103.60 |
| UMAP | 92/100 | 2.089 | 39.74 | 35.91 | 104.97 |

Every method has **32,000 optimal QPs**: 224,000 total, zero optimal_inaccurate, unsolved QPs, or trial exceptions. All methods retain48 charts,365–2122 columns/chart globally, rank60 controller bases, full numerical behavioral rank144, and one strongly connected directed transition component. All data, source and model hashes validated; all original37,800 columns retained.

Fresh K-means reproduces **all100 original rank60 distance arrays bit-for-bit**, with zero success flips and maximum difference0 (`baseline_reproduction.json`). Timing is freshly measured, not historical. Mean localization times (ms), in table order: .0712,.0712,.0705,.8449,.8381,1.9469,2.8441. Each method has31,500 timed online steps after the first5 steps per rollout are excluded; full320-step timing arrays are preserved.

Paired baseline-only / alternative-only successes: K-subspaces10/4, MPPCA9/4, spectral6/3, diffusion21/5, Isomap30/4, UMAP4/3. K-means has the highest observed success and lowest mean/median total time. The one-case gap against UMAP is not reliable evidence of a population advantage. Diffusion/Isomap performed worse with these fixed settings; do not generalize to all settings or all algorithm variants. The notebook includes Wilson intervals, paired errors and unadjusted exact McNemar results.

The notebook is populated with five executed code cells, six figures and an observed-comparison section. Numerical tests and full-checkpoint/hash validation passed. Figure inspection and hashes are recorded in `notebook_execution.json`. Fitting limits, landmarks, reused exploratory seeds and shared-computer wall-clock caveats are explicit.

## Exact protocol

- Baseline: `notebooks/baselines/baseline_graph_routed_atlas_deepc.ipynb`, waypoint-enabled, iterative N=10 references, Clarabel, **lambda_g=.05 explicitly**, 48 charts, rank60 linear uncentered behavioral bases, 320 steps. Config equality is asserted against the original .05/Clarabel rank-study manifest; the live newer lambda_g=0 default is overridden only in this isolated experiment.
- One immutable `data/select_iid_offline_hankel.npy`, shape144×37800, SHA256 `75c6e2e8376c00ac1fc032ecaaa31b9a3eb830b671f68a18e559b22f9263c45b`. All columns retained without reordering or recollection. No duplicated Hankel datasets per method.
- Six periodic latest-Yp configuration features, exactly baseline standardized cos(q1),cos(q2),sin(q1),sin(q2),cos(q1+q2),sin(q1+q2). Four input-history coordinates have zero weight. Shared feature arrays/preprocessing and landmark indices saved in `common_features.npz`.
- Fresh K-means fitting verified equal to the original baseline atlas; all seven methods get fresh timing/outcomes. No historical timing reuse.
- K-means; affine K-subspaces (2D planes,50 iterations); MPPCA (rank2,100 EM steps,tol1e-5,variance floor1e-5); sparse normalized spectral (48D); diffusion maps (4D,alpha1,time1); sklearn Isomap(4D); native UMAP(4D,min_dist.1,200epochs). Full settings and citations are in the notebook, `src/clustering_methods_models.py`, and immutable experiment manifest.
- Four embedding methods share 1500 offline landmarks sampled uniformly without replacement, seed1701. All original columns get native-geometry assignments. Spectral and diffusion use a sparse20-neighbor RBF graph and Nyström extension; Isomap uses its native transform; UMAP uses fixed-seed singleton transforms offline AND online to avoid batch dependence. Exact landmarks retain their saved training coordinates. Embedding-space K-means uses10 initializations,seed11,max300. No nearest-physical-centroid substitution.
- Medoids/transition graph rebuilt from original full labels using baseline physical-feature RMS distances, within-episode transition counts, threshold3, cost distance/sqrt(count). No embedding-space route costs. Behavioral SVD bases/bounds use the unchanged baseline routine; no nonlinear decoder. Supports must be>=61 and numerical behavioral ranks>=60; otherwise record method failure rather than reducing rank.
- One rollout worker, one numerical thread, rotate methods within each seed. Discarded warmup seed210/10steps per method. All320 per-step timing arrays saved; aggregate first5steps excluded. Offline fit/atlas construction and basis-cache time saved separately; online localization includes extension, goal selection, routing, matrix retrieval, QP construction/solve and total saved. Zero-action handling for unsolved QPs unchanged. Unreachable-goal exceptions reported explicitly as failed trials; no replacement routing/controller.
- Strict success: BOTH final30 and final100 mean<10mm,max<20mm. This is an exploratory reused seed panel, not untouched validation. No evaluation-based tuning.

## Fitting and preflight results

All seven atlases fitted, have48 charts, all supports>=365, numerical rank>=60, and one strongly connected directed transition component. Native assignment repeated/reloaded checks passed for fixed offline points. Four numerical tests passed: feature definition, memberships/cache reconstruction, PPCA likelihood against SciPy multivariate normal, and spectral/diffusion eigenfunction equations.

| Method | Min–max support | Fit + atlas seconds | Basis-cache seconds | Fit caveat |
|---|---|---:|---:|---|
| K-means | 558–1292 | .16 | .161 | Reproduces original atlas |
| K-subspaces | 520–1293 | 11.47 | .122 | Reached50 iterations without fixed assignments |
| MPPCA | 485–1088 | 13.05 | .131 | Reached100 iterations before tolerance |
| Spectral | 466–1387 | 1.11 | .153 | Landmark approximation |
| Diffusion | 521–2122 | .39 | .136 | Landmark approximation |
| Isomap | 365–1928 | 3.29 | .127 | Landmark approximation |
| UMAP | 423–1516 | 102.05 | .124 | Landmark approximation; native singleton transforms |

Fit times are exact values in each method's `artifact_manifest.json`; rounded here. These are end-to-end fit/atlas preparation timers, including full-column assignments and early artifact writes for alternatives, and a baseline reproduction check for K-means. They are not pure optimizer times; later artifact writes and numerical-rank diagnostics are outside the reported timers. UMAP includes initial compilation overhead. Fixed iteration caps are disclosed, not claimed converged. Do not retune based on outcomes. Primary-source references are embedded in the notebook.

## Artifacts and running process

Result root: `experiments/routed_deepc_iid/clustering_methods_100_clarabel_v1`.

- Specification copy: `user_specification.txt`.
- Runner: `src/clustering_methods_experiment.py`; models: `src/clustering_methods_models.py`; tests: `src/test_clustering_methods_experiment.py`.
- Seven baseline dependency files physically frozen under `frozen_protocol/src`; versioned UMAP0.5.9.post2 and pynndescent0.5.13 installed with no shared-environment changes under `dependencies/`.
- Each method has memberships, native gate.joblib, fitted parameters/embeddings, atlas, medoids, graph, controller cache, bases/singular values/bounds/supports, health and artifact manifest. Main manifest hashes sources, packages, data/config. Artifacts reload without reclustering.
- `--fit-only` completed successfully in terminal session17641. Numerical tests: terminal20283, exit0, four tests passed.
- Completed evaluation terminal session **91323**, PID14639. PID and launch UTC are preserved in `launch.json`.
- `run.lock` is exclusive; `run.log` logs each trial. `progress.json` updates after each seven-method seed block. Failures have explicit JSON/traceback, including per-seed failures. Rollouts are atomic validated full `method/seed211.npz` etc.
- Invocation: `/Users/kumiko/miniconda3/envs/gym-robotics-mujoco/bin/python -u src/clustering_methods_experiment.py` (redirect to a fresh resume log if needed). Never overwrite the original log when resuming.
- `summary.json` complete=true means all methods/seeds were attempted. **Check `all_methods_succeeded`, `offline_failures`, per-method `complete`, and trial failures**: attempted is not equivalent to700 valid rollouts.

## Completion instructions

Reuse the existing heartbeat `finish-atlas-deepc-rank-ablation` for this new task; the selected-rank1000 study is already finished and must not rerun. Stay quiet during ordinary progress.

1. Check actual process liveness from current launch.json, exclusive lock and fresh progress/checkpoints. Resume only if stopped after diagnosis. Source/provenance mismatches must be diagnosed, never bypassed. Preserve all artifacts and other tasks. Do not change frozen numerical code or hyperparameters after evaluation begins.
2. On completion, verify seven×100 paired seeds211–310 or report precise method/trial failures. Compare fresh K-means distance arrays/outcomes against original rank60/100-seed study if possible to confirm baseline reproduction; timing must remain fresh. Save a reproduction audit. Check all hashes and no dropped columns.
3. Execute `/Users/kumiko/miniconda3/envs/gym-robotics-mujoco/bin/python src/build_clustering_methods_notebook.py --execute`. Local Jupyter sockets may require escalation. This rewrites the requested old-name notebook with the full new report. Report-only fixes are allowed; preserve archived old notebook. It must have five executed code cells, no errors, six inline PNG figures, RUN_EXPERIMENTS=False.
4. Inspect all six PNGs under result `figures/`, fix report layout if necessary, and record successful visual inspection + figure hashes in `notebook_execution.json`. Verify `notebook_report.json`, all per-method provenance, success/error/timing and connectivity/status totals. Include caps/nonconvergence, landmark and exploratory-panel caveats in analysis.
5. Update this document and notebook index to complete, link/open the populated notebook, deliver a concise seven-method success/timing comparison and caveats. Pause the heartbeat; do not archive this task or modify other automations.
