# F1TENTH paper comparison across measurement noise levels

The paper table compares one tuned Atlas-DeePC configuration with Contextual-DPC using 30, 60 and 90 columns at each noise level. Every row uses 20 fresh paired online seeds, 531–550. Atlas K and rank d are selected separately at each noise level using development seeds 511–530, then frozen before any paper evaluation. Controller equations and weights stay fixed.

[Paper table](../../experiments/f1tenth/paper_nsr_20261027/report.md) · [LaTeX table](../../experiments/f1tenth/paper_nsr_20261027/table.tex) · [Notebook](../../notebooks/f1tenth/baselines/baseline_f1tenth_atlas_deepc.ipynb)

## Applied-input update

Current code supplies **applied inputs** to both Atlas-DeePC and Contextual-DPC histories and offline libraries. Acceleration is the RK4-weighted applied derivative, averaged over the ten internal 0.01 s steps; steering is the held angle. The simulator's limiting is state dependent, so using only the limiter value at the start of the interval would be insufficient. This uses simulator actuator telemetry, not noisy output differencing. The interval average gives the exact speed increment, but holding that average need not reproduce every nonlinear lateral state transition.

`U` / `U_applied` and `input_history` contain applied inputs. `U_commanded`, `command_history`, `warmup_commanded_inputs`, and internal applied samples are saved separately. Collision replay always uses commands to reproduce the original plant trajectory. The physical simulator, speed limit, objective and chart-selection rule are unchanged.

The new clean dataset is `offline_300_seconds_applied_v1.npz`; noisy versions also contain `applied_v1` in their names. They retain the original states/outputs and reconstruct applied inputs by exact command replay. The original offline data and historical results are preserved. **The retained paper table and earlier successful longer-history example used commanded inputs and have not been recomputed under this convention.** Current reproduction commands evaluate the frozen configurations with applied inputs and save a separate table.

## Noise and shared data

For outputs x, y, speed and yaw, `y_meas,j = y_true,j + sigma * s_j * epsilon_j`, with independent standard-normal epsilon. The fixed clean-offline channel standard deviations are approximately (15.2318 m, 16.1983 m, 1.63264 m/s, 1.59418 rad).

The levels are sigma=0,0.001,0.0025,0.005,0.01,0.025,0.05: 0%, 0.1%, 0.25%, 0.5%, 1%, 2.5% and 5%. NSR here is a **standard-deviation ratio**, not a power ratio; the noise/signal power ratio is sigma squared. The 10% level is excluded.

Each level has its own noisy offline dataset. Both methods receive the same file, with matching hashes. Noise is added once per raw output sample before alignment, clustering, SVD and contextual normalization; overlapping windows reuse the same measurement. Inputs and true states remain unchanged. The same physical noise standard deviations apply online. Offline seed 3408 is independent of online seeds; the same standardized offline draws are scaled across levels. There is one offline realization per level, so this is not a multiple-training-dataset robustness study.

The local contextual-DPC reconstruction supplies 300 seconds in 20 random-excitation episodes and 2,700 overlapping windows. These are not the authors' recorded data. The clean source and all seven noisy versions are retained as reproduction dependencies.

## Fixed task and controllers

São Paulo official varying-speed raceline, single-track dynamics with ideal steering-angle actuation, RK4 at 0.01 s, control interval 0.1 s. T_ini=5, T_f=10. Acceleration and steering bounds are 3 m/s² and 0.4189 rad. Initial position/yaw perturbations remain 0.05 m and 0.6 degrees, including at zero measurement noise.

Both controllers share a 15-step noisy reference-following initializer; it is never used as a fallback after handoff. Initialization is excluded from error and timing statistics but included in collision checks.

Atlas clusters noisy offline speeds and forms uncentered SVD chart bases. Online selection projects the latest five measured input/output pairs onto each chart's past image, using matching pose alignment. The three exact-zero anchor coordinates are removed for decomposition and restored as zeros. Rank cutoff is 1e-10; numerical ties select the smallest chart ID. Past images can span the entire 27-dimensional aligned history space, causing tied residuals and constant chart selection. The saved diagnostics must be consulted before claiming adaptive switching benefits.

Atlas minimizes `||Y_f g-y_ref||_Q² + ||U_f g||_R² + 100 ||Y_p g-y_ini||²`, with exact past inputs and future input bounds. Q=diag(100,100,1,1), R=diag(0.1,10). There is no g regularization, coefficient bound, affine constraint or fallback.

Contextual-DPC retains 30/60/90 columns, the same physical Q/R and measurements, its own standardized coordinates and input-increment penalty, an affine constraint, unsquared L2 coefficient penalty lambda_g=10 and L1 history slack lambda_sigma=1. The controller objectives are not identical. Both use single-threaded Clarabel.

## Selection, evaluation and retained files

The development target is Contextual-90's collision-free count and median successful-lap RMSE within 20%. Existing good settings are retained; deficient levels receive a broader feasible K/rank screen and local refinement. This target does not determine or filter the fresh paper outcomes. Failed levels remain in the table. When no lap succeeds, successful-lap RMSE is undefined and displayed as a dash. Atlas is tuned separately by noise level; this is not a robustness curve for one fixed Atlas configuration.

Success requires a full geometric lap and no simulator collision at any 0.01 s substep, including initialization. A geometric lap with a collision is not a success. RMSE uses true positions and is summarized over successful laps. Controller time includes reference construction, selection, assembly, solver setup and solve, excluding the first five calls. Timing includes failed prefixes and should be interpreted with the success counts.

Canonical results are in `experiments/f1tenth/paper_nsr_20261027/`: the table, 560 raw rollout records, frozen configuration choices, source snapshots, dataset hashes, independently verified collision replay and projection diagnostics. Compact selection ledgers preserve the search scope and outcomes. Superseded raw experiments and search trajectories are removed only after the retained table is verified; the cleanup manifest records their identities and hashes.

Independent validation reconstructs offline and online noise, verifies clustering on noisy data, checks projection decisions, paired warmup, constraints and replay-state agreement.

## Reproduction

To evaluate the frozen paper configurations with applied inputs in a fresh folder:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/run_f1tenth_paper.py \
  --output experiments/f1tenth/paper_applied_input_rerun
```

The notebook reads the retained table by default. Set `RUN_ALL_EXPERIMENTS=True` to reevaluate the 560 paper settings with applied inputs in its designated rerun folder. Reusing a folder with a different protocol is rejected. Reacher code and results are outside this cleanup.
