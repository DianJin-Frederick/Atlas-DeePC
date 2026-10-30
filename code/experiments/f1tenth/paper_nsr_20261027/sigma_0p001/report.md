# F1TENTH experimental past-history projection selection

| Method | Collision-free laps | Geometric laps | Median RMSE (cm) | Median time (ms) | P99 time (ms) |
|:---|---:|---:|---:|---:|---:|
| Atlas-DeePC (K=12, d=40) | 20/20 | 20/20 | 1.74 | 2.43 | 14.49 |
| Contextual-DPC (30 columns) | 20/20 | 20/20 | 4.87 | 3.88 | 11.82 |
| Contextual-DPC (60 columns) | 20/20 | 20/20 | 2.68 | 9.96 | 42.64 |
| Contextual-DPC (90 columns) | 20/20 | 20/20 | 2.06 | 13.76 | 47.62 |

RMSE is the median of per-rollout cross-track RMSE over collision-free completed laps. The methods share offline data, official varying-speed reference, evaluation seeds, noise and physical tracking/input weights. Their additional objective terms differ. Contextual DPC is our local reconstruction, not the authors’ released data or reported paper results.

Only configurations with at least one collision-free completed lap are displayed. All evaluated settings, including zero-success settings, remain in report.json and the raw records. Ranks exceeding the smallest chart membership are skipped and recorded in protocol.json.

Atlas comparisons keep training windows, Q/R, lambda_ini=100, solver tolerances and every task setting fixed. Chart memberships and centroids change with K; each rank within a K takes a prefix of the same uncentered SVD bases. No K/rank-specific tuning is performed. More rank changes model flexibility, not the amount of collected data. Timing pools all runs, including failed prefixes, so partial-success settings are not matched full-lap timing comparisons. These common seeds have been used in earlier exploratory comparisons.

T_ini=5, T_f=10, dt=0.1 s. Physical Q=diag(100,100,1,1), R=diag(0.1,10). Atlas uses lambda_ini=100, offline raw speed-based K-means charts and uncentered SVD, online past-window projection-residual selection, with no g regularization or coefficient bounds. Contextual uses the displayed column count, fixed effective lambda_g=10 with the unsquared L2 norm, lambda_sigma=1 with L1 history slack, affine and incremental-input constraints, and Clarabel. Larger column counts are not separately retuned.

Timing includes reference construction, selection, assembly, setup and solve, with the first five calls omitted. Offline preparation, simulation, logging and collision replay are excluded. A collision-free full lap requires state-verified simulator collision replay at every 0.01 s substep.

All results use projection-residual Atlas selection. Both controllers receive identical 15-step reference-following warmup histories; the initializer is never used after handoff. Warmup is excluded from error/timing metrics but included in collision replay. Source/data hashes, selection windows and all residuals are saved for verification.

Latest T_ini observed inputs/outputs, aligned to latest output pose; Euclidean projection onto the past rows [Up; Yp] of each unchanged full chart basis. Remove exact zero anchor x/y/yaw rows before SVD and embed the orthonormal basis back into 30 rows with exact zero anchor rows, for selection only. Fixed relative rank tolerance 1e-10. Explicit residual; smallest chart ID breaks numerical ties. Same 15-step initializer as full-window baseline. Full bases and QP unchanged.
Atlas uses past-history projection. At high ranks, restricted charts can span the entire aligned history space and cannot be distinguished by this residual; numerical ties select the lowest chart ID. Consult selection_diagnostics.json for measured chart use.

Noise: sigma=0.001 is the per-channel noise/signal standard-deviation ratio. Identical physical measurement standard deviations offline and online: [0.01523181116746967, 0.016198279997774737, 0.0016326426790134138, 0.0015941760591237071] for x/y/speed/yaw. Scales are fixed from clean offline global outputs; offline noise seed 3408. Noise is added before alignment, clustering and window extraction.
