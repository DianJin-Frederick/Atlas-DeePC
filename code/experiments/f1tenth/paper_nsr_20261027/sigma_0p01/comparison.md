# Retained partial-history selection experiment

Latest five observed input/output pairs, aligned to their final pose, projected onto the past rows of each full chart basis. Fixed 15-step warmup, data, noise, reference, QP and weights across configurations.

| K | d | Collision-free laps | Median RMSE (cm) |
|---:|---:|---:|---:|
| 10 | 44 | 12/20 | 8.80 |

Rows with zero successes are omitted here and retained in selection_diagnostics.json/report.json. RMSE includes only successful laps; these reused seeds are exploratory.

Numerics: exact zero anchor x/y/yaw rows are removed before decomposition and restored as exact zeros in the bases. Restricted-basis singular values use a fixed relative cutoff of 1e-10. The QP objective and constraints are unchanged.

All 80 rollouts verified. Selection diagnostics include numerical past ranks, chart use and tie rates.

The aligned five-sample query has 30 entries, with final x, y and yaw fixed to zero. A past-image rank of 27 therefore spans the entire admissible query space: an essentially zero residual is not evidence of predictive accuracy. Ties are resolved by smallest chart ID, without a centroid or controller fallback.
