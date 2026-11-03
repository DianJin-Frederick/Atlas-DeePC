# F1TENTH paper comparison across measurement NSR

Twenty fresh, paired online seeds (531–550) per setting. One Atlas (K,d) is selected per noise level using development seeds 511–530 and frozen before these evaluations. The offline realization is shared between methods at each level; noise is added before clustering. NSR denotes a standard-deviation ratio, not a power ratio.

| NSR (%) | Method | Collision-free laps | Geometric laps | Median successful RMSE (cm) | Median controller time (ms) |
|--:|:--|--:|--:|--:|--:|
| 0 | Atlas-DeePC (K=2, d=42) | 20/20 | 20/20 | 1.75 | 2.45 |
| 0 | Contextual-DPC (30 columns) | 20/20 | 20/20 | 5.20 | 3.87 |
| 0 | Contextual-DPC (60 columns) | 20/20 | 20/20 | 2.61 | 9.76 |
| 0 | Contextual-DPC (90 columns) | 20/20 | 20/20 | 1.94 | 13.70 |
| 0.1 | Atlas-DeePC (K=12, d=40) | 20/20 | 20/20 | 1.74 | 2.43 |
| 0.1 | Contextual-DPC (30 columns) | 20/20 | 20/20 | 4.87 | 3.88 |
| 0.1 | Contextual-DPC (60 columns) | 20/20 | 20/20 | 2.68 | 9.96 |
| 0.1 | Contextual-DPC (90 columns) | 20/20 | 20/20 | 2.06 | 13.76 |
| 0.25 | Atlas-DeePC (K=13, d=42) | 20/20 | 20/20 | 3.41 | 2.71 |
| 0.25 | Contextual-DPC (30 columns) | 20/20 | 20/20 | 6.10 | 3.90 |
| 0.25 | Contextual-DPC (60 columns) | 20/20 | 20/20 | 3.81 | 9.83 |
| 0.25 | Contextual-DPC (90 columns) | 20/20 | 20/20 | 3.03 | 13.68 |
| 0.5 | Atlas-DeePC (K=14, d=60) | 18/20 | 20/20 | 5.35 | 4.36 |
| 0.5 | Contextual-DPC (30 columns) | 7/20 | 20/20 | 9.79 | 4.00 |
| 0.5 | Contextual-DPC (60 columns) | 20/20 | 20/20 | 5.81 | 9.93 |
| 0.5 | Contextual-DPC (90 columns) | 20/20 | 20/20 | 4.67 | 13.62 |
| 1 | Atlas-DeePC (K=10, d=44) | 12/20 | 20/20 | 8.80 | 3.23 |
| 1 | Contextual-DPC (30 columns) | 0/20 | 4/20 | — | 3.95 |
| 1 | Contextual-DPC (60 columns) | 7/20 | 20/20 | 10.70 | 9.59 |
| 1 | Contextual-DPC (90 columns) | 11/20 | 20/20 | 8.14 | 13.34 |
| 2.5 | Atlas-DeePC (K=3, d=68) | 0/20 | 12/20 | — | 3.62 |
| 2.5 | Contextual-DPC (30 columns) | 0/20 | 0/20 | — | 3.86 |
| 2.5 | Contextual-DPC (60 columns) | 0/20 | 20/20 | — | 9.87 |
| 2.5 | Contextual-DPC (90 columns) | 0/20 | 20/20 | — | 13.51 |
| 5 | Atlas-DeePC (K=8, d=72) | 0/20 | 7/20 | — | 3.79 |
| 5 | Contextual-DPC (30 columns) | 0/20 | 0/20 | — | 4.04 |
| 5 | Contextual-DPC (60 columns) | 0/20 | 3/20 | — | 9.98 |
| 5 | Contextual-DPC (90 columns) | 0/20 | 11/20 | — | 13.57 |

RMSE uses collision-free completed laps only; a dash means none succeeded. Timing includes all controller calls after the first five, including failed prefixes. Geometric lap completion does not imply collision-free driving. Warmup is included in collision checks.

Only K and d are tuned for Atlas; its QP, physical weights, partial-history projection rule and tie handling remain fixed. Contextual-DPC retains 30/60/90 columns and its own objective terms. The methods use identical noisy offline observations, physical noise standard deviations and paired online seeds; their objectives are not identical.

The development target is Contextual-90 success count and RMSE within 20%; it is not imposed on the fresh paper results. Failed levels remain visible. When the contextual baseline has zero successes, successful-lap error comparability is undefined.

Projection residuals may tie because chart past images span the aligned history space. Consult selection_diagnostics.json before interpreting successes as an adaptive switching benefit. There is one offline noise realization per level, not multiple independent training datasets.

Observed selector behavior in this retained table: every nonzero-noise Atlas row has tied residuals at every decision and uses chart zero through the fixed tie rule. The zero-noise case has no ties, but also no chart switches. These results establish tracking performance, not an adaptive-switching benefit.
