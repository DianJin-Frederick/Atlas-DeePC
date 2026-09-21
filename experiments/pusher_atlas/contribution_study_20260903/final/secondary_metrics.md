# Complete secondary metrics

100 paired seeds per row; 24,000 control samples per row. All failures retained.

| Condition | Method | Success % [Wilson 95% CI] | Mean final cm | Mean capped settle s | Mean effort | RMS action increment | Joint-excursion samples | Recovered / eligible |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| disturbance | A_policy | 37 [28.2, 46.8] | 11.44 | 10.956 | 208.71 | 2.976 | 6322 / 24000 | 21 / 65 |
| disturbance | B_predictive | 4 [1.6, 9.8] | 26.19 | 11.825 | 175.28 | 2.792 | 5888 / 24000 | 1 / 9 |
| disturbance | C_combined | 13 [7.8, 21.0] | 13.84 | 11.541 | 180.58 | 2.807 | 5435 / 24000 | 3 / 65 |
| disturbance | D_selection | 15 [9.3, 23.3] | 16.31 | 11.453 | 168.83 | 2.710 | 4211 / 24000 | 4 / 65 |
| disturbance | E_regularization | 17 [10.9, 25.5] | 17.52 | 11.430 | 177.58 | 2.804 | 6388 / 24000 | 8 / 60 |
| friction | A_policy | 66 [56.3, 74.5] | 9.25 | 8.072 | 214.00 | 3.015 | 5123 / 24000 | 0 / 0 |
| friction | B_predictive | 13 [7.8, 21.0] | 23.19 | 11.167 | 177.47 | 2.807 | 5336 / 24000 | 0 / 0 |
| friction | C_combined | 60 [50.2, 69.1] | 9.92 | 8.437 | 190.05 | 2.889 | 5331 / 24000 | 0 / 0 |
| friction | D_selection | 66 [56.3, 74.5] | 10.81 | 7.804 | 184.96 | 2.864 | 4025 / 24000 | 0 / 0 |
| friction | E_regularization | 62 [52.2, 70.9] | 13.03 | 8.267 | 187.74 | 2.891 | 5915 / 24000 | 0 / 0 |
| mass | A_policy | 0 [0.0, 3.7] | 34.70 | 12.000 | 224.85 | 2.970 | 7083 / 24000 | 0 / 0 |
| mass | B_predictive | 7 [3.4, 13.7] | 29.56 | 11.618 | 163.30 | 2.652 | 9196 / 24000 | 0 / 0 |
| mass | C_combined | 0 [0.0, 3.7] | 34.38 | 12.000 | 171.77 | 2.547 | 9323 / 24000 | 0 / 0 |
| mass | D_selection | 6 [2.8, 12.5] | 34.19 | 11.542 | 138.56 | 2.246 | 9870 / 24000 | 0 / 0 |
| mass | E_regularization | 1 [0.2, 5.4] | 36.03 | 11.902 | 158.24 | 2.526 | 10025 / 24000 | 0 / 0 |
| nominal | A_policy | 66 [56.3, 74.5] | 8.88 | 8.067 | 214.48 | 3.015 | 5174 / 24000 | 0 / 0 |
| nominal | B_predictive | 13 [7.8, 21.0] | 23.49 | 11.222 | 177.39 | 2.808 | 5580 / 24000 | 0 / 0 |
| nominal | C_combined | 54 [44.3, 63.4] | 10.24 | 8.894 | 192.11 | 2.898 | 5252 / 24000 | 0 / 0 |
| nominal | D_selection | 70 [60.4, 78.1] | 10.76 | 7.637 | 186.26 | 2.875 | 3625 / 24000 | 0 / 0 |
| nominal | E_regularization | 66 [56.3, 74.5] | 13.57 | 8.101 | 186.48 | 2.876 | 6466 / 24000 | 0 / 0 |

Effort = dt × sum of squared applied torques; not energy. RMS increments include the change from zero initial control. Capped settling assigns failures 12 s; it is not an uncensored arrival-time mean. Recovery denominators are selected pre-disturbance goal-hold cases displaced outside the goal.

## Computation and solver diagnostics

Steady timing excludes the first 20 calls and each chart's first call. Times in ms; no simulation/rendering included.

| Condition | Method | Median / p95 / p99 | All-call p95 | Steady >50 ms % | Inaccurate / retries / fallbacks | Raw QP overshoot samples |
|---|---|---:|---:|---:|---:|---:|
| disturbance | A_policy | 0.19 / 0.33 / 0.51 | 0.33 | 0.00 | 0 / 0 / 0 | 0 |
| disturbance | B_predictive | 16.51 / 64.00 / 105.89 | 83.00 | 8.79 | 4 / 0 / 0 | 11897 |
| disturbance | C_combined | 15.73 / 65.67 / 131.91 | 102.91 | 8.29 | 15 / 1 / 0 | 6764 |
| disturbance | D_selection | 13.62 / 66.18 / 143.05 | 103.61 | 8.59 | 11 / 0 / 0 | 9834 |
| disturbance | E_regularization | 16.98 / 68.92 / 130.26 | 102.63 | 8.80 | 12 / 0 / 0 | 6361 |
| friction | A_policy | 0.19 / 0.33 / 0.64 | 0.34 | 0.00 | 0 / 0 / 0 | 0 |
| friction | B_predictive | 16.76 / 71.58 / 123.31 | 90.29 | 10.44 | 4 / 0 / 0 | 11445 |
| friction | C_combined | 15.21 / 62.83 / 125.87 | 96.92 | 7.46 | 15 / 1 / 0 | 6752 |
| friction | D_selection | 12.68 / 54.47 / 112.78 | 88.31 | 5.87 | 11 / 0 / 0 | 10809 |
| friction | E_regularization | 14.94 / 54.01 / 117.48 | 89.37 | 5.70 | 12 / 0 / 0 | 6708 |
| mass | A_policy | 0.19 / 0.31 / 0.55 | 0.32 | 0.00 | 0 / 0 / 0 | 0 |
| mass | B_predictive | 15.22 / 79.77 / 154.28 | 110.80 | 12.21 | 4 / 0 / 0 | 10006 |
| mass | C_combined | 14.68 / 83.27 / 157.64 | 114.09 | 12.35 | 15 / 1 / 0 | 5667 |
| mass | D_selection | 15.23 / 78.07 / 158.09 | 113.71 | 11.24 | 11 / 0 / 0 | 8733 |
| mass | E_regularization | 14.73 / 80.69 / 164.12 | 118.43 | 10.76 | 12 / 0 / 0 | 4679 |
| nominal | A_policy | 0.19 / 0.31 / 0.55 | 0.32 | 0.00 | 0 / 0 / 0 | 0 |
| nominal | B_predictive | 16.00 / 68.86 / 114.11 | 85.01 | 9.40 | 4 / 0 / 0 | 11537 |
| nominal | C_combined | 15.28 / 61.04 / 121.75 | 97.10 | 7.32 | 15 / 1 / 0 | 7067 |
| nominal | D_selection | 12.85 / 53.91 / 112.89 | 84.51 | 5.73 | 11 / 0 / 0 | 10768 |
| nominal | E_regularization | 15.02 / 56.25 / 126.91 | 95.16 | 6.00 | 12 / 0 / 0 | 6551 |

All applied torques respected ±2 within tolerance. Raw overshoots are before the common clipping stage; they are not applied-actuator violations.

## Paired success comparisons

C minus comparator, percentage points; pointwise 10,000-resample seed bootstrap intervals. Only C versus A/B belongs to the eight-test Holm family.

| Condition : comparison | Difference pp [95% CI] | C-only / comparator-only successes | Exact McNemar p | Holm p (primary only) |
|---|---:|---:|---:|---:|
| disturbance:C_minus_A_policy | -24 [-34, -14] | 3 / 27 | 8.43033e-06 | 5.0582e-05 |
| disturbance:C_minus_B_predictive | +9 [+2, +17] | 12 / 3 | 0.0351562 | 0.115837 |
| disturbance:C_minus_D_selection | -2 [-9, +5] | 6 / 8 | 0.790527 | exploratory |
| disturbance:C_minus_E_regularization | -4 [-12, +4] | 7 / 11 | 0.480682 | exploratory |
| friction:C_minus_A_policy | -6 [-16, +4] | 10 / 16 | 0.32694 | 0.653879 |
| friction:C_minus_B_predictive | +47 [+34, +59] | 54 / 7 | 4.32211e-10 | 3.45769e-09 |
| friction:C_minus_D_selection | -6 [-16, +4] | 11 / 17 | 0.344928 | exploratory |
| friction:C_minus_E_regularization | -2 [-13, +9] | 15 / 17 | 0.86005 | exploratory |
| mass:C_minus_A_policy | +0 [+0, +0] | 0 / 0 | 1 | 1 |
| mass:C_minus_B_predictive | -7 [-13, -2] | 0 / 7 | 0.015625 | 0.078125 |
| mass:C_minus_D_selection | -6 [-11, -2] | 0 / 6 | 0.03125 | exploratory |
| mass:C_minus_E_regularization | -1 [-3, +0] | 0 / 1 | 1 | exploratory |
| nominal:C_minus_A_policy | -12 [-22, -2] | 7 / 19 | 0.0289593 | 0.115837 |
| nominal:C_minus_B_predictive | +41 [+29, +53] | 48 / 7 | 1.30876e-08 | 9.16131e-08 |
| nominal:C_minus_D_selection | -16 [-27, -5] | 8 / 24 | 0.00700037 | exploratory |
| nominal:C_minus_E_regularization | -12 [-23, -1] | 9 / 21 | 0.0427739 | exploratory |

A degenerate [0, 0] empirical bootstrap interval with no discordances does not establish population equivalence. Secondary continuous-outcome paired estimates and confidence intervals are in `summary.json`; additional timing/recovery/resource details are in `detailed_metrics.json`.
