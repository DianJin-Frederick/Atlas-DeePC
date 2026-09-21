# Prediction-bound experiment audit and resume wording

Date: 2026-09-03. Source archive: `/Users/kumiko/Projects/prediction_error_bound.zip`.
The source archive and original notebook were not modified.

## Bottom line

There are numerical data behind the paper figure. They support a reduction in
**normalized bound slack**, not a reduction in prediction MSE. A separate,
assumption-screened one-step experiment also finds a **91.1844% median reduction
in raw bound magnitude**. This result uses the **true subspace distance** and
must not be described as an improvement in a fully data-computable certificate.

The recommended resume entry is in `resume_entry.tex`. Its numerical bullet
describes the NEW one-step experiment, not the exact experiment in the paper.
For a theory paper, the proofs are the primary contributions; the percentage
is optional supporting evidence, not a necessary justification of the project.

## Sources reviewed

- Archived `arxiv_submission_v2/v2.tex` and `main.pdf`, especially Theorem 2 and
  the numerical experiment. Relevant PDF pages 5-6 were rendered and inspected.
- [Author manuscript, v3](https://arxiv.org/html/2603.17256v3).
- [Kaviani, Markovsky and Ossareh, Theorem 2 / equation (16)](https://arxiv.org/html/2404.15098v1).
  The three terms in the notebook implementation match this public formula.
- `bound_comparison.ipynb`, `core/common.py`, `core/subspace.py`.
- `results/data/paper_fig4_theorem2_vs_our_bound_with_theorem9.csv`.

The supplied IEEE stamp URL could not be opened. The archived manuscript and
public author version were used instead; this is not a certification that the
final IEEE typeset version is identical.

## 1. What the saved plot data say

The CSV contains 50,000 system/noise rows: 1,000 generated systems and 50 noise
levels from 1e-8 to 1e-3. Each pair attempts 100 noise realizations. After the
notebook's signal-to-noise filter, 25,609 pairs, 513 systems, and 2,560,296
realizations contribute to the two main bound comparisons.

For each method, the stored metric is:

`maximum_over_accepted_noise_draws [100 * (bound - observed_error) / output_norm]`.

The maxima for the two methods can occur on different draws. Raw per-draw
bounds and errors are NOT retained in this CSV.

- At each noise level, compare the median stored gap across systems. The
  reduction `100 * (1 - median_ours / median_baseline)` ranges from **82.9120%
  to 84.1595%**; the median of those 50 reductions is **83.6103%**.
- Alternatively, take the median of the 25,609 paired gap reductions: **82.6087%**.
  These aggregation rules are different; do not interchange their numbers.
- The oracle-distance gap is smaller in all 25,609 saved paired rows.
- The fully computable plug-in variant has a **71.7939x median gap ratio**
  versus the baseline on 19,317 common finite rows, and is never smaller there.
  Thus the favorable percentage cannot be attributed to that variant.

`audit.py` independently replays the original first 3 generated systems,
including 15,000 attempted and 5,000 accepted noise draws. All three saved gap
columns and both acceptance-count columns match within rtol=0.002, atol=1e-9.
This is a sample replay, not a rerun of all five million attempted scenarios.

## 2. Issues to resolve before reusing the original figure as a bound claim

1. Theorem 2 bounds the FIRST future output; its coefficient uses `Y_f[:p, :]`.
   The notebook subtracts the error norm of the ENTIRE future trajectory and
   normalizes by the full future output norm. These are different quantities
   for `T_f > 1` (16,564 saved accepted system/noise pairs).
2. The theorem assumes `T_p >= n` and a local chordal-distance condition.
   The notebook filters on the baseline's `delta_sn > 0.6`, not those conditions.
   There are 1,442 saved accepted pairs with `T_p < n`. This does not prove a
   violation of the numerical bound, but they fall outside the theorem as stated.
3. The theorem assumes a feasible, noise-free initial trajectory. The original
   notebook independently perturbs the online initial output and passes that
   same perturbed history into both the clean and estimated predictors. Thus
   the measured error is not automatically the theorem's stated setting.
4. The paper describes per-time Euclidean noise norm at most N. Coordinatewise
   uniform noise in the notebook can exceed that norm for multiple outputs.
5. Even with matching assumptions, an oracle-distance bound and a computable
   bound use different information. A smaller oracle bound does not establish
   superiority of a deployable uncertainty certificate.

These are experimental interpretation/implementation concerns, not a claim
that the theoretical proofs are incorrect.

## 3. New one-step experiment

`one_step_check.py` uses the original generator and unchanged bound formula
implementations, with the following pre-specified structural restrictions:

- Generate all original 1,000 systems with seed 240415098.
- Select ALL systems with `T_f=1`, `T_p>=n`, and scalar output `p=1`: 226 systems.
  Selection is independent of measured performance.
- Use exact, feasible online initial histories. Perturb only the offline output.
- Use all 50 original noise levels and 20 fresh realizations each, seed 20260903.
- Retain the notebook's `delta_sn>0.6` filter and explicitly check
  `kappa <= sigma_min(M_hat)/(2*sqrt(2))` and positive `sigma_min`.
- Save both raw bound values, their corresponding observed errors, and
  conditioning quantities for EVERY accepted draw.

Results:

| Quantity | Result |
| --- | ---: |
| Attempted draws | 226,000 |
| Rejected by baseline signal-to-noise threshold | 89,140 |
| Additional local-distance-condition rejections | 0 |
| Accepted draws | 136,860 |
| Accepted system/noise pairs | 6,843 |
| Systems with accepted draws | 137 |
| Median paired raw-bound reduction | 91.1844% |
| Interquartile range of paired reductions | 85.6029%-94.8365% |
| Minimum / maximum paired reduction | 24.6303% / 99.9489% |
| Fraction with smaller oracle bound | 100% |
| Empirical bound coverage, both methods | 100% |

The primary raw metric is exactly
`median_over_accepted_draws [100 * (1 - oracle_bound / baseline_bound)]`.
It is NOT `100 * (1 - median(oracle_bound) / median(baseline_bound))`.
Repeated noise draws are not independent systems; report the 137-system scope.
The coverage check uses a 1e-12 numerical tolerance; it is an empirical check,
not a new proof or a calibrated confidence level.

Important limitations: this is scalar-output, one-step prediction for small
stable linear systems with exact online histories. It uses the TRUE subspace
distance. The baseline is kept exactly as published/implemented and retains
its allowance for online history noise even though this experiment uses exact
history. Consequently, this is a scoped comparison of certificate magnitudes,
not an equal-information, best-possible comparison of deployable certificates.
Do not claim a universal 91.2% improvement or improvement in MSE.

## Reproduce

Requires Python, NumPy, and pandas. No original notebook output cells are run.

```bash
python audit.py /Users/kumiko/Projects/prediction_error_bound.zip --replay-systems 3
python one_step_check.py /Users/kumiko/Projects/prediction_error_bound.zip
```

Outputs are written beside the scripts. `summary.json` describes the saved-data
audit; `one_step_summary.json` describes the new experiment. Raw accepted draws
are in `one_step_raw.csv.gz`, with pair and noise-level summaries in separate
CSVs. The source CSV SHA-256 is
`b48aa04a8aab8610ef8601cbbb090dc049106486b520c47d8865121942b559eb`.

## Wording to avoid

- "Reduced prediction error/MSE by 91%."
- "Developed a fully computable bound 91% tighter than existing methods."
- "Guaranteed 91% tighter bounds for dynamical systems."
- "Validated on five million systems."

If the oracle qualification distracts from your theory contribution, replace
the numeric resume bullet with:

> Evaluated prediction-error certificates across 50 noise levels, distinguishing
> the tightness of oracle-distance bounds from the conservatism introduced by
> fully data-computable uncertainty estimates.
