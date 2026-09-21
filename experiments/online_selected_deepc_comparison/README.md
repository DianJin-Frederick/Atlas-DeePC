# Online Trajectory-Selection DeePC Ablation

This is the controlled three-way comparison between graph-routed Atlas-DeePC
and two online trajectory-selection alternatives.

- **Atlas:** localize a graph chart and reuse its cached affine rank-60 basis.
- **Online selection:** rank all 37,800 full trajectory windows by raw L1
  distance to the measured-past/previous-prediction query, retain the top 200,
  and rebuild the same affine rank-60 basis and coefficient bounds online.
- **Raw online selection:** retain the same top 200 columns but use them directly,
  optimizing 200 coefficients with the affine constraint `sum(g)=1`.

Both use the same Reacher-v5 seed, 320-step rollout, offline Hankel data, graph
router and waypoint logic, 10-output representation, `T_ini=2`, `T_f=10`, costs,
constraints, regularization, CLARABEL solver, and one QP per step.

## Seed-1 result

| Method | Coefficients | Median total compute | p95 total compute | Final error | Mean final-100 | Mean prediction error | QP status |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Atlas cached chart | 60 | 26.43 ms | 34.34 ms | 0.79 mm | 0.79 mm | 2.02 mm | 320 optimal |
| Online top-200 to rank 60 | 60 | 49.42 ms | 57.74 ms | 9.89 mm | 5.82 mm | 3.34 mm | 320 optimal |
| Online raw top-200 | 200 | 88.47 ms | 122.21 ms | 284.82 mm | 285.24 mm | 155.01 mm | 320 optimal |

Atlas and rank-matched selection pass the existing strict final-30/final-100
success criteria; raw-200 fails. Rank-matched online selection is 1.87x slower,
primarily because full-library L1 ranking costs a
median 19.98 ms per step; rebuilding the rank-60 basis costs another 1.57 ms.
Atlas localization and cached matrix retrieval together are below 0.04 ms at
their medians. Raw-200 is 3.35x slower: selection costs 20.58 ms and its
200-coefficient QP solve costs 67.38 ms, versus 26.24 ms for Atlas's rank-60 QP.
Its optimal status but poor prediction/control indicate a representation and
regularization issue, not numerical solver failure. This remains a single-seed
controlled ablation, not a success-rate study.

The runner is `src/online_selected_deepc_comparison.py`. Frozen outputs are in
`seed1_n200_rank60/` and the executed project-root comparison notebook includes
the result.
