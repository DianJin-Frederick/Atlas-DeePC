# Matched simple-robot Atlas-DeePC pilots

This directory contains a paired 10-seed pilot of baseline Atlas-DeePC and
triggered contextual Atlas-DeePC on four `dm_control` tasks. The implementation
is `src/simple_robot_atlas_benchmark.py`.

## Frozen protocol

- 200 offline episodes with 200 steps each, seeds beginning at 2026.
- Select-DPC IID uniform input process `(a, b) = (0, 1)`.
- No stabilized offline episode is retained.
- `T_ini = 2`, `T_f = 10`, 48 physical-output charts, and rank cap 60.
- 128 cached contextual prototypes with 200-column supports.
- Observed chart-transition graph and chart-medoid waypoint routing.
- Contextual mode activates only after a causal 15-step progress test.
- Evaluation seeds 501 through 510; no dynamics model, IK, or feedback fallback.

## Results

| Task | Atlas strict success | Contextual Atlas strict success | Atlas median tail error | Contextual median tail error | Median QP time, Atlas / contextual |
|---|---:|---:|---:|---:|---:|
| PointMass Easy | 1/10 | 1/10 | 0.1185 m | 0.1330 m | 38.51 / 39.98 ms |
| Cart-pole Swing-up | 0/10 | 0/10 | 2.1279 composite | 2.0103 composite | 39.98 / 40.30 ms |
| Finger Turn Easy | 2/10 | 2/10 | 0.1437 m | 0.1314 m | 31.31 / 32.49 ms |
| Ball-in-Cup Catch | 0/10 | 0/10 | 0.2070 m | 0.0993 m | 33.32 / 33.87 ms |

The contextual basis was used for 35.6%, 93.0%, 74.0%, and 88.2% of control
steps, respectively.

## Coverage audit

| Task | Best instantaneous offline error | Offline samples below strict threshold |
|---|---:|---:|
| PointMass Easy | 0.03168 m | 0 |
| Cart-pole Swing-up | 1.95067 composite | 0 |
| Finger Turn Easy | 0.00008 m | 1,166 |
| Ball-in-Cup Catch | 0.01824 m | 29 |

The PointMass success starts 0.255 m from the target and reaches a minimum
error of 0.000036 m, so it is a genuine controlled approach rather than a
near-goal reset. Cart-pole cannot be interpreted as a controller comparison:
the IID library contains no near-upright behavior. Ball-in-cup contains a few
instantaneous near-catches but no stabilized catch episode; contextual charts
roughly halve median tail error without completing a strict catch. Finger has
target-region coverage, but both methods succeed on only two targets, exposing
a contact-mode and routing limitation.

These are diagnostic pilots, not statistically powered benchmark claims.
