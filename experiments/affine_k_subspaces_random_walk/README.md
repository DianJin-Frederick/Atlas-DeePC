# Random-Walk Affine K-Subspaces Atlas-DeePC

This experiment tests the requested pipeline:

1. Collect frozen Reacher trajectories with the Select-DPC input process
   `u_t = a u_(t-1) + b n_t`, using the random-walk profile `(a, b) = (1, 0.1)`.
2. Cluster length-12 trajectory columns with affine K-subspaces.
3. Recompute a centered SVD basis for every final cluster.
4. Express all affine charts relative to one global offset and concatenate their
   center displacements and SVD directions.
5. Feed that single matrix to the existing DeePC controller.

The controller is intentionally pure DeePC: IK references, feedback blending,
and feedback fallback are disabled.

## Run

Small end-to-end verification:

```bash
conda run -n gym-robotics-mujoco python src/affine_k_subspaces_random_walk.py --smoke --force-collect --force-cluster
```

Full experiment:

```bash
conda run -n gym-robotics-mujoco python src/affine_k_subspaces_random_walk.py --force-collect --force-cluster
```

Full outputs are written under `full/`; smoke-test outputs are written under
`smoke/`. Each directory contains the offline Hankel matrix, clustering and SVD
artifact, configuration, controller rollouts, and aggregate summary.

## Paper Protocol Note

The Select-DPC paper defines IID data as `(a, b) = (0, 1)` and random-walk data
as `(a, b) = (1, 0.1)`. Its reported Reacher experiment uses IID data. This
experiment deliberately uses the requested random-walk profile while retaining
the paper's 200 simulations and 200 steps as the full-run defaults.

## Initial Full Baseline

The first full run completed with:

- Hankel shape: `144 x 37800`
- Affine charts: `8`, with rank `12` per chart
- Concatenated affine matrix: `144 x 104`, numerical rank `103`
- Evaluation: `0/8` successful seeds
- Mean distance over the final 30 steps: `0.066` to `0.184` m
- Solver status: all 320 control steps were optimal for every seed

The affine K-subspaces objective decreased throughout all 20 configured
iterations, although assignments had not fully converged at the iteration cap.
This baseline therefore verifies the requested pipeline but does not yet show
that unrestricted concatenation of all charts is an effective nonlinear
controller. See `full/summary.json` for the complete per-seed results.
