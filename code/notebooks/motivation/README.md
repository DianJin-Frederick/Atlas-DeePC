# Paper motivation examples

Keep small illustrative examples for the paper introduction in this folder.

## Autonomous trajectory surface (simpler example)

Open [autonomous_trajectory_surface.ipynb](autonomous_trajectory_surface.ipynb).
The autonomous recurrence `y[k+2] = (y[k] + y[k+1]^2)/2` has scalar output
and two state coordinates. With `L = 3`, each plotted point is a complete
trajectory `(y0, y1, y2)`, directly in three-dimensional trajectory space.
Varying `(y0, y1)` over `[-1, 1]^2` gives the exact curved surface
`y2 = (y0 + y1^2)/2`. No coordinate projection is used. The paper figure
shows the shorter patch `y0 in [-0.6, 0.6]`, `y1 in [-1, 1]`, with the
original camera and axis proportions retained.

- Single-column, Times New Roman figure: [PDF](figures/autonomous_trajectory_paper.pdf),
  [SVG](figures/autonomous_trajectory_paper.svg), [PNG](figures/autonomous_trajectory_paper.png).
- Concise [LaTeX example](autonomous_trajectory_example.tex).
- [K-means and linear subspaces](autonomous_behavioral_atlas.ipynb): 12,000 complete
  trajectories, six groups, and uncentered rank-two SVD fits. Includes a
  [report and recorded plane equations](autonomous_behavioral_atlas.md),
  [overview](figures/autonomous_kmeans_overview.pdf), and
  [individual subspace views](figures/autonomous_kmeans_subspaces.pdf).
  A clearer [three-group version](figures/autonomous_kmeans_k3_overview.pdf)
  uses the same data; [its planes](figures/autonomous_kmeans_k3_subspaces.pdf)
  are also shown separately.
- [Trajectory data](figures/autonomous_trajectory_windows.npz) and
  [Hankel-column checks](figures/autonomous_trajectory_metrics.json).
- Reproduce with `python notebooks/motivation/autonomous_trajectory_surface.py`.
  Requires NumPy and Matplotlib, with Times New Roman installed.

## Controlled input–output trajectory surface

Use [input_output_trajectory_surface.ipynb](input_output_trajectory_surface.ipynb).
Every point represents one full input–output window of length
`L = T_ini + T_f = 2`, with `m = p = 1`, in four-dimensional trajectory space:
`w = (u0, u1, y0, y1)`. The nonlinear plant `y[k+1] = y[k]^2 + u[k]`, with
selected input pairs `u1 = -u0`, produces a curved two-dimensional family
inside a three-dimensional linear subspace. The figure uses an orthonormal
basis of that subspace, preserving complete vectors and Euclidean distances.
Its axes are basis coordinates `(z1, z2, z3)`, and the surface is a parabolic
cylinder. The notebook verifies actual Hankel-column construction, dynamics,
full-window linear reconstruction, distances, and full-space local affine fits.

- Paper figure: [PDF](figures/io_trajectory_paper.pdf),
  [SVG](figures/io_trajectory_paper.svg), [PNG](figures/io_trajectory_paper.png).
- Corrected [introduction paragraph and caption](io_trajectory_introduction.md)
  and [LaTeX snippet](io_trajectory_introduction.tex).
- [Full window data](figures/io_trajectory_windows.npz) and
  [numerical checks](figures/io_trajectory_metrics.json).
- [K-means behavioral atlas](io_behavioral_atlas.md): eight learned groups
  and their rank-two affine charts, fitted in the full R⁴ trajectory space.
  [Overview PDF](figures/io_behavioral_atlas.pdf),
  [cluster panel](figures/io_atlas_clusters.pdf), and
  [chart panel](figures/io_atlas_charts.pdf). Reproduce with
  `python notebooks/motivation/io_behavioral_atlas.py`.
- Reproduce with `python notebooks/motivation/io_trajectory_surface.py` from
  the project root, or run the notebook. Requires NumPy, SciPy, scikit-learn,
  Matplotlib, and Times New Roman for the publication figure.

## Nonlinear cylinder (earlier output-space illustration)

The earlier cylinder figure plots individual output samples, not full
input–output windows. Its introduction text and figure should not be used
to claim trajectory-space geometry; use the corrected example above.

Open [nonlinear_cylinder.ipynb](nonlinear_cylinder.ipynb) and run all cells with a
Python kernel containing NumPy, Matplotlib, SciPy, and scikit-learn. The notebook
is self-contained, uses fixed random seeds, and runs from this directory or the
project root. It needs no robot simulator or existing experiment data.

The plant has intrinsic state `(theta, z)` and measured output
`y = (cos(theta), sin(theta), z)`. Its noise-free measured trajectories lie
exactly on a cylinder in three dimensions. The notebook illustrates:

- Learned local affine geometry and one-step prediction on held-out rollouts.
- Intermediate output references from a graph of observed group transitions.
- Two future trajectory groups sharing one measured past, selected using the
  desired reference.

The affine predictor and contextual scores are explanatory surrogates, not an
implementation or performance evaluation of DeePC. Graph edges record observed
adjacency and do not certify reference trackability. Output-surface dimension
must not be confused with the dimension of input-output trajectory windows.

Saved outputs:

- Publication figure: [PDF](figures/cylinder_paper.pdf),
  [SVG](figures/cylinder_paper.svg), and [600 dpi PNG](figures/cylinder_paper.png).
  Uses Times New Roman at single-column size. Reproduce with
  `python notebooks/motivation/cylinder_paper_figure.py` from the project root.
- [Introduction paragraph and caption](cylinder_introduction.md), with a
  [LaTeX snippet](cylinder_introduction.tex).
- [Cylinder surface coverage](figures/cylinder_surface_coverage.png) and
  [vector PDF](figures/cylinder_surface_coverage.pdf): simulated sweeps with a
  cylinder wireframe, top view, and unwrapped angle-height view. These additional
  visualization rollouts do not replace the training or test data.
- [Overview](figures/cylinder_motivation.png) and
  [vector PDF](figures/cylinder_motivation.pdf).
- [Held-out diagnostics](figures/cylinder_diagnostics.png) and
  [vector PDF](figures/cylinder_diagnostics.pdf).
- [Numerical results](figures/cylinder_metrics.json).

The notebook includes a suggested paper caption and numerical checks. Re-running
it replaces these generated figures and results.
