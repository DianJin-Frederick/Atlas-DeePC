# Behavioral atlas for the fixed nonlinear example

Keep the system `y[k+1] = y[k]^2 + u[k]` and the selected offline family
`w = (u0, u1, y0, y1) = (a, -a, x, x^2+a)` in R⁴. The input restriction
`u1 = -u0` is retained; collecting arbitrary input pairs would produce a
different dataset, for which the current exact three-coordinate view does
not apply.

## What K-means and chart fitting produce

1. Generate 1,200 independent full Hankel columns with `x` uniform on `[-1,1]`
   and `a` uniform on `[-0.5,0.5]` (seed 29).
2. Run Euclidean K-means with `K=8` on all four coordinates, without coordinate
   standardization. K-means produces groups and centers, not predictive
   subspaces by itself.
3. In each group, subtract its center and retain the leading two right singular
   vectors as an orthonormal local basis.

For group j, the illustrative affine chart is

\[
\mathcal A_j=\{c_j+B_jg:g\in\mathbb R^2\},\qquad
c_j\in\mathbb R^4,\quad B_j\in\mathbb R^{4\times2}.
\]

The associated coordinates are `g = B_j.T @ (w-c_j)`, and the reconstruction
is `c_j + B_j @ g`. The atlas consists of the groups together with these local
representations. If used as affine behavioral columns, the center/affine
constraint must be retained; this is not equivalent to an uncentered rank-two
linear basis.

The exact isometry `z = Q.T @ w` from the surface example is used only for
display. Each center and basis is mapped by the same Q. K-means squared
distances and local residual norms are preserved. The left panel shows
colored complete windows; the right panel shows bounded portions of the
fitted planes, using convex hulls of their projected training samples.
The polygons are approximate chart footprints, not exact curved manifold
regions or invariant sets. Dots inside the patches mark their centers.

## What the atlas looks like

It is a collection of planar patches following the parabolic cylinder.
Their slopes change with the initial output x. K-means also separates windows
along the input direction a because it minimizes distances to centers;
it does not directly minimize local-plane reconstruction error. Group
boundaries are learned Voronoi regions, not manually prescribed operating
intervals. Neither overlaps of displayed patches nor neighboring clusters
establish a dynamically feasible transition graph.

Every group in this dataset still has exact uncentered rank three. If those
full linear spans were retained, each group would reproduce the same
containing subspace `u0+u1=0`. The local structure illustrated here comes from
**centered, rank-two affine approximation**, not exact linear spans.

## Held-out geometric diagnostic

All errors use 500 independent complete test windows in R⁴ and the same
nearest-center assignment. The test windows, including their future entries,
are used for geometric assignment; this is not online causal selection.

| Number of charts K | Full-window RMS reconstruction error |
|---:|---:|
| 1 | 0.226819 |
| 4 | 0.047679 |
| 8 | 0.019160 |
| 16 | 0.013118 |

These are results of this seeded dataset, not universal guarantees. More
charts add parameters and leave fewer observations per chart. The example
does not yet implement a DeePC optimization, contextual reference selection,
or graph routing.

## Outputs

- [Two-panel overview PDF](figures/io_behavioral_atlas.pdf) and
  [PNG](figures/io_behavioral_atlas.png).
- Single-column, title-free [cluster figure](figures/io_atlas_clusters.pdf)
  and [affine-chart figure](figures/io_atlas_charts.pdf).
- [Full atlas data and bases](figures/io_behavioral_atlas.npz) and
  [metrics](figures/io_behavioral_atlas_metrics.json).

All figures use Times New Roman. The separate panels are 3.5 inches wide.
Reproduce with `python notebooks/motivation/io_behavioral_atlas.py` from the
project root, or run the atlas section of `input_output_trajectory_surface.ipynb`.

Suggested caption: *An illustrative behavioral atlas for the nonlinear system
\(y_{k+1}=y_k^2+u_k\). (a) K-means partitions complete input–output windows
\(w=(u_0,u_1,y_0,y_1)\), selected with \(u_1=-u_0\), into eight groups.
(b) Centered rank-two SVD supplies a local affine chart for each group.
Both panels use the same exact orthonormal coordinates of the full vectors;
matching colors identify corresponding groups and charts. The planar patches
approximate the curved trajectory family.*
