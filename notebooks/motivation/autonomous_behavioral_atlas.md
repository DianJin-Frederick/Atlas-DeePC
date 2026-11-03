# K-means on autonomous trajectory data

The system is $y_{k+2}=(y_k+y_{k+1}^2)/2$. Each data vector is a complete
length-three trajectory $w=(y_0,y_1,y_2)$, obtained as a depth-three Hankel
column from an independently simulated rollout. We sample 12,000 training
initial conditions and 3,000 held-out initial conditions uniformly over
$y_0\in[-0.6,0.6]$, $y_1\in[-1,1]$, matching the shortened paper surface.
This is a selected patch of the behavior, not exhaustive data from every initial condition.
Independent short rollouts give coverage without the concentration near equilibria
that can arise from pooling long autonomous rollouts.

Euclidean K-means ($K=6$, 30 initializations, seed 20260929) operates directly on
all three trajectory coordinates, with no standardization or dimensionality reduction.
K-means returns labels and centroids, not linear subspaces. For each group, form
$W_j=[w^{(1)}\ \cdots\ w^{(N_j)}]$ and retain the first two left singular vectors
$U_j$ **without subtracting the mean**. The recorded linear subspace is
$\mathcal S_j=\operatorname{im}U_j=\{w:n_j^T w=0\}$, with projector $P_j=U_jU_j^T$.
It minimizes $\sum_{w\in C_j}\|w-P_jw\|_2^2$ among rank-two orthogonal projectors.

All six clusters have exact linear rank three: their exact spans are all
$\mathbb R^3$. The displayed planes are rank-two approximations. Every plane
passes through the origin; it is not a translated local tangent plane.
The K-means centroids are stored for assigning new trajectories, and are not
used as offsets in the linear models.

## Results

Held-out trajectories are assigned to the nearest training centroid.
The RMS Euclidean distance per complete trajectory is **0.03696** for the
six-plane model and **0.19450** for a single global rank-two linear plane.
These are projection/reconstruction errors, not future-output prediction or
closed-loop control errors. This run illustrates geometry, not selection of an optimal K.

| Group | Training windows | Plane equation ($n_j^T w=0$) | Test RMS |
|---|---:|---|---:|
| 1 | 1882 | $-0.3109y_0 +0.3899y_1 +0.8668y_2=0$ | 0.0439 |
| 2 | 1953 | $-0.4962y_0 +0.3442y_1 +0.7971y_2=0$ | 0.0452 |
| 3 | 2050 | $-0.4049y_0 +0.0036y_1 +0.9144y_2=0$ | 0.0183 |
| 4 | 2277 | $-0.4866y_0 -0.0214y_1 +0.8734y_2=0$ | 0.0211 |
| 5 | 1994 | $-0.4970y_0 -0.3440y_1 +0.7966y_2=0$ | 0.0449 |
| 6 | 1844 | $-0.3094y_0 -0.3941y_1 +0.8654y_2=0$ | 0.0407 |

## Reading the figures

- [Overview PDF](figures/autonomous_kmeans_overview.pdf): K-means groups and the fitted planes over the same surface.
- [Individual planes PDF](figures/autonomous_kmeans_subspaces.pdf): each group and its fitted plane shown separately.
- Colored dots are actual trajectories (160 per cluster are drawn for legibility; all 12,000 are used to fit).
- Gray geometry is the exact nonlinear surface. Shaded polygons are finite portions of infinite linear planes.
  Each polygon is the convex hull of the projected training group together with the origin.
  Thus a polygon indicates a displayed region, not a bounded subspace or the exact support of a cluster.
- The black point is the common origin. Small gray segments in the individual panels show orthogonal residuals.
- Camera orientation matches the earlier surface figure; the plot is enlarged within each panel for readability.

## Saved results

- [NPZ](figures/autonomous_kmeans_atlas.npz): training and test trajectories, training and test labels,
  centroids, orthonormal bases (shape 6 x 3 x 2), normals, projectors, singular values, and inertia.
  Cluster labels in this file are zero-based; display numbering is one-based.
- [JSON](figures/autonomous_kmeans_metrics.json): full-precision subspace matrices, normals, singular values,
  cluster sizes, fitting protocol, and errors.
- [CSV](figures/autonomous_kmeans_subspaces.csv): plane equations and per-cluster errors.
- [Notebook](autonomous_behavioral_atlas.ipynb): inspect results or set `RECOMPUTE=True` to repeat the experiment.

Reproduce from the project root with `python notebooks/motivation/autonomous_behavioral_atlas.py`.
Dependencies: NumPy, SciPy, scikit-learn, Matplotlib, and Times New Roman.

## Three groups for readability

The K=3 variant uses exactly the same training and held-out trajectories, with
unchanged uncentered rank-two fitting. Its updated viewpoint (elevation 35 degrees,
azimuth 160 degrees) exposes all three plane faces more clearly; the K=6 view
is retained.
The test RMS is **0.04493**, compared with **0.03696** for K=6 and **0.19450**
for a single global rank-two plane. All three clusters still have exact rank three.

- [Three-group overview](figures/autonomous_kmeans_k3_overview.pdf)
- [Three individual planes](figures/autonomous_kmeans_k3_subspaces.pdf)
- [Subspace bases and metrics](figures/autonomous_kmeans_k3_metrics.json)
- [Data and assignments](figures/autonomous_kmeans_k3_atlas.npz)
- [Plane equations](figures/autonomous_kmeans_k3_subspaces.csv)

Reproduce with `python notebooks/motivation/autonomous_behavioral_atlas.py --clusters 3`.
The original K=6 files are retained.
