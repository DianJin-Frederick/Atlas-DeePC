# Introduction text

> **Superseded for the paper's trajectory-space example.** This text describes
> instantaneous output geometry only. Use
> [io_trajectory_introduction.md](io_trajectory_introduction.md) for complete
> input–output trajectory windows.

A simple example illustrates the geometric motivation for local representations.
Consider the controlled nonlinear system

\[
\theta_{k+1}=\theta_k+h\omega_k,\qquad
z_{k+1}=z_k+h(v_k+a\sin\theta_k),\qquad
y_k=(\cos\theta_k,\sin\theta_k,z_k)^{\mathsf T},
\]

where \((\omega_k,v_k)\) is the input, \(h=0.12\), and \(a=0.35\).
Its measured outputs satisfy \(y_{1,k}^2+y_{2,k}^2=1\), so every rollout traces
a curve on a two-dimensional cylinder embedded in \(\mathbb R^3\).
Figure [X] shows a family of rollouts sampling this curved surface.
As the angular position changes, the same angular input produces motion in
different directions in the measured output coordinates. A single affine plane
cannot represent the entire cylinder, whereas local planes can approximate
restricted neighborhoods. This example provides a geometric motivation for
grouping offline data according to the operating condition and constructing
local predictive representations. The cylinder depicts the geometry of
instantaneous outputs; the input-output trajectory windows used for prediction
generally occupy a higher-dimensional space.

## Figure caption

**Curved output geometry of a simple nonlinear system.** Twelve simulated
rollouts, initialized at different angular positions, sample the invariant
cylinder \(y_1^2+y_2^2=1\) over \(z\in[-1,1]\). The orange curve highlights one
rollout; the circle and triangle mark its initial and final samples,
respectively. Blue curves show the other rollouts, and the gray wireframe
indicates the cylinder. Each rollout is a curve on the surface.

## Reproducibility details (optional methods note)

Each rollout contains 180 transitions and starts at \(z_0=-1\), with
\(\theta_0=-\pi+2\pi j/12\), \(j=0,\ldots,11\). Set
\(\omega_k=2\pi(1.25)/(180h)\) and
\(v_k=2/(180h)-a\sin\theta_k\). These inputs produce 1.25 revolutions and a
constant vertical increment, ending at \(z_{180}=1\), while satisfying
\(|\omega_k|\leq1.2\) and \(|v_k|\leq0.8\). Every plotted sample is generated
by the stated discrete-time dynamics. The sweep inputs are chosen to expose
the geometry; these visualization rollouts are separate from the notebook's
random training and test rollouts.

The figure illustrates local geometry; it does not establish graph reachability,
reference trackability, or a closed-loop DeePC performance improvement.

## Figure files

- `figures/cylinder_paper.pdf`: vector PDF with embedded Times New Roman fonts;
  recommended for LaTeX.
- `figures/cylinder_paper.svg`: vector SVG with glyphs outlined to preserve the
  Times New Roman appearance across applications.
- `figures/cylinder_paper.png`: 600 dpi PNG, 2100 × 2220 pixels.

The figure is sized at 3.5 × 3.7 inches, with 8 pt tick labels and 9 pt axis
labels. `cylinder_paper_figure.py` reproduces all three files. The accompanying
`cylinder_introduction.tex` contains the paragraph, equation, and figure block.
