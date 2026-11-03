# Paper example: curved geometry of complete input–output trajectories

The ready-to-insert [LaTeX passage](io_trajectory_introduction.tex) introduces
only the system, its complete input–output windows, and their exact
three-dimensional visualization. It contains a `wrapfigure` and caption.

A simple example illustrates the curved geometry of finite input–output
trajectories. Consider the scalar nonlinear system \(y_{k+1}=y_k^2+u_k\).
With \(m=p=1\) and \(T_{\mathrm{ini}}=T_{\mathrm f}=1\), the trajectory
length is \(L=2\), and each complete input–output window is
\(w=\operatorname{col}(u_0,u_1,y_0,y_1)\in\mathbb R^4\).
Such a vector is a column of the stacked depth-two input and output Hankel
matrices. Select \(y_0=x\) and \((u_0,u_1)=(a,-a)\), with
\(x\in[-1,1]\) and \(a\in[-0.5,0.5]\). The resulting trajectories are

\[
w(a,x)=\operatorname{col}(a,-a,x,x^2+a).
\]

The selected family satisfies \(u_0+u_1=0\) and is therefore contained in a
three-dimensional linear subspace of \(\mathbb R^4\). Choose the orthonormal
basis

\[
Q=\begin{bmatrix}
1/\sqrt3&0&-1/\sqrt6\\
-1/\sqrt3&0&1/\sqrt6\\
0&1&0\\
1/\sqrt3&0&2/\sqrt6
\end{bmatrix}.
\]

Then \(z=Q^{\mathsf T}w\), \(w=Qz\), and
\(\|w_i-w_j\|_2=\|z_i-z_j\|_2\). The picture retains both input samples
and both output samples through an invertible change of coordinates on the
selected subspace. Substitution yields

\[
z_1=\sqrt3a+x^2/\sqrt3,\qquad z_2=x,\qquad
z_3=\sqrt{2/3}\,x^2.
\]

The trajectories therefore form a two-dimensional surface satisfying
\(z_3=\sqrt{2/3}\,z_2^2\). This parabolic cylinder is curved despite having
only two intrinsic dimensions; the selected trajectories cannot all lie in
a single two-dimensional affine plane. The curvature reflects the nonlinear
input–output relation of the system and is preserved by the visualization.
The input restriction selects a visualizable family of valid trajectories;
it does not exhaust the system's finite-horizon behavior.

## Caption

**A curved family of complete input–output trajectories.** Each point represents
a length-two window \(w=(u_0,u_1,y_0,y_1)^{\mathsf T}\) of
\(y_{k+1}=y_k^2+u_k\), selected with \(u_1=-u_0\). The orthonormal coordinates
\(z=Q^{\mathsf T}w\) preserve the full vectors and their distances. The mesh
follows the surface \(z_3=\sqrt{2/3}\,z_2^2\); its curves do not represent
evolution in time.
