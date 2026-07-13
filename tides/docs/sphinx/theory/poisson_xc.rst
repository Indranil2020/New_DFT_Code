Grid Poisson and Exchange-Correlation
======================================

TIDES uses a dual real-space grid for density and potential evaluation,
with grid-based Poisson (Hartree) and libxc-backed exchange-correlation
:cite:`tides2026`.

Dual Grid
---------

The dual grid consists of:

* **Coarse grid** (spacing :math:`h_c`): orbital evaluation, density
  projection.
* **Fine grid** (spacing :math:`h_f \leq h_c`): density :math:`\rho(\mathbf{r})`,
  Hartree potential :math:`v_H(\mathbf{r})`, XC potential
  :math:`v_\text{xc}(\mathbf{r})`.

Four boundary conditions are supported: free (isolated molecule), wire (1-D
periodic), slab (2-D periodic), and bulk (3-D periodic).

Poisson Solver (Hartree)
------------------------

The Hartree potential is obtained by solving Poisson's equation:

.. math::

   \nabla^2 v_H(\mathbf{r}) = -4\pi\,\rho(\mathbf{r}).

For **free boundary conditions**, TIDES uses a direct summation with a
self-interaction regularization term. For **periodic** boundary conditions,
the solver uses cuFFT (GPU) or FFTW3 (CPU) fast Fourier transforms:

.. math::

   v_H(\mathbf{G}) = \frac{4\pi\,\rho(\mathbf{G})}{|\mathbf{G}|^2}.

The GPU Poisson kernel (cuFFT) achieves :math:`10^{-10}` accuracy on periodic
systems with adequate grid resolution.

Density Build
-------------

The density on the grid is built from the density matrix:

.. math::

   \rho(\mathbf{r}) = \sum_{\mu\nu} P_{\mu\nu}\,
   \phi_\mu(\mathbf{r})\, \phi_\nu(\mathbf{r}).

This is computed as a GEMM (BLAS ``dgemm``) from the basis function values on
the grid, making it compatible with R2/R3 (purification produces :math:`P`,
not orbitals).

V_mat Build
-----------

The potential matrix :math:`V_{\mu\nu}` is built by projecting the grid
potential onto the basis:

.. math::

   V_{\mu\nu} = \int v(\mathbf{r})\, \phi_\mu(\mathbf{r})\,
   \phi_\nu(\mathbf{r})\, d\mathbf{r}.

This is also a GEMM (``dgemm``), with the adjoint relationship
:math:`\text{BuildRho}^\dagger = \text{BuildHmat}` verified to
:math:`10^{-12}` (MET: :math:`1.67\times10^{-15}`).

Exchange-Correlation
--------------------

XC is evaluated via the fused Tier-0 XC engine (``xc::XcEval``) which supports:

* **LDA**: PW92 correlation + Slater exchange.
* **GGA**: PBE (and variants) via libxc.

The XC energy is:

.. math::

   E_\text{xc} = \int \epsilon_\text{xc}\!\bigl[\rho(\mathbf{r})\bigr]\,
   \rho(\mathbf{r})\, d\mathbf{r}.

The potential :math:`v_\text{xc} = \delta E_\text{xc} / \delta \rho` is
evaluated pointwise on the grid. The GPU XC kernel achieves Tier-0 accuracy
(:math:`5\times10^{-14}` relative vs CPU libxc).
