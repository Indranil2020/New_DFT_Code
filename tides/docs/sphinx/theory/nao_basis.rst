Numeric Atom-Centered Orbital Basis
====================================

The TIDES basis set consists of numeric atom-centered orbitals (NAOs) of the
form :cite:`tides2026`:

.. math::

   \phi_\mu(\mathbf{r}) = R_{nl}(r)\, Y_{lm}(\hat{r}),

where :math:`R_{nl}(r)` is a numerical radial function generated from a
confined-atom solver and :math:`Y_{lm}` are real spherical harmonics
(tesseral harmonics).

Confinement and Generation
--------------------------

Each radial function is obtained by solving the atomic Kohn-Sham equations
with a confining potential :math:`V_\text{conf}(r)` that forces the orbital to
decay smoothly to zero at a cutoff radius :math:`r_\text{cut}`:

.. math::

   V_\text{conf}(r) = \Bigl(\frac{r}{r_\text{cut}}\Bigr)^n
   \quad\text{for } r > r_\text{cut}.

The confinement produces orbitals that are strictly zero beyond
:math:`r_\text{cut}`, making all two-center integrals short-ranged. The
confined-atom solver uses a Numerov radial Schrödinger integrator on a
logarithmic grid.

Basis Set Recipes
-----------------

TIDES supports the following standard NAO recipes:

+-------+--------------------------+-----------------------------------+
| Label | Functions per atom (H)    | Functions per atom (C/N/O)        |
+=======+==========================+===================================+
| DZP   | 2s 1p (5 functions)       | 2s 2p 1d (14 functions)          |
+-------+--------------------------+-----------------------------------+
| TZP   | 3s 2p (7 functions)       | 3s 3p 2d (23 functions)           |
+-------+--------------------------+-----------------------------------+
| TZDP  | 3s 3p 2d (12 functions)   | 3s 3p 3d 1f (33 functions)       |
+-------+--------------------------+-----------------------------------+

Two-Center Integrals
--------------------

The overlap and kinetic energy matrices are assembled via analytic two-center
integrals factorized using Slater–Koster (SK) tables. The integral for a pair
:math:`(\phi_a, \phi_b)` at distance :math:`R` along direction
:math:`(\theta, \phi)` is:

.. math::

   S_{ab}(R, \theta, \phi) = \sum_L h_L(R)\,
   \mathcal{A}_{l_a m_a, l_b m_b}^{(L)}(\theta, \phi),

where :math:`h_L(R)` is the radial integral (tabulated on a 1-D grid and
evaluated via cubic spline) and :math:`\mathcal{A}^{(L)}` is the angular
coupling coefficient from the SK table.

The radial part :math:`h_L(R)` is computed by direct numerical integration on
the NAO radial grids with a Gauss–Legendre angular product grid:

.. math::

   h_L(R) = \int_0^\infty\!\! dr\, r^2 R_a(r)\,
   \oint\!\! d\Omega\; Y_{l_a m_a}(\hat{r})\,
   Y_{l_b m_b}(\widehat{r - \mathbf{R}})\,
   P_L(\cos\gamma).

For force evaluation, the spline derivative
:math:`h_L'(R) = \partial h_L / \partial R` is available analytically from
the cubic spline coefficients (see :ref:`scf-driver`).
