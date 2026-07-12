NAO Basis Theory
=================

The numeric atom-centered orbital (NAO) basis is defined as:

.. math::

   \phi_{n\ell m}(\mathbf{r}) = R_{n\ell}(r) \, Y_{\ell m}(\hat{r})

where :math:`R_{n\ell}(r)` is a radial function generated from a
confined-atom solve and :math:`Y_{\ell m}` are real spherical harmonics.

Confinement
-----------

The radial function is obtained by solving the atomic Schrödinger equation
with a confining potential:

.. math::

   V_{\text{conf}}(r) = \left(\frac{r}{r_c}\right)^{n_c}

where :math:`r_c` is the confining radius (default 5 Bohr) and :math:`n_c`
controls the smoothness (default 2).

Two-Center Integrals
--------------------

Two-center integrals (overlap :math:`S`, kinetic :math:`T`) are computed
using Slater-Koster decomposition:

.. math::

   S_{ab} = \sum_L h_L(R) \, \text{AngularFactor}(\ell_a, m_a, \ell_b, m_b, L, \theta, \phi)

where :math:`h_L(R)` is the radial integral tabulated via cubic spline and
the angular factor comes from the Slater-Koster rotation table.

Supported angular momentum pairs: ss, sp, ps, pp, sd, ds, pd, dp, dd.

Basis Recipes
-------------

TIDES supports DZP (double-zeta + polarization) and TZP (triple-zeta +
polarization) recipes. Each recipe generates a set of radial functions with
specific quantum numbers :math:`(n, \ell)` and confining radii.

.. math::

   \text{DZP}: \{2s, 2p, 2d\}, \quad \text{TZP}: \{3s, 3p, 2d\}
