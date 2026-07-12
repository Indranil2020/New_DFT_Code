Fermi-Operator Expansion (FOE)
==============================

Chebyshev Expansion of the Fermi Function
-----------------------------------------

At finite electronic temperature :math:`T_e`, the density matrix is:

.. math::
   P = f(H, \mu, T_e) = \frac{1}{1 + e^{\beta(H - \mu)}}

where :math:`\beta = 1/(k_B T_e)`. This is approximated by a Chebyshev
polynomial expansion of order :math:`n_p \sim \beta \cdot \Delta H`:

.. math::
   f(H) \approx \sum_{k=0}^{n_p} c_k T_k(\tilde{H})

where :math:`\tilde{H} = (H - c_{\min}) / (c_{\max} - c_{\min}) \in [-1, 1]`
and :math:`T_k` are Chebyshev polynomials.

Spectral Quadrature (SQ)
------------------------

The SPARC spectral quadrature method computes
:math:`\text{Tr}(f(H))` via Gauss quadrature, avoiding explicit matrix
functions:

.. math::
   \text{Tr}(f(H)) = \sum_{j} w_j \, e_j^T f(H) e_j

where :math:`\{e_j, w_j\}` are quadrature nodes. Each
:math:`e_j^T f(H) e_j` is computed via Chebyshev recurrence — a sequence of
GEMMs on the tile substrate.
