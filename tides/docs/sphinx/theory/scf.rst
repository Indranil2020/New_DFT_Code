SCF Driver and Convergence
==========================

The self-consistent field (SCF) loop iterates to find the ground-state
density matrix :math:`P` that satisfies:

.. math::

   H[P] \, C = S \, C \, \varepsilon, \quad P = \sum_{k}^{N_{\text{occ}}} C_k C_k^T

SCF Algorithm
-------------

1. Build the Hamiltonian :math:`H[P]` from the current density.
2. Solve the generalized eigenproblem (via solver broker).
3. Occupy the :math:`N_{\text{occ}}` lowest orbitals.
4. Mix the new and old density matrices.
5. Check convergence (energy or density change < tol).

Mixing Schemes
--------------

**Simple mixing:**

.. math::

   P_{\text{next}} = \alpha \, P_{\text{new}} + (1-\alpha) \, P_{\text{old}}

**Pulay/DIIS:** Find coefficients :math:`c_j` that minimize
:math:`\|\sum_j c_j R_j\|^2` subject to :math:`\sum c_j = 1`, where
:math:`R_j = F_j - P_j` is the residual. Then:

.. math::

   P_{\text{next}} = \sum_j c_j \, F_j

Löwdin Orthogonalization
------------------------

The overlap matrix :math:`S` is diagonalized and eigenvalues below a
threshold are filtered out to handle near-linear dependence:

.. math::

   X = V_{\text{retained}} \, \Lambda^{-1/2}, \quad H' = X^T H X
