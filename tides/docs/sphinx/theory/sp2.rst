Density-Matrix Purification (SP2)
================================

SP2 Iteration
-------------

The SP2 (second-order spectral projection) purification computes the density
matrix :math:`P = \theta(\mu I - H)` without diagonalization, via the
recursion:

.. math::
   P_{k+1} = \begin{cases}
       2P_k - P_k^2 & \text{if } \text{Tr}(P_k) < N_e/2 \\
       P_k^2 & \text{otherwise}
   \end{cases}

starting from :math:`P_0 = (c_{\min} I - H) / (c_{\min} - c_{\max})` where
:math:`c_{\min}, c_{\max}` bound the spectrum of :math:`H`.

Submatrix Method
----------------

For sparse :math:`H` (gapped systems), the submatrix method (NOLSM, Schäffer et
al.) solves the SP2 iteration as many small dense problems, one per atom
neighborhood. For each atom :math:`I`, define the local Hamiltonian
:math:`H_I` as the submatrix of :math:`H` restricted to the basis functions
within the truncation radius of atom :math:`I`. The local density matrix
:math:`P_I` is computed by SP2 on :math:`H_I`, and the global :math:`P` is
assembled from the diagonal blocks of all :math:`P_I`.

This maps the sparse problem to batched dense GEMM — the tile substrate's
sweet spot.
