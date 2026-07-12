.. _tile-substrate:

Tile Substrate and Mixed Precision
==================================

The tile substrate is the architectural foundation of TIDES: a block-sparse
matrix format (CSR-of-tiles) that maps every heavy operation to grouped GEMM
on tensor cores :cite:`tides2026`.

TileMat: CSR-of-Tiles
---------------------

A :class:`TileMat` stores a matrix as a collection of dense tiles
(dense :math:`T \times T` blocks) indexed by a CSR adjacency structure. This
naturally represents the block-sparse matrices arising in NAO DFT:

.. math::

   M = \sum_{(i,j) \in \mathcal{N}} M_{ij}\, e_i e_j^T,

where :math:`\mathcal{N}` is the set of non-zero tile pairs (determined by the
orbital cutoff radius).

Mixed-Precision Ozaki Scheme
----------------------------

The Ozaki scheme :cite:`tides2026` decomposes an FP64 matrix :math:`A` into a
sum of lower-precision matrices:

.. math::

   A = A^{(0)} + A^{(1)} + \cdots + A^{(n_s)},

where :math:`A^{(0)}` is representable in FP16/BF16 and the residual terms
:math:`A^{(k)}` capture the rounding error. Each term is multiplied using
tensor-core GEMM (FP16 input, FP32 accumulate), and the results are summed in
FP32.

Accuracy Guarantee
~~~~~~~~~~~~~~~~~~

For :math:`n_s` slices, the error in :math:`C = A \times B` is bounded by:

.. math::

   \|C_\text{Ozaki} - C_\text{FP64}\|
   \leq n_s \cdot \epsilon_\text{FP16} \cdot \|A\| \cdot \|B\|.

For typical DFT Hamiltonians, :math:`n_s = 2`–:math:`3` slices suffice for
FP64-equivalent accuracy (:math:`\leq 10^{-13}` relative on traces).

Filtered SpGEMM
---------------

The filtered sparse tile GEMM computes
:math:`C_{ij} = \sum_k A_{ik} B_{kj}` only for tiles :math:`(i, j)` in
:math:`\mathcal{N}`, with per-tile error tracking. The operation ledger
records the precision used per tile and the accumulated error, enabling
adaptive precision escalation.

ChFSI Subspace Reuse and Locking
---------------------------------

The Chebyshev-filtered subspace iteration (ChFSI) supports two optimizations:

1. **Subspace reuse**: the converged subspace from the previous SCF step is
   used as the initial guess, reducing the number of filter applications by
   :math:`\geq 2\times`.

2. **Locking/deflation**: converged eigenpairs (residual below
   :math:`\epsilon_\text{lock}`) are locked and removed from the active space,
   reducing the problem size as more eigenvalues converge.
