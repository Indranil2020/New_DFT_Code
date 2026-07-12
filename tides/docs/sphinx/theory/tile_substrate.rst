Tile Substrate and Mixed Precision
===================================

The tile substrate is TIDES' central architectural concept: a CSR-of-tiles
block-sparse matrix representation that serves as the "one true layer" for all
matrix operations.

TileMat Structure
-----------------

A TileMat stores a sparse collection of dense :math:`T_b \times T_b` tiles
(where :math:`T_b` is the block size, typically 16 or 32):

.. math::

   \text{TileMat} = \{(i, j, T_{ij}) : T_{ij} \in \mathbb{R}^{T_b \times T_b}\}

The sparsity pattern is stored as CSR (compressed sparse rows of tile blocks).

Mixed Precision (Ozaki)
------------------------

The Ozaki scheme represents a single FP64 value as a sum of FP16 slices:

.. math::

   x = \sum_{i=1}^{n} x_i, \quad x_i \in \text{FP16}

The GEMM is then performed in FP16 with FP64-emulated reductions:

.. math::

   C_{ij} = \sum_k A_{ik} B_{kj} = \sum_k \sum_p \sum_q A_{ik}^{(p)} B_{kj}^{(q)}

The error is bounded by:

.. math::

   \|C_{\text{mixed}} - C_{\text{exact}}\| \leq \epsilon_{\text{FP16}} \cdot \sqrt{N} \cdot \max|C|
