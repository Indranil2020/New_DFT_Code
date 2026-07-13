Mixed-Precision Ozaki Scheme
============================

Error-Free Transformations
--------------------------

The Ozaki scheme decomposes an FP64 matrix :math:`A` into a sum of
lower-precision matrices:

.. math::
   A = A^{(0)} + A^{(1)} + \cdots + A^{(n_s)}

where :math:`A^{(0)}` is representable in FP16/BF16 and the residual terms
:math:`A^{(k)}` capture the rounding error. Each term is multiplied using
tensor-core GEMM (FP16 input, FP32 accumulate), and the results are summed
in FP32.

Accuracy Guarantee
------------------

For :math:`n_s` slices, the error in :math:`C = A \times B` is bounded by:

.. math::
   \|C_{\text{Ozaki}} - C_{\text{FP64}}\| \leq n_s \cdot
       \epsilon_{\text{FP16}} \cdot \|A\| \cdot \|B\|

For typical DFT Hamiltonians, :math:`n_s = 2`–3 slices suffice for
FP64-equivalent accuracy (:math:`\leq 10^{-13}` relative on traces).
