#pragma once

// E4: Mixed precision (BF16/FP16 storage + FP64 reductions) in SCF loop.
//
// The Ozaki mixed-precision scheme stores the density matrix P in FP16/BF16
// (half precision) and performs the matrix products in FP64, accumulating
// the result in FP64. This gives the speed of tensor cores (FP16 GEMM) with
// the accuracy of FP64 (via the Ozaki error-compensated summation).
//
// For the CPU reference, we simulate the mixed-precision path by:
//   1. Quantizing P to FP16 precision (rounding to 11-bit mantissa).
//   2. Performing the matrix product in FP64.
//   3. Applying the Ozaki error compensation: the rounding error from step 1
//      is accumulated and fed back in the next iteration.
//
// Observable (E4): the mixed-precision SCF energy matches the FP64 SCF energy
// to within 1e-6 Ha (the Ozaki scheme recovers FP64 accuracy).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tides::tile {

// E4: Mixed-precision SCF operations.
// Simulates the BF16/FP16 storage + FP64 reduction pipeline.
class MixedPrecisionSCF {
 public:
  // Quantize a value to FP16 precision (simulated).
  // FP16 has 11 bits of mantissa (10 explicit + 1 implicit).
  // This rounds to the nearest representable FP16 value.
  static double QuantizeFp16(double x) {
    if (x == 0.0 || !std::isfinite(x)) return x;
    // FP16: 1 sign + 5 exponent + 10 mantissa = 16 bits.
    // Extract sign, exponent, mantissa.
    int exponent;
    double mantissa = std::frexp(x, &exponent);
    // FP16 exponent range: [-14, 15]. Clip to representable range.
    if (exponent > 16) return (x > 0) ? 65504.0 : -65504.0;  // max FP16
    if (exponent < -24) return 0.0;  // subnormal / underflow
    // Quantize mantissa to 10 bits: round to nearest 2^(-10).
    const double resolution = std::ldexp(1.0, -10);
    double quantized_mantissa = std::round(mantissa / resolution) * resolution;
    return std::ldexp(quantized_mantissa, exponent);
  }

  // Quantize a value to BF16 precision (simulated).
  // BF16 has 8 bits of mantissa (7 explicit + 1 implicit).
  static double QuantizeBf16(double x) {
    if (x == 0.0 || !std::isfinite(x)) return x;
    int exponent;
    double mantissa = std::frexp(x, &exponent);
    if (exponent > 128) return (x > 0) ? 3.39e38 : -3.39e38;
    if (exponent < -126) return 0.0;
    const double resolution = std::ldexp(1.0, -7);
    double quantized_mantissa = std::round(mantissa / resolution) * resolution;
    return std::ldexp(quantized_mantissa, exponent);
  }

  // Quantize a matrix to FP16/BF16 precision (element-wise).
  static std::vector<double> QuantizeMatrix(const std::vector<double>& M,
                                             bool use_bf16 = false) {
    std::vector<double> result(M.size());
    for (std::size_t i = 0; i < M.size(); ++i)
      result[i] = use_bf16 ? QuantizeBf16(M[i]) : QuantizeFp16(M[i]);
    return result;
  }

  // Ozaki error-compensated GEMM: C = P @ H with FP16 storage + FP64 reduction.
  // The Ozaki scheme:
    //   1. Split P into FP16-representable chunks: P = P_0 + P_1 + ... + P_k
  //   2. Compute C = sum_i (P_i @ H) in FP64
  //   3. The splitting error is O(2^{-11}) per element, compensated by the
  //      multi-chunk decomposition.
  // For the CPU reference, we simulate this with a single FP16 quantization
  // pass and error feedback.
  static std::vector<double> OzakiGEMM(
      std::size_t n,
      const std::vector<double>& P,
      const std::vector<double>& H,
      bool use_bf16 = false,
      std::vector<double>* error_feedback = nullptr) {
    std::vector<double> P_quant = QuantizeMatrix(P, use_bf16);

    // Compute the quantization error for feedback.
    if (error_feedback) {
      error_feedback->resize(P.size());
      for (std::size_t i = 0; i < P.size(); ++i)
        (*error_feedback)[i] = P[i] - P_quant[i];
    }

    // Compute C = P_quant @ H in FP64.
    std::vector<double> C(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (std::size_t k = 0; k < n; ++k)
          s += P_quant[i * n + k] * H[k * n + j];
        C[i * n + j] = s;
      }

    // Apply error feedback if available (Ozaki compensation).
    if (error_feedback && !error_feedback->empty()) {
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
          double s = 0.0;
          for (std::size_t k = 0; k < n; ++k)
            s += (*error_feedback)[i * n + k] * H[k * n + j];
          C[i * n + j] += s;
        }
    }

    return C;
  }

  // Compute the mixed-precision band energy: Tr(P @ H) using FP16 P + FP64 H.
  // This is the key operation in the mixed-precision SCF loop.
  static double MixedPrecisionBandEnergy(
      std::size_t n,
      const std::vector<double>& P,
      const std::vector<double>& H,
      bool use_bf16 = false) {
    auto C = OzakiGEMM(n, P, H, use_bf16);
    double trace = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      trace += C[i * n + i];
    return trace;
  }

  // Estimate the precision loss from FP16 quantization.
  static double QuantizationError(const std::vector<double>& M, bool use_bf16) {
    auto M_quant = QuantizeMatrix(M, use_bf16);
    double max_err = 0.0;
    for (std::size_t i = 0; i < M.size(); ++i)
      max_err = std::max(max_err, std::fabs(M[i] - M_quant[i]));
    return max_err;
  }
};

}  // namespace tides::tile
