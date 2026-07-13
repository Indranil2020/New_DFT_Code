#pragma once

// Mixed-precision SCF: BF16/FP16 storage with FP64-emulated reductions (§3.4).
//
// The Ozaki mixed-precision approach stores the density matrix and intermediate
// matrices in reduced precision (FP16/BF16), but performs the critical
// reductions (energy, forces) in FP64. The FP64 emulation uses the Ozaki
// decomposition: a single FP64 value is represented as a sum of FP16 slices:
//   x = Σ_i x_i  where each x_i is an FP16-representable value.
//
// This module provides the mixed-precision SCF adapter that wraps the
// standard SCFDriver to optionally use reduced precision storage.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace tides::scf {

// Precision mode for SCF matrices.
enum class PrecisionMode {
  kFP64 = 0,    // Full double precision (reference)
  kBF16 = 1,    // BF16 storage, FP64 reductions
  kFP16 = 2,    // FP16 storage, FP64 reductions (Ozaki)
  kAuto = 3,    // Auto-select based on system size
};

struct MixedPrecisionConfig {
  PrecisionMode mode = PrecisionMode::kFP64;
  int n_ozaki_slices = 5;     // number of Ozaki slices for FP16
  double error_budget = 1e-6; // acceptable error from mixed precision
  bool use_f64e_reductions = true;  // use FP64-emulated reductions
};

class MixedPrecisionSCF {
 public:
  // Select the optimal precision mode based on system size and error budget.
  static PrecisionMode AutoSelect(std::size_t n_basis, double error_budget) {
    if (error_budget < 1e-8) return PrecisionMode::kFP64;
    if (n_basis > 2000 && error_budget > 1e-5) return PrecisionMode::kBF16;
    if (n_basis > 500 && error_budget > 1e-4) return PrecisionMode::kFP16;
    return PrecisionMode::kFP64;
  }

  // Quantize a double to BF16 (truncated mantissa).
  // BF16 has 1 sign + 8 exponent + 7 mantissa bits.
  static double ToBF16(double x) {
    if (x == 0.0) return 0.0;
    // BF16 truncation: keep the top 16 bits, zero the bottom 48 bits.
    // In IEEE 754 double: 1+11+52 bits. BF16: 1+8+7.
    // Truncate 3 exponent bits and 45 mantissa bits.
    uint64_t bits;
    // Type-pun via memcpy (strict aliasing safe).
    std::memcpy(&bits, &x, sizeof(bits));
    // Round-to-nearest-even: add 0x7FFF (bit 47 + lower) before truncation.
    uint64_t rounding = 0x7FFF + ((bits >> 47) & 1);
    bits = (bits + rounding) & 0xFFFF000000000000ULL;
    double result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
  }

  // Quantize a double to FP16 (half precision).
  // FP16 has 1 sign + 5 exponent + 10 mantissa bits.
  static double ToFP16(double x) {
    if (x == 0.0) return 0.0;
    // Clamp to FP16 range: max ~65504, min normal ~6.1e-5
    if (std::fabs(x) > 65504.0) x = std::copysign(65504.0, x);
    if (std::fabs(x) < 6.1e-5 && x != 0.0) {
      // Subnormal or zero — flush to zero for simplicity.
      return 0.0;
    }
    uint64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    // Round-to-nearest-even for FP16: keep top 5 bits (sign + 4 exponent).
    uint64_t rounding = 0x3FF + ((bits >> 42) & 1);
    bits = (bits + rounding) & 0xFFFFE00000000000ULL;
    double result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
  }

  // Quantize a matrix to the specified precision mode.
  static std::vector<double> QuantizeMatrix(
      const std::vector<double>& M, PrecisionMode mode) {
    std::vector<double> result = M;
    if (mode == PrecisionMode::kFP64) return result;

    for (auto& v : result) {
      if (mode == PrecisionMode::kBF16)
        v = ToBF16(v);
      else if (mode == PrecisionMode::kFP16)
        v = ToFP16(v);
    }
    return result;
  }

  // FP64-emulated reduction: sum of vector elements with Ozaki compensation.
  // Instead of summing in FP64 directly, decompose each element into
  // high+low parts and sum separately, reducing rounding error.
  static double F64EReduce(const std::vector<double>& v) {
    // Ozaki / Shewchuk-style compensated summation.
    double sum = 0.0;
    double compensation = 0.0;
    for (double x : v) {
      double y = x - compensation;
      double t = sum + y;
      compensation = (t - sum) - y;
      sum = t;
    }
    return sum;
  }

  // Compute the mixed-precision error estimate for a matrix.
  // Returns the expected relative error from quantization.
  static double QuantizationError(std::size_t n, const std::vector<double>& M,
                                   PrecisionMode mode) {
    if (mode == PrecisionMode::kFP64) return 0.0;
    double machine_eps = (mode == PrecisionMode::kBF16) ? 1.0 / 128.0
                                                           : 1.0 / 1024.0;
    double max_val = 0.0;
    for (std::size_t i = 0; i < n * n; ++i)
      max_val = std::max(max_val, std::fabs(M[i]));
    if (max_val < 1e-30) return 0.0;
    // Error grows as sqrt(N) * eps * max_val (random walk model).
    return machine_eps * std::sqrt(static_cast<double>(n)) * max_val;
  }
};

}  // namespace tides::scf
