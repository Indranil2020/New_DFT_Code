// T1.4: Ozaki f64e GPU GEMM tests — vs CPU f64e long-double oracle.
// Validates that the CUDA Ozaki FP16-slice GEMM produces results within
// the analytical forward-error bound for adversarial dynamic-range inputs.

#include "tile/f64e_reference.hpp"
#include "tile/gemm_grouped.hpp"
#include "tile/ozaki.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

using tides::tile::CudaOzakiGemmResult;
using tides::tile::GemmF64eReference;
using tides::tile::GemmOzakiFp16Cuda;
using tides::tile::GemmOzakiFp16Reference;
using tides::tile::CudaRuntimeAvailable;

std::vector<double> MakeDynamicRangeMatrix(std::size_t rows, std::size_t cols,
                                           std::uint64_t salt) {
  std::vector<double> values(rows * cols, 0.0);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      const std::uint64_t key =
          (i + 5) * 0x9E3779B185EBCA87ULL ^
          (j + 13) * 0xC2B2AE3D27D4EB4FULL ^ salt;
      const double sign = (key & 1U) == 0 ? 1.0 : -1.0;
      if (key % 17 == 0) {
        values[i * cols + j] = sign * 1.0e9;
      } else if (key % 13 == 0) {
        values[i * cols + j] = sign * 1.0e-9;
      } else {
        values[i * cols + j] =
            sign * static_cast<double>((key % 4093U) + 1U) / 4093.0;
      }
    }
  }
  return values;
}

double MaxAbsDifference(const std::vector<double>& a,
                        const std::vector<double>& b) {
  double max_error = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    max_error = std::max(max_error, std::abs(a[i] - b[i]));
  }
  return max_error;
}

int TestOzakiGemmVsF64eOracle() {
  constexpr std::size_t m = 32;
  constexpr std::size_t k = 32;
  constexpr std::size_t n = 32;
  const std::vector<double> a = MakeDynamicRangeMatrix(m, k, 0xC0DEC0DEULL);
  const std::vector<double> b = MakeDynamicRangeMatrix(k, n, 0xFACEB00CULL);

  auto f64e = GemmF64eReference(m, k, n, a, b);
  if (!f64e.ok()) {
    std::cerr << "GemmF64eReference failed: " << f64e.status().message()
              << '\n';
    return 1;
  }

  auto cuda_ozaki = GemmOzakiFp16Cuda(m, k, n, a, b);
  if (!cuda_ozaki.ok()) {
    std::cerr << "GemmOzakiFp16Cuda failed: "
              << cuda_ozaki.status().message() << '\n';
    return 1;
  }

  const double max_diff =
      MaxAbsDifference(cuda_ozaki.value().values, f64e.value().values);
  const double bound = cuda_ozaki.value().analytical_abs_bound;

  std::cout << "cuda_ozaki_gemm: shape=" << m << "x" << k << "x" << n
            << " a_slices=" << cuda_ozaki.value().a_slice_count
            << " b_slices=" << cuda_ozaki.value().b_slice_count
            << " slice_pairs=" << cuda_ozaki.value().slice_pair_count
            << " kernel_ms=" << cuda_ozaki.value().kernel_ms
            << " max_diff_vs_f64e=" << max_diff
            << " analytical_bound=" << bound
            << " observed_error=" << cuda_ozaki.value().max_abs_error_vs_long_double
            << '\n';

  if (max_diff > bound && bound > 0.0) {
    std::cerr << "FAIL: max_diff=" << max_diff
              << " exceeds analytical_bound=" << bound << '\n';
    return 1;
  }
  if (cuda_ozaki.value().max_abs_error_vs_long_double > bound) {
    std::cerr << "FAIL: observed_error="
              << cuda_ozaki.value().max_abs_error_vs_long_double
              << " exceeds analytical_bound=" << bound << '\n';
    return 1;
  }
  return 0;
}

int TestOzakiGemmVsCpuOzaki() {
  constexpr std::size_t m = 16;
  constexpr std::size_t k = 16;
  constexpr std::size_t n = 16;
  const std::vector<double> a = MakeDynamicRangeMatrix(m, k, 0x12345678ULL);
  const std::vector<double> b = MakeDynamicRangeMatrix(k, n, 0xABCDEFULL);

  auto cpu_ozaki = GemmOzakiFp16Reference(m, k, n, a, b);
  if (!cpu_ozaki.ok()) {
    std::cerr << "GemmOzakiFp16Reference failed: "
              << cpu_ozaki.status().message() << '\n';
    return 1;
  }

  auto cuda_ozaki = GemmOzakiFp16Cuda(m, k, n, a, b);
  if (!cuda_ozaki.ok()) {
    std::cerr << "GemmOzakiFp16Cuda failed: "
              << cuda_ozaki.status().message() << '\n';
    return 1;
  }

  const double max_diff =
      MaxAbsDifference(cuda_ozaki.value().values, cpu_ozaki.value().values);

  std::cout << "cuda_ozaki_vs_cpu: shape=" << m << "x" << k << "x" << n
            << " max_diff=" << max_diff
            << " cpu_slices=" << cpu_ozaki.value().slice_pair_count
            << " cuda_slices=" << cuda_ozaki.value().slice_pair_count
            << '\n';

  // CUDA and CPU Ozaki should agree to within the CPU reference's own bound.
  const double tolerance =
      std::max(cpu_ozaki.value().max_fp64_forward_abs_bound,
               cuda_ozaki.value().analytical_abs_bound);
  if (max_diff > tolerance * 2.0 + 1e-12) {
    std::cerr << "FAIL: CUDA vs CPU Ozaki max_diff=" << max_diff
              << " exceeds 2x tolerance=" << tolerance * 2.0 << '\n';
    return 1;
  }
  return 0;
}

int TestOzakiGemmLargeShape() {
  constexpr std::size_t m = 64;
  constexpr std::size_t k = 64;
  constexpr std::size_t n = 64;
  const std::vector<double> a = MakeDynamicRangeMatrix(m, k, 0xDEADBEEFULL);
  const std::vector<double> b = MakeDynamicRangeMatrix(k, n, 0xCAFEBABEULL);

  auto f64e = GemmF64eReference(m, k, n, a, b);
  if (!f64e.ok()) {
    std::cerr << "GemmF64eReference failed: " << f64e.status().message()
              << '\n';
    return 1;
  }

  auto cuda_ozaki = GemmOzakiFp16Cuda(m, k, n, a, b);
  if (!cuda_ozaki.ok()) {
    std::cerr << "GemmOzakiFp16Cuda failed: "
              << cuda_ozaki.status().message() << '\n';
    return 1;
  }

  const double max_diff =
      MaxAbsDifference(cuda_ozaki.value().values, f64e.value().values);
  const double bound = cuda_ozaki.value().analytical_abs_bound;

  std::cout << "cuda_ozaki_gemm_large: shape=" << m << "x" << k << "x" << n
            << " slice_pairs=" << cuda_ozaki.value().slice_pair_count
            << " kernel_ms=" << cuda_ozaki.value().kernel_ms
            << " max_diff_vs_f64e=" << max_diff
            << " analytical_bound=" << bound << '\n';

  if (max_diff > bound && bound > 0.0) {
    std::cerr << "FAIL: large shape max_diff=" << max_diff
              << " exceeds analytical_bound=" << bound << '\n';
    return 1;
  }
  return 0;
}

int TestOzakiGemmLedger() {
  constexpr std::size_t m = 8;
  constexpr std::size_t k = 8;
  constexpr std::size_t n = 8;
  const std::vector<double> a = MakeDynamicRangeMatrix(m, k, 0xAAAULL);
  const std::vector<double> b = MakeDynamicRangeMatrix(k, n, 0xBBBULL);

  auto cuda_ozaki = GemmOzakiFp16Cuda(m, k, n, a, b);
  if (!cuda_ozaki.ok()) {
    std::cerr << "GemmOzakiFp16Cuda failed: "
              << cuda_ozaki.status().message() << '\n';
    return 1;
  }

  const auto& entries = cuda_ozaki.value().ledger.entries();
  if (entries.size() != 1) {
    std::cerr << "FAIL: expected 1 ledger entry, got " << entries.size()
              << '\n';
    return 1;
  }
  const auto& entry = entries[0];
  if (entry.operation != tides::tile::OperationKind::kGemm) {
    std::cerr << "FAIL: ledger operation is not kGemm\n";
    return 1;
  }
  if (entry.precision.storage != tides::tile::NumericFormat::kFloat16) {
    std::cerr << "FAIL: ledger storage is not kFloat16\n";
    return 1;
  }
  if (entry.precision.compute != tides::tile::NumericFormat::kFloat64Emulated) {
    std::cerr << "FAIL: ledger compute is not kFloat64Emulated\n";
    return 1;
  }
  if (entry.precision.determinism !=
      tides::tile::DeterminismMode::kDeterministic) {
    std::cerr << "FAIL: ledger determinism is not kDeterministic\n";
    return 1;
  }
  if (entry.candidates == 0) {
    std::cerr << "FAIL: ledger candidates is zero\n";
    return 1;
  }
  if (entry.budget.metric != tides::tile::ErrorMetric::kAbsolute) {
    std::cerr << "FAIL: ledger budget metric is not kAbsolute\n";
    return 1;
  }
  std::cout << "cuda_ozaki_ledger: entries=" << entries.size()
            << " label=" << entry.precision.label
            << " candidates=" << entry.candidates
            << " bound=" << entry.budget.bound << '\n';
  return 0;
}

}  // namespace

int main() {
  if (!CudaRuntimeAvailable()) {
    std::cout << "SKIP: CUDA runtime not available\n";
    return 77;
  }

  int failures = 0;
  failures += TestOzakiGemmVsF64eOracle();
  failures += TestOzakiGemmVsCpuOzaki();
  failures += TestOzakiGemmLargeShape();
  failures += TestOzakiGemmLedger();

  if (failures == 0) {
    std::cout << "All Ozaki CUDA GEMM tests passed.\n";
  } else {
    std::cerr << failures << " Ozaki CUDA GEMM test(s) failed.\n";
  }
  return failures;
}
