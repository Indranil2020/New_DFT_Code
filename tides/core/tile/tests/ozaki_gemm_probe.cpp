#include "tile/f64e_reference.hpp"
#include "tile/ozaki.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

using tides::tile::GemmF64eReference;
using tides::tile::GemmOzakiFp16Reference;

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

}  // namespace

int main() {
  constexpr std::size_t m = 32;
  constexpr std::size_t k = 32;
  constexpr std::size_t n = 32;
  const std::vector<double> a =
      MakeDynamicRangeMatrix(m, k, 0xC0DEC0DEULL);
  const std::vector<double> b =
      MakeDynamicRangeMatrix(k, n, 0xFACEB00CULL);

  const auto f64e_start = std::chrono::steady_clock::now();
  auto f64e = GemmF64eReference(m, k, n, a, b);
  const auto f64e_end = std::chrono::steady_clock::now();
  const auto ozaki_start = std::chrono::steady_clock::now();
  auto ozaki = GemmOzakiFp16Reference(m, k, n, a, b);
  const auto ozaki_end = std::chrono::steady_clock::now();
  if (!f64e.ok()) {
    std::cerr << "GemmF64eReference failed: " << f64e.status().message()
              << '\n';
    return 1;
  }
  if (!ozaki.ok()) {
    std::cerr << "GemmOzakiFp16Reference failed: "
              << ozaki.status().message() << '\n';
    return 1;
  }

  const double f64e_ms =
      std::chrono::duration<double, std::milli>(f64e_end - f64e_start).count();
  const double ozaki_ms =
      std::chrono::duration<double, std::milli>(ozaki_end - ozaki_start)
          .count();
  std::cout << "ozaki_gemm_probe: shape=" << m << "x" << k << "x" << n
            << " a_slices=" << ozaki.value().a_plan.slice_count
            << " b_slices=" << ozaki.value().b_plan.slice_count
            << " slice_pairs=" << ozaki.value().slice_pair_count
            << " f64e_ms=" << f64e_ms
            << " ozaki_ref_ms=" << ozaki_ms
            << " max_abs_diff="
            << MaxAbsDifference(ozaki.value().values, f64e.value().values)
            << " observed_bound="
            << ozaki.value().max_abs_error_vs_long_double
            << " fp64_forward_bound="
            << ozaki.value().max_fp64_forward_abs_bound << '\n';
  return 0;
}
