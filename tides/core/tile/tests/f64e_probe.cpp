#include "tile/f64e_reference.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

using tides::tile::DotF64eReference;
using tides::tile::GemmF64eReference;

double NaiveDot(const std::vector<double>& a, const std::vector<double>& b) {
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

}  // namespace

int main() {
  constexpr std::size_t repeats = 100000;
  std::vector<double> a;
  std::vector<double> b;
  a.reserve(3 * repeats);
  b.reserve(3 * repeats);
  for (std::size_t i = 0; i < repeats; ++i) {
    a.push_back(1.0e16);
    b.push_back(1.0);
    a.push_back(1.0);
    b.push_back(1.0);
    a.push_back(-1.0e16);
    b.push_back(1.0);
  }

  const auto naive_start = std::chrono::steady_clock::now();
  const double naive = NaiveDot(a, b);
  const auto naive_end = std::chrono::steady_clock::now();
  const auto f64e_start = std::chrono::steady_clock::now();
  auto f64e = DotF64eReference(a, b);
  const auto f64e_end = std::chrono::steady_clock::now();
  if (!f64e.ok()) {
    std::cerr << "DotF64eReference failed: " << f64e.status().message()
              << '\n';
    return 1;
  }

  constexpr std::size_t m = 48;
  constexpr std::size_t k = 48;
  constexpr std::size_t n = 48;
  std::vector<double> gemm_a(m * k, 0.0);
  std::vector<double> gemm_b(k * n, 0.0);
  for (std::size_t i = 0; i < gemm_a.size(); ++i) {
    gemm_a[i] = (i % 9 == 0) ? 1.0e8 : 1.0e-4;
  }
  for (std::size_t i = 0; i < gemm_b.size(); ++i) {
    gemm_b[i] = (i % 13 == 0) ? -1.0e8 : 1.0e-4;
  }
  const auto gemm_start = std::chrono::steady_clock::now();
  auto gemm = GemmF64eReference(m, k, n, gemm_a, gemm_b);
  const auto gemm_end = std::chrono::steady_clock::now();
  if (!gemm.ok()) {
    std::cerr << "GemmF64eReference failed: " << gemm.status().message()
              << '\n';
    return 1;
  }

  const double naive_ms =
      std::chrono::duration<double, std::milli>(naive_end - naive_start)
          .count();
  const double f64e_ms =
      std::chrono::duration<double, std::milli>(f64e_end - f64e_start)
          .count();
  const double gemm_ms =
      std::chrono::duration<double, std::milli>(gemm_end - gemm_start)
          .count();
  std::cout << "f64e_probe: dot_terms=" << a.size()
            << " naive=" << naive
            << " f64e=" << f64e.value().value
            << " expected=" << repeats
            << " naive_ms=" << naive_ms
            << " f64e_ms=" << f64e_ms
            << " gemm_shape=" << m << "x" << k << "x" << n
            << " gemm_ms=" << gemm_ms
            << " gemm_max_rounding_abs_error="
            << gemm.value().max_rounding_abs_error << '\n';
  return 0;
}
