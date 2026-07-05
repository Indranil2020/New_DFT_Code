#include "tile/reduce_f64e.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using tides::tile::DotF64eCuda;
using tides::tile::SumF64eCuda;
using tides::tile::TraceF64eCuda;

std::vector<double> MakeValues(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> mantissa(0.5, 1.0);
  std::uniform_int_distribution<int> exponent(-9, 9);
  std::bernoulli_distribution sign(0.5);
  std::vector<double> values(n, 0.0);
  for (double& value : values) {
    value = std::ldexp(mantissa(rng), exponent(rng) * 3);
    if (sign(rng)) {
      value = -value;
    }
  }
  return values;
}

}  // namespace

int main() {
  constexpr std::size_t n = 1U << 20U;
  const std::vector<double> a = MakeValues(n, 0xF64101ULL);
  const std::vector<double> b = MakeValues(n, 0xF64102ULL);

  const auto dot_start = std::chrono::steady_clock::now();
  auto dot = DotF64eCuda(a, b);
  const auto dot_end = std::chrono::steady_clock::now();
  if (!dot.ok()) {
    if (dot.status().message().find("cudaGetDeviceCount") !=
            std::string::npos ||
        dot.status().message().find("no CUDA-capable device") !=
            std::string::npos) {
      std::cerr << "CUDA runtime/device is not available: "
                << dot.status().message() << '\n';
      return 77;
    }
    std::cerr << "DotF64eCuda failed: " << dot.status().message() << '\n';
    return 1;
  }

  const auto sum_start = std::chrono::steady_clock::now();
  auto sum = SumF64eCuda(a);
  const auto sum_end = std::chrono::steady_clock::now();
  if (!sum.ok()) {
    std::cerr << "SumF64eCuda failed: " << sum.status().message() << '\n';
    return 1;
  }

  constexpr std::size_t edge = 1024;
  std::vector<double> matrix(edge * edge, 0.0);
  for (std::size_t i = 0; i < edge; ++i) {
    matrix[i * edge + i] = a[i];
  }
  const auto trace_start = std::chrono::steady_clock::now();
  auto trace = TraceF64eCuda(edge, edge, matrix);
  const auto trace_end = std::chrono::steady_clock::now();
  if (!trace.ok()) {
    std::cerr << "TraceF64eCuda failed: " << trace.status().message() << '\n';
    return 1;
  }

  const double dot_wall_ms =
      std::chrono::duration<double, std::milli>(dot_end - dot_start).count();
  const double sum_wall_ms =
      std::chrono::duration<double, std::milli>(sum_end - sum_start).count();
  const double trace_wall_ms =
      std::chrono::duration<double, std::milli>(trace_end - trace_start)
          .count();
  std::cout << "cuda_reduce_f64e_probe: n=" << n
            << " dot_wall_ms=" << dot_wall_ms
            << " dot_kernel_ms=" << dot.value().kernel_ms
            << " dot_gterms_s="
            << static_cast<double>(n) / (dot.value().kernel_ms * 1.0e6)
            << " dot_abs_error=" << dot.value().abs_error_vs_long_double
            << " dot_abs_bound=" << dot.value().analytical_abs_bound
            << " sum_wall_ms=" << sum_wall_ms
            << " sum_kernel_ms=" << sum.value().kernel_ms
            << " sum_gterms_s="
            << static_cast<double>(n) / (sum.value().kernel_ms * 1.0e6)
            << " trace_edge=" << edge
            << " trace_wall_ms=" << trace_wall_ms
            << " trace_kernel_ms=" << trace.value().kernel_ms
            << " trace_abs_error=" << trace.value().abs_error_vs_long_double
            << '\n';
  return 0;
}
