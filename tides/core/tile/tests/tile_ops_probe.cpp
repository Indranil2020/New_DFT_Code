#include "tile/ops.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

using tides::tile::AxpyFp64;
using tides::tile::FrobeniusNormF64eReference;
using tides::tile::TileMat;
using tides::tile::TraceF64eReference;

std::vector<double> MakeDense(std::size_t n, double keep_probability,
                              std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::bernoulli_distribution keep(keep_probability);
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  std::vector<double> dense(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (keep(rng)) {
        dense[i * n + j] = value(rng);
      }
    }
  }
  return dense;
}

}  // namespace

int main() {
  constexpr std::size_t n = 1024;
  constexpr std::uint32_t edge = 32;
  const std::vector<double> x_dense = MakeDense(n, 0.015, 0xABCULL);
  const std::vector<double> y_dense = MakeDense(n, 0.015, 0xDEFULL);
  auto x = TileMat::FromDense(n, n, x_dense, edge);
  auto y = TileMat::FromDense(n, n, y_dense, edge);
  if (!x.ok() || !y.ok()) {
    std::cerr << "TileMat construction failed\n";
    return 1;
  }

  const auto axpy_start = std::chrono::steady_clock::now();
  auto axpy = AxpyFp64(1.25, x.value(), -0.5, y.value());
  const auto axpy_end = std::chrono::steady_clock::now();
  if (!axpy.ok()) {
    std::cerr << "AxpyFp64 failed: " << axpy.status().message() << '\n';
    return 1;
  }

  const auto trace_start = std::chrono::steady_clock::now();
  auto trace = TraceF64eReference(axpy.value().matrix);
  const auto trace_end = std::chrono::steady_clock::now();
  const auto norm_start = std::chrono::steady_clock::now();
  auto norm = FrobeniusNormF64eReference(axpy.value().matrix);
  const auto norm_end = std::chrono::steady_clock::now();
  if (!trace.ok() || !norm.ok()) {
    std::cerr << "reduction failed\n";
    return 1;
  }

  const double axpy_ms =
      std::chrono::duration<double, std::milli>(axpy_end - axpy_start).count();
  const double trace_ms =
      std::chrono::duration<double, std::milli>(trace_end - trace_start).count();
  const double norm_ms =
      std::chrono::duration<double, std::milli>(norm_end - norm_start).count();
  std::cout << "tile_ops_probe: n=" << n << " edge=" << edge
            << " x_tiles=" << x.value().tile_count()
            << " y_tiles=" << y.value().tile_count()
            << " out_tiles=" << axpy.value().matrix.tile_count()
            << " axpy_ms=" << axpy_ms
            << " trace_ms=" << trace_ms
            << " norm_ms=" << norm_ms
            << " trace=" << trace.value().value
            << " norm=" << norm.value().value << '\n';
  return 0;
}
