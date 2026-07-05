#include "tile/gemm_grouped.hpp"
#include "tile/spgemm_filtered.hpp"
#include "tile/spgemm_filtered_cuda.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

using tides::tile::CudaRuntimeStatus;
using tides::tile::SpGemmFilteredFp64;
using tides::tile::SpGemmFilteredFp64Cuda;
using tides::tile::TileMat;

std::vector<double> MakeDense(std::size_t rows, std::size_t cols,
                              double keep_probability, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::bernoulli_distribution keep(keep_probability);
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  std::vector<double> dense(rows * cols, 0.0);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      if (keep(rng)) {
        dense[i * cols + j] = value(rng);
      }
    }
  }
  return dense;
}

}  // namespace

int main() {
  const auto runtime_status = CudaRuntimeStatus();
  if (!runtime_status.ok()) {
    std::cerr << "CUDA runtime/device is not available: "
              << runtime_status.message() << '\n';
    return 77;
  }
  constexpr std::size_t n = 256;
  constexpr std::uint32_t edge = 32;
  const std::vector<double> a_dense = MakeDense(n, n, 0.08, 0x111000ULL);
  const std::vector<double> b_dense = MakeDense(n, n, 0.08, 0x222000ULL);
  auto a = TileMat::FromDense(n, n, a_dense, edge);
  auto b = TileMat::FromDense(n, n, b_dense, edge);
  if (!a.ok() || !b.ok()) {
    std::cerr << "TileMat construction failed\n";
    return 1;
  }

  std::cout << "cuda_spgemm_probe: n=" << n << " edge=" << edge
            << " a_tiles=" << a.value().tile_count()
            << " b_tiles=" << b.value().tile_count() << '\n';
  for (const double eps : {0.0, 4.0, 16.0, 32.0, 64.0}) {
    const auto cpu_start = std::chrono::steady_clock::now();
    auto cpu = SpGemmFilteredFp64(a.value(), b.value(), eps);
    const auto cpu_end = std::chrono::steady_clock::now();
    const auto gpu_start = std::chrono::steady_clock::now();
    auto gpu = SpGemmFilteredFp64Cuda(a.value(), b.value(), eps);
    const auto gpu_end = std::chrono::steady_clock::now();
    if (!cpu.ok() || !gpu.ok()) {
      std::cerr << "SpGEMM failed\n";
      return 1;
    }
    const double cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    const double gpu_total_ms =
        std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    std::cout << "eps=" << eps
              << " candidates=" << gpu.value().ledger.candidate_products
              << " retained=" << gpu.value().ledger.retained_products
              << " dropped=" << gpu.value().ledger.dropped_products
              << " output_tiles=" << gpu.value().product.tile_count()
              << " ledger_bound="
              << gpu.value().ledger.dropped_frobenius_bound
              << " cpu_ms=" << cpu_ms
              << " gpu_total_ms=" << gpu_total_ms
              << " gpu_kernel_ms=" << gpu.value().kernel_ms << '\n';
  }
  return 0;
}
