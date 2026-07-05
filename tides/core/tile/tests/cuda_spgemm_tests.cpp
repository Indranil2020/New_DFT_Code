#include "tile/spgemm_filtered.hpp"
#include "tile/spgemm_filtered_cuda.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "tile/gemm_grouped.hpp"

namespace {

using tides::tile::CudaRuntimeStatus;
using tides::tile::SpGemmFilteredFp64;
using tides::tile::SpGemmFilteredFp64Cuda;
using tides::tile::TileMat;
using tides::tile::ValidateOperationLedgerEntry;

int Fail(const std::string& message) {
  std::cerr << "cuda_spgemm_tests: " << message << '\n';
  return 1;
}

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

double FrobeniusError(const std::vector<double>& a,
                      const std::vector<double>& b) {
  long double sum = 0.0L;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const long double diff =
        static_cast<long double>(a[i]) - static_cast<long double>(b[i]);
    sum += diff * diff;
  }
  return std::sqrt(static_cast<double>(sum));
}

}  // namespace

int main() {
  const auto runtime_status = CudaRuntimeStatus();
  if (!runtime_status.ok()) {
    std::cerr << "cuda_spgemm_tests: SKIP " << runtime_status.message()
              << '\n';
    return 77;
  }

  constexpr std::uint32_t edge = 16;
  const std::vector<double> a_dense = MakeDense(80, 80, 0.16, 0xABC000ULL);
  const std::vector<double> b_dense = MakeDense(80, 80, 0.14, 0xDEF000ULL);
  auto a = TileMat::FromDense(80, 80, a_dense, edge);
  auto b = TileMat::FromDense(80, 80, b_dense, edge);
  if (!a.ok() || !b.ok()) {
    return Fail("TileMat construction failed");
  }
  for (const double eps : {0.0, 4.0, 16.0, 64.0}) {
    auto cpu = SpGemmFilteredFp64(a.value(), b.value(), eps);
    auto gpu = SpGemmFilteredFp64Cuda(a.value(), b.value(), eps);
    if (!cpu.ok()) {
      return Fail("CPU SpGEMM failed: " + cpu.status().message());
    }
    if (!gpu.ok()) {
      return Fail("CUDA SpGEMM failed: " + gpu.status().message());
    }
    const double error =
        FrobeniusError(cpu.value().product.ToDense(),
                       gpu.value().product.ToDense());
    if (error > 1.0e-12 * (1.0 + cpu.value().product.FrobeniusNormFp64())) {
      return Fail("CUDA filtered SpGEMM differs from CPU oracle");
    }
    if (cpu.value().ledger.candidate_products !=
            gpu.value().ledger.candidate_products ||
        cpu.value().ledger.retained_products !=
            gpu.value().ledger.retained_products ||
        cpu.value().ledger.dropped_products !=
            gpu.value().ledger.dropped_products ||
        cpu.value().ledger.dropped_frobenius_bound !=
            gpu.value().ledger.dropped_frobenius_bound) {
      return Fail("CUDA SpGEMM ledger differs from CPU oracle");
    }
    if (!ValidateOperationLedgerEntry(
             gpu.value().ledger.operation_ledger.entries().front())
             .ok()) {
      return Fail("CUDA SpGEMM operation ledger is invalid");
    }
  }

  std::cout << "cuda_spgemm_tests: CUDA filtered SpGEMM matched CPU oracle\n";
  return 0;
}
