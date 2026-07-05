#include "tile/spgemm_filtered.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

using tides::tile::SpGemmFilteredFp64;
using tides::tile::TileMat;

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

std::vector<double> DenseMatmul(const std::vector<double>& a,
                                const std::vector<double>& b,
                                std::size_t n) {
  std::vector<double> c(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = 0; k < n; ++k) {
      const double aik = a[i * n + k];
      if (aik == 0.0) {
        continue;
      }
      for (std::size_t j = 0; j < n; ++j) {
        c[i * n + j] += aik * b[k * n + j];
      }
    }
  }
  return c;
}

double FrobeniusError(const std::vector<double>& a,
                      const std::vector<double>& b) {
  long double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const long double diff =
        static_cast<long double>(a[i]) - static_cast<long double>(b[i]);
    sum += diff * diff;
  }
  return std::sqrt(static_cast<double>(sum));
}

}  // namespace

int main() {
  constexpr std::size_t n = 256;
  constexpr std::uint32_t edge = 32;
  const std::vector<double> a_dense = MakeDense(n, 0.08, 0xA000ULL);
  const std::vector<double> b_dense = MakeDense(n, 0.08, 0xB000ULL);
  auto a = TileMat::FromDense(n, n, a_dense, edge);
  auto b = TileMat::FromDense(n, n, b_dense, edge);
  if (!a.ok() || !b.ok()) {
    std::cerr << "TileMat construction failed\n";
    return 1;
  }

  const std::vector<double> dense_ref = DenseMatmul(a_dense, b_dense, n);
  std::cout << "spgemm_filter_probe: n=" << n << " edge=" << edge
            << " a_tiles=" << a.value().tile_count()
            << " b_tiles=" << b.value().tile_count() << '\n';
  for (const double eps : {0.0, 1.0, 4.0, 8.0, 16.0, 32.0, 64.0}) {
    const auto start = std::chrono::steady_clock::now();
    auto filtered = SpGemmFilteredFp64(a.value(), b.value(), eps);
    const auto end = std::chrono::steady_clock::now();
    if (!filtered.ok()) {
      std::cerr << "SpGemmFilteredFp64 failed: "
                << filtered.status().message() << '\n';
      return 1;
    }
    const double ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    const double error =
        FrobeniusError(filtered.value().product.ToDense(), dense_ref);
    std::cout << "eps=" << eps
              << " candidates=" << filtered.value().ledger.candidate_products
              << " retained=" << filtered.value().ledger.retained_products
              << " dropped=" << filtered.value().ledger.dropped_products
              << " output_tiles=" << filtered.value().product.tile_count()
              << " error_frob=" << error
              << " ledger_bound="
              << filtered.value().ledger.dropped_frobenius_bound
              << " ms=" << ms << '\n';
  }
  return 0;
}
