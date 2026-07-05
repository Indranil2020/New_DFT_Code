#include "tile/spgemm_filtered.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using tides::tile::SpGemmFilteredFp64;
using tides::tile::Symmetry;
using tides::tile::TileMat;

int Fail(const std::string& message) {
  std::cerr << "spgemm_filtered_tests: " << message << '\n';
  return 1;
}

std::vector<double> MakeDense(std::size_t rows, std::size_t cols,
                              double keep_probability, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::bernoulli_distribution keep(keep_probability);
  std::uniform_real_distribution<double> value(-2.0, 2.0);
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

std::vector<double> MakeSymmetricDense(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::bernoulli_distribution keep(0.2);
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  std::vector<double> dense(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i; j < n; ++j) {
      if (i == j || keep(rng)) {
        const double v = value(rng);
        dense[i * n + j] = v;
        dense[j * n + i] = v;
      }
    }
  }
  return dense;
}

std::vector<double> DenseMatmul(const std::vector<double>& a,
                                const std::vector<double>& b,
                                std::size_t rows, std::size_t inner,
                                std::size_t cols) {
  std::vector<double> c(rows * cols, 0.0);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t k = 0; k < inner; ++k) {
      const double aik = a[i * inner + k];
      if (aik == 0.0) {
        continue;
      }
      for (std::size_t j = 0; j < cols; ++j) {
        c[i * cols + j] += aik * b[k * cols + j];
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

double FrobeniusNorm(const std::vector<double>& a) {
  long double sum = 0.0;
  for (const double v : a) {
    sum += static_cast<long double>(v) * static_cast<long double>(v);
  }
  return std::sqrt(static_cast<double>(sum));
}

double TileFrobeniusError(const std::vector<double>& a,
                          const std::vector<double>& b, std::size_t rows,
                          std::size_t cols, std::size_t block_row,
                          std::size_t block_col, std::uint32_t edge) {
  long double sum = 0.0;
  const std::size_t row0 = block_row * edge;
  const std::size_t col0 = block_col * edge;
  const std::size_t row_extent = std::min<std::size_t>(edge, rows - row0);
  const std::size_t col_extent = std::min<std::size_t>(edge, cols - col0);
  for (std::size_t i = 0; i < row_extent; ++i) {
    for (std::size_t j = 0; j < col_extent; ++j) {
      const long double diff =
          static_cast<long double>(a[(row0 + i) * cols + col0 + j]) -
          static_cast<long double>(b[(row0 + i) * cols + col0 + j]);
      sum += diff * diff;
    }
  }
  return std::sqrt(static_cast<double>(sum));
}

bool Near(const std::vector<double>& a, const std::vector<double>& b,
          double tolerance) {
  return FrobeniusError(a, b) <= tolerance * (1.0 + FrobeniusNorm(b));
}

int ExactUnfilteredMatchesDenseReference() {
  constexpr std::uint32_t edge = 16;
  for (int case_id = 0; case_id < 24; ++case_id) {
    const std::size_t rows = 9 + (case_id * 7) % 43;
    const std::size_t inner = 11 + (case_id * 5) % 39;
    const std::size_t cols = 13 + (case_id * 3) % 37;
    const std::vector<double> a_dense =
        MakeDense(rows, inner, 0.22, 0x1000ULL + case_id);
    const std::vector<double> b_dense =
        MakeDense(inner, cols, 0.19, 0x2000ULL + case_id);
    auto a = TileMat::FromDense(rows, inner, a_dense, edge);
    auto b = TileMat::FromDense(inner, cols, b_dense, edge);
    if (!a.ok() || !b.ok()) {
      return Fail("TileMat construction failed in exact case");
    }
    auto product = SpGemmFilteredFp64(a.value(), b.value(), 0.0);
    if (!product.ok()) {
      return Fail("SpGemmFilteredFp64 failed: " + product.status().message());
    }
    const std::vector<double> dense_ref =
        DenseMatmul(a_dense, b_dense, rows, inner, cols);
    if (!Near(product.value().product.ToDense(), dense_ref, 1.0e-14)) {
      return Fail("unfiltered SpGEMM differs from dense FP64 reference");
    }
    if (product.value().ledger.dropped_products != 0 ||
        product.value().ledger.dropped_frobenius_bound != 0.0) {
      return Fail("eps=0 dropped a nonzero tile product");
    }
    if (product.value().ledger.candidate_products !=
        product.value().ledger.retained_products) {
      return Fail("candidate/retained counts disagree for eps=0");
    }
  }
  return 0;
}

int FilteringLedgerBoundsDenseError() {
  constexpr std::uint32_t edge = 16;
  const std::vector<double> a_dense = MakeDense(80, 80, 0.14, 0x3000ULL);
  const std::vector<double> b_dense = MakeDense(80, 80, 0.12, 0x4000ULL);
  auto a = TileMat::FromDense(80, 80, a_dense, edge);
  auto b = TileMat::FromDense(80, 80, b_dense, edge);
  if (!a.ok() || !b.ok()) {
    return Fail("TileMat construction failed in filtering case");
  }
  const std::vector<double> dense_ref =
      DenseMatmul(a_dense, b_dense, 80, 80, 80);
  bool saw_dropped_products = false;
  for (const double eps : {0.1, 1.0, 10.0, 100.0}) {
    auto filtered = SpGemmFilteredFp64(a.value(), b.value(), eps);
    if (!filtered.ok()) {
      return Fail("filtered SpGEMM failed: " + filtered.status().message());
    }
    const double error =
        FrobeniusError(filtered.value().product.ToDense(), dense_ref);
    if (error > filtered.value().ledger.dropped_frobenius_bound + 1.0e-10) {
      return Fail("ledger bound was violated at eps=" + std::to_string(eps));
    }
    std::map<std::pair<std::size_t, std::size_t>, double> tile_bounds;
    for (const auto& bound : filtered.value().ledger.output_tile_bounds) {
      tile_bounds[{bound.block_row, bound.block_col}] =
          bound.frobenius_bound;
    }
    const std::vector<double> filtered_dense =
        filtered.value().product.ToDense();
    for (std::size_t br = 0; br < 5; ++br) {
      for (std::size_t bc = 0; bc < 5; ++bc) {
        const double tile_error = TileFrobeniusError(
            filtered_dense, dense_ref, 80, 80, br, bc, edge);
        const auto it = tile_bounds.find({br, bc});
        const double tile_bound =
            it == tile_bounds.end() ? 0.0 : it->second;
        if (tile_error > tile_bound + 1.0e-10) {
          return Fail("per-output tile ledger bound was violated");
        }
      }
    }
    if (filtered.value().ledger.candidate_products !=
        filtered.value().ledger.retained_products +
            filtered.value().ledger.dropped_products) {
      return Fail("candidate product count is not conserved");
    }
    saw_dropped_products =
        saw_dropped_products || filtered.value().ledger.dropped_products > 0;
  }
  if (!saw_dropped_products) {
    return Fail("filtering test never exercised a dropped tile product");
  }
  return 0;
}

int DeterministicRepeatedRuns() {
  constexpr std::uint32_t edge = 32;
  const std::vector<double> a_dense = MakeDense(96, 96, 0.11, 0x5000ULL);
  const std::vector<double> b_dense = MakeDense(96, 96, 0.10, 0x6000ULL);
  auto a = TileMat::FromDense(96, 96, a_dense, edge);
  auto b = TileMat::FromDense(96, 96, b_dense, edge);
  if (!a.ok() || !b.ok()) {
    return Fail("TileMat construction failed in deterministic case");
  }
  auto first = SpGemmFilteredFp64(a.value(), b.value(), 2.0);
  auto second = SpGemmFilteredFp64(a.value(), b.value(), 2.0);
  if (!first.ok() || !second.ok()) {
    return Fail("deterministic SpGEMM run failed");
  }
  if (first.value().product.ToDense() != second.value().product.ToDense()) {
    return Fail("repeated filtered SpGEMM changed product bits");
  }
  if (first.value().ledger.dropped_frobenius_bound !=
          second.value().ledger.dropped_frobenius_bound ||
      first.value().ledger.output_tile_bounds.size() !=
          second.value().ledger.output_tile_bounds.size()) {
    return Fail("repeated filtered SpGEMM changed ledger bits");
  }
  return 0;
}

int SymmetricInputsUseExpandedDenseSemantics() {
  constexpr std::uint32_t edge = 16;
  const std::vector<double> a_dense = MakeSymmetricDense(48, 0x7000ULL);
  const std::vector<double> b_dense = MakeSymmetricDense(48, 0x8000ULL);
  auto a = TileMat::FromDense(48, 48, a_dense, edge, Symmetry::kSymmetric);
  auto b = TileMat::FromDense(48, 48, b_dense, edge, Symmetry::kSymmetric);
  if (!a.ok() || !b.ok()) {
    return Fail("symmetric TileMat construction failed");
  }
  auto product = SpGemmFilteredFp64(a.value(), b.value(), 0.0);
  if (!product.ok()) {
    return Fail("symmetric SpGEMM failed: " + product.status().message());
  }
  const std::vector<double> dense_ref =
      DenseMatmul(a_dense, b_dense, 48, 48, 48);
  if (!Near(product.value().product.ToDense(), dense_ref, 1.0e-14)) {
    return Fail("symmetric input SpGEMM did not use full dense semantics");
  }
  return 0;
}

}  // namespace

int main() {
  if (const int rc = ExactUnfilteredMatchesDenseReference(); rc != 0) {
    return rc;
  }
  if (const int rc = FilteringLedgerBoundsDenseError(); rc != 0) {
    return rc;
  }
  if (const int rc = DeterministicRepeatedRuns(); rc != 0) {
    return rc;
  }
  if (const int rc = SymmetricInputsUseExpandedDenseSemantics(); rc != 0) {
    return rc;
  }
  std::cout << "spgemm_filtered_tests: analytical filter bounds passed\n";
  return 0;
}
