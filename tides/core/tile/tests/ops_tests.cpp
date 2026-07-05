#include "tile/ops.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using tides::tile::AxpyFp64;
using tides::tile::FrobeniusNormF64eReference;
using tides::tile::OperationKind;
using tides::tile::Symmetry;
using tides::tile::TileMat;
using tides::tile::TraceF64eReference;
using tides::tile::ValidateOperationLedgerEntry;

int Fail(const std::string& message) {
  std::cerr << "ops_tests: " << message << '\n';
  return 1;
}

std::vector<double> MakeDense(std::size_t rows, std::size_t cols,
                              double keep_probability, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::bernoulli_distribution keep(keep_probability);
  std::uniform_real_distribution<double> value(-3.0, 3.0);
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
  std::bernoulli_distribution keep(0.22);
  std::uniform_real_distribution<double> value(-2.0, 2.0);
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

bool ExactEqual(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

int AxpyMatchesDenseReference() {
  constexpr std::uint32_t edge = 16;
  for (int case_id = 0; case_id < 20; ++case_id) {
    const std::size_t rows = 7 + (case_id * 11) % 65;
    const std::size_t cols = 9 + (case_id * 13) % 69;
    const std::vector<double> x_dense =
        MakeDense(rows, cols, 0.18, 0x9000ULL + case_id);
    const std::vector<double> y_dense =
        MakeDense(rows, cols, 0.21, 0xA000ULL + case_id);
    auto x = TileMat::FromDense(rows, cols, x_dense, edge);
    auto y = TileMat::FromDense(rows, cols, y_dense, edge);
    if (!x.ok() || !y.ok()) {
      return Fail("TileMat construction failed");
    }
    const double alpha = 1.25;
    const double beta = -0.75;
    auto result = AxpyFp64(alpha, x.value(), beta, y.value());
    if (!result.ok()) {
      return Fail("AxpyFp64 failed: " + result.status().message());
    }
    std::vector<double> dense_ref(rows * cols, 0.0);
    for (std::size_t i = 0; i < dense_ref.size(); ++i) {
      dense_ref[i] = alpha * x_dense[i] + beta * y_dense[i];
    }
    if (!ExactEqual(result.value().matrix.ToDense(), dense_ref)) {
      return Fail("AXPY result differs from dense FP64 reference");
    }
    const auto& entries = result.value().ledger.entries();
    if (entries.size() != 1 || entries.front().operation != OperationKind::kAxpy) {
      return Fail("AXPY did not emit the expected ledger entry");
    }
    if (!ValidateOperationLedgerEntry(entries.front()).ok()) {
      return Fail("AXPY ledger entry failed validation");
    }
  }
  return 0;
}

int SymmetricAxpyPreservesSymmetry() {
  constexpr std::uint32_t edge = 32;
  const std::vector<double> x_dense = MakeSymmetricDense(70, 0xB000ULL);
  const std::vector<double> y_dense = MakeSymmetricDense(70, 0xC000ULL);
  auto x = TileMat::FromDense(70, 70, x_dense, edge, Symmetry::kSymmetric);
  auto y = TileMat::FromDense(70, 70, y_dense, edge, Symmetry::kSymmetric);
  if (!x.ok() || !y.ok()) {
    return Fail("symmetric TileMat construction failed");
  }
  auto result = AxpyFp64(0.5, x.value(), 2.0, y.value());
  if (!result.ok()) {
    return Fail("symmetric AxpyFp64 failed: " + result.status().message());
  }
  if (result.value().matrix.symmetry() != Symmetry::kSymmetric) {
    return Fail("symmetric AXPY did not preserve symmetry flag");
  }
  const std::vector<double> dense = result.value().matrix.ToDense();
  for (std::size_t i = 0; i < 70; ++i) {
    for (std::size_t j = i + 1; j < 70; ++j) {
      if (dense[i * 70 + j] != dense[j * 70 + i]) {
        return Fail("symmetric AXPY output is not exactly symmetric");
      }
    }
  }
  return 0;
}

int TraceAndNormReferences() {
  constexpr std::uint32_t edge = 16;
  const std::vector<double> dense = MakeDense(79, 83, 0.2, 0xD000ULL);
  auto matrix = TileMat::FromDense(79, 83, dense, edge);
  if (!matrix.ok()) {
    return Fail("TileMat construction failed for reductions");
  }
  auto trace = TraceF64eReference(matrix.value());
  auto norm = FrobeniusNormF64eReference(matrix.value());
  if (!trace.ok() || !norm.ok()) {
    return Fail("reduction operation failed");
  }
  double dense_trace = 0.0;
  long double dense_norm2 = 0.0;
  for (std::size_t i = 0; i < 79; ++i) {
    dense_trace += dense[i * 83 + i];
  }
  for (const double v : dense) {
    dense_norm2 += static_cast<long double>(v) * static_cast<long double>(v);
  }
  if (trace.value().value != dense_trace) {
    return Fail("trace reduction differs from dense reference");
  }
  if (norm.value().value != std::sqrt(static_cast<double>(dense_norm2))) {
    return Fail("norm reduction differs from dense reference");
  }
  if (!ValidateOperationLedgerEntry(trace.value().ledger.entries().front()).ok() ||
      !ValidateOperationLedgerEntry(norm.value().ledger.entries().front()).ok()) {
    return Fail("reduction ledger validation failed");
  }
  return 0;
}

int DeterministicRepeatedOps() {
  constexpr std::uint32_t edge = 64;
  const std::vector<double> x_dense = MakeDense(128, 128, 0.07, 0xE000ULL);
  const std::vector<double> y_dense = MakeDense(128, 128, 0.08, 0xF000ULL);
  auto x = TileMat::FromDense(128, 128, x_dense, edge);
  auto y = TileMat::FromDense(128, 128, y_dense, edge);
  if (!x.ok() || !y.ok()) {
    return Fail("TileMat construction failed for determinism");
  }
  auto first = AxpyFp64(-1.0, x.value(), 1.0, y.value());
  auto second = AxpyFp64(-1.0, x.value(), 1.0, y.value());
  if (!first.ok() || !second.ok()) {
    return Fail("deterministic AXPY failed");
  }
  if (first.value().matrix.ToDense() != second.value().matrix.ToDense()) {
    return Fail("repeated AXPY changed output bits");
  }
  auto first_trace = TraceF64eReference(first.value().matrix);
  auto second_trace = TraceF64eReference(second.value().matrix);
  if (!first_trace.ok() || !second_trace.ok() ||
      first_trace.value().value != second_trace.value().value) {
    return Fail("repeated trace changed output bits");
  }
  return 0;
}

}  // namespace

int main() {
  if (const int rc = AxpyMatchesDenseReference(); rc != 0) {
    return rc;
  }
  if (const int rc = SymmetricAxpyPreservesSymmetry(); rc != 0) {
    return rc;
  }
  if (const int rc = TraceAndNormReferences(); rc != 0) {
    return rc;
  }
  if (const int rc = DeterministicRepeatedOps(); rc != 0) {
    return rc;
  }
  std::cout << "ops_tests: deterministic TileMat ops passed\n";
  return 0;
}
