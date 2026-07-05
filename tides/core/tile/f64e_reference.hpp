#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "common/status.hpp"
#include "tile/precision.hpp"

namespace tides::tile {

struct F64eScalarResult {
  double value = 0.0;
  double rounding_abs_error = 0.0;
  OperationLedger ledger;
};

struct F64eMatrixResult {
  std::vector<double> values;
  double max_rounding_abs_error = 0.0;
  OperationLedger ledger;
};

[[nodiscard]] inline double AbsLongDoubleToDoubleError(long double exact,
                                                       double rounded) {
  const long double diff = exact - static_cast<long double>(rounded);
  return static_cast<double>(diff < 0.0L ? -diff : diff);
}

[[nodiscard]] inline Result<F64eScalarResult> DotF64eReference(
    const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) {
    return Status::InvalidArgument("f64e dot shape mismatch");
  }
  long double sum = 0.0L;
  for (std::size_t i = 0; i < a.size(); ++i) {
    sum += static_cast<long double>(a[i]) * static_cast<long double>(b[i]);
  }

  F64eScalarResult result;
  result.value = static_cast<double>(sum);
  result.rounding_abs_error =
      AbsLongDoubleToDoubleError(sum, result.value);
  const std::uint64_t terms = static_cast<std::uint64_t>(a.size());
  result.ledger.Add(OperationLedgerEntry{
      OperationKind::kDot,
      F64eReferencePrecision(),
      ErrorBudget{ErrorMetric::kAbsolute, result.rounding_abs_error,
                  "long-double oracle rounded to FP64"},
      result.rounding_abs_error,
      terms,
      terms,
      0,
      "CPU f64e-reference dot"});
  return result;
}

[[nodiscard]] inline Result<F64eMatrixResult> GemmF64eReference(
    std::size_t m, std::size_t k, std::size_t n,
    const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != m * k || b.size() != k * n) {
    return Status::InvalidArgument("f64e gemm shape mismatch");
  }

  F64eMatrixResult result;
  result.values.assign(m * n, 0.0);
  long double total_rounding_error = 0.0L;
  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t col = 0; col < n; ++col) {
      long double sum = 0.0L;
      for (std::size_t p = 0; p < k; ++p) {
        sum += static_cast<long double>(a[row * k + p]) *
               static_cast<long double>(b[p * n + col]);
      }
      const double rounded = static_cast<double>(sum);
      const double abs_error = AbsLongDoubleToDoubleError(sum, rounded);
      result.max_rounding_abs_error =
          std::max(result.max_rounding_abs_error, abs_error);
      total_rounding_error += static_cast<long double>(abs_error);
      result.values[row * n + col] = rounded;
    }
  }

  const std::uint64_t products =
      static_cast<std::uint64_t>(m) * static_cast<std::uint64_t>(k) *
      static_cast<std::uint64_t>(n);
  result.ledger.Add(OperationLedgerEntry{
      OperationKind::kGemm,
      F64eReferencePrecision(),
      ErrorBudget{ErrorMetric::kAbsolute,
                  static_cast<double>(total_rounding_error),
                  "sum of long-double-to-FP64 rounding errors"},
      static_cast<double>(total_rounding_error),
      products,
      products,
      0,
      "CPU f64e-reference dense GEMM"});
  return result;
}

[[nodiscard]] inline Result<F64eScalarResult> SumF64eReference(
    const std::vector<double>& values, OperationKind operation,
    const char* note) {
  long double sum = 0.0L;
  for (const double value : values) {
    sum += static_cast<long double>(value);
  }

  F64eScalarResult result;
  result.value = static_cast<double>(sum);
  result.rounding_abs_error =
      AbsLongDoubleToDoubleError(sum, result.value);
  const std::uint64_t terms = static_cast<std::uint64_t>(values.size());
  result.ledger.Add(OperationLedgerEntry{
      operation,
      F64eReferencePrecision(),
      ErrorBudget{ErrorMetric::kAbsolute, result.rounding_abs_error,
                  "long-double oracle rounded to FP64"},
      result.rounding_abs_error,
      terms,
      terms,
      0,
      note});
  return result;
}

}  // namespace tides::tile
