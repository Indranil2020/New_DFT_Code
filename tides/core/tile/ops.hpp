#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/status.hpp"
#include "tile/f64e_reference.hpp"
#include "tile/layout.hpp"
#include "tile/precision.hpp"

namespace tides::tile {

struct TileMatResult {
  TileMat matrix;
  OperationLedger ledger;
};

struct ScalarResult {
  double value = 0.0;
  OperationLedger ledger;
};

[[nodiscard]] inline Result<TileMatResult> AxpyFp64(double alpha,
                                                    const TileMat& x,
                                                    double beta,
                                                    const TileMat& y) {
  if (x.rows() != y.rows() || x.cols() != y.cols()) {
    return Status::InvalidArgument("AXPY shape mismatch");
  }
  if (x.tile_edge() != y.tile_edge()) {
    return Status::InvalidArgument("AXPY requires matching tile edges");
  }

  const std::vector<double> x_dense = x.ToDense();
  const std::vector<double> y_dense = y.ToDense();
  std::vector<double> out_dense(x_dense.size(), 0.0);
  for (std::size_t i = 0; i < out_dense.size(); ++i) {
    out_dense[i] = alpha * x_dense[i] + beta * y_dense[i];
  }

  const Symmetry symmetry =
      x.symmetry() == Symmetry::kSymmetric &&
              y.symmetry() == Symmetry::kSymmetric
          ? Symmetry::kSymmetric
          : Symmetry::kGeneral;
  auto out =
      TileMat::FromDense(x.rows(), x.cols(), out_dense, x.tile_edge(), symmetry);
  if (!out.ok()) {
    return out.status();
  }

  OperationLedger ledger;
  const std::uint64_t cells =
      static_cast<std::uint64_t>(x.rows()) * static_cast<std::uint64_t>(x.cols());
  ledger.Add(OperationLedgerEntry{
      OperationKind::kAxpy,
      Fp64ReferencePrecision(),
      ErrorBudget{ErrorMetric::kNone, 0.0, "FP64 reference axpy"},
      0.0,
      cells,
      cells,
      0,
      "CPU FP64 deterministic TileMat axpy"});
  return TileMatResult{out.take_value(), std::move(ledger)};
}

[[nodiscard]] inline Result<ScalarResult> TraceF64eReference(
    const TileMat& matrix) {
  ScalarResult result;
  const std::vector<double> dense = matrix.ToDense();
  const std::size_t diagonal = std::min(matrix.rows(), matrix.cols());
  std::vector<double> diagonal_values(diagonal, 0.0);
  for (std::size_t i = 0; i < diagonal; ++i) {
    diagonal_values[i] = dense[i * matrix.cols() + i];
  }
  auto sum = SumF64eReference(diagonal_values, OperationKind::kTrace,
                              "CPU f64e-reference trace reduction");
  if (!sum.ok()) {
    return sum.status();
  }
  result.value = sum.value().value;
  result.ledger = std::move(sum.value().ledger);
  return result;
}

[[nodiscard]] inline Result<ScalarResult> FrobeniusNormF64eReference(
    const TileMat& matrix) {
  ScalarResult result;
  const std::vector<double> dense = matrix.ToDense();
  std::vector<double> squares(dense.size(), 0.0);
  for (std::size_t i = 0; i < dense.size(); ++i) {
    squares[i] = dense[i] * dense[i];
  }
  auto sum = SumF64eReference(squares, OperationKind::kNorm,
                              "CPU f64e-reference Frobenius norm reduction");
  if (!sum.ok()) {
    return sum.status();
  }
  result.value = std::sqrt(sum.value().value);
  result.ledger = std::move(sum.value().ledger);
  return result;
}

}  // namespace tides::tile
