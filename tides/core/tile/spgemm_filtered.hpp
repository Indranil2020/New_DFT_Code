#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "common/status.hpp"
#include "tile/layout.hpp"
#include "tile/precision.hpp"

namespace tides::tile {

struct TileErrorBound {
  std::size_t block_row = 0;
  std::size_t block_col = 0;
  double frobenius_bound = 0.0;
};

struct ErrorLedger {
  double eps_filter = 0.0;
  std::size_t candidate_products = 0;
  std::size_t retained_products = 0;
  std::size_t dropped_products = 0;
  double dropped_frobenius_bound = 0.0;
  std::vector<TileErrorBound> output_tile_bounds;
  OperationLedger operation_ledger;
};

struct SpGemmFilteredResult {
  TileMat product;
  ErrorLedger ledger;
};

namespace detail {

struct ExpandedTile {
  std::size_t block_row = 0;
  std::size_t block_col = 0;
  std::size_t row_start = 0;
  std::size_t col_start = 0;
  std::size_t row_extent = 0;
  std::size_t col_extent = 0;
  std::vector<double> values;
  double frobenius_norm = 0.0;
};

[[nodiscard]] inline std::size_t CeilDiv(std::size_t value,
                                         std::size_t divisor) {
  return value == 0 ? 0 : 1 + (value - 1) / divisor;
}

[[nodiscard]] inline std::vector<ExpandedTile> ExpandToFullTiles(
    const TileMat& matrix) {
  const std::uint32_t edge = matrix.tile_edge();
  const std::size_t block_rows = CeilDiv(matrix.rows(), edge);
  const std::size_t block_cols = CeilDiv(matrix.cols(), edge);
  const std::vector<double> dense = matrix.ToDense();

  std::vector<ExpandedTile> tiles;
  for (std::size_t br = 0; br < block_rows; ++br) {
    for (std::size_t bc = 0; bc < block_cols; ++bc) {
      const std::size_t row0 = br * edge;
      const std::size_t col0 = bc * edge;
      const std::size_t row_extent =
          std::min<std::size_t>(edge, matrix.rows() - row0);
      const std::size_t col_extent =
          std::min<std::size_t>(edge, matrix.cols() - col0);
      ExpandedTile tile;
      tile.block_row = br;
      tile.block_col = bc;
      tile.row_start = row0;
      tile.col_start = col0;
      tile.row_extent = row_extent;
      tile.col_extent = col_extent;
      tile.values.assign(static_cast<std::size_t>(edge) * edge, 0.0);

      bool has_nonzero = false;
      long double norm2 = 0.0;
      for (std::size_t i = 0; i < row_extent; ++i) {
        for (std::size_t j = 0; j < col_extent; ++j) {
          const double v = dense[(row0 + i) * matrix.cols() + (col0 + j)];
          tile.values[i * edge + j] = v;
          has_nonzero = has_nonzero || v != 0.0;
          norm2 += static_cast<long double>(v) * static_cast<long double>(v);
        }
      }
      if (has_nonzero) {
        tile.frobenius_norm = std::sqrt(static_cast<double>(norm2));
        tiles.push_back(std::move(tile));
      }
    }
  }
  return tiles;
}

}  // namespace detail

[[nodiscard]] inline Result<SpGemmFilteredResult> SpGemmFilteredFp64(
    const TileMat& a, const TileMat& b, double eps_filter) {
  if (a.cols() != b.rows()) {
    return Status::InvalidArgument("SpGEMM dimension mismatch");
  }
  if (a.tile_edge() != b.tile_edge()) {
    return Status::InvalidArgument(
        "CPU SpGEMM reference currently requires matching tile edges");
  }
  if (eps_filter < 0.0 || std::isnan(eps_filter)) {
    return Status::InvalidArgument("eps_filter must be non-negative");
  }

  const std::uint32_t edge = a.tile_edge();
  const std::vector<detail::ExpandedTile> a_tiles =
      detail::ExpandToFullTiles(a);
  const std::vector<detail::ExpandedTile> b_tiles =
      detail::ExpandToFullTiles(b);

  const std::size_t b_block_rows = detail::CeilDiv(b.rows(), edge);
  std::vector<std::vector<const detail::ExpandedTile*>> b_by_row(b_block_rows);
  for (const detail::ExpandedTile& b_tile : b_tiles) {
    b_by_row[b_tile.block_row].push_back(&b_tile);
  }

  std::vector<double> c_dense(a.rows() * b.cols(), 0.0);
  ErrorLedger ledger;
  ledger.eps_filter = eps_filter;
  std::map<std::pair<std::size_t, std::size_t>, double> dropped_by_output;

  for (const detail::ExpandedTile& a_tile : a_tiles) {
    if (a_tile.block_col >= b_by_row.size()) {
      return Status::CorruptData("A tile column exceeds B block rows");
    }
    for (const detail::ExpandedTile* b_tile : b_by_row[a_tile.block_col]) {
      ++ledger.candidate_products;
      const double product_bound =
          a_tile.frobenius_norm * b_tile->frobenius_norm;
      if (product_bound < eps_filter) {
        ++ledger.dropped_products;
        ledger.dropped_frobenius_bound += product_bound;
        dropped_by_output[{a_tile.block_row, b_tile->block_col}] +=
            product_bound;
        continue;
      }

      ++ledger.retained_products;
      const std::size_t k_extent =
          std::min(a_tile.col_extent, b_tile->row_extent);
      for (std::size_t i = 0; i < a_tile.row_extent; ++i) {
        for (std::size_t j = 0; j < b_tile->col_extent; ++j) {
          double sum = 0.0;
          for (std::size_t k = 0; k < k_extent; ++k) {
            sum += a_tile.values[i * edge + k] *
                   b_tile->values[k * edge + j];
          }
          c_dense[(a_tile.row_start + i) * b.cols() +
                  (b_tile->col_start + j)] += sum;
        }
      }
    }
  }

  ledger.output_tile_bounds.reserve(dropped_by_output.size());
  for (const auto& [tile, bound] : dropped_by_output) {
    ledger.output_tile_bounds.push_back(
        TileErrorBound{tile.first, tile.second, bound});
  }
  ledger.operation_ledger.Add(OperationLedgerEntry{
      OperationKind::kSpGemmFiltered,
      Fp64ReferencePrecision(),
      ErrorBudget{ErrorMetric::kFrobenius, ledger.dropped_frobenius_bound,
                  "sum ||A_ik||_F ||B_kj||_F for dropped tile products"},
      ledger.dropped_frobenius_bound,
      static_cast<std::uint64_t>(ledger.candidate_products),
      static_cast<std::uint64_t>(ledger.retained_products),
      static_cast<std::uint64_t>(ledger.dropped_products),
      "CPU FP64 filtered SpGEMM oracle"});

  auto product = TileMat::FromDense(a.rows(), b.cols(), c_dense, edge);
  if (!product.ok()) {
    return product.status();
  }
  return SpGemmFilteredResult{product.take_value(), std::move(ledger)};
}

}  // namespace tides::tile
