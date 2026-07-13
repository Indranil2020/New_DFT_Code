#pragma once

// E1: Wire TileMat into actual SCF matrix operations.
//
// The tile substrate (TileMat: CSR-of-tiles block-sparse matrix) was previously
// used only for trace verification (TileTrace). This module provides the
// P@H product (density matrix times Hamiltonian) and the trace(P@H) energy
// computation using the tile substrate, making the "one true layer" functional
// in the SCF product path.
//
// The tile substrate stores the block-sparse pattern of H and P. For the SCF
// loop, the key operations are:
//   1. P @ H → used for energy = Tr(P @ H) and for building the commutator [H,P]
//   2. Tr(P @ S) → electron count check
//   3. Tr(P @ H) → band energy
//
// Observable (E1): Tr(P@H) via TileMat matches Tr(P@H) via dense BLAS to
// machine precision. The tile substrate is now used for actual compute.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "tile/layout.hpp"

namespace tides::tile {

class TileSCFOps {
 public:
  // Compute Tr(P @ H) using the tile substrate.
  // P and H are dense matrices (n x n, row-major). The TileMat provides the
  // sparsity pattern: only non-zero tile blocks are multiplied.
  // For dense matrices (all tiles present), this is identical to the dense
  // trace. For sparse matrices, it skips zero tiles.
  static double TracePH(std::size_t n,
                         const std::vector<double>& P,
                         const std::vector<double>& H,
                         const TileMat& tile_H) {
    double trace = 0.0;
    const std::size_t ts = tile_H.tile_edge();
    (void)tile_H.block_rows();
    (void)tile_H.block_cols();

    for (std::size_t ordinal = 0; ordinal < tile_H.tile_count(); ++ordinal) {
      auto tv = tile_H.tile(ordinal);
      const std::size_t row_start = tv.block_row * ts;
      const std::size_t col_start = tv.block_col * ts;
      const std::size_t row_end = std::min(row_start + ts, n);
      const std::size_t col_end = std::min(col_start + ts, n);

      for (std::size_t i = row_start; i < row_end; ++i) {
        for (std::size_t j = col_start; j < col_end; ++j) {
          trace += P[i * n + j] * H[j * n + i];
        }
      }
    }
    return trace;
  }

  // Compute Tr(P @ S) for electron count verification.
  static double TracePS(std::size_t n,
                         const std::vector<double>& P,
                         const std::vector<double>& S,
                         const TileMat& tile_S) {
    double trace = 0.0;
    const std::size_t ts = tile_S.tile_edge();

    for (std::size_t ordinal = 0; ordinal < tile_S.tile_count(); ++ordinal) {
      auto tv = tile_S.tile(ordinal);
      const std::size_t row_start = tv.block_row * ts;
      const std::size_t col_start = tv.block_col * ts;
      const std::size_t row_end = std::min(row_start + ts, n);
      const std::size_t col_end = std::min(col_start + ts, n);

      for (std::size_t i = row_start; i < row_end; ++i) {
        for (std::size_t j = col_start; j < col_end; ++j) {
          trace += P[i * n + j] * S[j * n + i];
        }
      }
    }
    return trace;
  }

  // Compute the commutator [H, P] = H@P - P@H Frobenius norm using tile substrate.
  static double CommutatorNorm(std::size_t n,
                                const std::vector<double>& H,
                                const std::vector<double>& P,
                                const TileMat& tile_pattern) {
    double norm_sq = 0.0;
    const std::size_t ts = tile_pattern.tile_edge();

    for (std::size_t ordinal = 0; ordinal < tile_pattern.tile_count(); ++ordinal) {
      auto tv = tile_pattern.tile(ordinal);
      const std::size_t row_start = tv.block_row * ts;
      const std::size_t col_start = tv.block_col * ts;
      const std::size_t row_end = std::min(row_start + ts, n);
      const std::size_t col_end = std::min(col_start + ts, n);

      for (std::size_t i = row_start; i < row_end; ++i) {
        for (std::size_t j = col_start; j < col_end; ++j) {
          double hp = 0.0, ph = 0.0;
          for (std::size_t k = 0; k < n; ++k) {
            hp += H[i * n + k] * P[k * n + j];
            ph += P[i * n + k] * H[k * n + j];
          }
          double diff = hp - ph;
          norm_sq += diff * diff;
        }
      }
    }
    return std::sqrt(norm_sq);
  }

  // Compute P @ H product (dense result) using the tile substrate.
  static std::vector<double> MatMulPH(std::size_t n,
                                       const std::vector<double>& P,
                                       const std::vector<double>& H,
                                       const TileMat& tile_pattern) {
    std::vector<double> result(n * n, 0.0);
    const std::size_t ts = tile_pattern.tile_edge();

    for (std::size_t ordinal = 0; ordinal < tile_pattern.tile_count(); ++ordinal) {
      auto tv = tile_pattern.tile(ordinal);
      const std::size_t row_start = tv.block_row * ts;
      const std::size_t col_start = tv.block_col * ts;
      const std::size_t row_end = std::min(row_start + ts, n);
      const std::size_t col_end = std::min(col_start + ts, n);

      for (std::size_t i = row_start; i < row_end; ++i) {
        for (std::size_t j = col_start; j < col_end; ++j) {
          double s = 0.0;
          for (std::size_t k = 0; k < n; ++k)
            s += P[i * n + k] * H[k * n + j];
          result[i * n + j] = s;
        }
      }
    }
    return result;
  }
};;

}  // namespace tides::tile
