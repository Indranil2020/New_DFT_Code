#pragma once

// Phase 3 (Inc 1): grid-blocked sparse-φ storage.
//
// Replaces the dense n_basis × n_grid orbital tables (nao_driver.hpp:1128-1184)
// with block-sparse storage: the grid is cut into BS³ cubic blocks, and each
// block stores φ only for the basis functions whose r_cut sphere intersects it.
// Every NAO is strictly zero beyond r_cut, so for fixed grid density the total
// stored φ is O(N) instead of the dense O(N²).  See
// tides-docs/PHASE3-grid-blocked-phi-design.md.
//
// This module is decoupled from NaoDriver: the caller supplies a φ evaluator
// (a lambda wrapping NaoDriver::EvalNaoBasisFn), so the block builder has no
// dependency on the basis/pseudopotential types and is unit-testable in
// isolation against any evaluator.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "grid/dual_grid.hpp"

namespace tides::grid {

// A cubic slab of grid points plus the basis functions active on it.
struct GridBlock {
  std::int32_t bx = 0, by = 0, bz = 0;         // block coordinates
  std::int32_t nx = 0, ny = 0, nz = 0;         // extent (clipped at grid edges)
  std::int32_t ix0 = 0, iy0 = 0, iz0 = 0;      // corner point (global indices)
  std::vector<std::int32_t> active;            // basis indices intersecting block
  // Compact φ, point-major within a function so a GEMM sees active fns as rows:
  //   phi[a * n_pts + p],  a in [0,active.size()), p in [0,n_pts)
  // Local point order: ix fastest, then iy, then iz (matches grid.flatten).
  std::vector<double> phi;

  [[nodiscard]] std::int64_t n_pts() const {
    return static_cast<std::int64_t>(nx) * ny * nz;
  }
};

struct BlockedPhi {
  std::vector<GridBlock> blocks;
  std::int32_t block_size = 0;
  std::size_t n_basis = 0;
  // Memory accounting (bytes) for the O(N) vs O(N²) report.
  std::int64_t dense_phi_bytes = 0;    // n_basis · n_grid · 8
  std::int64_t blocked_phi_bytes = 0;  // Σ active·n_pts · 8
  [[nodiscard]] double ratio() const {
    return blocked_phi_bytes > 0
               ? static_cast<double>(dense_phi_bytes) / blocked_phi_bytes
               : 0.0;
  }
};

// Squared distance from a point to an axis-aligned box [lo, hi] (0 inside).
inline double PointAabbDist2(const std::array<double, 3>& p,
                             const std::array<double, 3>& lo,
                             const std::array<double, 3>& hi) {
  double d2 = 0.0;
  for (int c = 0; c < 3; ++c) {
    const double v = p[c];
    const double excess = (v < lo[c]) ? (lo[c] - v) : (v > hi[c]) ? (v - hi[c]) : 0.0;
    d2 += excess * excess;
  }
  return d2;
}

// Build block-sparse φ.
//   centers[i], r_cut[i] : per basis function (i in [0, n_basis))
//   eval(i, x, y, z)     : φ_i at a Cartesian point (== NaoDriver::EvalNaoBasisFn)
// The active test is conservative (AABB–sphere): it never drops a nonzero
// function; it may keep a few functions that are ~0 in the block (harmless
// zero columns), which the reconstruction oracle tolerates.
template <typename Eval>
BlockedPhi BuildBlockedPhi(const UniformGrid3D& grid,
                           const std::vector<std::array<double, 3>>& centers,
                           const std::vector<double>& r_cut,
                           Eval&& eval,
                           std::int32_t block_size = 8) {
  BlockedPhi out;
  out.block_size = block_size;
  out.n_basis = centers.size();
  const std::size_t n0 = grid.n[0], n1 = grid.n[1], n2 = grid.n[2];
  const std::int64_t n_grid = static_cast<std::int64_t>(n0) * n1 * n2;
  out.dense_phi_bytes = static_cast<std::int64_t>(out.n_basis) * n_grid * 8;

  const std::int32_t BS = block_size;
  const std::int32_t nb0 = static_cast<std::int32_t>((n0 + BS - 1) / BS);
  const std::int32_t nb1 = static_cast<std::int32_t>((n1 + BS - 1) / BS);
  const std::int32_t nb2 = static_cast<std::int32_t>((n2 + BS - 1) / BS);

  out.blocks.resize(static_cast<std::size_t>(nb0) * nb1 * nb2);

  // Parallelizable over blocks (each writes only its own GridBlock).
  #pragma omp parallel for collapse(3) schedule(dynamic)
  for (std::int32_t bz = 0; bz < nb2; ++bz) {
    for (std::int32_t by = 0; by < nb1; ++by) {
      for (std::int32_t bx = 0; bx < nb0; ++bx) {
        const std::size_t blk_idx =
            (static_cast<std::size_t>(bz) * nb1 + by) * nb0 + bx;
        GridBlock& blk = out.blocks[blk_idx];
        blk.bx = bx; blk.by = by; blk.bz = bz;
        blk.ix0 = bx * BS; blk.iy0 = by * BS; blk.iz0 = bz * BS;
        blk.nx = static_cast<std::int32_t>(
            std::min<std::size_t>(BS, n0 - blk.ix0));
        blk.ny = static_cast<std::int32_t>(
            std::min<std::size_t>(BS, n1 - blk.iy0));
        blk.nz = static_cast<std::int32_t>(
            std::min<std::size_t>(BS, n2 - blk.iz0));

        // Block AABB in real coordinates.
        const std::array<double, 3> lo = {
            grid.origin[0] + grid.h[0] * blk.ix0,
            grid.origin[1] + grid.h[1] * blk.iy0,
            grid.origin[2] + grid.h[2] * blk.iz0};
        const std::array<double, 3> hi = {
            grid.origin[0] + grid.h[0] * (blk.ix0 + blk.nx - 1),
            grid.origin[1] + grid.h[1] * (blk.iy0 + blk.ny - 1),
            grid.origin[2] + grid.h[2] * (blk.iz0 + blk.nz - 1)};

        // Guard the active test by one cell diagonal so it is a guaranteed
        // over-approximation: any grid point with r <= r_cut is kept even at
        // the exact boundary (eval includes r == r_cut) and under floating
        // point. Extra kept functions are ~0 columns (harmless); a nonzero
        // point is never dropped.
        const double guard = std::sqrt(grid.h[0] * grid.h[0] +
                                       grid.h[1] * grid.h[1] +
                                       grid.h[2] * grid.h[2]);
        for (std::size_t i = 0; i < centers.size(); ++i) {
          const double reach = r_cut[i] + guard;
          if (PointAabbDist2(centers[i], lo, hi) <= reach * reach) {
            blk.active.push_back(static_cast<std::int32_t>(i));
          }
        }
        if (blk.active.empty()) continue;

        const std::int64_t npts = blk.n_pts();
        blk.phi.assign(blk.active.size() * static_cast<std::size_t>(npts), 0.0);
        for (std::size_t a = 0; a < blk.active.size(); ++a) {
          const std::int32_t bi = blk.active[a];
          double* prow = blk.phi.data() + a * npts;
          std::int64_t p = 0;
          for (std::int32_t lz = 0; lz < blk.nz; ++lz) {
            const double z = grid.origin[2] + grid.h[2] * (blk.iz0 + lz);
            for (std::int32_t ly = 0; ly < blk.ny; ++ly) {
              const double y = grid.origin[1] + grid.h[1] * (blk.iy0 + ly);
              for (std::int32_t lx = 0; lx < blk.nx; ++lx, ++p) {
                const double x = grid.origin[0] + grid.h[0] * (blk.ix0 + lx);
                prow[p] = eval(bi, x, y, z);
              }
            }
          }
        }
      }
    }
  }

  for (const auto& blk : out.blocks)
    out.blocked_phi_bytes +=
        static_cast<std::int64_t>(blk.phi.size()) * 8;
  return out;
}

// Oracle: reconstruct the dense n_basis × n_grid φ table (row-major
// [i*n_grid + g]) from the blocked storage, for A/B validation against the
// dense evaluation path.  Only for tests — defeats the memory savings.
inline std::vector<double> ReconstructDensePhi(const UniformGrid3D& grid,
                                               const BlockedPhi& bp) {
  const std::size_t n0 = grid.n[0], n1 = grid.n[1], n2 = grid.n[2];
  const std::int64_t n_grid = static_cast<std::int64_t>(n0) * n1 * n2;
  std::vector<double> dense(static_cast<std::size_t>(bp.n_basis) * n_grid, 0.0);
  for (const auto& blk : bp.blocks) {
    const std::int64_t npts = blk.n_pts();
    for (std::size_t a = 0; a < blk.active.size(); ++a) {
      const std::int64_t bi = blk.active[a];
      const double* prow = blk.phi.data() + a * npts;
      std::int64_t p = 0;
      for (std::int32_t lz = 0; lz < blk.nz; ++lz) {
        const std::size_t iz = blk.iz0 + lz;
        for (std::int32_t ly = 0; ly < blk.ny; ++ly) {
          const std::size_t iy = blk.iy0 + ly;
          for (std::int32_t lx = 0; lx < blk.nx; ++lx, ++p) {
            const std::size_t ix = blk.ix0 + lx;
            const std::size_t g = grid.flatten(ix, iy, iz);
            dense[bi * n_grid + g] = prow[p];
          }
        }
      }
    }
  }
  return dense;
}

}  // namespace tides::grid
