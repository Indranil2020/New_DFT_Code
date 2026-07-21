#pragma once

// Phase 3 (Inc 2): the three grid contractions on block-sparse φ.
// Each is per-block work on the block's active functions + a scatter into the
// small n_basis × n_basis matrix (overlap/vmat) or a write into the grid (ρ):
//   overlap  S_ij  = dv · Σ_g φ_i(g)φ_j(g)
//   rho      ρ(g)  = Σ_ij P_ij φ_i(g)φ_j(g)
//   vmat     H_ij  = dv · Σ_g v(g) φ_i(g)φ_j(g)
// Only pairs/points where the functions are actually nonzero are touched, so
// the cost is O(N) not O(N²).  See PHASE3-grid-blocked-phi-design.md.
//
// Inc 2 uses explicit per-block loops (provably correct); the GPU port (Inc 4)
// replaces the inner block contraction with a grouped GEMM on tensor cores.
// These reproduce the dense production ops (VmatBuilder::BuildRho/BuildHmat and
// the grid-S path) bit-for-bit up to summation order.

#include <cstddef>
#include <vector>

#include "grid/dual_grid.hpp"
#include "grid/grid_blocking.hpp"

namespace tides::grid {

// S_ij = dv · Σ_g φ_i(g) φ_j(g), returned dense row-major n_basis × n_basis.
inline std::vector<double> BlockedOverlap(const UniformGrid3D& grid,
                                          const BlockedPhi& bp) {
  const std::size_t n = bp.n_basis;
  const double dv = grid.h[0] * grid.h[1] * grid.h[2];
  std::vector<double> S(n * n, 0.0);
  // Per-block local S accumulated, then scattered; blocks are independent so
  // this parallelizes with a per-thread partial matrix reduced at the end.
  #pragma omp parallel
  {
    std::vector<double> S_local(n * n, 0.0);
    #pragma omp for schedule(dynamic) nowait
    for (std::size_t b = 0; b < bp.blocks.size(); ++b) {
      const GridBlock& blk = bp.blocks[b];
      const std::int64_t npts = blk.n_pts();
      const std::size_t na = blk.active.size();
      for (std::size_t a = 0; a < na; ++a) {
        const double* pa = blk.phi.data() + a * npts;
        const std::int32_t ia = blk.active[a];
        for (std::size_t c = a; c < na; ++c) {
          const double* pc = blk.phi.data() + c * npts;
          double s = 0.0;
          for (std::int64_t p = 0; p < npts; ++p) s += pa[p] * pc[p];
          s *= dv;
          const std::int32_t ic = blk.active[c];
          S_local[ia * n + ic] += s;
          if (ic != ia) S_local[ic * n + ia] += s;
        }
      }
    }
    #pragma omp critical
    for (std::size_t k = 0; k < n * n; ++k) S[k] += S_local[k];
  }
  return S;
}

// ρ(g) = Σ_ij P_ij φ_i(g) φ_j(g), P dense row-major n_basis × n_basis.
// Written on the full grid (points outside every block's support stay 0).
inline std::vector<double> BlockedRho(const UniformGrid3D& grid,
                                      const BlockedPhi& bp,
                                      const std::vector<double>& P) {
  const std::size_t n = bp.n_basis;
  const std::int64_t n_grid =
      static_cast<std::int64_t>(grid.n[0]) * grid.n[1] * grid.n[2];
  std::vector<double> rho(static_cast<std::size_t>(n_grid), 0.0);
  #pragma omp parallel for schedule(dynamic)
  for (std::size_t b = 0; b < bp.blocks.size(); ++b) {
    const GridBlock& blk = bp.blocks[b];
    const std::int64_t npts = blk.n_pts();
    const std::size_t na = blk.active.size();
    if (na == 0) continue;
    // temp[a,p] = Σ_c P[active[a],active[c]] φ[c,p]
    std::vector<double> temp(na * static_cast<std::size_t>(npts), 0.0);
    for (std::size_t a = 0; a < na; ++a) {
      const std::int32_t ia = blk.active[a];
      double* trow = temp.data() + a * npts;
      for (std::size_t c = 0; c < na; ++c) {
        const double Pac = P[ia * n + blk.active[c]];
        if (Pac == 0.0) continue;
        const double* pc = blk.phi.data() + c * npts;
        for (std::int64_t p = 0; p < npts; ++p) trow[p] += Pac * pc[p];
      }
    }
    // ρ_blk(p) = Σ_a φ[a,p]·temp[a,p], written at the block's global points.
    std::int64_t p = 0;
    for (std::int32_t lz = 0; lz < blk.nz; ++lz) {
      const std::size_t iz = blk.iz0 + lz;
      for (std::int32_t ly = 0; ly < blk.ny; ++ly) {
        const std::size_t iy = blk.iy0 + ly;
        for (std::int32_t lx = 0; lx < blk.nx; ++lx, ++p) {
          const std::size_t ix = blk.ix0 + lx;
          double r = 0.0;
          for (std::size_t a = 0; a < na; ++a)
            r += blk.phi[a * npts + p] * temp[a * npts + p];
          rho[grid.flatten(ix, iy, iz)] = r;
        }
      }
    }
  }
  return rho;
}

// H_ij = dv · Σ_g v(g) φ_i(g) φ_j(g), v on the full grid; returns dense n×n.
inline std::vector<double> BlockedVmat(const UniformGrid3D& grid,
                                       const BlockedPhi& bp,
                                       const std::vector<double>& v) {
  const std::size_t n = bp.n_basis;
  const double dv = grid.h[0] * grid.h[1] * grid.h[2];
  std::vector<double> H(n * n, 0.0);
  #pragma omp parallel
  {
    std::vector<double> H_local(n * n, 0.0);
    #pragma omp for schedule(dynamic) nowait
    for (std::size_t b = 0; b < bp.blocks.size(); ++b) {
      const GridBlock& blk = bp.blocks[b];
      const std::int64_t npts = blk.n_pts();
      const std::size_t na = blk.active.size();
      if (na == 0) continue;
      // Gather v at the block's points in local order.
      std::vector<double> vblk(static_cast<std::size_t>(npts));
      std::int64_t p = 0;
      for (std::int32_t lz = 0; lz < blk.nz; ++lz) {
        const std::size_t iz = blk.iz0 + lz;
        for (std::int32_t ly = 0; ly < blk.ny; ++ly) {
          const std::size_t iy = blk.iy0 + ly;
          for (std::int32_t lx = 0; lx < blk.nx; ++lx, ++p)
            vblk[p] = v[grid.flatten(blk.ix0 + lx, iy, iz)];
        }
      }
      for (std::size_t a = 0; a < na; ++a) {
        const double* pa = blk.phi.data() + a * npts;
        const std::int32_t ia = blk.active[a];
        for (std::size_t c = a; c < na; ++c) {
          const double* pc = blk.phi.data() + c * npts;
          double s = 0.0;
          for (std::int64_t q = 0; q < npts; ++q) s += vblk[q] * pa[q] * pc[q];
          s *= dv;
          const std::int32_t ic = blk.active[c];
          H_local[ia * n + ic] += s;
          if (ic != ia) H_local[ic * n + ia] += s;
        }
      }
    }
    #pragma omp critical
    for (std::size_t k = 0; k < n * n; ++k) H[k] += H_local[k];
  }
  return H;
}

}  // namespace tides::grid
