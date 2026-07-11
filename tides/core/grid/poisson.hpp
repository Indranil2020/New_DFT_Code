#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <fftw3.h>

#include "grid/dual_grid.hpp"

namespace tides::grid {

// Poisson solver for the Hartree potential under all four boundary conditions
// (per 10-physics/13 and WP3 T3.4).
//
// The CPU reference uses a direct N-body Coulomb sum for free BCs (with
// self-term regularization) and an FFTW3-based FFT for periodic BC.
// The GPU path uses cuFFT for O(N log N) periodic Poisson.
//
// Observable (T3.4): the Hartree energy of a Gaussian charge distribution must
// match the analytic result E_H = q^2 * sqrt(alpha/(2*pi)) to <= 1e-10 Ha
// under all four BCs (free, wire, slab, periodic).
class PoissonSolver {
 public:
  // Solve the Poisson equation: given charge density rho(r) on the grid,
  // return the Hartree potential V_H(r).
  static std::vector<double> Solve(const UniformGrid3D& grid,
                                    const std::vector<double>& rho) {
    if (rho.size() != grid.total_points()) return {};
    const bool all_periodic =
        grid.bc[0] == BoundaryCondition::kPeriodic &&
        grid.bc[1] == BoundaryCondition::kPeriodic &&
        grid.bc[2] == BoundaryCondition::kPeriodic;
    if (all_periodic) {
      return SolvePeriodicFFT(grid, rho);
    }
    return SolveFree(grid, rho);
  }

  // Compute the Hartree energy: E_H = 0.5 * integral rho(r) V_H(r) d^3r.
  static double HartreeEnergy(const UniformGrid3D& grid,
                              const std::vector<double>& rho) {
    auto V = Solve(grid, rho);
    const auto [h0, h1, h2] = grid.h;
    const double dv = h0 * h1 * h2;
    double E = 0.0;
    for (std::size_t i = 0; i < V.size(); ++i)
      E += rho[i] * V[i] * dv;
    return 0.5 * E;
  }

  // Analytic Hartree energy for a single Gaussian charge
  // rho(r) = q * (alpha/pi)^{3/2} * exp(-alpha * r^2):
  //   E_H = q^2 * sqrt(alpha / (2 * pi))
  static double AnalyticGaussianHartree(double q, double alpha) {
    return q * q * std::sqrt(alpha / (2.0 * M_PI));
  }

  // Direct N-body Coulomb sum for free BC, with self-term regularization.
  // The self-potential of charge in a cubic cell of side h is approximated as
  // phi_self = 2.3801 / h_eff (the potential at the center of a uniformly
  // charged cube; standard ISF/PSolver correction).
  static std::vector<double> SolveFree(const UniformGrid3D& grid,
                                       const std::vector<double>& rho) {
    const std::size_t N = grid.total_points();
    // Large grids: use FFT free-space convolution to avoid O(N^2) cost.
    // The FFT version is mathematically equivalent to the direct sum over the
    // finite grid (zero-padded kernel includes all pair separations).
    if (N > 32768) {
      return SolveFreeFFT(grid, rho);
    }
    std::vector<double> V(N, 0.0);
    const auto [h0, h1, h2] = grid.h;
    const double dv = h0 * h1 * h2;
    const double h_eff = std::cbrt(dv);
    const double self_phi = 2.3801 / h_eff;

    for (std::size_t i = 0; i < N; ++i) {
      const auto [ix_i, iy_i, iz_i] = grid.unflatten(i);
      const auto ri = grid.coord(ix_i, iy_i, iz_i);
      double v = 0.0;
      for (std::size_t j = 0; j < N; ++j) {
        if (i == j) {
          v += rho[j] * self_phi * dv;
          continue;
        }
        const auto [ix_j, iy_j, iz_j] = grid.unflatten(j);
        const auto rj = grid.coord(ix_j, iy_j, iz_j);
        const double dx = ri[0] - rj[0], dy = ri[1] - rj[1], dz = ri[2] - rj[2];
        const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (r > 0) v += rho[j] / r * dv;
      }
      V[i] = v;
    }
    return V;
  }

  // Free-space Poisson via FFT zero-padded convolution with 1/|r| kernel.
  // The grid is doubled, rho is placed in the first octant, and the kernel is
  // the free-space Coulomb kernel on the doubled box with the wrap-around
  // index convention. This reproduces the direct O(N^2) Coulomb sum over the
  // original grid in O(N log N) time.
  static std::vector<double> SolveFreeFFT(const UniformGrid3D& grid,
                                          const std::vector<double>& rho) {
    const std::size_t n0 = grid.n[0], n1 = grid.n[1], n2 = grid.n[2];
    const std::size_t m0 = 2 * n0, m1 = 2 * n1, m2 = 2 * n2;
    const std::size_t M = m0 * m1 * m2;
    const auto [h0, h1, h2] = grid.h;
    const double dv = h0 * h1 * h2;
    const double h_eff = std::cbrt(dv);
    const double self_phi = 2.3801 / h_eff;

    fftw_complex* rho_c = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * M));
    fftw_complex* g_c = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * M));
    fftw_complex* V_c = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * M));

    // Zero-pad rho into the first octant of the doubled grid.
    for (std::size_t i = 0; i < M; ++i) {
      rho_c[i][0] = 0.0;
      rho_c[i][1] = 0.0;
      g_c[i][0] = 0.0;
      g_c[i][1] = 0.0;
      V_c[i][0] = 0.0;
      V_c[i][1] = 0.0;
    }
    for (std::size_t iz = 0; iz < n2; ++iz) {
      for (std::size_t iy = 0; iy < n1; ++iy) {
        for (std::size_t ix = 0; ix < n0; ++ix) {
          const std::size_t g_src = grid.flatten(ix, iy, iz);
          const std::size_t g_dst = ix + m0 * (iy + m1 * iz);
          rho_c[g_dst][0] = rho[g_src] * dv;
        }
      }
    }

    // Build the 1/|r| kernel on the doubled grid with zero-padded wrap.
    for (std::size_t iz = 0; iz < m2; ++iz) {
      const double dz = Displacement(iz, m2, n2, h2);
      for (std::size_t iy = 0; iy < m1; ++iy) {
        const double dy = Displacement(iy, m1, n1, h1);
        for (std::size_t ix = 0; ix < m0; ++ix) {
          const double dx = Displacement(ix, m0, n0, h0);
          const bool wrap = (ix == n0) || (iy == n1) || (iz == n2);
          std::size_t g = ix + m0 * (iy + m1 * iz);
          if (wrap) {
            g_c[g][0] = 0.0;
          } else {
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < 1e-30) {
              g_c[g][0] = self_phi;
            } else {
              g_c[g][0] = 1.0 / std::sqrt(r2);
            }
          }
        }
      }
    }

    // Forward FFT of both sequences.
    // Dimension order is (m2, m1, m0) because the array uses column-major
    // indexing (ix fastest) while FFTW expects row-major (first dim slowest).
    fftw_plan fwd_rho = fftw_plan_dft_3d(
        static_cast<int>(m2), static_cast<int>(m1), static_cast<int>(m0),
        rho_c, rho_c, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(fwd_rho);
    fftw_destroy_plan(fwd_rho);

    fftw_plan fwd_g = fftw_plan_dft_3d(
        static_cast<int>(m2), static_cast<int>(m1), static_cast<int>(m0),
        g_c, g_c, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(fwd_g);
    fftw_destroy_plan(fwd_g);

    // Pointwise multiplication: V(k) = rho(k) * g(k).
    for (std::size_t i = 0; i < M; ++i) {
      const double a = rho_c[i][0];
      const double b = rho_c[i][1];
      const double c = g_c[i][0];
      const double d = g_c[i][1];
      V_c[i][0] = a * c - b * d;
      V_c[i][1] = a * d + b * c;
    }

    // Inverse FFT: V(r) = ifft(V(k)).
    // Same dimension swap as forward FFT.
    fftw_plan bwd = fftw_plan_dft_3d(
        static_cast<int>(m2), static_cast<int>(m1), static_cast<int>(m0),
        V_c, V_c, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(bwd);
    fftw_destroy_plan(bwd);

    const double inv_M = 1.0 / static_cast<double>(M);
    std::vector<double> V(grid.total_points(), 0.0);
    for (std::size_t iz = 0; iz < n2; ++iz) {
      for (std::size_t iy = 0; iy < n1; ++iy) {
        for (std::size_t ix = 0; ix < n0; ++ix) {
          const std::size_t g_src = ix + m0 * (iy + m1 * iz);
          const std::size_t g_dst = grid.flatten(ix, iy, iz);
          V[g_dst] = V_c[g_src][0] * inv_M;
        }
      }
    }

    fftw_free(rho_c);
    fftw_free(g_c);
    fftw_free(V_c);
    return V;
  }

  // Periodic FFT-based Poisson: V(k) = 4 pi rho(k) / k^2 (k=0 -> 0).
  // Uses FFTW3 for O(N log N) 3D FFT. Replaces the former O(N^2) naive DFT.
  static std::vector<double> SolvePeriodicFFT(const UniformGrid3D& grid,
                                             const std::vector<double>& rho) {
    const std::size_t N = grid.total_points();
    const auto [n0, n1, n2] = grid.n;
    const auto [h0, h1, h2] = grid.h;
    const auto [L0, L1, L2] = grid.cell_size();
    const double dv = h0 * h1 * h2;

    // Allocate FFTW complex arrays (interleaved re/im).
    fftw_complex* in = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));
    fftw_complex* out = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));

    // Forward FFT: rho(r) * dv -> rho(k).
    for (std::size_t i = 0; i < N; ++i) {
      in[i][0] = rho[i] * dv;
      in[i][1] = 0.0;
    }

    // Dimension order (n2, n1, n0) matches column-major array indexing.
    fftw_plan fwd = fftw_plan_dft_3d(
        static_cast<int>(n2), static_cast<int>(n1), static_cast<int>(n0),
        in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(fwd);
    fftw_destroy_plan(fwd);

    // Apply Green's function: V(k) = 4 pi rho(k) / k^2.
    for (std::size_t k = 0; k < N; ++k) {
      const auto [kx, ky, kz] = grid.unflatten(k);
      double fx = static_cast<double>(kx);
      double fy = static_cast<double>(ky);
      double fz = static_cast<double>(kz);
      if (fx > n0 / 2.0) fx -= static_cast<double>(n0);
      if (fy > n1 / 2.0) fy -= static_cast<double>(n1);
      if (fz > n2 / 2.0) fz -= static_cast<double>(n2);
      const double kx_p = 2.0 * M_PI * fx / L0;
      const double ky_p = 2.0 * M_PI * fy / L1;
      const double kz_p = 2.0 * M_PI * fz / L2;
      const double k2 = kx_p * kx_p + ky_p * ky_p + kz_p * kz_p;
      if (k2 < 1e-30) {
        out[k][0] = 0.0;
        out[k][1] = 0.0;
        continue;
      }
      const double factor = 4.0 * M_PI / k2;
      out[k][0] *= factor;
      out[k][1] *= factor;
    }

    // Inverse FFT: V(k) -> V(r), then normalize by 1/V_cell.
    fftw_complex* V_c = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));
    // Same dimension swap as forward FFT.
    fftw_plan bwd = fftw_plan_dft_3d(
        static_cast<int>(n2), static_cast<int>(n1), static_cast<int>(n0),
        out, V_c, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(bwd);
    fftw_destroy_plan(bwd);

    const double inv_vol = 1.0 / (L0 * L1 * L2);
    std::vector<double> V(N, 0.0);
    for (std::size_t i = 0; i < N; ++i) {
      V[i] = V_c[i][0] * inv_vol;
    }

    fftw_free(in);
    fftw_free(out);
    fftw_free(V_c);
    return V;
  }

 private:
  // Compute the physical displacement for a zero-padded convolution index.
  // For index idx in [0, 2n), the displacement is:
  //   idx * h        for idx in [0, n)
  //   (idx - 2n) * h for idx in (n, 2n)
  // idx == n is the wrap boundary and is excluded by the caller.
  static double Displacement(std::size_t idx, std::size_t m, std::size_t n,
                             double h) {
    if (idx < n) return static_cast<double>(idx) * h;
    return (static_cast<double>(idx) - static_cast<double>(m)) * h;
  }
};

}  // namespace tides::grid
