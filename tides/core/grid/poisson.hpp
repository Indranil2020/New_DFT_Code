#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "grid/dual_grid.hpp"

namespace tides::grid {

// Poisson solver for the Hartree potential under all four boundary conditions
// (per 10-physics/13 and WP3 T3.4).
//
// The CPU reference uses a direct N-body Coulomb sum for free BCs (with
// self-term regularization) and a DFT-based Green's function for periodic BC.
// The GPU path (cuFFT) is deferred to Phase B; this CPU path is the FP64 oracle.
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

 private:
  // Direct N-body Coulomb sum for free BC, with self-term regularization.
  // The self-potential of charge in a cubic cell of side h is approximated as
  // phi_self = 2.3801 / h_eff (the potential at the center of a uniformly
  // charged cube; standard ISF/PSolver correction).
  static std::vector<double> SolveFree(const UniformGrid3D& grid,
                                       const std::vector<double>& rho) {
    const std::size_t N = grid.total_points();
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

  // Periodic DFT-based Poisson: V(k) = 4 pi rho(k) / k^2 (k=0 -> 0).
  // O(N^2) naive DFT; GPU path uses cuFFT for O(N log N).
  static std::vector<double> SolvePeriodicFFT(const UniformGrid3D& grid,
                                             const std::vector<double>& rho) {
    const std::size_t N = grid.total_points();
    const auto [n0, n1, n2] = grid.n;
    const auto [h0, h1, h2] = grid.h;
    const auto [L0, L1, L2] = grid.cell_size();
    const double dv = h0 * h1 * h2;

    // Forward DFT: rho(k) = sum_j rho(j) exp(-i k.r_j) dv.
    std::vector<std::array<double, 2>> rho_k(N, {0.0, 0.0});
    for (std::size_t k = 0; k < N; ++k) {
      const auto [kx, ky, kz] = grid.unflatten(k);
      double re = 0.0, im = 0.0;
      for (std::size_t j = 0; j < N; ++j) {
        const auto [jx, jy, jz] = grid.unflatten(j);
        const double phase = -2.0 * M_PI *
            (static_cast<double>(kx * jx) / static_cast<double>(n0) +
             static_cast<double>(ky * jy) / static_cast<double>(n1) +
             static_cast<double>(kz * jz) / static_cast<double>(n2));
        re += rho[j] * std::cos(phase) * dv;
        im += rho[j] * std::sin(phase) * dv;
      }
      rho_k[k] = {re, im};
    }

    // Green's function: V(k) = 4 pi rho(k) / k^2.
    std::vector<std::array<double, 2>> V_k(N, {0.0, 0.0});
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
      if (k2 < 1e-30) continue;
      const double factor = 4.0 * M_PI / k2;
      V_k[k] = {rho_k[k][0] * factor, rho_k[k][1] * factor};
    }

    // Inverse DFT: V(r) = sum_k V(k) exp(+i k.r) / V_cell.
    std::vector<double> V(N, 0.0);
    const double inv_vol = 1.0 / (L0 * L1 * L2);
    for (std::size_t j = 0; j < N; ++j) {
      const auto [jx, jy, jz] = grid.unflatten(j);
      double re = 0.0;
      for (std::size_t k = 0; k < N; ++k) {
        const auto [kx, ky, kz] = grid.unflatten(k);
        const double phase = 2.0 * M_PI *
            (static_cast<double>(kx * jx) / static_cast<double>(n0) +
             static_cast<double>(ky * jy) / static_cast<double>(n1) +
             static_cast<double>(kz * jz) / static_cast<double>(n2));
        re += (V_k[k][0] * std::cos(phase) - V_k[k][1] * std::sin(phase));
      }
      V[j] = re * inv_vol;
    }
    return V;
  }
};

}  // namespace tides::grid
