#pragma once

// T3.2: rho builder — density from orbital products (CPU reference).
// rho(r) = sum_{ij} P_{ij} phi_i(r) phi_j(r)
// For a set of occupied orbitals psi_k(r) = sum_i c_{ik} phi_i(r) with
// occupancy f_k, rho(r) = sum_k f_k |psi_k(r)|^2.
//
// For the CPU reference we evaluate orbitals (Gaussian-type or NAO) on the
// grid and form the density directly. The GPU tile-batched path (T3.2
// production) uses WP1 TileMat grouped GEMM; this CPU path is the FP64 oracle
// (observable 1: "vs CPU <= 1e-9").
//
// Observable (2): integral(rho) = N_e <= 1e-10. This is the key physics gate.

#include <cmath>
#include <cstddef>
#include <vector>

#include "grid/dual_grid.hpp"

namespace tides::grid {

class RhoBuilder {
 public:
  // Build density from occupied orbitals. Each orbital is a real-valued
  // function on the fine grid; the density is sum_k f_k * |orb_k(r)|^2.
  static std::vector<double> BuildFromOrbitals(
      const UniformGrid3D& grid,
      const std::vector<std::vector<double>>& orbitals,
      const std::vector<double>& occupations) {
    const std::size_t N = grid.total_points();
    std::vector<double> rho(N, 0.0);
    for (std::size_t k = 0; k < orbitals.size() && k < occupations.size(); ++k) {
      if (orbitals[k].size() != N) continue;
      const double f = occupations[k];
      for (std::size_t i = 0; i < N; ++i)
        rho[i] += f * orbitals[k][i] * orbitals[k][i];
    }
    return rho;
  }

  // Integrate the density: integral rho dV. Should equal N_e (sum of occupations).
  static double Integral(const UniformGrid3D& grid,
                         const std::vector<double>& rho) {
    const auto [h0, h1, h2] = grid.h;
    const double dv = h0 * h1 * h2;
    double s = 0.0;
    for (double v : rho) s += v;
    return s * dv;
  }

  // Build a normalized Gaussian orbital on the grid (for testing).
  // psi(r) = (2a/pi)^{3/4} * exp(-a * r^2)
  static std::vector<double> GaussianOrbital(const UniformGrid3D& grid,
                                             double alpha,
                                             const std::array<double, 3>& center) {
    const std::size_t N = grid.total_points();
    std::vector<double> psi(N, 0.0);
    const double norm = std::pow(2.0 * alpha / M_PI, 0.75);
    for (std::size_t i = 0; i < N; ++i) {
      const auto [ix, iy, iz] = grid.unflatten(i);
      const auto [x, y, z] = grid.coord(ix, iy, iz);
      const double dx = x - center[0], dy = y - center[1], dz = z - center[2];
      psi[i] = norm * std::exp(-alpha * (dx * dx + dy * dy + dz * dz));
    }
    return psi;
  }
};

  // Build density from a density matrix (R2/R3 path — no eigenvectors needed).
  // rho(r) = sum_{mu,nu} P_{mu,nu} * phi_mu(r) * phi_nu(r)
  static std::vector<double> BuildFromDensityMatrix(
      const UniformGrid3D& grid,
      const std::vector<double>& P,       // [n_basis][n_basis]
      const std::vector<std::vector<double>>& phi,  // [n_basis][n_points]
      std::size_t n_basis) {
    const std::size_t N = grid.total_points();
    std::vector<double> rho(N, 0.0);
    for (std::size_t mu = 0; mu < n_basis; ++mu) {
      for (std::size_t nu = 0; nu < n_basis; ++nu) {
        const double p = P[mu * n_basis + nu];
        if (std::abs(p) < 1e-30) continue;
        for (std::size_t g = 0; g < N; ++g)
          rho[g] += p * phi[mu][g] * phi[nu][g];
      }
    }
    return rho;
  }

}  // namespace tides::grid
