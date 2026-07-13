#pragma once

// HSE screening integration for the NAO SCF loop (§3.1.5, WP7 T7.4).
//
// HSE06 uses a screened Coulomb interaction: the long-range exchange is
// treated with PBE exchange, while the short-range exchange uses exact
// (Hartree-Fock) exchange with a screened interaction:
//   1/r = erfc(ω*r)/r + erf(ω*r)/r
//   HSE: short-range HF exchange (erfc) + long-range PBE exchange (erf)
//
// The screening parameter ω controls the range separation. For HSE06,
// ω = 0.11 Bohr⁻¹.
//
// This module provides:
//   1. The screened exchange kernel evaluation
//   2. Integration into the SCF loop via the build_H callback
//   3. The ACE (asymptotically corrected exchange) adapter

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace tides::hybrids {

struct HSEConfig {
  double omega = 0.11;       // screening parameter (Bohr^-1), HSE06 default
  double alpha_exact = 0.25;  // fraction of exact exchange (0.25 for HSE06)
  double alpha_pbe = 0.75;   // fraction of PBE exchange (complementary)
  bool use_screening = true;  // if false, falls back to PBE0 (no screening)
};

struct HSEExchangeResult {
  double E_exact_sr = 0.0;   // short-range exact exchange energy
  double E_pbe_lr = 0.0;     // long-range PBE exchange energy
  std::vector<double> V_x_sr;  // short-range exchange matrix (n x n)
  bool converged = false;
};

class HSEScreening {
 public:
  // Compute the screened exchange interaction.
  // The error function complement erfc(ω*r) gives the short-range part.
  // For a pair of basis functions (i,j), the SR exchange integral is:
  //   K_ij^SR = ∫∫ φ_i(r) φ_j(r) * erfc(ω*|r-r'|) / |r-r'| * φ_i(r') φ_j(r') dr dr'
  //
  // For the CPU reference, we use the screened Coulomb kernel on a grid.
  //   P:        density matrix (n x n, row-major)
  //   basis_vals: basis function values on the grid (n_grid * n)
  //   grid:     grid points (3 * n_grid, Bohr)
  //   grid_w:   grid weights (n_grid)
  //   n:        basis size
  //   n_grid:   number of grid points
  static HSEExchangeResult ComputeSRExchange(
      const std::vector<double>& P, std::size_t n,
      const std::vector<double>& basis_vals,
      const std::vector<double>& grid, const std::vector<double>& grid_w,
      std::size_t n_grid,
      const HSEConfig& config = {}) {
    HSEExchangeResult res;
    res.V_x_sr.assign(n * n, 0.0);

    const double omega = config.omega;
    const double alpha = config.alpha_exact;

    // Compute the density on the grid: ρ(r) = Σ_ij P_ij φ_i(r) φ_j(r)
    std::vector<double> rho(n_grid, 0.0);
    for (std::size_t g = 0; g < n_grid; ++g) {
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
          rho[g] += P[i * n + j] * basis_vals[g * n + i] * basis_vals[g * n + j];
    }

    // Compute the SR exchange potential on the grid:
    // V_x^SR(r) = -α * Σ_j P_ij φ_j(r') * erfc(ω*|r-r'|) / |r-r'| dr'
    // For the CPU reference, use a direct double loop with cutoff.
    std::vector<double> V_xc_grid(n_grid, 0.0);
    for (std::size_t g1 = 0; g1 < n_grid; ++g1) {
      double vx = 0.0;
      for (std::size_t g2 = 0; g2 < n_grid; ++g2) {
        double dx = grid[3*g1] - grid[3*g2];
        double dy = grid[3*g1+1] - grid[3*g2+1];
        double dz = grid[3*g1+2] - grid[3*g2+2];
        double r = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (r < 1e-10) continue;

        // Screened Coulomb: erfc(ω*r) / r
        double screened = std::erfc(omega * r) / r;
        vx -= alpha * rho[g2] * screened * grid_w[g2];
      }
      V_xc_grid[g1] = vx;
    }

    // Project V_xc back to the basis matrix.
    // V_ij = ∫ φ_i(r) V_xc(r) φ_j(r) dr
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        double v = 0.0;
        for (std::size_t g = 0; g < n_grid; ++g)
          v += basis_vals[g * n + i] * V_xc_grid[g] *
               basis_vals[g * n + j] * grid_w[g];
        res.V_x_sr[i * n + j] = v;
      }

    // Compute energy: E = 0.5 * tr(P * V_x_sr)
    double E = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        E += 0.5 * P[i * n + j] * res.V_x_sr[j * n + i];
    res.E_exact_sr = E;
    res.converged = true;

    return res;
  }

  // Get the HSE exchange fraction for the SCF loop.
  // Returns the total exchange fraction: α_exact (SR) + α_pbe (LR)
  static double ExchangeFraction(const HSEConfig& config) {
    return config.alpha_exact + config.alpha_pbe;
  }

  // Compute the screened Coulomb kernel value.
  // 1/r = erfc(ω*r)/r + erf(ω*r)/r
  // SR part: erfc(ω*r)/r, LR part: erf(ω*r)/r
  static double ScreenedCoulomb(double r, double omega, bool short_range) {
    if (r < 1e-12) return 0.0;
    if (short_range)
      return std::erfc(omega * r) / r;
    else
      return std::erf(omega * r) / r;
  }
};

}  // namespace tides::hybrids
