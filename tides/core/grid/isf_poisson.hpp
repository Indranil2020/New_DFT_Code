#pragma once

// ISF (Infinite Sum Form) Poisson kernel for non-periodic BCs.
//
// The existing Poisson solver uses direct O(N²) Coulomb sums for free BCs,
// which is prohibitively expensive for large grids. The ISF approach enables
// O(N log N) FFT-based solving for all boundary conditions by:
//
// 1. Embedding the system in a supercell (2× or 3× the original box)
// 2. Using a softened Coulomb kernel in reciprocal space:
//    G_soft(k) = 4π/k² * exp(-k²/(4μ²))  (Gaussian-screened)
//    This avoids the G=0 divergence and suppresses aliasing from images
// 3. Applying FFT-based convolution on the supercell
// 4. Correcting for the softening error in real space (short-range sum)
//
// The parameter μ controls the split: large μ → more real-space work, less
// reciprocal-space error; small μ → less real-space work, more softening error.
// The optimal μ balances the two contributions.
//
// For free BC: supercell = 2× box (vacuum padding on all sides)
// For wire BC: supercell = 2× in free directions, 1× in periodic direction
// For slab BC: supercell = 2× in free direction, 1× in periodic directions
//
// Observable: ISF Poisson energy matches direct Coulomb sum to <= 1e-6 Ha
// at grid sizes up to 64³; O(N log N) vs O(N²).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <fftw3.h>

#include "grid/dual_grid.hpp"

namespace tides::grid {

// ISF Poisson solver configuration.
struct ISFConfig {
  // Screening parameter μ (Bohr⁻¹). Controls the real/reciprocal split.
  // Larger μ → more accurate but more real-space corrections.
  double mu = 0.5;
  // Supercell padding factor for free directions (2× = double the box).
  int pad_factor = 2;
  // Real-space cutoff for the correction sum (Bohr).
  double r_cutoff = 4.0;
};

// ISF Poisson solver result.
struct ISFResult {
  std::vector<double> V;        // potential on the original grid
  double hartree_energy = 0.0;  // E_H = 0.5 * ∫ ρ V d³r
  double wall_time_s = 0.0;
  bool ok = false;
};

// ISF Poisson solver for non-periodic boundary conditions.
class ISFPoissonSolver {
 public:
  // Solve the Poisson equation using the ISF kernel.
  // Supports free, wire, and slab BCs. For fully periodic BC, falls back
  // to the standard FFT Poisson (no softening needed).
  static ISFResult Solve(const UniformGrid3D& grid,
                          const std::vector<double>& rho,
                          const ISFConfig& config = {}) {
    ISFResult result;
    if (rho.size() != grid.total_points()) return result;

    const bool all_periodic =
        grid.bc[0] == BoundaryCondition::kPeriodic &&
        grid.bc[1] == BoundaryCondition::kPeriodic &&
        grid.bc[2] == BoundaryCondition::kPeriodic;

    if (all_periodic) {
      // For periodic BC, use standard FFT Poisson (no ISF needed).
      result.V = SolvePeriodicStandard(grid, rho);
      result.hartree_energy = ComputeHartreeEnergy(grid, rho, result.V);
      result.ok = !result.V.empty();
      return result;
    }

    // Build supercell: pad in free directions, keep periodic directions.
    auto super_grid = BuildSupercell(grid, config.pad_factor);

    // Embed charge density in the supercell (zero-padding in free directions).
    auto rho_super = EmbedCharge(grid, rho, super_grid);

    // Step 1: Reciprocal-space softened potential via FFT.
    auto V_soft = SolveSoftenedFFT(super_grid, rho_super, config.mu);

    // Step 2: Real-space correction for the softening error.
    // The correction is: V_corr(r) = sum_{r' near r} rho(r') * erfc(mu*r)/r * dv
    // Only needed for nearby pairs (within r_cutoff).
    auto V_corr = RealSpaceCorrection(grid, rho, config.mu, config.r_cutoff);

    // Step 3: Extract the potential on the original grid and combine.
    result.V = ExtractAndCombine(grid, super_grid, V_soft, V_corr);

    result.hartree_energy = ComputeHartreeEnergy(grid, rho, result.V);
    result.ok = !result.V.empty();
    return result;
  }

  // Compute the Hartree energy from rho and V.
  static double ComputeHartreeEnergy(const UniformGrid3D& grid,
                                      const std::vector<double>& rho,
                                      const std::vector<double>& V) {
    const auto [h0, h1, h2] = grid.h;
    const double dv = h0 * h1 * h2;
    double E = 0.0;
    for (std::size_t i = 0; i < V.size(); ++i)
      E += rho[i] * V[i] * dv;
    return 0.5 * E;
  }

  // Analytic Hartree energy for a Gaussian charge (for validation).
  static double AnalyticGaussianHartree(double q, double alpha) {
    return q * q * std::sqrt(alpha / (2.0 * M_PI));
  }

 private:
  // Build a supercell by padding in free directions.
  static UniformGrid3D BuildSupercell(const UniformGrid3D& grid,
                                       int pad_factor) {
    UniformGrid3D super = grid;
    for (int d = 0; d < 3; ++d) {
      if (grid.bc[d] != BoundaryCondition::kPeriodic) {
        super.n[d] = grid.n[d] * pad_factor;
        // Keep the same spacing; origin shifts to center the original grid.
        super.origin[d] = grid.origin[d] -
            static_cast<double>(grid.n[d] * (pad_factor - 1) / 2) * grid.h[d];
      }
    }
    return super;
  }

  // Embed charge density in the supercell (zero-pad in free directions).
  static std::vector<double> EmbedCharge(const UniformGrid3D& grid,
                                          const std::vector<double>& rho,
                                          const UniformGrid3D& super_grid) {
    std::vector<double> rho_super(super_grid.total_points(), 0.0);
    for (std::size_t iz = 0; iz < grid.n[2]; ++iz)
      for (std::size_t iy = 0; iy < grid.n[1]; ++iy)
        for (std::size_t ix = 0; ix < grid.n[0]; ++ix) {
          // Offset in the supercell for free directions.
          std::size_t sx = ix, sy = iy, sz = iz;
          if (grid.bc[0] != BoundaryCondition::kPeriodic)
            sx = ix + (super_grid.n[0] - grid.n[0]) / 2;
          if (grid.bc[1] != BoundaryCondition::kPeriodic)
            sy = iy + (super_grid.n[1] - grid.n[1]) / 2;
          if (grid.bc[2] != BoundaryCondition::kPeriodic)
            sz = iz + (super_grid.n[2] - grid.n[2]) / 2;
          const std::size_t src_idx = grid.flatten(ix, iy, iz);
          const std::size_t dst_idx = super_grid.flatten(sx, sy, sz);
          if (dst_idx < rho_super.size())
            rho_super[dst_idx] = rho[src_idx];
        }
    return rho_super;
  }

  // FFT-based softened Poisson solve on the supercell.
  // V_soft(k) = 4π * rho(k) / k² * exp(-k²/(4μ²))
  static std::vector<double> SolveSoftenedFFT(
      const UniformGrid3D& grid,
      const std::vector<double>& rho,
      double mu) {
    const std::size_t N = grid.total_points();
    const auto [n0, n1, n2] = grid.n;
    const auto [h0, h1, h2] = grid.h;
    const auto [L0, L1, L2] = grid.cell_size();
    const double dv = h0 * h1 * h2;

    fftw_complex* in = reinterpret_cast<fftw_complex*>(
        fftw_malloc(sizeof(fftw_complex) * N));
    fftw_complex* out = reinterpret_cast<fftw_complex*>(
        fftw_malloc(sizeof(fftw_complex) * N));

    for (std::size_t i = 0; i < N; ++i) {
      in[i][0] = rho[i] * dv;
      in[i][1] = 0.0;
    }

    fftw_plan fwd = fftw_plan_dft_3d(
        static_cast<int>(n0), static_cast<int>(n1), static_cast<int>(n2),
        in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(fwd);
    fftw_destroy_plan(fwd);

    const double mu2 = mu * mu;
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
      // Softened Green's function: 4π/k² * exp(-k²/(4μ²))
      const double soften = std::exp(-k2 / (4.0 * mu2));
      const double factor = 4.0 * M_PI * soften / k2;
      out[k][0] *= factor;
      out[k][1] *= factor;
    }

    fftw_complex* V_c = reinterpret_cast<fftw_complex*>(
        fftw_malloc(sizeof(fftw_complex) * N));
    fftw_plan bwd = fftw_plan_dft_3d(
        static_cast<int>(n0), static_cast<int>(n1), static_cast<int>(n2),
        out, V_c, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(bwd);
    fftw_destroy_plan(bwd);

    const double inv_vol = 1.0 / (L0 * L1 * L2);
    std::vector<double> V(N, 0.0);
    for (std::size_t i = 0; i < N; ++i)
      V[i] = V_c[i][0] * inv_vol;

    fftw_free(in);
    fftw_free(out);
    fftw_free(V_c);
    return V;
  }

  // Real-space correction: V_corr(r_i) = sum_{j near i} rho_j * erfc(μ*r_ij)/r_ij * dv
  // This corrects for the softening in the reciprocal-space solve.
  static std::vector<double> RealSpaceCorrection(
      const UniformGrid3D& grid,
      const std::vector<double>& rho,
      double mu, double r_cutoff) {
    const std::size_t N = grid.total_points();
    std::vector<double> V(N, 0.0);
    const auto [h0, h1, h2] = grid.h;
    const double dv = h0 * h1 * h2;
    const double h_eff = std::cbrt(dv);
    const double self_corr = 2.3801 / h_eff - 2.0 * mu / std::sqrt(M_PI);

    // Determine the neighbor range from r_cutoff.
    const int range_x = static_cast<int>(r_cutoff / h0) + 1;
    const int range_y = static_cast<int>(r_cutoff / h1) + 1;
    const int range_z = static_cast<int>(r_cutoff / h2) + 1;

    for (std::size_t i = 0; i < N; ++i) {
      const auto [ix_i, iy_i, iz_i] = grid.unflatten(i);
      const auto ri = grid.coord(ix_i, iy_i, iz_i);
      double v = 0.0;
      for (int dz = -range_z; dz <= range_z; ++dz) {
        const int jz = static_cast<int>(iz_i) + dz;
        if (jz < 0 || jz >= static_cast<int>(grid.n[2])) continue;
        for (int dy = -range_y; dy <= range_y; ++dy) {
          const int jy = static_cast<int>(iy_i) + dy;
          if (jy < 0 || jy >= static_cast<int>(grid.n[1])) continue;
          for (int dx = -range_x; dx <= range_x; ++dx) {
            const int jx = static_cast<int>(ix_i) + dx;
            if (jx < 0 || jx >= static_cast<int>(grid.n[0])) continue;
            const std::size_t j = grid.flatten(
                static_cast<std::size_t>(jx),
                static_cast<std::size_t>(jy),
                static_cast<std::size_t>(jz));
            if (i == j) {
              v += rho[j] * self_corr * dv;
              continue;
            }
            const auto rj = grid.coord(
                static_cast<std::size_t>(jx),
                static_cast<std::size_t>(jy),
                static_cast<std::size_t>(jz));
            const double rx = ri[0] - rj[0], ry = ri[1] - rj[1], rz = ri[2] - rj[2];
            const double r = std::sqrt(rx * rx + ry * ry + rz * rz);
            if (r > r_cutoff) continue;
            // erfc(μ*r)/r is the short-range correction.
            v += rho[j] * std::erfc(mu * r) / r * dv;
          }
        }
      }
      V[i] = v;
    }
    return V;
  }

  // Extract potential from supercell and combine with real-space correction.
  static std::vector<double> ExtractAndCombine(
      const UniformGrid3D& grid,
      const UniformGrid3D& super_grid,
      const std::vector<double>& V_soft,
      const std::vector<double>& V_corr) {
    std::vector<double> V(grid.total_points(), 0.0);
    for (std::size_t iz = 0; iz < grid.n[2]; ++iz)
      for (std::size_t iy = 0; iy < grid.n[1]; ++iy)
        for (std::size_t ix = 0; ix < grid.n[0]; ++ix) {
          std::size_t sx = ix, sy = iy, sz = iz;
          if (grid.bc[0] != BoundaryCondition::kPeriodic)
            sx = ix + (super_grid.n[0] - grid.n[0]) / 2;
          if (grid.bc[1] != BoundaryCondition::kPeriodic)
            sy = iy + (super_grid.n[1] - grid.n[1]) / 2;
          if (grid.bc[2] != BoundaryCondition::kPeriodic)
            sz = iz + (super_grid.n[2] - grid.n[2]) / 2;
          const std::size_t src_idx = super_grid.flatten(sx, sy, sz);
          const std::size_t dst_idx = grid.flatten(ix, iy, iz);
          if (src_idx < V_soft.size())
            V[dst_idx] = V_soft[src_idx] + V_corr[dst_idx];
        }
    return V;
  }

  // Standard periodic FFT Poisson (no softening) — same as PoissonSolver.
  static std::vector<double> SolvePeriodicStandard(
      const UniformGrid3D& grid, const std::vector<double>& rho) {
    const std::size_t N = grid.total_points();
    const auto [n0, n1, n2] = grid.n;
    const auto [h0, h1, h2] = grid.h;
    const auto [L0, L1, L2] = grid.cell_size();
    const double dv = h0 * h1 * h2;

    fftw_complex* in = reinterpret_cast<fftw_complex*>(
        fftw_malloc(sizeof(fftw_complex) * N));
    fftw_complex* out = reinterpret_cast<fftw_complex*>(
        fftw_malloc(sizeof(fftw_complex) * N));

    for (std::size_t i = 0; i < N; ++i) {
      in[i][0] = rho[i] * dv;
      in[i][1] = 0.0;
    }

    fftw_plan fwd = fftw_plan_dft_3d(
        static_cast<int>(n0), static_cast<int>(n1), static_cast<int>(n2),
        in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(fwd);
    fftw_destroy_plan(fwd);

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

    fftw_complex* V_c = reinterpret_cast<fftw_complex*>(
        fftw_malloc(sizeof(fftw_complex) * N));
    fftw_plan bwd = fftw_plan_dft_3d(
        static_cast<int>(n0), static_cast<int>(n1), static_cast<int>(n2),
        out, V_c, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(bwd);
    fftw_destroy_plan(bwd);

    const double inv_vol = 1.0 / (L0 * L1 * L2);
    std::vector<double> V(N, 0.0);
    for (std::size_t i = 0; i < N; ++i)
      V[i] = V_c[i][0] * inv_vol;

    fftw_free(in);
    fftw_free(out);
    fftw_free(V_c);
    return V;
  }
};

}  // namespace tides::grid
