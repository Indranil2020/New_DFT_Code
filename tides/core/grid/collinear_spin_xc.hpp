#pragma once

// Collinear spin XC evaluation.
//
// In collinear spin DFT, the magnetization is along a fixed axis (z).
// The two spin channels are treated independently:
//   rho = rho_up + rho_down  (total density)
//   m   = rho_up - rho_down  (magnetization)
//   zeta = m / rho            (spin polarization)
//
// The XC functional depends on both spin densities. libxc uses nspin=2
// with interleaved arrays:
//   rho[2*i+0] = rho_up[i],  rho[2*i+1] = rho_down[i]
//   sigma[3*i+0] = |grad rho_up|^2, sigma[3*i+1] = grad rho_up . grad rho_down,
//   sigma[3*i+2] = |grad rho_down|^2
//
// Observable: spin-polarized XC energy matches unpolarized at zeta=0;
// exchange energy scales as (1+zeta)^(4/3) + (1-zeta)^(4/3) / 2 vs unpolarized.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <xc.h>
#include <xc_funcs.h>

#include "grid/dual_grid.hpp"
#include "grid/libxc_wrapper.hpp"

namespace tides::grid {

// Result of a spin-polarized XC evaluation.
struct SpinXCResult {
  std::vector<double> eps_xc;       // per-particle energy (size np)
  std::vector<double> vxc_up;       // V_xc for spin-up (size np)
  std::vector<double> vxc_down;     // V_xc for spin-down (size np)
  double xc_energy = 0.0;           // total XC energy
  bool ok = false;
};

// Collinear spin XC evaluator using libxc with nspin=2.
class CollinearSpinXC {
 public:
  // Evaluate spin-polarized LDA (Slater X + PW92 C).
  //   rho_up, rho_down: spin densities on the grid (each size np)
  //   grid: the uniform grid
  static SpinXCResult EvaluateLDA(
      const UniformGrid3D& grid,
      const std::vector<double>& rho_up,
      const std::vector<double>& rho_down) {
    const std::size_t np = grid.total_points();
    SpinXCResult res;
    if (rho_up.size() != np || rho_down.size() != np) return res;

    // Build interleaved rho array for libxc: [up, down, up, down, ...]
    std::vector<double> rho_pol(2 * np, 0.0);
    for (std::size_t i = 0; i < np; ++i) {
      rho_pol[2 * i] = std::max(0.0, rho_up[i]);
      rho_pol[2 * i + 1] = std::max(0.0, rho_down[i]);
    }

    // Exchange (LDA_X, nspin=2)
    LibxcFunctional fx, fc;
    if (!fx.Init(kLibxc_LDA_X, XC_POLARIZED)) return res;
    if (!fc.Init(kLibxc_LDA_C_PW, XC_POLARIZED)) return res;

    auto rx = fx.EvalLDA(rho_pol, np);
    auto rc = fc.EvalLDA(rho_pol, np);

    res.eps_xc.resize(np, 0.0);
    res.vxc_up.resize(np, 0.0);
    res.vxc_down.resize(np, 0.0);

    const auto [h0, h1, h2] = grid.h;
    const double dv = h0 * h1 * h2;
    double energy = 0.0;

    for (std::size_t i = 0; i < np; ++i) {
      res.eps_xc[i] = rx.eps_xc[i] + rc.eps_xc[i];
      res.vxc_up[i] = rx.vrho[2 * i] + rc.vrho[2 * i];
      res.vxc_down[i] = rx.vrho[2 * i + 1] + rc.vrho[2 * i + 1];
      const double rho_tot = rho_pol[2 * i] + rho_pol[2 * i + 1];
      energy += res.eps_xc[i] * rho_tot * dv;
    }

    res.xc_energy = energy;
    res.ok = true;
    return res;
  }

  // Evaluate spin-polarized PBE GGA.
  static SpinXCResult EvaluatePBE(
      const UniformGrid3D& grid,
      const std::vector<double>& rho_up,
      const std::vector<double>& rho_down) {
    const std::size_t np = grid.total_points();
    SpinXCResult res;
    if (rho_up.size() != np || rho_down.size() != np) return res;

    const auto [n0, n1, n2] = grid.n;
    const auto [h0, h1, h2] = grid.h;

    // Build interleaved rho array.
    std::vector<double> rho_pol(2 * np, 0.0);
    for (std::size_t i = 0; i < np; ++i) {
      rho_pol[2 * i] = std::max(0.0, rho_up[i]);
      rho_pol[2 * i + 1] = std::max(0.0, rho_down[i]);
    }

    // Compute spin-polarized sigma: [uu, ud, dd] per point.
    auto sigma_up = ComputeSigmaSpin(n0, n1, n2, h0, h1, h2, rho_up);
    auto sigma_down = ComputeSigmaSpin(n0, n1, n2, h0, h1, h2, rho_down);
    auto sigma_cross = ComputeSigmaCross(n0, n1, n2, h0, h1, h2, rho_up, rho_down);

    std::vector<double> sigma_pol(3 * np, 0.0);
    for (std::size_t i = 0; i < np; ++i) {
      sigma_pol[3 * i] = sigma_up[i];
      sigma_pol[3 * i + 1] = sigma_cross[i];
      sigma_pol[3 * i + 2] = sigma_down[i];
    }

    // PBE exchange + correlation with nspin=2.
    LibxcFunctional fx, fc;
    if (!fx.Init(kLibxc_GGA_X_PBE, XC_POLARIZED)) return res;
    if (!fc.Init(kLibxc_GGA_C_PBE, XC_POLARIZED)) return res;

    auto rx = fx.EvalGGA(rho_pol, sigma_pol, np);
    auto rc = fc.EvalGGA(rho_pol, sigma_pol, np);

    res.eps_xc.resize(np, 0.0);
    res.vxc_up.resize(np, 0.0);
    res.vxc_down.resize(np, 0.0);

    const double dv = h0 * h1 * h2;
    double energy = 0.0;

    for (std::size_t i = 0; i < np; ++i) {
      res.eps_xc[i] = rx.eps_xc[i] + rc.eps_xc[i];
      res.vxc_up[i] = rx.vrho[2 * i] + rc.vrho[2 * i];
      res.vxc_down[i] = rx.vrho[2 * i + 1] + rc.vrho[2 * i + 1];
      const double rho_tot = rho_pol[2 * i] + rho_pol[2 * i + 1];
      energy += res.eps_xc[i] * rho_tot * dv;
    }

    res.xc_energy = energy;
    res.ok = true;
    return res;
  }

  // Compute spin polarization zeta = (rho_up - rho_down) / (rho_up + rho_down).
  static std::vector<double> ComputeZeta(
      const std::vector<double>& rho_up,
      const std::vector<double>& rho_down) {
    std::vector<double> zeta(rho_up.size(), 0.0);
    for (std::size_t i = 0; i < rho_up.size(); ++i) {
      const double tot = rho_up[i] + rho_down[i];
      if (tot > 1e-14)
        zeta[i] = (rho_up[i] - rho_down[i]) / tot;
    }
    return zeta;
  }

  // Check: at zeta=0 (equal spin densities), spin-polarized energy should
  // equal unpolarized energy.
  static bool CheckUnpolarizedLimit(const UniformGrid3D& grid,
                                     const std::vector<double>& rho) {
    const std::size_t np = grid.total_points();
    std::vector<double> rho_half(np);
    for (std::size_t i = 0; i < np; ++i)
      rho_half[i] = 0.5 * std::max(0.0, rho[i]);

    auto spin_res = EvaluateLDA(grid, rho_half, rho_half);
    auto unpol_res = LibxcFunctional::EvalLDAOnGrid(rho);

    if (!spin_res.ok) return false;

    const auto [h0, h1, h2] = grid.h;
    const double dv = h0 * h1 * h2;
    double unpol_energy = 0.0;
    for (std::size_t i = 0; i < np; ++i)
      unpol_energy += unpol_res.eps_xc[i] * std::max(0.0, rho[i]) * dv;

    return std::fabs(spin_res.xc_energy - unpol_energy) < 1e-10;
  }

 private:
  // Compute |grad rho_spin|^2 for a single spin channel.
  static std::vector<double> ComputeSigmaSpin(
      std::size_t n0, std::size_t n1, std::size_t n2,
      double h0, double h1, double h2,
      const std::vector<double>& rho) {
    std::size_t np = n0 * n1 * n2;
    std::vector<double> sigma(np, 0.0);
    auto idx = [n0, n1](std::size_t x, std::size_t y, std::size_t z) {
      return x + n0 * (y + n1 * z);
    };
    for (std::size_t iz = 0; iz < n2; ++iz)
      for (std::size_t iy = 0; iy < n1; ++iy)
        for (std::size_t ix = 0; ix < n0; ++ix) {
          const std::size_t g = idx(ix, iy, iz);
          if (rho[g] < 1e-14) { sigma[g] = 0.0; continue; }
          double dnx = 0.0, dny = 0.0, dnz = 0.0;
          if (ix > 0 && ix + 1 < n0)
            dnx = (rho[idx(ix + 1, iy, iz)] - rho[idx(ix - 1, iy, iz)]) / (2.0 * h0);
          if (iy > 0 && iy + 1 < n1)
            dny = (rho[idx(ix, iy + 1, iz)] - rho[idx(ix, iy - 1, iz)]) / (2.0 * h1);
          if (iz > 0 && iz + 1 < n2)
            dnz = (rho[idx(ix, iy, iz + 1)] - rho[idx(ix, iy, iz - 1)]) / (2.0 * h2);
          sigma[g] = dnx * dnx + dny * dny + dnz * dnz;
        }
    return sigma;
  }

  // Compute grad rho_up . grad rho_down (cross term for spin-polarized GGA).
  static std::vector<double> ComputeSigmaCross(
      std::size_t n0, std::size_t n1, std::size_t n2,
      double h0, double h1, double h2,
      const std::vector<double>& rho_up,
      const std::vector<double>& rho_down) {
    std::size_t np = n0 * n1 * n2;
    std::vector<double> sigma(np, 0.0);
    auto idx = [n0, n1](std::size_t x, std::size_t y, std::size_t z) {
      return x + n0 * (y + n1 * z);
    };
    for (std::size_t iz = 0; iz < n2; ++iz)
      for (std::size_t iy = 0; iy < n1; ++iy)
        for (std::size_t ix = 0; ix < n0; ++ix) {
          const std::size_t g = idx(ix, iy, iz);
          if (rho_up[g] < 1e-14 || rho_down[g] < 1e-14) {
            sigma[g] = 0.0;
            continue;
          }
          double dnx_up = 0.0, dny_up = 0.0, dnz_up = 0.0;
          double dnx_dn = 0.0, dny_dn = 0.0, dnz_dn = 0.0;
          if (ix > 0 && ix + 1 < n0) {
            dnx_up = (rho_up[idx(ix + 1, iy, iz)] - rho_up[idx(ix - 1, iy, iz)]) / (2.0 * h0);
            dnx_dn = (rho_down[idx(ix + 1, iy, iz)] - rho_down[idx(ix - 1, iy, iz)]) / (2.0 * h0);
          }
          if (iy > 0 && iy + 1 < n1) {
            dny_up = (rho_up[idx(ix, iy + 1, iz)] - rho_up[idx(ix, iy - 1, iz)]) / (2.0 * h1);
            dny_dn = (rho_down[idx(ix, iy + 1, iz)] - rho_down[idx(ix, iy - 1, iz)]) / (2.0 * h1);
          }
          if (iz > 0 && iz + 1 < n2) {
            dnz_up = (rho_up[idx(ix, iy, iz + 1)] - rho_up[idx(ix, iy, iz - 1)]) / (2.0 * h2);
            dnz_dn = (rho_down[idx(ix, iy, iz + 1)] - rho_down[idx(ix, iy, iz - 1)]) / (2.0 * h2);
          }
          sigma[g] = dnx_up * dnx_dn + dny_up * dny_dn + dnz_up * dnz_dn;
        }
    return sigma;
  }
};

}  // namespace tides::grid
