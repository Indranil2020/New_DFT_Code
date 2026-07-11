#pragma once

// T3.5: XC functional evaluation on the grid (CPU reference).
// The production path uses libxc (per 10-physics/12). For the CPU reference
// and testing, we evaluate LDA-PW92 directly using the validated WP2 LdaXC
// module, and cross-check against PySCF/libxc (the oracle).
//
// Observable (T3.5): He/Ne atom total energies vs PySCF <= 1e-8 Ha.
// This is validated by combining the grid XC evaluation with the WP2 atomic
// LDA solver. The grid XC here is the on-grid V_xc and E_xc density, which
// feeds the SCF energy assembly.
//
// For PBE/GGA the production path calls libxc; the CPU reference for GGA
// requires gradient evaluation on the grid, which is a straightforward
// finite-difference extension. The LDA path is fully self-contained.

#include <cmath>
#include <cstddef>
#include <vector>

#include "grid/dual_grid.hpp"
#include "grid/xc/functionals/compose.cuh"
#include "grid/xc/functionals/lda_pw92.cuh"
#include "grid/xc/functionals/lda_slater.cuh"

namespace tides::grid {

enum class XCFunctional {
  kLDA_PW92,  // Slater X + PW92 C (launch default per 10-physics/12)
  kPBE,       // PBE GGA (Phase A; needs gradient)
};

struct XCResult {
  std::vector<double> vxc;     // V_xc(r) on the grid
  std::vector<double> eps_xc;  // eps_xc(r) = energy density per particle
};

class XCGridEvaluator {
 public:
  // Evaluate LDA XC on the grid given the density rho(r).
  // Spin-unpolarized (zeta = 0).
  static XCResult EvaluateLDA(const UniformGrid3D& grid,
                              const std::vector<double>& rho) {
    XCResult res;
    res.vxc.resize(rho.size(), 0.0);
    res.eps_xc.resize(rho.size(), 0.0);
    for (std::size_t i = 0; i < rho.size(); ++i) {
      const double n = std::max(0.0, rho[i]);
      const auto eval =
          xc::LdaSlater::Eval(n) + xc::LdaPw92::Eval(n);
      res.vxc[i] = eval.vrho;
      res.eps_xc[i] = eval.eps;
    }
    return res;
  }

  // Compute the XC energy: E_xc = integral eps_xc(r) * rho(r) d^3r.
  static double XCEnergy(const UniformGrid3D& grid, const XCResult& xc,
                         const std::vector<double>& rho) {
    const auto [h0, h1, h2] = grid.h;
    const double dv = h0 * h1 * h2;
    double E = 0.0;
    for (std::size_t i = 0; i < rho.size(); ++i)
      E += xc.eps_xc[i] * rho[i] * dv;
    return E;
  }

  // Evaluate the PBE GGA exchange enhancement factor F(s) where s is the
  // reduced density gradient. This is the analytical PBE F(s) formula:
  //   F(s) = 1 + kappa - kappa / (1 + mu * s^2 / kappa)
  // with kappa = 0.804, mu = 0.2195149727645171 (PBE parameters).
  // The full PBE needs |grad rho|, which requires a grid gradient; this
  // helper is tested analytically and feeds the GGA path.
  static double PBE_EnhancementFactor(double s) {
    const double kappa = 0.804;
    const double mu = 0.2195149727645171;
    return 1.0 + kappa - kappa / (1.0 + mu * s * s / kappa);
  }

  // Reduced density gradient s = |grad n| / (2 * (3 pi^2)^{1/3} * n^{4/3}).
  // Computed via central finite differences on the grid.
  static std::vector<double> ReducedGradient(const UniformGrid3D& grid,
                                             const std::vector<double>& rho) {
    const auto [n0, n1, n2] = grid.n;
    const auto [h0, h1, h2] = grid.h;
    std::vector<double> s(rho.size(), 0.0);
    const double prefac = 2.0 * std::pow(3.0 * M_PI * M_PI, 1.0 / 3.0);
    for (std::size_t iz = 0; iz < n2; ++iz)
      for (std::size_t iy = 0; iy < n1; ++iy)
        for (std::size_t ix = 0; ix < n0; ++ix) {
          const std::size_t g = grid.flatten(ix, iy, iz);
          const double n = rho[g];
          if (n < 1e-12) { s[g] = 0.0; continue; }
          // Central differences for grad n (zero at boundaries).
          const double dnx = (ix > 0 && ix + 1 < n0)
              ? (rho[grid.flatten(ix + 1, iy, iz)] - rho[grid.flatten(ix - 1, iy, iz)]) / (2 * h0)
              : 0.0;
          const double dny = (iy > 0 && iy + 1 < n1)
              ? (rho[grid.flatten(ix, iy + 1, iz)] - rho[grid.flatten(ix, iy - 1, iz)]) / (2 * h1)
              : 0.0;
          const double dnz = (iz > 0 && iz + 1 < n2)
              ? (rho[grid.flatten(ix, iy, iz + 1)] - rho[grid.flatten(ix, iy, iz - 1)]) / (2 * h2)
              : 0.0;
          const double grad_n = std::sqrt(dnx * dnx + dny * dny + dnz * dnz);
          s[g] = grad_n / (prefac * std::pow(n, 4.0 / 3.0));
        }
    return s;
  }
};

}  // namespace tides::grid
