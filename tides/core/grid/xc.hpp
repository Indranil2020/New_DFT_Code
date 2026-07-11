#pragma once

// CPU reference XC evaluation for testing and validation only (§2.3).
// Production code must call tides::grid::xc::XcEval (via xc_host_bridge.hpp
// for host data). This class exists to provide an independent CPU oracle
// for cross-checking GPU results.

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
  // Tested analytically; production GGA uses XcEval with analytic gradients.
  static double PBE_EnhancementFactor(double s) {
    const double kappa = 0.804;
    const double mu = 0.2195149727645171;
    return 1.0 + kappa - kappa / (1.0 + mu * s * s / kappa);
  }

};

}  // namespace tides::grid
