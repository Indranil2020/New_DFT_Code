#pragma once

// AUDIT T-X1.1: LDA Slater exchange functor.
// eps_x = -(3/pi)^(1/3) * rho^(1/3) * (3/4) = -0.75 * (3/pi)^(1/3) * rho^(1/3)
// v_x = d(rho*eps_x)/d(rho) = -(3/pi)^(1/3) * rho^(1/3) = eps_x / 0.75 * (4/3)
// More precisely: v_x = -(4/3) * (3/(4*pi))^(1/3) * rho^(1/3)
//                  = -(3/pi)^(1/3) * (4/3) * rho^(1/3) / (4^(1/3) * 4^(1/3))
// Standard form: v_x = -(3/pi)^(1/3) * rho^(1/3) * (4/3)
//
// For spin-polarized: v_x,sigma = 2^(1/3) * v_x(unpolarized with rho_sigma)
//
// CPU-compilable for oracle testing.

#include <cmath>
#include "grid/xc/functionals/common.cuh"

namespace tides::grid::xc {

struct LdaSlater {
  // Energy per particle: eps_x(rho)
  static TIDES_HD double Eps(double rho) {
    if (rho < detail::kRhoMin) return 0.0;
    const double cbrt_rho = std::cbrt(rho);
    return -0.75 * std::cbrt(3.0 / M_PI) * cbrt_rho;
  }

  // Potential: v_x(rho) = d(rho*eps_x)/d(rho)
  static TIDES_HD double Vrho(double rho) {
    if (rho < detail::kRhoMin) return 0.0;
    const double cbrt_rho = std::cbrt(rho);
    return -std::cbrt(3.0 / M_PI) * cbrt_rho;  // = -(4/3) * (3/(4*pi))^(1/3) * rho^(1/3)
  }

  // Energy density: rho * eps_x
  static TIDES_HD double EnergyDensity(double rho) {
    return rho * Eps(rho);
  }
};

}  // namespace tides::grid::xc
