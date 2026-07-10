#pragma once

// AUDIT T-X0.2/T-X1.1: LDA PW92 correlation functor with analytic derivatives.
// Replaces the finite-difference d(eps_c)/d(rs) hack (audit B2).
//
// PW92 parametrization (Perdew & Wang, PRB 45, 13244 (1992)):
//   eps_c(rs) = -2*a*(1 + a1*rs) * ln(1 + 1/(2*a*Q))
//   Q = b1*sqrt(rs) + b2*rs + b3*rs*sqrt(rs) + b4*rs^2
//
// Analytic derivative:
//   d(eps_c)/d(rs) = -2*a1 * ln(1 + 1/(2*a*Q))
//                    - 2*a*(1 + a1*rs) * (1/(1 + 1/(2*a*Q))) * (-1/(2*a*Q^2)) * dQ/d(rs)
//   dQ/d(rs) = b1/(2*sqrt(rs)) + b2 + 3*b3*sqrt(rs)/2 + 2*b4*rs
//
// v_c = d(rho*eps_c)/d(rho) = eps_c + rho * d(eps_c)/d(rho)
//      = eps_c - (rs/3) * d(eps_c)/d(rs)
//
// CPU-compilable for oracle testing.

#include <cmath>
#include "grid/xc/functionals/common.cuh"

namespace tides::grid::xc {

struct LdaPw92 {
  // PW92 parameters (paramagnetic, unpolarized).
  static constexpr double a_  = 0.0310907;
  static constexpr double a1_ = 0.21370;
  static constexpr double b1_ = 7.5957;
  static constexpr double b2_ = 3.5876;
  static constexpr double b3_ = 1.6382;
  static constexpr double b4_ = 0.49294;

  // Q(rs) = b1*sqrt(rs) + b2*rs + b3*rs*sqrt(rs) + b4*rs^2
  static TIDES_HD double Q(double rs) {
    const double sqrt_rs = std::sqrt(rs);
    return b1_ * sqrt_rs + b2_ * rs + b3_ * rs * sqrt_rs + b4_ * rs * rs;
  }

  // dQ/d(rs) = b1/(2*sqrt(rs)) + b2 + 3*b3*sqrt(rs)/2 + 2*b4*rs
  static TIDES_HD double DQDrs(double rs) {
    if (rs < 1e-15) return 0.0;
    const double sqrt_rs = std::sqrt(rs);
    return b1_ / (2.0 * sqrt_rs) + b2_ + 1.5 * b3_ * sqrt_rs + 2.0 * b4_ * rs;
  }

  // eps_c(rs): correlation energy per particle.
  static TIDES_HD double EpsRs(double rs) {
    if (rs < 1e-15 || rs > 1e10) return 0.0;
    const double q = Q(rs);
    const double arg = 1.0 + 1.0 / (2.0 * a_ * q);
    return -2.0 * a_ * (1.0 + a1_ * rs) * std::log(arg);
  }

  // Analytic d(eps_c)/d(rs) — replaces finite-difference hack (audit B2).
  static TIDES_HD double DEpsDrs(double rs) {
    if (rs < 1e-15 || rs > 1e10) return 0.0;
    const double q = Q(rs);
    const double dq = DQDrs(rs);
    const double two_aq = 2.0 * a_ * q;
    const double log_arg = std::log(1.0 + 1.0 / two_aq);

    // d(eps_c)/d(rs) = -2*a1*ln(1 + 1/(2aQ))
    //                  - 2a*(1+a1*rs) * (-1/(2aQ^2)) / (1 + 1/(2aQ)) * dQ/d(rs)
    const double term1 = -2.0 * a1_ * log_arg;
    const double denom = (1.0 + 1.0 / two_aq) * two_aq * two_aq;
    const double term2 = 2.0 * a_ * (1.0 + a1_ * rs) * dq / denom;
    return term1 + term2;
  }

  // eps_c(n): convenience wrapper converting density to rs.
  static TIDES_HD double Eps(double rho) {
    if (rho < detail::kRhoMin) return 0.0;
    return EpsRs(detail::RhoToRs(rho));
  }

  // v_c(rho) = eps_c + rho * d(eps_c)/d(rho)
  //          = eps_c - (rs/3) * d(eps_c)/d(rs)
  static TIDES_HD double Vrho(double rho) {
    if (rho < detail::kRhoMin) return 0.0;
    const double rs = detail::RhoToRs(rho);
    const double eps_c = EpsRs(rs);
    const double deps_drs = DEpsDrs(rs);
    return eps_c - (rs / 3.0) * deps_drs;
  }

  // Energy density: rho * eps_c
  static TIDES_HD double EnergyDensity(double rho) {
    return rho * Eps(rho);
  }
};

}  // namespace tides::grid::xc
