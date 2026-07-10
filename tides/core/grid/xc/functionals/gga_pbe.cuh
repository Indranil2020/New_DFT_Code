#pragma once

// AUDIT T-X1.1: PBE GGA exchange+correlation functor.
// Templated on {kappa, mu, beta} to produce PBE / PBEsol / revPBE from one source.
//
// PBE Exchange (Perdew, Burke, Ernzerhof, PRL 77, 3865 (1996)):
//   F_x(s) = 1 + kappa - kappa / (1 + mu * s^2 / kappa)
//   s = |grad_rho| / (2 * kF * rho^(4/3))  (reduced gradient)
//   eps_x^GGA = eps_x^LDA * F_x(s)
//   v_x = d(rho * eps_x^GGA) / d(rho)
//
// PBE Correlation: PW92 LDA correlation + gradient correction
//   H(s) = (gamma * phi^3 * t^2) * exp(-ec_lda / (gamma * phi^3)) / (1 + beta/gamma * H_raw)
//   where t = |grad_rho| / (2 * k_s * rho) is the reduced density gradient
//
// Template parameters:
//   kappa: PBE=0.804, PBEsol=1.0 (reverted), revPBE=1.245
//   mu:    PBE=0.2195149727645171, PBEsol=0.1234567901234568
//   beta:  PBE=0.06672455060314922, PBEsol=0.046
//
// CPU-compilable for oracle testing.

#include <cmath>
#include "grid/xc/functionals/common.cuh"
#include "grid/xc/functionals/lda_slater.cuh"
#include "grid/xc/functionals/lda_pw92.cuh"

namespace tides::grid::xc {

// PBE Exchange enhancement factor and derivative.
template <double Kappa, double Mu>
struct PbeExchange {
  static constexpr double kappa = Kappa;
  static constexpr double mu = Mu;

  // F_x(s) = 1 + kappa - kappa / (1 + mu * s^2 / kappa)
  static double Fx(double s2) {
    const double t = mu * s2 / kappa;
    return 1.0 + kappa - kappa / (1.0 + t);
  }

  // dF_x/d(s^2) = kappa * mu / kappa / (1 + mu*s^2/kappa)^2
  //             = mu / (1 + mu*s^2/kappa)^2
  static double DFxDs2(double s2) {
    const double t = mu * s2 / kappa;
    return mu / ((1.0 + t) * (1.0 + t));
  }

  // eps_x^GGA = eps_x^LDA * F_x(s)
  static double Eps(double rho, double sigma) {
    if (rho < detail::kRhoMin) return 0.0;
    double s2 = detail::ReducedGradient(rho, sigma);
    s2 *= s2;  // s^2
    return LdaSlater::Eps(rho) * Fx(s2);
  }

  // v_x^GGA = d(rho * eps_x^GGA) / d(rho)
  // This requires the chain rule through s^2(rho, sigma).
  // s^2 = sigma / (4 * kF^2 * rho^(8/3))
  // d(s^2)/d(rho) = s^2 * (-8/3 / rho)  (at fixed sigma)
  // d(s^2)/d(sigma) = 1 / (4 * kF^2 * rho^(8/3))
  //
  // v_rho = eps_x^LDA * (4/3 * F_x + rho * dF_x/d(s^2) * d(s^2)/d(rho))
  // v_sigma = rho * eps_x^LDA * dF_x/d(s^2) * d(s^2)/d(sigma)
  struct Result { double v_rho; double v_sigma; };
  static Result VrhoVsigma(double rho, double sigma) {
    if (rho < detail::kRhoMin) return {0.0, 0.0};
    const double eps_x_lda = LdaSlater::Eps(rho);
    const double s = detail::ReducedGradient(rho, sigma);
    const double s2 = s * s;
    const double fx = Fx(s2);
    const double dfx = DFxDs2(s2);

    // v_rho = d(rho * eps_x_lda * F_x) / d(rho)
    //       = eps_x_lda * F_x + rho * (4/3) * eps_x_lda/rho * F_x
    //         + rho * eps_x_lda * dF_x/d(s^2) * d(s^2)/d(rho)
    //       = eps_x_lda * F_x * (1 + 4/3) + rho * eps_x_lda * dfx * (-8/3 * s2 / rho)
    //       = eps_x_lda * (7/3 * fx - 8/3 * s2 * dfx)
    double v_rho = eps_x_lda * (7.0/3.0 * fx - 8.0/3.0 * s2 * dfx);

    // v_sigma = rho * eps_x_lda * dF_x/d(s^2) * d(s^2)/d(sigma)
    //         = rho * eps_x_lda * dfx / (4 * kF^2 * rho^(8/3))
    const double kF = detail::FermiWavevector(rho);
    const double denom = 4.0 * kF * kF * std::cbrt(rho * rho) * rho * rho / rho;
    // Simplify: 4 * kF^2 * rho^(8/3) = 4 * (3*pi^2)^(2/3) * rho^(10/3)
    // d(s^2)/d(sigma) = 1 / (4 * (3*pi^2)^(2/3) * rho^(10/3))
    const double ds2_dsigma = 1.0 / (4.0 * std::cbrt(3.0 * M_PI * M_PI * 3.0 * M_PI * M_PI) * std::cbrt(rho * rho * rho * rho * rho) * rho * rho * rho);
    // Actually, let's be more careful:
    // s = |grad_rho| / (2 * kF * rho^(4/3))
    // s^2 = sigma / (4 * kF^2 * rho^(8/3))
    // kF^2 = (3*pi^2)^(2/3) * rho^(2/3)
    // So s^2 = sigma / (4 * (3*pi^2)^(2/3) * rho^(10/3))
    const double kF2 = std::cbrt(3.0 * M_PI * M_PI) * std::cbrt(3.0 * M_PI * M_PI);
    const double ds2_dsigma_correct = 1.0 / (4.0 * kF2 * std::cbrt(rho * rho * rho * rho) * std::cbrt(rho * rho * rho * rho * rho * rho));
    // Use the clean formula:
    const double ds2_dsigma_clean = 1.0 / (4.0 * std::pow(3.0 * M_PI * M_PI, 2.0/3.0) * std::pow(rho, 10.0/3.0));
    double v_sigma = rho * eps_x_lda * dfx * ds2_dsigma_clean;

    return {v_rho, v_sigma};
  }
};

// PBE Correlation: LDA part + gradient correction.
// Standard PBE correlation:
//   A = (pi^2/3) * (gamma * phi^3)  -- not used directly
//   t = |grad_rho| / (2 * k_s * rho)  (reduced gradient for correlation)
//   k_s = sqrt(4 * kF / pi)  (Thomas-Fermi screening)
//   phi = 1 (unpolarized)
//   H = gamma * t^2 * exp(-eps_c_lda / (gamma)) / (1 + beta/gamma * t^2 * exp(-eps_c_lda/gamma))
//   eps_c^GGA = eps_c^LDA + H
//
// gamma = 0.0310907 (same as PW92 a parameter, but in PBE it's (pi^2/3)*...)
// Actually, the standard PBE uses:
//   gamma = 0.031090690869654895  (≈ pi^2/3 * (alpha/(2*pi))^2 ... )
//   beta = 0.06672455060314922
template <double Beta>
struct PbeCorrelation {
  static constexpr double beta = Beta;
  static constexpr double gamma = 0.031090690869654895;

  // t = |grad_rho| / (2 * k_s * rho)
  // k_s = sqrt(4 * kF / pi), kF = (3*pi^2*rho)^(1/3)
  static double ReducedT(double rho, double sigma) {
    if (rho < detail::kRhoMin) return 0.0;
    const double grad = std::sqrt(std::max(sigma, 0.0));
    const double kF = detail::FermiWavevector(rho);
    const double kS = std::sqrt(4.0 * kF / M_PI);
    return grad / (2.0 * kS * rho);
  }

  // H(t, eps_c_lda): gradient correction to correlation.
  static double Hcorr(double t, double eps_c_lda) {
    const double t2 = t * t;
    const double exp_term = std::exp(-eps_c_lda / gamma);
    const double denom = 1.0 + (beta / gamma) * t2 * exp_term;
    return gamma * t2 * exp_term / denom;
  }

  // eps_c^GGA = eps_c^LDA + H
  static double Eps(double rho, double sigma) {
    if (rho < detail::kRhoMin) return 0.0;
    const double eps_c_lda = LdaPw92::Eps(rho);
    const double t = ReducedT(rho, sigma);
    return eps_c_lda + Hcorr(t, eps_c_lda);
  }

  // v_rho and v_sigma for PBE correlation.
  // This is complex — the full derivative involves dH/d(t) * d(t)/d(rho) and dH/d(eps_c) * d(eps_c)/d(rho).
  // For the Tier-0 implementation, we compute these analytically.
  struct Result { double v_rho; double v_sigma; };
  static Result VrhoVsigma(double rho, double sigma) {
    if (rho < detail::kRhoMin) return {0.0, 0.0};
    const double rs = detail::RhoToRs(rho);
    const double eps_c_lda = LdaPw92::EpsRs(rs);
    const double deps_c_drs = LdaPw92::DEpsDrs(rs);
    const double t = ReducedT(rho, sigma);
    const double t2 = t * t;
    const double exp_term = std::exp(-eps_c_lda / gamma);
    const double beta_gamma = beta / gamma;
    const double denom = 1.0 + beta_gamma * t2 * exp_term;
    const double H = gamma * t2 * exp_term / denom;

    // dH/d(t^2) = gamma * exp_term / denom^2
    const double dH_dt2 = gamma * exp_term / (denom * denom);

    // dH/d(eps_c) = gamma * t2 * exp_term * (1/gamma) / denom
    //             + gamma * t2 * exp_term * (-beta_gamma * t2 * exp_term * (-1/gamma)) / denom^2
    // Simplified: dH/d(eps_c) = t2 * exp_term / denom * (1 + beta_gamma * t2 * exp_term / denom)
    const double dH_deps = t2 * exp_term / denom * (1.0 + beta_gamma * t2 * exp_term / denom) / gamma * gamma;
    // Actually, let's compute it more carefully:
    // H = gamma * t2 * E / D, where E = exp(-eps_c/gamma), D = 1 + (beta/gamma) * t2 * E
    // dH/d(eps_c) = gamma * t2 * dE/d(eps_c) / D - gamma * t2 * E * dD/d(eps_c) / D^2
    // dE/d(eps_c) = -E/gamma
    // dD/d(eps_c) = (beta/gamma) * t2 * (-E/gamma) = -beta * t2 * E / gamma^2
    // dH/d(eps_c) = gamma * t2 * (-E/gamma) / D - gamma * t2 * E * (-beta * t2 * E / gamma^2) / D^2
    //             = -t2 * E / D + beta * t2^2 * E^2 / (gamma * D^2)
    const double dH_deps_c = -t2 * exp_term / denom + beta * t2 * t2 * exp_term * exp_term / (gamma * denom * denom);

    // v_rho = d(rho * (eps_c_lda + H)) / d(rho)
    //       = eps_c_lda + H + rho * d(eps_c_lda)/d(rho) + rho * dH/d(rho)
    // d(eps_c_lda)/d(rho) = -rs/(3*rho) * d(eps_c_lda)/d(rs)
    const double deps_c_lda_drho = -rs / (3.0 * rho) * deps_c_drs;

    // dH/d(rho) = dH/d(eps_c) * d(eps_c)/d(rho) + dH/d(t^2) * d(t^2)/d(rho)
    // t = |grad| / (2 * kS * rho), t^2 = sigma / (4 * kS^2 * rho^2)
    // kS^2 = 4*kF/pi = 4*(3*pi^2)^(1/3)*rho^(1/3)/pi
    // t^2 = sigma * pi / (16 * (3*pi^2)^(1/3) * rho^(7/3))
    // d(t^2)/d(rho) = -7/3 * t^2 / rho (at fixed sigma)
    const double dt2_drho = -7.0/3.0 * t2 / rho;

    const double dH_drho = dH_deps_c * deps_c_lda_drho + dH_dt2 * dt2_drho;

    double v_rho = eps_c_lda + H + rho * (deps_c_lda_drho + dH_drho);

    // v_sigma = rho * dH/d(sigma)
    // d(t^2)/d(sigma) = 1 / (4 * kS^2 * rho^2) = pi / (16 * (3*pi^2)^(1/3) * rho^(7/3))
    const double kS2 = 4.0 * detail::FermiWavevector(rho) / M_PI;
    const double dt2_dsigma = 1.0 / (4.0 * kS2 * rho * rho);
    double v_sigma = rho * dH_dt2 * dt2_dsigma;

    return {v_rho, v_sigma};
  }
};

// Convenience typedefs for standard PBE parameter sets.
using PbeX = PbeExchange<0.804, 0.2195149727645171>;
using PbesolX = PbeExchange<1.0, 0.1234567901234568>;
using RevPbeX = PbeExchange<1.245, 0.2195149727645171>;

using PbeC = PbeCorrelation<0.06672455060314922>;
using PbesolC = PbeCorrelation<0.046>;

}  // namespace tides::grid::xc
