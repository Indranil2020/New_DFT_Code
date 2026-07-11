#pragma once

#include "grid/xc/functionals/common.cuh"

namespace tides::grid::xc {

// Becke 88 (B88) exchange functional.  The libxc implementation uses the
// dimensionless argument x = 2^(1/3) sqrt(sigma) / rho^(4/3) and the
// enhancement factor
//   F = 1 + (2/9) beta cbrt(3)^2 cbrt(pi) cbrt(4) cbrt(2)^2 (sigma/rho^(8/3))
//         / (1 + gamma beta x asinh(x))
// with beta = 0.0042 and gamma = 6.0.
struct GgaB88 {
  static constexpr Family kFamily = Family::kGga;
  static constexpr double kDensityThreshold = 1.0e-15;
  static constexpr double kSigmaThreshold = 1.0e-20;
  static constexpr double beta = 0.0042;
  static constexpr double gamma = 6.0;

  TIDES_XC_HOST_DEVICE static GgaEvaluation Eval(double rho, double sigma) {
    if (rho <= kDensityThreshold) return {};
    const double sigma_eff =
        sigma < kSigmaThreshold * kSigmaThreshold
            ? kSigmaThreshold * kSigmaThreshold
            : sigma;

    const detail::DualRhoSigma density = detail::MakeRhoVariable(rho);
    const detail::DualRhoSigma grad = detail::MakeSigmaVariable(sigma_eff);

    const detail::DualRhoSigma rho_third = detail::Cbrt(density);
    const detail::DualRhoSigma rho_4_3 = rho_third * density;
    const detail::DualRhoSigma rho_8_3 = rho_4_3 * rho_4_3;

    const detail::DualRhoSigma sqrt_sigma = detail::Sqrt(grad);
    const double cbrt2 = detail::Cbrt(2.0);
    const detail::DualRhoSigma x = cbrt2 * sqrt_sigma / rho_4_3;

    const detail::DualRhoSigma asinh_x = detail::Asinh(x);
    const detail::DualRhoSigma denominator = 1.0 + gamma * beta * x * asinh_x;

    const double cbrt3 = detail::Cbrt(3.0);
    const double cbrt_pi = detail::Cbrt(detail::kPi);
    const double cbrt4 = detail::Cbrt(4.0);
    const double cbrt2_sq = cbrt2 * cbrt2;
    const double prefactor =
        (2.0 / 9.0) * beta * cbrt3 * cbrt3 * cbrt_pi * cbrt4 * cbrt2_sq;

    const detail::DualRhoSigma correction =
        prefactor * grad / (rho_8_3 * denominator);

    const detail::DualRhoSigma enhancement = 1.0 + correction;
    const detail::DualRhoSigma eps_per_particle =
        -0.75 * detail::Cbrt(3.0 / detail::kPi) * rho_third * enhancement;

    const detail::DualRhoSigma energy = density * eps_per_particle;
    return {eps_per_particle.value, energy.d_rho, energy.d_sigma};
  }
};

}  // namespace tides::grid::xc
