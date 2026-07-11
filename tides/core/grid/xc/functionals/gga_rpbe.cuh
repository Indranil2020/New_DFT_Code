#pragma once

#include "grid/xc/functionals/common.cuh"

namespace tides::grid::xc {

// RPBE exchange (Hammer, Hansen, Norskov) in unpolarized form.  The RPBE
// enhancement factor is F = 1 + kappa * (1 - exp(-mu * s^2 / kappa)), where
// s is the reduced gradient.  The constants are the standard PBE values.
struct GgaRpbe {
  static constexpr Family kFamily = Family::kGga;
  static constexpr double kDensityThreshold = 1.0e-15;
  static constexpr double kSigmaThreshold = 1.0e-20;
  static constexpr double kappa = 0.8040;
  static constexpr double mu = 0.2195149727645171;

  TIDES_XC_HOST_DEVICE static GgaEvaluation Eval(double rho, double sigma) {
    if (rho <= kDensityThreshold) return {};
    const double sigma_eff =
        sigma < kSigmaThreshold * kSigmaThreshold
            ? kSigmaThreshold * kSigmaThreshold
            : sigma;

    const detail::DualRhoSigma density = detail::MakeRhoVariable(rho);
    const detail::DualRhoSigma grad = detail::MakeSigmaVariable(sigma_eff);

    const detail::DualRhoSigma rho_third = detail::Cbrt(density);
    const detail::DualRhoSigma rho_8_3 = rho_third * rho_third * rho_third *
                                         rho_third * rho_third * rho_third *
                                         rho_third * rho_third;

    // s^2 = sigma / (4 * (3 pi^2)^(2/3) * rho^(8/3))
    const double cbrt_3pi2 = detail::Cbrt(3.0 * detail::kPi * detail::kPi);
    const double s2_denominator = 4.0 * cbrt_3pi2 * cbrt_3pi2;
    const detail::DualRhoSigma s2 = grad / (s2_denominator * rho_8_3);

    const detail::DualRhoSigma enhancement =
        1.0 + kappa * (1.0 - detail::Exp(-(mu / kappa) * s2));

    const detail::DualRhoSigma eps_per_particle =
        -0.75 * detail::Cbrt(3.0 / detail::kPi) * rho_third * enhancement;

    const detail::DualRhoSigma energy = density * eps_per_particle;
    return {eps_per_particle.value, energy.d_rho, energy.d_sigma};
  }
};

}  // namespace tides::grid::xc
