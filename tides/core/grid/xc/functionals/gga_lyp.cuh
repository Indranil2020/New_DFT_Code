#pragma once

#include "grid/xc/functionals/common.cuh"

namespace tides::grid::xc {

// Lee-Yang-Parr (LYP) correlation functional.  The unpolarized expression is
// derived from libxc's maple2c gga_c_lyp after the zeta=1 screening terms are
// set to unity (the unpolarized limit).  Forward-mode differentiation over
// rho and sigma gives the exact vrho and vsigma.
struct GgaLyp {
  static constexpr Family kFamily = Family::kGga;
  static constexpr double kDensityThreshold = 1.0e-15;
  static constexpr double kSigmaThreshold = 1.0e-20;
  static constexpr double a = 0.04918;
  static constexpr double b = 0.132;
  static constexpr double c = 0.2533;
  static constexpr double d = 0.349;

  TIDES_XC_HOST_DEVICE static GgaEvaluation Eval(double rho, double sigma) {
    if (rho <= kDensityThreshold) return {};
    const double sigma_eff =
        sigma < kSigmaThreshold * kSigmaThreshold
            ? kSigmaThreshold * kSigmaThreshold
            : sigma;

    const detail::DualRhoSigma density = detail::MakeRhoVariable(rho);
    const detail::DualRhoSigma grad = detail::MakeSigmaVariable(sigma_eff);

    // x = rho^{-1/3}
    const detail::DualRhoSigma x = detail::Cbrt(1.0 / density);
    const detail::DualRhoSigma t5 = 1.0 / (1.0 + d * x);
    const detail::DualRhoSigma t8 = b * detail::Exp(-c * x);
    const detail::DualRhoSigma t16 = x * (c + d * t5);

    // x^8 = rho^{-8/3}
    const detail::DualRhoSigma x2 = x * x;
    const detail::DualRhoSigma x4 = x2 * x2;
    const detail::DualRhoSigma x8 = x4 * x4;

    const detail::DualRhoSigma t62 =
        grad * x8 * (3.0 + 7.0 * t16) / 72.0 -
        (3.0 / 10.0) * detail::Cbrt(detail::kPi * detail::kPi * detail::kPi * detail::kPi);

    const detail::DualRhoSigma eps_per_particle = a * (t8 * t5 * t62 - t5);
    const detail::DualRhoSigma energy = density * eps_per_particle;

    return {eps_per_particle.value, energy.d_rho, energy.d_sigma};
  }
};

}  // namespace tides::grid::xc
