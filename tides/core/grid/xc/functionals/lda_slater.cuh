#pragma once

#include "grid/xc/functionals/common.cuh"

namespace tides::grid::xc {

struct LdaSlater {
  TIDES_XC_HOST_DEVICE static double Eps(double rho) {
    // Mirrors libxc's generated LDA_X guard: rho / 2 <= dens_threshold.
    if (rho * 0.5 <= detail::kLdaPw92DensityThreshold) return 0.0;
    return -0.75 * detail::Cbrt(3.0 / detail::kPi) * detail::Cbrt(rho);
  }

  TIDES_XC_HOST_DEVICE static double V(double rho) {
    return (4.0 / 3.0) * Eps(rho);
  }

  TIDES_XC_HOST_DEVICE static GgaEvaluation Eval(double rho) {
    if (rho * 0.5 <= detail::kLdaPw92DensityThreshold) return {};
    const detail::DualRhoSigma density = detail::MakeRhoVariable(rho);
    const detail::DualRhoSigma eps =
        -0.75 * detail::Cbrt(3.0 / detail::kPi) * detail::Cbrt(density);
    const detail::DualRhoSigma energy = density * eps;
    return {eps.value, energy.d_rho, 0.0};
  }
};

}  // namespace tides::grid::xc
