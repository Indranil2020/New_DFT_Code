#pragma once

#include "grid/xc/functionals/common.cuh"

namespace tides::grid::xc {

// Perdew-Wang 1992 correlation constants.  The parameter sets mirror libxc's
// LDA_C_PW exactly; the modified PBE parameterization is intentionally absent.
struct LdaPw92 {
  struct Correlation {
    double eps = 0.0;
    double d_eps_d_rs = 0.0;
  };

  TIDES_XC_HOST_DEVICE static Correlation EvalParamagnetic(double rs) {
    return Eval(rs, 0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294);
  }

  TIDES_XC_HOST_DEVICE static Correlation EvalFerromagnetic(double rs) {
    return Eval(rs, 0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517);
  }

  TIDES_XC_HOST_DEVICE static double RsFromDensity(double rho) {
    return detail::Cbrt(3.0 / (4.0 * detail::kPi * rho));
  }

  TIDES_XC_HOST_DEVICE static double EpsCorrelation(double rho) {
    if (rho < detail::kLdaPw92DensityThreshold) return 0.0;
    return EvalParamagnetic(RsFromDensity(rho)).eps;
  }

  TIDES_XC_HOST_DEVICE static double VCorrelation(double rho) {
    if (rho < detail::kLdaPw92DensityThreshold) return 0.0;
    const double rs = RsFromDensity(rho);
    const Correlation correlation = EvalParamagnetic(rs);
    return correlation.eps - rs * correlation.d_eps_d_rs / 3.0;
  }

  TIDES_XC_HOST_DEVICE static GgaEvaluation Eval(double rho) {
    if (rho < detail::kLdaPw92DensityThreshold) return {};
    return {EpsCorrelation(rho), VCorrelation(rho), 0.0};
  }

 private:
  TIDES_XC_HOST_DEVICE static Correlation Eval(
      double rs, double a, double alpha1, double beta1, double beta2,
      double beta3, double beta4) {
    const double sqrt_rs = detail::Sqrt(rs);
    const double q = beta1 * sqrt_rs + beta2 * rs + beta3 * rs * sqrt_rs +
                     beta4 * rs * rs;
    const double one_plus_alpha = 1.0 + alpha1 * rs;
    const double log_argument = 1.0 + 1.0 / (2.0 * a * q);
    const double log_value = detail::Log(log_argument);
    const double d_q_d_rs = beta1 / (2.0 * sqrt_rs) + beta2 +
                             1.5 * beta3 * sqrt_rs + 2.0 * beta4 * rs;
    const double d_log_argument_d_rs = -d_q_d_rs / (2.0 * a * q * q);

    Correlation result;
    result.eps = -2.0 * a * one_plus_alpha * log_value;
    result.d_eps_d_rs = -2.0 * a *
        (alpha1 * log_value +
         one_plus_alpha * d_log_argument_d_rs / log_argument);
    return result;
  }
};

}  // namespace tides::grid::xc
