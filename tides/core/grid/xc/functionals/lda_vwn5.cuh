#pragma once

#include "grid/xc/functionals/common.cuh"

namespace tides::grid::xc {

// Vosko-Wilk-Nusair (VWN5) correlation functional.  This is the unpolarized
// (paramagnetic) branch; the spin interpolation is added later with the same
// zeta-scaling as the rest of the LDA suite.
struct LdaVwn5 {
  // VWN5 paramagnetic parameters: A, b, c, x0.
  // The formula is the standard one from libxc's LDA_C_VWN:
  //   e_c = A [ ln(x^2 / (x^2 + b x + c))
  //          + (2 b / Q) atan(Q / (2 x + b))
  //          - (b x0 / (x0^2 + b x0 + c)) (
  //              ln((x - x0)^2 / (x^2 + b x + c))
  //            + (2 (2 x0 + b) / Q) atan(Q / (2 x + b)) ) ]
  // with x = sqrt(rs), Q = sqrt(4 c - b^2).
  TIDES_XC_HOST_DEVICE static GgaEvaluation Eval(double rho) {
    if (rho <= detail::kLdaPw92DensityThreshold) return {};
    const detail::DualRhoSigma density = detail::MakeRhoVariable(rho);
    const detail::DualRhoSigma rs =
        detail::Cbrt(3.0 / (4.0 * detail::kPi * density));
    const detail::DualRhoSigma eps = EvalParamagnetic(rs);
    const detail::DualRhoSigma energy = density * eps;
    return {eps.value, energy.d_rho, 0.0};
  }

  TIDES_XC_HOST_DEVICE static double EpsCorrelation(double rho) {
    if (rho <= detail::kLdaPw92DensityThreshold) return 0.0;
    const double rs = detail::Cbrt(3.0 / (4.0 * detail::kPi * rho));
    return EvalRs(detail::Dual(rs, 0.0)).value;
  }

  TIDES_XC_HOST_DEVICE static double VCorrelation(double rho) {
    return Eval(rho).vrho;
  }

 private:
  struct Params {
    double A;
    double b;
    double c;
    double x0;
  };

  TIDES_XC_HOST_DEVICE static constexpr Params Paramagnetic() {
    return {0.0310907, 3.72744, 12.9352, -0.10498};
  }

  TIDES_XC_HOST_DEVICE static detail::DualRhoSigma EvalParamagnetic(
      const detail::DualRhoSigma& rs) {
    const detail::Dual rs_one(rs.value, rs.d_rho);
    const detail::Dual eps = EvalRs(rs_one);
    return {eps.value, eps.derivative, 0.0};
  }

  // The energy density as a function of rs (dual one-variable version).
  TIDES_XC_HOST_DEVICE static detail::Dual EvalRs(const detail::Dual& rs) {
    const Params p = Paramagnetic();
    const detail::Dual x = detail::Sqrt(rs);
    const detail::Dual x2 = x * x;
    const detail::Dual bx = p.b * x;
    const detail::Dual D = x2 + bx + p.c;
    const double Q = detail::Sqrt(4.0 * p.c - p.b * p.b);

    // The derivative of log((x - x0)^2 / D) is handled with Dual as long as
    // the argument is positive.  For x > |x0| this is fine.
    const detail::Dual term1 = detail::Log(x2 / D);
    const detail::Dual term2 =
        (2.0 * p.b / Q) * detail::Atan(Q / (2.0 * x + p.b));

    const double x0_2 = p.x0 * p.x0;
    const double denom_x0 = x0_2 + p.b * p.x0 + p.c;
    const double coeff_x0 = p.b * p.x0 / denom_x0;
    const detail::Dual arg_x0 =
        ((x - p.x0) * (x - p.x0)) / D;
    const detail::Dual term3_inner =
        detail::Log(arg_x0) +
        (2.0 * (2.0 * p.x0 + p.b) / Q) *
            detail::Atan(Q / (2.0 * x + p.b));

    return p.A * (term1 + term2 - coeff_x0 * term3_inner);
  }
};

}  // namespace tides::grid::xc
