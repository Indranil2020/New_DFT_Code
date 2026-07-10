#pragma once

#include "grid/xc/functionals/common.cuh"

namespace tides::grid::xc {

struct GgaEvaluation {
  double eps = 0.0;
  double vrho = 0.0;
  double vsigma = 0.0;
};

struct PbeParameters {
  static constexpr double kappa = 0.8040;
  static constexpr double mu = 0.2195149727645171;
  static constexpr double beta = 0.06672455060314922;
};

struct PbeSolParameters {
  static constexpr double kappa = 0.8040;
  static constexpr double mu = 10.0 / 81.0;
  static constexpr double beta = 0.046;
};

struct RevPbeParameters {
  static constexpr double kappa = 1.245;
  static constexpr double mu = 0.2195149727645171;
  static constexpr double beta = 0.06672455060314922;
};

namespace detail {

// Forward-mode differentiation of the local energy density e(rho, sigma).
// This is analytic chain-rule differentiation, not a finite difference.
struct DualRhoSigma {
  double value;
  double d_rho;
  double d_sigma;

  TIDES_XC_HOST_DEVICE DualRhoSigma(double value_in = 0.0,
                                    double d_rho_in = 0.0,
                                    double d_sigma_in = 0.0)
      : value(value_in), d_rho(d_rho_in), d_sigma(d_sigma_in) {}
};

TIDES_XC_HOST_DEVICE inline DualRhoSigma operator+(
    DualRhoSigma lhs, DualRhoSigma rhs) {
  return {lhs.value + rhs.value, lhs.d_rho + rhs.d_rho,
          lhs.d_sigma + rhs.d_sigma};
}

TIDES_XC_HOST_DEVICE inline DualRhoSigma operator-(
    DualRhoSigma lhs, DualRhoSigma rhs) {
  return {lhs.value - rhs.value, lhs.d_rho - rhs.d_rho,
          lhs.d_sigma - rhs.d_sigma};
}

TIDES_XC_HOST_DEVICE inline DualRhoSigma operator-(DualRhoSigma value) {
  return {-value.value, -value.d_rho, -value.d_sigma};
}

TIDES_XC_HOST_DEVICE inline DualRhoSigma operator*(
    DualRhoSigma lhs, DualRhoSigma rhs) {
  return {lhs.value * rhs.value,
          lhs.d_rho * rhs.value + lhs.value * rhs.d_rho,
          lhs.d_sigma * rhs.value + lhs.value * rhs.d_sigma};
}

TIDES_XC_HOST_DEVICE inline DualRhoSigma operator/(
    DualRhoSigma lhs, DualRhoSigma rhs) {
  const double inverse = 1.0 / rhs.value;
  const double quotient = lhs.value * inverse;
  return {quotient,
          (lhs.d_rho - quotient * rhs.d_rho) * inverse,
          (lhs.d_sigma - quotient * rhs.d_sigma) * inverse};
}

TIDES_XC_HOST_DEVICE inline DualRhoSigma Cbrt(DualRhoSigma value) {
  const double root = detail::Cbrt(value.value);
  const double factor = 1.0 / (3.0 * root * root);
  return {root, factor * value.d_rho, factor * value.d_sigma};
}

TIDES_XC_HOST_DEVICE inline DualRhoSigma Sqrt(DualRhoSigma value) {
  const double root = detail::Sqrt(value.value);
  const double factor = 0.5 / root;
  return {root, factor * value.d_rho, factor * value.d_sigma};
}

TIDES_XC_HOST_DEVICE inline DualRhoSigma Log(DualRhoSigma value) {
  const double factor = 1.0 / value.value;
  return {detail::Log(value.value), factor * value.d_rho, factor * value.d_sigma};
}

TIDES_XC_HOST_DEVICE inline DualRhoSigma Log1p(DualRhoSigma value) {
  const double factor = 1.0 / (1.0 + value.value);
  return {detail::Log1p(value.value), factor * value.d_rho,
          factor * value.d_sigma};
}

TIDES_XC_HOST_DEVICE inline DualRhoSigma Exp(DualRhoSigma value) {
  const double exponential = detail::Exp(value.value);
  return {exponential, exponential * value.d_rho,
          exponential * value.d_sigma};
}

TIDES_XC_HOST_DEVICE inline DualRhoSigma Expm1(DualRhoSigma value) {
  const double exponential = detail::Exp(value.value);
  return {detail::Expm1(value.value), exponential * value.d_rho,
          exponential * value.d_sigma};
}

}  // namespace detail

// PBE-family unpolarized GGA. The correlation expression follows libxc's
// checked-in LDA_C_PW_MOD/PBE algebra and threshold semantics exactly. The
// parameter type changes only the documented PBE-family constants.
template <class Parameters>
struct GgaPbe {
  static constexpr double kExchangeDensityThreshold = 1.0e-15;
  static constexpr double kExchangeSigmaThreshold = 1.0e-20;
  static constexpr double kCorrelationDensityThreshold = 1.0e-12;
  static constexpr double kCorrelationSigmaThreshold = 1.0e-16;

  TIDES_XC_HOST_DEVICE static GgaEvaluation Eval(double rho, double sigma) {
    const GgaEvaluation exchange = EvalExchange(rho, sigma);
    const GgaEvaluation correlation = EvalCorrelation(rho, sigma);
    return {exchange.eps + correlation.eps, exchange.vrho + correlation.vrho,
            exchange.vsigma + correlation.vsigma};
  }

  // Exposed separately for compile-time composition of PBE-family hybrids.
  TIDES_XC_HOST_DEVICE static GgaEvaluation EvalExchange(double rho,
                                                           double sigma) {
    // The generated unpolarized LDA-X/PBE-X branch screens rho/2.
    if (rho * 0.5 <= kExchangeDensityThreshold) return {};
    const double sigma_eff = sigma < kExchangeSigmaThreshold * kExchangeSigmaThreshold
        ? kExchangeSigmaThreshold * kExchangeSigmaThreshold : sigma;
    const double cbrt_three = detail::Cbrt(3.0);
    const double cbrt_pi = detail::Cbrt(detail::kPi);
    const double cbrt_two = detail::Cbrt(2.0);
    const double rho_third = detail::Cbrt(rho);
    const double rho_two_thirds = rho_third * rho_third;
    const double rho_squared = rho * rho;
    const double pbe_denominator = Parameters::kappa + Parameters::mu *
        detail::Cbrt(6.0) / (cbrt_pi * cbrt_pi * cbrt_pi * cbrt_pi) *
        sigma_eff * cbrt_two * cbrt_two /
        (24.0 * rho_two_thirds * rho_squared);
    const double enhancement = 1.0 + Parameters::kappa *
        (1.0 - Parameters::kappa / pbe_denominator);
    const double eps = -0.75 * cbrt_three / cbrt_pi * rho_third * enhancement;

    // Direct analytic derivatives in the same operation order as libxc's
    // maple2c PBE-X worker.  This avoids losing the X/C cancellation in vσ.
    const double rho_cubed = rho_squared * rho;
    const double kappa_squared = Parameters::kappa * Parameters::kappa;
    const double t58 = cbrt_three / cbrt_pi / rho_third / rho_cubed *
        kappa_squared;
    const double inverse_denominator_squared =
        Parameters::mu / (pbe_denominator * pbe_denominator);
    const double t64 = sigma_eff * cbrt_two * cbrt_two /
        (cbrt_pi * cbrt_pi * cbrt_pi * cbrt_pi);
    const double t65 = inverse_denominator_squared * detail::Cbrt(6.0) * t64;
    const double rho_derivative_density = -cbrt_three / cbrt_pi /
        rho_two_thirds * enhancement / 8.0 + t58 * t65 / 24.0;
    const double vrho = 2.0 * rho * rho_derivative_density -
        0.75 * cbrt_three / cbrt_pi * rho_third * enhancement;
    const double t79 = inverse_denominator_squared * detail::Cbrt(6.0) /
        (cbrt_pi * cbrt_pi * cbrt_pi * cbrt_pi) * cbrt_two * cbrt_two;
    const double vsigma = 2.0 * rho * (-cbrt_three / cbrt_pi /
        rho_third / rho_squared * kappa_squared * t79 / 64.0);
    return {eps, vrho, vsigma};
  }

  TIDES_XC_HOST_DEVICE static GgaEvaluation EvalCorrelation(double rho,
                                                             double sigma) {
    if (rho < kCorrelationDensityThreshold) return {};
    // This is the unpolarized LDA_C_PW_MOD/PBE-C algebra used by libxc.
    // Forward-mode differentiation supplies the exact first derivatives of
    // the local energy density without numerical differencing.
    const double sigma_eff =
        sigma < kCorrelationSigmaThreshold * kCorrelationSigmaThreshold
            ? kCorrelationSigmaThreshold * kCorrelationSigmaThreshold
            : sigma;
    const detail::DualRhoSigma density(rho, 1.0, 0.0);
    const detail::DualRhoSigma gradient(sigma_eff, 0.0, 1.0);
    constexpr double kLibxcCbrtTwo = 1.259921049894873164767210607278228350570;
    constexpr double kLibxcCbrtThree = 1.442249570307408382321638310780109588392;
    constexpr double kLibxcCbrtFour = 1.587401051968199474751705639272308260391;
    const double cbrt_three = kLibxcCbrtThree;
    const double cbrt_one_over_pi = detail::Cbrt(1.0 / detail::kPi);
    const double cbrt_four = kLibxcCbrtFour;
    const double cbrt_two = kLibxcCbrtTwo;
    const detail::DualRhoSigma rho_third = detail::Cbrt(density);
    const detail::DualRhoSigma rho_two_thirds = rho_third * rho_third;
    const detail::DualRhoSigma rs_four =
        cbrt_three * cbrt_one_over_pi * cbrt_four * cbrt_four / rho_third;
    const detail::DualRhoSigma sqrt_rs_four = detail::Sqrt(rs_four);
    const detail::DualRhoSigma rs_four_three_halves = rs_four * sqrt_rs_four;
    const detail::DualRhoSigma q =
        3.79785 * sqrt_rs_four + 0.8969 * rs_four +
        0.204775 * rs_four_three_halves +
        0.123235 * (cbrt_three * cbrt_three *
                     cbrt_one_over_pi * cbrt_one_over_pi * cbrt_four /
                     rho_two_thirds);
    const detail::DualRhoSigma lda =
        -0.0621814 * (1.0 + 0.053425 * rs_four) *
        detail::Log1p(16.081979498692535067 / q);
    constexpr double gamma = 0.031090690869654895034;
    const double beta_over_gamma = Parameters::beta / gamma;
    const detail::DualRhoSigma a = beta_over_gamma /
        detail::Expm1(-lda / gamma);
    const detail::DualRhoSigma t_squared =
        gradient / (rho_third * density * density) * cbrt_two *
        (cbrt_three * cbrt_three / cbrt_one_over_pi * cbrt_four) / 96.0;
    const detail::DualRhoSigma pbe_argument =
        t_squared + a * t_squared * t_squared;
    const detail::DualRhoSigma denominator = 1.0 + a * pbe_argument;
    // This exact identity avoids the LDA/H cancellation in the PBE-C
    // saturation limit: eps = gamma*log1p(-b / ((a+b) * denominator)).
    const detail::DualRhoSigma eps = gamma * detail::Log1p(
        -beta_over_gamma / ((a + beta_over_gamma) * denominator));
    const detail::DualRhoSigma energy = density * eps;
    const double vsigma = rho * Parameters::beta * t_squared.d_sigma *
        (1.0 + 2.0 * a.value * t_squared.value) /
        (denominator.value *
         (1.0 + (a.value + beta_over_gamma) * pbe_argument.value));
    return {eps.value, energy.d_rho, vsigma};
  }
};

using GgaPbeStandard = GgaPbe<PbeParameters>;

}  // namespace tides::grid::xc
