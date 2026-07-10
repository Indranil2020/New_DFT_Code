#include "grid/libxc_wrapper.hpp"
#include "grid/xc/functionals/gga_pbe.cuh"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#ifndef TIDES_XC_RUNG0_REL
#error "TIDES_XC_RUNG0_REL must be configured from verification/tolerances.yaml"
#endif

namespace {

using tides::grid::LibxcFunctional;
using tides::grid::kLibxc_GGA_C_PBE;
using tides::grid::kLibxc_GGA_X_PBE;
using tides::grid::xc::GgaPbeStandard;

constexpr double kRung0RelativeTolerance = TIDES_XC_RUNG0_REL;

double RelativeError(double observed, double expected) {
  return std::abs(observed - expected) / std::max(std::abs(expected), 1.0e-16);
}

bool MatchesOracle(double observed, double expected) {
  return std::abs(observed - expected) <= kRung0RelativeTolerance ||
         RelativeError(observed, expected) <= kRung0RelativeTolerance;
}

bool AssembledVsigmaMatchesOracle(double observed_exchange,
                                  double observed_correlation,
                                  double expected_exchange,
                                  double expected_correlation) {
  // PBE X and C v_sigma can nearly cancel.  Assess the assembled result on
  // the component scale while retaining strict component checks below.
  const double scale = std::max(
      {std::abs(expected_exchange + expected_correlation),
       std::abs(expected_exchange) + std::abs(expected_correlation), 1.0e-16});
  return std::abs((observed_exchange + observed_correlation) -
                  (expected_exchange + expected_correlation)) <=
      kRung0RelativeTolerance * scale;
}

struct PbeCorrelationReference {
  long double eps = 0.0L;
  long double vrho = 0.0L;
  long double vsigma = 0.0L;
};

struct HighDual {
  long double value = 0.0L;
  long double d_rho = 0.0L;

  HighDual(long double value_in = 0.0L, long double derivative_in = 0.0L)
      : value(value_in), d_rho(derivative_in) {}
};

HighDual operator+(HighDual lhs, HighDual rhs) {
  return {lhs.value + rhs.value, lhs.d_rho + rhs.d_rho};
}

HighDual operator-(HighDual value) {
  return {-value.value, -value.d_rho};
}

HighDual operator*(HighDual lhs, HighDual rhs) {
  return {lhs.value * rhs.value,
          lhs.d_rho * rhs.value + lhs.value * rhs.d_rho};
}

HighDual operator/(HighDual lhs, HighDual rhs) {
  const long double inverse = 1.0L / rhs.value;
  const long double quotient = lhs.value * inverse;
  return {quotient, (lhs.d_rho - quotient * rhs.d_rho) * inverse};
}

HighDual Cbrt(HighDual value) {
  const long double root = std::cbrt(value.value);
  return {root, value.d_rho / (3.0L * root * root)};
}

HighDual Sqrt(HighDual value) {
  const long double root = std::sqrt(value.value);
  return {root, value.d_rho / (2.0L * root)};
}

HighDual Log1p(HighDual value) {
  return {std::log1p(value.value), value.d_rho / (1.0L + value.value)};
}

HighDual Expm1(HighDual value) {
  const long double exponential = std::exp(value.value);
  return {std::expm1(value.value), exponential * value.d_rho};
}

// Stable PBE-C reference evaluated at extended precision from the same
// double inputs supplied to libxc and the device functor.  This specifically
// covers the PBE-C saturation regime where libxc's generated v_sigma worker
// subtracts nearly equal large terms.
PbeCorrelationReference StablePbeCorrelationReference(double rho,
                                                       double sigma) {
  if (rho < 1.0e-12) return {};
  using Real = long double;
  constexpr Real kPi = 3.141592653589793238462643383279502884L;
  constexpr Real kBeta = 0.06672455060314922L;
  constexpr Real kGamma = 0.031090690869654895034L;
  const HighDual density(static_cast<Real>(rho), 1.0L);
  const Real sigma_eff = std::max(static_cast<Real>(sigma), 1.0e-32L);
  const Real cbrt_three = std::cbrt(3.0L);
  const Real cbrt_one_over_pi = std::cbrt(1.0L / kPi);
  const Real cbrt_four = std::cbrt(4.0L);
  const Real cbrt_two = std::cbrt(2.0L);
  const HighDual rho_third = Cbrt(density);
  const HighDual rs_four = cbrt_three * cbrt_one_over_pi * cbrt_four *
      cbrt_four / rho_third;
  const HighDual q_lda = 3.79785L * Sqrt(rs_four) + 0.8969L * rs_four +
      0.204775L * rs_four * Sqrt(rs_four) +
      0.123235L * cbrt_three * cbrt_three * cbrt_one_over_pi *
      cbrt_one_over_pi * cbrt_four / (rho_third * rho_third);
  const HighDual lda = -0.0621814L * (1.0L + 0.053425L * rs_four) *
      Log1p(16.081979498692535067L / q_lda);
  const Real beta_over_gamma = kBeta / kGamma;
  const HighDual a = beta_over_gamma / Expm1(-lda / kGamma);
  const Real dx_dsigma = cbrt_two * cbrt_three * cbrt_three /
      cbrt_one_over_pi * cbrt_four /
      (96.0L * rho_third.value * density.value * density.value);
  const HighDual x = sigma_eff * cbrt_two * cbrt_three * cbrt_three /
      cbrt_one_over_pi * cbrt_four /
      (96.0L * rho_third * density * density);
  const HighDual pbe_argument = x + a * x * x;
  const HighDual denominator = 1.0L + a * pbe_argument;
  const HighDual eps = kGamma * Log1p(
      -beta_over_gamma / ((a + beta_over_gamma) * denominator));
  const HighDual energy = density * eps;
  const Real vsigma = density.value * kBeta * dx_dsigma *
      (1.0L + 2.0L * a.value * x.value) /
      (denominator.value *
       (1.0L + (a.value + beta_over_gamma) * pbe_argument.value));
  return {eps.value, energy.d_rho, vsigma};
}

bool MatchesExtendedPrecision(double observed, long double expected) {
  const long double error = std::abs(static_cast<long double>(observed) - expected);
  const long double scale = std::max(
      std::abs(expected), std::numeric_limits<long double>::min());
  return error / scale <= static_cast<long double>(kRung0RelativeTolerance);
}

double ExtendedRelativeError(double observed, long double expected) {
  const long double scale = std::max(
      std::abs(expected), std::numeric_limits<long double>::min());
  return static_cast<double>(
      std::abs(static_cast<long double>(observed) - expected) / scale);
}

int Fail(const char* message) {
  std::cerr << "pbe_functor_oracle: " << message << '\n';
  return 1;
}

}  // namespace

int main() {
  std::vector<double> rho;
  std::vector<double> sigma;
  constexpr std::size_t kDensityPoints = 64;
  constexpr std::size_t kReducedGradientPoints = 48;
  constexpr double kPi = 3.141592653589793238462643383279502884;
  // The design's logarithmic s interval includes zero explicitly, then uses
  // 47 log-spaced positive samples through 1e3.
  std::vector<double> reduced_gradients = {0.0};
  reduced_gradients.reserve(kReducedGradientPoints);
  for (std::size_t i = 0; i + 1 < kReducedGradientPoints; ++i) {
    const double fraction = static_cast<double>(i) /
        static_cast<double>(kReducedGradientPoints - 2);
    reduced_gradients.push_back(std::exp(
        std::log(1.0e-12) + fraction * std::log(1.0e15)));
  }
  rho.reserve(kDensityPoints * kReducedGradientPoints + 9);
  sigma.reserve(kDensityPoints * kReducedGradientPoints + 9);
  for (std::size_t i = 0; i < kDensityPoints; ++i) {
    const double fraction = static_cast<double>(i) /
        static_cast<double>(kDensityPoints - 1);
    const double density = std::exp(
        std::log(1.0e-15) + fraction * std::log(1.0e19));
    const double density_four_thirds =
        density * std::cbrt(density);
    const double sigma_prefactor =
        2.0 * std::cbrt(3.0 * kPi * kPi) * density_four_thirds;
    for (double reduced_gradient : reduced_gradients) {
      rho.push_back(density);
      sigma.push_back(sigma_prefactor * sigma_prefactor *
                      reduced_gradient * reduced_gradient);
    }
  }
  // libxc's density and sigma clamps use mixed strict/non-strict boundaries;
  // preserve those semantics instead of relying on incidental lattice hits.
  const auto append_case = [&rho, &sigma](double density, double gradient_sq) {
    rho.push_back(density);
    sigma.push_back(gradient_sq);
  };
  append_case(0.0, 0.0);
  append_case(2.0e-15, 0.0);  // GGA-X screens rho/2 <= 1e-15.
  append_case(std::nextafter(2.0e-15,
                             std::numeric_limits<double>::infinity()), 0.0);
  append_case(std::nextafter(1.0e-12, 0.0), 0.0);  // GGA-C screens rho < 1e-12.
  append_case(1.0e-12, 0.0);
  append_case(1.0, 0.0);
  append_case(1.0, 1.0e-40);  // GGA-X sigma clamp squared.
  append_case(1.0, std::nextafter(1.0e-40,
                                  std::numeric_limits<double>::infinity()));
  append_case(1.0, 1.0e-32);  // GGA-C sigma clamp squared.

  LibxcFunctional exchange;
  LibxcFunctional correlation;
  if (!exchange.Init(kLibxc_GGA_X_PBE, XC_UNPOLARIZED) ||
      !correlation.Init(kLibxc_GGA_C_PBE, XC_UNPOLARIZED)) {
    return Fail("failed to initialize the libxc PBE oracle");
  }
  const auto libxc_x = exchange.EvalGGA(rho, sigma, rho.size());
  const auto libxc_c = correlation.EvalGGA(rho, sigma, rho.size());

  double max_eps_relative_error = 0.0;
  double max_vrho_relative_error = 0.0;
  double max_vsigma_relative_error = 0.0;
  double max_exchange_eps_relative_error = 0.0;
  double max_correlation_eps_relative_error = 0.0;
  double max_exchange_vsigma_relative_error = 0.0;
  double max_correlation_vsigma_relative_error = 0.0;
  double max_correlation_reference_eps_relative_error = 0.0;
  double max_correlation_reference_vrho_relative_error = 0.0;
  double max_correlation_reference_vsigma_relative_error = 0.0;
  std::size_t max_correlation_eps_point = 0;
  std::size_t max_correlation_vsigma_point = 0;
  std::size_t libxc_precision_divergence_count = 0;
  for (std::size_t point = 0; point < rho.size(); ++point) {
    const auto observed = GgaPbeStandard::Eval(rho[point], sigma[point]);
    const auto observed_exchange =
        GgaPbeStandard::EvalExchange(rho[point], sigma[point]);
    const auto observed_correlation =
        GgaPbeStandard::EvalCorrelation(rho[point], sigma[point]);
    const bool correlation_active = rho[point] >= 1.0e-12;
    const auto correlation_reference =
        StablePbeCorrelationReference(rho[point], sigma[point]);
    if (correlation_active) {
      max_correlation_reference_eps_relative_error = std::max(
          max_correlation_reference_eps_relative_error,
          ExtendedRelativeError(observed_correlation.eps,
                                correlation_reference.eps));
      max_correlation_reference_vrho_relative_error = std::max(
          max_correlation_reference_vrho_relative_error,
          ExtendedRelativeError(observed_correlation.vrho,
                                correlation_reference.vrho));
      max_correlation_reference_vsigma_relative_error = std::max(
          max_correlation_reference_vsigma_relative_error,
          ExtendedRelativeError(observed_correlation.vsigma,
                                correlation_reference.vsigma));
    }
    const double expected_eps = libxc_x.eps_xc[point] + libxc_c.eps_xc[point];
    const double expected_vrho = libxc_x.vrho[point] + libxc_c.vrho[point];
    const double expected_vsigma = libxc_x.vsigma[point] + libxc_c.vsigma[point];
    max_eps_relative_error = std::max(max_eps_relative_error,
                                      RelativeError(observed.eps, expected_eps));
    max_vrho_relative_error = std::max(max_vrho_relative_error,
                                       RelativeError(observed.vrho, expected_vrho));
    max_vsigma_relative_error = std::max(max_vsigma_relative_error,
                                         RelativeError(observed.vsigma, expected_vsigma));
    max_exchange_eps_relative_error = std::max(
        max_exchange_eps_relative_error,
        RelativeError(observed_exchange.eps, libxc_x.eps_xc[point]));
    const double correlation_eps_error =
        RelativeError(observed_correlation.eps, libxc_c.eps_xc[point]);
    if (correlation_eps_error > max_correlation_eps_relative_error) {
      max_correlation_eps_relative_error = correlation_eps_error;
      max_correlation_eps_point = point;
    }
    max_exchange_vsigma_relative_error = std::max(
        max_exchange_vsigma_relative_error,
        RelativeError(observed_exchange.vsigma, libxc_x.vsigma[point]));
    const double correlation_vsigma_error =
        RelativeError(observed_correlation.vsigma, libxc_c.vsigma[point]);
    if (correlation_vsigma_error > max_correlation_vsigma_relative_error) {
      max_correlation_vsigma_relative_error = correlation_vsigma_error;
      max_correlation_vsigma_point = point;
    }
    const bool correlation_matches_reference = !correlation_active ||
        (MatchesExtendedPrecision(observed_correlation.eps,
                                  correlation_reference.eps) &&
         MatchesExtendedPrecision(observed_correlation.vrho,
                                  correlation_reference.vrho) &&
         MatchesExtendedPrecision(observed_correlation.vsigma,
                                  correlation_reference.vsigma));
    const bool libxc_correlation_matches_reference = !correlation_active ||
        (MatchesExtendedPrecision(libxc_c.eps_xc[point],
                                  correlation_reference.eps) &&
         MatchesExtendedPrecision(libxc_c.vrho[point],
                                  correlation_reference.vrho) &&
         MatchesExtendedPrecision(libxc_c.vsigma[point],
                                  correlation_reference.vsigma));
    if (correlation_active && !libxc_correlation_matches_reference) {
      ++libxc_precision_divergence_count;
    }
    const bool raw_libxc_matches = !libxc_correlation_matches_reference ||
        (MatchesOracle(observed.eps, expected_eps) &&
         MatchesOracle(observed.vrho, expected_vrho) &&
         AssembledVsigmaMatchesOracle(
             observed_exchange.vsigma, observed_correlation.vsigma,
             libxc_x.vsigma[point], libxc_c.vsigma[point]) &&
         MatchesOracle(observed_correlation.eps, libxc_c.eps_xc[point]) &&
         MatchesOracle(observed_correlation.vrho, libxc_c.vrho[point]) &&
         MatchesOracle(observed_correlation.vsigma, libxc_c.vsigma[point]));
    if (!correlation_matches_reference || !raw_libxc_matches ||
        !MatchesOracle(observed_exchange.eps, libxc_x.eps_xc[point]) ||
        !MatchesOracle(observed_exchange.vrho, libxc_x.vrho[point]) ||
        !MatchesOracle(observed_exchange.vsigma, libxc_x.vsigma[point])) {
      std::cerr << std::setprecision(17)
                << "rho=" << rho[point] << " sigma=" << sigma[point]
                << " observed={" << observed.eps << ',' << observed.vrho << ','
                << observed.vsigma << "} expected={" << expected_eps << ','
                << expected_vrho << ',' << expected_vsigma << "}"
                << " c_observed={" << observed_correlation.eps << ','
                << observed_correlation.vrho << ',' << observed_correlation.vsigma
                << "} c_expected={" << libxc_c.eps_xc[point] << ','
                << libxc_c.vrho[point] << ',' << libxc_c.vsigma[point] << "}"
                << " c_reference={" << static_cast<double>(correlation_reference.eps)
                << ',' << static_cast<double>(correlation_reference.vrho)
                << ',' << static_cast<double>(correlation_reference.vsigma) << "}"
                << " libxc_c_precision_ok=" << libxc_correlation_matches_reference
                << " c_reference_ok=" << correlation_matches_reference << '\n';
      return Fail("PBE component or combined derivative differs from libxc");
    }
  }

  std::cout << "pbe_functor_oracle: points=" << rho.size()
            << " max_raw_eps_rel=" << max_eps_relative_error
            << " max_raw_vrho_rel=" << max_vrho_relative_error
            << " max_raw_vsigma_rel=" << max_vsigma_relative_error
            << " max_x_eps_rel=" << max_exchange_eps_relative_error
            << " max_raw_c_eps_rel=" << max_correlation_eps_relative_error
            << " max_x_vsigma_rel=" << max_exchange_vsigma_relative_error
            << " max_raw_c_vsigma_rel=" << max_correlation_vsigma_relative_error
            << " max_c_reference_eps_rel="
            << max_correlation_reference_eps_relative_error
            << " max_c_reference_vrho_rel="
            << max_correlation_reference_vrho_relative_error
            << " max_c_reference_vsigma_rel="
            << max_correlation_reference_vsigma_relative_error
            << " libxc_c_precision_divergences="
            << libxc_precision_divergence_count
            << " c_eps_rho=" << rho[max_correlation_eps_point]
            << " c_eps_sigma=" << sigma[max_correlation_eps_point]
            << " c_eps_observed=" << GgaPbeStandard::EvalCorrelation(
                   rho[max_correlation_eps_point], sigma[max_correlation_eps_point]).eps
            << " c_eps_expected=" << libxc_c.eps_xc[max_correlation_eps_point]
            << " c_vsigma_rho=" << rho[max_correlation_vsigma_point]
            << " c_vsigma_sigma=" << sigma[max_correlation_vsigma_point]
            << " c_vsigma_observed=" << GgaPbeStandard::EvalCorrelation(
                   rho[max_correlation_vsigma_point], sigma[max_correlation_vsigma_point]).vsigma
            << " c_vsigma_expected=" << libxc_c.vsigma[max_correlation_vsigma_point]
            << " reference_tolerance=" << kRung0RelativeTolerance << '\n';
  return 0;
}
