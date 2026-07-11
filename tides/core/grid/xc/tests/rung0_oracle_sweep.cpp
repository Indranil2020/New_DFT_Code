#include "grid/libxc_wrapper.hpp"
#include "grid/xc/functionals/common.cuh"
#include "grid/xc/functionals/lda_pw92.cuh"
#include "grid/xc/functionals/lda_slater.cuh"
#include "grid/xc/xc_engine.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#ifndef TIDES_XC_RUNG0_REL
#error "TIDES_XC_RUNG0_REL must be configured from verification/tolerances.yaml"
#endif

namespace {

using tides::grid::LibxcFunctional;
using tides::grid::kLibxc_LDA_C_PW;
using tides::grid::kLibxc_LDA_X;
using tides::grid::xc::LdaPw92;
using tides::grid::xc::LdaSlater;

constexpr double kRung0RelativeTolerance = TIDES_XC_RUNG0_REL;

double RelativeError(double observed, double expected) {
  const double denominator = std::max(std::abs(expected), 1.0e-16);
  return std::abs(observed - expected) / denominator;
}

int Fail(const char* message) {
  std::cerr << "xc_rung0_oracle: " << message << '\n';
  return 1;
}

}  // namespace

int main() {
  std::vector<double> rho = {
      0.5 * tides::grid::xc::detail::kLdaPw92DensityThreshold,
      tides::grid::xc::detail::kLdaPw92DensityThreshold,
      2.0 * tides::grid::xc::detail::kLdaPw92DensityThreshold,
  };
  for (int exponent = -14; exponent <= 4; ++exponent) {
    for (int mantissa = 1; mantissa <= 3; ++mantissa) {
      rho.push_back(static_cast<double>(mantissa) * std::pow(10.0, exponent));
    }
  }

  LibxcFunctional exchange;
  LibxcFunctional correlation;
  if (!exchange.Init(kLibxc_LDA_X, XC_UNPOLARIZED) ||
      !correlation.Init(kLibxc_LDA_C_PW, XC_UNPOLARIZED)) {
    return Fail("failed to initialize the libxc LDA-PW92 oracle");
  }
  const auto libxc_x = exchange.EvalLDA(rho, rho.size());
  const auto libxc_c = correlation.EvalLDA(rho, rho.size());

  double max_eps_relative_error = 0.0;
  double max_vrho_relative_error = 0.0;
  std::size_t max_eps_point = 0;
  std::size_t max_vrho_point = 0;
  double max_eps_observed = 0.0;
  double max_eps_expected = 0.0;
  double max_vrho_observed = 0.0;
  double max_vrho_expected = 0.0;
  for (std::size_t point = 0; point < rho.size(); ++point) {
    const double eps = LdaSlater::Eps(rho[point]) +
                       LdaPw92::EpsCorrelation(rho[point]);
    const double vrho = LdaSlater::V(rho[point]) +
                        LdaPw92::VCorrelation(rho[point]);
    const double expected_eps = libxc_x.eps_xc[point] + libxc_c.eps_xc[point];
    const double expected_vrho = libxc_x.vrho[point] + libxc_c.vrho[point];
    const double eps_error = RelativeError(eps, expected_eps);
    const double vrho_error = RelativeError(vrho, expected_vrho);
    if (eps_error > max_eps_relative_error) {
      max_eps_relative_error = eps_error;
      max_eps_point = point;
      max_eps_observed = eps;
      max_eps_expected = expected_eps;
    }
    if (vrho_error > max_vrho_relative_error) {
      max_vrho_relative_error = vrho_error;
      max_vrho_point = point;
      max_vrho_observed = vrho;
      max_vrho_expected = expected_vrho;
    }
  }

  std::cout << "xc_rung0_host: points=" << rho.size()
            << " x_threshold=" << exchange.DensityThreshold()
            << " c_threshold=" << correlation.DensityThreshold()
            << " max_eps_rel=" << max_eps_relative_error
            << " max_vrho_rel=" << max_vrho_relative_error
            << " eps_rho=" << rho[max_eps_point]
            << " eps_observed=" << max_eps_observed
            << " eps_expected=" << max_eps_expected
            << " eps_x=" << libxc_x.eps_xc[max_eps_point]
            << " eps_c=" << libxc_c.eps_xc[max_eps_point]
            << " vrho_rho=" << rho[max_vrho_point]
            << " vrho_observed=" << max_vrho_observed
            << " vrho_expected=" << max_vrho_expected
            << " vrho_x=" << libxc_x.vrho[max_vrho_point]
            << " vrho_c=" << libxc_c.vrho[max_vrho_point]
            << " tolerance=" << kRung0RelativeTolerance << '\n';
  if (max_eps_relative_error > kRung0RelativeTolerance ||
      max_vrho_relative_error > kRung0RelativeTolerance) {
    return Fail("shared LDA-PW92 arithmetic differs from the libxc oracle");
  }
  return 0;
}
