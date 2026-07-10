#include "grid/libxc_wrapper.hpp"
#include "grid/xc/xc_arena.hpp"
#include "grid/xc/xc_engine.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#ifndef TIDES_XC_RUNG0_REL
#error "TIDES_XC_RUNG0_REL must be configured from verification/tolerances.yaml"
#endif

namespace {

using tides::grid::kLibxc_MGGA_C_TPSS;
using tides::grid::kLibxc_MGGA_X_TPSS;
using tides::grid::kLibxc_MGGA_C_SCAN;
using tides::grid::kLibxc_MGGA_X_SCAN;
using tides::grid::kLibxc_MGGA_C_R2SCAN;
using tides::grid::kLibxc_MGGA_X_R2SCAN;
using tides::grid::kLibxc_MGGA_C_M06_2X;
using tides::grid::kLibxc_HYB_MGGA_X_M06_2X;
using tides::grid::LibxcFunctional;
using tides::grid::xc::Family;
using tides::grid::xc::Functional;
using tides::grid::xc::PrecisionPolicy;
using tides::grid::xc::XcArena;
using tides::grid::xc::XcEval;
using tides::grid::xc::XcGridIn;
using tides::grid::xc::XcGridOut;
using tides::grid::xc::XcSpec;

constexpr double kRung0RelativeTolerance = TIDES_XC_RUNG0_REL;

double RelativeError(double observed, double expected) {
  return std::abs(observed - expected) / std::max(std::abs(expected), 1.0e-16);
}

bool MatchesOracle(double observed, double expected) {
  return std::abs(observed - expected) <= kRung0RelativeTolerance ||
         RelativeError(observed, expected) <= kRung0RelativeTolerance;
}

constexpr double kGradientAbsoluteTolerance = 1.0e-12;

bool MatchesGradient(double observed, double expected) {
  return std::abs(observed - expected) <= kGradientAbsoluteTolerance ||
         RelativeError(observed, expected) <= kRung0RelativeTolerance;
}

int Fail(const char* message) {
  std::cerr << "xc_mgga_pol_device: " << message << '\n';
  return 1;
}

struct MggaTestCase {
  const char* name;
  int x_id;
  int c_id;
  Functional functional;
  bool xhyb;
};

std::vector<double> UniformTau(const std::vector<double>& rho) {
  std::vector<double> tau(rho.size());
  constexpr double c = (3.0 / 10.0) * std::pow(6.0 * M_PI * M_PI, 2.0 / 3.0);
  for (std::size_t i = 0; i < rho.size(); ++i) {
    tau[i] = c * std::pow(rho[i], 5.0 / 3.0);
  }
  return tau;
}

int RunMggaPolTest(const MggaTestCase& test) {
  const std::vector<double> rho_total = {0.1, 0.25, 1.0, 2.0, 10.0, 100.0};
  const std::vector<double> weights = {0.5, 0.75, 1.0, 1.25, 1.5, 2.0};
  const std::size_t np = rho_total.size();

  std::vector<double> rho_up(np), rho_down(np);
  std::vector<double> gradient_up_x(np), gradient_up_y(np), gradient_up_z(np);
  std::vector<double> gradient_down_x(np), gradient_down_y(np), gradient_down_z(np);
  std::vector<double> sigma_up(np), sigma_down(np), sigma_ud(np);
  std::vector<double> tau_up, tau_down;

  for (std::size_t point = 0; point < np; ++point) {
    rho_up[point] = 0.4 * rho_total[point];
    rho_down[point] = 0.6 * rho_total[point];

    const double gradient_norm = 0.05 * static_cast<double>(point + 1);
    const double scale = gradient_norm / std::sqrt(14.0);
    gradient_up_x[point] = 0.6 * scale;
    gradient_up_y[point] = 0.6 * 2.0 * scale;
    gradient_up_z[point] = 0.6 * 3.0 * scale;
    gradient_down_x[point] = 0.8 * scale;
    gradient_down_y[point] = 0.8 * 2.0 * scale;
    gradient_down_z[point] = 0.8 * 3.0 * scale;

    sigma_up[point] = gradient_up_x[point] * gradient_up_x[point] +
                      gradient_up_y[point] * gradient_up_y[point] +
                      gradient_up_z[point] * gradient_up_z[point];
    sigma_down[point] = gradient_down_x[point] * gradient_down_x[point] +
                        gradient_down_y[point] * gradient_down_y[point] +
                        gradient_down_z[point] * gradient_down_z[point];
    sigma_ud[point] = gradient_up_x[point] * gradient_down_x[point] +
                      gradient_up_y[point] * gradient_down_y[point] +
                      gradient_up_z[point] * gradient_down_z[point];

    tau_up = UniformTau(rho_up);
    tau_down = UniformTau(rho_down);
  }

  std::vector<double> rho_libxc(2 * np);
  std::vector<double> sigma_libxc(3 * np);
  std::vector<double> tau_libxc(2 * np);
  std::vector<double> lapl_libxc(2 * np, 0.0);
  for (std::size_t point = 0; point < np; ++point) {
    rho_libxc[2 * point] = rho_up[point];
    rho_libxc[2 * point + 1] = rho_down[point];
    sigma_libxc[3 * point] = sigma_up[point];
    sigma_libxc[3 * point + 1] = sigma_ud[point];
    sigma_libxc[3 * point + 2] = sigma_down[point];
    tau_libxc[2 * point] = tau_up[point];
    tau_libxc[2 * point + 1] = tau_down[point];
  }

  // libxc mGGA functional is a composite of separate x and c ids.
  // For the oracle, we evaluate x and c separately and combine.
  LibxcFunctional x_func;
  if (!x_func.Init(test.x_id, XC_POLARIZED)) {
    return Fail((std::string("failed to initialize the libxc ") + test.name + " x oracle").c_str());
  }
  LibxcFunctional c_func;
  if (!c_func.Init(test.c_id, XC_POLARIZED)) {
    return Fail((std::string("failed to initialize the libxc ") + test.name + " c oracle").c_str());
  }

  const auto x_oracle = x_func.EvalMGGA(rho_libxc, sigma_libxc, lapl_libxc, tau_libxc, np);
  const auto c_oracle = c_func.EvalMGGA(rho_libxc, sigma_libxc, lapl_libxc, tau_libxc, np);

  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) return Fail("cudaStreamCreate failed");
  XcArena arena;
  const auto reserve_status = arena.Reserve(np, 2, true, true, 1, stream);
  if (!reserve_status.ok()) {
    std::cerr << reserve_status.message() << '\n';
    return 1;
  }
  const std::size_t stride = arena.capacity();

  std::vector<double> grad(6 * stride, 0.0);
  std::vector<double> tau_arena(2 * stride, 0.0);
  std::vector<double> rho_arena(2 * stride, 0.0);
  for (std::size_t point = 0; point < np; ++point) {
    rho_arena[point] = rho_up[point];
    rho_arena[stride + point] = rho_down[point];
    grad[point] = gradient_up_x[point];
    grad[stride + point] = gradient_up_y[point];
    grad[2 * stride + point] = gradient_up_z[point];
    grad[3 * stride + point] = gradient_down_x[point];
    grad[4 * stride + point] = gradient_down_y[point];
    grad[5 * stride + point] = gradient_down_z[point];
    tau_arena[point] = tau_up[point];
    tau_arena[stride + point] = tau_down[point];
  }

  if (cudaMemcpyAsync(arena.rho(), rho_arena.data(), rho_arena.size() * sizeof(double),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(arena.grad(), grad.data(), grad.size() * sizeof(double),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(arena.tau(), tau_arena.data(), tau_arena.size() * sizeof(double),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(arena.weights(), weights.data(), np * sizeof(double),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    return Fail("mGGA polarized input upload failed");
  }

  XcSpec spec;
  spec.family = Family::kMgga;
  spec.nspin = 2;
  spec.terms = {{test.functional, 1.0}};
  spec.precision = PrecisionPolicy::kFloat64;
  XcGridIn input{arena.rho(), arena.grad(), arena.tau(), arena.weights(),
                 static_cast<std::int64_t>(np), static_cast<std::int64_t>(stride), 1,
                 arena.sys_offsets()};
  XcGridOut output{arena.wv_rho(), arena.wv_grad(), arena.wv_tau(),
                   arena.exc_per_system()};
  const auto eval_status = XcEval(spec, input, output, stream);
  if (!eval_status.ok()) {
    std::cerr << eval_status.message() << '\n';
    return 1;
  }

  std::vector<double> observed_wv_rho(2 * stride);
  std::vector<double> observed_wv_grad(6 * stride);
  std::vector<double> observed_wv_tau(2 * stride);
  double observed_energy = 0.0;
  if (cudaMemcpyAsync(observed_wv_rho.data(), arena.wv_rho(),
                      observed_wv_rho.size() * sizeof(double),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(observed_wv_grad.data(), arena.wv_grad(),
                      observed_wv_grad.size() * sizeof(double),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(observed_wv_tau.data(), arena.wv_tau(),
                      observed_wv_tau.size() * sizeof(double),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(&observed_energy, arena.exc_per_system(), sizeof(double),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    return Fail("mGGA polarized output download failed");
  }

  spec.deterministic = true;
  std::uint64_t deterministic_energy_bits = 0;
  double deterministic_energy = 0.0;
  for (int run = 0; run < 100; ++run) {
    const auto deterministic_status = XcEval(spec, input, output, stream);
    if (!deterministic_status.ok()) {
      std::cerr << deterministic_status.message() << '\n';
      return 1;
    }
    if (cudaMemcpyAsync(&deterministic_energy, arena.exc_per_system(), sizeof(double),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      return Fail("deterministic mGGA polarized energy download failed");
    }
    const std::uint64_t observed_bits = std::bit_cast<std::uint64_t>(deterministic_energy);
    if (run == 0) {
      deterministic_energy_bits = observed_bits;
    } else if (observed_bits != deterministic_energy_bits) {
      return Fail("deterministic mGGA polarized energy changed across repeated runs");
    }
  }

  const auto release_status = arena.Release(stream);
  if (!release_status.ok()) {
    std::cerr << release_status.message() << '\n';
    return 1;
  }
  if (cudaStreamDestroy(stream) != cudaSuccess) return Fail("cudaStreamDestroy failed");

  double expected_energy = 0.0;
  double max_wv_rho_rel = 0.0;
  double max_wv_grad_rel = 0.0;
  double max_wv_tau_rel = 0.0;
  bool all_wv_rho_match = true;
  bool all_wv_grad_match = true;
  bool all_wv_tau_match = true;
  for (std::size_t point = 0; point < np; ++point) {
    const double eps = x_oracle.eps_xc[point] + c_oracle.eps_xc[point];
    const double vrho_up = x_oracle.vrho[2 * point] + c_oracle.vrho[2 * point];
    const double vrho_down = x_oracle.vrho[2 * point + 1] + c_oracle.vrho[2 * point + 1];
    const double vsigma_up = x_oracle.vsigma[3 * point] + c_oracle.vsigma[3 * point];
    const double vsigma_ud = x_oracle.vsigma[3 * point + 1] + c_oracle.vsigma[3 * point + 1];
    const double vsigma_down = x_oracle.vsigma[3 * point + 2] + c_oracle.vsigma[3 * point + 2];
    const double vtau_up = x_oracle.vtau[2 * point] + c_oracle.vtau[2 * point];
    const double vtau_down = x_oracle.vtau[2 * point + 1] + c_oracle.vtau[2 * point + 1];
    expected_energy +=
        weights[point] * (rho_up[point] + rho_down[point]) * eps;

    const double expected_wv_rho_up = weights[point] * vrho_up;
    const double expected_wv_rho_down = weights[point] * vrho_down;
    max_wv_rho_rel = std::max(
        max_wv_rho_rel,
        std::max(RelativeError(observed_wv_rho[point], expected_wv_rho_up),
                 RelativeError(observed_wv_rho[stride + point], expected_wv_rho_down)));
    all_wv_rho_match = all_wv_rho_match &&
        MatchesOracle(observed_wv_rho[point], expected_wv_rho_up) &&
        MatchesOracle(observed_wv_rho[stride + point], expected_wv_rho_down);

    const double expected_wv_tau_up = weights[point] * vtau_up;
    const double expected_wv_tau_down = weights[point] * vtau_down;
    max_wv_tau_rel = std::max(
        max_wv_tau_rel,
        std::max(RelativeError(observed_wv_tau[point], expected_wv_tau_up),
                 RelativeError(observed_wv_tau[stride + point], expected_wv_tau_down)));
    all_wv_tau_match = all_wv_tau_match &&
        MatchesOracle(observed_wv_tau[point], expected_wv_tau_up) &&
        MatchesOracle(observed_wv_tau[stride + point], expected_wv_tau_down);

    const double wv_up[3] = {
        weights[point] * (2.0 * vsigma_up * gradient_up_x[point] +
                          vsigma_ud * gradient_down_x[point]),
        weights[point] * (2.0 * vsigma_up * gradient_up_y[point] +
                          vsigma_ud * gradient_down_y[point]),
        weights[point] * (2.0 * vsigma_up * gradient_up_z[point] +
                          vsigma_ud * gradient_down_z[point])};
    const double wv_down[3] = {
        weights[point] * (2.0 * vsigma_down * gradient_down_x[point] +
                          vsigma_ud * gradient_up_x[point]),
        weights[point] * (2.0 * vsigma_down * gradient_down_y[point] +
                          vsigma_ud * gradient_up_y[point]),
        weights[point] * (2.0 * vsigma_down * gradient_down_z[point] +
                          vsigma_ud * gradient_up_z[point])};
    for (std::size_t component = 0; component < 3; ++component) {
      const double observed_up = observed_wv_grad[component * stride + point];
      const double observed_down = observed_wv_grad[(component + 3) * stride + point];
      const double rel_up = RelativeError(observed_up, wv_up[component]);
      const double rel_down = RelativeError(observed_down, wv_down[component]);
      max_wv_grad_rel = std::max(max_wv_grad_rel, std::max(rel_up, rel_down));
      all_wv_grad_match = all_wv_grad_match &&
          MatchesGradient(observed_up, wv_up[component]) &&
          MatchesGradient(observed_down, wv_down[component]);
      if (rel_up > 1.0e-12 || rel_down > 1.0e-12) {
        std::cout << "  " << test.name << " point=" << point << " comp=" << component
                  << " rho_up=" << rho_up[point] << " rho_down=" << rho_down[point]
                  << " obs_up=" << observed_up << " exp_up=" << wv_up[component]
                  << " rel_up=" << rel_up << " obs_down=" << observed_down
                  << " exp_down=" << wv_down[component] << " rel_down=" << rel_down
                  << '\n';
      }
    }
  }
  const double energy_rel = RelativeError(observed_energy, expected_energy);
  const double deterministic_energy_rel =
      RelativeError(deterministic_energy, expected_energy);
  std::cout << "xc_mgga_pol_device_" << test.name << ": points=" << np
            << " stride=" << stride << " max_wv_rho_rel=" << max_wv_rho_rel
            << " max_wv_grad_rel=" << max_wv_grad_rel
            << " max_wv_tau_rel=" << max_wv_tau_rel
            << " energy_rel=" << energy_rel
            << " deterministic_energy_rel=" << deterministic_energy_rel
            << " deterministic_runs=100"
            << " tolerance=" << kRung0RelativeTolerance << '\n';
  if (!all_wv_rho_match || !all_wv_grad_match || !all_wv_tau_match ||
      energy_rel > kRung0RelativeTolerance ||
      deterministic_energy_rel > kRung0RelativeTolerance) {
    return Fail("device polarized mGGA weighted outputs differ from the libxc oracle");
  }
  return 0;
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: CUDA runtime not available\n";
    return 77;
  }

  if (RunMggaPolTest({"TPSS", kLibxc_MGGA_X_TPSS, kLibxc_MGGA_C_TPSS, Functional::kTpss, false}) != 0) {
    return 1;
  }
  if (RunMggaPolTest({"SCAN", kLibxc_MGGA_X_SCAN, kLibxc_MGGA_C_SCAN, Functional::kScan, false}) != 0) {
    return 1;
  }
  if (RunMggaPolTest({"R2SCAN", kLibxc_MGGA_X_R2SCAN, kLibxc_MGGA_C_R2SCAN, Functional::kR2scan, false}) != 0) {
    return 1;
  }
  if (RunMggaPolTest({"M06_2X", kLibxc_HYB_MGGA_X_M06_2X, kLibxc_MGGA_C_M06_2X, Functional::kM06_2x, true}) != 0) {
    return 1;
  }
  return 0;
}
