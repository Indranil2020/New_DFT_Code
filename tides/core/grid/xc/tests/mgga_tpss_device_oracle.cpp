#include "grid/libxc_wrapper.hpp"
#include "grid/xc/xc_arena.hpp"
#include "grid/xc/xc_engine.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#ifndef TIDES_XC_RUNG0_REL
#error "TIDES_XC_RUNG0_REL must be configured from verification/tolerances.yaml"
#endif

namespace {

using tides::grid::LibxcFunctional;
using tides::grid::kLibxc_MGGA_C_TPSS;
using tides::grid::kLibxc_MGGA_X_TPSS;
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

int Fail(const char* message) {
  std::cerr << "xc_mgga_tpss_device: " << message << '\n';
  return 1;
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: CUDA runtime not available\n";
    return 77;
  }

  const std::vector<double> rho = {0.1, 0.25, 1.0, 2.0, 10.0, 100.0};
  const std::vector<double> weights = {0.5, 0.75, 1.0, 1.25, 1.5, 2.0};
  std::vector<double> sigma(rho.size());
  std::vector<double> lapl(rho.size(), 0.0);
  std::vector<double> tau(rho.size());
  std::vector<double> gradient_x(rho.size());
  std::vector<double> gradient_y(rho.size());
  std::vector<double> gradient_z(rho.size());
  for (std::size_t point = 0; point < rho.size(); ++point) {
    const double gradient_norm = 0.05 * static_cast<double>(point + 1);
    const double scale = gradient_norm / std::sqrt(14.0);
    gradient_x[point] = scale;
    gradient_y[point] = 2.0 * scale;
    gradient_z[point] = 3.0 * scale;
    sigma[point] = gradient_norm * gradient_norm;
    tau[point] = 0.1 * std::pow(rho[point], 5.0 / 3.0);
  }

  LibxcFunctional exchange;
  LibxcFunctional correlation;
  if (!exchange.Init(kLibxc_MGGA_X_TPSS, XC_UNPOLARIZED) ||
      !correlation.Init(kLibxc_MGGA_C_TPSS, XC_UNPOLARIZED)) {
    return Fail("failed to initialize the libxc TPSS oracle");
  }
  const auto libxc_x = exchange.EvalMGGA(rho, sigma, lapl, tau, rho.size());
  const auto libxc_c = correlation.EvalMGGA(rho, sigma, lapl, tau, rho.size());

  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) return Fail("cudaStreamCreate failed");
  XcArena arena;
  const auto reserve_status = arena.Reserve(rho.size(), 1, true, true, 1, stream);
  if (!reserve_status.ok()) {
    std::cerr << reserve_status.message() << '\n';
    return 1;
  }
  const std::size_t stride = arena.capacity();

  std::vector<double> grad(3 * stride, 0.0);
  for (std::size_t point = 0; point < rho.size(); ++point) {
    grad[point] = gradient_x[point];
    grad[stride + point] = gradient_y[point];
    grad[2 * stride + point] = gradient_z[point];
  }
  const std::size_t scalar_bytes = rho.size() * sizeof(double);
  if (cudaMemcpyAsync(arena.rho(), rho.data(), scalar_bytes, cudaMemcpyHostToDevice,
                      stream) != cudaSuccess ||
      cudaMemcpyAsync(arena.weights(), weights.data(), scalar_bytes,
                      cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(arena.grad(), grad.data(), grad.size() * sizeof(double),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(arena.tau(), tau.data(), scalar_bytes, cudaMemcpyHostToDevice,
                      stream) != cudaSuccess) {
    return Fail("TPSS input upload failed");
  }

  XcSpec spec;
  spec.family = Family::kMgga;
  spec.nspin = 1;
  spec.terms = {{Functional::kTpss, 1.0}};
  spec.precision = PrecisionPolicy::kFloat64;
  XcGridIn input{arena.rho(), arena.grad(), arena.tau(), arena.weights(),
                 static_cast<std::int64_t>(rho.size()),
                 static_cast<std::int64_t>(stride), 1, arena.sys_offsets()};
  XcGridOut output{arena.wv_rho(), arena.wv_grad(), arena.wv_tau(),
                   arena.exc_per_system()};
  const auto eval_status = XcEval(spec, input, output, stream);
  if (!eval_status.ok()) {
    std::cerr << eval_status.message() << '\n';
    return 1;
  }

  std::vector<double> observed_wv_rho(rho.size());
  std::vector<double> observed_wv_grad(3 * stride);
  std::vector<double> observed_wv_tau(rho.size());
  double observed_energy = 0.0;
  if (cudaMemcpyAsync(observed_wv_rho.data(), arena.wv_rho(), scalar_bytes,
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(observed_wv_grad.data(), arena.wv_grad(),
                      observed_wv_grad.size() * sizeof(double),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(observed_wv_tau.data(), arena.wv_tau(), scalar_bytes,
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(&observed_energy, arena.exc_per_system(), sizeof(double),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    return Fail("TPSS output download failed");
  }

  spec.deterministic = true;
  double deterministic_energy = 0.0;
  std::uint64_t deterministic_energy_bits = 0;
  for (int run = 0; run < 100; ++run) {
    const auto deterministic_status = XcEval(spec, input, output, stream);
    if (!deterministic_status.ok()) {
      std::cerr << deterministic_status.message() << '\n';
      return 1;
    }
    if (cudaMemcpyAsync(&deterministic_energy, arena.exc_per_system(), sizeof(double),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      return Fail("deterministic TPSS energy download failed");
    }
    const std::uint64_t observed_bits = std::bit_cast<std::uint64_t>(deterministic_energy);
    if (run == 0) {
      deterministic_energy_bits = observed_bits;
    } else if (observed_bits != deterministic_energy_bits) {
      return Fail("deterministic TPSS energy changed across repeated runs");
    }
  }

  const auto release_status = arena.Release(stream);
  if (!release_status.ok()) {
    std::cerr << release_status.message() << '\n';
    return 1;
  }
  if (cudaStreamDestroy(stream) != cudaSuccess) return Fail("cudaStreamDestroy failed");

  double expected_energy = 0.0;
  double max_wv_rho_relative_error = 0.0;
  double max_wv_grad_relative_error = 0.0;
  double max_wv_tau_relative_error = 0.0;
  bool all_wv_rho_match = true;
  bool all_wv_grad_match = true;
  bool all_wv_tau_match = true;
  for (std::size_t point = 0; point < rho.size(); ++point) {
    const double eps = libxc_x.eps_xc[point] + libxc_c.eps_xc[point];
    const double vrho = libxc_x.vrho[point] + libxc_c.vrho[point];
    const double vsigma = libxc_x.vsigma[point] + libxc_c.vsigma[point];
    const double vtau = libxc_x.vtau[point] + libxc_c.vtau[point];
    expected_energy += weights[point] * rho[point] * eps;
    const double expected_wv_rho = weights[point] * vrho;
    const double expected_wv_tau = weights[point] * vtau;
    max_wv_rho_relative_error = std::max(
        max_wv_rho_relative_error,
        RelativeError(observed_wv_rho[point], expected_wv_rho));
    all_wv_rho_match = all_wv_rho_match &&
        MatchesOracle(observed_wv_rho[point], expected_wv_rho);
    max_wv_tau_relative_error = std::max(
        max_wv_tau_relative_error,
        RelativeError(observed_wv_tau[point], expected_wv_tau));
    all_wv_tau_match = all_wv_tau_match &&
        MatchesOracle(observed_wv_tau[point], expected_wv_tau);
    const double weighted_vsigma = 2.0 * weights[point] * vsigma;
    const double expected_gradient[] = {
        weighted_vsigma * gradient_x[point],
        weighted_vsigma * gradient_y[point],
        weighted_vsigma * gradient_z[point],
    };
    for (std::size_t component = 0; component < 3; ++component) {
      const double observed_grad = observed_wv_grad[component * stride + point];
      const double expected_grad = expected_gradient[component];
      max_wv_grad_relative_error = std::max(
          max_wv_grad_relative_error,
          RelativeError(observed_grad, expected_grad));
      all_wv_grad_match = all_wv_grad_match &&
          MatchesOracle(observed_grad, expected_grad);
    }
  }
  const double energy_relative_error = RelativeError(observed_energy, expected_energy);
  const double deterministic_energy_relative_error =
      RelativeError(deterministic_energy, expected_energy);
  std::cout << "xc_mgga_tpss_device: points=" << rho.size()
            << " stride=" << stride
            << " max_wv_rho_rel=" << max_wv_rho_relative_error
            << " max_wv_grad_rel=" << max_wv_grad_relative_error
            << " max_wv_tau_rel=" << max_wv_tau_relative_error
            << " energy_rel=" << energy_relative_error
            << " deterministic_energy_rel=" << deterministic_energy_relative_error
            << " deterministic_runs=100"
            << " tolerance=" << kRung0RelativeTolerance << '\n';
  if (!all_wv_rho_match || !all_wv_grad_match || !all_wv_tau_match ||
      energy_relative_error > kRung0RelativeTolerance ||
      deterministic_energy_relative_error > kRung0RelativeTolerance) {
    return Fail("device TPSS weighted outputs differ from the libxc oracle");
  }
  return 0;
}
