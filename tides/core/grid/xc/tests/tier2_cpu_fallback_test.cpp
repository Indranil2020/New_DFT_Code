// T-X4.3: Test the Tier-2 CPU libxc fallback for mixed/exotic XC specs.
//
// This triggers the fallback path by requesting a non-standard coefficient on
// an otherwise Tier-0 functional (PBE with coefficient 0.5).  XcEval must not
// match the fast Tier-0 path, so it routes to tier2::LaunchCpuFallback.
// The CPU oracle is the same libxc evaluation scaled by 0.5.

#include "grid/libxc_wrapper.hpp"
#include "grid/xc/xc_arena.hpp"
#include "grid/xc/xc_engine.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#ifndef TIDES_XC_RUNG0_REL
#error "TIDES_XC_RUNG0_REL must be configured from verification/tolerances.yaml"
#endif

namespace {

using tides::grid::LibxcFunctional;
using tides::grid::kLibxc_GGA_C_PBE;
using tides::grid::kLibxc_GGA_X_PBE;
using tides::grid::xc::Family;
using tides::grid::xc::Functional;
using tides::grid::xc::PrecisionPolicy;
using tides::grid::xc::XcArena;
using tides::grid::xc::XcEval;
using tides::grid::xc::XcGridIn;
using tides::grid::xc::XcGridOut;
using tides::grid::xc::XcSpec;

constexpr double kRelativeTolerance = TIDES_XC_RUNG0_REL;

double RelativeError(double observed, double expected) {
  return std::abs(observed - expected) / std::max(std::abs(expected), 1.0e-16);
}

bool Matches(double observed, double expected) {
  return std::abs(observed - expected) <= kRelativeTolerance ||
         RelativeError(observed, expected) <= kRelativeTolerance;
}

int Fail(const char* message) {
  std::cerr << "xc_tier2_cpu_fallback: " << message << '\n';
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
  std::vector<double> gradient_x(rho.size());
  std::vector<double> gradient_y(rho.size());
  std::vector<double> gradient_z(rho.size());
  std::vector<double> sigma(rho.size());
  for (std::size_t point = 0; point < rho.size(); ++point) {
    const double gradient_norm = 0.05 * static_cast<double>(point + 1);
    const double scale = gradient_norm / std::sqrt(14.0);
    gradient_x[point] = scale;
    gradient_y[point] = 2.0 * scale;
    gradient_z[point] = 3.0 * scale;
    sigma[point] = gradient_norm * gradient_norm;
  }

  LibxcFunctional exchange;
  LibxcFunctional correlation;
  if (!exchange.Init(kLibxc_GGA_X_PBE, XC_UNPOLARIZED) ||
      !correlation.Init(kLibxc_GGA_C_PBE, XC_UNPOLARIZED)) {
    return Fail("failed to initialize the libxc PBE oracle");
  }
  const auto libxc_x = exchange.EvalGGA(rho, sigma, rho.size());
  const auto libxc_c = correlation.EvalGGA(rho, sigma, rho.size());

  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) return Fail("cudaStreamCreate failed");
  XcArena arena;
  const double coefficient = 0.5;
  const auto reserve_status = arena.Reserve(rho.size(), 1, true, false, 1, stream);
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
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    return Fail("PBE input upload failed");
  }

  XcSpec spec;
  spec.family = Family::kGga;
  spec.nspin = 1;
  // A single term with coefficient != 1.0 bypasses the Tier-0 fast path and
  // routes through the Tier-2 synchronous CPU fallback.
  spec.terms = {{Functional::kPbe, coefficient}};
  spec.precision = PrecisionPolicy::kFloat64;
  XcGridIn input{arena.rho(), arena.grad(), nullptr, arena.weights(),
                 static_cast<std::int64_t>(rho.size()),
                 static_cast<std::int64_t>(stride), 1, arena.sys_offsets()};
  XcGridOut output{arena.wv_rho(), arena.wv_grad(), nullptr,
                   arena.exc_per_system()};
  const auto eval_status = XcEval(spec, input, output, stream);
  if (!eval_status.ok()) {
    std::cerr << eval_status.message() << '\n';
    return 1;
  }

  std::vector<double> observed_wv_rho(rho.size());
  std::vector<double> observed_wv_grad(3 * stride);
  double observed_energy = 0.0;
  if (cudaMemcpyAsync(observed_wv_rho.data(), arena.wv_rho(), scalar_bytes,
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(observed_wv_grad.data(), arena.wv_grad(),
                      observed_wv_grad.size() * sizeof(double),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(&observed_energy, arena.exc_per_system(), sizeof(double),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    return Fail("Tier-2 output download failed");
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
  bool all_match = true;
  for (std::size_t point = 0; point < rho.size(); ++point) {
    const double eps = libxc_x.eps_xc[point] + libxc_c.eps_xc[point];
    const double vrho = libxc_x.vrho[point] + libxc_c.vrho[point];
    const double vsigma = libxc_x.vsigma[point] + libxc_c.vsigma[point];
    expected_energy += weights[point] * rho[point] * (coefficient * eps);
    const double expected_wv_rho = weights[point] * (coefficient * vrho);
    max_wv_rho_rel = std::max(max_wv_rho_rel, RelativeError(observed_wv_rho[point], expected_wv_rho));
    all_match = all_match && Matches(observed_wv_rho[point], expected_wv_rho);
    const double weighted_vsigma = 2.0 * weights[point] * (coefficient * vsigma);
    const double expected_gradient[3] = {
        weighted_vsigma * gradient_x[point],
        weighted_vsigma * gradient_y[point],
        weighted_vsigma * gradient_z[point]};
    for (std::size_t component = 0; component < 3; ++component) {
      const double observed_grad = observed_wv_grad[component * stride + point];
      const double expected_grad = expected_gradient[component];
      max_wv_grad_rel = std::max(max_wv_grad_rel, RelativeError(observed_grad, expected_grad));
      all_match = all_match && Matches(observed_grad, expected_grad);
    }
  }
  const double energy_rel = RelativeError(observed_energy, expected_energy);
  std::cout << "xc_tier2_cpu_fallback: points=" << rho.size()
            << " stride=" << stride
            << " max_wv_rho_rel=" << max_wv_rho_rel
            << " max_wv_grad_rel=" << max_wv_grad_rel
            << " energy_rel=" << energy_rel
            << " tolerance=" << kRelativeTolerance << '\n';
  if (!all_match || energy_rel > kRelativeTolerance) {
    return Fail("Tier-2 CPU fallback weighted outputs differ from the libxc oracle");
  }
  return 0;
}
