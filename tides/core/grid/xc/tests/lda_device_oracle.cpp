#include "grid/libxc_wrapper.hpp"
#include "grid/xc/functionals/lda_pw92.cuh"
#include "grid/xc/functionals/lda_slater.cuh"
#include "grid/xc/xc_arena.hpp"
#include "grid/xc/xc_engine.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#ifndef TIDES_XC_RUNG0_REL
#error "TIDES_XC_RUNG0_REL must be configured from verification/tolerances.yaml"
#endif

namespace {

using tides::grid::LibxcFunctional;
using tides::grid::kLibxc_LDA_C_PW;
using tides::grid::kLibxc_LDA_X;
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

int Fail(const char* message) {
  std::cerr << "xc_rung0_device: " << message << '\n';
  return 1;
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: CUDA runtime not available\n";
    return 77;
  }
  std::vector<double> rho;
  for (int exponent = -14; exponent <= 4; ++exponent) {
    for (int mantissa = 1; mantissa <= 3; ++mantissa) {
      rho.push_back(static_cast<double>(mantissa) * std::pow(10.0, exponent));
    }
  }
  const std::vector<double> weights(rho.size(), 1.0);

  LibxcFunctional exchange;
  LibxcFunctional correlation;
  if (!exchange.Init(kLibxc_LDA_X, XC_UNPOLARIZED) ||
      !correlation.Init(kLibxc_LDA_C_PW, XC_UNPOLARIZED)) {
    return Fail("failed to initialize libxc");
  }
  const auto libxc_x = exchange.EvalLDA(rho, rho.size());
  const auto libxc_c = correlation.EvalLDA(rho, rho.size());
  std::vector<double> expected_vrho(rho.size());
  std::vector<double> expected_eps(rho.size());
  for (std::size_t point = 0; point < rho.size(); ++point) {
    expected_vrho[point] = libxc_x.vrho[point] + libxc_c.vrho[point];
    expected_eps[point] = libxc_x.eps_xc[point] + libxc_c.eps_xc[point];
  }

  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) return Fail("cudaStreamCreate failed");
  XcArena arena;
  const auto reserve_status = arena.Reserve(rho.size(), 1, false, false, 1, stream);
  if (!reserve_status.ok()) {
    std::cerr << reserve_status.message() << '\n';
    return 1;
  }
  const std::size_t reserved_capacity = arena.capacity();
  double* const first_rho_pointer = arena.rho();
  const auto repeat_reserve_status = arena.Reserve(rho.size(), 1, false, false, 1, stream);
  if (!repeat_reserve_status.ok() || arena.rho() != first_rho_pointer) {
    return Fail("same-shape arena reserve allocated during the iteration path");
  }
  const std::size_t bytes = rho.size() * sizeof(double);
  if (cudaMemcpyAsync(arena.rho(), rho.data(), bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(arena.weights(), weights.data(), bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    return Fail("input upload failed");
  }

  XcSpec spec;
  spec.family = Family::kLda;
  spec.nspin = 1;
  spec.terms = {{Functional::kLdaPw92, 1.0}};
  spec.precision = PrecisionPolicy::kFloat64;
  XcGridIn input{arena.rho(), nullptr, nullptr, arena.weights(),
                 static_cast<std::int64_t>(rho.size()),
                 static_cast<std::int64_t>(arena.capacity()), 1,
                 arena.sys_offsets()};
  XcGridOut output{arena.wv_rho(), nullptr, nullptr, arena.exc_per_system()};
  const auto eval_status = XcEval(spec, input, output, stream);
  if (!eval_status.ok()) {
    std::cerr << eval_status.message() << '\n';
    return 1;
  }

  std::vector<double> observed_vrho(rho.size());
  double observed_energy = 0.0;
  if (cudaMemcpyAsync(observed_vrho.data(), arena.wv_rho(), bytes,
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(&observed_energy, arena.exc_per_system(), sizeof(double),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    return Fail("output download failed");
  }
  const auto release_status = arena.Release(stream);
  if (!release_status.ok()) {
    std::cerr << release_status.message() << '\n';
    return 1;
  }
  if (cudaStreamDestroy(stream) != cudaSuccess) return Fail("cudaStreamDestroy failed");

  double max_vrho_relative_error = 0.0;
  for (std::size_t point = 0; point < rho.size(); ++point) {
    max_vrho_relative_error = std::max(max_vrho_relative_error,
                                       RelativeError(observed_vrho[point], expected_vrho[point]));
  }
  const double expected_energy = std::inner_product(
      rho.begin(), rho.end(), expected_eps.begin(), 0.0);
  const double energy_relative_error = RelativeError(observed_energy, expected_energy);
  std::cout << "xc_rung0_device: points=" << rho.size()
            << " capacity=" << reserved_capacity
            << " max_vrho_rel=" << max_vrho_relative_error
            << " energy_rel=" << energy_relative_error
            << " tolerance=" << kRung0RelativeTolerance << '\n';
  if (max_vrho_relative_error > kRung0RelativeTolerance ||
      energy_relative_error > kRung0RelativeTolerance) {
    return Fail("device Tier-0 LDA-PW92 differs from the libxc oracle");
  }
  return 0;
}
