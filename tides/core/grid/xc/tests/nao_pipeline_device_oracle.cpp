#include "grid/rho_build_gpu.hpp"
#include "grid/vmat_build_gpu.hpp"
#include "grid/xc/functionals/gga_pbe.cuh"
#include "grid/xc/xc_arena.hpp"
#include "grid/xc/xc_engine.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#ifndef TIDES_XC_RUNG0_REL
#error "TIDES_XC_RUNG0_REL must be configured from verification/tolerances.yaml"
#endif

namespace {

using tides::grid::BuildGgaVmatDevice;
using tides::grid::BuildRhoGradientDevice;
using tides::grid::GgaVmatDeviceIn;
using tides::grid::RhoGradientDeviceIn;
using tides::grid::xc::Family;
using tides::grid::xc::Functional;
using tides::grid::xc::GgaPbeStandard;
using tides::grid::xc::PrecisionPolicy;
using tides::grid::xc::XcArena;
using tides::grid::xc::XcEval;
using tides::grid::xc::XcGridIn;
using tides::grid::xc::XcGridOut;
using tides::grid::xc::XcSpec;

constexpr std::size_t kBasis = 3;
constexpr std::size_t kPoints = 11;
constexpr std::size_t kStride = 512;
constexpr double kRung0RelativeTolerance = TIDES_XC_RUNG0_REL;

std::size_t PhiIndex(std::size_t basis, std::size_t point) {
  return basis * kStride + point;
}

std::size_t GradPhiIndex(std::size_t component, std::size_t basis,
                         std::size_t point) {
  return (component * kBasis + basis) * kStride + point;
}

double RelativeError(double observed, double expected) {
  return std::abs(observed - expected) / std::max(std::abs(expected), 1.0e-16);
}

int Fail(const char* message) {
  std::cerr << "xc_nao_device_pipeline: " << message << '\n';
  return 1;
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: CUDA runtime not available\n";
    return 77;
  }

  const std::array<double, kBasis * kBasis> density_matrix = {{
      1.20, 0.12, -0.04,
      0.12, 0.95, 0.08,
      -0.04, 0.08, 0.75,
  }};
  constexpr std::array<std::array<double, 3>, kBasis> centers = {{
      {{-0.35, 0.05, -0.10}},
      {{0.30, -0.20, 0.20}},
      {{0.10, 0.30, 0.25}},
  }};
  constexpr std::array<double, kBasis> alpha = {{0.7, 1.1, 0.85}};
  std::vector<double> phi(kBasis * kStride, 0.0);
  std::vector<double> grad_phi(3 * kBasis * kStride, 0.0);
  std::array<double, kPoints> weights{};
  for (std::size_t point = 0; point < kPoints; ++point) {
    const double t = -0.9 + 1.8 * static_cast<double>(point) /
        static_cast<double>(kPoints - 1);
    const std::array<double, 3> coordinate = {
        t, 0.25 * std::sin(1.3 * t), 0.20 * std::cos(0.8 * t)};
    weights[point] = 0.10 + 0.01 * static_cast<double>(point);
    for (std::size_t basis = 0; basis < kBasis; ++basis) {
      std::array<double, 3> delta{};
      double radius_squared = 0.0;
      for (std::size_t component = 0; component < 3; ++component) {
        delta[component] = coordinate[component] - centers[basis][component];
        radius_squared += delta[component] * delta[component];
      }
      const double value = std::exp(-alpha[basis] * radius_squared);
      phi[PhiIndex(basis, point)] = value;
      for (std::size_t component = 0; component < 3; ++component) {
        grad_phi[GradPhiIndex(component, basis, point)] =
            -2.0 * alpha[basis] * delta[component] * value;
      }
    }
  }

  std::array<double, kPoints> expected_rho{};
  std::array<double, 3 * kPoints> expected_grad{};
  for (std::size_t point = 0; point < kPoints; ++point) {
    for (std::size_t mu = 0; mu < kBasis; ++mu) {
      for (std::size_t nu = 0; nu < kBasis; ++nu) {
        const double coefficient = density_matrix[mu * kBasis + nu];
        const double phi_mu = phi[PhiIndex(mu, point)];
        const double phi_nu = phi[PhiIndex(nu, point)];
        expected_rho[point] += coefficient * phi_mu * phi_nu;
        for (std::size_t component = 0; component < 3; ++component) {
          expected_grad[component * kPoints + point] += coefficient *
              (grad_phi[GradPhiIndex(component, mu, point)] * phi_nu +
               phi_mu * grad_phi[GradPhiIndex(component, nu, point)]);
        }
      }
    }
  }

  std::array<double, kBasis * kBasis> expected_vmat{};
  double expected_energy = 0.0;
  for (std::size_t point = 0; point < kPoints; ++point) {
    const double gx = expected_grad[point];
    const double gy = expected_grad[kPoints + point];
    const double gz = expected_grad[2 * kPoints + point];
    const auto evaluation = GgaPbeStandard::Eval(
        expected_rho[point], gx * gx + gy * gy + gz * gz);
    const double wv_rho = weights[point] * evaluation.vrho;
    const std::array<double, 3> wv_grad = {
        2.0 * weights[point] * evaluation.vsigma * gx,
        2.0 * weights[point] * evaluation.vsigma * gy,
        2.0 * weights[point] * evaluation.vsigma * gz,
    };
    expected_energy += weights[point] * expected_rho[point] * evaluation.eps;
    for (std::size_t mu = 0; mu < kBasis; ++mu) {
      for (std::size_t nu = 0; nu < kBasis; ++nu) {
        const double phi_mu = phi[PhiIndex(mu, point)];
        const double phi_nu = phi[PhiIndex(nu, point)];
        double contribution = wv_rho * phi_mu * phi_nu;
        for (std::size_t component = 0; component < 3; ++component) {
          contribution += wv_grad[component] *
              (grad_phi[GradPhiIndex(component, mu, point)] * phi_nu +
               phi_mu * grad_phi[GradPhiIndex(component, nu, point)]);
        }
        expected_vmat[mu * kBasis + nu] += contribution;
      }
    }
  }

  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) return Fail("cudaStreamCreate failed");
  double* d_density_matrix = nullptr;
  double* d_phi = nullptr;
  double* d_grad_phi = nullptr;
  double* d_vmat = nullptr;
  auto cleanup = [&]() {
    if (d_density_matrix != nullptr) cudaFree(d_density_matrix);
    if (d_phi != nullptr) cudaFree(d_phi);
    if (d_grad_phi != nullptr) cudaFree(d_grad_phi);
    if (d_vmat != nullptr) cudaFree(d_vmat);
    cudaStreamDestroy(stream);
  };
  if (cudaMalloc(reinterpret_cast<void**>(&d_density_matrix),
                 density_matrix.size() * sizeof(double)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_phi), phi.size() * sizeof(double)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_grad_phi),
                 grad_phi.size() * sizeof(double)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_vmat),
                 kBasis * kBasis * sizeof(double)) != cudaSuccess) {
    cleanup();
    return Fail("device allocation failed");
  }
  if (cudaMemcpyAsync(d_density_matrix, density_matrix.data(),
                      density_matrix.size() * sizeof(double), cudaMemcpyHostToDevice,
                      stream) != cudaSuccess ||
      cudaMemcpyAsync(d_phi, phi.data(), phi.size() * sizeof(double),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(d_grad_phi, grad_phi.data(), grad_phi.size() * sizeof(double),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    cleanup();
    return Fail("input upload failed");
  }

  XcArena arena;
  const auto reserve_status = arena.Reserve(kPoints, 1, true, false, 1, stream);
  if (!reserve_status.ok()) {
    std::cerr << reserve_status.message() << '\n';
    cleanup();
    return 1;
  }
  if (arena.capacity() != kStride) {
    cleanup();
    return Fail("test requires the documented 512-point arena padding");
  }
  if (cudaMemcpyAsync(arena.weights(), weights.data(), kPoints * sizeof(double),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    cleanup();
    return Fail("weight upload failed");
  }
  const RhoGradientDeviceIn rho_input{
      d_density_matrix, d_phi, d_grad_phi, static_cast<std::int64_t>(kBasis),
      static_cast<std::int64_t>(kPoints), static_cast<std::int64_t>(kStride)};
  const auto rho_status = BuildRhoGradientDevice(rho_input, arena.rho(), arena.grad(),
                                                  stream);
  if (!rho_status.ok()) {
    std::cerr << rho_status.message() << '\n';
    cleanup();
    return 1;
  }
  XcSpec spec;
  spec.family = Family::kGga;
  spec.nspin = 1;
  spec.terms = {{Functional::kPbe, 1.0}};
  spec.precision = PrecisionPolicy::kFloat64;
  XcGridIn xc_input{arena.rho(), arena.grad(), nullptr, arena.weights(),
                    static_cast<std::int64_t>(kPoints),
                    static_cast<std::int64_t>(kStride), 1, arena.sys_offsets()};
  XcGridOut xc_output{arena.wv_rho(), arena.wv_grad(), nullptr,
                      arena.exc_per_system()};
  const auto xc_status = XcEval(spec, xc_input, xc_output, stream);
  if (!xc_status.ok()) {
    std::cerr << xc_status.message() << '\n';
    cleanup();
    return 1;
  }
  const GgaVmatDeviceIn vmat_input{
      d_phi, d_grad_phi, arena.wv_rho(), arena.wv_grad(),
      static_cast<std::int64_t>(kBasis), static_cast<std::int64_t>(kPoints),
      static_cast<std::int64_t>(kStride)};
  const auto vmat_status = BuildGgaVmatDevice(vmat_input, d_vmat, stream);
  if (!vmat_status.ok()) {
    std::cerr << vmat_status.message() << '\n';
    cleanup();
    return 1;
  }

  std::vector<double> observed_rho(kStride);
  std::vector<double> observed_grad(3 * kStride);
  std::vector<double> observed_wv_rho(kStride);
  std::vector<double> observed_wv_grad(3 * kStride);
  std::array<double, kBasis * kBasis> observed_vmat{};
  double observed_energy = 0.0;
  if (cudaMemcpyAsync(observed_rho.data(), arena.rho(), observed_rho.size() * sizeof(double),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(observed_grad.data(), arena.grad(), observed_grad.size() * sizeof(double),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(observed_wv_rho.data(), arena.wv_rho(),
                      observed_wv_rho.size() * sizeof(double), cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess ||
      cudaMemcpyAsync(observed_wv_grad.data(), arena.wv_grad(),
                      observed_wv_grad.size() * sizeof(double), cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess ||
      cudaMemcpyAsync(observed_vmat.data(), d_vmat, observed_vmat.size() * sizeof(double),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(&observed_energy, arena.exc_per_system(), sizeof(double),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    cleanup();
    return Fail("output download failed");
  }
  const auto release_status = arena.Release(stream);
  if (!release_status.ok()) {
    std::cerr << release_status.message() << '\n';
    cleanup();
    return 1;
  }
  cudaStreamSynchronize(stream);
  cleanup();

  double max_rho_error = 0.0;
  double max_gradient_error = 0.0;
  double max_vmat_error = 0.0;
  for (std::size_t point = 0; point < kPoints; ++point) {
    max_rho_error = std::max(max_rho_error,
                             std::abs(observed_rho[point] - expected_rho[point]));
    for (std::size_t component = 0; component < 3; ++component) {
      max_gradient_error = std::max(
          max_gradient_error,
          std::abs(observed_grad[component * kStride + point] -
                   expected_grad[component * kPoints + point]));
    }
  }
  for (std::size_t index = 0; index < observed_vmat.size(); ++index) {
    max_vmat_error = std::max(max_vmat_error,
                              std::abs(observed_vmat[index] - expected_vmat[index]));
  }
  const double energy_relative_error = RelativeError(observed_energy, expected_energy);
  std::cout << "xc_nao_device_pipeline: points=" << kPoints
            << " stride=" << kStride
            << " max_rho_abs=" << max_rho_error
            << " max_grad_abs=" << max_gradient_error
            << " max_vmat_abs=" << max_vmat_error
            << " energy_rel=" << energy_relative_error
            << " tolerance=" << kRung0RelativeTolerance << '\n';
  if (max_rho_error > 1.0e-10 || max_gradient_error > 1.0e-10 ||
      max_vmat_error > 1.0e-10 || energy_relative_error > kRung0RelativeTolerance) {
    return Fail("device-resident rho-to-XC-to-GGA-vmat pipeline differs from CPU oracle");
  }
  return 0;
}
