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

using tides::grid::kLibxc_LDA_X;
using tides::grid::kLibxc_LDA_C_PW;
using tides::grid::kLibxc_LDA_C_VWN;
using tides::grid::kLibxc_GGA_X_PBE;
using tides::grid::kLibxc_GGA_C_PBE;
using tides::grid::kLibxc_GGA_C_PBE_SOL;
using tides::grid::kLibxc_GGA_X_PBE_SOL;
using tides::grid::kLibxc_GGA_X_PBE_R;
using tides::grid::kLibxc_GGA_X_RPBE;
using tides::grid::kLibxc_GGA_X_B88;
using tides::grid::kLibxc_GGA_C_LYP;
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
  std::cerr << "xc_tier0_pol_device: " << message << '\n';
  return 1;
}

// ---------------------------------------------------------------------------
// LDA polarized test
// ---------------------------------------------------------------------------

int RunLdaPolTest(const char* name, int x_id, int c_id, Functional functional) {
  const std::vector<double> rho_total = {0.1, 0.25, 1.0, 2.0, 10.0, 100.0};
  const std::vector<double> weights = {0.5, 0.75, 1.0, 1.25, 1.5, 2.0};
  const std::size_t np = rho_total.size();

  std::vector<double> rho_up(np), rho_down(np);
  for (std::size_t i = 0; i < np; ++i) {
    rho_up[i] = 0.4 * rho_total[i];
    rho_down[i] = 0.6 * rho_total[i];
  }

  std::vector<double> rho_libxc(2 * np);
  for (std::size_t i = 0; i < np; ++i) {
    rho_libxc[2 * i] = rho_up[i];
    rho_libxc[2 * i + 1] = rho_down[i];
  }

  LibxcFunctional x_func, c_func;
  if (!x_func.Init(x_id, XC_POLARIZED)) {
    return Fail((std::string("failed to init libxc ") + name + " x oracle").c_str());
  }
  if (!c_func.Init(c_id, XC_POLARIZED)) {
    return Fail((std::string("failed to init libxc ") + name + " c oracle").c_str());
  }

  const auto x_oracle = x_func.EvalLDA(rho_libxc, np);
  const auto c_oracle = c_func.EvalLDA(rho_libxc, np);

  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) return Fail("cudaStreamCreate failed");
  XcArena arena;
  const auto reserve_status = arena.Reserve(np, 2, false, false, 1, stream);
  if (!reserve_status.ok()) {
    std::cerr << reserve_status.message() << '\n';
    return 1;
  }
  const std::size_t stride = arena.capacity();

  std::vector<double> rho_arena(2 * stride, 0.0);
  for (std::size_t i = 0; i < np; ++i) {
    rho_arena[i] = rho_up[i];
    rho_arena[stride + i] = rho_down[i];
  }

  if (cudaMemcpyAsync(arena.rho(), rho_arena.data(), rho_arena.size() * sizeof(double),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(arena.weights(), weights.data(), np * sizeof(double),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    return Fail("LDA polarized input upload failed");
  }

  XcSpec spec;
  spec.family = Family::kLda;
  spec.nspin = 2;
  spec.terms = {{functional, 1.0}};
  spec.precision = PrecisionPolicy::kFloat64;
  XcGridIn input{arena.rho(), nullptr, nullptr, arena.weights(),
                 static_cast<std::int64_t>(np), static_cast<std::int64_t>(stride), 1,
                 arena.sys_offsets()};
  XcGridOut output{arena.wv_rho(), nullptr, nullptr, arena.exc_per_system()};
  const auto eval_status = XcEval(spec, input, output, stream);
  if (!eval_status.ok()) {
    std::cerr << eval_status.message() << '\n';
    return 1;
  }

  std::vector<double> observed_wv_rho(2 * stride);
  double observed_energy = 0.0;
  if (cudaMemcpyAsync(observed_wv_rho.data(), arena.wv_rho(),
                      observed_wv_rho.size() * sizeof(double),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(&observed_energy, arena.exc_per_system(), sizeof(double),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    return Fail("LDA polarized output download failed");
  }

  spec.deterministic = true;
  std::uint64_t deterministic_energy_bits = 0;
  double deterministic_energy = 0.0;
  for (int run = 0; run < 100; ++run) {
    const auto det_status = XcEval(spec, input, output, stream);
    if (!det_status.ok()) {
      std::cerr << det_status.message() << '\n';
      return 1;
    }
    if (cudaMemcpyAsync(&deterministic_energy, arena.exc_per_system(), sizeof(double),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      return Fail("deterministic LDA polarized energy download failed");
    }
    const std::uint64_t observed_bits = std::bit_cast<std::uint64_t>(deterministic_energy);
    if (run == 0) {
      deterministic_energy_bits = observed_bits;
    } else if (observed_bits != deterministic_energy_bits) {
      return Fail("deterministic LDA polarized energy changed across repeated runs");
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
  bool all_wv_rho_match = true;
  for (std::size_t i = 0; i < np; ++i) {
    const double eps = x_oracle.eps_xc[i] + c_oracle.eps_xc[i];
    const double vrho_up = x_oracle.vrho[2 * i] + c_oracle.vrho[2 * i];
    const double vrho_down = x_oracle.vrho[2 * i + 1] + c_oracle.vrho[2 * i + 1];
    expected_energy += weights[i] * (rho_up[i] + rho_down[i]) * eps;

    const double expected_wv_rho_up = weights[i] * vrho_up;
    const double expected_wv_rho_down = weights[i] * vrho_down;
    max_wv_rho_rel = std::max(
        max_wv_rho_rel,
        std::max(RelativeError(observed_wv_rho[i], expected_wv_rho_up),
                 RelativeError(observed_wv_rho[stride + i], expected_wv_rho_down)));
    all_wv_rho_match = all_wv_rho_match &&
        MatchesOracle(observed_wv_rho[i], expected_wv_rho_up) &&
        MatchesOracle(observed_wv_rho[stride + i], expected_wv_rho_down);
  }
  const double energy_rel = RelativeError(observed_energy, expected_energy);
  const double det_energy_rel = RelativeError(deterministic_energy, expected_energy);
  std::cout << "xc_tier0_pol_lda_" << name << ": points=" << np
            << " stride=" << stride << " max_wv_rho_rel=" << max_wv_rho_rel
            << " energy_rel=" << energy_rel
            << " det_energy_rel=" << det_energy_rel
            << " det_runs=100 tolerance=" << kRung0RelativeTolerance << '\n';
  if (!all_wv_rho_match || energy_rel > kRung0RelativeTolerance ||
      det_energy_rel > kRung0RelativeTolerance) {
    return Fail("device polarized LDA outputs differ from libxc oracle");
  }
  return 0;
}

// ---------------------------------------------------------------------------
// GGA polarized test
// ---------------------------------------------------------------------------

int RunGgaPolTest(const char* name, int x_id, int c_id, Functional functional) {
  const std::vector<double> rho_total = {0.1, 0.25, 1.0, 2.0, 10.0, 100.0};
  const std::vector<double> weights = {0.5, 0.75, 1.0, 1.25, 1.5, 2.0};
  const std::size_t np = rho_total.size();

  std::vector<double> rho_up(np), rho_down(np);
  std::vector<double> gradient_up_x(np), gradient_up_y(np), gradient_up_z(np);
  std::vector<double> gradient_down_x(np), gradient_down_y(np), gradient_down_z(np);
  std::vector<double> sigma_up(np), sigma_down(np), sigma_ud(np);

  for (std::size_t i = 0; i < np; ++i) {
    rho_up[i] = 0.4 * rho_total[i];
    rho_down[i] = 0.6 * rho_total[i];

    const double gradient_norm = 0.05 * static_cast<double>(i + 1);
    const double scale = gradient_norm / std::sqrt(14.0);
    gradient_up_x[i] = 0.6 * scale;
    gradient_up_y[i] = 0.6 * 2.0 * scale;
    gradient_up_z[i] = 0.6 * 3.0 * scale;
    gradient_down_x[i] = 0.8 * scale;
    gradient_down_y[i] = 0.8 * 2.0 * scale;
    gradient_down_z[i] = 0.8 * 3.0 * scale;

    sigma_up[i] = gradient_up_x[i] * gradient_up_x[i] +
                  gradient_up_y[i] * gradient_up_y[i] +
                  gradient_up_z[i] * gradient_up_z[i];
    sigma_down[i] = gradient_down_x[i] * gradient_down_x[i] +
                    gradient_down_y[i] * gradient_down_y[i] +
                    gradient_down_z[i] * gradient_down_z[i];
    sigma_ud[i] = gradient_up_x[i] * gradient_down_x[i] +
                  gradient_up_y[i] * gradient_down_y[i] +
                  gradient_up_z[i] * gradient_down_z[i];
  }

  std::vector<double> rho_libxc(2 * np);
  std::vector<double> sigma_libxc(3 * np);
  for (std::size_t i = 0; i < np; ++i) {
    rho_libxc[2 * i] = rho_up[i];
    rho_libxc[2 * i + 1] = rho_down[i];
    sigma_libxc[3 * i] = sigma_up[i];
    sigma_libxc[3 * i + 1] = sigma_ud[i];
    sigma_libxc[3 * i + 2] = sigma_down[i];
  }

  LibxcFunctional x_func, c_func;
  if (!x_func.Init(x_id, XC_POLARIZED)) {
    return Fail((std::string("failed to init libxc ") + name + " x oracle").c_str());
  }
  if (!c_func.Init(c_id, XC_POLARIZED)) {
    return Fail((std::string("failed to init libxc ") + name + " c oracle").c_str());
  }

  const auto x_oracle = x_func.EvalGGA(rho_libxc, sigma_libxc, np);
  const auto c_oracle = c_func.EvalGGA(rho_libxc, sigma_libxc, np);

  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) return Fail("cudaStreamCreate failed");
  XcArena arena;
  const auto reserve_status = arena.Reserve(np, 2, true, false, 1, stream);
  if (!reserve_status.ok()) {
    std::cerr << reserve_status.message() << '\n';
    return 1;
  }
  const std::size_t stride = arena.capacity();

  std::vector<double> grad(6 * stride, 0.0);
  std::vector<double> rho_arena(2 * stride, 0.0);
  for (std::size_t i = 0; i < np; ++i) {
    rho_arena[i] = rho_up[i];
    rho_arena[stride + i] = rho_down[i];
    grad[i] = gradient_up_x[i];
    grad[stride + i] = gradient_up_y[i];
    grad[2 * stride + i] = gradient_up_z[i];
    grad[3 * stride + i] = gradient_down_x[i];
    grad[4 * stride + i] = gradient_down_y[i];
    grad[5 * stride + i] = gradient_down_z[i];
  }

  if (cudaMemcpyAsync(arena.rho(), rho_arena.data(), rho_arena.size() * sizeof(double),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(arena.grad(), grad.data(), grad.size() * sizeof(double),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(arena.weights(), weights.data(), np * sizeof(double),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    return Fail("GGA polarized input upload failed");
  }

  XcSpec spec;
  spec.family = Family::kGga;
  spec.nspin = 2;
  spec.terms = {{functional, 1.0}};
  spec.precision = PrecisionPolicy::kFloat64;
  XcGridIn input{arena.rho(), arena.grad(), nullptr, arena.weights(),
                 static_cast<std::int64_t>(np), static_cast<std::int64_t>(stride), 1,
                 arena.sys_offsets()};
  XcGridOut output{arena.wv_rho(), arena.wv_grad(), nullptr, arena.exc_per_system()};
  const auto eval_status = XcEval(spec, input, output, stream);
  if (!eval_status.ok()) {
    std::cerr << eval_status.message() << '\n';
    return 1;
  }

  std::vector<double> observed_wv_rho(2 * stride);
  std::vector<double> observed_wv_grad(6 * stride);
  double observed_energy = 0.0;
  if (cudaMemcpyAsync(observed_wv_rho.data(), arena.wv_rho(),
                      observed_wv_rho.size() * sizeof(double),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(observed_wv_grad.data(), arena.wv_grad(),
                      observed_wv_grad.size() * sizeof(double),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(&observed_energy, arena.exc_per_system(), sizeof(double),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    return Fail("GGA polarized output download failed");
  }

  spec.deterministic = true;
  std::uint64_t deterministic_energy_bits = 0;
  double deterministic_energy = 0.0;
  for (int run = 0; run < 100; ++run) {
    const auto det_status = XcEval(spec, input, output, stream);
    if (!det_status.ok()) {
      std::cerr << det_status.message() << '\n';
      return 1;
    }
    if (cudaMemcpyAsync(&deterministic_energy, arena.exc_per_system(), sizeof(double),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      return Fail("deterministic GGA polarized energy download failed");
    }
    const std::uint64_t observed_bits = std::bit_cast<std::uint64_t>(deterministic_energy);
    if (run == 0) {
      deterministic_energy_bits = observed_bits;
    } else if (observed_bits != deterministic_energy_bits) {
      return Fail("deterministic GGA polarized energy changed across repeated runs");
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
  bool all_wv_rho_match = true;
  bool all_wv_grad_match = true;
  for (std::size_t i = 0; i < np; ++i) {
    const double eps = x_oracle.eps_xc[i] + c_oracle.eps_xc[i];
    const double vrho_up = x_oracle.vrho[2 * i] + c_oracle.vrho[2 * i];
    const double vrho_down = x_oracle.vrho[2 * i + 1] + c_oracle.vrho[2 * i + 1];
    const double vsigma_up = x_oracle.vsigma[3 * i] + c_oracle.vsigma[3 * i];
    const double vsigma_ud = x_oracle.vsigma[3 * i + 1] + c_oracle.vsigma[3 * i + 1];
    const double vsigma_down = x_oracle.vsigma[3 * i + 2] + c_oracle.vsigma[3 * i + 2];
    expected_energy += weights[i] * (rho_up[i] + rho_down[i]) * eps;

    const double expected_wv_rho_up = weights[i] * vrho_up;
    const double expected_wv_rho_down = weights[i] * vrho_down;
    max_wv_rho_rel = std::max(
        max_wv_rho_rel,
        std::max(RelativeError(observed_wv_rho[i], expected_wv_rho_up),
                 RelativeError(observed_wv_rho[stride + i], expected_wv_rho_down)));
    all_wv_rho_match = all_wv_rho_match &&
        MatchesOracle(observed_wv_rho[i], expected_wv_rho_up) &&
        MatchesOracle(observed_wv_rho[stride + i], expected_wv_rho_down);

    const double wv_up[3] = {
        weights[i] * (2.0 * vsigma_up * gradient_up_x[i] +
                      vsigma_ud * gradient_down_x[i]),
        weights[i] * (2.0 * vsigma_up * gradient_up_y[i] +
                      vsigma_ud * gradient_down_y[i]),
        weights[i] * (2.0 * vsigma_up * gradient_up_z[i] +
                      vsigma_ud * gradient_down_z[i])};
    const double wv_down[3] = {
        weights[i] * (2.0 * vsigma_down * gradient_down_x[i] +
                      vsigma_ud * gradient_up_x[i]),
        weights[i] * (2.0 * vsigma_down * gradient_down_y[i] +
                      vsigma_ud * gradient_up_y[i]),
        weights[i] * (2.0 * vsigma_down * gradient_down_z[i] +
                      vsigma_ud * gradient_up_z[i])};
    for (std::size_t comp = 0; comp < 3; ++comp) {
      const double observed_up = observed_wv_grad[comp * stride + i];
      const double observed_down = observed_wv_grad[(comp + 3) * stride + i];
      const double rel_up = RelativeError(observed_up, wv_up[comp]);
      const double rel_down = RelativeError(observed_down, wv_down[comp]);
      max_wv_grad_rel = std::max(max_wv_grad_rel, std::max(rel_up, rel_down));
      all_wv_grad_match = all_wv_grad_match &&
          MatchesGradient(observed_up, wv_up[comp]) &&
          MatchesGradient(observed_down, wv_down[comp]);
      if (rel_up > 1.0e-12 || rel_down > 1.0e-12) {
        std::cout << "  " << name << " point=" << i << " comp=" << comp
                  << " obs_up=" << observed_up << " exp_up=" << wv_up[comp]
                  << " rel_up=" << rel_up << " obs_down=" << observed_down
                  << " exp_down=" << wv_down[comp] << " rel_down=" << rel_down
                  << '\n';
      }
    }
  }
  const double energy_rel = RelativeError(observed_energy, expected_energy);
  const double det_energy_rel = RelativeError(deterministic_energy, expected_energy);
  std::cout << "xc_tier0_pol_gga_" << name << ": points=" << np
            << " stride=" << stride << " max_wv_rho_rel=" << max_wv_rho_rel
            << " max_wv_grad_rel=" << max_wv_grad_rel
            << " energy_rel=" << energy_rel
            << " det_energy_rel=" << det_energy_rel
            << " det_runs=100 tolerance=" << kRung0RelativeTolerance << '\n';
  if (!all_wv_rho_match || !all_wv_grad_match ||
      energy_rel > kRung0RelativeTolerance ||
      det_energy_rel > kRung0RelativeTolerance) {
    return Fail("device polarized GGA outputs differ from libxc oracle");
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

  // LDA polarized tests
  if (RunLdaPolTest("PW92", kLibxc_LDA_X, kLibxc_LDA_C_PW, Functional::kLdaPw92) != 0) return 1;
  if (RunLdaPolTest("SVWN5", kLibxc_LDA_X, kLibxc_LDA_C_VWN, Functional::kSvwn5) != 0) return 1;

  // GGA polarized tests
  if (RunGgaPolTest("PBE", kLibxc_GGA_X_PBE, kLibxc_GGA_C_PBE, Functional::kPbe) != 0) return 1;
  if (RunGgaPolTest("PBEsol", kLibxc_GGA_X_PBE_SOL, kLibxc_GGA_C_PBE_SOL, Functional::kPbeSol) != 0) return 1;
  if (RunGgaPolTest("revPBE", kLibxc_GGA_X_PBE_R, kLibxc_GGA_C_PBE, Functional::kRevPbe) != 0) return 1;
  if (RunGgaPolTest("RPBE", kLibxc_GGA_X_RPBE, kLibxc_GGA_C_PBE, Functional::kRpbe) != 0) return 1;
  if (RunGgaPolTest("BLYP", kLibxc_GGA_X_B88, kLibxc_GGA_C_LYP, Functional::kBlyp) != 0) return 1;

  return 0;
}
