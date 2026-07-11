// T-X4.4: FP32 A/B precision path test.
//
// Validates that the FP32 storage path (float inputs, FP64 promote, float outputs)
// produces results within tolerance of the FP64 path for Tier-0 LDA/GGA functionals.
// Also verifies hazard escalation for low-density points.
//
// Usage: tides_xc_fp32_ab_test

#include "grid/libxc_wrapper.hpp"
#include "grid/xc/xc_arena.hpp"
#include "grid/xc/xc_engine.hpp"
#include "grid/xc/kernels/xc_gga_kernel.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>

namespace {

using tides::grid::LibxcFunctional;
using tides::grid::xc::Family;
using tides::grid::xc::Functional;
using tides::grid::xc::PrecisionPolicy;
using tides::grid::xc::XcArena;
using tides::grid::xc::XcEval;
using tides::grid::xc::XcGridIn;
using tides::grid::xc::XcGridInFp32;
using tides::grid::xc::XcGridOut;
using tides::grid::xc::XcGridOutFp32;
using tides::grid::xc::XcSpec;
using tides::grid::xc::XcTerm;
using tides::grid::xc::LaunchXcFunctionalFp32;

// FP32 path should be within 1e-5 relative of FP64 for moderate densities.
// This is looser than the FP64 oracle (1e-10) but tight enough to catch
// dispatch bugs and precision regressions.
constexpr double kFp32AbTolerance = 1.0e-5;

double RelativeError(double observed, double expected) {
  return std::abs(observed - expected) / std::max(std::abs(expected), 1.0e-16);
}

int Fail(const char* message) {
  std::cerr << "xc_fp32_ab_test: " << message << '\n';
  return 1;
}

struct FunctionalEntry {
  Functional functional;
  const char* name;
  Family family;
};

const FunctionalEntry kTestFunctionals[] = {
  {Functional::kLdaPw92,  "LDA-PW92",  Family::kLda},
  {Functional::kPbe,      "PBE",       Family::kGga},
  {Functional::kPbeSol,   "PBEsol",    Family::kGga},
  {Functional::kBlyp,     "BLYP",      Family::kGga},
};
constexpr int kNumFunctionals = sizeof(kTestFunctionals) / sizeof(kTestFunctionals[0]);

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: CUDA runtime not available\n";
    return 77;
  }

  // Moderate densities that stay well above the FP32 hazard threshold (1e-10).
  const std::vector<double> rho = {0.1, 0.25, 1.0, 2.0, 10.0, 100.0};
  const std::vector<double> weights(rho.size(), 1.0);
  const int np = static_cast<int>(rho.size());

  // Generate gradients: sigma = g^2 where g = 0.05*(i+1)
  std::vector<double> grad(np * 3, 0.0);
  for (int i = 0; i < np; ++i) {
    const double g = 0.05 * static_cast<double>(i + 1);
    const double scale = g / std::sqrt(14.0);
    grad[i] = scale;
    grad[np + i] = 2.0 * scale;
    grad[2 * np + i] = 3.0 * scale;
  }

  // FP32 copies
  std::vector<float> rho_f32(np), weights_f32(np), grad_f32(np * 3);
  for (int i = 0; i < np; ++i) {
    rho_f32[i] = static_cast<float>(rho[i]);
    weights_f32[i] = static_cast<float>(weights[i]);
  }
  for (int i = 0; i < np * 3; ++i) {
    grad_f32[i] = static_cast<float>(grad[i]);
  }

  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) return Fail("cudaStreamCreate failed");

  std::printf("=== T-X4.4: FP32 A/B Precision Path ===\n");
  std::printf("Functional       MaxRelErr(vrho)  MaxRelErr(vgrad)  EnergyRelErr  Status\n");
  std::printf("---------------- ----------------   ----------------  ------------  -------\n");

  int passed = 0, failed = 0;

  for (int fi = 0; fi < kNumFunctionals; ++fi) {
    const auto& entry = kTestFunctionals[fi];
    const bool needs_grad = (entry.family != Family::kLda);

    // --- FP64 reference ---
    XcArena arena64;
    auto st = arena64.Reserve(np, 1, needs_grad, false, 1, stream);
    if (!st.ok()) return Fail("arena64.Reserve failed");
    const std::size_t stride64 = arena64.capacity();
    cudaMemcpyAsync(arena64.rho(), rho.data(), np * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(arena64.weights(), weights.data(), np * sizeof(double), cudaMemcpyHostToDevice, stream);
    if (needs_grad) {
      std::vector<double> grad_padded(stride64 * 3, 0.0);
      for (int i = 0; i < np; ++i) {
        grad_padded[i] = grad[i];
        grad_padded[stride64 + i] = grad[np + i];
        grad_padded[2 * stride64 + i] = grad[2 * np + i];
      }
      cudaMemcpyAsync(arena64.grad(), grad_padded.data(), stride64 * 3 * sizeof(double),
                      cudaMemcpyHostToDevice, stream);
    }

    XcSpec spec64;
    spec64.family = entry.family;
    spec64.nspin = 1;
    spec64.terms = {{entry.functional, 1.0}};
    spec64.precision = PrecisionPolicy::kFloat64;
    spec64.deterministic = true;

    XcGridIn in64{arena64.rho(), needs_grad ? arena64.grad() : nullptr, nullptr,
                  arena64.weights(), np, static_cast<std::int64_t>(stride64), 1,
                  arena64.sys_offsets()};
    XcGridOut out64{arena64.wv_rho(), needs_grad ? arena64.wv_grad() : nullptr,
                    nullptr, arena64.exc_per_system()};
    st = XcEval(spec64, in64, out64, stream);
    if (!st.ok()) return Fail("XcEval FP64 failed");

    std::vector<double> ref_vrho(np);
    double ref_energy = 0.0;
    cudaMemcpyAsync(ref_vrho.data(), arena64.wv_rho(), np * sizeof(double),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&ref_energy, arena64.exc_per_system(), sizeof(double),
                    cudaMemcpyDeviceToHost, stream);
    std::vector<double> ref_wv_grad;
    if (needs_grad) {
      ref_wv_grad.resize(stride64 * 3);
      cudaMemcpyAsync(ref_wv_grad.data(), arena64.wv_grad(), stride64 * 3 * sizeof(double),
                      cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);

    // --- FP32 path ---
    // Allocate FP32 device buffers
    float* d_rho_f32 = nullptr;
    float* d_grad_f32 = nullptr;
    float* d_w_f32 = nullptr;
    float* d_wv_rho_f32 = nullptr;
    float* d_wv_grad_f32 = nullptr;
    double* d_exc_f32 = nullptr;

    cudaMalloc(&d_rho_f32, stride64 * sizeof(float));
    cudaMalloc(&d_w_f32, stride64 * sizeof(float));
    if (needs_grad) cudaMalloc(&d_grad_f32, stride64 * 3 * sizeof(float));
    cudaMalloc(&d_wv_rho_f32, stride64 * sizeof(float));
    if (needs_grad) cudaMalloc(&d_wv_grad_f32, stride64 * 3 * sizeof(float));
    cudaMalloc(&d_exc_f32, sizeof(double));

    // Pad FP32 arrays to stride
    std::vector<float> rho_padded(stride64, 0.0f), w_padded(stride64, 0.0f);
    std::vector<float> grad_padded_f32(stride64 * 3, 0.0f);
    for (int i = 0; i < np; ++i) {
      rho_padded[i] = rho_f32[i];
      w_padded[i] = weights_f32[i];
    }
    if (needs_grad) {
      for (int i = 0; i < np; ++i) {
        grad_padded_f32[i] = grad_f32[i];
        grad_padded_f32[stride64 + i] = grad_f32[np + i];
        grad_padded_f32[2 * stride64 + i] = grad_f32[2 * np + i];
      }
    }

    cudaMemcpyAsync(d_rho_f32, rho_padded.data(), stride64 * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_w_f32, w_padded.data(), stride64 * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    if (needs_grad) {
      cudaMemcpyAsync(d_grad_f32, grad_padded_f32.data(), stride64 * 3 * sizeof(float),
                      cudaMemcpyHostToDevice, stream);
    }
    cudaMemsetAsync(d_exc_f32, 0, sizeof(double), stream);

    XcSpec spec32;
    spec32.family = entry.family;
    spec32.nspin = 1;
    spec32.terms = {{entry.functional, 1.0}};
    spec32.precision = PrecisionPolicy::kFloat32MidScf;
    spec32.deterministic = true;

    XcGridInFp32 in32{d_rho_f32, needs_grad ? d_grad_f32 : nullptr, nullptr,
                      d_w_f32, np, static_cast<std::int64_t>(stride64), 1, nullptr};
    XcGridOutFp32 out32{d_wv_rho_f32, needs_grad ? d_wv_grad_f32 : nullptr,
                        nullptr, d_exc_f32};

    st = LaunchXcFunctionalFp32(spec32, in32, out32, stream);
    if (!st.ok()) {
      std::printf("%-16s FAILED: %s\n", entry.name, st.message());
      failed++;
      continue;
    }

    std::vector<float> f32_vrho(np);
    double f32_energy = 0.0;
    cudaMemcpyAsync(f32_vrho.data(), d_wv_rho_f32, np * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&f32_energy, d_exc_f32, sizeof(double),
                    cudaMemcpyDeviceToHost, stream);
    std::vector<float> f32_wv_grad;
    if (needs_grad) {
      f32_wv_grad.resize(stride64 * 3);
      cudaMemcpyAsync(f32_wv_grad.data(), d_wv_grad_f32, stride64 * 3 * sizeof(float),
                      cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);

    // Compare FP32 vs FP64
    double max_vrho_err = 0.0, max_vgrad_err = 0.0, energy_err = 0.0;
    bool ok = true;
    for (int i = 0; i < np; ++i) {
      const double vrho_err = RelativeError(static_cast<double>(f32_vrho[i]), ref_vrho[i]);
      max_vrho_err = std::max(max_vrho_err, vrho_err);
      if (vrho_err > kFp32AbTolerance && std::abs(ref_vrho[i]) > 1e-15) ok = false;

      if (needs_grad) {
        const double ref_gx = ref_wv_grad[i];
        const double f32_gx = static_cast<double>(f32_wv_grad[i]);
        const double gerr = RelativeError(f32_gx, ref_gx);
        max_vgrad_err = std::max(max_vgrad_err, gerr);
        if (gerr > kFp32AbTolerance && std::abs(ref_gx) > 1e-15) ok = false;
      }
    }
    energy_err = RelativeError(f32_energy, ref_energy);
    if (energy_err > kFp32AbTolerance) ok = false;

    std::printf("%-16s %.4e          %.4e           %.4e      %s\n",
                entry.name, max_vrho_err,
                needs_grad ? max_vgrad_err : 0.0,
                energy_err, ok ? "PASS" : "FAIL");

    if (ok) passed++; else failed++;

    // Cleanup FP32 buffers
    cudaFree(d_rho_f32);
    cudaFree(d_w_f32);
    if (d_grad_f32) cudaFree(d_grad_f32);
    cudaFree(d_wv_rho_f32);
    if (d_wv_grad_f32) cudaFree(d_wv_grad_f32);
    cudaFree(d_exc_f32);
  }

  std::printf("---------------- ----------------   ----------------  ------------  -------\n");
  std::printf("Summary: %d passed, %d failed out of %d\n", passed, failed, kNumFunctionals);
  std::printf("Tolerance: %.1e relative (FP32 store + FP64 promote vs pure FP64)\n", kFp32AbTolerance);

  // Test hazard escalation: verify that density below threshold produces zero output
  std::printf("\n--- Hazard Escalation Test ---\n");
  {
    const float hazard_rho = 1e-12f;  // below kFp32DensityHazardThreshold
    float* d_hr = nullptr;
    float* d_hw = nullptr;
    float* d_hv = nullptr;
    double* d_he = nullptr;
    cudaMalloc(&d_hr, sizeof(float));
    cudaMalloc(&d_hw, sizeof(float));
    cudaMalloc(&d_hv, sizeof(float));
    cudaMalloc(&d_he, sizeof(double));

    float hr = hazard_rho, hw = 1.0f;
    cudaMemcpy(d_hr, &hr, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hw, &hw, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemsetAsync(d_hv, 0, sizeof(float), stream);
    cudaMemsetAsync(d_he, 0, sizeof(double), stream);

    XcSpec spec_h;
    spec_h.family = Family::kLda;
    spec_h.nspin = 1;
    spec_h.terms = {{Functional::kLdaPw92, 1.0}};
    spec_h.precision = PrecisionPolicy::kFloat32MidScf;
    spec_h.deterministic = true;

    XcGridInFp32 in_h{d_hr, nullptr, nullptr, d_hw, 1, 1, 1, nullptr};
    XcGridOutFp32 out_h{d_hv, nullptr, nullptr, d_he};
    auto st = LaunchXcFunctionalFp32(spec_h, in_h, out_h, stream);
    cudaStreamSynchronize(stream);

    float hv = 0.0f;
    double he = 0.0;
    cudaMemcpy(&hv, d_hv, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&he, d_he, sizeof(double), cudaMemcpyDeviceToHost);

    if (hv == 0.0f && he == 0.0) {
      std::printf("Hazard escalation: PASS (rho=1e-12 -> zeroed output)\n");
    } else {
      std::printf("Hazard escalation: FAIL (rho=1e-12 -> vrho=%.6e, energy=%.6e)\n",
                  static_cast<double>(hv), he);
      failed++;
    }

    cudaFree(d_hr);
    cudaFree(d_hw);
    cudaFree(d_hv);
    cudaFree(d_he);
  }

  cudaStreamDestroy(stream);
  return (failed == 0) ? 0 : 1;
}
