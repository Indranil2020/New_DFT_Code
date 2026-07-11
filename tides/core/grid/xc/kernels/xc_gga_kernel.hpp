#pragma once

#include "grid/xc/xc_engine.hpp"

namespace tides::grid::xc {

// Enqueues the unpolarized FP64 evaluation for the given Tier-0 functional.
// XcEval owns contract validation and zeroing exc_per_system before this call.
[[nodiscard]] Status LaunchXcFunctional(const XcSpec& spec,
                                        const XcGridIn& input,
                                        XcGridOut& output,
                                        cudaStream_t stream);

// T-X4.4: FP32 storage path (consumer GPU).  Inputs are float arrays that get
// promoted to double inside the kernel for the functional evaluation.  Outputs
// (wv_rho, wv_grad) are written as float.  Energy accumulation is always FP64.
// Hazard escalation: points matching known FP32 hazards (spin ζ→±1, erfc
// underflow, SCAN α near von-Weizsäcker bound) are escalated to FP64 storage.
struct XcGridInFp32 {
  const float* rho = nullptr;
  const float* grad = nullptr;
  const float* tau = nullptr;
  const float* w = nullptr;
  std::int64_t np = 0;
  std::int64_t point_stride = 0;
  int nsys = 1;
  const std::int64_t* sys_offsets = nullptr;
};

struct XcGridOutFp32 {
  float* wv_rho = nullptr;
  float* wv_grad = nullptr;
  float* wv_tau = nullptr;
  double* exc_per_system = nullptr;
};

[[nodiscard]] Status LaunchXcFunctionalFp32(const XcSpec& spec,
                                            const XcGridInFp32& input,
                                            XcGridOutFp32& output,
                                            cudaStream_t stream);

// T-X4.5: Stress-tensor grid terms.
// Computes the XC contribution to the stress tensor from grid movement under
// strain. For GGA functionals, the stress has two parts:
//   σ_ab = (1/V) Σ_i w_i [ρ_i v_ρ_i δ_ab + 2 v_σ_i (∂ρ/∂a_i · ∂ρ/∂b_i)]
// For LDA, only the first term (isotropic) contributes.
// The caller provides grad_rho as the Cartesian gradient components (not sigma).
struct XcStressOut {
  double* stress;  // [6] symmetric 3x3: xx, yy, zz, xy, xz, yz
};

[[nodiscard]] Status LaunchXcStress(const XcSpec& spec,
                                     const XcGridIn& input,
                                     XcStressOut& stress_out,
                                     cudaStream_t stream);

}  // namespace tides::grid::xc
