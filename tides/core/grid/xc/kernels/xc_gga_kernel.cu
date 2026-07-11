#include "grid/xc/kernels/xc_gga_kernel.hpp"

#include "grid/xc/functional_dispatch.hpp"
#include "grid/xc/tier0/tier0_pol.hpp"
#include "grid/xc/tier1/mgga_tpss.hpp"
#include "grid/xc/tier1/mgga_scan.hpp"
#include "grid/xc/tier1/mgga_r2scan.hpp"
#include "grid/xc/tier1/mgga_m06_2x.hpp"
#include "grid/xc/tier1/rsh_hse06.hpp"
#include "grid/xc/tier1/rsh_wb97x.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <string>

namespace tides::grid::xc {
namespace {

[[nodiscard]] Status CudaStatus(cudaError_t error, const char* context) {
  if (error == cudaSuccess) return Status::Ok();
  return Status::IoError(std::string(context) + ": " + cudaGetErrorString(error));
}

constexpr int kThreads = 256;

// Generic kernel for any LDA/GGA/RSH functor that returns a GgaEvaluation.
// Func::kFamily determines whether grad and wv_grad are touched.  mGGA
// functionals are handled by a separate kernel once their functors are added.
template <class Func>
__global__ void FunctionalKernel(const double* __restrict__ rho,
                                 const double* __restrict__ grad,
                                 const double* __restrict__ weights,
                                 double* __restrict__ wv_rho,
                                 double* __restrict__ wv_grad,
                                 double* __restrict__ exc,
                                 std::int64_t np, std::int64_t point_stride,
                                 bool fast_reduction) {
  double local_energy = 0.0;
  for (std::int64_t point = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x;
       point < np;
       point += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const double density = rho[point];
    const double weight = weights[point];
    GgaEvaluation evaluation;
    if constexpr (Func::kFamily == Family::kLda) {
      evaluation = Func::Eval(density);
    } else {
      const double gx = grad[point];
      const double gy = grad[point_stride + point];
      const double gz = grad[2 * point_stride + point];
      const double sigma = gx * gx + gy * gy + gz * gz;
      evaluation = Func::Eval(density, sigma);
      const double weighted_gradient = 2.0 * weight * evaluation.vsigma;
      wv_grad[point] = weighted_gradient * gx;
      wv_grad[point_stride + point] = weighted_gradient * gy;
      wv_grad[2 * point_stride + point] = weighted_gradient * gz;
    }
    wv_rho[point] = weight * evaluation.vrho;
    if (fast_reduction) local_energy += weight * density * evaluation.eps;
  }
  if (!fast_reduction) return;
  for (int offset = 16; offset > 0; offset /= 2) {
    local_energy += __shfl_down_sync(0xffffffff, local_energy, offset);
  }
  if ((threadIdx.x & 31) == 0) atomicAdd(exc, local_energy);
}

template <class Func>
__global__ void FunctionalDeterministicEnergyKernel(
    const double* __restrict__ rho, const double* __restrict__ grad,
    const double* __restrict__ weights, double* __restrict__ exc,
    std::int64_t np, std::int64_t point_stride) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  double energy = 0.0;
  for (std::int64_t point = 0; point < np; ++point) {
    const double density = rho[point];
    GgaEvaluation evaluation;
    if constexpr (Func::kFamily == Family::kLda) {
      evaluation = Func::Eval(density);
    } else {
      const double gx = grad[point];
      const double gy = grad[point_stride + point];
      const double gz = grad[2 * point_stride + point];
      const double sigma = gx * gx + gy * gy + gz * gz;
      evaluation = Func::Eval(density, sigma);
    }
    energy += weights[point] * density * evaluation.eps;
  }
  exc[0] = energy;
}

template <class Func>
[[nodiscard]] Status LaunchFunctionalKernel(const XcGridIn& input,
                                            XcGridOut& output,
                                            cudaStream_t stream,
                                            bool deterministic) {
  const std::int64_t required_blocks = (input.np + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(std::min<std::int64_t>(required_blocks, 65535));
  FunctionalKernel<Func><<<blocks, kThreads, 0, stream>>>(
      input.rho, input.grad, input.w, output.wv_rho, output.wv_grad,
      output.exc_per_system, input.np, input.point_stride, !deterministic);
  Status status = CudaStatus(cudaGetLastError(), "FunctionalKernel launch");
  if (!status.ok() || !deterministic) return status;
  FunctionalDeterministicEnergyKernel<Func><<<1, 1, 0, stream>>>(
      input.rho, input.grad, input.w, output.exc_per_system, input.np,
      input.point_stride);
  return CudaStatus(cudaGetLastError(), "FunctionalDeterministicEnergyKernel launch");
}

}  // namespace

// T-X4.4: FP32 storage path kernel.
// Reads float inputs, promotes to double for functional evaluation,
// writes float outputs. Energy accumulation is always FP64.
// Hazard escalation: points with density below the FP32 safe threshold
// are skipped (output zeroed) to avoid catastrophic cancellation.
constexpr double kFp32DensityHazardThreshold = 1.0e-10;

template <class Func>
__global__ void FunctionalKernelFp32(
    const float* __restrict__ rho,
    const float* __restrict__ grad,
    const float* __restrict__ weights,
    float* __restrict__ wv_rho,
    float* __restrict__ wv_grad,
    double* __restrict__ exc,
    std::int64_t np, std::int64_t point_stride,
    bool fast_reduction) {
  double local_energy = 0.0;
  for (std::int64_t point = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x;
       point < np;
       point += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    // Promote to double for functional evaluation
    const double density = static_cast<double>(rho[point]);
    const double weight = static_cast<double>(weights[point]);

    // Hazard escalation: skip points where FP32 storage loses too much
    // precision. These points should be handled by a FP64 fallback path.
    if (density < kFp32DensityHazardThreshold) {
      wv_rho[point] = 0.0f;
      if constexpr (Func::kFamily != Family::kLda) {
        wv_grad[point] = 0.0f;
        wv_grad[point_stride + point] = 0.0f;
        wv_grad[2 * point_stride + point] = 0.0f;
      }
      continue;
    }

    GgaEvaluation evaluation;
    if constexpr (Func::kFamily == Family::kLda) {
      evaluation = Func::Eval(density);
    } else {
      const double gx = static_cast<double>(grad[point]);
      const double gy = static_cast<double>(grad[point_stride + point]);
      const double gz = static_cast<double>(grad[2 * point_stride + point]);
      const double sigma = gx * gx + gy * gy + gz * gz;
      evaluation = Func::Eval(density, sigma);
      const double weighted_gradient = 2.0 * weight * evaluation.vsigma;
      wv_grad[point] = static_cast<float>(weighted_gradient * gx);
      wv_grad[point_stride + point] = static_cast<float>(weighted_gradient * gy);
      wv_grad[2 * point_stride + point] = static_cast<float>(weighted_gradient * gz);
    }
    wv_rho[point] = static_cast<float>(weight * evaluation.vrho);
    if (fast_reduction) local_energy += weight * density * evaluation.eps;
  }
  if (!fast_reduction) return;
  for (int offset = 16; offset > 0; offset /= 2) {
    local_energy += __shfl_down_sync(0xffffffff, local_energy, offset);
  }
  if ((threadIdx.x & 31) == 0) atomicAdd(exc, local_energy);
}

template <class Func>
__global__ void FunctionalDeterministicEnergyKernelFp32(
    const float* __restrict__ rho,
    const float* __restrict__ grad,
    const float* __restrict__ weights,
    double* __restrict__ exc,
    std::int64_t np, std::int64_t point_stride) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  double energy = 0.0;
  for (std::int64_t point = 0; point < np; ++point) {
    const double density = static_cast<double>(rho[point]);
    if (density < kFp32DensityHazardThreshold) continue;
    GgaEvaluation evaluation;
    if constexpr (Func::kFamily == Family::kLda) {
      evaluation = Func::Eval(density);
    } else {
      const double gx = static_cast<double>(grad[point]);
      const double gy = static_cast<double>(grad[point_stride + point]);
      const double gz = static_cast<double>(grad[2 * point_stride + point]);
      const double sigma = gx * gx + gy * gy + gz * gz;
      evaluation = Func::Eval(density, sigma);
    }
    energy += static_cast<double>(weights[point]) * density * evaluation.eps;
  }
  exc[0] = energy;
}

template <class Func>
[[nodiscard]] Status LaunchFunctionalKernelFp32(const XcGridInFp32& input,
                                                 XcGridOutFp32& output,
                                                 cudaStream_t stream,
                                                 bool deterministic) {
  const std::int64_t required_blocks = (input.np + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(std::min<std::int64_t>(required_blocks, 65535));
  FunctionalKernelFp32<Func><<<blocks, kThreads, 0, stream>>>(
      input.rho, input.grad, input.w, output.wv_rho, output.wv_grad,
      output.exc_per_system, input.np, input.point_stride, !deterministic);
  Status status = CudaStatus(cudaGetLastError(), "FunctionalKernelFp32 launch");
  if (!status.ok() || !deterministic) return status;
  FunctionalDeterministicEnergyKernelFp32<Func><<<1, 1, 0, stream>>>(
      input.rho, input.grad, input.w, output.exc_per_system, input.np,
      input.point_stride);
  return CudaStatus(cudaGetLastError(), "FunctionalDeterministicEnergyKernelFp32 launch");
}

Status LaunchXcFunctionalFp32(const XcSpec& spec, const XcGridInFp32& input,
                              XcGridOutFp32& output, cudaStream_t stream) {
  // FP32 path currently supports Tier-0 LDA/GGA functionals only.
  // mGGA/RSH (Tier-1) remain FP64-only due to α/erfc hazard complexity.
  switch (spec.terms[0].functional) {
    case Functional::kLdaPw92:
      return LaunchFunctionalKernelFp32<LdaPw92Functor>(input, output, stream, spec.deterministic);
    case Functional::kSvwn5:
      return LaunchFunctionalKernelFp32<Svwn5Functor>(input, output, stream, spec.deterministic);
    case Functional::kPbe:
      return LaunchFunctionalKernelFp32<PbeFunctor>(input, output, stream, spec.deterministic);
    case Functional::kPbeSol:
      return LaunchFunctionalKernelFp32<PbeSolFunctor>(input, output, stream, spec.deterministic);
    case Functional::kRevPbe:
      return LaunchFunctionalKernelFp32<RevPbeFunctor>(input, output, stream, spec.deterministic);
    case Functional::kRpbe:
      return LaunchFunctionalKernelFp32<RpbeFunctor>(input, output, stream, spec.deterministic);
    case Functional::kBlyp:
      return LaunchFunctionalKernelFp32<BlypFunctor>(input, output, stream, spec.deterministic);
    case Functional::kB3lyp:
      return LaunchFunctionalKernelFp32<B3lypFunctor>(input, output, stream, spec.deterministic);
    case Functional::kPbe0:
      return LaunchFunctionalKernelFp32<Pbe0Functor>(input, output, stream, spec.deterministic);
    default:
      return Status::Unimplemented(
          "FP32 mid-SCF path supports Tier-0 LDA/GGA only. "
          "mGGA/RSH functionals require FP64 (SCAN α, erfc hazards).");
  }
}

// T-X4.5: Stress-tensor kernel.
// Computes σ_ab = Σ_i w_i [ρ_i v_ρ_i δ_ab + 2 v_σ_i (∂ρ/∂a · ∂ρ/∂b)]
// For LDA: only the isotropic term ρ * vrho * δ_ab.
// Output: 6-component symmetric tensor [xx, yy, zz, xy, xz, yz].
template <class Func>
__global__ void StressKernel(const double* __restrict__ rho,
                              const double* __restrict__ grad,
                              const double* __restrict__ weights,
                              double* __restrict__ stress,
                              std::int64_t np, std::int64_t point_stride) {
  // Use shared memory for partial sums (6 components).
  __shared__ double sdata[6 * kThreads];
  int tid = threadIdx.x;
  for (int c = 0; c < 6; ++c) sdata[c * kThreads + tid] = 0.0;

  for (std::int64_t point = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x;
       point < np;
       point += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const double density = rho[point];
    const double weight = weights[point];

    GgaEvaluation evaluation;
    if constexpr (Func::kFamily == Family::kLda) {
      evaluation = Func::Eval(density);
      // Isotropic contribution: w * rho * vrho * delta_ab
      const double isotropic = weight * density * evaluation.vrho;
      sdata[0 * kThreads + tid] += isotropic;  // xx
      sdata[1 * kThreads + tid] += isotropic;  // yy
      sdata[2 * kThreads + tid] += isotropic;  // zz
    } else {
      const double gx = grad[point];
      const double gy = grad[point_stride + point];
      const double gz = grad[2 * point_stride + point];
      const double sigma = gx * gx + gy * gy + gz * gz;
      evaluation = Func::Eval(density, sigma);
      const double isotropic = weight * density * evaluation.vrho;
      const double two_w_vs = 2.0 * weight * evaluation.vsigma;
      sdata[0 * kThreads + tid] += isotropic + two_w_vs * gx * gx;  // xx
      sdata[1 * kThreads + tid] += isotropic + two_w_vs * gy * gy;  // yy
      sdata[2 * kThreads + tid] += isotropic + two_w_vs * gz * gz;  // zz
      sdata[3 * kThreads + tid] += two_w_vs * gx * gy;              // xy
      sdata[4 * kThreads + tid] += two_w_vs * gx * gz;              // xz
      sdata[5 * kThreads + tid] += two_w_vs * gy * gz;              // yz
    }
  }

  __syncthreads();
  // Tree reduction within block
  for (int offset = kThreads / 2; offset > 0; offset /= 2) {
    if (tid < offset) {
      for (int c = 0; c < 6; ++c)
        sdata[c * kThreads + tid] += sdata[c * kThreads + tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    for (int c = 0; c < 6; ++c)
      atomicAdd(&stress[c], sdata[c * kThreads]);
  }
}

template <class Func>
[[nodiscard]] Status LaunchStressKernel(const XcSpec& spec,
                                         const XcGridIn& input,
                                         XcStressOut& stress_out,
                                         cudaStream_t stream) {
  cudaMemsetAsync(stress_out.stress, 0, 6 * sizeof(double), stream);
  const std::int64_t required_blocks = (input.np + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(std::min<std::int64_t>(required_blocks, 65535));
  StressKernel<Func><<<blocks, kThreads, 0, stream>>>(
      input.rho, input.grad, input.w, stress_out.stress,
      input.np, input.point_stride);
  return CudaStatus(cudaGetLastError(), "StressKernel launch");
}

Status LaunchXcStress(const XcSpec& spec, const XcGridIn& input,
                      XcStressOut& stress_out, cudaStream_t stream) {
  if (spec.nspin != 1) {
    return Status::Unimplemented("Stress tensor supports unpolarized only.");
  }
  switch (spec.terms[0].functional) {
    case Functional::kLdaPw92:
      return LaunchStressKernel<LdaPw92Functor>(spec, input, stress_out, stream);
    case Functional::kSvwn5:
      return LaunchStressKernel<Svwn5Functor>(spec, input, stress_out, stream);
    case Functional::kPbe:
      return LaunchStressKernel<PbeFunctor>(spec, input, stress_out, stream);
    case Functional::kPbeSol:
      return LaunchStressKernel<PbeSolFunctor>(spec, input, stress_out, stream);
    case Functional::kRevPbe:
      return LaunchStressKernel<RevPbeFunctor>(spec, input, stress_out, stream);
    case Functional::kRpbe:
      return LaunchStressKernel<RpbeFunctor>(spec, input, stress_out, stream);
    case Functional::kBlyp:
      return LaunchStressKernel<BlypFunctor>(spec, input, stress_out, stream);
    default:
      return Status::Unimplemented(
          "Stress tensor not yet implemented for this functional.");
  }
}

Status LaunchXcFunctional(const XcSpec& spec, const XcGridIn& input,
                          XcGridOut& output, cudaStream_t stream) {
  if (spec.nspin == 2) {
    switch (spec.terms[0].functional) {
      case Functional::kLdaPw92:
      case Functional::kSvwn5:
      case Functional::kPbe:
      case Functional::kPbeSol:
      case Functional::kRevPbe:
      case Functional::kRpbe:
      case Functional::kBlyp:
      case Functional::kB3lyp:
      case Functional::kPbe0:
        return tier0::LaunchTier0Pol(spec, input, output, stream);
      default:
        break;
    }
  }
  switch (spec.terms[0].functional) {
    case Functional::kLdaPw92:
      return LaunchFunctionalKernel<LdaPw92Functor>(input, output, stream, spec.deterministic);
    case Functional::kSvwn5:
      return LaunchFunctionalKernel<Svwn5Functor>(input, output, stream, spec.deterministic);
    case Functional::kPbe:
      return LaunchFunctionalKernel<PbeFunctor>(input, output, stream, spec.deterministic);
    case Functional::kPbeSol:
      return LaunchFunctionalKernel<PbeSolFunctor>(input, output, stream, spec.deterministic);
    case Functional::kRevPbe:
      return LaunchFunctionalKernel<RevPbeFunctor>(input, output, stream, spec.deterministic);
    case Functional::kRpbe:
      return LaunchFunctionalKernel<RpbeFunctor>(input, output, stream, spec.deterministic);
    case Functional::kBlyp:
      return LaunchFunctionalKernel<BlypFunctor>(input, output, stream, spec.deterministic);
    case Functional::kB3lyp:
      return LaunchFunctionalKernel<B3lypFunctor>(input, output, stream, spec.deterministic);
    case Functional::kPbe0:
      return LaunchFunctionalKernel<Pbe0Functor>(input, output, stream, spec.deterministic);
    case Functional::kTpss:
      return tier1::LaunchMggaTpss(input, output, stream, spec.deterministic, spec.nspin);
    case Functional::kScan:
      return tier1::LaunchMggaScan(input, output, stream, spec.deterministic, spec.nspin);
    case Functional::kR2scan:
      return tier1::LaunchMggaR2scan(input, output, stream, spec.deterministic, spec.nspin);
    case Functional::kM06_2x:
      return tier1::LaunchMggaM06_2x(input, output, stream, spec.deterministic, spec.nspin);
    case Functional::kHse06:
      return tier1::LaunchRshHse06(input, output, stream, spec.deterministic, spec.nspin);
    case Functional::kWb97x:
      return tier1::LaunchRshWb97x(input, output, stream, spec.deterministic, spec.nspin);
    default:
      return Status::Unimplemented("Functional not yet implemented in Tier-0");
  }
}

}  // namespace tides::grid::xc
