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
