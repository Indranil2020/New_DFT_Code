#include "grid/xc/kernels/xc_gga_kernel.hpp"

#include "grid/xc/functionals/gga_pbe.cuh"

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

__global__ void PbeGgaKernel(const double* __restrict__ rho,
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
    const double gx = grad[point];
    const double gy = grad[point_stride + point];
    const double gz = grad[2 * point_stride + point];
    const double sigma = gx * gx + gy * gy + gz * gz;
    const GgaEvaluation evaluation = GgaPbeStandard::Eval(rho[point], sigma);
    const double weight = weights[point];
    const double weighted_gradient = 2.0 * weight * evaluation.vsigma;
    wv_rho[point] = weight * evaluation.vrho;
    wv_grad[point] = weighted_gradient * gx;
    wv_grad[point_stride + point] = weighted_gradient * gy;
    wv_grad[2 * point_stride + point] = weighted_gradient * gz;
    if (fast_reduction) local_energy += weight * rho[point] * evaluation.eps;
  }
  if (!fast_reduction) return;
  for (int offset = 16; offset > 0; offset /= 2) {
    local_energy += __shfl_down_sync(0xffffffff, local_energy, offset);
  }
  if ((threadIdx.x & 31) == 0) atomicAdd(exc, local_energy);
}

__global__ void PbeGgaDeterministicEnergyKernel(
    const double* __restrict__ rho, const double* __restrict__ grad,
    const double* __restrict__ weights, double* __restrict__ exc,
    std::int64_t np, std::int64_t point_stride) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  double energy = 0.0;
  for (std::int64_t point = 0; point < np; ++point) {
    const double gx = grad[point];
    const double gy = grad[point_stride + point];
    const double gz = grad[2 * point_stride + point];
    const GgaEvaluation evaluation =
        GgaPbeStandard::Eval(rho[point], gx * gx + gy * gy + gz * gz);
    energy += weights[point] * rho[point] * evaluation.eps;
  }
  exc[0] = energy;
}

}  // namespace

Status LaunchPbeGgaKernel(const XcGridIn& input, XcGridOut& output,
                          cudaStream_t stream, bool deterministic) {
  constexpr int kThreads = 256;
  const std::int64_t required_blocks = (input.np + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(std::min<std::int64_t>(required_blocks, 65535));
  PbeGgaKernel<<<blocks, kThreads, 0, stream>>>(
      input.rho, input.grad, input.w, output.wv_rho, output.wv_grad,
      output.exc_per_system, input.np, input.point_stride, !deterministic);
  Status status = CudaStatus(cudaGetLastError(), "PbeGgaKernel launch");
  if (!status.ok() || !deterministic) return status;
  PbeGgaDeterministicEnergyKernel<<<1, 1, 0, stream>>>(
      input.rho, input.grad, input.w, output.exc_per_system, input.np,
      input.point_stride);
  return CudaStatus(cudaGetLastError(), "PbeGgaDeterministicEnergyKernel launch");
}

}  // namespace tides::grid::xc
