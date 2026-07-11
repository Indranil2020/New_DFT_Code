#include "grid/xc/functionals/mgga_tpss.cuh"
#include "grid/xc/tier1/mgga_pol_kernel.cuh"
#include "grid/xc/xc_engine.hpp"
#include "common/status.hpp"

#include <algorithm>
#include <cuda_runtime.h>

namespace tides::grid::xc::tier1 {

namespace {

using tides::grid::xc::MggaTpssFunctor;
using tides::grid::xc::MggaTpssPolFunctor;

constexpr int kThreads = 128;

__global__ __launch_bounds__(128)
void MggaTpssKernel(const double* __restrict__ rho,
                               const double* __restrict__ grad,
                               const double* __restrict__ tau,
                               const double* __restrict__ weights,
                               double* __restrict__ wv_rho,
                               double* __restrict__ wv_grad,
                               double* __restrict__ wv_tau,
                               double* __restrict__ exc,
                               std::int64_t np, std::int64_t point_stride,
                               bool fast_reduction) {
  double local_energy = 0.0;
  for (std::int64_t point = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       point < np; point += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const double density = rho[point];
    const double weight = weights[point];
    const double gx = grad[point];
    const double gy = grad[point_stride + point];
    const double gz = grad[2 * point_stride + point];
    const double sigma = gx * gx + gy * gy + gz * gz;
    MggaEvaluation evaluation = MggaTpssFunctor::Eval(density, sigma, tau[point]);
    const double weighted_gradient = 2.0 * weight * evaluation.vsigma;
    wv_grad[point] = weighted_gradient * gx;
    wv_grad[point_stride + point] = weighted_gradient * gy;
    wv_grad[2 * point_stride + point] = weighted_gradient * gz;
    wv_rho[point] = weight * evaluation.vrho;
    wv_tau[point] = weight * evaluation.vtau;
    if (fast_reduction) local_energy += weight * density * evaluation.eps;
  }
  if (!fast_reduction) return;
  for (int offset = 16; offset > 0; offset /= 2) {
    local_energy += __shfl_down_sync(0xffffffff, local_energy, offset);
  }
  if ((threadIdx.x & 31) == 0) atomicAdd(exc, local_energy);
}

__global__ void MggaTpssDeterministicEnergyKernel(const double* __restrict__ rho,
                                                  const double* __restrict__ grad,
                                                  const double* __restrict__ tau,
                                                  const double* __restrict__ weights,
                                                  double* __restrict__ exc,
                                                  std::int64_t np,
                                                  std::int64_t point_stride) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  double energy = 0.0;
  for (std::int64_t point = 0; point < np; ++point) {
    const double density = rho[point];
    const double gx = grad[point];
    const double gy = grad[point_stride + point];
    const double gz = grad[2 * point_stride + point];
    const double sigma = gx * gx + gy * gy + gz * gz;
    MggaEvaluation evaluation = MggaTpssFunctor::Eval(density, sigma, tau[point]);
    energy += weights[point] * density * evaluation.eps;
  }
  exc[0] = energy;
}

}  // namespace

Status LaunchMggaTpss(const XcGridIn& input, XcGridOut& output,
                      cudaStream_t stream, bool deterministic, int nspin) {
  const std::int64_t required_blocks = (input.np + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(std::min<std::int64_t>(required_blocks, 65535));
  if (nspin == 2) {
    MggaPolKernelScalar<MggaTpssPolFunctor><<<blocks, kThreads, 0, stream>>>(
        input.rho, input.grad, input.tau, input.w, output.wv_rho,
        output.wv_tau, output.exc_per_system, input.np,
        input.point_stride, !deterministic);
    if (cudaGetLastError() != cudaSuccess) return Status::IoError("MggaTpssPolKernelScalar launch");
    MggaPolKernelGrad<MggaTpssPolFunctor><<<blocks, kThreads, 0, stream>>>(
        input.rho, input.grad, input.tau, input.w, output.wv_grad,
        input.np, input.point_stride);
    if (cudaGetLastError() != cudaSuccess) return Status::IoError("MggaTpssPolKernelGrad launch");
    if (!deterministic) return Status::Ok();
    MggaPolDeterministicEnergyKernel<MggaTpssPolFunctor><<<1, 1, 0, stream>>>(
        input.rho, input.grad, input.tau, input.w, output.exc_per_system,
        input.np, input.point_stride);
    if (cudaGetLastError() != cudaSuccess) return Status::IoError("MggaTpssPolDeterministicEnergyKernel launch");
    return Status::Ok();
  }
  MggaTpssKernel<<<blocks, kThreads, 0, stream>>>(
      input.rho, input.grad, input.tau, input.w, output.wv_rho,
      output.wv_grad, output.wv_tau, output.exc_per_system, input.np,
      input.point_stride, !deterministic);
  if (cudaGetLastError() != cudaSuccess) return Status::IoError("MggaTpssKernel launch");
  if (!deterministic) return Status::Ok();
  MggaTpssDeterministicEnergyKernel<<<1, 1, 0, stream>>>(
      input.rho, input.grad, input.tau, input.w, output.exc_per_system,
      input.np, input.point_stride);
  if (cudaGetLastError() != cudaSuccess) return Status::IoError("MggaTpssDeterministicEnergyKernel launch");
  return Status::Ok();
}

}  // namespace tides::grid::xc::tier1
