#pragma once

// Generic polarized (nspin=2) mGGA kernel and deterministic energy kernel.
// Each mGGA functional provides a PolFunctor that returns an MggaPolEvaluation
// for a single point given rho[2], sigma[3], lapl[2], and tau[2].

#include "grid/xc/functionals/common.cuh"
#include "grid/xc/xc_engine.hpp"

#include <cuda_runtime.h>

namespace tides::grid::xc::tier1 {

// Pass 1: compute eps, vrho, vtau (scalar outputs + energy reduction).
// Split from gradient computation to reduce register pressure (design doc §5).
template <typename PolFunctor>
__global__ __launch_bounds__(128)
void MggaPolKernelScalar(const double* __restrict__ rho,
                              const double* __restrict__ grad,
                              const double* __restrict__ tau,
                              const double* __restrict__ weights,
                              double* __restrict__ wv_rho,
                              double* __restrict__ wv_tau,
                              double* __restrict__ exc,
                              std::int64_t np, std::int64_t point_stride,
                              bool fast_reduction) {
  PolFunctor pol;
  double local_energy = 0.0;
  for (std::int64_t point = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       point < np; point += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const double rho_up = rho[point];
    const double rho_down = rho[point_stride + point];
    const double weight = weights[point];
    const double gx_up = grad[point];
    const double gy_up = grad[point_stride + point];
    const double gz_up = grad[2 * point_stride + point];
    const double gx_down = grad[3 * point_stride + point];
    const double gy_down = grad[4 * point_stride + point];
    const double gz_down = grad[5 * point_stride + point];
    const double rho_in[2] = {rho_up, rho_down};
    const double sigma_in[3] = {
        gx_up * gx_up + gy_up * gy_up + gz_up * gz_up,
        gx_up * gx_down + gy_up * gy_down + gz_up * gz_down,
        gx_down * gx_down + gy_down * gy_down + gz_down * gz_down};
    const double lapl_in[2] = {0.0, 0.0};
    const double tau_in[2] = {tau[point], tau[point_stride + point]};

    MggaPolEvaluation evaluation = pol(rho_in, sigma_in, lapl_in, tau_in);

    wv_rho[point] = weight * evaluation.vrho[0];
    wv_rho[point_stride + point] = weight * evaluation.vrho[1];

    wv_tau[point] = weight * evaluation.vtau[0];
    wv_tau[point_stride + point] = weight * evaluation.vtau[1];

    if (fast_reduction) {
      local_energy += weight * (rho_up + rho_down) * evaluation.eps;
    }
  }
  if (!fast_reduction) return;
  for (int offset = 16; offset > 0; offset /= 2) {
    local_energy += __shfl_down_sync(0xffffffff, local_energy, offset);
  }
  if ((threadIdx.x & 31) == 0) atomicAdd(exc, local_energy);
}

// Pass 2: compute vsigma → wv_grad (gradient outputs only).
// Re-reads inputs but uses fewer registers than the fused kernel.
template <typename PolFunctor>
__global__ __launch_bounds__(128)
void MggaPolKernelGrad(const double* __restrict__ rho,
                              const double* __restrict__ grad,
                              const double* __restrict__ tau,
                              const double* __restrict__ weights,
                              double* __restrict__ wv_grad,
                              std::int64_t np, std::int64_t point_stride) {
  PolFunctor pol;
  for (std::int64_t point = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       point < np; point += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const double weight = weights[point];
    const double gx_up = grad[point];
    const double gy_up = grad[point_stride + point];
    const double gz_up = grad[2 * point_stride + point];
    const double gx_down = grad[3 * point_stride + point];
    const double gy_down = grad[4 * point_stride + point];
    const double gz_down = grad[5 * point_stride + point];
    const double rho_up = rho[point];
    const double rho_down = rho[point_stride + point];
    const double rho_in[2] = {rho_up, rho_down};
    const double sigma_in[3] = {
        gx_up * gx_up + gy_up * gy_up + gz_up * gz_up,
        gx_up * gx_down + gy_up * gy_down + gz_up * gz_down,
        gx_down * gx_down + gy_down * gy_down + gz_down * gz_down};
    const double lapl_in[2] = {0.0, 0.0};
    const double tau_in[2] = {tau[point], tau[point_stride + point]};

    MggaPolEvaluation evaluation = pol(rho_in, sigma_in, lapl_in, tau_in);

    const double wv0 = 2.0 * weight * evaluation.vsigma[0];
    const double wv1 = weight * evaluation.vsigma[1];
    const double wv2 = 2.0 * weight * evaluation.vsigma[2];
    wv_grad[point] = wv0 * gx_up + wv1 * gx_down;
    wv_grad[point_stride + point] = wv0 * gy_up + wv1 * gy_down;
    wv_grad[2 * point_stride + point] = wv0 * gz_up + wv1 * gz_down;
    wv_grad[3 * point_stride + point] = wv1 * gx_up + wv2 * gx_down;
    wv_grad[4 * point_stride + point] = wv1 * gy_up + wv2 * gy_down;
    wv_grad[5 * point_stride + point] = wv1 * gz_up + wv2 * gz_down;
  }
}

template <typename PolFunctor>
__global__ void MggaPolDeterministicEnergyKernel(const double* __restrict__ rho,
                                                 const double* __restrict__ grad,
                                                 const double* __restrict__ tau,
                                                 const double* __restrict__ weights,
                                                 double* __restrict__ exc,
                                                 std::int64_t np,
                                                 std::int64_t point_stride) {
  PolFunctor pol;
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  double energy = 0.0;
  for (std::int64_t point = 0; point < np; ++point) {
    const double rho_up = rho[point];
    const double rho_down = rho[point_stride + point];
    const double gx_up = grad[point];
    const double gy_up = grad[point_stride + point];
    const double gz_up = grad[2 * point_stride + point];
    const double gx_down = grad[3 * point_stride + point];
    const double gy_down = grad[4 * point_stride + point];
    const double gz_down = grad[5 * point_stride + point];
    const double rho_in[2] = {rho_up, rho_down};
    const double sigma_in[3] = {
        gx_up * gx_up + gy_up * gy_up + gz_up * gz_up,
        gx_up * gx_down + gy_up * gy_down + gz_up * gz_down,
        gx_down * gx_down + gy_down * gy_down + gz_down * gz_down};
    const double lapl_in[2] = {0.0, 0.0};
    const double tau_in[2] = {tau[point], tau[point_stride + point]};

    MggaPolEvaluation evaluation = pol(rho_in, sigma_in, lapl_in, tau_in);
    energy += weights[point] * (rho_up + rho_down) * evaluation.eps;
  }
  exc[0] = energy;
}

}  // namespace tides::grid::xc::tier1
