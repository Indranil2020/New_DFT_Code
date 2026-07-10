#pragma once

// Generic polarized (nspin=2) LDA and GGA kernels for Tier-0 functionals.
// Each functional provides a PolFunctor with operator()(rho[2]) for LDA or
// operator()(rho[2], sigma[3]) for GGA, returning LdaPolEvaluation or
// GgaPolEvaluation respectively.

#include "grid/xc/functionals/common.cuh"
#include "grid/xc/xc_engine.hpp"

#include <cuda_runtime.h>

namespace tides::grid::xc::tier0 {

// ---------------------------------------------------------------------------
// LDA polarized kernels
// ---------------------------------------------------------------------------

template <typename PolFunctor>
__global__ void LdaPolKernel(const double* __restrict__ rho,
                             const double* __restrict__ weights,
                             double* __restrict__ wv_rho,
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
    const double rho_in[2] = {rho_up, rho_down};

    LdaPolEvaluation evaluation = pol(rho_in);

    wv_rho[point] = weight * evaluation.vrho[0];
    wv_rho[point_stride + point] = weight * evaluation.vrho[1];

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

template <typename PolFunctor>
__global__ void LdaPolDeterministicEnergyKernel(
    const double* __restrict__ rho,
    const double* __restrict__ weights,
    double* __restrict__ exc,
    std::int64_t np, std::int64_t point_stride) {
  PolFunctor pol;
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  double energy = 0.0;
  for (std::int64_t point = 0; point < np; ++point) {
    const double rho_up = rho[point];
    const double rho_down = rho[point_stride + point];
    const double rho_in[2] = {rho_up, rho_down};
    LdaPolEvaluation evaluation = pol(rho_in);
    energy += weights[point] * (rho_up + rho_down) * evaluation.eps;
  }
  exc[0] = energy;
}

// ---------------------------------------------------------------------------
// GGA polarized kernels
// ---------------------------------------------------------------------------

template <typename PolFunctor>
__global__ void GgaPolKernel(const double* __restrict__ rho,
                             const double* __restrict__ grad,
                             const double* __restrict__ weights,
                             double* __restrict__ wv_rho,
                             double* __restrict__ wv_grad,
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

    GgaPolEvaluation evaluation = pol(rho_in, sigma_in);

    const double wv0 = 2.0 * weight * evaluation.vsigma[0];
    const double wv1 = weight * evaluation.vsigma[1];
    const double wv2 = 2.0 * weight * evaluation.vsigma[2];
    wv_grad[point] = wv0 * gx_up + wv1 * gx_down;
    wv_grad[point_stride + point] = wv0 * gy_up + wv1 * gy_down;
    wv_grad[2 * point_stride + point] = wv0 * gz_up + wv1 * gz_down;
    wv_grad[3 * point_stride + point] = wv1 * gx_up + wv2 * gx_down;
    wv_grad[4 * point_stride + point] = wv1 * gy_up + wv2 * gy_down;
    wv_grad[5 * point_stride + point] = wv1 * gz_up + wv2 * gz_down;

    wv_rho[point] = weight * evaluation.vrho[0];
    wv_rho[point_stride + point] = weight * evaluation.vrho[1];

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

template <typename PolFunctor>
__global__ void GgaPolDeterministicEnergyKernel(
    const double* __restrict__ rho,
    const double* __restrict__ grad,
    const double* __restrict__ weights,
    double* __restrict__ exc,
    std::int64_t np, std::int64_t point_stride) {
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
    GgaPolEvaluation evaluation = pol(rho_in, sigma_in);
    energy += weights[point] * (rho_up + rho_down) * evaluation.eps;
  }
  exc[0] = energy;
}

}  // namespace tides::grid::xc::tier0
