#pragma once

// Shared XC kernel utilities and reduction helpers.
//
// This header contains the templated point-evaluation kernels, the block/warp
// reduction machinery, and the function declarations used by the per-family
// launcher files (xc_lda_kernel.cu, xc_gga_kernel.cu, xc_mgga_kernel.cu).

#include "grid/xc/xc_engine.hpp"
#include "grid/xc/kernels/xc_gga_kernel.hpp"
#include "grid/xc/functionals/common.cuh"
#include "common/status.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <string>

namespace tides::grid::xc {

namespace kernels {

constexpr int kThreads = 256;

// Generic kernel for any LDA/GGA functor that returns a GgaEvaluation.
// Func::kFamily determines whether grad and wv_grad are touched.
template <class Func, bool kMultiSystem = false>
__global__ void FunctionalKernel(const double* __restrict__ rho,
                                 const double* __restrict__ grad,
                                 const double* __restrict__ weights,
                                 double* __restrict__ wv_rho,
                                 double* __restrict__ wv_grad,
                                 double* __restrict__ exc,
                                 const std::int64_t* __restrict__ sys_offsets,
                                 int nsys,
                                 std::int64_t np, std::int64_t point_stride,
                                 bool fast_reduction) {
  double local_energy = 0.0;
  std::int64_t local_sys = 0;
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
    if (fast_reduction) {
      if constexpr (kMultiSystem) {
        while (local_sys + 1 < nsys && point >= sys_offsets[local_sys + 1])
          ++local_sys;
      }
      local_energy += weight * density * evaluation.eps;
    }
  }
  if (!fast_reduction) return;
  if constexpr (kMultiSystem) {
    atomicAdd(exc + local_sys, local_energy);
    return;
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    local_energy += __shfl_down_sync(0xffffffff, local_energy, offset);
  }
  if ((threadIdx.x & 31) == 0) atomicAdd(exc, local_energy);
}

// Deterministic energy accumulator for single- or multi-system grids.
template <class Func>
__global__ void FunctionalDeterministicEnergyKernel(
    const double* __restrict__ rho, const double* __restrict__ grad,
    const double* __restrict__ weights, double* __restrict__ exc,
    const std::int64_t* __restrict__ sys_offsets,
    int nsys,
    std::int64_t np, std::int64_t point_stride) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  for (int s = 0; s < nsys; ++s) {
    const std::int64_t begin = sys_offsets[s];
    const std::int64_t end = sys_offsets[s + 1];
    double energy = 0.0;
    for (std::int64_t point = begin; point < end; ++point) {
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
    exc[s] = energy;
  }
}

// FP32 storage path kernel.
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
    const double density = static_cast<double>(rho[point]);
    const double weight = static_cast<double>(weights[point]);

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

// Stress-tensor kernel.
template <class Func>
__global__ void StressKernel(const double* __restrict__ rho,
                              const double* __restrict__ grad,
                              const double* __restrict__ weights,
                              double* __restrict__ stress,
                              std::int64_t np, std::int64_t point_stride) {
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
      const double isotropic = weight * density * evaluation.vrho;
      sdata[0 * kThreads + tid] += isotropic;
      sdata[1 * kThreads + tid] += isotropic;
      sdata[2 * kThreads + tid] += isotropic;
    } else {
      const double gx = grad[point];
      const double gy = grad[point_stride + point];
      const double gz = grad[2 * point_stride + point];
      const double sigma = gx * gx + gy * gy + gz * gz;
      evaluation = Func::Eval(density, sigma);
      const double isotropic = weight * density * evaluation.vrho;
      const double two_w_vs = 2.0 * weight * evaluation.vsigma;
      sdata[0 * kThreads + tid] += isotropic + two_w_vs * gx * gx;
      sdata[1 * kThreads + tid] += isotropic + two_w_vs * gy * gy;
      sdata[2 * kThreads + tid] += isotropic + two_w_vs * gz * gz;
      sdata[3 * kThreads + tid] += two_w_vs * gx * gy;
      sdata[4 * kThreads + tid] += two_w_vs * gx * gz;
      sdata[5 * kThreads + tid] += two_w_vs * gy * gz;
    }
  }

  __syncthreads();
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

// Launch helpers for the kernels above.

[[nodiscard]] inline Status CudaStatus(cudaError_t error, const char* context) {
  if (error == cudaSuccess) return Status::Ok();
  return Status::IoError(std::string(context) + ": " + cudaGetErrorString(error));
}

template <class Func>
[[nodiscard]] Status LaunchFunctionalKernel(const XcGridIn& input,
                                            XcGridOut& output,
                                            cudaStream_t stream,
                                            bool deterministic) {
  const std::int64_t* sys_offsets = input.sys_offsets;
  std::int64_t* d_default_sys_offsets = nullptr;
  if (sys_offsets == nullptr) {
    std::int64_t host_offsets[2] = {0, input.np};
    cudaError_t err = cudaMallocAsync(reinterpret_cast<void**>(&d_default_sys_offsets),
                                      sizeof(host_offsets), stream);
    if (err != cudaSuccess) return CudaStatus(err, "cudaMallocAsync default sys_offsets");
    err = cudaMemcpyAsync(d_default_sys_offsets, host_offsets, sizeof(host_offsets),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
      cudaFreeAsync(d_default_sys_offsets, stream);
      return CudaStatus(err, "cudaMemcpyAsync default sys_offsets");
    }
    sys_offsets = d_default_sys_offsets;
  }
  const std::int64_t required_blocks = (input.np + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(std::min<std::int64_t>(required_blocks, 65535));
  if (input.nsys > 1) {
    FunctionalKernel<Func, true><<<blocks, kThreads, 0, stream>>>(
        input.rho, input.grad, input.w, output.wv_rho, output.wv_grad,
        output.exc_per_system, sys_offsets, input.nsys,
        input.np, input.point_stride, !deterministic);
  } else {
    FunctionalKernel<Func, false><<<blocks, kThreads, 0, stream>>>(
        input.rho, input.grad, input.w, output.wv_rho, output.wv_grad,
        output.exc_per_system, sys_offsets, input.nsys,
        input.np, input.point_stride, !deterministic);
  }
  Status status = CudaStatus(cudaGetLastError(), "FunctionalKernel launch");
  if (!status.ok() || !deterministic) {
    if (d_default_sys_offsets) cudaFreeAsync(d_default_sys_offsets, stream);
    return status;
  }
  FunctionalDeterministicEnergyKernel<Func><<<1, 1, 0, stream>>>(
      input.rho, input.grad, input.w, output.exc_per_system,
      sys_offsets, input.nsys, input.np, input.point_stride);
  status = CudaStatus(cudaGetLastError(), "FunctionalDeterministicEnergyKernel launch");
  if (d_default_sys_offsets) cudaFreeAsync(d_default_sys_offsets, stream);
  return status;
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

}  // namespace kernels

// Per-family launch functions used by the dispatcher in xc_gga_kernel.cu.
Status LaunchLdaFunctional(const XcSpec& spec, const XcGridIn& input,
                           XcGridOut& output, cudaStream_t stream);
Status LaunchGgaFunctional(const XcSpec& spec, const XcGridIn& input,
                           XcGridOut& output, cudaStream_t stream);
Status LaunchMggaFunctional(const XcSpec& spec, const XcGridIn& input,
                            XcGridOut& output, cudaStream_t stream);
Status LaunchRshFunctional(const XcSpec& spec, const XcGridIn& input,
                           XcGridOut& output, cudaStream_t stream);

Status LaunchLdaFunctionalFp32(const XcSpec& spec, const XcGridInFp32& input,
                               XcGridOutFp32& output, cudaStream_t stream);
Status LaunchGgaFunctionalFp32(const XcSpec& spec, const XcGridInFp32& input,
                               XcGridOutFp32& output, cudaStream_t stream);

Status LaunchLdaStress(const XcSpec& spec, const XcGridIn& input,
                       XcStressOut& stress_out, cudaStream_t stream);
Status LaunchGgaStress(const XcSpec& spec, const XcGridIn& input,
                       XcStressOut& stress_out, cudaStream_t stream);

}  // namespace tides::grid::xc
