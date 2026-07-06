// T3.2: GPU rho builder — density from orbital products on the fine grid.
//
// rho(r) = sum_k f_k * |orb_k(r)|^2
//
// The kernel processes grid points in parallel: each thread computes the
// density contribution at one grid point by iterating over occupied orbitals.
// This is the #1 memory-bound kernel — the GPU path enables tile-batched
// orbital products for large systems.
//
// Observables (T3.2):
//   (1) vs CPU <= 1e-9
//   (2) integral(n) = N_e <= 1e-10
//   (3) >= 60% HBM roofline on RTX (throughput recorded)

#include "grid/rho_build.hpp"
#include "grid/dual_grid.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/status.hpp"
#include "tile/precision.hpp"

namespace tides::grid {
namespace {

[[nodiscard]] Status CudaStatus(cudaError_t error, const char* context) {
  if (error == cudaSuccess) {
    return Status::Ok();
  }
  return Status::IoError(std::string(context) + ": " +
                         cudaGetErrorString(error));
}

// Kernel: each thread processes one grid point, iterating over all orbitals.
// rho[point] = sum_k f_k * orb_k[point]^2
__global__ void RhoBuildKernel(
    const double* orbitals,  // [n_orbitals][n_points] row-major
    const double* occupations,  // [n_orbitals]
    double* rho,  // [n_points]
    std::size_t n_orbitals,
    std::size_t n_points) {
  const std::size_t point = static_cast<std::size_t>(blockIdx.x) *
                                blockDim.x + threadIdx.x;
  if (point >= n_points) return;
  double sum = 0.0;
  for (std::size_t k = 0; k < n_orbitals; ++k) {
    const double psi = orbitals[k * n_points + point];
    sum += occupations[k] * psi * psi;
  }
  rho[point] = sum;
}

// Kernel: integrate density on the grid. Each thread accumulates a partial sum;
// a final reduction on the host completes the integral.
__global__ void RhoIntegralKernel(
    const double* rho,
    std::size_t n_points,
    double dv,
    double* partial_sums,
    std::size_t n_blocks) {
  extern __shared__ double sdata[];
  const std::size_t tid = threadIdx.x;
  const std::size_t bid = blockIdx.x;
  const std::size_t block_size = blockDim.x;

  double sum = 0.0;
  for (std::size_t i = bid * block_size + tid; i < n_points;
       i += gridDim.x * block_size) {
    sum += rho[i];
  }
  sdata[tid] = sum;
  __syncthreads();

  // Tree reduction.
  for (std::size_t s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_sums[bid] = sdata[0] * dv;
  }
}

}  // namespace

struct RhoBuildGpuResult {
  std::vector<double> rho;
  double integral = 0.0;
  double kernel_ms = 0.0;
  std::size_t n_points = 0;
  std::size_t n_orbitals = 0;
  tides::tile::OperationLedger ledger;
};

[[nodiscard]] bool RhoBuildCudaAvailable() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

[[nodiscard]] Result<RhoBuildGpuResult> RhoBuildCuda(
    const UniformGrid3D& grid,
    const std::vector<std::vector<double>>& orbitals,
    const std::vector<double>& occupations) {
  const std::size_t n_points = grid.total_points();
  const std::size_t n_orbitals = orbitals.size();

  RhoBuildGpuResult result;
  result.n_points = n_points;
  result.n_orbitals = n_orbitals;

  if (n_points == 0 || n_orbitals == 0) {
    result.rho.assign(n_points, 0.0);
    return result;
  }

  if (!RhoBuildCudaAvailable()) {
    return Status::IoError("CUDA runtime not available for rho builder");
  }

  // Flatten orbitals into a single contiguous array [n_orbitals * n_points].
  std::vector<double> flat_orbitals(n_orbitals * n_points);
  for (std::size_t k = 0; k < n_orbitals; ++k) {
    if (orbitals[k].size() != n_points) {
      return Status::InvalidArgument("orbital size mismatch with grid");
    }
    std::copy(orbitals[k].begin(), orbitals[k].end(),
              flat_orbitals.begin() + k * n_points);
  }

  // Allocate device memory.
  double* d_orbitals = nullptr;
  double* d_occupations = nullptr;
  double* d_rho = nullptr;

  auto cleanup = [&]() {
    if (d_orbitals) cudaFree(d_orbitals);
    if (d_occupations) cudaFree(d_occupations);
    if (d_rho) cudaFree(d_rho);
  };

  const std::size_t orb_bytes = flat_orbitals.size() * sizeof(double);
  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_orbitals), orb_bytes);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMalloc orbitals"); }
  err = cudaMemcpy(d_orbitals, flat_orbitals.data(), orb_bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMemcpy orbitals"); }

  const std::size_t occ_bytes = occupations.size() * sizeof(double);
  err = cudaMalloc(reinterpret_cast<void**>(&d_occupations), occ_bytes);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMalloc occupations"); }
  err = cudaMemcpy(d_occupations, occupations.data(), occ_bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMemcpy occupations"); }

  const std::size_t rho_bytes = n_points * sizeof(double);
  err = cudaMalloc(reinterpret_cast<void**>(&d_rho), rho_bytes);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMalloc rho"); }

  // Launch rho build kernel.
  const int threads = 256;
  const int blocks = static_cast<int>((n_points + threads - 1) / threads);
  auto kernel_start = std::chrono::steady_clock::now();
  RhoBuildKernel<<<blocks, threads>>>(
      d_orbitals, d_occupations, d_rho, n_orbitals, n_points);
  err = cudaDeviceSynchronize();
  auto kernel_end = std::chrono::steady_clock::now();
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "RhoBuildKernel"); }

  result.kernel_ms =
      std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();

  // Copy rho back.
  result.rho.resize(n_points);
  err = cudaMemcpy(result.rho.data(), d_rho, rho_bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMemcpy rho D2H"); }

  // Compute integral on GPU via reduction.
  const auto [h0, h1, h2] = grid.h;
  const double dv = h0 * h1 * h2;
  const int red_threads = 256;
  const int red_blocks = static_cast<int>((n_points + red_threads - 1) / red_threads);
  std::vector<double> partial_sums(red_blocks, 0.0);
  double* d_partial = nullptr;
  err = cudaMalloc(reinterpret_cast<void**>(&d_partial), red_blocks * sizeof(double));
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMalloc partial"); }

  RhoIntegralKernel<<<red_blocks, red_threads, red_threads * sizeof(double)>>>(
      d_rho, n_points, dv, d_partial, static_cast<std::size_t>(red_blocks));
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    cudaFree(d_partial);
    cleanup();
    return CudaStatus(err, "RhoIntegralKernel");
  }

  err = cudaMemcpy(partial_sums.data(), d_partial,
                   red_blocks * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_partial);
  cleanup();
  if (err != cudaSuccess) { return CudaStatus(err, "cudaMemcpy partial D2H"); }

  result.integral = 0.0;
  for (double s : partial_sums) result.integral += s;

  // Ledger.
  tides::tile::PrecisionDescriptor desc;
  desc.storage = tides::tile::NumericFormat::kFloat64;
  desc.compute = tides::tile::NumericFormat::kFloat64;
  desc.reduction = tides::tile::NumericFormat::kFloat64;
  desc.determinism = tides::tile::DeterminismMode::kDeterministic;
  desc.label = "cuda-rho-build";
  const std::uint64_t candidates =
      static_cast<std::uint64_t>(n_orbitals) * n_points;
  result.ledger.Add(tides::tile::OperationLedgerEntry{
      tides::tile::OperationKind::kReduction,
      desc,
      tides::tile::ErrorBudget{tides::tile::ErrorMetric::kAbsolute, 0.0,
          "GPU rho build vs CPU reference"},
      0.0, candidates, candidates, 0,
      "CUDA rho builder (orbital products on fine grid)"});

  return result;
}

}  // namespace tides::grid
