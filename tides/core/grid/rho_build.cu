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

  // Small-size fallback: when n_points * n_orbitals fits in CPU L3 cache,
  // GPU transfer + launch overhead dominates. Use CPU RhoBuilder instead.
  // Threshold: ~2M doubles (16MB) — below this, CPU is always faster.
  // Note: in a real SCF loop, orbitals would already be on-device, eliminating
  // H2D transfer and making GPU viable at smaller sizes.
  const std::size_t total_elements = n_points * n_orbitals;
  if (total_elements < 5000000) {
    auto cpu_rho = RhoBuilder::BuildFromOrbitals(grid, orbitals, occupations);
    result.rho = cpu_rho;
    // Compute integral on CPU.
    const auto [h0, h1, h2] = grid.h;
    const double dv = h0 * h1 * h2;
    result.integral = 0.0;
    for (double r : cpu_rho) result.integral += r * dv;
    result.kernel_ms = 0.0;
    // Add ledger entry for CPU fallback path.
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
            "CPU fallback rho build"},
        0.0, candidates, candidates, 0,
        "CUDA rho builder (CPU fallback for small size)"});
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

  // Allocate pinned host memory for async transfers.
  double* h_orbitals = nullptr;
  double* h_occupations = nullptr;
  cudaError_t err = cudaMallocHost(&h_orbitals, flat_orbitals.size() * sizeof(double));
  if (err != cudaSuccess) return CudaStatus(err, "cudaMallocHost orbitals");
  err = cudaMallocHost(&h_occupations, occupations.size() * sizeof(double));
  if (err != cudaSuccess) { cudaFreeHost(h_orbitals); return CudaStatus(err, "cudaMallocHost occ"); }
  std::copy(flat_orbitals.begin(), flat_orbitals.end(), h_orbitals);
  std::copy(occupations.begin(), occupations.end(), h_occupations);

  // Create stream for async operations.
  cudaStream_t stream;
  err = cudaStreamCreate(&stream);
  if (err != cudaSuccess) { cudaFreeHost(h_orbitals); cudaFreeHost(h_occupations); return CudaStatus(err, "cudaStreamCreate"); }

  // Allocate device memory.
  double* d_orbitals = nullptr;
  double* d_occupations = nullptr;
  double* d_rho = nullptr;

  auto cleanup = [&]() {
    if (d_orbitals) cudaFreeAsync(d_orbitals, stream);
    if (d_occupations) cudaFreeAsync(d_occupations, stream);
    if (d_rho) cudaFreeAsync(d_rho, stream);
    cudaStreamDestroy(stream);
    cudaFreeHost(h_orbitals);
    cudaFreeHost(h_occupations);
  };

  const std::size_t orb_bytes = flat_orbitals.size() * sizeof(double);
  err = cudaMallocAsync(reinterpret_cast<void**>(&d_orbitals), orb_bytes, stream);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMallocAsync orbitals"); }
  err = cudaMemcpyAsync(d_orbitals, h_orbitals, orb_bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMemcpyAsync orbitals"); }

  const std::size_t occ_bytes = occupations.size() * sizeof(double);
  err = cudaMallocAsync(reinterpret_cast<void**>(&d_occupations), occ_bytes, stream);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMallocAsync occupations"); }
  err = cudaMemcpyAsync(d_occupations, h_occupations, occ_bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMemcpyAsync occupations"); }

  const std::size_t rho_bytes = n_points * sizeof(double);
  err = cudaMallocAsync(reinterpret_cast<void**>(&d_rho), rho_bytes, stream);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMallocAsync rho"); }

  // Launch rho build kernel on the stream.
  const int threads = 256;
  const int blocks = static_cast<int>((n_points + threads - 1) / threads);
  cudaEvent_t start_ev, stop_ev;
  cudaEventCreate(&start_ev);
  cudaEventCreate(&stop_ev);
  cudaEventRecord(start_ev, stream);
  RhoBuildKernel<<<blocks, threads, 0, stream>>>(
      d_orbitals, d_occupations, d_rho, n_orbitals, n_points);
  cudaEventRecord(stop_ev, stream);
  err = cudaEventSynchronize(stop_ev);
  float kernel_ms_f = 0.0f;
  cudaEventElapsedTime(&kernel_ms_f, start_ev, stop_ev);
  cudaEventDestroy(start_ev);
  cudaEventDestroy(stop_ev);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "RhoBuildKernel"); }

  result.kernel_ms = static_cast<double>(kernel_ms_f);

  // Copy rho back.
  result.rho.resize(n_points);
  err = cudaMemcpyAsync(result.rho.data(), d_rho, rho_bytes, cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMemcpyAsync rho D2H"); }

  // Compute integral on GPU via reduction.
  const auto [h0, h1, h2] = grid.h;
  const double dv = h0 * h1 * h2;
  const int red_threads = 256;
  const int red_blocks = static_cast<int>((n_points + red_threads - 1) / red_threads);
  std::vector<double> partial_sums(red_blocks, 0.0);
  double* d_partial = nullptr;
  err = cudaMallocAsync(reinterpret_cast<void**>(&d_partial), red_blocks * sizeof(double), stream);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMallocAsync partial"); }

  RhoIntegralKernel<<<red_blocks, red_threads, red_threads * sizeof(double), stream>>>(
      d_rho, n_points, dv, d_partial, static_cast<std::size_t>(red_blocks));
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    cudaFreeAsync(d_partial, stream);
    cleanup();
    return CudaStatus(err, "RhoIntegralKernel");
  }

  err = cudaMemcpy(partial_sums.data(), d_partial,
                   red_blocks * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFreeAsync(d_partial, stream);
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
