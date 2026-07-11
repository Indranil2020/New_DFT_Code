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
#include "grid/rho_build_gpu.hpp"
#include "grid/dual_grid.hpp"
#include "grid/gpu_arena.hpp"

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

// Device-native reference implementation for the NAO product contract.  The
// future TileMat backend may replace this O(nbasis^2) loop, but it must retain
// these borrowed-pointer and stream semantics.
__global__ void RhoGradientDeviceKernel(
    const double* density_matrix, const double* phi, const double* grad_phi,
    double* rho, double* grad, std::int64_t nbasis, std::int64_t np,
    std::int64_t point_stride) {
  const std::int64_t point = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  if (point >= np) return;
  double density = 0.0;
  double gradient_x = 0.0;
  double gradient_y = 0.0;
  double gradient_z = 0.0;
  const std::int64_t basis_plane = nbasis * point_stride;
  for (std::int64_t mu = 0; mu < nbasis; ++mu) {
    const double phi_mu = phi[mu * point_stride + point];
    const double dphi_mu_x = grad_phi[mu * point_stride + point];
    const double dphi_mu_y = grad_phi[basis_plane + mu * point_stride + point];
    const double dphi_mu_z = grad_phi[2 * basis_plane + mu * point_stride + point];
    for (std::int64_t nu = 0; nu < nbasis; ++nu) {
      const double p_mu_nu = density_matrix[mu * nbasis + nu];
      const double phi_nu = phi[nu * point_stride + point];
      const double dphi_nu_x = grad_phi[nu * point_stride + point];
      const double dphi_nu_y = grad_phi[basis_plane + nu * point_stride + point];
      const double dphi_nu_z = grad_phi[2 * basis_plane + nu * point_stride + point];
      density += p_mu_nu * phi_mu * phi_nu;
      gradient_x += p_mu_nu * (dphi_mu_x * phi_nu + phi_mu * dphi_nu_x);
      gradient_y += p_mu_nu * (dphi_mu_y * phi_nu + phi_mu * dphi_nu_y);
      gradient_z += p_mu_nu * (dphi_mu_z * phi_nu + phi_mu * dphi_nu_z);
    }
  }
  rho[point] = density;
  grad[point] = gradient_x;
  grad[point_stride + point] = gradient_y;
  grad[2 * point_stride + point] = gradient_z;
}

// Device-native kinetic energy density build for mGGA.
// tau = (1/2) sum_mn P_mn (grad phi_m . grad phi_n)
__global__ void TauBuildDeviceKernel(
    const double* density_matrix, const double* grad_phi,
    double* tau, std::int64_t nbasis, std::int64_t np,
    std::int64_t point_stride) {
  const std::int64_t point = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  if (point >= np) return;
  double kinetic = 0.0;
  const std::int64_t basis_plane = nbasis * point_stride;
  for (std::int64_t mu = 0; mu < nbasis; ++mu) {
    const double dphi_mu_x = grad_phi[mu * point_stride + point];
    const double dphi_mu_y = grad_phi[basis_plane + mu * point_stride + point];
    const double dphi_mu_z = grad_phi[2 * basis_plane + mu * point_stride + point];
    for (std::int64_t nu = 0; nu < nbasis; ++nu) {
      const double p_mu_nu = density_matrix[mu * nbasis + nu];
      const double dphi_nu_x = grad_phi[nu * point_stride + point];
      const double dphi_nu_y = grad_phi[basis_plane + nu * point_stride + point];
      const double dphi_nu_z = grad_phi[2 * basis_plane + nu * point_stride + point];
      kinetic += p_mu_nu * (dphi_mu_x * dphi_nu_x +
                            dphi_mu_y * dphi_nu_y +
                            dphi_mu_z * dphi_nu_z);
    }
  }
  tau[point] = 0.5 * kinetic;
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

  // AUDIT B10: Use GPU arena for persistent device buffers and stream.
  // Previously: cudaMalloc/cudaFree/cudaStreamCreate per call → residency defect.
  GpuArena& arena = GpuArena::Instance();
  cudaStream_t stream = arena.Stream();

  // Allocate device memory via arena (reuses cached blocks).
  const std::size_t orb_bytes = flat_orbitals.size() * sizeof(double);
  double* d_orbitals = static_cast<double*>(arena.Alloc(orb_bytes));
  if (!d_orbitals) return Status::IoError("arena.Alloc failed for orbitals");

  const std::size_t occ_bytes = occupations.size() * sizeof(double);
  double* d_occupations = static_cast<double*>(arena.Alloc(occ_bytes));
  if (!d_occupations) { arena.Free(d_orbitals); return Status::IoError("arena.Alloc failed for occ"); }

  const std::size_t rho_bytes = n_points * sizeof(double);
  double* d_rho = static_cast<double*>(arena.Alloc(rho_bytes));
  if (!d_rho) { arena.Free(d_orbitals); arena.Free(d_occupations); return Status::IoError("arena.Alloc failed for rho"); }

  // Async H2D on arena stream.
  err = arena.H2D(d_orbitals, h_orbitals, orb_bytes);
  if (err != cudaSuccess) { arena.Free(d_orbitals); arena.Free(d_occupations); arena.Free(d_rho); cudaFreeHost(h_orbitals); cudaFreeHost(h_occupations); return CudaStatus(err, "arena.H2D orbitals"); }
  err = arena.H2D(d_occupations, h_occupations, occ_bytes);
  if (err != cudaSuccess) { arena.Free(d_orbitals); arena.Free(d_occupations); arena.Free(d_rho); cudaFreeHost(h_orbitals); cudaFreeHost(h_occupations); return CudaStatus(err, "arena.H2D occ"); }

  // Launch rho build kernel on the arena stream.
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
  if (err != cudaSuccess) { arena.Free(d_orbitals); arena.Free(d_occupations); arena.Free(d_rho); cudaFreeHost(h_orbitals); cudaFreeHost(h_occupations); return CudaStatus(err, "RhoBuildKernel"); }

  result.kernel_ms = static_cast<double>(kernel_ms_f);

  // Copy rho back via arena D2H.
  result.rho.resize(n_points);
  err = arena.D2H(result.rho.data(), d_rho, rho_bytes);
  if (err != cudaSuccess) { arena.Free(d_orbitals); arena.Free(d_occupations); arena.Free(d_rho); cudaFreeHost(h_orbitals); cudaFreeHost(h_occupations); return CudaStatus(err, "arena.D2H rho"); }

  // Compute integral on GPU via reduction.
  const auto [h0, h1, h2] = grid.h;
  const double dv = h0 * h1 * h2;
  const int red_threads = 256;
  const int red_blocks = static_cast<int>((n_points + red_threads - 1) / red_threads);
  std::vector<double> partial_sums(red_blocks, 0.0);
  double* d_partial = static_cast<double*>(arena.Alloc(red_blocks * sizeof(double)));
  if (!d_partial) { arena.Free(d_orbitals); arena.Free(d_occupations); arena.Free(d_rho); cudaFreeHost(h_orbitals); cudaFreeHost(h_occupations); return Status::IoError("arena.Alloc failed for partial"); }

  RhoIntegralKernel<<<red_blocks, red_threads, red_threads * sizeof(double), stream>>>(
      d_rho, n_points, dv, d_partial, static_cast<std::size_t>(red_blocks));
  err = arena.Sync();
  if (err != cudaSuccess) {
    arena.Free(d_partial); arena.Free(d_orbitals); arena.Free(d_occupations); arena.Free(d_rho);
    cudaFreeHost(h_orbitals); cudaFreeHost(h_occupations);
    return CudaStatus(err, "RhoIntegralKernel");
  }

  err = cudaMemcpy(partial_sums.data(), d_partial,
                   red_blocks * sizeof(double), cudaMemcpyDeviceToHost);

  // Return arena blocks to pool (no cudaFree).
  arena.Free(d_partial);
  arena.Free(d_orbitals);
  arena.Free(d_occupations);
  arena.Free(d_rho);
  cudaFreeHost(h_orbitals);
  cudaFreeHost(h_occupations);

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

Status BuildRhoGradientDevice(const RhoGradientDeviceIn& input, double* rho,
                              double* grad, cudaStream_t stream) {
  if (input.nbasis <= 0 || input.np < 0 || input.point_stride < input.np) {
    return Status::InvalidArgument(
        "rho/gradient device build requires nbasis > 0, np >= 0, and point_stride >= np");
  }
  if (input.np == 0) return Status::Ok();
  if (input.density_matrix == nullptr || input.phi == nullptr ||
      input.grad_phi == nullptr || rho == nullptr || grad == nullptr) {
    return Status::InvalidArgument(
        "rho/gradient device build requires non-null device pointers");
  }
  constexpr int kThreads = 128;
  const std::int64_t required_blocks = (input.np + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(std::min<std::int64_t>(required_blocks, 65535));
  RhoGradientDeviceKernel<<<blocks, kThreads, 0, stream>>>(
      input.density_matrix, input.phi, input.grad_phi, rho, grad, input.nbasis,
      input.np, input.point_stride);
  return CudaStatus(cudaGetLastError(), "RhoGradientDeviceKernel launch");
}

Status BuildTauDevice(const RhoGradientDeviceIn& input, double* tau,
                       cudaStream_t stream) {
  if (input.nbasis <= 0 || input.np < 0 || input.point_stride < input.np) {
    return Status::InvalidArgument(
        "tau device build requires nbasis > 0, np >= 0, and point_stride >= np");
  }
  if (input.np == 0) return Status::Ok();
  if (input.density_matrix == nullptr || input.grad_phi == nullptr ||
      tau == nullptr) {
    return Status::InvalidArgument(
        "tau device build requires non-null density_matrix, grad_phi, and tau");
  }
  constexpr int kThreads = 128;
  const std::int64_t required_blocks = (input.np + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(std::min<std::int64_t>(required_blocks, 65535));
  TauBuildDeviceKernel<<<blocks, kThreads, 0, stream>>>(
      input.density_matrix, input.grad_phi, tau, input.nbasis, input.np,
      input.point_stride);
  return CudaStatus(cudaGetLastError(), "TauBuildDeviceKernel launch");
}

}  // namespace tides::grid
