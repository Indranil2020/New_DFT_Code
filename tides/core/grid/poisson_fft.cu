// T3.4: GPU Poisson solver via cuFFT for periodic boundary conditions.
//
// Solves nabla^2 V = -4*pi*rho using FFT:
//   1. Forward FFT: rho(r) -> rho(k)
//   2. Apply Green's function: V(k) = 4*pi*rho(k) / k^2  (k=0 -> 0)
//   3. Inverse FFT: V(k) -> V(r)
//
// Observable (T3.4): Gaussian-charge analytic <= 1e-10 Ha under all four BCs.
// This implementation handles periodic BC; free BC uses the CPU direct Coulomb
// sum (which is the FP64 oracle). The GPU cuFFT path replaces the O(N^2) naive
// DFT with O(N log N).

#include "grid/poisson.hpp"
#include "grid/dual_grid.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <array>
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
  return Status::IoError(std::string(context) + " (CUDA): " +
                         cudaGetErrorString(error));
}

[[nodiscard]] Status CufftStatus(cufftResult error, const char* context) {
  if (error == CUFFT_SUCCESS) {
    return Status::Ok();
  }
  return Status::IoError(std::string(context) + " (cuFFT): " +
                         std::to_string(error));
}

// Kernel: apply Green's function in reciprocal space.
// V(k) = 4*pi*rho(k) / k^2 for k != 0, V(0) = 0.
__global__ void ApplyGreensFunctionKernel(
    cufftDoubleComplex* V_k,
    const cufftDoubleComplex* rho_k,
    const double* kx_sq,
    const double* ky_sq,
    const double* kz_sq,
    int n0, int n1, int n2) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n0 * n1 * n2;
  if (idx >= total) return;
  const int iz = idx / (n0 * n1);
  const int rem = idx % (n0 * n1);
  const int iy = rem / n0;
  const int ix = rem % n0;
  const double k2 = kx_sq[ix] + ky_sq[iy] + kz_sq[iz];
  if (k2 < 1e-30) {
    V_k[idx].x = 0.0;
    V_k[idx].y = 0.0;
  } else {
    const double factor = 4.0 * M_PI / k2;
    V_k[idx].x = rho_k[idx].x * factor;
    V_k[idx].y = rho_k[idx].y * factor;
  }
}

}  // namespace

struct PoissonFftGpuResult {
  std::vector<double> V;
  double hartree_energy = 0.0;
  double kernel_ms = 0.0;
  std::size_t grid_size = 0;
  tides::tile::OperationLedger ledger;
};

[[nodiscard]] bool PoissonFftCudaAvailable() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

[[nodiscard]] Result<PoissonFftGpuResult> PoissonFftCuda(
    const UniformGrid3D& grid,
    const std::vector<double>& rho) {
  const std::size_t N = grid.total_points();
  PoissonFftGpuResult result;
  result.grid_size = N;

  if (N == 0) {
    return result;
  }

  if (rho.size() != N) {
    return Status::InvalidArgument("rho size mismatch with grid");
  }

  if (!PoissonFftCudaAvailable()) {
    return Status::IoError("CUDA runtime not available for Poisson FFT");
  }

  const auto [n0, n1, n2] = grid.n;
  const auto [h0, h1, h2] = grid.h;
  const auto [L0, L1, L2] = grid.cell_size();
  const double dv = h0 * h1 * h2;

  // Prepare k-space squared magnitudes for each axis.
  std::vector<double> kx_sq(n0), ky_sq(n1), kz_sq(n2);
  for (std::size_t i = 0; i < n0; ++i) {
    double fx = static_cast<double>(i);
    if (fx > n0 / 2.0) fx -= static_cast<double>(n0);
    const double kx = 2.0 * M_PI * fx / L0;
    kx_sq[i] = kx * kx;
  }
  for (std::size_t i = 0; i < n1; ++i) {
    double fy = static_cast<double>(i);
    if (fy > n1 / 2.0) fy -= static_cast<double>(n1);
    const double ky = 2.0 * M_PI * fy / L1;
    ky_sq[i] = ky * ky;
  }
  for (std::size_t i = 0; i < n2; ++i) {
    double fz = static_cast<double>(i);
    if (fz > n2 / 2.0) fz -= static_cast<double>(n2);
    const double kz = 2.0 * M_PI * fz / L2;
    kz_sq[i] = kz * kz;
  }

  // Prepare complex input: rho scaled by dv (DFT convention).
  std::vector<cufftDoubleComplex> host_rho_k(N);
  for (std::size_t i = 0; i < N; ++i) {
    host_rho_k[i].x = rho[i] * dv;
    host_rho_k[i].y = 0.0;
  }

  // Allocate device memory.
  cufftDoubleComplex* d_rho_k = nullptr;
  cufftDoubleComplex* d_V_k = nullptr;
  double* d_kx_sq = nullptr;
  double* d_ky_sq = nullptr;
  double* d_kz_sq = nullptr;
  cufftHandle plan_fwd = 0, plan_inv = 0;

  auto cleanup = [&]() {
    if (d_rho_k) cudaFree(d_rho_k);
    if (d_V_k) cudaFree(d_V_k);
    if (d_kx_sq) cudaFree(d_kx_sq);
    if (d_ky_sq) cudaFree(d_ky_sq);
    if (d_kz_sq) cudaFree(d_kz_sq);
    if (plan_fwd) cufftDestroy(plan_fwd);
    if (plan_inv) cufftDestroy(plan_inv);
  };

  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_rho_k),
                                N * sizeof(cufftDoubleComplex));
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMalloc rho_k"); }
  err = cudaMemcpy(d_rho_k, host_rho_k.data(), N * sizeof(cufftDoubleComplex),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMemcpy rho_k"); }

  err = cudaMalloc(reinterpret_cast<void**>(&d_V_k), N * sizeof(cufftDoubleComplex));
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMalloc V_k"); }

  err = cudaMalloc(reinterpret_cast<void**>(&d_kx_sq), n0 * sizeof(double));
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMalloc kx_sq"); }
  err = cudaMemcpy(d_kx_sq, kx_sq.data(), n0 * sizeof(double), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMemcpy kx_sq"); }
  err = cudaMalloc(reinterpret_cast<void**>(&d_ky_sq), n1 * sizeof(double));
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMalloc ky_sq"); }
  err = cudaMemcpy(d_ky_sq, ky_sq.data(), n1 * sizeof(double), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMemcpy ky_sq"); }
  err = cudaMalloc(reinterpret_cast<void**>(&d_kz_sq), n2 * sizeof(double));
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMalloc kz_sq"); }
  err = cudaMemcpy(d_kz_sq, kz_sq.data(), n2 * sizeof(double), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMemcpy kz_sq"); }

  // Create C2C FFT plans (Z2Z for double-precision complex-to-complex).
  cufftResult cufft_err = cufftPlan3d(&plan_fwd,
      static_cast<int>(n2), static_cast<int>(n1), static_cast<int>(n0),
      CUFFT_Z2Z);
  if (cufft_err != CUFFT_SUCCESS) {
    cleanup();
    return CufftStatus(cufft_err, "cufftPlan3d Z2Z fwd");
  }
  cufft_err = cufftPlan3d(&plan_inv,
      static_cast<int>(n2), static_cast<int>(n1), static_cast<int>(n0),
      CUFFT_Z2Z);
  if (cufft_err != CUFFT_SUCCESS) {
    cleanup();
    return CufftStatus(cufft_err, "cufftPlan3d Z2Z inv");
  }

  auto kernel_start = std::chrono::steady_clock::now();

  // Forward FFT: rho(r) -> rho(k).
  cufft_err = cufftExecZ2Z(plan_fwd, d_rho_k, d_rho_k, CUFFT_FORWARD);
  if (cufft_err != CUFFT_SUCCESS) {
    cleanup();
    return CufftStatus(cufft_err, "cufftExecZ2Z fwd");
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "sync after fwd Z2Z"); }

  // Apply Green's function: V(k) = 4*pi*rho(k) / k^2.
  const int total = static_cast<int>(N);
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  ApplyGreensFunctionKernel<<<blocks, threads>>>(
      d_V_k, d_rho_k, d_kx_sq, d_ky_sq, d_kz_sq,
      static_cast<int>(n0), static_cast<int>(n1), static_cast<int>(n2));
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "sync after Green's"); }

  // Inverse FFT: V(k) -> V(r).
  cufft_err = cufftExecZ2Z(plan_inv, d_V_k, d_rho_k, CUFFT_INVERSE);
  if (cufft_err != CUFFT_SUCCESS) {
    cleanup();
    return CufftStatus(cufft_err, "cufftExecZ2Z inv");
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "sync after inv Z2Z"); }

  auto kernel_end = std::chrono::steady_clock::now();
  result.kernel_ms =
      std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();

  // Copy result back.
  std::vector<cufftDoubleComplex> host_V_k(N);
  err = cudaMemcpy(host_V_k.data(), d_rho_k, N * sizeof(cufftDoubleComplex),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMemcpy V D2H"); }

  cleanup();

  // Normalize: cuFFT does NOT normalize the inverse transform.
  // Our DFT convention:
  //   rho_k = sum_j rho_j * exp(-ikr) * dv  (pre-scaled by dv)
  //   V_r = sum_k V_k * exp(+ikr) / V_cell
  // cuFFT fwd (no scaling): sum_j input_j * exp(-ikr) = dv * DFT(rho)
  // cuFFT inv (no scaling): sum_k V_k * exp(+ikr)
  // We want (1/V_cell) * sum_k V_k * exp(+ikr) = cuFFT_inv / V_cell.
  // V_cell = N * dv.
  result.V.resize(N);
  const double normalize = 1.0 / (static_cast<double>(N) * dv);
  for (std::size_t i = 0; i < N; ++i) {
    result.V[i] = host_V_k[i].x * normalize;
  }

  // Compute Hartree energy: E_H = 0.5 * integral rho * V * dv.
  double E = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    E += rho[i] * result.V[i] * dv;
  }
  result.hartree_energy = 0.5 * E;

  // Ledger.
  tides::tile::PrecisionDescriptor desc;
  desc.storage = tides::tile::NumericFormat::kFloat64;
  desc.compute = tides::tile::NumericFormat::kFloat64;
  desc.reduction = tides::tile::NumericFormat::kFloat64;
  desc.determinism = tides::tile::DeterminismMode::kDeterministic;
  desc.label = "cuda-poisson-fft";
  result.ledger.Add(tides::tile::OperationLedgerEntry{
      tides::tile::OperationKind::kPoissonSolve,
      desc,
      tides::tile::ErrorBudget{tides::tile::ErrorMetric::kAbsolute, 0.0,
          "GPU cuFFT Poisson vs CPU reference"},
      0.0, static_cast<std::uint64_t>(N), static_cast<std::uint64_t>(N), 0,
      "CUDA cuFFT periodic Poisson solver"});

  return result;
}

}  // namespace tides::grid
