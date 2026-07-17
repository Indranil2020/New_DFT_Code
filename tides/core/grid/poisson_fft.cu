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
#include "grid/poisson_fft_gpu.hpp"
#include "grid/dual_grid.hpp"
#include "grid/gpu_arena.hpp"

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

// Kernel: pointwise complex multiplication for FFT-based convolution.
// V(k) = a(k) * b(k) (complex multiply).
__global__ void ComplexMultiplyKernel(
    cufftDoubleComplex* out,
    const cufftDoubleComplex* a,
    const cufftDoubleComplex* b,
    int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const double ar = a[idx].x, ai = a[idx].y;
  const double br = b[idx].x, bi = b[idx].y;
  out[idx].x = ar * br - ai * bi;
  out[idx].y = ar * bi + ai * br;
}

}  // namespace

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

  // Small-size fallback: for grids ≤ 32³ (32768 points), CPU FFTW is faster
  // than GPU cuFFT due to CUDA context + cuFFT plan creation overhead.
  // At 32³: GPU=2.87ms vs CPU=1.76ms. At 16³: GPU=5.5ms vs CPU=15ms (GPU wins).
  // Threshold: 32768 (32³) — below this, use CPU.
  if (N <= 32768) {
    auto t0 = std::chrono::steady_clock::now();
    auto V_cpu = PoissonSolver::SolvePeriodicFFT(grid, rho);
    auto t1 = std::chrono::steady_clock::now();
    result.V = V_cpu;
    result.kernel_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    // Compute Hartree energy: E_H = 0.5 * integral rho * V d^3r.
    const auto [h0, h1, h2] = grid.h;
    const double dv = h0 * h1 * h2;
    double E = 0.0;
    for (std::size_t i = 0; i < N; ++i)
      E += rho[i] * V_cpu[i] * dv;
    result.hartree_energy = 0.5 * E;
    // Add ledger entry for CPU fallback path.
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
            "CPU fallback Poisson FFT"},
        0.0, N, N, 0,
        "CUDA Poisson FFT (CPU fallback for small grid)"});
    return result;
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

  // AUDIT B10: Use GPU arena for persistent device buffers and stream.
  GpuArena& arena = GpuArena::Instance();
  cudaStream_t stream = arena.Stream();

  // Allocate device memory via arena (reuses cached blocks).
  cufftDoubleComplex* d_rho_k = nullptr;
  cufftDoubleComplex* d_V_k = nullptr;
  double* d_kx_sq = nullptr;
  double* d_ky_sq = nullptr;
  double* d_kz_sq = nullptr;
  cufftHandle plan_fwd = 0, plan_inv = 0;

  auto cleanup = [&]() {
    if (d_rho_k) arena.Free(d_rho_k);
    if (d_V_k) arena.Free(d_V_k);
    if (d_kx_sq) arena.Free(d_kx_sq);
    if (d_ky_sq) arena.Free(d_ky_sq);
    if (d_kz_sq) arena.Free(d_kz_sq);
    if (plan_fwd) cufftDestroy(plan_fwd);
    if (plan_inv) cufftDestroy(plan_inv);
  };

  d_rho_k = static_cast<cufftDoubleComplex*>(
      arena.Alloc(N * sizeof(cufftDoubleComplex)));
  if (!d_rho_k) { cleanup(); return Status::IoError("arena.Alloc failed for rho_k"); }
  cudaError_t err = arena.H2D(d_rho_k, host_rho_k.data(),
                               N * sizeof(cufftDoubleComplex));
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "arena.H2D rho_k"); }

  d_V_k = static_cast<cufftDoubleComplex*>(
      arena.Alloc(N * sizeof(cufftDoubleComplex)));
  if (!d_V_k) { cleanup(); return Status::IoError("arena.Alloc failed for V_k"); }

  d_kx_sq = static_cast<double*>(arena.Alloc(n0 * sizeof(double)));
  if (!d_kx_sq) { cleanup(); return Status::IoError("arena.Alloc failed for kx_sq"); }
  err = arena.H2D(d_kx_sq, kx_sq.data(), n0 * sizeof(double));
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "arena.H2D kx_sq"); }

  d_ky_sq = static_cast<double*>(arena.Alloc(n1 * sizeof(double)));
  if (!d_ky_sq) { cleanup(); return Status::IoError("arena.Alloc failed for ky_sq"); }
  err = arena.H2D(d_ky_sq, ky_sq.data(), n1 * sizeof(double));
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "arena.H2D ky_sq"); }

  d_kz_sq = static_cast<double*>(arena.Alloc(n2 * sizeof(double)));
  if (!d_kz_sq) { cleanup(); return Status::IoError("arena.Alloc failed for kz_sq"); }
  err = arena.H2D(d_kz_sq, kz_sq.data(), n2 * sizeof(double));
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "arena.H2D kz_sq"); }

  // Create C2C FFT plans on the arena stream.
  cufftResult cufft_err = cufftPlan3d(&plan_fwd,
      static_cast<int>(n2), static_cast<int>(n1), static_cast<int>(n0),
      CUFFT_Z2Z);
  if (cufft_err != CUFFT_SUCCESS) {
    cleanup();
    return CufftStatus(cufft_err, "cufftPlan3d Z2Z fwd");
  }
  cufftSetStream(plan_fwd, stream);
  cufft_err = cufftPlan3d(&plan_inv,
      static_cast<int>(n2), static_cast<int>(n1), static_cast<int>(n0),
      CUFFT_Z2Z);
  if (cufft_err != CUFFT_SUCCESS) {
    cleanup();
    return CufftStatus(cufft_err, "cufftPlan3d Z2Z inv");
  }
  cufftSetStream(plan_inv, stream);

  auto kernel_start = std::chrono::steady_clock::now();

  // Forward FFT: rho(r) -> rho(k).
  cufft_err = cufftExecZ2Z(plan_fwd, d_rho_k, d_rho_k, CUFFT_FORWARD);
  if (cufft_err != CUFFT_SUCCESS) {
    cleanup();
    return CufftStatus(cufft_err, "cufftExecZ2Z fwd");
  }
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "sync after fwd Z2Z"); }

  // Apply Green's function: V(k) = 4*pi*rho(k) / k^2.
  const int total = static_cast<int>(N);
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  ApplyGreensFunctionKernel<<<blocks, threads, 0, stream>>>(
      d_V_k, d_rho_k, d_kx_sq, d_ky_sq, d_kz_sq,
      static_cast<int>(n0), static_cast<int>(n1), static_cast<int>(n2));
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "sync after Green's"); }

  // Inverse FFT: V(k) -> V(r).
  cufft_err = cufftExecZ2Z(plan_inv, d_V_k, d_rho_k, CUFFT_INVERSE);
  if (cufft_err != CUFFT_SUCCESS) {
    cleanup();
    return CufftStatus(cufft_err, "cufftExecZ2Z inv");
  }
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "sync after inv Z2Z"); }

  auto kernel_end = std::chrono::steady_clock::now();
  result.kernel_ms =
      std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();

  // Copy result back via arena D2H.
  std::vector<cufftDoubleComplex> host_V_k(N);
  err = arena.D2H(host_V_k.data(), d_rho_k, N * sizeof(cufftDoubleComplex));
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "arena.D2H V"); }

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

// GPU free-space Poisson solver via zero-padded cuFFT convolution.
// Ports PoissonSolver::SolveFreeFFT to GPU: doubles the grid, zero-pads rho
// into the first octant, builds the 1/|r| Coulomb kernel, forward-FFT both,
// pointwise multiply, inverse-FFT, extract first octant.
// This eliminates the D2H transfer for molecular Poisson in the SCF loop.
[[nodiscard]] Result<PoissonFftGpuResult> PoissonFreeCuda(
    const UniformGrid3D& grid,
    const std::vector<double>& rho) {
  const std::size_t N = grid.total_points();
  PoissonFftGpuResult result;
  result.grid_size = N;

  if (N == 0) return result;
  if (rho.size() != N)
    return Status::InvalidArgument("rho size mismatch with grid");

  // Small grids: CPU FFTW is faster (cuFFT plan creation overhead).
  if (N <= 32768) {
    auto t0 = std::chrono::steady_clock::now();
    auto V_cpu = PoissonSolver::SolveFreeFFT(grid, rho);
    auto t1 = std::chrono::steady_clock::now();
    result.V = V_cpu;
    result.kernel_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    const auto [h0, h1, h2] = grid.h;
    const double dv = h0 * h1 * h2;
    double E = 0.0;
    for (std::size_t i = 0; i < N; ++i)
      E += rho[i] * V_cpu[i] * dv;
    result.hartree_energy = 0.5 * E;
    tides::tile::PrecisionDescriptor desc;
    desc.storage = tides::tile::NumericFormat::kFloat64;
    desc.compute = tides::tile::NumericFormat::kFloat64;
    desc.reduction = tides::tile::NumericFormat::kFloat64;
    desc.determinism = tides::tile::DeterminismMode::kDeterministic;
    desc.label = "cuda-poisson-free";
    result.ledger.Add(tides::tile::OperationLedgerEntry{
        tides::tile::OperationKind::kPoissonSolve,
        desc,
        tides::tile::ErrorBudget{tides::tile::ErrorMetric::kAbsolute, 0.0,
            "CPU fallback free-space Poisson"},
        0.0, N, N, 0,
        "CUDA free-space Poisson (CPU fallback for small grid)"});
    return result;
  }

  if (!PoissonFftCudaAvailable())
    return Status::IoError("CUDA runtime not available for free-space Poisson");

  const auto [n0, n1, n2] = grid.n;
  const auto [h0, h1, h2] = grid.h;
  const std::size_t m0 = 2 * n0, m1 = 2 * n1, m2 = 2 * n2;
  const std::size_t M = m0 * m1 * m2;
  const double dv = h0 * h1 * h2;
  const double h_eff = std::cbrt(dv);
  const double self_phi = 2.3801 / h_eff;

  // Prepare zero-padded rho and 1/|r| kernel on host.
  std::vector<cufftDoubleComplex> host_rho(M);
  std::vector<cufftDoubleComplex> host_g(M);
  for (std::size_t i = 0; i < M; ++i) {
    host_rho[i].x = 0.0; host_rho[i].y = 0.0;
    host_g[i].x = 0.0;   host_g[i].y = 0.0;
  }
  // Place rho*dv in the first octant.
  for (std::size_t iz = 0; iz < n2; ++iz)
    for (std::size_t iy = 0; iy < n1; ++iy)
      for (std::size_t ix = 0; ix < n0; ++ix) {
        const std::size_t src = grid.flatten(ix, iy, iz);
        const std::size_t dst = ix + m0 * (iy + m1 * iz);
        host_rho[dst].x = rho[src] * dv;
      }
  // Build the 1/|r| kernel on the doubled grid.
  for (std::size_t iz = 0; iz < m2; ++iz) {
    const double dz = (iz < n2) ? static_cast<double>(iz) * h2
                                : (static_cast<double>(iz) - static_cast<double>(m2)) * h2;
    for (std::size_t iy = 0; iy < m1; ++iy) {
      const double dy = (iy < n1) ? static_cast<double>(iy) * h1
                                  : (static_cast<double>(iy) - static_cast<double>(m1)) * h1;
      for (std::size_t ix = 0; ix < m0; ++ix) {
        const double dx = (ix < n0) ? static_cast<double>(ix) * h0
                                    : (static_cast<double>(ix) - static_cast<double>(m0)) * h0;
        const bool wrap = (ix == n0) || (iy == n1) || (iz == n2);
        const std::size_t g = ix + m0 * (iy + m1 * iz);
        if (wrap) {
          host_g[g].x = 0.0;
        } else {
          const double r2 = dx * dx + dy * dy + dz * dz;
          if (r2 < 1e-30)
            host_g[g].x = self_phi;
          else
            host_g[g].x = 1.0 / std::sqrt(r2);
        }
      }
    }
  }

  // Upload to device via GPU arena.
  GpuArena& arena = GpuArena::Instance();
  cudaStream_t stream = arena.Stream();

  cufftDoubleComplex* d_rho = nullptr;
  cufftDoubleComplex* d_g = nullptr;
  cufftDoubleComplex* d_V = nullptr;
  cufftHandle plan_fwd_rho = 0, plan_fwd_g = 0, plan_inv = 0;

  auto cleanup = [&]() {
    if (d_rho) arena.Free(d_rho);
    if (d_g) arena.Free(d_g);
    if (d_V) arena.Free(d_V);
    if (plan_fwd_rho) cufftDestroy(plan_fwd_rho);
    if (plan_fwd_g) cufftDestroy(plan_fwd_g);
    if (plan_inv) cufftDestroy(plan_inv);
  };

  d_rho = static_cast<cufftDoubleComplex*>(
      arena.Alloc(M * sizeof(cufftDoubleComplex)));
  if (!d_rho) { cleanup(); return Status::IoError("arena.Alloc failed for rho"); }
  cudaError_t err = arena.H2D(d_rho, host_rho.data(),
                               M * sizeof(cufftDoubleComplex));
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "arena.H2D rho"); }

  d_g = static_cast<cufftDoubleComplex*>(
      arena.Alloc(M * sizeof(cufftDoubleComplex)));
  if (!d_g) { cleanup(); return Status::IoError("arena.Alloc failed for g"); }
  err = arena.H2D(d_g, host_g.data(), M * sizeof(cufftDoubleComplex));
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "arena.H2D g"); }

  d_V = static_cast<cufftDoubleComplex*>(
      arena.Alloc(M * sizeof(cufftDoubleComplex)));
  if (!d_V) { cleanup(); return Status::IoError("arena.Alloc failed for V"); }

  // Create cuFFT plans (dimension order: m2, m1, m0 for column-major arrays).
  cufftResult cufft_err = cufftPlan3d(&plan_fwd_rho,
      static_cast<int>(m2), static_cast<int>(m1), static_cast<int>(m0),
      CUFFT_Z2Z);
  if (cufft_err != CUFFT_SUCCESS) {
    cleanup(); return CufftStatus(cufft_err, "cufftPlan3d Z2Z fwd rho");
  }
  cufftSetStream(plan_fwd_rho, stream);

  cufft_err = cufftPlan3d(&plan_fwd_g,
      static_cast<int>(m2), static_cast<int>(m1), static_cast<int>(m0),
      CUFFT_Z2Z);
  if (cufft_err != CUFFT_SUCCESS) {
    cleanup(); return CufftStatus(cufft_err, "cufftPlan3d Z2Z fwd g");
  }
  cufftSetStream(plan_fwd_g, stream);

  cufft_err = cufftPlan3d(&plan_inv,
      static_cast<int>(m2), static_cast<int>(m1), static_cast<int>(m0),
      CUFFT_Z2Z);
  if (cufft_err != CUFFT_SUCCESS) {
    cleanup(); return CufftStatus(cufft_err, "cufftPlan3d Z2Z inv");
  }
  cufftSetStream(plan_inv, stream);

  auto kernel_start = std::chrono::steady_clock::now();

  // Forward FFT: rho(r) -> rho(k), g(r) -> g(k).
  cufft_err = cufftExecZ2Z(plan_fwd_rho, d_rho, d_rho, CUFFT_FORWARD);
  if (cufft_err != CUFFT_SUCCESS) {
    cleanup(); return CufftStatus(cufft_err, "cufftExecZ2Z fwd rho");
  }
  cufft_err = cufftExecZ2Z(plan_fwd_g, d_g, d_g, CUFFT_FORWARD);
  if (cufft_err != CUFFT_SUCCESS) {
    cleanup(); return CufftStatus(cufft_err, "cufftExecZ2Z fwd g");
  }
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "sync after fwd"); }

  // Pointwise multiply: V(k) = rho(k) * g(k).
  {
    const int total = static_cast<int>(M);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    ComplexMultiplyKernel<<<blocks, threads, 0, stream>>>(
        d_V, d_rho, d_g, total);
  }
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "sync after multiply"); }

  // Inverse FFT: V(k) -> V(r).
  cufft_err = cufftExecZ2Z(plan_inv, d_V, d_V, CUFFT_INVERSE);
  if (cufft_err != CUFFT_SUCCESS) {
    cleanup(); return CufftStatus(cufft_err, "cufftExecZ2Z inv");
  }
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "sync after inv"); }

  auto kernel_end = std::chrono::steady_clock::now();
  result.kernel_ms =
      std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();

  // Download result.
  std::vector<cufftDoubleComplex> host_V(M);
  err = arena.D2H(host_V.data(), d_V, M * sizeof(cufftDoubleComplex));
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "arena.D2H V"); }

  cleanup();

  // Extract first octant and normalize by 1/M.
  result.V.resize(N);
  const double inv_M = 1.0 / static_cast<double>(M);
  for (std::size_t iz = 0; iz < n2; ++iz)
    for (std::size_t iy = 0; iy < n1; ++iy)
      for (std::size_t ix = 0; ix < n0; ++ix) {
        const std::size_t src = ix + m0 * (iy + m1 * iz);
        const std::size_t dst = grid.flatten(ix, iy, iz);
        result.V[dst] = host_V[src].x * inv_M;
      }

  // Hartree energy: E_H = 0.5 * integral rho * V * dv.
  double E = 0.0;
  for (std::size_t i = 0; i < N; ++i)
    E += rho[i] * result.V[i] * dv;
  result.hartree_energy = 0.5 * E;

  // Ledger.
  tides::tile::PrecisionDescriptor desc;
  desc.storage = tides::tile::NumericFormat::kFloat64;
  desc.compute = tides::tile::NumericFormat::kFloat64;
  desc.reduction = tides::tile::NumericFormat::kFloat64;
  desc.determinism = tides::tile::DeterminismMode::kDeterministic;
  desc.label = "cuda-poisson-free";
  result.ledger.Add(tides::tile::OperationLedgerEntry{
      tides::tile::OperationKind::kPoissonSolve,
      desc,
      tides::tile::ErrorBudget{tides::tile::ErrorMetric::kAbsolute, 0.0,
          "GPU cuFFT free-space Poisson vs CPU reference"},
      0.0, static_cast<std::uint64_t>(N), static_cast<std::uint64_t>(M), 0,
      "CUDA cuFFT free-space Poisson (zero-padded convolution)"});

  return result;
}

// =============================================================================
// Device-resident free-space Poisson solver (cached plans + device data flow)
// =============================================================================

namespace {

// GPU kernel: zero-pad rho into first octant of doubled grid and scale by dv.
// Input: d_rho (np_total doubles in grid.flatten layout)
// Output: d_rho_pad (M doubles, first octant filled, rest zero)
__global__ void ZeroPadRhoKernel(
    double* d_rho_pad,
    const double* d_rho,
    double dv,
    int n0, int n1, int n2,
    int m0, int m1) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n0 * n1 * n2;
  if (idx >= total) return;
  const int iz = idx / (n0 * n1);
  const int rem = idx % (n0 * n1);
  const int iy = rem / n0;
  const int ix = rem % n0;
  const int dst = ix + m0 * (iy + m1 * iz);
  d_rho_pad[dst] = d_rho[idx] * dv;
}

// GPU kernel: extract first octant from V_pad (real) and normalize by 1/M.
__global__ void ExtractOctantKernel(
    double* d_V_out,
    const double* d_V_pad,
    double inv_M,
    int n0, int n1, int n2,
    int m0, int m1) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n0 * n1 * n2;
  if (idx >= total) return;
  const int iz = idx / (n0 * n1);
  const int rem = idx % (n0 * n1);
  const int iy = rem / n0;
  const int ix = rem % n0;
  const int src = ix + m0 * (iy + m1 * iz);
  d_V_out[idx] = d_V_pad[src] * inv_M;
}

// GPU kernel: compute Hartree energy = 0.5 * sum(rho * V * dv) via reduction.
// Uses a single block for simplicity (grid sizes are modest).
__global__ void HartreeEnergyKernel(
    double* d_energy,
    const double* d_rho,
    const double* d_V,
    double dv,
    int n) {
  extern __shared__ double sdata[];
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int total = n;

  double partial = 0.0;
  for (int i = bid * blockDim.x + tid; i < total; i += gridDim.x * blockDim.x) {
    partial += d_rho[i] * d_V[i] * dv;
  }
  sdata[tid] = partial;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid == 0) atomicAdd(d_energy, 0.5 * sdata[0]);
}

// GPU kernel: zero a double array.
__global__ void ZeroRealKernel(
    double* d, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  d[idx] = 0.0;
}

}  // namespace

Result<PoissonFreeDeviceResult> PoissonFreeDeviceCache::Solve(
    const UniformGrid3D& grid,
    const double* d_rho,
    double* d_V_out,
    cudaStream_t stream) {

  const std::size_t N = grid.total_points();
  if (N == 0) return Status::InvalidArgument("PoissonFreeDeviceCache: N=0");

  PoissonFreeDeviceResult result;
  result.grid_size = N;

  const auto [n0, n1, n2] = grid.n;
  const auto [h0, h1, h2] = grid.h;
  const double dv = h0 * h1 * h2;

  // Small grids: fall back to the host API.
  if (N <= 32768) {
    std::vector<double> rho_host(N);
    cudaMemcpyAsync(rho_host.data(), d_rho, N * sizeof(double),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    auto V_cpu = PoissonSolver::SolveFreeFFT(grid, rho_host);
    cudaMemcpyAsync(d_V_out, V_cpu.data(), N * sizeof(double),
                    cudaMemcpyHostToDevice, stream);
    double E = 0.0;
    for (std::size_t i = 0; i < N; ++i)
      E += rho_host[i] * V_cpu[i] * dv;
    result.hartree_energy = 0.5 * E;
    return result;
  }

  const std::size_t m0 = 2 * n0, m1 = 2 * n1, m2 = 2 * n2;
  const std::size_t M = m0 * m1 * m2;
  const double h_eff = std::cbrt(dv);
  const double self_phi = 2.3801 / h_eff;
  const double inv_M = 1.0 / static_cast<double>(M);

  // Initialize or reinitialize cache if grid dimensions changed.
  if (!initialized_ || cached_n0_ != n0 || cached_n1_ != n1 || cached_n2_ != n2) {
    if (initialized_) {
      if (plan_fwd_) cufftDestroy(plan_fwd_);
      if (plan_inv_) cufftDestroy(plan_inv_);
      if (d_g_) cudaFree(d_g_);
      if (d_rho_pad_) cudaFree(d_rho_pad_);
      if (d_fft_) cudaFree(d_fft_);
      if (d_energy_) cudaFree(d_energy_);
    }

    m0_ = m0; m1_ = m1; m2_ = m2; M_ = M;
    M_hc_ = (m0 / 2 + 1) * m1 * m2;  // hermitian half-size for R2C/C2R
    cached_n0_ = n0; cached_n1_ = n1; cached_n2_ = n2;

    // Allocate device buffers.
    // d_rho_pad: real array [M] doubles (zero-padded rho)
    // d_fft_: complex array [M_hc] (hermitian half for R2C/C2R)
    // d_g_: complex array [M_hc] (FFT'd kernel, hermitian half)
    cudaMalloc(&d_g_, M_hc_ * sizeof(cufftDoubleComplex));
    cudaMalloc(&d_rho_pad_, M * sizeof(double));
    cudaMalloc(&d_fft_, M_hc_ * sizeof(cufftDoubleComplex));
    cudaMalloc(&d_energy_, sizeof(double));

    // Build and upload 1/|r| kernel on host (one-time cost).
    // Build as real array, then R2C FFT to get hermitian half.
    std::vector<double> host_g_real(M);
    for (std::size_t iz = 0; iz < m2; ++iz) {
      const double dz = (iz < n2) ? static_cast<double>(iz) * h2
                                  : (static_cast<double>(iz) - static_cast<double>(m2)) * h2;
      for (std::size_t iy = 0; iy < m1; ++iy) {
        const double dy = (iy < n1) ? static_cast<double>(iy) * h1
                                    : (static_cast<double>(iy) - static_cast<double>(m1)) * h1;
        for (std::size_t ix = 0; ix < m0; ++ix) {
          const double dx = (ix < n0) ? static_cast<double>(ix) * h0
                                      : (static_cast<double>(ix) - static_cast<double>(m0)) * h0;
          const bool wrap = (ix == n0) || (iy == n1) || (iz == n2);
          const std::size_t g = ix + m0 * (iy + m1 * iz);
          if (wrap) {
            host_g_real[g] = 0.0;
          } else {
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < 1e-30) {
              host_g_real[g] = self_phi;
            } else {
              host_g_real[g] = 1.0 / std::sqrt(r2);
            }
          }
        }
      }
    }

    // Create cuFFT plans: D2Z (real-to-complex) forward, Z2D (complex-to-real) inverse.
    // Dimension order: m2, m1, m0 for column-major.
    cufftResult cufft_err = cufftPlan3d(&plan_fwd_,
        static_cast<int>(m2), static_cast<int>(m1), static_cast<int>(m0),
        CUFFT_D2Z);
    if (cufft_err != CUFFT_SUCCESS)
      return CufftStatus(cufft_err, "PoissonFreeDeviceCache: cufftPlan3d D2Z fwd");
    cufftSetStream(plan_fwd_, stream);

    cufft_err = cufftPlan3d(&plan_inv_,
        static_cast<int>(m2), static_cast<int>(m1), static_cast<int>(m0),
        CUFFT_Z2D);
    if (cufft_err != CUFFT_SUCCESS)
      return CufftStatus(cufft_err, "PoissonFreeDeviceCache: cufftPlan3d Z2D inv");
    cufftSetStream(plan_inv_, stream);

    // Forward FFT the kernel (one-time): real -> hermitian half.
    // Upload real kernel, D2Z transform, result in d_g_.
    double* d_g_real = nullptr;
    cudaMalloc(&d_g_real, M * sizeof(double));
    cudaMemcpyAsync(d_g_real, host_g_real.data(), M * sizeof(double),
                    cudaMemcpyHostToDevice, stream);
    cufft_err = cufftExecD2Z(plan_fwd_, d_g_real, d_g_);
    if (cufft_err != CUFFT_SUCCESS) {
      cudaFree(d_g_real);
      return CufftStatus(cufft_err, "PoissonFreeDeviceCache: cufftExecD2Z fwd g");
    }
    cudaFree(d_g_real);

    initialized_ = true;
  }

  // Update stream on cached plans (stream may change across Run calls).
  cufftSetStream(plan_fwd_, stream);
  cufftSetStream(plan_inv_, stream);

  // Profiling mode: set TIDES_POISSON_PROFILE=1 to enable CUDA event timing.
  // Event creation + sync adds ~10-30ms overhead, so disabled by default.
  static const bool profile = [] {
    const char* e = std::getenv("TIDES_POISSON_PROFILE");
    return e && e[0] == '1';
  }();

  cudaEvent_t ev0, ev1, ev2, ev3, ev4, ev5, ev6, ev7;
  if (profile) {
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventCreate(&ev2); cudaEventCreate(&ev3);
    cudaEventCreate(&ev4); cudaEventCreate(&ev5);
    cudaEventCreate(&ev6); cudaEventCreate(&ev7);
    cudaEventRecord(ev0, stream);
  }

  // Step 1: Zero-pad rho into d_rho_pad (real array, M doubles).
  cudaMemsetAsync(d_rho_pad_, 0, M_ * sizeof(double), stream);
  if (profile) cudaEventRecord(ev1, stream);
  {
    const int total = static_cast<int>(N);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    ZeroPadRhoKernel<<<blocks, threads, 0, stream>>>(
        d_rho_pad_, d_rho, dv,
        static_cast<int>(n0), static_cast<int>(n1), static_cast<int>(n2),
        static_cast<int>(m0_), static_cast<int>(m1_));
  }
  if (profile) cudaEventRecord(ev2, stream);

  // Step 2: Forward D2Z FFT: real rho_pad -> hermitian half d_fft_.
  cufftResult cufft_err = cufftExecD2Z(plan_fwd_, d_rho_pad_, d_fft_);
  if (cufft_err != CUFFT_SUCCESS)
    return CufftStatus(cufft_err, "PoissonFreeDeviceCache: cufftExecD2Z fwd rho");
  if (profile) cudaEventRecord(ev3, stream);

  // Step 3: Pointwise multiply V(k) = rho(k) * g(k) in hermitian half.
  {
    const int total = static_cast<int>(M_hc_);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    ComplexMultiplyKernel<<<blocks, threads, 0, stream>>>(
        d_fft_, d_fft_, d_g_, total);
  }
  if (profile) cudaEventRecord(ev4, stream);

  // Step 4: Inverse Z2D FFT: hermitian half d_fft_ -> real d_rho_pad_.
  // Z2D overwrites input, result is real in d_rho_pad_ (M doubles).
  cufft_err = cufftExecZ2D(plan_inv_, d_fft_, d_rho_pad_);
  if (cufft_err != CUFFT_SUCCESS)
    return CufftStatus(cufft_err, "PoissonFreeDeviceCache: cufftExecZ2D inv");
  if (profile) cudaEventRecord(ev5, stream);

  // Step 5: Extract first octant and normalize.
  {
    const int total = static_cast<int>(N);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    ExtractOctantKernel<<<blocks, threads, 0, stream>>>(
        d_V_out, d_rho_pad_, inv_M,
        static_cast<int>(n0), static_cast<int>(n1), static_cast<int>(n2),
        static_cast<int>(m0_), static_cast<int>(m1_));
  }
  if (profile) cudaEventRecord(ev6, stream);

  // Step 6: Compute Hartree energy on device (async, no sync).
  cudaMemsetAsync(d_energy_, 0, sizeof(double), stream);
  {
    const int threads = 256;
    const int blocks = 256;
    const int smem = threads * sizeof(double);
    HartreeEnergyKernel<<<blocks, threads, smem, stream>>>(
        d_energy_, d_rho, d_V_out, dv, static_cast<int>(N));
  }
  // Energy D2H deferred to caller — no blocking cudaMemcpyAsync here.
  // The caller syncs the stream when needed and can read d_energy_ if required.
  if (profile) cudaEventRecord(ev7, stream);

  if (profile) {
    // Sync to get event timings (also needed for energy D2H).
    cudaStreamSynchronize(stream);

    float ms;
    cudaEventElapsedTime(&ms, ev0, ev1); result.memset_pad_ms = ms;
    cudaEventElapsedTime(&ms, ev1, ev2); result.zero_pad_ms = ms;
    cudaEventElapsedTime(&ms, ev2, ev3); result.fft_fwd_ms = ms;
    cudaEventElapsedTime(&ms, ev3, ev4); result.multiply_ms = ms;
    cudaEventElapsedTime(&ms, ev4, ev5); result.fft_inv_ms = ms;
    cudaEventElapsedTime(&ms, ev5, ev6); result.extract_ms = ms;
    cudaEventElapsedTime(&ms, ev6, ev7); result.energy_ms = ms;
    result.kernel_ms = result.memset_pad_ms + result.zero_pad_ms +
        result.fft_fwd_ms + result.multiply_ms + result.fft_inv_ms +
        result.extract_ms + result.energy_ms;
    result.fft_n0 = m0_; result.fft_n1 = m1_; result.fft_n2 = m2_;

    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    cudaEventDestroy(ev2); cudaEventDestroy(ev3);
    cudaEventDestroy(ev4); cudaEventDestroy(ev5);
    cudaEventDestroy(ev6); cudaEventDestroy(ev7);
  } else {
    // No sync — caller owns the stream and will sync when they need V_H or energy.
    // kernel_ms remains 0; substep fields remain 0.
    result.fft_n0 = m0_; result.fft_n1 = m1_; result.fft_n2 = m2_;
  }

  return result;
}

void PoissonFreeDeviceCache::Release() {
  if (plan_fwd_) { cufftDestroy(plan_fwd_); plan_fwd_ = 0; }
  if (plan_inv_) { cufftDestroy(plan_inv_); plan_inv_ = 0; }
  if (d_g_) { cudaFree(d_g_); d_g_ = nullptr; }
  if (d_rho_pad_) { cudaFree(d_rho_pad_); d_rho_pad_ = nullptr; }
  if (d_fft_) { cudaFree(d_fft_); d_fft_ = nullptr; }
  if (d_energy_) { cudaFree(d_energy_); d_energy_ = nullptr; }
  initialized_ = false;
}

PoissonFreeDeviceCache::~PoissonFreeDeviceCache() {
  Release();
}

}  // namespace tides::grid
