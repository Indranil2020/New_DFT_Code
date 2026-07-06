// T3.5: GPU XC functional evaluation on the grid.
//
// Evaluates LDA-PW92 (Slater X + PW92 C) on the GPU grid in parallel.
// Each thread processes one grid point: computes V_xc and eps_xc from rho(r).
//
// Observable (T3.5): He/Ne totals vs PySCF <= 1e-8 Ha.
// The GPU path matches the CPU LDA evaluation exactly (same formulas), so
// GPU vs CPU agreement is machine-precision. The PySCF cross-validation is
// done at the SCF energy assembly level (combining grid XC with atomic solver).

#include "grid/xc.hpp"
#include "grid/dual_grid.hpp"
#include "grid/libxc_wrapper.hpp"

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

// Device-side LDA exchange: Slater X-alpha (alpha=2/3).
// eps_x(n, zeta) = -3/4 (3/pi)^(1/3) * f(zeta) * n^(1/3)
__device__ double EpsXDevice(double n, double zeta) {
  if (n <= 0.0) return 0.0;
  const double fz = 0.5 * (pow(1.0 + zeta, 4.0 / 3.0) +
                           pow(1.0 - zeta, 4.0 / 3.0));
  return -0.75 * pow(3.0 / M_PI, 1.0 / 3.0) * fz * pow(n, 1.0 / 3.0);
}

// Device-side PW92 correlation (paramagnetic).
__device__ double EpsCParamagneticDevice(double rs) {
  const double a = 0.0310907, a1 = 0.2137;
  const double b1 = 7.5957, b2 = 3.5876, b3 = 1.6382, b4 = 0.49294;
  const double sr = sqrt(rs);
  const double Q = b1 * sr + b2 * rs + b3 * rs * sr + b4 * rs * rs;
  return -2.0 * a * (1.0 + a1 * rs) * log(1.0 + 1.0 / (2.0 * a * Q));
}

// Device-side PW92 correlation (ferromagnetic).
__device__ double EpsCFerromagneticDevice(double rs) {
  const double a = 0.015545, a1 = 0.20548;
  const double b1 = 14.1189, b2 = 6.1977, b3 = 3.3662, b4 = 0.62517;
  const double sr = sqrt(rs);
  const double Q = b1 * sr + b2 * rs + b3 * rs * sr + b4 * rs * rs;
  return -2.0 * a * (1.0 + a1 * rs) * log(1.0 + 1.0 / (2.0 * a * Q));
}

// Device-side d(eps_c)/d(rs) via central finite difference.
__device__ double DEpsCParamagneticDRsDevice(double rs) {
  const double h = 1e-6 * (1.0 + rs);
  return (EpsCParamagneticDevice(rs + h) - EpsCParamagneticDevice(rs - h)) / (2.0 * h);
}

__device__ double DEpsCFerromagneticDRsDevice(double rs) {
  const double h = 1e-6 * (1.0 + rs);
  return (EpsCFerromagneticDevice(rs + h) - EpsCFerromagneticDevice(rs - h)) / (2.0 * h);
}

// Device-side spin polarization factor.
__device__ double SpinPolFactorDevice(double zeta) {
  const double fz = (pow(1.0 + zeta, 4.0 / 3.0) +
                     pow(1.0 - zeta, 4.0 / 3.0) - 2.0) /
                    (pow(2.0, 4.0 / 3.0) - 2.0);
  if (fz < 0.0) return 0.0;
  if (fz > 1.0) return 1.0;
  return fz;
}

// Device-side total eps_xc(n, zeta).
__device__ double EpsXCDevice(double n, double zeta) {
  if (n <= 0.0) return 0.0;
  const double ex = EpsXDevice(n, zeta);
  const double rs = pow(3.0 / (4.0 * M_PI * n), 1.0 / 3.0);
  const double ep = EpsCParamagneticDevice(rs);
  if (fabs(zeta) < 1e-12) return ex + ep;
  const double ef = EpsCFerromagneticDevice(rs);
  const double fz = SpinPolFactorDevice(zeta);
  return ex + ep + (ef - ep) * fz;
}

// Device-side V_xc(n, zeta).
__device__ double VXCDevice(double n, double zeta) {
  if (n <= 0.0) return 0.0;
  const double vx = (4.0 / 3.0) * EpsXDevice(n, zeta);
  const double rs = pow(3.0 / (4.0 * M_PI * n), 1.0 / 3.0);
  const double ep = EpsCParamagneticDevice(rs);
  const double dep = DEpsCParamagneticDRsDevice(rs);
  if (fabs(zeta) < 1e-12) {
    return vx + ep - (rs / 3.0) * dep;
  }
  const double ef = EpsCFerromagneticDevice(rs);
  const double def = DEpsCFerromagneticDRsDevice(rs);
  const double fz = SpinPolFactorDevice(zeta);
  const double ec = ep + (ef - ep) * fz;
  const double dec_drs = dep + (def - dep) * fz;
  return vx + ec - (rs / 3.0) * dec_drs;
}

// Kernel: evaluate LDA XC on the grid. Each thread processes one grid point.
__global__ void XCEvalKernel(
    const double* rho,
    double* vxc,
    double* eps_xc,
    std::size_t n_points,
    double zeta) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n_points) return;
  const double n = rho[idx] > 0.0 ? rho[idx] : 0.0;
  vxc[idx] = VXCDevice(n, zeta);
  eps_xc[idx] = EpsXCDevice(n, zeta);
}

// Kernel: compute XC energy via reduction. E_xc = sum eps_xc * rho * dv.
__global__ void XCEnergyKernel(
    const double* eps_xc,
    const double* rho,
    std::size_t n_points,
    double dv,
    double* partial_sums) {
  extern __shared__ double sdata[];
  const std::size_t tid = threadIdx.x;
  const std::size_t bid = blockIdx.x;
  const std::size_t block_size = blockDim.x;

  double sum = 0.0;
  for (std::size_t i = bid * block_size + tid; i < n_points;
       i += gridDim.x * block_size) {
    sum += eps_xc[i] * rho[i] * dv;
  }
  sdata[tid] = sum;
  __syncthreads();

  for (std::size_t s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_sums[bid] = sdata[0];
  }
}

}  // namespace

struct XCGpuResult {
  std::vector<double> vxc;
  std::vector<double> eps_xc;
  double xc_energy = 0.0;
  double kernel_ms = 0.0;
  std::size_t n_points = 0;
  tides::tile::OperationLedger ledger;
};

[[nodiscard]] bool XCCudaAvailable() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

[[nodiscard]] Result<XCGpuResult> XCEvalLdaCuda(
    const UniformGrid3D& grid,
    const std::vector<double>& rho,
    double zeta) {
  const std::size_t N = grid.total_points();
  XCGpuResult result;
  result.n_points = N;

  if (N == 0) {
    return result;
  }

  if (rho.size() != N) {
    return Status::InvalidArgument("rho size mismatch with grid");
  }

  if (!XCCudaAvailable()) {
    return Status::IoError("CUDA runtime not available for XC evaluation");
  }

  const auto [h0, h1, h2] = grid.h;
  const double dv = h0 * h1 * h2;

  // Allocate device memory.
  double* d_rho = nullptr;
  double* d_vxc = nullptr;
  double* d_eps_xc = nullptr;

  auto cleanup = [&]() {
    if (d_rho) cudaFree(d_rho);
    if (d_vxc) cudaFree(d_vxc);
    if (d_eps_xc) cudaFree(d_eps_xc);
  };

  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_rho), N * sizeof(double));
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMalloc rho"); }
  err = cudaMemcpy(d_rho, rho.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMemcpy rho"); }

  err = cudaMalloc(reinterpret_cast<void**>(&d_vxc), N * sizeof(double));
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMalloc vxc"); }
  err = cudaMalloc(reinterpret_cast<void**>(&d_eps_xc), N * sizeof(double));
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMalloc eps_xc"); }

  // Launch XC evaluation kernel.
  const int threads = 256;
  const int blocks = static_cast<int>((N + threads - 1) / threads);
  auto kernel_start = std::chrono::steady_clock::now();
  XCEvalKernel<<<blocks, threads>>>(d_rho, d_vxc, d_eps_xc, N, zeta);
  err = cudaDeviceSynchronize();
  auto kernel_end = std::chrono::steady_clock::now();
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "XCEvalKernel"); }

  result.kernel_ms =
      std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();

  // Copy results back.
  result.vxc.resize(N);
  result.eps_xc.resize(N);
  err = cudaMemcpy(result.vxc.data(), d_vxc, N * sizeof(double), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMemcpy vxc D2H"); }
  err = cudaMemcpy(result.eps_xc.data(), d_eps_xc, N * sizeof(double), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMemcpy eps_xc D2H"); }

  // Compute XC energy via GPU reduction.
  const int red_threads = 256;
  const int red_blocks = static_cast<int>((N + red_threads - 1) / red_threads);
  std::vector<double> partial_sums(red_blocks, 0.0);
  double* d_partial = nullptr;
  err = cudaMalloc(reinterpret_cast<void**>(&d_partial), red_blocks * sizeof(double));
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMalloc partial"); }

  XCEnergyKernel<<<red_blocks, red_threads, red_threads * sizeof(double)>>>(
      d_eps_xc, d_rho, N, dv, d_partial);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    cudaFree(d_partial);
    cleanup();
    return CudaStatus(err, "XCEnergyKernel");
  }

  err = cudaMemcpy(partial_sums.data(), d_partial,
                   red_blocks * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_partial);
  cleanup();
  if (err != cudaSuccess) { return CudaStatus(err, "cudaMemcpy partial D2H"); }

  result.xc_energy = 0.0;
  for (double s : partial_sums) result.xc_energy += s;

  // Ledger.
  tides::tile::PrecisionDescriptor desc;
  desc.storage = tides::tile::NumericFormat::kFloat64;
  desc.compute = tides::tile::NumericFormat::kFloat64;
  desc.reduction = tides::tile::NumericFormat::kFloat64;
  desc.determinism = tides::tile::DeterminismMode::kDeterministic;
  desc.label = "cuda-xc-lda";
  result.ledger.Add(tides::tile::OperationLedgerEntry{
      tides::tile::OperationKind::kXcFunctional,
      desc,
      tides::tile::ErrorBudget{tides::tile::ErrorMetric::kAbsolute, 0.0,
          "GPU LDA XC vs CPU reference"},
      0.0, static_cast<std::uint64_t>(N), static_cast<std::uint64_t>(N), 0,
      "CUDA LDA-PW92 XC evaluation on grid"});

  return result;
}

// PBE GGA evaluation via libxc (CPU) + GPU energy reduction.
// The functional evaluation is done on CPU via libxc, then the energy
// reduction is performed on GPU for consistency with the LDA path.
[[nodiscard]] Result<XCGpuResult> XCEvalPbeCuda(
    const UniformGrid3D& grid,
    const std::vector<double>& rho) {
  const std::size_t N = grid.total_points();
  XCGpuResult result;
  result.n_points = N;

  if (N == 0) return result;
  if (rho.size() != N)
    return Status::InvalidArgument("rho size mismatch with grid");

  const auto [n0, n1, n2] = grid.n;
  const auto [h0, h1, h2] = grid.h;
  const double dv = h0 * h1 * h2;

  // Evaluate PBE on CPU using libxc.
  auto pbe = LibxcFunctional::EvalPBEOnGrid(n0, n1, n2, h0, h1, h2, rho);

  // Copy results to GPU for energy reduction.
  double* d_rho = nullptr;
  double* d_eps_xc = nullptr;
  double* d_partial = nullptr;

  auto cleanup = [&]() {
    if (d_rho) cudaFree(d_rho);
    if (d_eps_xc) cudaFree(d_eps_xc);
    if (d_partial) cudaFree(d_partial);
  };

  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_rho), N * sizeof(double));
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMalloc rho"); }
  err = cudaMemcpy(d_rho, rho.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMemcpy rho"); }

  err = cudaMalloc(reinterpret_cast<void**>(&d_eps_xc), N * sizeof(double));
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMalloc eps_xc"); }
  err = cudaMemcpy(d_eps_xc, pbe.eps_xc.data(), N * sizeof(double),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMemcpy eps_xc"); }

  // Energy reduction on GPU.
  const int red_threads = 256;
  const int red_blocks = static_cast<int>((N + red_threads - 1) / red_threads);
  std::vector<double> partial_sums(red_blocks, 0.0);
  err = cudaMalloc(reinterpret_cast<void**>(&d_partial),
                   red_blocks * sizeof(double));
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMalloc partial"); }

  XCEnergyKernel<<<red_blocks, red_threads, red_threads * sizeof(double)>>>(
      d_eps_xc, d_rho, N, dv, d_partial);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "XCEnergyKernel"); }

  err = cudaMemcpy(partial_sums.data(), d_partial,
                   red_blocks * sizeof(double), cudaMemcpyDeviceToHost);
  cleanup();
  if (err != cudaSuccess) return CudaStatus(err, "cudaMemcpy partial D2H");

  result.xc_energy = 0.0;
  for (double s : partial_sums) result.xc_energy += s;
  result.vxc = std::move(pbe.vxc);
  result.eps_xc = std::move(pbe.eps_xc);

  // Ledger.
  tides::tile::PrecisionDescriptor desc;
  desc.storage = tides::tile::NumericFormat::kFloat64;
  desc.compute = tides::tile::NumericFormat::kFloat64;
  desc.reduction = tides::tile::NumericFormat::kFloat64;
  desc.determinism = tides::tile::DeterminismMode::kDeterministic;
  desc.label = "cuda-xc-pbe";
  result.ledger.Add(tides::tile::OperationLedgerEntry{
      tides::tile::OperationKind::kXcFunctional,
      desc,
      tides::tile::ErrorBudget{tides::tile::ErrorMetric::kAbsolute, 0.0,
          "GPU PBE XC (libxc) vs CPU reference"},
      0.0, static_cast<std::uint64_t>(N), static_cast<std::uint64_t>(N), 0,
      "CUDA PBE GGA XC evaluation (libxc functional + GPU reduction)"});

  return result;
}

}  // namespace tides::grid
