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

// AUDIT B2: Analytic derivative of PW92 correlation energy w.r.t. rs.
// Replaces central finite differences (h=1e-6) that introduce ~1e-11 noise
// into the "oracle" comparisons (A5). The analytic form is exact and ~3× cheaper.
//
// eps_c(rs) = -2a(1 + a1*rs) * ln(1 + 1/(2a*Q(rs)))
// Q = b1*sqrt(rs) + b2*rs + b3*rs*sqrt(rs) + b4*rs^2
// Q' = b1/(2*sqrt(rs)) + b2 + (3/2)*b3*sqrt(rs) + 2*b4*rs
// d(eps_c)/d(rs) = -2a*a1*ln(1 + 1/(2a*Q))
//                  + (-2a*(1+a1*rs)) * (-Q' / (2a*Q^2 + Q))
__device__ double DEpsCParamagneticDRsDevice(double rs) {
  const double a = 0.0310907, a1 = 0.2137;
  const double b1 = 7.5957, b2 = 3.5876, b3 = 1.6382, b4 = 0.49294;
  const double sr = sqrt(rs);
  const double Q = b1 * sr + b2 * rs + b3 * rs * sr + b4 * rs * rs;
  const double Qprime = b1 / (2.0 * sr) + b2 + 1.5 * b3 * sr + 2.0 * b4 * rs;
  const double log_term = log(1.0 + 1.0 / (2.0 * a * Q));
  const double dlog = -Qprime / (2.0 * a * Q * Q + Q);
  return -2.0 * a * a1 * log_term + (-2.0 * a * (1.0 + a1 * rs)) * dlog;
}

__device__ double DEpsCFerromagneticDRsDevice(double rs) {
  const double a = 0.015545, a1 = 0.20548;
  const double b1 = 14.1189, b2 = 6.1977, b3 = 3.3662, b4 = 0.62517;
  const double sr = sqrt(rs);
  const double Q = b1 * sr + b2 * rs + b3 * rs * sr + b4 * rs * rs;
  const double Qprime = b1 / (2.0 * sr) + b2 + 1.5 * b3 * sr + 2.0 * b4 * rs;
  const double log_term = log(1.0 + 1.0 / (2.0 * a * Q));
  const double dlog = -Qprime / (2.0 * a * Q * Q + Q);
  return -2.0 * a * a1 * log_term + (-2.0 * a * (1.0 + a1 * rs)) * dlog;
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

  // AUDIT B10: Use persistent arena + stream instead of per-call alloc/free/sync.
  GpuArena& arena = GpuArena::Instance();

  // Allocate device memory via arena (reuses cached buffers).
  double* d_rho = static_cast<double*>(arena.Alloc(N * sizeof(double)));
  double* d_vxc = static_cast<double*>(arena.Alloc(N * sizeof(double)));
  double* d_eps_xc = static_cast<double*>(arena.Alloc(N * sizeof(double)));

  if (!d_rho || !d_vxc || !d_eps_xc) {
    if (d_rho) arena.Free(d_rho);
    if (d_vxc) arena.Free(d_vxc);
    if (d_eps_xc) arena.Free(d_eps_xc);
    return Status::IoError("GPU arena allocation failed for XC evaluation");
  }

  // Async H2D on persistent stream.
  cudaError_t err = arena.H2D(d_rho, rho.data(), N * sizeof(double));
  if (err != cudaSuccess) {
    arena.Free(d_rho); arena.Free(d_vxc); arena.Free(d_eps_xc);
    return CudaStatus(err, "arena H2D rho");
  }

  // Launch XC evaluation kernel on arena stream.
  const int threads = 256;
  const int blocks = static_cast<int>((N + threads - 1) / threads);
  auto kernel_start = std::chrono::steady_clock::now();
  XCEvalKernel<<<blocks, threads, 0, arena.Stream()>>>(d_rho, d_vxc, d_eps_xc, N, zeta);
  err = arena.Sync();
  auto kernel_end = std::chrono::steady_clock::now();
  if (err != cudaSuccess) {
    arena.Free(d_rho); arena.Free(d_vxc); arena.Free(d_eps_xc);
    return CudaStatus(err, "XCEvalKernel");
  }

  result.kernel_ms =
      std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();

  // Copy results back via arena async D2H.
  result.vxc.resize(N);
  result.eps_xc.resize(N);
  err = arena.D2H(result.vxc.data(), d_vxc, N * sizeof(double));
  if (err != cudaSuccess) {
    arena.Free(d_rho); arena.Free(d_vxc); arena.Free(d_eps_xc);
    return CudaStatus(err, "arena D2H vxc");
  }
  err = arena.D2H(result.eps_xc.data(), d_eps_xc, N * sizeof(double));
  if (err != cudaSuccess) {
    arena.Free(d_rho); arena.Free(d_vxc); arena.Free(d_eps_xc);
    return CudaStatus(err, "arena D2H eps_xc");
  }
  arena.Sync();

  // Compute XC energy via GPU reduction.
  const int red_threads = 256;
  const int red_blocks = static_cast<int>((N + red_threads - 1) / red_threads);
  std::vector<double> partial_sums(red_blocks, 0.0);
  double* d_partial = static_cast<double*>(arena.Alloc(red_blocks * sizeof(double)));
  if (!d_partial) {
    arena.Free(d_rho); arena.Free(d_vxc); arena.Free(d_eps_xc);
    return Status::IoError("GPU arena allocation failed for reduction");
  }

  XCEnergyKernel<<<red_blocks, red_threads, red_threads * sizeof(double), arena.Stream()>>>(
      d_eps_xc, d_rho, N, dv, d_partial);
  err = arena.Sync();
  if (err != cudaSuccess) {
    arena.Free(d_rho); arena.Free(d_vxc); arena.Free(d_eps_xc); arena.Free(d_partial);
    return CudaStatus(err, "XCEnergyKernel");
  }

  err = arena.D2H(partial_sums.data(), d_partial, red_blocks * sizeof(double));
  // Return all blocks to arena.
  arena.Free(d_rho); arena.Free(d_vxc); arena.Free(d_eps_xc); arena.Free(d_partial);
  if (err != cudaSuccess) { return CudaStatus(err, "arena D2H partial"); }

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

// AUDIT B1: PBE CUDA path deleted.
// The previous implementation was broken (missing -2∇·(v_σ∇ρ) GGA term)
// and was dead code (driver is LDA-only). It also used the anti-pattern
// of CPU libxc eval + GPU-only reduction with two PCIe round-trips.
// PBE will be implemented as part of the fused Tier-0 XC engine (P2.7).
[[nodiscard]] Result<XCGpuResult> XCEvalPbeCuda(
    const UniformGrid3D& grid,
    const std::vector<double>& rho) {
  (void)grid;
  (void)rho;
  return Status::Unimplemented(
      "PBE CUDA path deleted (audit B1): missing GGA term, dead code. "
      "Use fused Tier-0 XC engine when implemented (P2.7).");
}

}  // namespace tides::grid
