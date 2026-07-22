// Phase 3 (Inc 4a): GPU port of the five blocked-φ ops from blocked_ops.hpp.
//
// Each op mirrors the CPU reference math bit-for-bit in double precision:
//   1. BlockedOverlapGpu        — S = dv · Σ_block φφᵀ scattered into n×n
//   2. BlockedRhoGpu            — ρ(g) = Σ_ij P_ij φ_i φ_j, per block
//   3. BlockedRhoWithGradGpu    — ρ and ∇ρ (needs grad)
//   4. BlockedVmatGpu           — H_ij = dv · Σ_g v(g) φ_i φ_j, per block
//   5. BlockedGgaVmatGpu        — 4-plane GGA contraction (needs grad)
//
// Per-block strategy (matching the task spec):
//   - Upload each block's φ/grad + active list to device once.
//   - Per-block cuBLAS DGEMM for the skinny contraction.
//   - Scatter into the small n×n result with atomicAdd (S/H are small).
//   - For ρ/∇ρ, a custom reduction kernel writes grid-point results directly.
//
// The GGA plane convention follows VmatBuilder::BuildGgaHmatGemm:
//   wv_grad_c already carries 2·dv·vsigma·∇ρ_c — the builder does NOT
//   re-multiply.  This is the convention that bit E6/E8.
//
// Acceptance: max|GPU − CPU| ≤ 1e-8 for all five ops (standalone nvcc test).

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "grid/grid_blocking.hpp"
#include "grid/blocked_ops.hpp"  // for BlockedRhoGrad

namespace tides::grid {

// ---------------------------------------------------------------------------
// Error checking helpers (typed, on every CUDA call).
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::string cuda_error_string(cudaError_t e,
                                                     const char* ctx) {
  return std::string(ctx) + ": " + cudaGetErrorString(e);
}

[[nodiscard]] inline std::string cublas_error_string(cublasStatus_t e,
                                                      const char* ctx) {
  return std::string(ctx) + ": cuBLAS error " +
         std::to_string(static_cast<int>(e));
}

#define TIDES_CUDA_CHECK(expr)                                      \
  do {                                                              \
    cudaError_t _tides_e = (expr);                                   \
    if (_tides_e != cudaSuccess)                                    \
      throw std::runtime_error(::tides::grid::cuda_error_string(    \
          _tides_e, #expr));                                        \
  } while (0)

#define TIDES_CUBLAS_CHECK(expr)                                     \
  do {                                                               \
    cublasStatus_t _tides_e = (expr);                                \
    if (_tides_e != CUBLAS_STATUS_SUCCESS)                           \
      throw std::runtime_error(                                      \
          ::tides::grid::cublas_error_string(_tides_e, #expr));     \
  } while (0)

// ---------------------------------------------------------------------------
// RAII device buffer — no raw owning pointers in host code.
// ---------------------------------------------------------------------------
struct DeviceMem {
  void* ptr = nullptr;
  std::size_t sz = 0;

  explicit DeviceMem(std::size_t bytes) : sz(bytes) {
    if (bytes > 0) {
      TIDES_CUDA_CHECK(cudaMalloc(&ptr, bytes));
    }
  }
  ~DeviceMem() {
    if (ptr) cudaFree(ptr);
  }
  DeviceMem(const DeviceMem&) = delete;
  DeviceMem& operator=(const DeviceMem&) = delete;
  DeviceMem(DeviceMem&& o) noexcept : ptr(o.ptr), sz(o.sz) {
    o.ptr = nullptr;
    o.sz = 0;
  }
  DeviceMem& operator=(DeviceMem&& o) noexcept {
    if (this != &o) {
      if (ptr) cudaFree(ptr);
      ptr = o.ptr;
      sz = o.sz;
      o.ptr = nullptr;
      o.sz = 0;
    }
    return *this;
  }
  template <typename T>
  T* as() {
    return static_cast<T*>(ptr);
  }
  template <typename T>
  const T* as() const {
    return static_cast<const T*>(ptr);
  }
};

// Upload a host range to a freshly allocated device buffer.
inline DeviceMem to_device(const void* host, std::size_t bytes) {
  DeviceMem d(bytes);
  if (bytes > 0) {
    TIDES_CUDA_CHECK(
        cudaMemcpy(d.ptr, host, bytes, cudaMemcpyHostToDevice));
  }
  return d;
}

// ---------------------------------------------------------------------------
// RAII cuBLAS handle.
// ---------------------------------------------------------------------------
struct CublasHandle {
  cublasHandle_t h = nullptr;
  explicit CublasHandle(bool deterministic) {
    TIDES_CUBLAS_CHECK(cublasCreate(&h));
    if (deterministic) {
      TIDES_CUBLAS_CHECK(cublasSetMathMode(
          h, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    }
  }
  ~CublasHandle() {
    if (h) cublasDestroy(h);
  }
  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;
  operator cublasHandle_t() const { return h; }
};

// ---------------------------------------------------------------------------
// Device kernels (anonymous namespace — internal linkage).
// ---------------------------------------------------------------------------
namespace {

// Scatter a small na×na block into the global n×n matrix via atomicAdd.
// The block matrix is symmetric, so we scatter all (a, c) pairs.
__global__ void ScatterBlockKernel(const double* blk, const int* active,
                                   int na, int n, double* S) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= na * na) return;
  int a = idx / na;
  int c = idx % na;
  atomicAdd(&S[static_cast<std::int64_t>(active[a]) * n + active[c]],
            blk[idx]);
}

// Scale a [na][npts] array by a per-point weight: out[a,p] = w[p] * in[a,p].
__global__ void ScaleRowsKernel(double* out, const double* in,
                                const double* w, int na, int npts) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  int a = blockIdx.y * blockDim.y + threadIdx.y;
  if (p >= npts || a >= na) return;
  out[a * npts + p] = w[p] * in[a * npts + p];
}

// ρ reduction: for each point p in the block, compute ρ = Σ_a φ[a,p]·temp[a,p]
// and write to the global grid at the correct flat index.
__global__ void RhoReduceKernel(const double* phi, const double* temp,
                                double* rho_global, int na, int npts,
                                int nx, int ny, int ix0, int iy0, int iz0,
                                int gn0, int gn1) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= npts) return;
  int lz = p / (nx * ny);
  int rem = p % (nx * ny);
  int ly = rem / nx;
  int lx = rem % nx;
  std::int64_t g = static_cast<std::int64_t>(ix0 + lx) +
                   gn0 * (static_cast<std::int64_t>(iy0 + ly) +
                          gn1 * static_cast<std::int64_t>(iz0 + lz));
  double r = 0.0;
  for (int a = 0; a < na; ++a)
    r += phi[a * npts + p] * temp[a * npts + p];
  rho_global[g] = r;
}

// ρ + ∇ρ reduction (GGA): also computes grad_c = 2·Σ_a ∂φ_c[a,p]·temp[a,p].
__global__ void RhoGradReduceKernel(
    const double* phi, const double* temp, const double* grad,
    double* rho_global, double* gx_global, double* gy_global,
    double* gz_global, int na, int npts, int plane, int nx, int ny,
    int ix0, int iy0, int iz0, int gn0, int gn1) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= npts) return;
  int lz = p / (nx * ny);
  int rem = p % (nx * ny);
  int ly = rem / nx;
  int lx = rem % nx;
  std::int64_t g = static_cast<std::int64_t>(ix0 + lx) +
                   gn0 * (static_cast<std::int64_t>(iy0 + ly) +
                          gn1 * static_cast<std::int64_t>(iz0 + lz));
  double r = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;
  for (int a = 0; a < na; ++a) {
    double t = temp[a * npts + p];
    r += phi[a * npts + p] * t;
    gx += grad[0 * plane + a * npts + p] * t;
    gy += grad[1 * plane + a * npts + p] * t;
    gz += grad[2 * plane + a * npts + p] * t;
  }
  rho_global[g] = r;
  gx_global[g] = 2.0 * gx;
  gy_global[g] = 2.0 * gy;
  gz_global[g] = 2.0 * gz;
}

}  // namespace

// ===========================================================================
// 1. BlockedOverlapGpu  ≡  BlockedOverlap
//    S_ij = dv · Σ_g φ_i(g) φ_j(g), per-block DGEMM + atomicAdd scatter.
// ===========================================================================
inline std::vector<double> BlockedOverlapGpu(const UniformGrid3D& grid,
                                             const BlockedPhi& bp,
                                             bool deterministic = false) {
  const std::size_t n = bp.n_basis;
  const double dv = grid.h[0] * grid.h[1] * grid.h[2];
  std::vector<double> S(n * n, 0.0);

  DeviceMem d_S(n * n * sizeof(double));
  TIDES_CUDA_CHECK(cudaMemset(d_S.ptr, 0, n * n * sizeof(double)));

  CublasHandle handle(deterministic);

  for (const auto& blk : bp.blocks) {
    const int na = static_cast<int>(blk.active.size());
    if (na == 0) continue;
    const int npts = static_cast<int>(blk.n_pts());
    if (npts == 0) continue;

    DeviceMem d_phi = to_device(
        blk.phi.data(), blk.phi.size() * sizeof(double));
    DeviceMem d_active = to_device(
        blk.active.data(), blk.active.size() * sizeof(int));
    DeviceMem d_Sblk(static_cast<std::size_t>(na) * na * sizeof(double));

    // Sblk = dv · φ @ φᵀ   (φ is na×npts row-major = npts×na col-major)
    // Col-major: Sblk_cm = dv · φ_cmᵀ @ φ_cm
    const double alpha = dv, beta = 0.0;
    TIDES_CUBLAS_CHECK(cublasDgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, na, na, npts, &alpha,
        d_phi.as<double>(), npts, d_phi.as<double>(), npts, &beta,
        d_Sblk.as<double>(), na));

    // Scatter into global S with atomicAdd.
    int threads = 256;
    int blocks = (na * na + threads - 1) / threads;
    ScatterBlockKernel<<<blocks, threads>>>(
        d_Sblk.as<double>(), d_active.as<int>(), na, static_cast<int>(n),
        d_S.as<double>());
    TIDES_CUDA_CHECK(cudaGetLastError());
  }

  TIDES_CUDA_CHECK(cudaDeviceSynchronize());
  TIDES_CUDA_CHECK(cudaMemcpy(S.data(), d_S.ptr, n * n * sizeof(double),
                              cudaMemcpyDeviceToHost));
  return S;
}

// ===========================================================================
// 2. BlockedRhoGpu  ≡  BlockedRho
//    ρ(g) = Σ_ij P_ij φ_i(g) φ_j(g), per-block DGEMM + reduction kernel.
// ===========================================================================
inline std::vector<double> BlockedRhoGpu(const UniformGrid3D& grid,
                                         const BlockedPhi& bp,
                                         const std::vector<double>& P,
                                         bool deterministic = false) {
  const std::size_t n = bp.n_basis;
  const std::int64_t n_grid =
      static_cast<std::int64_t>(grid.n[0]) * grid.n[1] * grid.n[2];
  std::vector<double> rho(static_cast<std::size_t>(n_grid), 0.0);

  DeviceMem d_rho(static_cast<std::size_t>(n_grid) * sizeof(double));
  TIDES_CUDA_CHECK(cudaMemset(d_rho.ptr, 0, n_grid * sizeof(double)));

  CublasHandle handle(deterministic);
  const int gn0 = static_cast<int>(grid.n[0]);
  const int gn1 = static_cast<int>(grid.n[1]);

  for (const auto& blk : bp.blocks) {
    const int na = static_cast<int>(blk.active.size());
    if (na == 0) continue;
    const int npts = static_cast<int>(blk.n_pts());
    if (npts == 0) continue;

    // Extract P_sub[na×na] from full P using active list.
    std::vector<double> Psub(static_cast<std::size_t>(na) * na);
    for (int a = 0; a < na; ++a) {
      const int ia = blk.active[a];
      for (int c = 0; c < na; ++c)
        Psub[a * na + c] = P[static_cast<std::size_t>(ia) * n + blk.active[c]];
    }

    DeviceMem d_phi = to_device(
        blk.phi.data(), blk.phi.size() * sizeof(double));
    DeviceMem d_Psub = to_device(
        Psub.data(), Psub.size() * sizeof(double));
    DeviceMem d_temp(static_cast<std::size_t>(na) * npts * sizeof(double));

    // temp = P_sub @ φ   (row-major: (na×na) @ (na×npts) → na×npts)
    // Col-major: temp_cm = φ_cm @ P_sub_cm  (npts×na = npts×na @ na×na)
    const double alpha = 1.0, beta = 0.0;
    TIDES_CUBLAS_CHECK(cublasDgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, npts, na, na, &alpha,
        d_phi.as<double>(), npts, d_Psub.as<double>(), na, &beta,
        d_temp.as<double>(), npts));

    // ρ(p) = Σ_a φ[a,p] · temp[a,p], written to global grid.
    int threads = 256;
    int blocks = (npts + threads - 1) / threads;
    RhoReduceKernel<<<blocks, threads>>>(
        d_phi.as<double>(), d_temp.as<double>(), d_rho.as<double>(), na, npts,
        blk.nx, blk.ny, blk.ix0, blk.iy0, blk.iz0, gn0, gn1);
    TIDES_CUDA_CHECK(cudaGetLastError());
  }

  TIDES_CUDA_CHECK(cudaDeviceSynchronize());
  TIDES_CUDA_CHECK(cudaMemcpy(rho.data(), d_rho.ptr,
                              n_grid * sizeof(double),
                              cudaMemcpyDeviceToHost));
  return rho;
}

// ===========================================================================
// 3. BlockedRhoWithGradGpu  ≡  BlockedRhoWithGrad
//    ρ and ∇ρ together (GGA).  Needs AddBlockedGrad on bp.
//    ∇ρ_c(g) = 2·Σ_i ∂φ_i,c(g) · temp_i(g),  temp = P·φ  (P symmetric)
// ===========================================================================
inline BlockedRhoGrad BlockedRhoWithGradGpu(
    const UniformGrid3D& grid, const BlockedPhi& bp,
    const std::vector<double>& P, bool deterministic = false) {
  const std::size_t n = bp.n_basis;
  const std::int64_t n_grid =
      static_cast<std::int64_t>(grid.n[0]) * grid.n[1] * grid.n[2];

  BlockedRhoGrad out;
  out.rho.assign(static_cast<std::size_t>(n_grid), 0.0);
  out.grad_x.assign(static_cast<std::size_t>(n_grid), 0.0);
  out.grad_y.assign(static_cast<std::size_t>(n_grid), 0.0);
  out.grad_z.assign(static_cast<std::size_t>(n_grid), 0.0);

  DeviceMem d_rho(n_grid * sizeof(double));
  DeviceMem d_gx(n_grid * sizeof(double));
  DeviceMem d_gy(n_grid * sizeof(double));
  DeviceMem d_gz(n_grid * sizeof(double));
  TIDES_CUDA_CHECK(cudaMemset(d_rho.ptr, 0, n_grid * sizeof(double)));
  TIDES_CUDA_CHECK(cudaMemset(d_gx.ptr, 0, n_grid * sizeof(double)));
  TIDES_CUDA_CHECK(cudaMemset(d_gy.ptr, 0, n_grid * sizeof(double)));
  TIDES_CUDA_CHECK(cudaMemset(d_gz.ptr, 0, n_grid * sizeof(double)));

  CublasHandle handle(deterministic);
  const int gn0 = static_cast<int>(grid.n[0]);
  const int gn1 = static_cast<int>(grid.n[1]);

  for (const auto& blk : bp.blocks) {
    const int na = static_cast<int>(blk.active.size());
    if (na == 0) continue;
    const int npts = static_cast<int>(blk.n_pts());
    if (npts == 0) continue;
    const int plane = na * npts;

    // Extract P_sub.
    std::vector<double> Psub(static_cast<std::size_t>(na) * na);
    for (int a = 0; a < na; ++a) {
      const int ia = blk.active[a];
      for (int c = 0; c < na; ++c)
        Psub[a * na + c] = P[static_cast<std::size_t>(ia) * n + blk.active[c]];
    }

    DeviceMem d_phi = to_device(
        blk.phi.data(), blk.phi.size() * sizeof(double));
    DeviceMem d_grad = to_device(
        blk.grad.data(), blk.grad.size() * sizeof(double));
    DeviceMem d_Psub = to_device(
        Psub.data(), Psub.size() * sizeof(double));
    DeviceMem d_temp(static_cast<std::size_t>(na) * npts * sizeof(double));

    // temp = P_sub @ φ  (same GEMM as rho).
    const double alpha = 1.0, beta = 0.0;
    TIDES_CUBLAS_CHECK(cublasDgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, npts, na, na, &alpha,
        d_phi.as<double>(), npts, d_Psub.as<double>(), na, &beta,
        d_temp.as<double>(), npts));

    // ρ(p) = Σ_a φ[a,p]·temp[a,p];  ∇ρ_c(p) = 2·Σ_a grad_c[a,p]·temp[a,p].
    int threads = 256;
    int blocks = (npts + threads - 1) / threads;
    RhoGradReduceKernel<<<blocks, threads>>>(
        d_phi.as<double>(), d_temp.as<double>(), d_grad.as<double>(),
        d_rho.as<double>(), d_gx.as<double>(), d_gy.as<double>(),
        d_gz.as<double>(), na, npts, plane, blk.nx, blk.ny, blk.ix0,
        blk.iy0, blk.iz0, gn0, gn1);
    TIDES_CUDA_CHECK(cudaGetLastError());
  }

  TIDES_CUDA_CHECK(cudaDeviceSynchronize());
  TIDES_CUDA_CHECK(cudaMemcpy(out.rho.data(), d_rho.ptr,
                              n_grid * sizeof(double),
                              cudaMemcpyDeviceToHost));
  TIDES_CUDA_CHECK(cudaMemcpy(out.grad_x.data(), d_gx.ptr,
                              n_grid * sizeof(double),
                              cudaMemcpyDeviceToHost));
  TIDES_CUDA_CHECK(cudaMemcpy(out.grad_y.data(), d_gy.ptr,
                              n_grid * sizeof(double),
                              cudaMemcpyDeviceToHost));
  TIDES_CUDA_CHECK(cudaMemcpy(out.grad_z.data(), d_gz.ptr,
                              n_grid * sizeof(double),
                              cudaMemcpyDeviceToHost));
  return out;
}

// ===========================================================================
// 4. BlockedVmatGpu  ≡  BlockedVmat
//    H_ij = dv · Σ_g v(g) φ_i(g) φ_j(g), per-block DGEMM + atomicAdd scatter.
// ===========================================================================
inline std::vector<double> BlockedVmatGpu(const UniformGrid3D& grid,
                                         const BlockedPhi& bp,
                                         const std::vector<double>& v,
                                         bool deterministic = false) {
  const std::size_t n = bp.n_basis;
  const double dv = grid.h[0] * grid.h[1] * grid.h[2];
  std::vector<double> H(n * n, 0.0);

  DeviceMem d_H(n * n * sizeof(double));
  TIDES_CUDA_CHECK(cudaMemset(d_H.ptr, 0, n * n * sizeof(double)));

  CublasHandle handle(deterministic);

  for (const auto& blk : bp.blocks) {
    const int na = static_cast<int>(blk.active.size());
    if (na == 0) continue;
    const int npts = static_cast<int>(blk.n_pts());
    if (npts == 0) continue;

    // Gather v at the block's grid points (local order), matching the CPU.
    std::vector<double> vblk(static_cast<std::size_t>(npts));
    {
      std::int64_t p = 0;
      for (std::int32_t lz = 0; lz < blk.nz; ++lz) {
        const std::size_t iz = blk.iz0 + lz;
        for (std::int32_t ly = 0; ly < blk.ny; ++ly) {
          const std::size_t iy = blk.iy0 + ly;
          for (std::int32_t lx = 0; lx < blk.nx; ++lx, ++p)
            vblk[p] = v[grid.flatten(blk.ix0 + lx, iy, iz)];
        }
      }
    }

    DeviceMem d_phi = to_device(
        blk.phi.data(), blk.phi.size() * sizeof(double));
    DeviceMem d_vblk = to_device(
        vblk.data(), vblk.size() * sizeof(double));
    DeviceMem d_vphi(static_cast<std::size_t>(na) * npts * sizeof(double));
    DeviceMem d_active = to_device(
        blk.active.data(), blk.active.size() * sizeof(int));
    DeviceMem d_Hblk(static_cast<std::size_t>(na) * na * sizeof(double));

    // Scale: vphi[a,p] = vblk[p] · φ[a,p].
    {
      dim3 block(32, 8);
      dim3 grid_dim((npts + block.x - 1) / block.x,
                    (na + block.y - 1) / block.y);
      ScaleRowsKernel<<<grid_dim, block>>>(
          d_vphi.as<double>(), d_phi.as<double>(), d_vblk.as<double>(),
          na, npts);
      TIDES_CUDA_CHECK(cudaGetLastError());
    }

    // Hblk = dv · vphi @ φᵀ  (na×na).
    const double alpha = dv, beta = 0.0;
    TIDES_CUBLAS_CHECK(cublasDgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, na, na, npts, &alpha,
        d_vphi.as<double>(), npts, d_phi.as<double>(), npts, &beta,
        d_Hblk.as<double>(), na));

    // Scatter into global H.
    int threads = 256;
    int blocks = (na * na + threads - 1) / threads;
    ScatterBlockKernel<<<blocks, threads>>>(
        d_Hblk.as<double>(), d_active.as<int>(), na, static_cast<int>(n),
        d_H.as<double>());
    TIDES_CUDA_CHECK(cudaGetLastError());
  }

  TIDES_CUDA_CHECK(cudaDeviceSynchronize());
  TIDES_CUDA_CHECK(cudaMemcpy(H.data(), d_H.ptr, n * n * sizeof(double),
                              cudaMemcpyDeviceToHost));
  return H;
}

// ===========================================================================
// 5. BlockedGgaVmatGpu  ≡  BlockedGgaVmat
//    H_ij = Σ_g [ wv_rho·φ_i φ_j
//              + Σ_c wv_grad_c·(∂φ_i,c φ_j + φ_i ∂φ_j,c) ]
//    wv_rho and wv_grad already carry dv (no extra dv — E15 convention).
//    Decomposition (matching BuildGgaHmatGemm):
//      H = (wr·φ) @ φᵀ  +  Σ_c [ (wg_c·∂φ_c) @ φᵀ + φ @ (wg_c·∂φ_c)ᵀ ]
// ===========================================================================
inline std::vector<double> BlockedGgaVmatGpu(
    const UniformGrid3D& grid, const BlockedPhi& bp,
    const std::vector<double>& wv_rho, const std::vector<double>& wv_grad_x,
    const std::vector<double>& wv_grad_y, const std::vector<double>& wv_grad_z,
    bool deterministic = false) {
  const std::size_t n = bp.n_basis;
  std::vector<double> H(n * n, 0.0);

  DeviceMem d_H(n * n * sizeof(double));
  TIDES_CUDA_CHECK(cudaMemset(d_H.ptr, 0, n * n * sizeof(double)));

  CublasHandle handle(deterministic);

  // The four weighted-field planes, gathered per block.
  const std::vector<double>* wv_planes[4] = {&wv_rho, &wv_grad_x,
                                              &wv_grad_y, &wv_grad_z};

  for (const auto& blk : bp.blocks) {
    const int na = static_cast<int>(blk.active.size());
    if (na == 0) continue;
    const int npts = static_cast<int>(blk.n_pts());
    if (npts == 0) continue;
    const int plane = na * npts;

    // Gather the 4 weighted planes at the block's points (local order).
    std::array<std::vector<double>, 4> wp;
    for (int pl = 0; pl < 4; ++pl) {
      wp[pl].resize(npts);
      std::int64_t p = 0;
      for (std::int32_t lz = 0; lz < blk.nz; ++lz) {
        const std::size_t iz = blk.iz0 + lz;
        for (std::int32_t ly = 0; ly < blk.ny; ++ly) {
          const std::size_t iy = blk.iy0 + ly;
          for (std::int32_t lx = 0; lx < blk.nx; ++lx, ++p)
            wp[pl][p] = (*wv_planes[pl])[grid.flatten(blk.ix0 + lx, iy, iz)];
        }
      }
    }

    DeviceMem d_phi = to_device(
        blk.phi.data(), blk.phi.size() * sizeof(double));
    DeviceMem d_grad = to_device(
        blk.grad.data(), blk.grad.size() * sizeof(double));
    DeviceMem d_active = to_device(
        blk.active.data(), blk.active.size() * sizeof(int));
    DeviceMem d_Hblk(static_cast<std::size_t>(na) * na * sizeof(double));
    TIDES_CUDA_CHECK(cudaMemset(d_Hblk.ptr, 0, na * na * sizeof(double)));

    // Temp buffer for scaled arrays.
    DeviceMem d_temp(static_cast<std::size_t>(na) * npts * sizeof(double));

    const double alpha = 1.0;

    // --- Rho part: Hblk += (wr·φ) @ φᵀ ---
    {
      DeviceMem d_wp = to_device(wp[0].data(), npts * sizeof(double));
      dim3 block(32, 8);
      dim3 grid_dim((npts + block.x - 1) / block.x,
                    (na + block.y - 1) / block.y);
      ScaleRowsKernel<<<grid_dim, block>>>(
          d_temp.as<double>(), d_phi.as<double>(), d_wp.as<double>(), na,
          npts);
      TIDES_CUDA_CHECK(cudaGetLastError());

      double beta = 0.0;  // init Hblk
      TIDES_CUBLAS_CHECK(cublasDgemm(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, na, na, npts, &alpha,
          d_temp.as<double>(), npts, d_phi.as<double>(), npts, &beta,
          d_Hblk.as<double>(), na));
    }

    // --- Grad parts: for each direction c, Hblk += (wg·∂φ_c)@φᵀ + φ@(wg·∂φ_c)ᵀ
    // ---
    for (int c = 0; c < 3; ++c) {
      DeviceMem d_wp = to_device(wp[c + 1].data(), npts * sizeof(double));
      dim3 block(32, 8);
      dim3 grid_dim((npts + block.x - 1) / block.x,
                    (na + block.y - 1) / block.y);
      // Scale grad_c by wg_c.
      ScaleRowsKernel<<<grid_dim, block>>>(
          d_temp.as<double>(),
          d_grad.as<double>() + c * plane,  // offset to grad plane c
          d_wp.as<double>(), na, npts);
      TIDES_CUDA_CHECK(cudaGetLastError());

      // GEMM 1: W = (wg·∂φ_c) @ φᵀ → Hblk += W
      double beta = 1.0;
      TIDES_CUBLAS_CHECK(cublasDgemm(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, na, na, npts, &alpha,
          d_temp.as<double>(), npts, d_phi.as<double>(), npts, &beta,
          d_Hblk.as<double>(), na));

      // GEMM 2: W = φ @ (wg·∂φ_c)ᵀ → Hblk += W  (the transpose term)
      TIDES_CUBLAS_CHECK(cublasDgemm(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, na, na, npts, &alpha,
          d_phi.as<double>(), npts, d_temp.as<double>(), npts, &beta,
          d_Hblk.as<double>(), na));
    }

    // Scatter Hblk into global H.
    int threads = 256;
    int blocks = (na * na + threads - 1) / threads;
    ScatterBlockKernel<<<blocks, threads>>>(
        d_Hblk.as<double>(), d_active.as<int>(), na, static_cast<int>(n),
        d_H.as<double>());
    TIDES_CUDA_CHECK(cudaGetLastError());
  }

  TIDES_CUDA_CHECK(cudaDeviceSynchronize());
  TIDES_CUDA_CHECK(cudaMemcpy(H.data(), d_H.ptr, n * n * sizeof(double),
                              cudaMemcpyDeviceToHost));
  return H;
}

}  // namespace tides::grid
