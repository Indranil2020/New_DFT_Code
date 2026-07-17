// T3.3: GPU v(r) -> H tiles adjoint map.
//
// H_ij = integral v(r) * phi_i(r) * phi_j(r) d^3r
//
// The kernel processes (i, j) pairs in parallel. Each thread block handles
// one (i, j) pair and reduces over grid points using shared memory.
//
// Observable (T3.3): adjointness |<A P, w> - <P, A^T w>| <= 1e-12 (FP64).

#include "grid/vmat_build_gpu.hpp"
#include "grid/vmat_build.hpp"
#include "grid/gpu_arena.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "common/status.hpp"
#include "tile/precision.hpp"

namespace tides::grid {
namespace {

[[nodiscard]] Status CudaStatus(cudaError_t error, const char* context) {
  if (error == cudaSuccess) return Status::Ok();
  return Status::IoError(std::string(context) + ": " +
                         cudaGetErrorString(error));
}

[[nodiscard]] Status CublasStatus(cublasStatus_t s, const char* ctx) {
  if (s == CUBLAS_STATUS_SUCCESS) return Status::Ok();
  return Status::IoError(std::string(ctx) + ": cuBLAS error " +
                         std::to_string(static_cast<int>(s)));
}

// Scale each column g of phi by wv[g]:  temp[mu, g] = wv[g] * phi[mu, g]
// 2D grid: (g, mu) for coalesced memory access along g.
__global__ void ScaleColumnsKernel(
    double* out, const double* phi, const double* wv,
    std::int64_t nbasis, std::int64_t np, std::int64_t stride) {
  const std::int64_t g = blockIdx.x * blockDim.x + threadIdx.x;
  const std::int64_t mu = blockIdx.y * blockDim.y + threadIdx.y;
  if (g >= np || mu >= nbasis) return;
  out[mu * stride + g] = wv[g] * phi[mu * stride + g];
}

// FP32 version: scale + cast to float in one pass.
__global__ void ScaleColumnsKernelF32(
    float* out, const double* phi, const double* wv,
    std::int64_t nbasis, std::int64_t np, std::int64_t stride) {
  const std::int64_t g = blockIdx.x * blockDim.x + threadIdx.x;
  const std::int64_t mu = blockIdx.y * blockDim.y + threadIdx.y;
  if (g >= np || mu >= nbasis) return;
  out[mu * stride + g] = static_cast<float>(wv[g] * phi[mu * stride + g]);
}

// Cast double array to float.
__global__ void CastToF32Kernel(
    float* out, const double* in, std::int64_t n) {
  std::int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = static_cast<float>(in[i]);
}

// Cast float array to double.
__global__ void CastToF64Kernel(
    double* out, const float* in, std::int64_t n) {
  std::int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = static_cast<double>(in[i]);
}

// Accumulate FP32 matrix into FP64: V_f64[i,j] += (double)W_f32[i,j]
__global__ void AccumulateF32ToF64Kernel(
    double* V, const float* W, std::int64_t n) {
  std::int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n * n) return;
  V[idx] += static_cast<double>(W[idx]);
}

// 2D slice cast: out[mu*stride + g] = (float)in[mu*stride + g] for g in [0,k_cur)
__global__ void CastSliceToF32Kernel(
    float* out, const double* in,
    std::int64_t nbasis, std::int64_t k_cur, std::int64_t stride) {
  const std::int64_t g = blockIdx.x * blockDim.x + threadIdx.x;
  const std::int64_t mu = blockIdx.y * blockDim.y + threadIdx.y;
  if (g >= k_cur || mu >= nbasis) return;
  out[mu * stride + g] = static_cast<float>(in[mu * stride + g]);
}

// Symmetrize-accumulate FP32 into FP64: V_f64 += W_f32 + W_f32^T
__global__ void SymmetrizeAccumulateF32ToF64Kernel(
    double* V, const float* W, std::int64_t n) {
  std::int64_t mu = blockIdx.x * blockDim.x + threadIdx.x;
  if (mu >= n) return;
  for (std::int64_t nu = mu; nu < n; ++nu) {
    const double val = static_cast<double>(W[mu * n + nu] + W[nu * n + mu]);
    V[mu * n + nu] += val;
    if (nu != mu) V[nu * n + mu] += val;
  }
}

// Add W + W^T to V (symmetric accumulation). V += W + W^T.
// FP32 version.
__global__ void SymmetrizeAddKernelF32(
    float* V, const float* W, std::int64_t n) {
  const std::int64_t mu = blockIdx.x * blockDim.x + threadIdx.x;
  if (mu >= n) return;
  for (std::int64_t nu = mu; nu < n; ++nu) {
    const float val = W[mu * n + nu] + W[nu * n + mu];
    V[mu * n + nu] += val;
    if (nu != mu) V[nu * n + mu] += val;
  }
}

// Add W + W^T to V (symmetric accumulation). V += W + W^T.
__global__ void SymmetrizeAddKernel(
    double* V, const double* W, std::int64_t n) {
  const std::int64_t mu = blockIdx.x * blockDim.x + threadIdx.x;
  if (mu >= n) return;
  for (std::int64_t nu = mu; nu < n; ++nu) {
    const double val = W[mu * n + nu] + W[nu * n + mu];
    V[mu * n + nu] += val;
    if (nu != mu) V[nu * n + mu] += val;
  }
}

// Symmetrize-accumulate a strided block from concatenated W:
// W is col-major with leading dim ldw. Block starts at row offset.
// V[mu,nu] += W[offset+mu, nu] + W[offset+nu, mu]
// W[offset+i, j] = W_col[offset + i + j*ldw]
__global__ void SymmetrizeAddStridedKernel(
    double* V, const double* W,
    std::int64_t offset, std::int64_t n, std::int64_t ldw) {
  const std::int64_t mu = blockIdx.x * blockDim.x + threadIdx.x;
  if (mu >= n) return;
  for (std::int64_t nu = mu; nu < n; ++nu) {
    const double val = W[offset + mu + nu * ldw] + W[offset + nu + mu * ldw];
    V[mu * n + nu] += val;
    if (nu != mu) V[nu * n + mu] += val;
  }
}

// Add a strided block from concatenated W to V (no symmetrize):
// V[mu,nu] += W[offset+mu, nu]  for all mu, nu
// W is col-major with leading dim ldw.
__global__ void StridedAddKernel(
    double* V, const double* W,
    std::int64_t offset, std::int64_t n, std::int64_t ldw) {
  const std::int64_t mu = blockIdx.x * blockDim.x + threadIdx.x;
  if (mu >= n) return;
  for (std::int64_t nu = 0; nu < n; ++nu) {
    V[mu * n + nu] += W[offset + mu + nu * ldw];
  }
}

// Grid screening kernels for GGA vmat (same concept as LDA path).
__global__ void GgaBuildCompactionIndexKernel(
    const double* phi, std::int64_t nbasis, std::int64_t np,
    std::int64_t stride, int* index, int* compact_count) {
  std::int64_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= np) return;
  bool active = false;
  for (std::int64_t mu = 0; mu < nbasis; ++mu) {
    if (phi[mu * stride + g] != 0.0) { active = true; break; }
  }
  if (active) {
    int pos = atomicAdd(compact_count, 1);
    index[pos] = static_cast<int>(g);
  }
}

__global__ void GgaCompactPhiKernel(
    double* out, const double* phi, const int* index,
    std::int64_t nbasis, std::int64_t np_compact, std::int64_t stride) {
  std::int64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
  std::int64_t mu = blockIdx.y * blockDim.y + threadIdx.y;
  if (pos >= np_compact || mu >= nbasis) return;
  out[mu * np_compact + pos] = phi[mu * stride + index[pos]];
}

// Compact grad_phi: grad is [3][nbasis][stride], compact to [3][nbasis][np_compact]
__global__ void GgaCompactGradKernel(
    double* out, const double* grad, const int* index,
    std::int64_t nbasis, std::int64_t np_compact, std::int64_t stride) {
  std::int64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
  std::int64_t mu = blockIdx.y * blockDim.y + threadIdx.y;
  std::int64_t comp = threadIdx.z;
  if (pos >= np_compact || mu >= nbasis || comp >= 3) return;
  std::int64_t basis_plane = nbasis * stride;
  out[comp * nbasis * np_compact + mu * np_compact + pos] =
      grad[comp * basis_plane + mu * stride + index[pos]];
}

__global__ void GgaCompactWvKernel(
    double* out, const double* wv, const int* index, std::int64_t np_compact) {
  std::int64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= np_compact) return;
  out[pos] = wv[index[pos]];
}

// Scale compacted columns: out[mu * np_c + pos] = wv[pos] * in[mu * np_c + pos]
__global__ void ScaleCompactColumnsKernel(
    double* out, const double* in, const double* wv,
    std::int64_t nbasis, std::int64_t np_c) {
  std::int64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
  std::int64_t mu = blockIdx.y * blockDim.y + threadIdx.y;
  if (pos >= np_c || mu >= nbasis) return;
  out[mu * np_c + pos] = wv[pos] * in[mu * np_c + pos];
}

// cuBLAS handle + temp buffers for GEMM-based GGA vmat.
// Screen buffers are sized to np_compact (not full stride) after counting.
// Full-path d_temp_g is allocated only if screening cannot be used.
struct GgaVmatGemmCache {
  cublasHandle_t handle = nullptr;
  double* d_temp_g = nullptr;      // [4 * nbasis * stride] full unscreened path
  double* d_W = nullptr;           // [nbasis * nbasis] temp GEMM block (grad)
  double* d_W4 = nullptr;          // [4 * nbasis * nbasis] full unscreened concat
  // Grid screening buffers.
  int* d_compact_index = nullptr;      // [stride]
  double* d_phi_compact = nullptr;     // [nbasis * np_c]
  double* d_grad_compact = nullptr;    // [3 * nbasis * np_c]
  double* d_wv_rho_c = nullptr;        // [np_c]
  double* d_wv_grad_c = nullptr;       // [3 * np_c]
  double* d_temp_g_c = nullptr;        // [nbasis * np_c] single scaled plane
  int* d_screen_count = nullptr;
  std::int64_t np_compact = 0;
  std::int64_t screen_cap = 0;         // capacity of compact buffers
  bool screen_initialized = false;
  bool compact_ready = false;          // phi/grad compacted for current index
  std::int64_t cached_nbasis = 0, cached_stride = 0;

  void free_screen_data() {
    if (d_phi_compact) cudaFree(d_phi_compact);
    if (d_grad_compact) cudaFree(d_grad_compact);
    if (d_wv_rho_c) cudaFree(d_wv_rho_c);
    if (d_wv_grad_c) cudaFree(d_wv_grad_c);
    if (d_temp_g_c) cudaFree(d_temp_g_c);
    d_phi_compact = nullptr;
    d_grad_compact = nullptr;
    d_wv_rho_c = nullptr;
    d_wv_grad_c = nullptr;
    d_temp_g_c = nullptr;
    screen_cap = 0;
    compact_ready = false;
  }

  void free_screen_all() {
    free_screen_data();
    if (d_compact_index) cudaFree(d_compact_index);
    if (d_screen_count) cudaFree(d_screen_count);
    d_compact_index = nullptr;
    d_screen_count = nullptr;
    screen_initialized = false;
    np_compact = 0;
  }

  ~GgaVmatGemmCache() {
    if (handle) cublasDestroy(handle);
    if (d_temp_g) cudaFree(d_temp_g);
    if (d_W) cudaFree(d_W);
    free_screen_all();
  }

  bool ensure(std::int64_t nbasis, std::int64_t stride, cudaStream_t stream) {
    if (!handle) {
      if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return false;
      cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
    }
    cublasSetStream(handle, stream);
    if (cached_nbasis != nbasis || cached_stride != stride) {
      if (d_temp_g) cudaFree(d_temp_g);
      if (d_W) cudaFree(d_W);
      if (d_W4) cudaFree(d_W4);
      d_temp_g = nullptr;
      d_W = nullptr;
      d_W4 = nullptr;
      free_screen_all();
      // W and W4 are needed for both screen and full paths.
      if (cudaMalloc(&d_W, nbasis * nbasis * sizeof(double)) != cudaSuccess) {
        d_W = nullptr;
        return false;
      }
      if (cudaMalloc(&d_W4, 4 * nbasis * nbasis * sizeof(double)) != cudaSuccess) {
        d_W4 = nullptr;
        return false;
      }
      cached_nbasis = nbasis;
      cached_stride = stride;
    }
    return d_W != nullptr && d_W4 != nullptr;
  }

  // Phase-1: index + counter only (cheap).
  bool ensure_screen_index(std::int64_t stride) {
    if (d_compact_index && d_screen_count) return true;
    free_screen_all();
    if (cudaMalloc(&d_compact_index, stride * sizeof(int)) != cudaSuccess) {
      d_compact_index = nullptr;
      return false;
    }
    if (cudaMalloc(&d_screen_count, sizeof(int)) != cudaSuccess) {
      cudaFree(d_compact_index);
      d_compact_index = nullptr;
      d_screen_count = nullptr;
      return false;
    }
    return true;
  }

  // Phase-2: size compact buffers exactly to np_c (avoids multi-GB OOM).
  // d_temp_g_c is [4 * nbasis * np_c] for batched 4-plane GEMM.
  bool ensure_screen_data(std::int64_t nbasis, std::int64_t np_c) {
    if (np_c <= 0) return false;
    if (screen_cap >= np_c && d_phi_compact && d_grad_compact &&
        d_wv_rho_c && d_wv_grad_c && d_temp_g_c) {
      return true;
    }
    free_screen_data();
    if (cudaMalloc(&d_phi_compact, nbasis * np_c * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_grad_compact, 3 * nbasis * np_c * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_wv_rho_c, np_c * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_wv_grad_c, 3 * np_c * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_temp_g_c, 4 * nbasis * np_c * sizeof(double)) != cudaSuccess) {
      free_screen_data();
      fprintf(stderr, "[gga] screen data alloc FAILED nbasis=%ld np_c=%ld\n",
              (long)nbasis, (long)np_c);
      return false;
    }
    screen_cap = np_c;
    return true;
  }

  bool ensure_full_temp(std::int64_t nbasis, std::int64_t stride) {
    if (d_temp_g && d_W4) return true;
    if (!d_temp_g) {
      if (cudaMalloc(&d_temp_g, 4 * nbasis * stride * sizeof(double)) != cudaSuccess) {
        d_temp_g = nullptr;
        fprintf(stderr, "[gga] full temp alloc FAILED nbasis=%ld stride=%ld\n",
                (long)nbasis, (long)stride);
        return false;
      }
    }
    // d_W4 is now allocated in ensure().
    return d_W4 != nullptr;
  }
};

GgaVmatGemmCache& gga_vmat_gemm_cache() {
  static GgaVmatGemmCache c;
  return c;
}

// Kernel: each block computes one H[i, j] by reducing over all grid points.
// Threads in the block cooperatively iterate over grid points in chunks.
__global__ void VmatBuildKernel(
    const double* orbitals,  // [n_orb][n_points] row-major
    const double* v,         // [n_points]
    double* H,               // [n_orb][n_orb] output
    std::size_t n_orb,
    std::size_t n_points,
    double dv) {
  // Map blockIdx to (i, j) pair using 2D grid.
  const std::size_t i = static_cast<std::size_t>(blockIdx.x);
  const std::size_t j = static_cast<std::size_t>(blockIdx.y);
  if (i >= n_orb || j >= n_orb) return;
  if (j < i) return;  // only compute upper triangle

  const double* phi_i = orbitals + i * n_points;
  const double* phi_j = orbitals + j * n_points;

  // Cooperative reduction over grid points.
  double partial = 0.0;
  for (std::size_t g = static_cast<std::size_t>(threadIdx.x);
       g < n_points; g += static_cast<std::size_t>(blockDim.x)) {
    partial += v[g] * phi_i[g] * phi_j[g];
  }

  // Shared memory reduction.
  __shared__ double sdata[256];
  sdata[threadIdx.x] = partial;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const double val = sdata[0] * dv;
    H[i * n_orb + j] = val;
    H[j * n_orb + i] = val;  // symmetric
  }
}

// Direct device-native GGA adjoint for the frozen weighted-field contract.
// A TileMat/GEMM backend may replace this reduction, but it must not insert a
// divergence stencil or reapply quadrature weights.
__global__ void GgaVmatDeviceKernel(
    const double* phi, const double* grad_phi, const double* wv_rho,
    const double* wv_grad, double* vmat, std::int64_t nbasis,
    std::int64_t np, std::int64_t point_stride) {
  const std::int64_t mu = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t nu = static_cast<std::int64_t>(blockIdx.y);
  if (mu >= nbasis || nu >= nbasis || nu < mu) return;
  const std::int64_t basis_plane = nbasis * point_stride;
  double partial = 0.0;
  for (std::int64_t point = threadIdx.x; point < np; point += blockDim.x) {
    const double phi_mu = phi[mu * point_stride + point];
    const double phi_nu = phi[nu * point_stride + point];
    double value = wv_rho[point] * phi_mu * phi_nu;
    for (std::int64_t component = 0; component < 3; ++component) {
      const std::int64_t grad_offset = component * basis_plane;
      const double dphi_mu = grad_phi[grad_offset + mu * point_stride + point];
      const double dphi_nu = grad_phi[grad_offset + nu * point_stride + point];
      value += wv_grad[component * point_stride + point] *
          (dphi_mu * phi_nu + phi_mu * dphi_nu);
    }
    partial += value;
  }
  __shared__ double reduction[256];
  reduction[threadIdx.x] = partial;
  __syncthreads();
  for (int offset = 128; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) reduction[threadIdx.x] += reduction[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    vmat[mu * nbasis + nu] = reduction[0];
    vmat[nu * nbasis + mu] = reduction[0];
  }
}

// Direct device-native mGGA adjoint.  Adds the weighted v_tau term
// wv_tau * (grad phi_m . grad phi_n) to the GGA expression.
__global__ void MggaVmatDeviceKernel(
    const double* phi, const double* grad_phi, const double* wv_rho,
    const double* wv_grad, const double* wv_tau, double* vmat,
    std::int64_t nbasis, std::int64_t np, std::int64_t point_stride) {
  const std::int64_t mu = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t nu = static_cast<std::int64_t>(blockIdx.y);
  if (mu >= nbasis || nu >= nbasis || nu < mu) return;
  const std::int64_t basis_plane = nbasis * point_stride;
  double partial = 0.0;
  for (std::int64_t point = threadIdx.x; point < np; point += blockDim.x) {
    const double phi_mu = phi[mu * point_stride + point];
    const double phi_nu = phi[nu * point_stride + point];
    double value = wv_rho[point] * phi_mu * phi_nu;
    double tau_dot = 0.0;
    for (std::int64_t component = 0; component < 3; ++component) {
      const std::int64_t grad_offset = component * basis_plane;
      const double dphi_mu = grad_phi[grad_offset + mu * point_stride + point];
      const double dphi_nu = grad_phi[grad_offset + nu * point_stride + point];
      value += wv_grad[component * point_stride + point] *
          (dphi_mu * phi_nu + phi_mu * dphi_nu);
      tau_dot += dphi_mu * dphi_nu;
    }
    value += wv_tau[point] * tau_dot;
    partial += value;
  }
  __shared__ double reduction[256];
  reduction[threadIdx.x] = partial;
  __syncthreads();
  for (int offset = 128; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) reduction[threadIdx.x] += reduction[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    vmat[mu * nbasis + nu] = reduction[0];
    vmat[nu * nbasis + mu] = reduction[0];
  }
}

// AUDIT B10: Arena-based device allocation helpers.
template <typename T>
Status CopyToDeviceArena(GpuArena& arena, const std::vector<T>& host, T** device) {
  if (host.empty()) { *device = nullptr; return Status::Ok(); }
  const std::size_t bytes = host.size() * sizeof(T);
  *device = static_cast<T*>(arena.Alloc(bytes));
  if (*device == nullptr) return Status::IoError("arena.Alloc failed");
  cudaError_t error = arena.H2D(*device, host.data(), bytes);
  if (error != cudaSuccess) { arena.Free(*device); *device = nullptr; }
  return CudaStatus(error, "arena.H2D");
}

template <typename T>
void FreeDeviceArena(GpuArena& arena, T* ptr) { if (ptr) arena.Free(ptr); }

}  // namespace

[[nodiscard]] bool VmatCudaAvailable() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

[[nodiscard]] Result<VmatGpuResult> VmatBuildCuda(
    const UniformGrid3D& grid,
    const std::vector<std::vector<double>>& orbitals,
    const std::vector<double>& v) {
  const std::size_t N = grid.total_points();
  const std::size_t n_orb = orbitals.size();
  VmatGpuResult result;
  result.H.assign(n_orb * n_orb, 0.0);
  result.n_points = N;
  result.n_orb = n_orb;

  if (N == 0 || n_orb == 0) return result;
  if (v.size() != N)
    return Status::InvalidArgument("v size mismatch with grid");
  for (std::size_t k = 0; k < n_orb; ++k)
    if (orbitals[k].size() != N)
      return Status::InvalidArgument("orbital size mismatch with grid");

  // Small-size fallback: when n_orb * n_points is small, GPU transfer +
  // launch overhead dominates. Use CPU VmatBuilder instead.
  // At 16³×4: GPU=0.82ms vs CPU=0.07ms. At 32³×16: GPU=1875ms vs CPU=4.4ms.
  // At 48³×32: GPU=1725ms vs CPU=71ms (CPU still faster due to kernel design).
  // Threshold: 5M elements — below this, CPU is always faster.
  const std::size_t total_elements = n_orb * N;
  if (total_elements < 5000000) {
    auto t0 = std::chrono::steady_clock::now();
    auto H_cpu = VmatBuilder::BuildHmat(grid, orbitals, v);
    auto t1 = std::chrono::steady_clock::now();
    result.H = H_cpu;
    result.kernel_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
  }

  if (!VmatCudaAvailable())
    return Status::IoError("CUDA runtime not available for vmat build");

  const auto [h0, h1, h2] = grid.h;
  const double dv = h0 * h1 * h2;

  // Flatten orbitals into a single array [n_orb * N].
  std::vector<double> flat_orb(n_orb * N, 0.0);
  for (std::size_t k = 0; k < n_orb; ++k)
    std::copy(orbitals[k].begin(), orbitals[k].end(),
              flat_orb.begin() + k * N);

  // AUDIT B10: Use GPU arena for persistent device buffers and stream.
  GpuArena& arena = GpuArena::Instance();
  cudaStream_t stream = arena.Stream();

  // Allocate device memory via arena.
  double* d_orb = nullptr;
  double* d_v = nullptr;
  double* d_H = nullptr;

  auto cleanup = [&]() {
    FreeDeviceArena(arena, d_orb);
    FreeDeviceArena(arena, d_v);
    FreeDeviceArena(arena, d_H);
  };

  auto st = CopyToDeviceArena(arena, flat_orb, &d_orb);
  if (!st.ok()) { cleanup(); return st; }
  st = CopyToDeviceArena(arena, v, &d_v);
  if (!st.ok()) { cleanup(); return st; }

  const std::size_t mat_bytes = n_orb * n_orb * sizeof(double);
  d_H = static_cast<double*>(arena.Alloc(mat_bytes));
  if (!d_H) { cleanup(); return Status::IoError("arena.Alloc failed for H"); }
  cudaMemsetAsync(d_H, 0, mat_bytes, stream);

  // Launch with 2D grid: (i, j) for upper triangle on arena stream.
  dim3 grid_dim(static_cast<unsigned int>(n_orb),
                static_cast<unsigned int>(n_orb));
  const int threads = 256;
  cudaEvent_t start_ev, stop_ev;
  cudaEventCreate(&start_ev);
  cudaEventCreate(&stop_ev);
  cudaEventRecord(start_ev, stream);
  VmatBuildKernel<<<grid_dim, threads, 0, stream>>>(
      d_orb, d_v, d_H, n_orb, N, dv);
  cudaEventRecord(stop_ev, stream);
  arena.Sync();
  float kernel_ms_f = 0.0f;
  cudaEventElapsedTime(&kernel_ms_f, start_ev, stop_ev);
  cudaEventDestroy(start_ev);
  cudaEventDestroy(stop_ev);
  result.kernel_ms = static_cast<double>(kernel_ms_f);

  cudaError_t err = arena.D2H(result.H.data(), d_H, mat_bytes);
  cleanup();
  if (err != cudaSuccess) return CudaStatus(err, "arena.D2H H");

  // Ledger.
  tides::tile::PrecisionDescriptor desc;
  desc.storage = tides::tile::NumericFormat::kFloat64;
  desc.compute = tides::tile::NumericFormat::kFloat64;
  desc.reduction = tides::tile::NumericFormat::kFloat64;
  desc.determinism = tides::tile::DeterminismMode::kDeterministic;
  desc.label = "cuda-vmat-build";
  const std::uint64_t candidates =
      static_cast<std::uint64_t>(n_orb) * n_orb * N;
  result.ledger.Add(tides::tile::OperationLedgerEntry{
      tides::tile::OperationKind::kGemm,
      desc,
      tides::tile::ErrorBudget{tides::tile::ErrorMetric::kAbsolute, 0.0,
          "GPU v->H adjoint vs CPU reference"},
      0.0, candidates, candidates, 0,
      "CUDA v->H adjoint map (potential to Hamiltonian tile)"});
  return result;
}

Status BuildGgaVmatDevice(const GgaVmatDeviceIn& input, double* vmat,
                          cudaStream_t stream) {
  if (input.nbasis <= 0 || input.np < 0 || input.point_stride < input.np) {
    return Status::InvalidArgument(
        "GGA vmat device build requires nbasis > 0, np >= 0, and point_stride >= np");
  }
  if (input.phi == nullptr || input.grad_phi == nullptr ||
      input.wv_rho == nullptr || input.wv_grad == nullptr || vmat == nullptr) {
    return Status::InvalidArgument(
        "GGA vmat device build requires non-null device pointers");
  }
  if (input.np == 0) {
    return CudaStatus(cudaMemsetAsync(
                          vmat, 0,
                          static_cast<std::size_t>(input.nbasis) * input.nbasis *
                              sizeof(double),
                          stream),
                      "cudaMemsetAsync GGA vmat");
  }

  // GEMM path: V = Phi * diag(wv_rho) * Phi^T
  //             + sum_a [G_a * diag(wv_grad_a) * Phi^T + its transpose]
  //
  // Decomposition:
  //   1. temp_rho = wv_rho * Phi;   V = temp_rho * Phi^T         (1 GEMM)
  //   2. For each direction a:
  //      temp_ga = wv_grad_a * G_a;  W = temp_ga * Phi^T          (1 GEMM)
  //      V += W + W^T                                               (symmetrize)
  const bool use_gemm = [] {
    const char* e = std::getenv("TIDES_VMAT_GEMM");
    return (e == nullptr || e[0] != '0');
  }();

  if (use_gemm && input.nbasis >= 4) {
    auto& gc = gga_vmat_gemm_cache();
    if (gc.ensure(input.nbasis, input.point_stride, stream)) {
      const int n = static_cast<int>(input.nbasis);
      const int k = static_cast<int>(input.np);
      const int lda = static_cast<int>(input.point_stride);
      const double alpha = 1.0, beta0 = 0.0;
      const std::int64_t basis_plane = input.nbasis * input.point_stride;
      const int n4 = 4 * n;
      bool done = false;

      const bool use_screen = [] {
        const char* e = std::getenv("TIDES_GRID_SCREEN");
        return (e == nullptr || e[0] != '0');
      }();

      if (use_screen && gc.ensure_screen_index(input.point_stride)) {
        if (!gc.screen_initialized) {
          cudaMemsetAsync(gc.d_screen_count, 0, sizeof(int), stream);
          {
            int threads = 256;
            int blocks = (static_cast<int>(input.np) + threads - 1) / threads;
            GgaBuildCompactionIndexKernel<<<blocks, threads, 0, stream>>>(
                input.phi, input.nbasis, input.np, input.point_stride,
                gc.d_compact_index, gc.d_screen_count);
          }
          int host_count = 0;
          cudaMemcpyAsync(&host_count, gc.d_screen_count,
                          sizeof(int), cudaMemcpyDeviceToHost, stream);
          cudaStreamSynchronize(stream);
          gc.np_compact = host_count;
          gc.screen_initialized = true;

          fprintf(stderr, "[screen] GGA np=%ld np_compact=%ld (%.1f%% active)\n",
                  (long)input.np, (long)gc.np_compact,
                  100.0 * gc.np_compact / input.np);
        }

        if (gc.np_compact > 0 && gc.np_compact < input.np &&
            gc.ensure_screen_data(input.nbasis, gc.np_compact)) {
          if (!gc.compact_ready) {
            {
              dim3 block(32, 4);
              dim3 grid((static_cast<unsigned int>(gc.np_compact) + block.x - 1) / block.x,
                        (static_cast<unsigned int>(input.nbasis) + block.y - 1) / block.y);
              GgaCompactPhiKernel<<<grid, block, 0, stream>>>(
                  gc.d_phi_compact, input.phi, gc.d_compact_index,
                  input.nbasis, gc.np_compact, input.point_stride);
            }
            {
              dim3 block(32, 4, 3);
              dim3 grid((static_cast<unsigned int>(gc.np_compact) + block.x - 1) / block.x,
                        (static_cast<unsigned int>(input.nbasis) + block.y - 1) / block.y,
                        1);
              GgaCompactGradKernel<<<grid, block, 0, stream>>>(
                  gc.d_grad_compact, input.grad_phi, gc.d_compact_index,
                  input.nbasis, gc.np_compact, input.point_stride);
            }
            gc.compact_ready = true;
          }

          const int k_c = static_cast<int>(gc.np_compact);
          const std::int64_t bp_c = input.nbasis * gc.np_compact;
          const int n4 = 4 * n;
          bool ok = true;

          cudaMemsetAsync(vmat, 0, n * n * sizeof(double), stream);

          // --- Batched: scale all 4 planes, then 1 GEMM ---
          // Plane 0: wv_rho * phi
          // Plane a (1..3): wv_grad_a * grad_phi_a
          {
            int threads = 256;
            int blocks = (k_c + threads - 1) / threads;
            GgaCompactWvKernel<<<blocks, threads, 0, stream>>>(
                gc.d_wv_rho_c, input.wv_rho, gc.d_compact_index, gc.np_compact);
          }
          {
            dim3 block(32, 4);
            dim3 grid((static_cast<unsigned int>(gc.np_compact) + block.x - 1) / block.x,
                      (static_cast<unsigned int>(input.nbasis) + block.y - 1) / block.y);
            ScaleCompactColumnsKernel<<<grid, block, 0, stream>>>(
                gc.d_temp_g_c, gc.d_phi_compact, gc.d_wv_rho_c,
                input.nbasis, gc.np_compact);
          }
          for (int a = 0; a < 3; ++a) {
            {
              int threads = 256;
              int blocks = (k_c + threads - 1) / threads;
              GgaCompactWvKernel<<<blocks, threads, 0, stream>>>(
                  gc.d_wv_grad_c + a * gc.np_compact,
                  input.wv_grad + a * input.point_stride,
                  gc.d_compact_index, gc.np_compact);
            }
            {
              dim3 block(32, 4);
              dim3 grid((static_cast<unsigned int>(gc.np_compact) + block.x - 1) / block.x,
                        (static_cast<unsigned int>(input.nbasis) + block.y - 1) / block.y);
              ScaleCompactColumnsKernel<<<grid, block, 0, stream>>>(
                  gc.d_temp_g_c + (a + 1) * bp_c,
                  gc.d_grad_compact + a * bp_c,
                  gc.d_wv_grad_c + a * gc.np_compact,
                  input.nbasis, gc.np_compact);
            }
          }

          // Single batched GEMM: W4[4n x n] = temp4^T * phi_compact
          {
            cublasStatus_t cs = cublasGemmEx(
                gc.handle, CUBLAS_OP_T, CUBLAS_OP_N,
                n4, n, k_c,
                &alpha,
                gc.d_temp_g_c, CUDA_R_64F, k_c,
                gc.d_phi_compact, CUDA_R_64F, k_c,
                &beta0,
                gc.d_W4, CUDA_R_64F, n4,
                CUDA_R_64F, CUBLAS_GEMM_DEFAULT);
            if (cs != CUBLAS_STATUS_SUCCESS) ok = false;
          }
          // Add rho block (plane 0) + symmetrize 3 grad blocks (planes 1..3)
          if (ok) {
            const int threads = 64;
            const int blocks = (n + threads - 1) / threads;
            StridedAddKernel<<<blocks, threads, 0, stream>>>(
                vmat, gc.d_W4, 0, n, n4);
            for (int a = 1; a < 4; ++a) {
              SymmetrizeAddStridedKernel<<<blocks, threads, 0, stream>>>(
                  vmat, gc.d_W4, a * n, n, n4);
            }
          }
          done = ok;
        }
      }

      if (!done && gc.ensure_full_temp(input.nbasis, input.point_stride)) {
        cudaMemsetAsync(vmat, 0, n * n * sizeof(double), stream);

        {
          dim3 block(32, 4);
          dim3 grid((static_cast<unsigned int>(input.np) + block.x - 1) / block.x,
                    (static_cast<unsigned int>(input.nbasis) + block.y - 1) / block.y);
          ScaleColumnsKernel<<<grid, block, 0, stream>>>(
              gc.d_temp_g, input.phi, input.wv_rho,
              input.nbasis, input.np, input.point_stride);
        }
        for (int a = 0; a < 3; ++a) {
          const double* grad_a = input.grad_phi + a * basis_plane;
          const double* wv_g_a = input.wv_grad + a * input.point_stride;
          double* dst = gc.d_temp_g + (a + 1) * basis_plane;
          dim3 block(32, 4);
          dim3 grid((static_cast<unsigned int>(input.np) + block.x - 1) / block.x,
                    (static_cast<unsigned int>(input.nbasis) + block.y - 1) / block.y);
          ScaleColumnsKernel<<<grid, block, 0, stream>>>(
              dst, grad_a, wv_g_a,
              input.nbasis, input.np, input.point_stride);
        }

        cublasStatus_t cs = cublasGemmEx(
            gc.handle, CUBLAS_OP_T, CUBLAS_OP_N,
            n4, n, k,
            &alpha,
            gc.d_temp_g, CUDA_R_64F, lda,
            input.phi, CUDA_R_64F, lda,
            &beta0,
            gc.d_W4, CUDA_R_64F, n4,
            CUDA_R_64F, CUBLAS_GEMM_DEFAULT);
        if (cs == CUBLAS_STATUS_SUCCESS) {
          const int threads = 64;
          const int blocks = (n + threads - 1) / threads;
          StridedAddKernel<<<blocks, threads, 0, stream>>>(
              vmat, gc.d_W4, 0, n, n4);
          for (int a = 1; a < 4; ++a) {
            SymmetrizeAddStridedKernel<<<blocks, threads, 0, stream>>>(
                vmat, gc.d_W4, a * n, n, n4);
          }
          done = true;
        }
      }

      if (done) return Status::Ok();
    }
  }

  // Fallback: original per-(mu,nu) reduction kernel.
  {
    dim3 grid_dim(static_cast<unsigned int>(input.nbasis),
                  static_cast<unsigned int>(input.nbasis));
    GgaVmatDeviceKernel<<<grid_dim, 256, 0, stream>>>(
        input.phi, input.grad_phi, input.wv_rho, input.wv_grad, vmat,
        input.nbasis, input.np, input.point_stride);
    return CudaStatus(cudaGetLastError(), "GgaVmatDeviceKernel launch");
  }
}

Status BuildMggaVmatDevice(const MggaVmatDeviceIn& input, double* vmat,
                           cudaStream_t stream) {
  if (input.nbasis <= 0 || input.np < 0 || input.point_stride < input.np) {
    return Status::InvalidArgument(
        "mGGA vmat device build requires nbasis > 0, np >= 0, and point_stride >= np");
  }
  if (input.phi == nullptr || input.grad_phi == nullptr ||
      input.wv_rho == nullptr || input.wv_grad == nullptr ||
      input.wv_tau == nullptr || vmat == nullptr) {
    return Status::InvalidArgument(
        "mGGA vmat device build requires non-null device pointers");
  }
  if (input.np == 0) {
    return CudaStatus(cudaMemsetAsync(
                          vmat, 0,
                          static_cast<std::size_t>(input.nbasis) * input.nbasis *
                              sizeof(double),
                          stream),
                      "cudaMemsetAsync mGGA vmat");
  }
  dim3 grid_dim(static_cast<unsigned int>(input.nbasis),
                static_cast<unsigned int>(input.nbasis));
  MggaVmatDeviceKernel<<<grid_dim, 256, 0, stream>>>(
      input.phi, input.grad_phi, input.wv_rho, input.wv_grad, input.wv_tau,
      vmat, input.nbasis, input.np, input.point_stride);
  return CudaStatus(cudaGetLastError(), "MggaVmatDeviceKernel launch");
}

}  // namespace tides::grid
