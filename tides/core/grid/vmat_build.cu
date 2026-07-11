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
  if (error == cudaSuccess) return Status::Ok();
  return Status::IoError(std::string(context) + ": " +
                         cudaGetErrorString(error));
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
  dim3 grid_dim(static_cast<unsigned int>(input.nbasis),
                static_cast<unsigned int>(input.nbasis));
  GgaVmatDeviceKernel<<<grid_dim, 256, 0, stream>>>(
      input.phi, input.grad_phi, input.wv_rho, input.wv_grad, vmat,
      input.nbasis, input.np, input.point_stride);
  return CudaStatus(cudaGetLastError(), "GgaVmatDeviceKernel launch");
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
