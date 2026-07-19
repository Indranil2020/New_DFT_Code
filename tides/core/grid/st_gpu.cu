#include "grid/st_gpu.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <mutex>
#include <vector>

#include "common/status.hpp"
#include "grid/gpu_arena.hpp"

namespace tides::grid {

namespace {

cublasHandle_t* GetCachedCublasHandle() {
  static std::once_flag flag;
  static cublasHandle_t* handle = nullptr;
  std::call_once(flag, [&]() {
    handle = new cublasHandle_t;
    cublasCreate(handle);
  });
  return handle;
}

}  // namespace

StGpuResult BuildStFromGridGpu(const std::vector<double>& phi_flat,
                               const std::vector<double>& grad_flat,
                               std::size_t n, std::size_t np_total,
                               double dv) {
  StGpuResult result;
  if (n == 0 || np_total == 0) {
    result.status = Status::InvalidArgument("BuildStFromGridGpu: empty input");
    return result;
  }

  const int nn = static_cast<int>(n);
  const int kk = static_cast<int>(np_total);
  const int nn3 = 3 * nn;

  // phi_flat and grad_flat are row-major [n][np] and [3][n][np].
  // Interpreted as column-major with leading dimension np, the columns are the
  // basis (or basis*component) vectors.  Computing C = A^T A then gives the
  // overlap/kinetic matrices (with T summed over the three gradient components).

  cublasHandle_t handle = *GetCachedCublasHandle();
  GpuArena& arena = GpuArena::Instance();
  cudaStream_t stream = arena.Stream();
  cublasSetStream(handle, stream);

  const std::size_t phi_bytes = n * np_total * sizeof(double);
  const std::size_t grad_bytes = 3 * phi_bytes;
  const std::size_t st_bytes = n * n * sizeof(double);
  const std::size_t t_full_bytes = nn3 * nn3 * sizeof(double);

  double* d_phi = static_cast<double*>(arena.Alloc(phi_bytes));
  double* d_grad = static_cast<double*>(arena.Alloc(grad_bytes));
  double* d_S = static_cast<double*>(arena.Alloc(st_bytes));
  double* d_T_full = static_cast<double*>(arena.Alloc(t_full_bytes));
  if (!d_phi || !d_grad || !d_S || !d_T_full) {
    arena.Free(d_phi); arena.Free(d_grad); arena.Free(d_S); arena.Free(d_T_full);
    result.status = Status::IoError("BuildStFromGridGpu: device allocation failed");
    return result;
  }

  cudaError_t err = cudaMemcpyAsync(d_phi, phi_flat.data(), phi_bytes,
                                    cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    arena.Free(d_phi); arena.Free(d_grad); arena.Free(d_S); arena.Free(d_T_full);
    result.status = Status::IoError("BuildStFromGridGpu: phi H2D failed");
    return result;
  }

  err = cudaMemcpyAsync(d_grad, grad_flat.data(), grad_bytes,
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    arena.Free(d_phi); arena.Free(d_grad); arena.Free(d_S); arena.Free(d_T_full);
    result.status = Status::IoError("BuildStFromGridGpu: grad H2D failed");
    return result;
  }

  cudaMemsetAsync(d_S, 0, st_bytes, stream);
  cudaMemsetAsync(d_T_full, 0, t_full_bytes, stream);

  // S = dv * Phi^T * Phi
  double alpha = dv;
  double beta = 0.0;
  cublasStatus_t cst = cublasDgemm(
      handle, CUBLAS_OP_T, CUBLAS_OP_N,
      nn, nn, kk,
      &alpha, d_phi, kk, d_phi, kk,
      &beta, d_S, nn);
  if (cst != CUBLAS_STATUS_SUCCESS) {
    arena.Free(d_phi); arena.Free(d_grad); arena.Free(d_S); arena.Free(d_T_full);
    result.status = Status::IoError("BuildStFromGridGpu: S dgemm failed");
    return result;
  }

  // T = (dv/2) * sum_c grad_c^T grad_c.
  // Stack the three gradient components into a [3n x np] matrix G so that
  // G^T G contains the three n x n diagonal blocks we need.
  alpha = 0.5 * dv;
  cst = cublasDgemm(
      handle, CUBLAS_OP_T, CUBLAS_OP_N,
      nn3, nn3, kk,
      &alpha, d_grad, kk, d_grad, kk,
      &beta, d_T_full, nn3);
  if (cst != CUBLAS_STATUS_SUCCESS) {
    arena.Free(d_phi); arena.Free(d_grad); arena.Free(d_S); arena.Free(d_T_full);
    result.status = Status::IoError("BuildStFromGridGpu: T dgemm failed");
    return result;
  }

  result.S.assign(n * n, 0.0);
  result.T.assign(n * n, 0.0);
  err = cudaMemcpyAsync(result.S.data(), d_S, st_bytes, cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    arena.Free(d_phi); arena.Free(d_grad); arena.Free(d_S); arena.Free(d_T_full);
    result.status = Status::IoError("BuildStFromGridGpu: S D2H failed");
    return result;
  }

  std::vector<double> T_full(nn3 * nn3, 0.0);
  err = cudaMemcpyAsync(T_full.data(), d_T_full, t_full_bytes, cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    arena.Free(d_phi); arena.Free(d_grad); arena.Free(d_S); arena.Free(d_T_full);
    result.status = Status::IoError("BuildStFromGridGpu: T D2H failed");
    return result;
  }

  cudaStreamSynchronize(stream);

  // Extract the diagonal 3 n x n blocks and accumulate into T.
  for (int c = 0; c < 3; ++c) {
    const int base = c * nn;
    for (int i = 0; i < nn; ++i) {
      for (int j = 0; j < nn; ++j) {
        // d_T_full is column-major: index(row, col) = row + col * ldc.
        result.T[i * nn + j] += T_full[(base + i) + (base + j) * nn3];
      }
    }
  }

  arena.Free(d_phi);
  arena.Free(d_grad);
  arena.Free(d_S);
  arena.Free(d_T_full);

  result.status = Status::Ok();
  return result;
}

}  // namespace tides::grid
