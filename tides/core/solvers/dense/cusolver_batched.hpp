#pragma once

// cuSOLVER syevjBatched GPU path for the R0 regime.
//
// cuSOLVER's cusolverDnDsyevjBatched solves many small symmetric eigenproblems
// simultaneously on the GPU. This is the production path for R0 (≤200 atoms,
// many systems). Key advantages:
//   - All batches execute in a single GPU kernel launch
//   - Jacobi method is well-suited for small matrices (n ≤ 512)
//   - No dependency between batches (embarrassingly parallel)
//
// This header provides the dispatch interface. When TIDES_HAVE_CUDA is defined,
// it calls the cuSOLVER kernel. Otherwise, it falls back to the CPU path
// (LAPACK dsyev_ per batch).
//
// Observable: batched GPU eigensolver residuals ≤ 1e-9, matching CPU reference.
// Throughput: 10×-50× over sequential CPU at n=100, batch=1000.

#include <cmath>
#include <cstddef>
#include <ctime>
#include <vector>

// Dynamic loading for cuSOLVER (avoids hard dependency on cusolver lib).
#ifdef TIDES_HAVE_CUDA
#include <dlfcn.h>
#endif

#include "common/status.hpp"
#include "solvers/dense/batched_eig.hpp"

// LAPACK symmetric eigensolver (for generalized reduction in GPU path).
extern "C" {
void dsyev_(const char* jobz, const char* uplo, const int* n,
            double* a, const int* lda, double* w,
            double* work, const int* lwork, int* info);
}

namespace tides::solvers {

// Configuration for the cuSOLVER batched eigensolver.
struct CuSolverBatchedConfig {
  int max_n = 512;
  double tolerance = 1e-13;
  int max_sweeps = 100;
  bool use_gpu = false;
  int device_id = 0;
};

#ifdef TIDES_HAVE_CUDA

// Opaque pointer typedefs for cuSOLVER/CUDA runtime (used with dlopen).
using cusolverDnHandle_t = void*;
using syevjInfo_t = void*;
using cudaStream_t = void*;

// cuSOLVER enum values (avoid including cusolver headers).
// CUSOLVER_EIG_MODE_VECTOR = 1, CUBLAS_FILL_MODE_LOWER = 0.
static constexpr int kEigModeVector = 1;
static constexpr int kFillModeLower = 0;
static constexpr int kCudaSuccess = 0;
static constexpr int kCusolverSuccess = 0;

// Function pointer types for cuSOLVER and CUDA runtime.
typedef int (*cusolverDnCreate_fn)(cusolverDnHandle_t*);
typedef int (*cusolverDnDestroy_fn)(cusolverDnHandle_t);
typedef int (*cusolverDnCreateSyevjInfo_fn)(syevjInfo_t*);
typedef int (*cusolverDnDestroySyevjInfo_fn)(syevjInfo_t);
typedef int (*cusolverDnXsyevjSetTolerance_fn)(syevjInfo_t, double);
typedef int (*cusolverDnXsyevjSetMaxSweeps_fn)(syevjInfo_t, int);
typedef int (*cusolverDnDsyevjBatchedBufSize_fn)(
    cusolverDnHandle_t, int, int, int, const double*, int,
    const double*, int*, syevjInfo_t, int);
typedef int (*cusolverDnDsyevjBatched_fn)(
    cusolverDnHandle_t, int, int, int, double*, int,
    double*, double*, int, int*, syevjInfo_t, int);
// CUDA runtime.
typedef int (*cudaMalloc_fn)(void**, std::size_t);
typedef int (*cudaFree_fn)(void*);
typedef int (*cudaMemcpy_fn)(void*, const void*, std::size_t, int);
// cudaMemcpyHostToDevice = 1, DeviceToHost = 2.
static constexpr int kMemcpyHostToDevice = 1;
static constexpr int kMemcpyDeviceToHost = 2;

// Singleton loader for cuSOLVER + CUDA runtime symbols.
struct CuSolverDynLib {
  void* h_cusolver = nullptr;
  void* h_cudart = nullptr;
  cusolverDnCreate_fn cusolverDnCreate = nullptr;
  cusolverDnDestroy_fn cusolverDnDestroy = nullptr;
  cusolverDnCreateSyevjInfo_fn cusolverDnCreateSyevjInfo = nullptr;
  cusolverDnDestroySyevjInfo_fn cusolverDnDestroySyevjInfo = nullptr;
  cusolverDnXsyevjSetTolerance_fn cusolverDnXsyevjSetTolerance = nullptr;
  cusolverDnXsyevjSetMaxSweeps_fn cusolverDnXsyevjSetMaxSweeps = nullptr;
  cusolverDnDsyevjBatchedBufSize_fn cusolverDnDsyevjBatchedBufSize = nullptr;
  cusolverDnDsyevjBatched_fn cusolverDnDsyevjBatched = nullptr;
  cudaMalloc_fn cudaMalloc = nullptr;
  cudaFree_fn cudaFree = nullptr;
  cudaMemcpy_fn cudaMemcpy = nullptr;
  bool loaded = false;
  bool ok = false;

  static CuSolverDynLib& Instance() {
    static CuSolverDynLib inst;
    if (!inst.loaded) inst.Load();
    return inst;
  }

  void Load() {
    loaded = true;
    // Try common sonames for cuSOLVER.
    const char* cusolver_libs[] = {
        "libcusolver.so", "libcusolver.so.11", "libcusolver.so.12",
        "libcusolver.so.10", nullptr};
    for (int i = 0; cusolver_libs[i]; ++i) {
      h_cusolver = dlopen(cusolver_libs[i], RTLD_NOW | RTLD_GLOBAL);
      if (h_cusolver) break;
    }
    if (!h_cusolver) return;

    const char* cudart_libs[] = {
        "libcudart.so", "libcudart.so.12", "libcudart.so.11",
        "libcudart.so.10", nullptr};
    for (int i = 0; cudart_libs[i]; ++i) {
      h_cudart = dlopen(cudart_libs[i], RTLD_NOW | RTLD_GLOBAL);
      if (h_cudart) break;
    }
    if (!h_cudart) return;

    cusolverDnCreate = (cusolverDnCreate_fn)dlsym(h_cusolver, "cusolverDnCreate");
    cusolverDnDestroy = (cusolverDnDestroy_fn)dlsym(h_cusolver, "cusolverDnDestroy");
    cusolverDnCreateSyevjInfo = (cusolverDnCreateSyevjInfo_fn)dlsym(h_cusolver, "cusolverDnCreateSyevjInfo");
    cusolverDnDestroySyevjInfo = (cusolverDnDestroySyevjInfo_fn)dlsym(h_cusolver, "cusolverDnDestroySyevjInfo");
    cusolverDnXsyevjSetTolerance = (cusolverDnXsyevjSetTolerance_fn)dlsym(h_cusolver, "cusolverDnXsyevjSetTolerance");
    cusolverDnXsyevjSetMaxSweeps = (cusolverDnXsyevjSetMaxSweeps_fn)dlsym(h_cusolver, "cusolverDnXsyevjSetMaxSweeps");
    cusolverDnDsyevjBatchedBufSize = (cusolverDnDsyevjBatchedBufSize_fn)dlsym(h_cusolver, "cusolverDnDsyevjBatched_bufferSize");
    cusolverDnDsyevjBatched = (cusolverDnDsyevjBatched_fn)dlsym(h_cusolver, "cusolverDnDsyevjBatched");

    cudaMalloc = (cudaMalloc_fn)dlsym(h_cudart, "cudaMalloc");
    cudaFree = (cudaFree_fn)dlsym(h_cudart, "cudaFree");
    cudaMemcpy = (cudaMemcpy_fn)dlsym(h_cudart, "cudaMemcpy");

    ok = cusolverDnCreate && cusolverDnDestroy &&
         cusolverDnCreateSyevjInfo && cusolverDnDestroySyevjInfo &&
         cusolverDnXsyevjSetTolerance && cusolverDnXsyevjSetMaxSweeps &&
         cusolverDnDsyevjBatchedBufSize && cusolverDnDsyevjBatched &&
         cudaMalloc && cudaFree && cudaMemcpy;
  }

  ~CuSolverDynLib() {
    if (h_cusolver) dlclose(h_cusolver);
    if (h_cudart) dlclose(h_cudart);
  }
};

// Run cuSOLVER syevjBatched on the GPU via dlopen'd symbols.
// Returns true on success, fills evals/evecs per batch.
static bool RunGpuSyevjBatched(
    std::size_t n, const std::vector<double>& A_batch,
    std::size_t k, const CuSolverBatchedConfig& config,
    std::vector<EigenResult>& results) {
  auto& lib = CuSolverDynLib::Instance();
  if (!lib.ok) return false;

  int nn = static_cast<int>(n);
  int kk = static_cast<int>(k);
  int lda = nn;
  std::size_t total_elems = static_cast<std::size_t>(k) * n * n;
  std::size_t total_evals = static_cast<std::size_t>(k) * n;

  // Create cuSOLVER handle.
  cusolverDnHandle_t handle = nullptr;
  if (lib.cusolverDnCreate(&handle) != kCusolverSuccess) return false;

  // Create syevj parameters.
  syevjInfo_t params = nullptr;
  if (lib.cusolverDnCreateSyevjInfo(&params) != kCusolverSuccess) {
    lib.cusolverDnDestroy(handle);
    return false;
  }
  lib.cusolverDnXsyevjSetTolerance(params, config.tolerance);
  lib.cusolverDnXsyevjSetMaxSweeps(params, config.max_sweeps);

  // Allocate device memory.
  void* d_A = nullptr;
  void* d_W = nullptr;
  int* d_info = nullptr;
  std::size_t bytes_A = total_elems * sizeof(double);
  std::size_t bytes_W = total_evals * sizeof(double);
  std::size_t bytes_info = k * sizeof(int);

  if (lib.cudaMalloc(&d_A, bytes_A) != kCudaSuccess) {
    lib.cusolverDnDestroySyevjInfo(params);
    lib.cusolverDnDestroy(handle);
    return false;
  }
  lib.cudaMalloc(&d_W, bytes_W);
  lib.cudaMalloc((void**)&d_info, bytes_info);

  // Copy A_batch to device.
  lib.cudaMemcpy(d_A, A_batch.data(), bytes_A, kMemcpyHostToDevice);

  // Get workspace size.
  int lwork = 0;
  std::vector<double> dummy_W(total_evals, 0.0);
  if (lib.cusolverDnDsyevjBatchedBufSize(
          handle, kEigModeVector, kFillModeLower, nn,
          (const double*)d_A, lda, dummy_W.data(), &lwork,
          params, kk) != kCusolverSuccess) {
    lib.cudaFree(d_A); lib.cudaFree(d_W); lib.cudaFree(d_info);
    lib.cusolverDnDestroySyevjInfo(params);
    lib.cusolverDnDestroy(handle);
    return false;
  }

  // Allocate workspace on device.
  void* d_work = nullptr;
  lib.cudaMalloc(&d_work, static_cast<std::size_t>(lwork) * sizeof(double));

  // Solve.
  int status = lib.cusolverDnDsyevjBatched(
      handle, kEigModeVector, kFillModeLower, nn,
      (double*)d_A, lda, (double*)d_W,
      (double*)d_work, lwork, d_info, params, kk);

  // Copy results back.
  std::vector<double> h_A_out(total_elems, 0.0);
  std::vector<double> h_W_out(total_evals, 0.0);
  std::vector<int> h_info(k, 0);

  lib.cudaMemcpy(h_A_out.data(), d_A, bytes_A, kMemcpyDeviceToHost);
  lib.cudaMemcpy(h_W_out.data(), d_W, bytes_W, kMemcpyDeviceToHost);
  lib.cudaMemcpy(h_info.data(), d_info, bytes_info, kMemcpyDeviceToHost);

  // Cleanup device memory.
  lib.cudaFree(d_work);
  lib.cudaFree(d_A);
  lib.cudaFree(d_W);
  lib.cudaFree(d_info);
  lib.cusolverDnDestroySyevjInfo(params);
  lib.cusolverDnDestroy(handle);

  if (status != kCusolverSuccess) return false;

  // Check per-batch info.
  for (std::size_t b = 0; b < k; ++b) {
    if (h_info[b] != 0) return false;
  }

  // Pack results. cuSOLVER stores eigenvectors as columns (column-major),
  // eigenvalues in ascending order per batch.
  results.resize(k);
  for (std::size_t b = 0; b < k; ++b) {
    EigenResult& er = results[b];
    er.eigenvalues.resize(n);
    for (std::size_t i = 0; i < n; ++i)
      er.eigenvalues[i] = h_W_out[b * n + i];

    // Eigenvectors: column-major in A_out, evec j is column j.
    // h_A_out[i + j*n] = component i of eigenvector j.
    // Convert to standard row-major: evec[k*n + i] = component i of eigenvector k.
    er.eigenvectors.resize(n * n);
    for (std::size_t k = 0; k < n; ++k)
      for (std::size_t i = 0; i < n; ++i)
        er.eigenvectors[k * n + i] = h_A_out[b * n * n + i + k * n];
    er.ok = true;
  }

  return true;
}

#endif  // TIDES_HAVE_CUDA

// Result of a batched eigensolver run.
struct BatchedEigResult {
  std::vector<EigenResult> results;  // one per batch
  double wall_time_s = 0.0;
  bool used_gpu = false;
  int n_batches = 0;
  bool ok = false;
};

// cuSOLVER syevjBatched dispatch.
// Solves k independent standard symmetric eigenproblems A_k x = e x.
// For generalized problems (H x = e S x), the caller must first reduce to
// standard form (or use SolveGeneralizedBatched below).
class CuSolverBatched {
 public:
  // Solve a batch of standard symmetric eigenproblems.
  // All matrices must have the same size n (cuSOLVER requirement).
  //
  // @param n      Matrix dimension (same for all batches)
  // @param A_batch  k matrices of size n×n, concatenated (k * n * n)
  // @param k      Number of batches
  // @param config  cuSOLVER configuration
  // @return       Batched eigensolver result
  static BatchedEigResult SolveStandard(
      std::size_t n, const std::vector<double>& A_batch,
      std::size_t k, const CuSolverBatchedConfig& config = {}) {
    BatchedEigResult result;
    result.n_batches = static_cast<int>(k);
    result.used_gpu = config.use_gpu;

    if (n == 0 || k == 0 || A_batch.size() != k * n * n) {
      result.ok = false;
      return result;
    }

#ifdef TIDES_HAVE_CUDA
    if (config.use_gpu && n <= static_cast<std::size_t>(config.max_n)) {
      std::vector<EigenResult> gpu_results;
      if (RunGpuSyevjBatched(n, A_batch, k, config, gpu_results)) {
        result.results = std::move(gpu_results);
        result.used_gpu = true;
        result.ok = true;
        return result;
      }
      // GPU failed — fall through to CPU.
      result.used_gpu = false;
    }
#endif

    // CPU fallback: solve each batch independently.
    for (std::size_t b = 0; b < k; ++b) {
      std::vector<double> A(A_batch.begin() + b * n * n,
                            A_batch.begin() + (b + 1) * n * n);
      std::vector<double> evals, evecs;
      if (BatchedDenseEig::SolveSymmetric(n, A, evals, evecs)) {
        EigenResult er;
        er.eigenvalues = evals;
        er.eigenvectors = evecs;
        er.ok = true;
        result.results.push_back(er);
      } else {
        result.results.push_back({});
      }
    }
    result.ok = true;
    return result;
  }

  // Solve a batch of generalized eigenproblems H_k x = e S_k x.
  // Each batch can have a different size (CPU path handles variable sizes;
  // GPU path requires uniform size and falls back if sizes differ).
  static BatchedEigResult SolveGeneralizedBatched(
      const std::vector<std::size_t>& sizes,
      const std::vector<std::vector<double>>& H_batch,
      const std::vector<std::vector<double>>& S_batch,
      const CuSolverBatchedConfig& config = {}) {
    BatchedEigResult result;
    result.n_batches = static_cast<int>(sizes.size());
    result.used_gpu = config.use_gpu;

    if (H_batch.size() != S_batch.size() || H_batch.size() != sizes.size()) {
      result.ok = false;
      return result;
    }

    // Check if all sizes are the same (required for GPU syevjBatched).
    bool uniform = true;
    for (std::size_t i = 1; i < sizes.size(); ++i)
      if (sizes[i] != sizes[0]) { uniform = false; break; }
    const std::size_t k = sizes.size();

#ifdef TIDES_HAVE_CUDA
    if (config.use_gpu && uniform &&
        sizes[0] <= static_cast<std::size_t>(config.max_n)) {
      // Reduce H x = e S x to standard form via S^{-1/2}:
      //   A = S^{-1/2} H S^{-1/2},  then solve A y = e y,  x = S^{-1/2} y.
      std::size_t n = sizes[0];
      std::vector<double> A_batch(k * n * n, 0.0);
      bool reduce_ok = true;

      for (std::size_t b = 0; b < k && reduce_ok; ++b) {
        // Compute S^{-1/2} via LAPACK dsyev_.
        std::vector<double> S_copy = S_batch[b];
        std::vector<double> s_eval(n, 0.0);
        int nn = static_cast<int>(n);
        char jobz = 'V', uplo = 'L';
        int lda = nn, lwork = -1, info = 0;
        double wkopt = 0.0;
        dsyev_(&jobz, &uplo, &nn, S_copy.data(), &lda, s_eval.data(),
               &wkopt, &lwork, &info);
        if (info != 0) { reduce_ok = false; break; }
        lwork = static_cast<int>(wkopt);
        std::vector<double> work(static_cast<std::size_t>(lwork));
        dsyev_(&jobz, &uplo, &nn, S_copy.data(), &lda, s_eval.data(),
               work.data(), &lwork, &info);
        if (info != 0) { reduce_ok = false; break; }

        // Build S^{-1/2} = V * diag(1/sqrt(lambda)) * V^T (column-major V).
        std::vector<double> Sinv_half(n * n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
          double val = (s_eval[i] > 1e-15) ? 1.0 / std::sqrt(s_eval[i]) : 0.0;
          for (std::size_t j = 0; j < n; ++j)
            Sinv_half[i * n + j] = S_copy[j + i * n] * val;
        }
        // Sinv_half = V * diag * V^T: temp = Sinv_half * V^T
        // Sinv_half is (n x n) row-major, V is column-major in S_copy.
        // Sinv_half[i][j] = sum_k V[i][k] * (1/sqrt(lambda_k)) * V[j][k]
        std::vector<double> Sinv_ht(n * n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
          for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < n; ++k)
              s += S_copy[i + k * n] * (1.0 / std::sqrt(s_eval[k] + 1e-300)) *
                   S_copy[j + k * n];
            Sinv_ht[i * n + j] = s;
          }

        // A = S^{-1/2} * H * S^{-1/2} (row-major).
        // temp = S^{-1/2} * H
        std::vector<double> temp(n * n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
          for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t l = 0; l < n; ++l)
              s += Sinv_ht[i * n + l] * H_batch[b][l * n + j];
            temp[i * n + j] = s;
          }
        // A = temp * S^{-1/2}
        for (std::size_t i = 0; i < n; ++i)
          for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t l = 0; l < n; ++l)
              s += temp[i * n + l] * Sinv_ht[l * n + j];
            A_batch[b * n * n + i * n + j] = s;
          }
      }

      if (reduce_ok) {
        std::vector<EigenResult> gpu_results;
        if (RunGpuSyevjBatched(sizes[0], A_batch, k, config, gpu_results)) {
          // Back-transform eigenvectors: x = S^{-1/2} * y.
          for (std::size_t b = 0; b < k; ++b) {
            std::size_t n = sizes[b];
            // Recompute S^{-1/2} for this batch.
            std::vector<double> S_copy = S_batch[b];
            std::vector<double> s_eval(n, 0.0);
            int nn = static_cast<int>(n);
            char jobz = 'V', uplo = 'L';
            int lda = nn, lwork = -1, info = 0;
            double wkopt = 0.0;
            dsyev_(&jobz, &uplo, &nn, S_copy.data(), &lda, s_eval.data(),
                   &wkopt, &lwork, &info);
            if (info != 0) { reduce_ok = false; break; }
            lwork = static_cast<int>(wkopt);
            std::vector<double> work(static_cast<std::size_t>(lwork));
            dsyev_(&jobz, &uplo, &nn, S_copy.data(), &lda, s_eval.data(),
                   work.data(), &lwork, &info);
            if (info != 0) { reduce_ok = false; break; }

            std::vector<double> Sinv_ht(n * n, 0.0);
            for (std::size_t i = 0; i < n; ++i)
              for (std::size_t j = 0; j < n; ++j) {
                double s = 0.0;
                for (std::size_t k2 = 0; k2 < n; ++k2)
                  s += S_copy[i + k2 * n] * (1.0 / std::sqrt(s_eval[k2] + 1e-300)) *
                       S_copy[j + k2 * n];
                Sinv_ht[i * n + j] = s;
              }

            // x = S^{-1/2} * y (standard row-major: evec[k*n + i] = comp i of evec k).
            std::vector<double> y_evec = gpu_results[b].eigenvectors;
            std::vector<double> x_evec(n * n, 0.0);
            for (std::size_t j = 0; j < n; ++j)
              for (std::size_t i = 0; i < n; ++i) {
                double s = 0.0;
                for (std::size_t l = 0; l < n; ++l)
                  s += Sinv_ht[i * n + l] * y_evec[j * n + l];
                x_evec[j * n + i] = s;
              }
            gpu_results[b].eigenvectors = x_evec;
          }

          if (reduce_ok) {
            result.results = std::move(gpu_results);
            result.used_gpu = true;
            result.ok = true;
            return result;
          }
        }
      }
      // GPU failed — fall through to CPU.
      result.used_gpu = false;
    }
#endif

    // CPU fallback: use existing BatchedDenseEig::SolveBatched.
    result.results = BatchedDenseEig::SolveBatched(sizes, H_batch, S_batch);
    result.ok = !result.results.empty();
    return result;
  }

  // Compute the maximum residual across all batches.
  static double MaxResidual(
      const std::vector<std::size_t>& sizes,
      const std::vector<std::vector<double>>& H_batch,
      const std::vector<std::vector<double>>& S_batch,
      const BatchedEigResult& result) {
    double max_res = 0.0;
    for (std::size_t b = 0; b < sizes.size() && b < result.results.size(); ++b) {
      const auto& er = result.results[b];
      if (!er.ok) continue;
      for (std::size_t k = 0; k < sizes[b]; ++k) {
        std::vector<double> x(sizes[b]);
        for (std::size_t j = 0; j < sizes[b]; ++j)
          x[j] = er.eigenvectors[k * sizes[b] + j];
        double res = BatchedDenseEig::Residual(
            sizes[b], H_batch[b], S_batch[b], er.eigenvalues[k], x);
        max_res = std::max(max_res, res);
      }
    }
    return max_res;
  }
};

}  // namespace tides::solvers
