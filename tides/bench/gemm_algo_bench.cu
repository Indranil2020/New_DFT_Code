// Micro-benchmark: test cuBLAS DGEMM algorithms for vmat-shaped GEMMs.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

int main() {
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);

  // GEMM: C = A^T * B, where A and B are [k x n] col-major (lda=k)
  // C is [n x n] col-major (ldc=n)
  // This matches V = Phi^T * temp where Phi and temp are [stride x n] col-major

  for (int n : {96, 400, 800}) {
    for (int k : {171000, 500000, 1000000}) {
      if (n == 96 && k > 171000) continue;
      if (n == 400 && k > 500000) continue;

      double *d_A, *d_B, *d_C;
      cudaMalloc(&d_A, k * n * sizeof(double));
      cudaMalloc(&d_B, k * n * sizeof(double));
      cudaMalloc(&d_C, n * n * sizeof(double));
      cudaMemset(d_A, 1, k * n * sizeof(double));
      cudaMemset(d_B, 2, k * n * sizeof(double));

      double alpha = 1.0, beta = 0.0;
      float best_ms = 1e9;
      int best_algo = -1;

      // Test default first.
      {
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     n, n, k, &alpha,
                     d_A, CUDA_R_64F, k,
                     d_B, CUDA_R_64F, k,
                     &beta,
                     d_C, CUDA_R_64F, n,
                     CUDA_R_64F, CUBLAS_GEMM_DEFAULT);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  n=%d k=%d DEFAULT: %.2f ms\n", n, k, ms);
        best_ms = ms;
        best_algo = -1;
        cudaEventDestroy(start); cudaEventDestroy(stop);
      }

      // Test algorithms 0-23.
      for (int algo = 0; algo <= 23; ++algo) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        // Warmup.
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     n, n, k, &alpha,
                     d_A, CUDA_R_64F, k,
                     d_B, CUDA_R_64F, k,
                     &beta,
                     d_C, CUDA_R_64F, n,
                     CUDA_R_64F, static_cast<cublasGemmAlgo_t>(algo));

        cudaEventRecord(start);
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     n, n, k, &alpha,
                     d_A, CUDA_R_64F, k,
                     d_B, CUDA_R_64F, k,
                     &beta,
                     d_C, CUDA_R_64F, n,
                     CUDA_R_64F, static_cast<cublasGemmAlgo_t>(algo));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        if (ms < best_ms) {
          best_ms = ms;
          best_algo = algo;
        }
        cudaEventDestroy(start); cudaEventDestroy(stop);
      }

      printf("  n=%d k=%d BEST: algo=%d %.2f ms (default was %.2f ms)\n",
             n, k, best_algo, best_ms, best_ms);
      // Actually print default time separately
      {
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     n, n, k, &alpha,
                     d_A, CUDA_R_64F, k,
                     d_B, CUDA_R_64F, k,
                     &beta,
                     d_C, CUDA_R_64F, n,
                     CUDA_R_64F, CUBLAS_GEMM_DEFAULT);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  n=%d k=%d DEFAULT: %.2f ms, BEST: algo=%d %.2f ms (%.1fx)\n",
               n, k, ms, best_algo, best_ms, ms / best_ms);
        cudaEventDestroy(start); cudaEventDestroy(stop);
      }

      cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }
  }

  cublasDestroy(handle);
  return 0;
}
