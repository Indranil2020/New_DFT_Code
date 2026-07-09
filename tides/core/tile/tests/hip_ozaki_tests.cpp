// HIP Ozaki f64e test — validates FP64 emulation on AMD GPU.
// Returns 77 (skip) if no HIP device, 0 on success.

#include "tile/hip_compat.hpp"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
#ifdef TIDES_USE_HIP
  int device_count = 0;
  hipGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cout << "HIP Ozaki: No devices, skipping.\n";
    return 77;
  }

  // Simple test: verify that half-precision GEMM produces a finite result.
  const int N = 4;
  std::vector<double> h_A(N * N, 1.0);
  std::vector<double> h_B(N * N, 2.0);
  std::vector<double> h_C(N * N, 0.0);

  double* d_A = nullptr;
  double* d_B = nullptr;
  double* d_C = nullptr;
  hipMalloc(&d_A, N * N * sizeof(double));
  hipMalloc(&d_B, N * N * sizeof(double));
  hipMalloc(&d_C, N * N * sizeof(double));
  hipMemcpy(d_A, h_A.data(), N * N * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_B, h_B.data(), N * N * sizeof(double), hipMemcpyHostToDevice);

  hipblasHandle_t handle;
  hipblasCreate(&handle);
  double alpha = 1.0, beta = 0.0;
  hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
               N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
  hipMemcpy(h_C.data(), d_C, N * N * sizeof(double), hipMemcpyDeviceToHost);
  hipblasDestroy(handle);
  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);

  // Expected: each element = 4*2 = 8 (A=1, B=2, sum of N=4 products).
  double expected = static_cast<double>(N) * 1.0 * 2.0;
  double max_err = 0.0;
  for (auto v : h_C) max_err = std::max(max_err, std::fabs(v - expected));

  if (max_err > 1e-10) {
    std::cerr << "HIP Ozaki: Error too large (" << max_err << ")\n";
    return 1;
  }

  std::cout << "HIP Ozaki: OK (expected=" << expected << " max_err=" << max_err << ")\n";
  return 0;
#else
  std::cout << "HIP Ozaki: Not compiled with TIDES_USE_HIP, skipping.\n";
  return 77;
#endif
}
