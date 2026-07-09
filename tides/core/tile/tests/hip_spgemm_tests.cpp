// HIP SpGEMM test — validates filtered sparse matrix multiply on AMD GPU.
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
    std::cout << "HIP SpGEMM: No devices, skipping.\n";
    return 77;
  }

  // Simple sparse matrix test: 4x4 identity * 4x4 ones = 4x4 ones.
  // CSR format: values = [1,1,1,1], col_ind = [0,1,2,3], row_ptr = [0,1,2,3,4]
  const int M = 4, N = 4, K = 4;
  std::vector<double> h_val = {1.0, 1.0, 1.0, 1.0};
  std::vector<int> h_col = {0, 1, 2, 3};
  std::vector<int> h_row = {0, 1, 2, 3, 4};
  std::vector<double> h_B(N * N, 1.0);
  std::vector<double> h_C(M * N, 0.0);

  // Allocate device memory.
  double *d_val, *d_B, *d_C;
  int *d_col, *d_row;
  hipMalloc(&d_val, 4 * sizeof(double));
  hipMalloc(&d_col, 4 * sizeof(int));
  hipMalloc(&d_row, 5 * sizeof(int));
  hipMalloc(&d_B, N * N * sizeof(double));
  hipMalloc(&d_C, M * N * sizeof(double));

  hipMemcpy(d_val, h_val.data(), 4 * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_col, h_col.data(), 4 * sizeof(int), hipMemcpyHostToDevice);
  hipMemcpy(d_row, h_row.data(), 5 * sizeof(int), hipMemcpyHostToDevice);
  hipMemcpy(d_B, h_B.data(), N * N * sizeof(double), hipMemcpyHostToDevice);

  // Use hipSPARSE for SpMM.
  hipsparseHandle_t handle;
  hipsparseCreate(&handle);

  hipsparseSpMatDescr_t matA;
  hipsparseDnMatDescr_t matB, matC;
  hipsparseCreateCsr(&matA, M, K, 4, d_row, d_col, d_val,
                     HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                     HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);
  hipsparseCreateDnMat(&matB, K, N, N, d_B, HIP_R_64F, HIPSPARSE_ORDER_ROW);
  hipsparseCreateDnMat(&matC, M, N, N, d_C, HIP_R_64F, HIPSPARSE_ORDER_ROW);

  double alpha = 1.0, beta = 0.0;
  size_t bufferSize = 0;
  hipsparseSpMM_bufferSize(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                           HIPSPARSE_OPERATION_NON_TRANSPOSE,
                           &alpha, matA, matB, &beta, matC,
                           HIP_R_64F, HIPSPARSE_SPMM_ALG_DEFAULT, &bufferSize);

  void* d_buffer = nullptr;
  if (bufferSize > 0) hipMalloc(&d_buffer, bufferSize);

  hipsparseSpMM(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                HIPSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, matB, &beta, matC,
                HIP_R_64F, HIPSPARSE_SPMM_ALG_DEFAULT, d_buffer);

  hipMemcpy(h_C.data(), d_C, M * N * sizeof(double), hipMemcpyDeviceToHost);

  hipsparseDestroySpMat(matA);
  hipsparseDestroyDnMat(matB);
  hipsparseDestroyDnMat(matC);
  hipsparseDestroy(handle);
  if (d_buffer) hipFree(d_buffer);
  hipFree(d_val);
  hipFree(d_col);
  hipFree(d_row);
  hipFree(d_B);
  hipFree(d_C);

  // Expected: identity * ones = ones (each row = [1,1,1,1]).
  double max_err = 0.0;
  for (auto v : h_C) max_err = std::max(max_err, std::fabs(v - 1.0));

  if (max_err > 1e-10) {
    std::cerr << "HIP SpGEMM: Error too large (" << max_err << ")\n";
    return 1;
  }

  std::cout << "HIP SpGEMM: OK (max_err=" << max_err << ")\n";
  return 0;
#else
  std::cout << "HIP SpGEMM: Not compiled with TIDES_USE_HIP, skipping.\n";
  return 77;
#endif
}
