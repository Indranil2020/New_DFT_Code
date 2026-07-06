#include "tile/gemm_grouped.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

int main() {
  const auto status = tides::tile::CudaRuntimeStatus();
  if (!status.ok()) {
    double* device_value = nullptr;
    const cudaError_t malloc_error =
        cudaMalloc(reinterpret_cast<void**>(&device_value), sizeof(double));
    if (malloc_error == cudaSuccess) {
      cudaFree(device_value);
      std::cerr << "cuda_runtime_link_probe: cudaGetDeviceCount failed but "
                   "cudaMalloc succeeded: "
                << status.message() << '\n';
      return 1;
    }

    tides::tile::CudaGemmProblem problem;
    problem.m = 1;
    problem.k = 1;
    problem.n = 1;
    problem.a = {2.0};
    problem.b = {3.0};
    auto gemm = tides::tile::GroupedGemmFp64Cuda({problem});
    if (gemm.ok()) {
      std::cerr << "cuda_runtime_link_probe: cudaGetDeviceCount failed but "
                   "grouped GEMM succeeded: "
                << status.message() << '\n';
      return 1;
    }
    std::cerr << "cuda_runtime_link_probe: SKIP " << status.message()
              << "; cudaMalloc=" << cudaGetErrorString(malloc_error)
              << "; grouped_gemm=" << gemm.status().message() << '\n';
    return 77;
  }
  std::cout << "cuda_runtime_link_probe: CUDA runtime visible\n";
  return 0;
}
