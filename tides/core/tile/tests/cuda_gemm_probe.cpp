#include "tile/f64e_reference.hpp"
#include "tile/gemm_grouped.hpp"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

using tides::tile::CudaGemmProblem;
using tides::tile::CudaRuntimeStatus;
using tides::tile::GemmF64eReference;
using tides::tile::BuildGroupedGemmFp16AccumCudaPlan;
using tides::tile::GroupedGemmFp16AccumCuda;
using tides::tile::GroupedGemmFp64Cuda;
using tides::tile::RunGroupedGemmFp16AccumCudaPlan;

std::vector<double> MakeDense(std::size_t rows, std::size_t cols,
                              std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  std::vector<double> dense(rows * cols, 0.0);
  for (double& v : dense) {
    v = value(rng);
  }
  return dense;
}

struct ShapeKey {
  std::uint32_t m = 0;
  std::uint32_t k = 0;
  std::uint32_t n = 0;

  bool operator<(const ShapeKey& other) const {
    if (m != other.m) return m < other.m;
    if (k != other.k) return k < other.k;
    return n < other.n;
  }
};

struct CublasBucket {
  ShapeKey shape;
  std::vector<__half> a;
  std::vector<__half> b;
  std::vector<float> c;
};

std::string CublasStatusName(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "NOT_SUPPORTED";
    default:
      return "UNKNOWN";
  }
}

bool BuildCublasBuckets(const std::vector<CudaGemmProblem>& problems,
                        std::vector<CublasBucket>* buckets,
                        std::string* error) {
  std::map<ShapeKey, CublasBucket> by_shape;
  for (const CudaGemmProblem& problem : problems) {
    ShapeKey key{problem.m, problem.k, problem.n};
    CublasBucket& bucket = by_shape[key];
    bucket.shape = key;
    bucket.a.reserve(bucket.a.size() + problem.a.size());
    bucket.b.reserve(bucket.b.size() + problem.b.size());
    for (const double value : problem.a) {
      if (!std::isfinite(value) || std::abs(value) > 65504.0) {
        *error = "cuBLAS baseline input outside finite FP16 range";
        return false;
      }
      bucket.a.push_back(__float2half_rn(static_cast<float>(value)));
    }
    for (const double value : problem.b) {
      if (!std::isfinite(value) || std::abs(value) > 65504.0) {
        *error = "cuBLAS baseline input outside finite FP16 range";
        return false;
      }
      bucket.b.push_back(__float2half_rn(static_cast<float>(value)));
    }
    bucket.c.resize(bucket.c.size() +
                    static_cast<std::size_t>(problem.m) * problem.n, 0.0F);
  }
  buckets->clear();
  buckets->reserve(by_shape.size());
  for (auto& entry : by_shape) {
    buckets->push_back(std::move(entry.second));
  }
  return true;
}

template <typename T>
bool CopyVectorToDevice(const std::vector<T>& host, T** device,
                        std::string* error) {
  if (host.empty()) {
    *device = nullptr;
    return true;
  }
  cudaError_t cuda_error =
      cudaMalloc(reinterpret_cast<void**>(device), host.size() * sizeof(T));
  if (cuda_error != cudaSuccess) {
    *error = std::string("cudaMalloc: ") + cudaGetErrorString(cuda_error);
    return false;
  }
  cuda_error = cudaMemcpy(*device, host.data(), host.size() * sizeof(T),
                          cudaMemcpyHostToDevice);
  if (cuda_error != cudaSuccess) {
    cudaFree(*device);
    *device = nullptr;
    *error = std::string("cudaMemcpy H2D: ") + cudaGetErrorString(cuda_error);
    return false;
  }
  return true;
}

bool RunCublasFp16BatchedBaseline(const std::vector<CudaGemmProblem>& problems,
                                  double* kernel_ms, std::size_t* buckets,
                                  std::string* error) {
  std::vector<CublasBucket> host_buckets;
  if (!BuildCublasBuckets(problems, &host_buckets, error)) {
    return false;
  }
  *buckets = host_buckets.size();

  cublasHandle_t handle = nullptr;
  cublasStatus_t cublas_status = cublasCreate(&handle);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    *error = "cublasCreate: " + CublasStatusName(cublas_status);
    return false;
  }
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaError_t cuda_error = cudaEventCreate(&start);
  if (cuda_error != cudaSuccess) {
    cublasDestroy(handle);
    *error = std::string("cudaEventCreate start: ") +
             cudaGetErrorString(cuda_error);
    return false;
  }
  cuda_error = cudaEventCreate(&stop);
  if (cuda_error != cudaSuccess) {
    cudaEventDestroy(start);
    cublasDestroy(handle);
    *error = std::string("cudaEventCreate stop: ") +
             cudaGetErrorString(cuda_error);
    return false;
  }

  std::vector<__half*> device_a(host_buckets.size(), nullptr);
  std::vector<__half*> device_b(host_buckets.size(), nullptr);
  std::vector<float*> device_c(host_buckets.size(), nullptr);

  for (std::size_t i = 0; i < host_buckets.size(); ++i) {
    CublasBucket& bucket = host_buckets[i];
    if (!CopyVectorToDevice(bucket.a, &device_a[i], error) ||
        !CopyVectorToDevice(bucket.b, &device_b[i], error) ||
        !CopyVectorToDevice(bucket.c, &device_c[i], error)) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cublasDestroy(handle);
      return false;
    }
  }

  const float alpha = 1.0F;
  const float beta = 0.0F;
  auto run_bucket = [&](std::size_t i) -> cublasStatus_t {
    const CublasBucket& bucket = host_buckets[i];
    const long long a_tile =
        static_cast<long long>(bucket.shape.m) * bucket.shape.k;
    const long long b_tile =
        static_cast<long long>(bucket.shape.k) * bucket.shape.n;
    const long long c_tile =
        static_cast<long long>(bucket.shape.m) * bucket.shape.n;
    const int batch_count =
        static_cast<int>(bucket.c.size() / static_cast<std::size_t>(c_tile));
    const int m = static_cast<int>(bucket.shape.n);
    const int n = static_cast<int>(bucket.shape.m);
    const int k = static_cast<int>(bucket.shape.k);
    return cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, device_b[i],
        CUDA_R_16F, m, b_tile, device_a[i], CUDA_R_16F, k, a_tile, &beta,
        device_c[i], CUDA_R_32F, m, c_tile, batch_count,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  };

  for (std::size_t i = 0; i < host_buckets.size(); ++i) {
    cublas_status = run_bucket(i);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cublasDestroy(handle);
      *error = "cublasGemmStridedBatchedEx warmup: " +
               CublasStatusName(cublas_status);
      return false;
    }
  }
  cuda_error = cudaDeviceSynchronize();
  if (cuda_error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    *error = std::string("cuBLAS baseline warmup synchronize: ") +
             cudaGetErrorString(cuda_error);
    return false;
  }

  cudaEventRecord(start);
  for (std::size_t i = 0; i < host_buckets.size(); ++i) {
    cublas_status = run_bucket(i);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cublasDestroy(handle);
      *error = "cublasGemmStridedBatchedEx: " +
               CublasStatusName(cublas_status);
      return false;
    }
  }
  cudaEventRecord(stop);
  cuda_error = cudaEventSynchronize(stop);
  if (cuda_error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    *error = std::string("cuBLAS baseline synchronize: ") +
             cudaGetErrorString(cuda_error);
    return false;
  }
  float elapsed = 0.0F;
  cudaEventElapsedTime(&elapsed, start, stop);
  *kernel_ms = static_cast<double>(elapsed);

  for (std::size_t i = 0; i < host_buckets.size(); ++i) {
    cudaFree(device_a[i]);
    cudaFree(device_b[i]);
    cudaFree(device_c[i]);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cublasDestroy(handle);
  return true;
}

bool SetLtLayoutBatch(cublasLtMatrixLayout_t layout, int batch_count,
                      long long stride, std::string* error) {
  cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
      sizeof(batch_count));
  if (status != CUBLAS_STATUS_SUCCESS) {
    *error = "cublasLtMatrixLayoutSetAttribute batch_count: " +
             CublasStatusName(status);
    return false;
  }
  status = cublasLtMatrixLayoutSetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
      sizeof(stride));
  if (status != CUBLAS_STATUS_SUCCESS) {
    *error = "cublasLtMatrixLayoutSetAttribute stride: " +
             CublasStatusName(status);
    return false;
  }
  return true;
}

bool RunCublasLtFp16BatchedBaseline(
    const std::vector<CudaGemmProblem>& problems, double* kernel_ms,
    std::size_t* buckets, std::string* error) {
  std::vector<CublasBucket> host_buckets;
  if (!BuildCublasBuckets(problems, &host_buckets, error)) {
    return false;
  }
  *buckets = host_buckets.size();

  cublasLtHandle_t handle = nullptr;
  cublasStatus_t status = cublasLtCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    *error = "cublasLtCreate: " + CublasStatusName(status);
    return false;
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaError_t cuda_error = cudaEventCreate(&start);
  if (cuda_error != cudaSuccess) {
    cublasLtDestroy(handle);
    *error = std::string("cudaEventCreate start: ") +
             cudaGetErrorString(cuda_error);
    return false;
  }
  cuda_error = cudaEventCreate(&stop);
  if (cuda_error != cudaSuccess) {
    cudaEventDestroy(start);
    cublasLtDestroy(handle);
    *error = std::string("cudaEventCreate stop: ") +
             cudaGetErrorString(cuda_error);
    return false;
  }

  std::vector<__half*> device_a(host_buckets.size(), nullptr);
  std::vector<__half*> device_b(host_buckets.size(), nullptr);
  std::vector<float*> device_c(host_buckets.size(), nullptr);
  for (std::size_t i = 0; i < host_buckets.size(); ++i) {
    CublasBucket& bucket = host_buckets[i];
    if (!CopyVectorToDevice(bucket.a, &device_a[i], error) ||
        !CopyVectorToDevice(bucket.b, &device_b[i], error) ||
        !CopyVectorToDevice(bucket.c, &device_c[i], error)) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cublasLtDestroy(handle);
      return false;
    }
  }

  cublasLtMatmulDesc_t op = nullptr;
  status = cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasLtDestroy(handle);
    *error = "cublasLtMatmulDescCreate: " + CublasStatusName(status);
    return false;
  }
  const cublasOperation_t no_transpose = CUBLAS_OP_N;
  status = cublasLtMatmulDescSetAttribute(
      op, CUBLASLT_MATMUL_DESC_TRANSA, &no_transpose, sizeof(no_transpose));
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulDescSetAttribute(
        op, CUBLASLT_MATMUL_DESC_TRANSB, &no_transpose,
        sizeof(no_transpose));
  }
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulDescDestroy(op);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasLtDestroy(handle);
    *error = "cublasLtMatmulDescSetAttribute transpose: " +
             CublasStatusName(status);
    return false;
  }

  auto run_bucket = [&](std::size_t i) -> cublasStatus_t {
    const CublasBucket& bucket = host_buckets[i];
    const long long a_tile =
        static_cast<long long>(bucket.shape.m) * bucket.shape.k;
    const long long b_tile =
        static_cast<long long>(bucket.shape.k) * bucket.shape.n;
    const long long c_tile =
        static_cast<long long>(bucket.shape.m) * bucket.shape.n;
    const int batch_count =
        static_cast<int>(bucket.c.size() / static_cast<std::size_t>(c_tile));
    const int m = static_cast<int>(bucket.shape.n);
    const int n = static_cast<int>(bucket.shape.m);
    const int k = static_cast<int>(bucket.shape.k);

    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasStatus_t st =
        cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16F, m, k, m);
    if (st == CUBLAS_STATUS_SUCCESS) {
      st = cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16F, k, n, k);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
      st = cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, m, n, m);
    }
    if (st != CUBLAS_STATUS_SUCCESS) {
      if (a_desc != nullptr) cublasLtMatrixLayoutDestroy(a_desc);
      if (b_desc != nullptr) cublasLtMatrixLayoutDestroy(b_desc);
      if (c_desc != nullptr) cublasLtMatrixLayoutDestroy(c_desc);
      return st;
    }
    bool attrs_ok = SetLtLayoutBatch(a_desc, batch_count, b_tile, error) &&
                    SetLtLayoutBatch(b_desc, batch_count, a_tile, error) &&
                    SetLtLayoutBatch(c_desc, batch_count, c_tile, error);
    if (!attrs_ok) {
      cublasLtMatrixLayoutDestroy(a_desc);
      cublasLtMatrixLayoutDestroy(b_desc);
      cublasLtMatrixLayoutDestroy(c_desc);
      return CUBLAS_STATUS_INVALID_VALUE;
    }
    const float alpha = 1.0F;
    const float beta = 0.0F;
    st = cublasLtMatmul(handle, op, &alpha, device_b[i], a_desc, device_a[i],
                        b_desc, &beta, device_c[i], c_desc, device_c[i],
                        c_desc, nullptr, nullptr, 0, nullptr);
    cublasLtMatrixLayoutDestroy(a_desc);
    cublasLtMatrixLayoutDestroy(b_desc);
    cublasLtMatrixLayoutDestroy(c_desc);
    return st;
  };

  for (std::size_t i = 0; i < host_buckets.size(); ++i) {
    status = run_bucket(i);
    if (status != CUBLAS_STATUS_SUCCESS) {
      cublasLtMatmulDescDestroy(op);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cublasLtDestroy(handle);
      *error = error->empty()
                   ? "cublasLtMatmul warmup: " + CublasStatusName(status)
                   : *error;
      return false;
    }
  }
  cuda_error = cudaDeviceSynchronize();
  if (cuda_error != cudaSuccess) {
    cublasLtMatmulDescDestroy(op);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasLtDestroy(handle);
    *error = std::string("cuBLASLt baseline warmup synchronize: ") +
             cudaGetErrorString(cuda_error);
    return false;
  }

  cudaEventRecord(start);
  for (std::size_t i = 0; i < host_buckets.size(); ++i) {
    status = run_bucket(i);
    if (status != CUBLAS_STATUS_SUCCESS) {
      cublasLtMatmulDescDestroy(op);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cublasLtDestroy(handle);
      *error = error->empty() ? "cublasLtMatmul: " + CublasStatusName(status)
                              : *error;
      return false;
    }
  }
  cudaEventRecord(stop);
  cuda_error = cudaEventSynchronize(stop);
  if (cuda_error != cudaSuccess) {
    cublasLtMatmulDescDestroy(op);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasLtDestroy(handle);
    *error = std::string("cuBLASLt baseline synchronize: ") +
             cudaGetErrorString(cuda_error);
    return false;
  }
  float elapsed = 0.0F;
  cudaEventElapsedTime(&elapsed, start, stop);
  *kernel_ms = static_cast<double>(elapsed);

  for (std::size_t i = 0; i < host_buckets.size(); ++i) {
    cudaFree(device_a[i]);
    cudaFree(device_b[i]);
    cudaFree(device_c[i]);
  }
  cublasLtMatmulDescDestroy(op);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cublasLtDestroy(handle);
  return true;
}

}  // namespace

int main() {
  const auto runtime_status = CudaRuntimeStatus();
  if (!runtime_status.ok()) {
    std::cerr << "CUDA runtime/device is not available: "
              << runtime_status.message() << '\n';
    return 77;
  }
  std::vector<CudaGemmProblem> problems;
  for (std::size_t i = 0; i < 128; ++i) {
    const std::uint32_t edge = i % 3 == 0 ? 16U : (i % 3 == 1 ? 32U : 64U);
    CudaGemmProblem problem;
    problem.m = edge;
    problem.k = edge;
    problem.n = edge;
    problem.a_scale = 1.0 + 0.03125 * static_cast<double>(i % 17);
    problem.b_scale = (i % 2 == 0) ? 0.5 : -0.25;
    problem.epilogue_scale = (i % 5 == 0) ? -2.0 : 1.0;
    problem.a = MakeDense(edge, edge, 0x500000ULL + i);
    problem.b = MakeDense(edge, edge, 0x600000ULL + i);
    problems.push_back(std::move(problem));
  }

  const auto cpu_start = std::chrono::steady_clock::now();
  std::uint64_t products = 0;
  double scale_abs_sum = 0.0;
  for (const CudaGemmProblem& problem : problems) {
    auto cpu = GemmF64eReference(problem.m, problem.k, problem.n, problem.a,
                                problem.b);
    if (!cpu.ok()) {
      std::cerr << "CPU reference failed: " << cpu.status().message() << '\n';
      return 1;
    }
    products += static_cast<std::uint64_t>(problem.m) * problem.k * problem.n;
    scale_abs_sum +=
        std::abs(problem.a_scale * problem.b_scale * problem.epilogue_scale);
  }
  const auto cpu_end = std::chrono::steady_clock::now();
  const auto gpu_start = std::chrono::steady_clock::now();
  auto gpu = GroupedGemmFp64Cuda(problems);
  const auto gpu_end = std::chrono::steady_clock::now();
  const auto mixed_start = std::chrono::steady_clock::now();
  auto mixed = GroupedGemmFp16AccumCuda(problems);
  const auto mixed_end = std::chrono::steady_clock::now();
  const auto plan_build_start = std::chrono::steady_clock::now();
  auto mixed_plan = BuildGroupedGemmFp16AccumCudaPlan(problems);
  const auto plan_build_end = std::chrono::steady_clock::now();
  if (!gpu.ok()) {
    std::cerr << "GroupedGemmFp64Cuda failed: " << gpu.status().message()
              << '\n';
    return 1;
  }
  if (!mixed.ok()) {
    std::cerr << "GroupedGemmFp16AccumCuda failed: "
              << mixed.status().message() << '\n';
    return 1;
  }
  if (!mixed_plan.ok()) {
    std::cerr << "BuildGroupedGemmFp16AccumCudaPlan failed: "
              << mixed_plan.status().message() << '\n';
    return 1;
  }
  for (int warmup = 0; warmup < 5; ++warmup) {
    auto planned_warmup = RunGroupedGemmFp16AccumCudaPlan(mixed_plan.value());
    if (!planned_warmup.ok()) {
      std::cerr << "RunGroupedGemmFp16AccumCudaPlan warmup failed: "
                << planned_warmup.status().message() << '\n';
      return 1;
    }
  }
  const auto planned_start = std::chrono::steady_clock::now();
  tides::tile::CudaGroupedGemmResult planned_best;
  bool planned_best_valid = false;
  constexpr int kTimedRepeats = 10;
  for (int sample = 0; sample < kTimedRepeats; ++sample) {
    auto planned = RunGroupedGemmFp16AccumCudaPlan(mixed_plan.value());
    if (!planned.ok()) {
      std::cerr << "RunGroupedGemmFp16AccumCudaPlan failed: "
                << planned.status().message() << '\n';
      return 1;
    }
    if (planned.value().c_tiles != mixed.value().c_tiles) {
      std::cerr << "planned mixed GEMM differs from one-shot mixed GEMM\n";
      return 1;
    }
    if (!planned_best_valid ||
        planned.value().kernel_ms < planned_best.kernel_ms) {
      planned_best = std::move(planned.value());
      planned_best_valid = true;
    }
  }
  const auto planned_end = std::chrono::steady_clock::now();
  double cublas_kernel_ms = 0.0;
  std::size_t cublas_shape_buckets = 0;
  std::string cublas_error;
  bool cublas_ok = false;
  for (int sample = 0; sample < 5; ++sample) {
    double sample_ms = 0.0;
    std::size_t sample_buckets = 0;
    std::string sample_error;
    const bool sample_ok = RunCublasFp16BatchedBaseline(
        problems, &sample_ms, &sample_buckets, &sample_error);
    if (!sample_ok) {
      cublas_error = sample_error;
      continue;
    }
    if (!cublas_ok || sample_ms < cublas_kernel_ms) {
      cublas_ok = true;
      cublas_kernel_ms = sample_ms;
      cublas_shape_buckets = sample_buckets;
    }
  }
  double cublaslt_kernel_ms = 0.0;
  std::size_t cublaslt_shape_buckets = 0;
  std::string cublaslt_error;
  bool cublaslt_ok = false;
  for (int sample = 0; sample < 5; ++sample) {
    double sample_ms = 0.0;
    std::size_t sample_buckets = 0;
    std::string sample_error;
    const bool sample_ok = RunCublasLtFp16BatchedBaseline(
        problems, &sample_ms, &sample_buckets, &sample_error);
    if (!sample_ok) {
      cublaslt_error = sample_error;
      continue;
    }
    if (!cublaslt_ok || sample_ms < cublaslt_kernel_ms) {
      cublaslt_ok = true;
      cublaslt_kernel_ms = sample_ms;
      cublaslt_shape_buckets = sample_buckets;
    }
  }

  const double cpu_ms =
      std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
  const double gpu_total_ms =
      std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
  const double mixed_total_ms =
      std::chrono::duration<double, std::milli>(mixed_end - mixed_start)
          .count();
  const double plan_build_ms =
      std::chrono::duration<double, std::milli>(plan_build_end -
                                                plan_build_start)
          .count();
  const double planned_total_ms =
      std::chrono::duration<double, std::milli>(planned_end - planned_start)
          .count();
  const double flops = 2.0 * static_cast<double>(products);
  std::cout << "cuda_gemm_probe: problems=" << problems.size()
            << " products=" << products
            << " scale_abs_sum=" << scale_abs_sum
            << " cpu_ref_ms=" << cpu_ms
            << " gpu_total_ms=" << gpu_total_ms
            << " gpu_kernel_ms=" << gpu.value().kernel_ms
            << " kernel_gflops=" << flops / (gpu.value().kernel_ms * 1.0e6)
            << " mixed_total_ms=" << mixed_total_ms
            << " mixed_kernel_ms=" << mixed.value().kernel_ms
            << " mixed_postprocess_ms=" << mixed.value().postprocess_ms
            << " mixed_shape_buckets="
            << mixed.value().backend_shape_buckets
            << " mixed_kernel_gflops="
            << flops / (mixed.value().kernel_ms * 1.0e6)
            << " plan_build_ms=" << plan_build_ms
            << " planned_total_ms=" << planned_total_ms
            << " planned_samples=" << kTimedRepeats
            << " planned_kernel_ms=" << planned_best.kernel_ms
            << " planned_postprocess_ms=" << planned_best.postprocess_ms
            << " planned_shape_buckets="
            << planned_best.backend_shape_buckets
            << " planned_device_total_ms="
            << planned_best.kernel_ms + planned_best.postprocess_ms
            << " planned_kernel_gflops="
            << flops / (planned_best.kernel_ms * 1.0e6)
            << " cublas_shape_buckets=" << cublas_shape_buckets
            << " cublas_status=" << (cublas_ok ? "ok" : cublas_error)
            << " cublas_kernel_ms=" << cublas_kernel_ms
            << " cublas_kernel_gflops="
            << (cublas_ok ? flops / (cublas_kernel_ms * 1.0e6) : 0.0)
            << " cublaslt_shape_buckets=" << cublaslt_shape_buckets
            << " cublaslt_status=" << (cublaslt_ok ? "ok" : cublaslt_error)
            << " cublaslt_kernel_ms=" << cublaslt_kernel_ms
            << " cublaslt_kernel_gflops="
            << (cublaslt_ok ? flops / (cublaslt_kernel_ms * 1.0e6) : 0.0)
            << " planned_vs_cublas_ratio="
            << (cublas_ok ? planned_best.kernel_ms / cublas_kernel_ms : 0.0)
            << " planned_vs_cublaslt_ratio="
            << (cublaslt_ok ? planned_best.kernel_ms / cublaslt_kernel_ms
                            : 0.0)
            << " mixed_abs_bound="
            << mixed.value().ledger.entries().front().budget.bound
            << '\n';
  return 0;
}
