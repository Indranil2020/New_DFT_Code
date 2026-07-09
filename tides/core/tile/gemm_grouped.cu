#include "tile/gemm_grouped.hpp"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using namespace nvcuda;

namespace tides::tile {
namespace {

constexpr int kBlockEdge = 16;

[[nodiscard]] Status CudaStatus(cudaError_t error, const char* context) {
  if (error == cudaSuccess) {
    return Status::Ok();
  }
  return Status::IoError(std::string(context) + ": " +
                         cudaGetErrorString(error));
}

[[nodiscard]] const char* CublasStatusName(cublasStatus_t status) {
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

[[nodiscard]] Status CublasStatus(cublasStatus_t status, const char* context) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return Status::Ok();
  }
  return Status::IoError(std::string(context) + ": " +
                         CublasStatusName(status));
}

constexpr int kGemmTileK = 16;

__global__ void GroupedGemmKernel(const double* a_all, const double* b_all,
                                  double* c_all,
                                  const std::uint64_t* a_offsets,
                                  const std::uint64_t* b_offsets,
                                  const std::uint64_t* c_offsets,
                                  const std::uint32_t* ms,
                                  const std::uint32_t* ks,
                                  const std::uint32_t* ns,
                                  const double* epilogue_scales) {
  const std::uint32_t problem = static_cast<std::uint32_t>(blockIdx.z);
  const std::uint32_t row =
      static_cast<std::uint32_t>(blockIdx.y * blockDim.y + threadIdx.y);
  const std::uint32_t col =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const std::uint32_t m = ms[problem];
  const std::uint32_t k = ks[problem];
  const std::uint32_t n = ns[problem];

  const double* a = a_all + a_offsets[problem];
  const double* b = b_all + b_offsets[problem];
  double* c = c_all + c_offsets[problem];

  if (k >= static_cast<std::uint32_t>(kGemmTileK)) {
    extern __shared__ double smem[];
    double* s_a = smem;
    double* s_b = smem + kBlockEdge * kGemmTileK;

    double sum = 0.0;
    const std::uint32_t num_tiles = (k + kGemmTileK - 1) / kGemmTileK;
    for (std::uint32_t t = 0; t < num_tiles; ++t) {
      const std::uint32_t k_base = t * kGemmTileK;
      const std::uint32_t local_k = threadIdx.x;
      if (row < m && local_k < kGemmTileK && k_base + local_k < k) {
        s_a[threadIdx.y * kGemmTileK + local_k] = a[row * k + k_base + local_k];
      } else {
        s_a[threadIdx.y * kGemmTileK + local_k] = 0.0;
      }
      const std::uint32_t b_row = k_base + threadIdx.y;
      if (col < n && threadIdx.y < kGemmTileK && b_row < k) {
        s_b[threadIdx.y * kBlockEdge + threadIdx.x] = b[b_row * n + col];
      } else {
        s_b[threadIdx.y * kBlockEdge + threadIdx.x] = 0.0;
      }
      __syncthreads();

      if (row < m && col < n) {
        for (std::uint32_t p = 0; p < kGemmTileK && k_base + p < k; ++p) {
          sum += s_a[threadIdx.y * kGemmTileK + p] *
                 s_b[p * kBlockEdge + threadIdx.x];
        }
      }
      __syncthreads();
    }
    if (row < m && col < n) {
      c[row * n + col] = epilogue_scales[problem] * sum;
    }
  } else {
    if (row >= m || col >= n) {
      return;
    }
    double sum = 0.0;
    for (std::uint32_t p = 0; p < k; ++p) {
      sum += a[row * k + p] * b[p * n + col];
    }
    c[row * n + col] = epilogue_scales[problem] * sum;
  }
}

__global__ void GroupedGemmFp16AccumKernel(
    const __half* a_all, const __half* b_all, double* c_all,
    const std::uint64_t* a_offsets, const std::uint64_t* b_offsets,
    const std::uint64_t* c_offsets, const std::uint32_t* ms,
    const std::uint32_t* ks, const std::uint32_t* ns,
    const double* epilogue_scales) {
  const std::uint32_t problem = static_cast<std::uint32_t>(blockIdx.z);
  const std::uint32_t row =
      static_cast<std::uint32_t>(blockIdx.y * blockDim.y + threadIdx.y);
  const std::uint32_t col =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const std::uint32_t m = ms[problem];
  const std::uint32_t k = ks[problem];
  const std::uint32_t n = ns[problem];
  if (row >= m || col >= n) {
    return;
  }

  const __half* a = a_all + a_offsets[problem];
  const __half* b = b_all + b_offsets[problem];
  double* c = c_all + c_offsets[problem];
  float sum = 0.0F;
  for (std::uint32_t p = 0; p < k; ++p) {
    sum = fmaf(__half2float(a[row * k + p]), __half2float(b[p * n + col]),
               sum);
  }
  c[row * n + col] =
      epilogue_scales[problem] * static_cast<double>(sum);
}

// WMMA-based FP16 GEMM kernel using Ampere tensor cores.
// Each warp computes a 16x16 output tile via mma.sync 16x16x16.
// Block has 4 warps (128 threads) → 2x2 output tile grid per block.
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kWarpsPerBlock = 4;
constexpr int kWmmaTileX = 2;  // 2 tiles in N direction per block
constexpr int kWmmaTileY = 2;  // 2 tiles in M direction per block

__global__ void GroupedGemmFp16AccumWmmaKernel(
    const __half* a_all, const __half* b_all, double* c_all,
    const std::uint64_t* a_offsets, const std::uint64_t* b_offsets,
    const std::uint64_t* c_offsets, const std::uint32_t* ms,
    const std::uint32_t* ks, const std::uint32_t* ns,
    const double* epilogue_scales) {
  const std::uint32_t problem = static_cast<std::uint32_t>(blockIdx.z);
  const std::uint32_t m = ms[problem];
  const std::uint32_t k = ks[problem];
  const std::uint32_t n = ns[problem];

  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  // Each warp handles one 16x16 output tile
  const int warp_row = warp_id / kWmmaTileX;  // 0 or 1
  const int warp_col = warp_id % kWmmaTileX;  // 0 or 1

  const int tile_row = static_cast<int>(blockIdx.y) * kWmmaTileY + warp_row;
  const int tile_col = static_cast<int>(blockIdx.x) * kWmmaTileX + warp_col;

  const int row0 = tile_row * kWmmaM;
  const int col0 = tile_col * kWmmaN;
  if (row0 >= static_cast<int>(m) || col0 >= static_cast<int>(n)) return;

  const __half* a = a_all + a_offsets[problem];
  const __half* b = b_all + b_offsets[problem];
  double* c = c_all + c_offsets[problem];

  wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;

  wmma::fill_fragment(c_frag, 0.0f);

  for (int ki = 0; ki < static_cast<int>(k); ki += kWmmaK) {
    int k_remaining = static_cast<int>(k) - ki;
    if (k_remaining >= kWmmaK) {
      // Full tile — use WMMA
      wmma::load_matrix_sync(a_frag, a + row0 * k + ki, k);
      wmma::load_matrix_sync(b_frag, b + ki * n + col0, n);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    } else {
      // Partial K tile — pad with zeros in shared memory
      __shared__ __half a_pad[kWmmaTileY * kWmmaM * kWmmaK];
      __shared__ __half b_pad[kWmmaTileX * kWmmaN * kWmmaK];
      __half* a_smem = a_pad + warp_row * kWmmaM * kWmmaK;
      __half* b_smem = b_pad + warp_col * kWmmaN * kWmmaK;

      // Cooperatively load and zero-pad the K dimension
      for (int idx = lane; idx < kWmmaM * kWmmaK; idx += 32) {
        int i = idx / kWmmaK;
        int p = idx % kWmmaK;
        int r = row0 + i;
        a_smem[i * kWmmaK + p] = (r < static_cast<int>(m) && p < k_remaining)
            ? a[r * k + ki + p] : __float2half(0.0f);
      }
      for (int idx = lane; idx < kWmmaN * kWmmaK; idx += 32) {
        int j = idx / kWmmaK;
        int p = idx % kWmmaK;
        int cc = col0 + j;
        b_smem[j * kWmmaK + p] = (cc < static_cast<int>(n) && p < k_remaining)
            ? b[(ki + p) * n + cc] : __float2half(0.0f);
      }
      __syncwarp();
      wmma::load_matrix_sync(a_frag, a_smem, kWmmaK);
      wmma::load_matrix_sync(b_frag, b_smem, kWmmaK);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
  }

  // Store with epilogue scaling
  for (int i = 0; i < c_frag.num_elements; ++i) {
    c_frag.x[i] = static_cast<float>(c_frag.x[i] * static_cast<float>(epilogue_scales[problem]));
  }

  // Store to global memory via shared memory
  __shared__ float c_smem[kWmmaTileY * kWmmaM * kWmmaTileX * kWmmaN];
  float* tile_smem = c_smem + warp_id * kWmmaM * kWmmaN;

  if (row0 + kWmmaM <= static_cast<int>(m) && col0 + kWmmaN <= static_cast<int>(n)) {
    // Full tile — use store_matrix_sync into shared memory
    wmma::store_matrix_sync(tile_smem, c_frag, kWmmaN, wmma::mem_row_major);
    __syncwarp();
    if (lane < kWmmaM * kWmmaN) {
      int i = lane / kWmmaN;
      int j = lane % kWmmaN;
      c[(row0 + i) * n + col0 + j] = static_cast<double>(tile_smem[i * kWmmaN + j]);
    }
  } else {
    // Boundary tile — element-wise store from fragment
    for (int i = 0; i < kWmmaM; ++i) {
      int r = row0 + i;
      if (r >= static_cast<int>(m)) break;
      for (int j = 0; j < kWmmaN; ++j) {
        int cc = col0 + j;
        if (cc >= static_cast<int>(n)) break;
        if (lane == 0) {
          c[r * n + cc] = static_cast<double>(c_frag.x[i * kWmmaN + j]);
        }
      }
    }
  }
}

__global__ void ScaleBucketFloatGemmOutputKernel(
    const float* c_bucket, double* c_all, std::uint32_t first_problem,
    const std::uint64_t* c_offsets, const double* epilogue_scales,
    std::uint32_t m, std::uint32_t n) {
  const std::uint32_t local_problem = static_cast<std::uint32_t>(blockIdx.z);
  const std::uint32_t global_problem = first_problem + local_problem;
  const std::uint32_t row =
      static_cast<std::uint32_t>(blockIdx.y * blockDim.y + threadIdx.y);
  const std::uint32_t col =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m || col >= n) {
    return;
  }
  const std::uint64_t local_index =
      static_cast<std::uint64_t>(local_problem) * m * n + row * n + col;
  const std::uint64_t output_index = c_offsets[global_problem] + row * n + col;
  c_all[output_index] = epilogue_scales[global_problem] *
                        static_cast<double>(c_bucket[local_index]);
}

template <typename T>
Status CopyToDevice(const std::vector<T>& host, T** device) {
  if (host.empty()) {
    *device = nullptr;
    return Status::Ok();
  }
  const std::size_t bytes = host.size() * sizeof(T);
  cudaError_t error = cudaMalloc(reinterpret_cast<void**>(device), bytes);
  if (error != cudaSuccess) {
    return CudaStatus(error, "cudaMalloc");
  }
  void* pinned = nullptr;
  error = cudaMallocHost(&pinned, bytes);
  if (error == cudaSuccess) {
    std::memcpy(pinned, host.data(), bytes);
    error = cudaMemcpy(*device, pinned, bytes, cudaMemcpyHostToDevice);
    cudaFreeHost(pinned);
  } else {
    error = cudaMemcpy(*device, host.data(), bytes, cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) {
    cudaFree(*device);
    *device = nullptr;
    return CudaStatus(error, "cudaMemcpy H2D");
  }
  return Status::Ok();
}

template <typename T>
Status CopySpanToDevice(const T* host, std::size_t count, T** device) {
  if (count == 0) {
    *device = nullptr;
    return Status::Ok();
  }
  const std::size_t bytes = count * sizeof(T);
  cudaError_t error = cudaMalloc(reinterpret_cast<void**>(device), bytes);
  if (error != cudaSuccess) {
    return CudaStatus(error, "cudaMalloc");
  }
  void* pinned = nullptr;
  error = cudaMallocHost(&pinned, bytes);
  if (error == cudaSuccess) {
    std::memcpy(pinned, host, bytes);
    error = cudaMemcpy(*device, pinned, bytes, cudaMemcpyHostToDevice);
    cudaFreeHost(pinned);
  } else {
    error = cudaMemcpy(*device, host, bytes, cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) {
    cudaFree(*device);
    *device = nullptr;
    return CudaStatus(error, "cudaMemcpy H2D");
  }
  return Status::Ok();
}

template <typename T>
void FreeDevice(T* ptr) {
  if (ptr != nullptr) {
    cudaFree(ptr);
  }
}

struct PackedProblems {
  std::vector<std::uint64_t> a_offsets;
  std::vector<std::uint64_t> b_offsets;
  std::vector<std::uint64_t> c_offsets;
  std::vector<std::uint32_t> ms;
  std::vector<std::uint32_t> ks;
  std::vector<std::uint32_t> ns;
  std::vector<double> epilogue_scales;
  std::vector<double> a_all;
  std::vector<double> b_all;
  std::vector<double> c_all;
  std::uint32_t max_m = 0;
  std::uint32_t max_n = 0;
  std::uint64_t products = 0;
  bool uses_nontrivial_scale = false;
};

struct MixedShapeBucket {
  std::uint32_t m = 0;
  std::uint32_t k = 0;
  std::uint32_t n = 0;
  std::uint32_t first_problem = 0;
  std::uint64_t a_offset = 0;
  std::uint64_t b_offset = 0;
  std::uint64_t c_offset = 0;
  std::uint32_t count = 0;
};

struct PackedMixedProblems {
  std::vector<std::uint64_t> a_offsets;
  std::vector<std::uint64_t> b_offsets;
  std::vector<std::uint64_t> c_offsets;
  std::vector<std::uint32_t> ms;
  std::vector<std::uint32_t> ks;
  std::vector<std::uint32_t> ns;
  std::vector<double> epilogue_scales;
  std::vector<__half> a_all;
  std::vector<__half> b_all;
  std::vector<double> c_all;
  std::vector<std::size_t> original_indices;
  std::vector<MixedShapeBucket> buckets;
  std::uint32_t max_m = 0;
  std::uint32_t max_n = 0;
  std::uint64_t products = 0;
  double max_abs_error_bound = 0.0;
};

Result<PackedProblems> PackProblems(
    const std::vector<CudaGemmProblem>& problems) {
  if (problems.empty()) {
    return Status::InvalidArgument("grouped GEMM problem list is empty");
  }
  PackedProblems packed;
  packed.a_offsets.assign(problems.size(), 0);
  packed.b_offsets.assign(problems.size(), 0);
  packed.c_offsets.assign(problems.size(), 0);
  packed.ms.assign(problems.size(), 0);
  packed.ks.assign(problems.size(), 0);
  packed.ns.assign(problems.size(), 0);
  packed.epilogue_scales.assign(problems.size(), 1.0);

  for (std::size_t i = 0; i < problems.size(); ++i) {
    const CudaGemmProblem& problem = problems[i];
    if (problem.m == 0 || problem.k == 0 || problem.n == 0) {
      return Status::InvalidArgument("grouped GEMM dimensions must be nonzero");
    }
    const std::size_t a_size =
        static_cast<std::size_t>(problem.m) * problem.k;
    const std::size_t b_size =
        static_cast<std::size_t>(problem.k) * problem.n;
    const std::size_t c_size =
        static_cast<std::size_t>(problem.m) * problem.n;
    if (problem.a.size() != a_size || problem.b.size() != b_size) {
      return Status::InvalidArgument("grouped GEMM payload size mismatch");
    }
    if (!std::isfinite(problem.a_scale) ||
        !std::isfinite(problem.b_scale) ||
        !std::isfinite(problem.epilogue_scale)) {
      return Status::InvalidArgument("grouped GEMM scales must be finite");
    }
    packed.a_offsets[i] = static_cast<std::uint64_t>(packed.a_all.size());
    packed.b_offsets[i] = static_cast<std::uint64_t>(packed.b_all.size());
    packed.c_offsets[i] = static_cast<std::uint64_t>(packed.c_all.size());
    packed.ms[i] = problem.m;
    packed.ks[i] = problem.k;
    packed.ns[i] = problem.n;
    packed.epilogue_scales[i] =
        problem.a_scale * problem.b_scale * problem.epilogue_scale;
    if (!std::isfinite(packed.epilogue_scales[i])) {
      return Status::InvalidArgument(
          "grouped GEMM combined epilogue scale must be finite");
    }
    packed.uses_nontrivial_scale =
        packed.uses_nontrivial_scale ||
        problem.a_scale != 1.0 || problem.b_scale != 1.0 ||
        problem.epilogue_scale != 1.0;
    packed.a_all.insert(packed.a_all.end(), problem.a.begin(), problem.a.end());
    packed.b_all.insert(packed.b_all.end(), problem.b.begin(), problem.b.end());
    packed.c_all.resize(packed.c_all.size() + c_size, 0.0);
    packed.max_m = std::max(packed.max_m, problem.m);
    packed.max_n = std::max(packed.max_n, problem.n);
    packed.products += static_cast<std::uint64_t>(problem.m) * problem.k *
                       problem.n;
  }
  return packed;
}

[[nodiscard]] double Fp32DotForwardAbsBound(
    std::size_t k, long double sum_abs_products) {
  constexpr long double unit_roundoff =
      static_cast<long double>(std::numeric_limits<float>::epsilon()) / 2.0L;
  const long double ku = static_cast<long double>(k) * unit_roundoff;
  const long double gamma = ku < 1.0L ? ku / (1.0L - ku) : ku;
  return static_cast<double>(gamma * sum_abs_products);
}

[[nodiscard]] Result<__half> ToHalfFinite(double value) {
  if (!std::isfinite(value) ||
      std::abs(value) > static_cast<double>(65504.0F)) {
    return Status::OutOfRange(
        "mixed grouped GEMM payload is outside finite FP16 range");
  }
  const __half half_value = __float2half_rn(static_cast<float>(value));
  const float roundtrip = __half2float(half_value);
  if (!std::isfinite(roundtrip)) {
    return Status::OutOfRange(
        "mixed grouped GEMM payload rounded to non-finite FP16");
  }
  return half_value;
}

[[nodiscard]] double MixedProblemMaxAbsErrorBound(
    const CudaGemmProblem& problem, const std::vector<float>& a_quantized,
    const std::vector<float>& b_quantized) {
  double max_bound = 0.0;
  const long double scale_abs =
      std::abs(static_cast<long double>(problem.a_scale) *
               static_cast<long double>(problem.b_scale) *
               static_cast<long double>(problem.epilogue_scale));
  for (std::uint32_t row = 0; row < problem.m; ++row) {
    for (std::uint32_t col = 0; col < problem.n; ++col) {
      long double quantization_error = 0.0L;
      long double quantized_sum_abs = 0.0L;
      for (std::uint32_t p = 0; p < problem.k; ++p) {
        const std::size_t a_index = row * problem.k + p;
        const std::size_t b_index = p * problem.n + col;
        const long double exact =
            static_cast<long double>(problem.a[a_index]) *
            static_cast<long double>(problem.b[b_index]);
        const long double quantized =
            static_cast<long double>(a_quantized[a_index]) *
            static_cast<long double>(b_quantized[b_index]);
        const long double diff = exact - quantized;
        quantization_error += diff < 0.0L ? -diff : diff;
        quantized_sum_abs += quantized < 0.0L ? -quantized : quantized;
      }
      const long double bound =
          scale_abs *
          (quantization_error +
           Fp32DotForwardAbsBound(problem.k, quantized_sum_abs));
      max_bound = std::max(max_bound, static_cast<double>(bound));
    }
  }
  return max_bound;
}

Result<PackedMixedProblems> PackMixedProblems(
    const std::vector<CudaGemmProblem>& problems) {
  if (problems.empty()) {
    return Status::InvalidArgument("mixed grouped GEMM problem list is empty");
  }
  PackedMixedProblems packed;
  packed.a_offsets.assign(problems.size(), 0);
  packed.b_offsets.assign(problems.size(), 0);
  packed.c_offsets.assign(problems.size(), 0);
  packed.ms.assign(problems.size(), 0);
  packed.ks.assign(problems.size(), 0);
  packed.ns.assign(problems.size(), 0);
  packed.epilogue_scales.assign(problems.size(), 1.0);

  std::vector<std::size_t> order(problems.size(), 0);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(),
                   [&](std::size_t lhs, std::size_t rhs) {
                     const CudaGemmProblem& a = problems[lhs];
                     const CudaGemmProblem& b = problems[rhs];
                     if (a.m != b.m) return a.m < b.m;
                     if (a.k != b.k) return a.k < b.k;
                     if (a.n != b.n) return a.n < b.n;
                     return lhs < rhs;
                   });
  packed.original_indices.reserve(problems.size());

  for (std::size_t packed_i = 0; packed_i < order.size(); ++packed_i) {
    const std::size_t original_i = order[packed_i];
    const CudaGemmProblem& problem = problems[original_i];
    if (problem.m == 0 || problem.k == 0 || problem.n == 0) {
      return Status::InvalidArgument(
          "mixed grouped GEMM dimensions must be nonzero");
    }
    const std::size_t a_size =
        static_cast<std::size_t>(problem.m) * problem.k;
    const std::size_t b_size =
        static_cast<std::size_t>(problem.k) * problem.n;
    const std::size_t c_size =
        static_cast<std::size_t>(problem.m) * problem.n;
    if (problem.a.size() != a_size || problem.b.size() != b_size) {
      return Status::InvalidArgument(
          "mixed grouped GEMM payload size mismatch");
    }
    if (!std::isfinite(problem.a_scale) ||
        !std::isfinite(problem.b_scale) ||
        !std::isfinite(problem.epilogue_scale)) {
      return Status::InvalidArgument("mixed grouped GEMM scales must be finite");
    }

    std::vector<float> a_quantized(a_size, 0.0F);
    std::vector<float> b_quantized(b_size, 0.0F);
    packed.a_offsets[packed_i] =
        static_cast<std::uint64_t>(packed.a_all.size());
    packed.b_offsets[packed_i] =
        static_cast<std::uint64_t>(packed.b_all.size());
    packed.c_offsets[packed_i] =
        static_cast<std::uint64_t>(packed.c_all.size());
    packed.ms[packed_i] = problem.m;
    packed.ks[packed_i] = problem.k;
    packed.ns[packed_i] = problem.n;
    packed.original_indices.push_back(original_i);
    packed.epilogue_scales[packed_i] =
        problem.a_scale * problem.b_scale * problem.epilogue_scale;
    if (!std::isfinite(packed.epilogue_scales[packed_i])) {
      return Status::InvalidArgument(
          "mixed grouped GEMM combined epilogue scale must be finite");
    }
    if (packed.buckets.empty() || packed.buckets.back().m != problem.m ||
        packed.buckets.back().k != problem.k ||
        packed.buckets.back().n != problem.n) {
      packed.buckets.push_back(MixedShapeBucket{
          problem.m,
          problem.k,
          problem.n,
          static_cast<std::uint32_t>(packed_i),
          packed.a_offsets[packed_i],
          packed.b_offsets[packed_i],
          packed.c_offsets[packed_i],
          0});
    }
    ++packed.buckets.back().count;
    for (std::size_t j = 0; j < a_size; ++j) {
      auto half_value = ToHalfFinite(problem.a[j]);
      if (!half_value.ok()) return half_value.status();
      packed.a_all.push_back(half_value.value());
      a_quantized[j] = __half2float(half_value.value());
    }
    for (std::size_t j = 0; j < b_size; ++j) {
      auto half_value = ToHalfFinite(problem.b[j]);
      if (!half_value.ok()) return half_value.status();
      packed.b_all.push_back(half_value.value());
      b_quantized[j] = __half2float(half_value.value());
    }
    packed.c_all.resize(packed.c_all.size() + c_size, 0.0);
    packed.max_m = std::max(packed.max_m, problem.m);
    packed.max_n = std::max(packed.max_n, problem.n);
    packed.products += static_cast<std::uint64_t>(problem.m) * problem.k *
                       problem.n;
    packed.max_abs_error_bound = std::max(
        packed.max_abs_error_bound,
        MixedProblemMaxAbsErrorBound(problem, a_quantized, b_quantized));
  }
  return packed;
}

struct DevicePackedProblems {
  double* a = nullptr;
  double* b = nullptr;
  double* c = nullptr;
  std::uint64_t* a_offsets = nullptr;
  std::uint64_t* b_offsets = nullptr;
  std::uint64_t* c_offsets = nullptr;
  std::uint32_t* ms = nullptr;
  std::uint32_t* ks = nullptr;
  std::uint32_t* ns = nullptr;
  double* epilogue_scales = nullptr;
};

struct DevicePackedMixedProblems {
  __half* a = nullptr;
  __half* b = nullptr;
  double* c = nullptr;
  std::uint64_t* a_offsets = nullptr;
  std::uint64_t* b_offsets = nullptr;
  std::uint64_t* c_offsets = nullptr;
  std::uint32_t* ms = nullptr;
  std::uint32_t* ks = nullptr;
  std::uint32_t* ns = nullptr;
  double* epilogue_scales = nullptr;
};

struct DeviceMixedShapeBucket {
  __half* a = nullptr;
  __half* b = nullptr;
  float* c = nullptr;
  std::uint32_t m = 0;
  std::uint32_t k = 0;
  std::uint32_t n = 0;
  std::uint32_t first_problem = 0;
  std::uint32_t count = 0;
};

void FreeDevicePackedProblems(DevicePackedProblems* device) {
  FreeDevice(device->a);
  FreeDevice(device->b);
  FreeDevice(device->c);
  FreeDevice(device->a_offsets);
  FreeDevice(device->b_offsets);
  FreeDevice(device->c_offsets);
  FreeDevice(device->ms);
  FreeDevice(device->ks);
  FreeDevice(device->ns);
  FreeDevice(device->epilogue_scales);
  *device = {};
}

void FreeDevicePackedMixedProblems(DevicePackedMixedProblems* device) {
  FreeDevice(device->a);
  FreeDevice(device->b);
  FreeDevice(device->c);
  FreeDevice(device->a_offsets);
  FreeDevice(device->b_offsets);
  FreeDevice(device->c_offsets);
  FreeDevice(device->ms);
  FreeDevice(device->ks);
  FreeDevice(device->ns);
  FreeDevice(device->epilogue_scales);
  *device = {};
}

void FreeDeviceMixedShapeBuckets(
    std::vector<DeviceMixedShapeBucket>* buckets) {
  for (DeviceMixedShapeBucket& bucket : *buckets) {
    FreeDevice(bucket.a);
    FreeDevice(bucket.b);
    FreeDevice(bucket.c);
    bucket = {};
  }
  buckets->clear();
}

Status CopyPackedToDevice(const PackedProblems& packed,
                          DevicePackedProblems* device) {
  Status status = CopyToDevice(packed.a_all, &device->a);
  if (!status.ok()) return status;
  status = CopyToDevice(packed.b_all, &device->b);
  if (!status.ok()) return status;
  status = CopyToDevice(packed.c_all, &device->c);
  if (!status.ok()) return status;
  status = CopyToDevice(packed.a_offsets, &device->a_offsets);
  if (!status.ok()) return status;
  status = CopyToDevice(packed.b_offsets, &device->b_offsets);
  if (!status.ok()) return status;
  status = CopyToDevice(packed.c_offsets, &device->c_offsets);
  if (!status.ok()) return status;
  status = CopyToDevice(packed.ms, &device->ms);
  if (!status.ok()) return status;
  status = CopyToDevice(packed.ks, &device->ks);
  if (!status.ok()) return status;
  status = CopyToDevice(packed.ns, &device->ns);
  if (!status.ok()) return status;
  status = CopyToDevice(packed.epilogue_scales, &device->epilogue_scales);
  return status;
}

Status CopyMixedPackedToDevice(const PackedMixedProblems& packed,
                               DevicePackedMixedProblems* device) {
  Status status = CopyToDevice(packed.a_all, &device->a);
  if (!status.ok()) return status;
  status = CopyToDevice(packed.b_all, &device->b);
  if (!status.ok()) return status;
  status = CopyToDevice(packed.c_all, &device->c);
  if (!status.ok()) return status;
  status = CopyToDevice(packed.a_offsets, &device->a_offsets);
  if (!status.ok()) return status;
  status = CopyToDevice(packed.b_offsets, &device->b_offsets);
  if (!status.ok()) return status;
  status = CopyToDevice(packed.c_offsets, &device->c_offsets);
  if (!status.ok()) return status;
  status = CopyToDevice(packed.ms, &device->ms);
  if (!status.ok()) return status;
  status = CopyToDevice(packed.ks, &device->ks);
  if (!status.ok()) return status;
  status = CopyToDevice(packed.ns, &device->ns);
  if (!status.ok()) return status;
  status = CopyToDevice(packed.epilogue_scales, &device->epilogue_scales);
  return status;
}

Status CopyMixedShapeBucketsToDevice(
    const PackedMixedProblems& packed,
    std::vector<DeviceMixedShapeBucket>* device_buckets) {
  device_buckets->clear();
  device_buckets->reserve(packed.buckets.size());
  for (const MixedShapeBucket& bucket : packed.buckets) {
    DeviceMixedShapeBucket device_bucket;
    device_bucket.m = bucket.m;
    device_bucket.k = bucket.k;
    device_bucket.n = bucket.n;
    device_bucket.first_problem = bucket.first_problem;
    device_bucket.count = bucket.count;

    const std::size_t a_count =
        static_cast<std::size_t>(bucket.count) * bucket.m * bucket.k;
    const std::size_t b_count =
        static_cast<std::size_t>(bucket.count) * bucket.k * bucket.n;
    const std::size_t c_count =
        static_cast<std::size_t>(bucket.count) * bucket.m * bucket.n;
    Status status =
        CopySpanToDevice(packed.a_all.data() + bucket.a_offset, a_count,
                         &device_bucket.a);
    if (!status.ok()) {
      FreeDeviceMixedShapeBuckets(device_buckets);
      return status;
    }
    status = CopySpanToDevice(packed.b_all.data() + bucket.b_offset, b_count,
                              &device_bucket.b);
    if (!status.ok()) {
      FreeDevice(device_bucket.a);
      FreeDeviceMixedShapeBuckets(device_buckets);
      return status;
    }
    cudaError_t error =
        cudaMalloc(reinterpret_cast<void**>(&device_bucket.c),
                   c_count * sizeof(float));
    if (error != cudaSuccess) {
      FreeDevice(device_bucket.a);
      FreeDevice(device_bucket.b);
      FreeDeviceMixedShapeBuckets(device_buckets);
      return CudaStatus(error, "cudaMalloc mixed bucket C");
    }
    error = cudaMemset(device_bucket.c, 0, c_count * sizeof(float));
    if (error != cudaSuccess) {
      FreeDevice(device_bucket.a);
      FreeDevice(device_bucket.b);
      FreeDevice(device_bucket.c);
      FreeDeviceMixedShapeBuckets(device_buckets);
      return CudaStatus(error, "cudaMemset mixed bucket C");
    }
    device_buckets->push_back(device_bucket);
  }
  return Status::Ok();
}

std::vector<std::vector<double>> SplitOutputTiles(
    const std::vector<CudaGemmProblem>& problems,
    const std::vector<double>& c_all,
    const std::vector<std::uint64_t>& c_offsets) {
  std::vector<std::vector<double>> c_tiles;
  c_tiles.reserve(problems.size());
  for (std::size_t i = 0; i < problems.size(); ++i) {
    const std::size_t begin = static_cast<std::size_t>(c_offsets[i]);
    const std::size_t count =
        static_cast<std::size_t>(problems[i].m) * problems[i].n;
    c_tiles.emplace_back(c_all.begin() + begin,
                         c_all.begin() + begin + count);
  }
  return c_tiles;
}

std::vector<std::vector<double>> SplitMixedOutputTilesFromPacked(
    const std::vector<std::uint32_t>& ms,
    const std::vector<std::uint32_t>& ns,
    const std::vector<double>& c_all,
    const std::vector<std::uint64_t>& c_offsets,
    const std::vector<std::size_t>& original_indices) {
  std::vector<std::vector<double>> c_tiles(ms.size());
  for (std::size_t i = 0; i < ms.size(); ++i) {
    const std::size_t begin = static_cast<std::size_t>(c_offsets[i]);
    const std::size_t count = static_cast<std::size_t>(ms[i]) * ns[i];
    c_tiles[original_indices[i]] =
        std::vector<double>(c_all.begin() + begin, c_all.begin() + begin + count);
  }
  return c_tiles;
}

OperationLedgerEntry CudaGemmLedgerEntry(std::uint64_t products,
                                         bool uses_nontrivial_scale,
                                         const char* note) {
  PrecisionDescriptor precision{
      NumericFormat::kFloat64, NumericFormat::kFloat64,
      NumericFormat::kFloat64, DeterminismMode::kDeterministic,
      uses_nontrivial_scale, true, "cuda-fp64-reference"};
  if (uses_nontrivial_scale) {
    precision.label = "cuda-fp64-scale-epilogue-reference";
  }
  return OperationLedgerEntry{
      OperationKind::kGemm,
      precision,
      ErrorBudget{ErrorMetric::kAbsolute, 0.0,
                  "CUDA FP64 grouped GEMM reference"},
      0.0,
      products,
      products,
      0,
      note};
}

OperationLedgerEntry CudaMixedGemmLedgerEntry(std::uint64_t products,
                                              double max_abs_error_bound) {
  return OperationLedgerEntry{
      OperationKind::kGemm,
      PrecisionDescriptor{NumericFormat::kFloat16, NumericFormat::kFloat32,
                          NumericFormat::kFloat32,
                          DeterminismMode::kDeterministic, true, true,
                          "cuda-fp16-store-fp32-accum-reference"},
      ErrorBudget{ErrorMetric::kAbsolute, max_abs_error_bound,
                  "FP16 input quantization plus FP32 accumulation bound"},
      max_abs_error_bound,
      products,
      products,
      0,
      "CUDA FP16-store FP32-accumulate grouped tile GEMM reference kernel"};
}

Status CreateCublasLtHandle(cublasLtHandle_t* handle, cudaStream_t stream) {
  (void)stream;
  cublasStatus_t status = cublasLtCreate(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasStatus(status, "cublasLtCreate");
  }
  return Status::Ok();
}

struct CublasLtMatmulPref {
  cublasLtMatmulPreference_t pref = nullptr;
  ~CublasLtMatmulPref() {
    if (pref != nullptr) cublasLtMatmulPreferenceDestroy(pref);
  }
};

struct CublasLtMatmulDescHolder {
  cublasLtMatmulDesc_t desc = nullptr;
  ~CublasLtMatmulDescHolder() {
    if (desc != nullptr) cublasLtMatmulDescDestroy(desc);
  }
};

struct CublasLtLayoutHolder {
  cublasLtMatrixLayout_t layout = nullptr;
  ~CublasLtLayoutHolder() {
    if (layout != nullptr) cublasLtMatrixLayoutDestroy(layout);
  }
};

Status SetLtBatchAttr(cublasLtMatrixLayout_t layout, int batch_count,
                      long long stride) {
  cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
      sizeof(batch_count));
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasStatus(status, "cublasLtMatrixLayoutSetAttribute batch_count");
  }
  status = cublasLtMatrixLayoutSetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
      sizeof(stride));
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasStatus(status, "cublasLtMatrixLayoutSetAttribute stride");
  }
  return Status::Ok();
}

Status RunMixedCublasLtBuckets(
    cublasLtHandle_t handle,
    const std::vector<DeviceMixedShapeBucket>& device_buckets) {
  const float alpha = 1.0F;
  const float beta = 0.0F;
  const cublasOperation_t no_transpose = CUBLAS_OP_N;

  CublasLtMatmulDescHolder op_holder;
  cublasStatus_t status = cublasLtMatmulDescCreate(&op_holder.desc,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasStatus(status, "cublasLtMatmulDescCreate");
  }
  status = cublasLtMatmulDescSetAttribute(
      op_holder.desc, CUBLASLT_MATMUL_DESC_TRANSA, &no_transpose,
      sizeof(no_transpose));
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulDescSetAttribute(
        op_holder.desc, CUBLASLT_MATMUL_DESC_TRANSB, &no_transpose,
        sizeof(no_transpose));
  }
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasStatus(status, "cublasLtMatmulDescSetAttribute transpose");
  }

  for (const DeviceMixedShapeBucket& bucket : device_buckets) {
    const int m = static_cast<int>(bucket.n);
    const int n = static_cast<int>(bucket.m);
    const int k = static_cast<int>(bucket.k);
    const long long a_stride =
        static_cast<long long>(bucket.m) * bucket.k;
    const long long b_stride =
        static_cast<long long>(bucket.k) * bucket.n;
    const long long c_stride =
        static_cast<long long>(bucket.m) * bucket.n;
    const int batch_count = static_cast<int>(bucket.count);

    CublasLtLayoutHolder a_desc, b_desc, c_desc;
    status = cublasLtMatrixLayoutCreate(&a_desc.layout, CUDA_R_16F, m, k, m);
    if (status == CUBLAS_STATUS_SUCCESS) {
      status = cublasLtMatrixLayoutCreate(&b_desc.layout, CUDA_R_16F, k, n, k);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
      status = cublasLtMatrixLayoutCreate(&c_desc.layout, CUDA_R_32F, m, n, m);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
      return CublasStatus(status, "cublasLtMatrixLayoutCreate");
    }

    Status attr_status = SetLtBatchAttr(a_desc.layout, batch_count, b_stride);
    if (!attr_status.ok()) return attr_status;
    attr_status = SetLtBatchAttr(b_desc.layout, batch_count, a_stride);
    if (!attr_status.ok()) return attr_status;
    attr_status = SetLtBatchAttr(c_desc.layout, batch_count, c_stride);
    if (!attr_status.ok()) return attr_status;

    status = cublasLtMatmul(handle, op_holder.desc, &alpha, bucket.b,
                            a_desc.layout, bucket.a, b_desc.layout, &beta,
                            bucket.c, c_desc.layout, bucket.c, c_desc.layout,
                            nullptr, nullptr, 0, nullptr);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return CublasStatus(status, "cublasLtMatmul");
    }
  }

  return Status::Ok();
}

Status LaunchMixedScaleKernel(const PackedMixedProblems& packed,
                              const DevicePackedMixedProblems& device,
                              const std::vector<DeviceMixedShapeBucket>&
                                  device_buckets,
                              cudaStream_t stream) {
  (void)packed;
  const dim3 block(kBlockEdge, kBlockEdge);
  for (const DeviceMixedShapeBucket& bucket : device_buckets) {
    const dim3 grid((bucket.n + kBlockEdge - 1) / kBlockEdge,
                    (bucket.m + kBlockEdge - 1) / kBlockEdge,
                    static_cast<unsigned int>(bucket.count));
    ScaleBucketFloatGemmOutputKernel<<<grid, block, 0, stream>>>(
        bucket.c, device.c, bucket.first_problem, device.c_offsets,
        device.epilogue_scales, bucket.m, bucket.n);
    const Status status =
        CudaStatus(cudaGetLastError(), "ScaleBucketFloatGemmOutputKernel launch");
    if (!status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

Status TimedRunMixedCublasLtAndScale(cublasLtHandle_t handle,
                                   const PackedMixedProblems& packed,
                                   const DevicePackedMixedProblems& device,
                                   const std::vector<DeviceMixedShapeBucket>&
                                       device_buckets,
                                   cudaStream_t stream, double* gemm_ms,
                                   double* scale_ms) {
  cudaEvent_t start = nullptr;
  cudaEvent_t after_gemm = nullptr;
  cudaEvent_t stop = nullptr;
  cudaError_t error = cudaEventCreate(&start);
  if (error != cudaSuccess) {
    return CudaStatus(error, "cudaEventCreate mixed start");
  }
  error = cudaEventCreate(&after_gemm);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    return CudaStatus(error, "cudaEventCreate mixed after_gemm");
  }
  error = cudaEventCreate(&stop);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(after_gemm);
    return CudaStatus(error, "cudaEventCreate mixed stop");
  }
  cudaEventRecord(start, stream);
  Status status = RunMixedCublasLtBuckets(handle, device_buckets);
  if (!status.ok()) {
    cudaEventDestroy(start);
    cudaEventDestroy(after_gemm);
    cudaEventDestroy(stop);
    return status;
  }
  cudaEventRecord(after_gemm, stream);
  status = LaunchMixedScaleKernel(packed, device, device_buckets, stream);
  if (!status.ok()) {
    cudaEventDestroy(start);
    cudaEventDestroy(after_gemm);
    cudaEventDestroy(stop);
    return status;
  }
  cudaEventRecord(stop, stream);
  error = cudaEventSynchronize(stop);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(after_gemm);
    cudaEventDestroy(stop);
    return CudaStatus(error, "mixed cuBLAS grouped GEMM synchronize");
  }
  float gemm_elapsed_ms = 0.0F;
  float scale_elapsed_ms = 0.0F;
  cudaEventElapsedTime(&gemm_elapsed_ms, start, after_gemm);
  cudaEventElapsedTime(&scale_elapsed_ms, after_gemm, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(after_gemm);
  cudaEventDestroy(stop);
  *gemm_ms = static_cast<double>(gemm_elapsed_ms);
  *scale_ms = static_cast<double>(scale_elapsed_ms);
  return Status::Ok();
}

}  // namespace

struct CudaGroupedGemmFp16AccumPlan::Impl {
  PackedMixedProblems packed;
  DevicePackedMixedProblems device;
  std::vector<DeviceMixedShapeBucket> cublas_buckets;
  cublasLtHandle_t handle = nullptr;
};

CudaGroupedGemmFp16AccumPlan::CudaGroupedGemmFp16AccumPlan() = default;

CudaGroupedGemmFp16AccumPlan::CudaGroupedGemmFp16AccumPlan(Impl* impl)
    : impl_(impl) {}

CudaGroupedGemmFp16AccumPlan::~CudaGroupedGemmFp16AccumPlan() {
  if (impl_ != nullptr) {
    if (impl_->handle != nullptr) {
      cublasLtDestroy(impl_->handle);
    }
    FreeDeviceMixedShapeBuckets(&impl_->cublas_buckets);
    FreeDevicePackedMixedProblems(&impl_->device);
    delete impl_;
    impl_ = nullptr;
  }
}

CudaGroupedGemmFp16AccumPlan::CudaGroupedGemmFp16AccumPlan(
    CudaGroupedGemmFp16AccumPlan&& other) noexcept
    : impl_(other.impl_) {
  other.impl_ = nullptr;
}

CudaGroupedGemmFp16AccumPlan& CudaGroupedGemmFp16AccumPlan::operator=(
    CudaGroupedGemmFp16AccumPlan&& other) noexcept {
  if (this != &other) {
    if (impl_ != nullptr) {
      if (impl_->handle != nullptr) {
        cublasLtDestroy(impl_->handle);
      }
      FreeDeviceMixedShapeBuckets(&impl_->cublas_buckets);
      FreeDevicePackedMixedProblems(&impl_->device);
      delete impl_;
    }
    impl_ = other.impl_;
    other.impl_ = nullptr;
  }
  return *this;
}

bool CudaGroupedGemmFp16AccumPlan::valid() const {
  return impl_ != nullptr;
}

bool CudaRuntimeAvailable() {
  return CudaRuntimeStatus().ok();
}

Status CudaRuntimeStatus() {
  cudaFree(nullptr);
  cudaGetLastError();
  cudaError_t last_error = cudaSuccess;
  for (int attempt = 0; attempt < 5; ++attempt) {
    int count = 0;
    const cudaError_t error = cudaGetDeviceCount(&count);
    if (error == cudaSuccess) {
      if (count == 0) {
        return Status::IoError("cudaGetDeviceCount returned zero devices");
      }
      return Status::Ok();
    }
    last_error = error;
    cudaGetLastError();
    if (attempt + 1 < 5) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
  return CudaStatus(last_error, "cudaGetDeviceCount");
}

Result<CudaGroupedGemmResult> GroupedGemmFp64Cuda(
    const std::vector<CudaGemmProblem>& problems) {
  const Status runtime_status = CudaRuntimeStatus();
  if (!runtime_status.ok()) {
    return runtime_status;
  }
  cudaError_t error = cudaSuccess;

  auto packed_result = PackProblems(problems);
  if (!packed_result.ok()) {
    return packed_result.status();
  }
  PackedProblems packed = packed_result.take_value();

  DevicePackedProblems device;
  Status status = CopyPackedToDevice(packed, &device);
  if (!status.ok()) {
    FreeDevicePackedProblems(&device);
    return status;
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error != cudaSuccess) {
    return CudaStatus(error, "cudaEventCreate start");
  }
  error = cudaEventCreate(&stop);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    return CudaStatus(error, "cudaEventCreate stop");
  }

  const dim3 block(kBlockEdge, kBlockEdge);
  const dim3 grid((packed.max_n + kBlockEdge - 1) / kBlockEdge,
                  (packed.max_m + kBlockEdge - 1) / kBlockEdge,
                  static_cast<unsigned int>(problems.size()));
  const std::size_t gemm_smem =
      static_cast<std::size_t>(kBlockEdge) * kGemmTileK * 2 * sizeof(double);

  cudaEventRecord(start);
  GroupedGemmKernel<<<grid, block, gemm_smem>>>(device.a, device.b, device.c,
                                     device.a_offsets, device.b_offsets,
                                     device.c_offsets, device.ms, device.ks,
                                     device.ns, device.epilogue_scales);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return CudaStatus(error, "GroupedGemmKernel launch");
  }
  cudaEventRecord(stop);
  error = cudaEventSynchronize(stop);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return CudaStatus(error, "GroupedGemmKernel synchronize");
  }
  float kernel_ms = 0.0F;
  cudaEventElapsedTime(&kernel_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  error = cudaMemcpy(packed.c_all.data(), device.c,
                     packed.c_all.size() * sizeof(double),
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    FreeDevicePackedProblems(&device);
    return CudaStatus(error, "cudaMemcpy D2H");
  }

  FreeDevicePackedProblems(&device);

  CudaGroupedGemmResult result;
  result.kernel_ms = static_cast<double>(kernel_ms);
  result.c_tiles = SplitOutputTiles(problems, packed.c_all, packed.c_offsets);
  result.ledger.Add(CudaGemmLedgerEntry(
      packed.products, packed.uses_nontrivial_scale,
      "CUDA FP64 grouped tile GEMM reference kernel"));
  return result;
}

Result<CudaGroupedGemmResult> GroupedGemmFp16AccumCuda(
    const std::vector<CudaGemmProblem>& problems) {
  const Status runtime_status = CudaRuntimeStatus();
  if (!runtime_status.ok()) {
    return runtime_status;
  }
  cudaError_t error = cudaSuccess;

  auto packed_result = PackMixedProblems(problems);
  if (!packed_result.ok()) {
    return packed_result.status();
  }
  PackedMixedProblems packed = packed_result.take_value();

  DevicePackedMixedProblems device;
  Status status = CopyMixedPackedToDevice(packed, &device);
  if (!status.ok()) {
    FreeDevicePackedMixedProblems(&device);
    return status;
  }
  std::vector<DeviceMixedShapeBucket> cublas_buckets;
  status = CopyMixedShapeBucketsToDevice(packed, &cublas_buckets);
  if (!status.ok()) {
    FreeDevicePackedMixedProblems(&device);
    return status;
  }

  cublasLtHandle_t handle = nullptr;
  status = CreateCublasLtHandle(&handle, nullptr);
  if (!status.ok()) {
    FreeDeviceMixedShapeBuckets(&cublas_buckets);
    FreeDevicePackedMixedProblems(&device);
    return status;
  }

  double kernel_ms = 0.0;
  double postprocess_ms = 0.0;
  status = TimedRunMixedCublasLtAndScale(handle, packed, device, cublas_buckets,
                                       nullptr, &kernel_ms, &postprocess_ms);
  if (!status.ok()) {
    cublasLtDestroy(handle);
    FreeDeviceMixedShapeBuckets(&cublas_buckets);
    FreeDevicePackedMixedProblems(&device);
    return status;
  }

  error = cudaMemcpy(packed.c_all.data(), device.c,
                     packed.c_all.size() * sizeof(double),
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    cublasLtDestroy(handle);
    FreeDeviceMixedShapeBuckets(&cublas_buckets);
    FreeDevicePackedMixedProblems(&device);
    return CudaStatus(error, "cudaMemcpy mixed D2H");
  }

  cublasLtDestroy(handle);
  FreeDeviceMixedShapeBuckets(&cublas_buckets);
  FreeDevicePackedMixedProblems(&device);

  CudaGroupedGemmResult result;
  result.kernel_ms = kernel_ms;
  result.postprocess_ms = postprocess_ms;
  result.backend_shape_buckets = packed.buckets.size();
  result.c_tiles = SplitMixedOutputTilesFromPacked(
      packed.ms, packed.ns, packed.c_all, packed.c_offsets,
      packed.original_indices);
  result.ledger.Add(
      CudaMixedGemmLedgerEntry(packed.products, packed.max_abs_error_bound));
  return result;
}

Result<CudaGroupedGemmFp16AccumPlan> BuildGroupedGemmFp16AccumCudaPlan(
    const std::vector<CudaGemmProblem>& problems) {
  const Status runtime_status = CudaRuntimeStatus();
  if (!runtime_status.ok()) {
    return runtime_status;
  }

  auto packed_result = PackMixedProblems(problems);
  if (!packed_result.ok()) {
    return packed_result.status();
  }

  auto* impl = new CudaGroupedGemmFp16AccumPlan::Impl;
  impl->packed = packed_result.take_value();
  const Status status = CopyMixedPackedToDevice(impl->packed, &impl->device);
  if (!status.ok()) {
    FreeDevicePackedMixedProblems(&impl->device);
    delete impl;
    return status;
  }
  const Status bucket_status =
      CopyMixedShapeBucketsToDevice(impl->packed, &impl->cublas_buckets);
  if (!bucket_status.ok()) {
    FreeDevicePackedMixedProblems(&impl->device);
    delete impl;
    return bucket_status;
  }
  const Status handle_status = CreateCublasLtHandle(&impl->handle, nullptr);
  if (!handle_status.ok()) {
    FreeDeviceMixedShapeBuckets(&impl->cublas_buckets);
    FreeDevicePackedMixedProblems(&impl->device);
    delete impl;
    return handle_status;
  }
  return CudaGroupedGemmFp16AccumPlan(impl);
}

Result<CudaGroupedGemmResult> RunGroupedGemmFp16AccumCudaPlan(
    const CudaGroupedGemmFp16AccumPlan& plan) {
  if (plan.impl_ == nullptr) {
    return Status::InvalidArgument("mixed grouped GEMM CUDA plan is empty");
  }

  PackedMixedProblems& packed = plan.impl_->packed;
  DevicePackedMixedProblems& device = plan.impl_->device;
  cublasLtHandle_t handle = plan.impl_->handle;
  double kernel_ms = 0.0;
  double postprocess_ms = 0.0;
  Status status = TimedRunMixedCublasLtAndScale(
      handle, packed, device, plan.impl_->cublas_buckets, nullptr, &kernel_ms,
      &postprocess_ms);
  if (!status.ok()) {
    return status;
  }

  cudaError_t error = cudaMemcpy(packed.c_all.data(), device.c,
                                 packed.c_all.size() * sizeof(double),
                                 cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    return CudaStatus(error, "cudaMemcpy planned mixed D2H");
  }

  CudaGroupedGemmResult result;
  result.kernel_ms = kernel_ms;
  result.postprocess_ms = postprocess_ms;
  result.backend_shape_buckets = packed.buckets.size();
  result.c_tiles = SplitMixedOutputTilesFromPacked(
      packed.ms, packed.ns, packed.c_all, packed.c_offsets,
      packed.original_indices);
  result.ledger.Add(
      CudaMixedGemmLedgerEntry(packed.products, packed.max_abs_error_bound));
  return result;
}

Result<CudaGraphReplayResult> GroupedGemmFp64CudaGraphReplay(
    const std::vector<CudaGemmProblem>& problems, std::uint32_t repeats) {
  if (repeats == 0) {
    return Status::InvalidArgument("graph replay repeats must be nonzero");
  }
  const Status runtime_status = CudaRuntimeStatus();
  if (!runtime_status.ok()) {
    return runtime_status;
  }

  auto packed_result = PackProblems(problems);
  if (!packed_result.ok()) {
    return packed_result.status();
  }
  PackedProblems packed = packed_result.take_value();
  DevicePackedProblems device;
  Status status = CopyPackedToDevice(packed, &device);
  if (!status.ok()) {
    FreeDevicePackedProblems(&device);
    return status;
  }

  cudaStream_t stream = nullptr;
  cudaError_t error = cudaStreamCreate(&stream);
  if (error != cudaSuccess) {
    FreeDevicePackedProblems(&device);
    return CudaStatus(error, "cudaStreamCreate");
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error != cudaSuccess) {
    cudaStreamDestroy(stream);
    FreeDevicePackedProblems(&device);
    return CudaStatus(error, "cudaEventCreate start");
  }
  error = cudaEventCreate(&stop);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    FreeDevicePackedProblems(&device);
    return CudaStatus(error, "cudaEventCreate stop");
  }

  const dim3 block(kBlockEdge, kBlockEdge);
  const dim3 grid((packed.max_n + kBlockEdge - 1) / kBlockEdge,
                  (packed.max_m + kBlockEdge - 1) / kBlockEdge,
                  static_cast<unsigned int>(problems.size()));

  const auto raw_wall_start = std::chrono::steady_clock::now();
  cudaEventRecord(start, stream);
  const std::size_t graph_gemm_smem =
      static_cast<std::size_t>(kBlockEdge) * kGemmTileK * 2 * sizeof(double);
  for (std::uint32_t i = 0; i < repeats; ++i) {
    GroupedGemmKernel<<<grid, block, graph_gemm_smem, stream>>>(
        device.a, device.b, device.c, device.a_offsets, device.b_offsets,
        device.c_offsets, device.ms, device.ks, device.ns,
        device.epilogue_scales);
  }
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    FreeDevicePackedProblems(&device);
    return CudaStatus(error, "raw repeated GroupedGemmKernel launch");
  }
  cudaEventRecord(stop, stream);
  error = cudaEventSynchronize(stop);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    FreeDevicePackedProblems(&device);
    return CudaStatus(error, "raw repeated GroupedGemmKernel sync");
  }
  float raw_ms = 0.0F;
  cudaEventElapsedTime(&raw_ms, start, stop);
  const auto raw_wall_end = std::chrono::steady_clock::now();

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  const auto graph_setup_start = std::chrono::steady_clock::now();
  error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    FreeDevicePackedProblems(&device);
    return CudaStatus(error, "cudaStreamBeginCapture");
  }
  for (std::uint32_t i = 0; i < repeats; ++i) {
    GroupedGemmKernel<<<grid, block, graph_gemm_smem, stream>>>(
        device.a, device.b, device.c, device.a_offsets, device.b_offsets,
        device.c_offsets, device.ms, device.ks, device.ns,
        device.epilogue_scales);
  }
  error = cudaStreamEndCapture(stream, &graph);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    FreeDevicePackedProblems(&device);
    return CudaStatus(error, "cudaStreamEndCapture");
  }
  error = cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
  if (error != cudaSuccess) {
    cudaGraphDestroy(graph);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    FreeDevicePackedProblems(&device);
    return CudaStatus(error, "cudaGraphInstantiate");
  }
  const auto graph_setup_end = std::chrono::steady_clock::now();

  const auto graph_wall_start = std::chrono::steady_clock::now();
  cudaEventRecord(start, stream);
  error = cudaGraphLaunch(graph_exec, stream);
  if (error != cudaSuccess) {
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    FreeDevicePackedProblems(&device);
    return CudaStatus(error, "cudaGraphLaunch");
  }
  cudaEventRecord(stop, stream);
  error = cudaEventSynchronize(stop);
  if (error != cudaSuccess) {
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    FreeDevicePackedProblems(&device);
    return CudaStatus(error, "cudaGraph replay sync");
  }
  float graph_ms = 0.0F;
  cudaEventElapsedTime(&graph_ms, start, stop);
  const auto graph_wall_end = std::chrono::steady_clock::now();

  error = cudaMemcpy(packed.c_all.data(), device.c,
                     packed.c_all.size() * sizeof(double),
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    FreeDevicePackedProblems(&device);
    return CudaStatus(error, "cudaMemcpy D2H graph replay");
  }

  cudaGraphExecDestroy(graph_exec);
  cudaGraphDestroy(graph);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  FreeDevicePackedProblems(&device);

  CudaGraphReplayResult result;
  result.repeats = repeats;
  result.raw_repeated_kernel_ms = static_cast<double>(raw_ms);
  result.raw_repeated_wall_ms =
      std::chrono::duration<double, std::milli>(raw_wall_end - raw_wall_start)
          .count();
  result.graph_setup_ms =
      std::chrono::duration<double, std::milli>(graph_setup_end -
                                                graph_setup_start)
          .count();
  result.graph_replay_ms = static_cast<double>(graph_ms);
  result.graph_replay_wall_ms =
      std::chrono::duration<double, std::milli>(graph_wall_end -
                                                graph_wall_start)
          .count();
  result.c_tiles = SplitOutputTiles(problems, packed.c_all, packed.c_offsets);
  result.ledger.Add(CudaGemmLedgerEntry(
      packed.products * repeats, packed.uses_nontrivial_scale,
      "CUDA graph replay FP64 grouped GEMM kernel"));
  return result;
}

Result<CudaGraphReplayResult> GroupedGemmFp16AccumCudaGraphReplay(
    const std::vector<CudaGemmProblem>& problems, std::uint32_t repeats) {
  if (repeats == 0) {
    return Status::InvalidArgument(
        "mixed graph replay repeats must be nonzero");
  }
  const Status runtime_status = CudaRuntimeStatus();
  if (!runtime_status.ok()) {
    return runtime_status;
  }

  auto packed_result = PackMixedProblems(problems);
  if (!packed_result.ok()) {
    return packed_result.status();
  }
  PackedMixedProblems packed = packed_result.take_value();
  DevicePackedMixedProblems device;
  Status status = CopyMixedPackedToDevice(packed, &device);
  if (!status.ok()) {
    FreeDevicePackedMixedProblems(&device);
    return status;
  }

  cudaStream_t stream = nullptr;
  cudaError_t error = cudaStreamCreate(&stream);
  if (error != cudaSuccess) {
    FreeDevicePackedMixedProblems(&device);
    return CudaStatus(error, "cudaStreamCreate mixed graph");
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error != cudaSuccess) {
    cudaStreamDestroy(stream);
    FreeDevicePackedMixedProblems(&device);
    return CudaStatus(error, "cudaEventCreate mixed graph start");
  }
  error = cudaEventCreate(&stop);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    FreeDevicePackedMixedProblems(&device);
    return CudaStatus(error, "cudaEventCreate mixed graph stop");
  }

  const dim3 wmma_block(128);
  const dim3 wmma_grid(
      (packed.max_n + kWmmaTileX * kWmmaN - 1) / (kWmmaTileX * kWmmaN),
      (packed.max_m + kWmmaTileY * kWmmaM - 1) / (kWmmaTileY * kWmmaM),
      static_cast<unsigned int>(problems.size()));

  const auto raw_wall_start = std::chrono::steady_clock::now();
  cudaEventRecord(start, stream);
  for (std::uint32_t i = 0; i < repeats; ++i) {
    GroupedGemmFp16AccumWmmaKernel<<<wmma_grid, wmma_block, 0, stream>>>(
        device.a, device.b, device.c, device.a_offsets, device.b_offsets,
        device.c_offsets, device.ms, device.ks, device.ns,
        device.epilogue_scales);
  }
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    FreeDevicePackedMixedProblems(&device);
    return CudaStatus(error,
                      "raw repeated GroupedGemmFp16AccumWmmaKernel launch");
  }
  cudaEventRecord(stop, stream);
  error = cudaEventSynchronize(stop);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    FreeDevicePackedMixedProblems(&device);
    return CudaStatus(error, "raw repeated mixed GroupedGemmKernel sync");
  }
  float raw_ms = 0.0F;
  cudaEventElapsedTime(&raw_ms, start, stop);
  const auto raw_wall_end = std::chrono::steady_clock::now();

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  const auto graph_setup_start = std::chrono::steady_clock::now();
  error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    FreeDevicePackedMixedProblems(&device);
    return CudaStatus(error, "cudaStreamBeginCapture mixed");
  }
  for (std::uint32_t i = 0; i < repeats; ++i) {
    GroupedGemmFp16AccumWmmaKernel<<<wmma_grid, wmma_block, 0, stream>>>(
        device.a, device.b, device.c, device.a_offsets, device.b_offsets,
        device.c_offsets, device.ms, device.ks, device.ns,
        device.epilogue_scales);
  }
  error = cudaStreamEndCapture(stream, &graph);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    FreeDevicePackedMixedProblems(&device);
    return CudaStatus(error, "cudaStreamEndCapture mixed");
  }
  error = cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
  if (error != cudaSuccess) {
    cudaGraphDestroy(graph);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    FreeDevicePackedMixedProblems(&device);
    return CudaStatus(error, "cudaGraphInstantiate mixed");
  }
  const auto graph_setup_end = std::chrono::steady_clock::now();

  const auto graph_wall_start = std::chrono::steady_clock::now();
  cudaEventRecord(start, stream);
  error = cudaGraphLaunch(graph_exec, stream);
  if (error != cudaSuccess) {
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    FreeDevicePackedMixedProblems(&device);
    return CudaStatus(error, "cudaGraphLaunch mixed");
  }
  cudaEventRecord(stop, stream);
  error = cudaEventSynchronize(stop);
  if (error != cudaSuccess) {
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    FreeDevicePackedMixedProblems(&device);
    return CudaStatus(error, "cudaGraph mixed replay sync");
  }
  float graph_ms = 0.0F;
  cudaEventElapsedTime(&graph_ms, start, stop);
  const auto graph_wall_end = std::chrono::steady_clock::now();

  error = cudaMemcpy(packed.c_all.data(), device.c,
                     packed.c_all.size() * sizeof(double),
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    FreeDevicePackedMixedProblems(&device);
    return CudaStatus(error, "cudaMemcpy D2H mixed graph replay");
  }

  cudaGraphExecDestroy(graph_exec);
  cudaGraphDestroy(graph);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  FreeDevicePackedMixedProblems(&device);

  CudaGraphReplayResult result;
  result.repeats = repeats;
  result.raw_repeated_kernel_ms = static_cast<double>(raw_ms);
  result.raw_repeated_wall_ms =
      std::chrono::duration<double, std::milli>(raw_wall_end - raw_wall_start)
          .count();
  result.graph_setup_ms =
      std::chrono::duration<double, std::milli>(graph_setup_end -
                                                graph_setup_start)
          .count();
  result.graph_replay_ms = static_cast<double>(graph_ms);
  result.graph_replay_wall_ms =
      std::chrono::duration<double, std::milli>(graph_wall_end -
                                                graph_wall_start)
          .count();
  result.c_tiles = SplitMixedOutputTilesFromPacked(
      packed.ms, packed.ns, packed.c_all, packed.c_offsets,
      packed.original_indices);
  result.ledger.Add(CudaMixedGemmLedgerEntry(
      packed.products * repeats, packed.max_abs_error_bound));
  return result;
}

Result<CudaGraphReplayResult> RunGroupedGemmFp16AccumCudaPlanGraphReplay(
    const CudaGroupedGemmFp16AccumPlan& plan, std::uint32_t repeats) {
  if (repeats == 0) {
    return Status::InvalidArgument(
        "planned mixed graph replay repeats must be nonzero");
  }
  if (plan.impl_ == nullptr) {
    return Status::InvalidArgument(
        "planned mixed graph replay CUDA plan is empty");
  }

  PackedMixedProblems& packed = plan.impl_->packed;
  DevicePackedMixedProblems& device = plan.impl_->device;
  cudaStream_t stream = nullptr;
  cudaError_t error = cudaStreamCreate(&stream);
  if (error != cudaSuccess) {
    return CudaStatus(error, "cudaStreamCreate planned mixed graph");
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error != cudaSuccess) {
    cudaStreamDestroy(stream);
    return CudaStatus(error, "cudaEventCreate planned mixed graph start");
  }
  error = cudaEventCreate(&stop);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    return CudaStatus(error, "cudaEventCreate planned mixed graph stop");
  }

  const dim3 wmma_block(128);
  const dim3 wmma_grid(
      (packed.max_n + kWmmaTileX * kWmmaN - 1) / (kWmmaTileX * kWmmaN),
      (packed.max_m + kWmmaTileY * kWmmaM - 1) / (kWmmaTileY * kWmmaM),
      static_cast<unsigned int>(packed.ms.size()));

  const auto raw_wall_start = std::chrono::steady_clock::now();
  cudaEventRecord(start, stream);
  for (std::uint32_t i = 0; i < repeats; ++i) {
    GroupedGemmFp16AccumWmmaKernel<<<wmma_grid, wmma_block, 0, stream>>>(
        device.a, device.b, device.c, device.a_offsets, device.b_offsets,
        device.c_offsets, device.ms, device.ks, device.ns,
        device.epilogue_scales);
  }
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    return CudaStatus(error,
                      "raw repeated planned GroupedGemmFp16AccumWmmaKernel launch");
  }
  cudaEventRecord(stop, stream);
  error = cudaEventSynchronize(stop);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    return CudaStatus(error,
                      "raw repeated planned mixed GroupedGemmKernel sync");
  }
  float raw_ms = 0.0F;
  cudaEventElapsedTime(&raw_ms, start, stop);
  const auto raw_wall_end = std::chrono::steady_clock::now();

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  const auto graph_setup_start = std::chrono::steady_clock::now();
  error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    return CudaStatus(error, "cudaStreamBeginCapture planned mixed");
  }
  for (std::uint32_t i = 0; i < repeats; ++i) {
    GroupedGemmFp16AccumWmmaKernel<<<wmma_grid, wmma_block, 0, stream>>>(
        device.a, device.b, device.c, device.a_offsets, device.b_offsets,
        device.c_offsets, device.ms, device.ks, device.ns,
        device.epilogue_scales);
  }
  error = cudaStreamEndCapture(stream, &graph);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    return CudaStatus(error, "cudaStreamEndCapture planned mixed");
  }
  error = cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
  if (error != cudaSuccess) {
    cudaGraphDestroy(graph);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    return CudaStatus(error, "cudaGraphInstantiate planned mixed");
  }
  const auto graph_setup_end = std::chrono::steady_clock::now();

  const auto graph_wall_start = std::chrono::steady_clock::now();
  cudaEventRecord(start, stream);
  error = cudaGraphLaunch(graph_exec, stream);
  if (error != cudaSuccess) {
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    return CudaStatus(error, "cudaGraphLaunch planned mixed");
  }
  cudaEventRecord(stop, stream);
  error = cudaEventSynchronize(stop);
  if (error != cudaSuccess) {
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    return CudaStatus(error, "cudaGraph planned mixed replay sync");
  }
  float graph_ms = 0.0F;
  cudaEventElapsedTime(&graph_ms, start, stop);
  const auto graph_wall_end = std::chrono::steady_clock::now();

  error = cudaMemcpy(packed.c_all.data(), device.c,
                     packed.c_all.size() * sizeof(double),
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    return CudaStatus(error, "cudaMemcpy D2H planned mixed graph replay");
  }

  cudaGraphExecDestroy(graph_exec);
  cudaGraphDestroy(graph);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);

  CudaGraphReplayResult result;
  result.repeats = repeats;
  result.raw_repeated_kernel_ms = static_cast<double>(raw_ms);
  result.raw_repeated_wall_ms =
      std::chrono::duration<double, std::milli>(raw_wall_end - raw_wall_start)
          .count();
  result.graph_setup_ms =
      std::chrono::duration<double, std::milli>(graph_setup_end -
                                                graph_setup_start)
          .count();
  result.graph_replay_ms = static_cast<double>(graph_ms);
  result.graph_replay_wall_ms =
      std::chrono::duration<double, std::milli>(graph_wall_end -
                                                graph_wall_start)
          .count();
  result.c_tiles = SplitMixedOutputTilesFromPacked(
      packed.ms, packed.ns, packed.c_all, packed.c_offsets,
      packed.original_indices);
  result.ledger.Add(CudaMixedGemmLedgerEntry(
      packed.products * repeats, packed.max_abs_error_bound));
  return result;
}

}  // namespace tides::tile
