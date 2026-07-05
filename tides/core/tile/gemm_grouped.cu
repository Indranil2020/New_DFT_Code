#include "tile/gemm_grouped.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

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
  if (row >= m || col >= n) {
    return;
  }

  const double* a = a_all + a_offsets[problem];
  const double* b = b_all + b_offsets[problem];
  double* c = c_all + c_offsets[problem];
  double sum = 0.0;
  for (std::uint32_t p = 0; p < k; ++p) {
    sum += a[row * k + p] * b[p * n + col];
  }
  c[row * n + col] = epilogue_scales[problem] * sum;
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
  error = cudaMemcpy(*device, host.data(), bytes, cudaMemcpyHostToDevice);
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

  for (std::size_t i = 0; i < problems.size(); ++i) {
    const CudaGemmProblem& problem = problems[i];
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
          "mixed grouped GEMM combined epilogue scale must be finite");
    }
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

std::vector<std::vector<double>> SplitOutputTilesFromPacked(
    const std::vector<std::uint32_t>& ms,
    const std::vector<std::uint32_t>& ns,
    const std::vector<double>& c_all,
    const std::vector<std::uint64_t>& c_offsets) {
  std::vector<std::vector<double>> c_tiles;
  c_tiles.reserve(ms.size());
  for (std::size_t i = 0; i < ms.size(); ++i) {
    const std::size_t begin = static_cast<std::size_t>(c_offsets[i]);
    const std::size_t count = static_cast<std::size_t>(ms[i]) * ns[i];
    c_tiles.emplace_back(c_all.begin() + begin,
                         c_all.begin() + begin + count);
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

}  // namespace

struct CudaGroupedGemmFp16AccumPlan::Impl {
  PackedMixedProblems packed;
  DevicePackedMixedProblems device;
};

CudaGroupedGemmFp16AccumPlan::CudaGroupedGemmFp16AccumPlan() = default;

CudaGroupedGemmFp16AccumPlan::CudaGroupedGemmFp16AccumPlan(Impl* impl)
    : impl_(impl) {}

CudaGroupedGemmFp16AccumPlan::~CudaGroupedGemmFp16AccumPlan() {
  if (impl_ != nullptr) {
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
  int count = 0;
  const cudaError_t error = cudaGetDeviceCount(&count);
  if (error != cudaSuccess) {
    return CudaStatus(error, "cudaGetDeviceCount");
  }
  if (count == 0) {
    return Status::IoError("cudaGetDeviceCount returned zero devices");
  }
  return Status::Ok();
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

  cudaEventRecord(start);
  GroupedGemmKernel<<<grid, block>>>(device.a, device.b, device.c,
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

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error != cudaSuccess) {
    FreeDevicePackedMixedProblems(&device);
    return CudaStatus(error, "cudaEventCreate mixed start");
  }
  error = cudaEventCreate(&stop);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    FreeDevicePackedMixedProblems(&device);
    return CudaStatus(error, "cudaEventCreate mixed stop");
  }

  const dim3 block(kBlockEdge, kBlockEdge);
  const dim3 grid((packed.max_n + kBlockEdge - 1) / kBlockEdge,
                  (packed.max_m + kBlockEdge - 1) / kBlockEdge,
                  static_cast<unsigned int>(problems.size()));

  cudaEventRecord(start);
  GroupedGemmFp16AccumKernel<<<grid, block>>>(
      device.a, device.b, device.c, device.a_offsets, device.b_offsets,
      device.c_offsets, device.ms, device.ks, device.ns,
      device.epilogue_scales);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    FreeDevicePackedMixedProblems(&device);
    return CudaStatus(error, "GroupedGemmFp16AccumKernel launch");
  }
  cudaEventRecord(stop);
  error = cudaEventSynchronize(stop);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    FreeDevicePackedMixedProblems(&device);
    return CudaStatus(error, "GroupedGemmFp16AccumKernel synchronize");
  }
  float kernel_ms = 0.0F;
  cudaEventElapsedTime(&kernel_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  error = cudaMemcpy(packed.c_all.data(), device.c,
                     packed.c_all.size() * sizeof(double),
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    FreeDevicePackedMixedProblems(&device);
    return CudaStatus(error, "cudaMemcpy mixed D2H");
  }

  FreeDevicePackedMixedProblems(&device);

  CudaGroupedGemmResult result;
  result.kernel_ms = static_cast<double>(kernel_ms);
  result.c_tiles = SplitOutputTiles(problems, packed.c_all, packed.c_offsets);
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
  return CudaGroupedGemmFp16AccumPlan(impl);
}

Result<CudaGroupedGemmResult> RunGroupedGemmFp16AccumCudaPlan(
    const CudaGroupedGemmFp16AccumPlan& plan) {
  if (plan.impl_ == nullptr) {
    return Status::InvalidArgument("mixed grouped GEMM CUDA plan is empty");
  }

  PackedMixedProblems& packed = plan.impl_->packed;
  DevicePackedMixedProblems& device = plan.impl_->device;
  cudaError_t error = cudaSuccess;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error != cudaSuccess) {
    return CudaStatus(error, "cudaEventCreate planned mixed start");
  }
  error = cudaEventCreate(&stop);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    return CudaStatus(error, "cudaEventCreate planned mixed stop");
  }

  const dim3 block(kBlockEdge, kBlockEdge);
  const dim3 grid((packed.max_n + kBlockEdge - 1) / kBlockEdge,
                  (packed.max_m + kBlockEdge - 1) / kBlockEdge,
                  static_cast<unsigned int>(packed.ms.size()));

  cudaEventRecord(start);
  GroupedGemmFp16AccumKernel<<<grid, block>>>(
      device.a, device.b, device.c, device.a_offsets, device.b_offsets,
      device.c_offsets, device.ms, device.ks, device.ns,
      device.epilogue_scales);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return CudaStatus(error, "planned GroupedGemmFp16AccumKernel launch");
  }
  cudaEventRecord(stop);
  error = cudaEventSynchronize(stop);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return CudaStatus(error, "planned GroupedGemmFp16AccumKernel synchronize");
  }
  float kernel_ms = 0.0F;
  cudaEventElapsedTime(&kernel_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  error = cudaMemcpy(packed.c_all.data(), device.c,
                     packed.c_all.size() * sizeof(double),
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    return CudaStatus(error, "cudaMemcpy planned mixed D2H");
  }

  CudaGroupedGemmResult result;
  result.kernel_ms = static_cast<double>(kernel_ms);
  result.c_tiles = SplitOutputTilesFromPacked(packed.ms, packed.ns,
                                              packed.c_all, packed.c_offsets);
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
  for (std::uint32_t i = 0; i < repeats; ++i) {
    GroupedGemmKernel<<<grid, block, 0, stream>>>(
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
    GroupedGemmKernel<<<grid, block, 0, stream>>>(
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

  const dim3 block(kBlockEdge, kBlockEdge);
  const dim3 grid((packed.max_n + kBlockEdge - 1) / kBlockEdge,
                  (packed.max_m + kBlockEdge - 1) / kBlockEdge,
                  static_cast<unsigned int>(problems.size()));

  const auto raw_wall_start = std::chrono::steady_clock::now();
  cudaEventRecord(start, stream);
  for (std::uint32_t i = 0; i < repeats; ++i) {
    GroupedGemmFp16AccumKernel<<<grid, block, 0, stream>>>(
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
                      "raw repeated GroupedGemmFp16AccumKernel launch");
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
    GroupedGemmFp16AccumKernel<<<grid, block, 0, stream>>>(
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
  result.c_tiles = SplitOutputTiles(problems, packed.c_all, packed.c_offsets);
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

  const dim3 block(kBlockEdge, kBlockEdge);
  const dim3 grid((packed.max_n + kBlockEdge - 1) / kBlockEdge,
                  (packed.max_m + kBlockEdge - 1) / kBlockEdge,
                  static_cast<unsigned int>(packed.ms.size()));

  const auto raw_wall_start = std::chrono::steady_clock::now();
  cudaEventRecord(start, stream);
  for (std::uint32_t i = 0; i < repeats; ++i) {
    GroupedGemmFp16AccumKernel<<<grid, block, 0, stream>>>(
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
                      "raw repeated planned GroupedGemmFp16AccumKernel launch");
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
    GroupedGemmFp16AccumKernel<<<grid, block, 0, stream>>>(
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
  result.c_tiles = SplitOutputTilesFromPacked(packed.ms, packed.ns,
                                              packed.c_all, packed.c_offsets);
  result.ledger.Add(CudaMixedGemmLedgerEntry(
      packed.products * repeats, packed.max_abs_error_bound));
  return result;
}

}  // namespace tides::tile
