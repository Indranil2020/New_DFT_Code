#include "tile/reduce_f64e.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "tile/f64e_reference.hpp"

namespace tides::tile {
namespace {

constexpr int kReductionBlock = 256;
constexpr int kMaxReductionBlocks = 1024;

struct DeviceDoubleDouble {
  double hi;
  double lo;
};

[[nodiscard]] Status CudaStatus(cudaError_t error, const char* context) {
  if (error == cudaSuccess) {
    return Status::Ok();
  }
  return Status::IoError(std::string(context) + ": " +
                         cudaGetErrorString(error));
}

template <typename T>
[[nodiscard]] Status CopyToDevice(const std::vector<T>& host, T** device) {
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
[[nodiscard]] Status AllocateDevice(std::size_t count, T** device) {
  if (count == 0) {
    *device = nullptr;
    return Status::Ok();
  }
  const cudaError_t error =
      cudaMalloc(reinterpret_cast<void**>(device), count * sizeof(T));
  if (error != cudaSuccess) {
    return CudaStatus(error, "cudaMalloc");
  }
  return Status::Ok();
}

template <typename T>
void FreeDevice(T* ptr) {
  if (ptr != nullptr) {
    cudaFree(ptr);
  }
}

__device__ DeviceDoubleDouble TwoSum(double a, double b) {
  const double sum = a + b;
  const double b_virtual = sum - a;
  const double error = (a - (sum - b_virtual)) + (b - b_virtual);
  return {sum, error};
}

__device__ void AddDouble(DeviceDoubleDouble* acc, double value) {
  DeviceDoubleDouble first = TwoSum(acc->hi, value);
  DeviceDoubleDouble second = TwoSum(first.hi, acc->lo + first.lo);
  acc->hi = second.hi;
  acc->lo = second.lo;
}

__device__ void AddDoubleDouble(DeviceDoubleDouble* acc,
                                DeviceDoubleDouble value) {
  AddDouble(acc, value.hi);
  AddDouble(acc, value.lo);
}

__device__ DeviceDoubleDouble ProductDoubleDouble(double a, double b) {
  const double product = a * b;
  const double residual = fma(a, b, -product);
  return {product, residual};
}

__global__ void DotPartialKernel(const double* a, const double* b,
                                 std::size_t n, double* partial_hi,
                                 double* partial_lo) {
  __shared__ double hi[kReductionBlock];
  __shared__ double lo[kReductionBlock];
  const int tid = threadIdx.x;
  DeviceDoubleDouble acc{0.0, 0.0};
  for (std::size_t i = blockIdx.x * blockDim.x + tid; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    AddDoubleDouble(&acc, ProductDoubleDouble(a[i], b[i]));
  }
  hi[tid] = acc.hi;
  lo[tid] = acc.lo;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      DeviceDoubleDouble merged{hi[tid], lo[tid]};
      AddDoubleDouble(&merged, DeviceDoubleDouble{hi[tid + stride],
                                                  lo[tid + stride]});
      hi[tid] = merged.hi;
      lo[tid] = merged.lo;
    }
    __syncthreads();
  }
  if (tid == 0) {
    partial_hi[blockIdx.x] = hi[0];
    partial_lo[blockIdx.x] = lo[0];
  }
}

__global__ void SumPartialKernel(const double* values, std::size_t n,
                                 double* partial_hi, double* partial_lo) {
  __shared__ double hi[kReductionBlock];
  __shared__ double lo[kReductionBlock];
  const int tid = threadIdx.x;
  DeviceDoubleDouble acc{0.0, 0.0};
  for (std::size_t i = blockIdx.x * blockDim.x + tid; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    AddDouble(&acc, values[i]);
  }
  hi[tid] = acc.hi;
  lo[tid] = acc.lo;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      DeviceDoubleDouble merged{hi[tid], lo[tid]};
      AddDoubleDouble(&merged, DeviceDoubleDouble{hi[tid + stride],
                                                  lo[tid + stride]});
      hi[tid] = merged.hi;
      lo[tid] = merged.lo;
    }
    __syncthreads();
  }
  if (tid == 0) {
    partial_hi[blockIdx.x] = hi[0];
    partial_lo[blockIdx.x] = lo[0];
  }
}

__global__ void TracePartialKernel(const double* values, std::size_t rows,
                                   std::size_t cols, std::size_t diagonal,
                                   double* partial_hi, double* partial_lo) {
  __shared__ double hi[kReductionBlock];
  __shared__ double lo[kReductionBlock];
  const int tid = threadIdx.x;
  DeviceDoubleDouble acc{0.0, 0.0};
  for (std::size_t i = blockIdx.x * blockDim.x + tid; i < diagonal;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    AddDouble(&acc, values[i * cols + i]);
  }
  (void)rows;
  hi[tid] = acc.hi;
  lo[tid] = acc.lo;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      DeviceDoubleDouble merged{hi[tid], lo[tid]};
      AddDoubleDouble(&merged, DeviceDoubleDouble{hi[tid + stride],
                                                  lo[tid + stride]});
      hi[tid] = merged.hi;
      lo[tid] = merged.lo;
    }
    __syncthreads();
  }
  if (tid == 0) {
    partial_hi[blockIdx.x] = hi[0];
    partial_lo[blockIdx.x] = lo[0];
  }
}

__global__ void FinalReductionKernel(const double* partial_hi,
                                     const double* partial_lo,
                                     std::size_t count, double* out) {
  __shared__ double hi[kReductionBlock];
  __shared__ double lo[kReductionBlock];
  const int tid = threadIdx.x;
  DeviceDoubleDouble acc{0.0, 0.0};
  for (std::size_t i = tid; i < count; i += blockDim.x) {
    AddDoubleDouble(&acc, DeviceDoubleDouble{partial_hi[i], partial_lo[i]});
  }
  hi[tid] = acc.hi;
  lo[tid] = acc.lo;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      DeviceDoubleDouble merged{hi[tid], lo[tid]};
      AddDoubleDouble(&merged, DeviceDoubleDouble{hi[tid + stride],
                                                  lo[tid + stride]});
      hi[tid] = merged.hi;
      lo[tid] = merged.lo;
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[0] = hi[0] + lo[0];
  }
}

[[nodiscard]] PrecisionDescriptor CudaF64eReductionPrecision() {
  PrecisionDescriptor descriptor;
  descriptor.storage = NumericFormat::kFloat64;
  descriptor.compute = NumericFormat::kFloat64Emulated;
  descriptor.reduction = NumericFormat::kFloat64Emulated;
  descriptor.determinism = DeterminismMode::kDeterministic;
  descriptor.ordered_reductions = true;
  descriptor.label = "cuda-double-double-f64e-reduction-reference";
  return descriptor;
}

[[nodiscard]] double Fp64ForwardAbsBound(std::size_t terms,
                                         long double sum_abs_terms) {
  constexpr long double unit_roundoff =
      static_cast<long double>(std::numeric_limits<double>::epsilon()) / 2.0L;
  const long double work =
      static_cast<long double>(std::max<std::size_t>(1, 2 * terms + 64));
  const long double gamma =
      work * unit_roundoff < 1.0L
          ? (work * unit_roundoff) / (1.0L - work * unit_roundoff)
          : work * unit_roundoff;
  return static_cast<double>(gamma * sum_abs_terms);
}

[[nodiscard]] std::uint32_t BlockCount(std::size_t terms) {
  const std::size_t blocks =
      (terms + static_cast<std::size_t>(kReductionBlock) - 1) /
      static_cast<std::size_t>(kReductionBlock);
  return static_cast<std::uint32_t>(
      std::max<std::size_t>(1, std::min<std::size_t>(kMaxReductionBlocks,
                                                     blocks)));
}

[[nodiscard]] Result<CudaF64eReductionResult> EmptyReduction(
    OperationKind operation, const char* note) {
  CudaF64eReductionResult result;
  result.ledger.Add(OperationLedgerEntry{
      operation,
      CudaF64eReductionPrecision(),
      ErrorBudget{ErrorMetric::kAbsolute, 0.0,
                  "empty deterministic f64e reduction"},
      0.0,
      0,
      0,
      0,
      note});
  return result;
}

[[nodiscard]] Result<double> LaunchDotReduction(const std::vector<double>& a,
                                                const std::vector<double>& b,
                                                double* kernel_ms) {
  double* device_a = nullptr;
  double* device_b = nullptr;
  double* partial_hi = nullptr;
  double* partial_lo = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  const std::uint32_t blocks = BlockCount(a.size());

  Status status = CopyToDevice(a, &device_a);
  if (!status.ok()) return status;
  status = CopyToDevice(b, &device_b);
  if (!status.ok()) {
    FreeDevice(device_a);
    return status;
  }
  status = AllocateDevice(blocks, &partial_hi);
  if (!status.ok()) {
    FreeDevice(device_a);
    FreeDevice(device_b);
    return status;
  }
  status = AllocateDevice(blocks, &partial_lo);
  if (!status.ok()) {
    FreeDevice(device_a);
    FreeDevice(device_b);
    FreeDevice(partial_hi);
    return status;
  }

  cudaError_t error = cudaEventCreate(&start);
  if (error != cudaSuccess) {
    FreeDevice(device_a);
    FreeDevice(device_b);
    FreeDevice(partial_hi);
    FreeDevice(partial_lo);
    return CudaStatus(error, "cudaEventCreate reduction start");
  }
  error = cudaEventCreate(&stop);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    FreeDevice(device_a);
    FreeDevice(device_b);
    FreeDevice(partial_hi);
    FreeDevice(partial_lo);
    return CudaStatus(error, "cudaEventCreate reduction stop");
  }

  cudaEventRecord(start);
  DotPartialKernel<<<blocks, kReductionBlock>>>(device_a, device_b, a.size(),
                                                partial_hi, partial_lo);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    FreeDevice(device_a);
    FreeDevice(device_b);
    FreeDevice(partial_hi);
    FreeDevice(partial_lo);
    return CudaStatus(error, "DotPartialKernel launch");
  }
  FinalReductionKernel<<<1, kReductionBlock>>>(partial_hi, partial_lo, blocks,
                                               partial_hi);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    FreeDevice(device_a);
    FreeDevice(device_b);
    FreeDevice(partial_hi);
    FreeDevice(partial_lo);
    return CudaStatus(error, "FinalReductionKernel launch");
  }
  cudaEventRecord(stop);
  error = cudaEventSynchronize(stop);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    FreeDevice(device_a);
    FreeDevice(device_b);
    FreeDevice(partial_hi);
    FreeDevice(partial_lo);
    return CudaStatus(error, "CUDA f64e dot reduction synchronize");
  }
  float elapsed = 0.0F;
  cudaEventElapsedTime(&elapsed, start, stop);
  *kernel_ms = static_cast<double>(elapsed);

  double value = 0.0;
  error = cudaMemcpy(&value, partial_hi, sizeof(double),
                     cudaMemcpyDeviceToHost);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  FreeDevice(device_a);
  FreeDevice(device_b);
  FreeDevice(partial_hi);
  FreeDevice(partial_lo);
  if (error != cudaSuccess) {
    return CudaStatus(error, "cudaMemcpy dot reduction D2H");
  }
  return value;
}

enum class ReductionPayload {
  kSum,
  kTrace,
};

[[nodiscard]] Result<double> LaunchLinearReduction(
    const std::vector<double>& values, std::size_t rows, std::size_t cols,
    ReductionPayload payload, double* kernel_ms) {
  double* device_values = nullptr;
  double* partial_hi = nullptr;
  double* partial_lo = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  const std::size_t terms =
      payload == ReductionPayload::kTrace ? std::min(rows, cols)
                                          : values.size();
  const std::uint32_t blocks = BlockCount(terms);

  Status status = CopyToDevice(values, &device_values);
  if (!status.ok()) return status;
  status = AllocateDevice(blocks, &partial_hi);
  if (!status.ok()) {
    FreeDevice(device_values);
    return status;
  }
  status = AllocateDevice(blocks, &partial_lo);
  if (!status.ok()) {
    FreeDevice(device_values);
    FreeDevice(partial_hi);
    return status;
  }

  cudaError_t error = cudaEventCreate(&start);
  if (error != cudaSuccess) {
    FreeDevice(device_values);
    FreeDevice(partial_hi);
    FreeDevice(partial_lo);
    return CudaStatus(error, "cudaEventCreate reduction start");
  }
  error = cudaEventCreate(&stop);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    FreeDevice(device_values);
    FreeDevice(partial_hi);
    FreeDevice(partial_lo);
    return CudaStatus(error, "cudaEventCreate reduction stop");
  }

  cudaEventRecord(start);
  if (payload == ReductionPayload::kTrace) {
    TracePartialKernel<<<blocks, kReductionBlock>>>(
        device_values, rows, cols, terms, partial_hi, partial_lo);
  } else {
    SumPartialKernel<<<blocks, kReductionBlock>>>(
        device_values, terms, partial_hi, partial_lo);
  }
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    FreeDevice(device_values);
    FreeDevice(partial_hi);
    FreeDevice(partial_lo);
    return CudaStatus(error, payload == ReductionPayload::kTrace
                                 ? "TracePartialKernel launch"
                                 : "SumPartialKernel launch");
  }
  FinalReductionKernel<<<1, kReductionBlock>>>(partial_hi, partial_lo, blocks,
                                               partial_hi);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    FreeDevice(device_values);
    FreeDevice(partial_hi);
    FreeDevice(partial_lo);
    return CudaStatus(error, "FinalReductionKernel launch");
  }
  cudaEventRecord(stop);
  error = cudaEventSynchronize(stop);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    FreeDevice(device_values);
    FreeDevice(partial_hi);
    FreeDevice(partial_lo);
    return CudaStatus(error, "CUDA f64e linear reduction synchronize");
  }
  float elapsed = 0.0F;
  cudaEventElapsedTime(&elapsed, start, stop);
  *kernel_ms = static_cast<double>(elapsed);

  double value = 0.0;
  error = cudaMemcpy(&value, partial_hi, sizeof(double),
                     cudaMemcpyDeviceToHost);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  FreeDevice(device_values);
  FreeDevice(partial_hi);
  FreeDevice(partial_lo);
  if (error != cudaSuccess) {
    return CudaStatus(error, "cudaMemcpy linear reduction D2H");
  }
  return value;
}

[[nodiscard]] bool AllFinite(const std::vector<double>& values) {
  for (const double value : values) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] Result<CudaF64eReductionResult> BuildResult(
    double value, double kernel_ms, long double exact, long double sum_abs,
    std::size_t terms, OperationKind operation, const char* note) {
  if (!std::isfinite(value)) {
    return Status::OutOfRange("CUDA f64e reduction produced non-finite value");
  }
  CudaF64eReductionResult result;
  result.value = value;
  result.kernel_ms = kernel_ms;
  result.abs_error_vs_long_double = AbsLongDoubleToDoubleError(exact, value);
  const double forward_bound = Fp64ForwardAbsBound(terms, sum_abs);
  const double final_rounding = AbsLongDoubleToDoubleError(
      exact, static_cast<double>(exact));
  result.analytical_abs_bound = forward_bound + final_rounding;
  result.ledger.Add(OperationLedgerEntry{
      operation,
      CudaF64eReductionPrecision(),
      ErrorBudget{ErrorMetric::kAbsolute, result.analytical_abs_bound,
                  "deterministic double-double CUDA reduction forward bound"},
      result.abs_error_vs_long_double,
      static_cast<std::uint64_t>(terms),
      static_cast<std::uint64_t>(terms),
      0,
      note});
  return result;
}

}  // namespace

Result<CudaF64eReductionResult> DotF64eCuda(
    const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) {
    return Status::InvalidArgument("CUDA f64e dot shape mismatch");
  }
  if (a.empty()) {
    return EmptyReduction(OperationKind::kDot, "CUDA f64e dot reduction");
  }
  if (!AllFinite(a) || !AllFinite(b)) {
    return Status::InvalidArgument("CUDA f64e dot requires finite inputs");
  }
  long double exact = 0.0L;
  long double sum_abs = 0.0L;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const long double product =
        static_cast<long double>(a[i]) * static_cast<long double>(b[i]);
    if (!std::isfinite(static_cast<double>(product))) {
      return Status::OutOfRange(
          "CUDA f64e dot product is outside finite FP64 range");
    }
    exact += product;
    sum_abs += product < 0.0L ? -product : product;
  }

  double kernel_ms = 0.0;
  auto value = LaunchDotReduction(a, b, &kernel_ms);
  if (!value.ok()) {
    return value.status();
  }
  return BuildResult(value.value(), kernel_ms, exact, sum_abs, a.size(),
                     OperationKind::kDot, "CUDA f64e dot reduction");
}

Result<CudaF64eReductionResult> SumF64eCuda(
    const std::vector<double>& values) {
  if (values.empty()) {
    return EmptyReduction(OperationKind::kReduction,
                          "CUDA f64e linear sum reduction");
  }
  if (!AllFinite(values)) {
    return Status::InvalidArgument("CUDA f64e sum requires finite inputs");
  }
  long double exact = 0.0L;
  long double sum_abs = 0.0L;
  for (const double value : values) {
    exact += static_cast<long double>(value);
    sum_abs += std::abs(static_cast<long double>(value));
  }

  double kernel_ms = 0.0;
  auto value =
      LaunchLinearReduction(values, 0, 0, ReductionPayload::kSum, &kernel_ms);
  if (!value.ok()) {
    return value.status();
  }
  return BuildResult(value.value(), kernel_ms, exact, sum_abs, values.size(),
                     OperationKind::kReduction,
                     "CUDA f64e linear sum reduction");
}

Result<CudaF64eReductionResult> TraceF64eCuda(
    std::size_t rows, std::size_t cols, const std::vector<double>& values) {
  if (values.size() != rows * cols) {
    return Status::InvalidArgument("CUDA f64e trace shape mismatch");
  }
  const std::size_t diagonal = std::min(rows, cols);
  if (diagonal == 0) {
    return EmptyReduction(OperationKind::kTrace, "CUDA f64e trace reduction");
  }
  if (!AllFinite(values)) {
    return Status::InvalidArgument("CUDA f64e trace requires finite inputs");
  }
  long double exact = 0.0L;
  long double sum_abs = 0.0L;
  for (std::size_t i = 0; i < diagonal; ++i) {
    const long double value = static_cast<long double>(values[i * cols + i]);
    exact += value;
    sum_abs += value < 0.0L ? -value : value;
  }

  double kernel_ms = 0.0;
  auto value = LaunchLinearReduction(values, rows, cols,
                                     ReductionPayload::kTrace, &kernel_ms);
  if (!value.ok()) {
    return value.status();
  }
  return BuildResult(value.value(), kernel_ms, exact, sum_abs, diagonal,
                     OperationKind::kTrace, "CUDA f64e trace reduction");
}

}  // namespace tides::tile
