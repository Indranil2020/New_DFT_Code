#include "tile/spgemm_filtered_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "tile/gemm_grouped.hpp"

namespace tides::tile {
namespace {

constexpr int kBlockEdge = 16;

struct ProductTaskHost {
  std::uint64_t a_offset = 0;
  std::uint64_t b_offset = 0;
  std::uint32_t k_extent = 0;
};

struct OutputTileHost {
  std::uint32_t block_row = 0;
  std::uint32_t block_col = 0;
  std::uint32_t row_extent = 0;
  std::uint32_t col_extent = 0;
  std::uint32_t task_begin = 0;
  std::uint32_t task_end = 0;
};

struct RetainedTaskBuild {
  std::size_t a_index = 0;
  std::size_t b_index = 0;
  std::uint32_t k_extent = 0;
};

[[nodiscard]] Status CudaStatus(cudaError_t error, const char* context) {
  if (error == cudaSuccess) {
    return Status::Ok();
  }
  return Status::IoError(std::string(context) + ": " +
                         cudaGetErrorString(error));
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  [[nodiscard]] Status CopyFromHost(const std::vector<T>& host,
                                    const char* context) {
    if (host.empty()) {
      return Status::Ok();
    }
    const std::size_t bytes = host.size() * sizeof(T);
    cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&ptr_), bytes);
    if (error != cudaSuccess) {
      return CudaStatus(error, context);
    }
    void* pinned = nullptr;
    error = cudaMallocHost(&pinned, bytes);
    if (error == cudaSuccess) {
      std::memcpy(pinned, host.data(), bytes);
      error = cudaMemcpy(ptr_, pinned, bytes, cudaMemcpyHostToDevice);
      cudaFreeHost(pinned);
    } else {
      error = cudaMemcpy(ptr_, host.data(), bytes, cudaMemcpyHostToDevice);
    }
    if (error != cudaSuccess) {
      cudaFree(ptr_);
      ptr_ = nullptr;
      return CudaStatus(error, context);
    }
    return Status::Ok();
  }

  [[nodiscard]] T* get() const { return ptr_; }

 private:
  T* ptr_ = nullptr;
};

class EventHandle {
 public:
  EventHandle() = default;
  EventHandle(const EventHandle&) = delete;
  EventHandle& operator=(const EventHandle&) = delete;

  ~EventHandle() {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
  }

  [[nodiscard]] Status Create(const char* context) {
    const cudaError_t error = cudaEventCreate(&event_);
    if (error != cudaSuccess) {
      event_ = nullptr;
      return CudaStatus(error, context);
    }
    return Status::Ok();
  }

  [[nodiscard]] cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

__global__ void FilteredSpGemmOutputTileKernel(
    const double* a_values, const double* b_values, double* c_dense,
    const ProductTaskHost* tasks, const OutputTileHost* outputs,
    std::uint32_t output_count, std::uint32_t edge, std::uint64_t c_cols) {
  const std::uint32_t output = static_cast<std::uint32_t>(blockIdx.z);
  if (output >= output_count) {
    return;
  }
  const OutputTileHost meta = outputs[output];
  const std::uint32_t local_row =
      static_cast<std::uint32_t>(blockIdx.y * blockDim.y + threadIdx.y);
  const std::uint32_t local_col =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const std::uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
  const std::uint32_t block_threads = blockDim.x * blockDim.y;

  extern __shared__ double smem[];
  double* s_a = smem;
  double* s_b = smem + edge * edge;

  double sum = 0.0;
  for (std::uint32_t task_id = meta.task_begin; task_id < meta.task_end;
       ++task_id) {
    const ProductTaskHost task = tasks[task_id];
    const double* a = a_values + task.a_offset;
    const double* b = b_values + task.b_offset;

    for (std::uint32_t idx = tid; idx < edge * edge; idx += block_threads) {
      s_a[idx] = a[idx];
      s_b[idx] = b[idx];
    }
    __syncthreads();

    if (local_row < meta.row_extent && local_col < meta.col_extent) {
      for (std::uint32_t k = 0; k < task.k_extent; ++k) {
        sum += s_a[local_row * edge + k] * s_b[k * edge + local_col];
      }
    }
    __syncthreads();
  }

  if (local_row < meta.row_extent && local_col < meta.col_extent) {
    const std::uint64_t row =
        static_cast<std::uint64_t>(meta.block_row) * edge + local_row;
    const std::uint64_t col =
        static_cast<std::uint64_t>(meta.block_col) * edge + local_col;
    c_dense[row * c_cols + col] = sum;
  }
}

}  // namespace

Result<CudaSpGemmFilteredResult> SpGemmFilteredFp64Cuda(
    const TileMat& a, const TileMat& b, double eps_filter) {
  if (a.cols() != b.rows()) {
    return Status::InvalidArgument("CUDA SpGEMM dimension mismatch");
  }
  if (a.tile_edge() != b.tile_edge()) {
    return Status::InvalidArgument(
        "CUDA SpGEMM reference requires matching tile edges");
  }
  if (eps_filter < 0.0 || std::isnan(eps_filter)) {
    return Status::InvalidArgument("eps_filter must be non-negative");
  }
  const Status runtime_status = CudaRuntimeStatus();
  if (!runtime_status.ok()) {
    return runtime_status;
  }

  const std::uint32_t edge = a.tile_edge();
  const std::size_t tile_cells = static_cast<std::size_t>(edge) * edge;
  const std::vector<detail::ExpandedTile> a_tiles =
      detail::ExpandToFullTiles(a);
  const std::vector<detail::ExpandedTile> b_tiles =
      detail::ExpandToFullTiles(b);
  const std::size_t b_block_rows = detail::CeilDiv(b.rows(), edge);
  std::vector<std::vector<std::size_t>> b_by_row(b_block_rows);
  for (std::size_t i = 0; i < b_tiles.size(); ++i) {
    b_by_row[b_tiles[i].block_row].push_back(i);
  }

  ErrorLedger ledger;
  ledger.eps_filter = eps_filter;
  std::map<std::pair<std::size_t, std::size_t>, double> dropped_by_output;
  std::map<std::pair<std::size_t, std::size_t>, std::vector<RetainedTaskBuild>>
      retained_by_output;

  for (std::size_t ai = 0; ai < a_tiles.size(); ++ai) {
    const detail::ExpandedTile& a_tile = a_tiles[ai];
    if (a_tile.block_col >= b_by_row.size()) {
      return Status::CorruptData("A tile column exceeds B block rows");
    }
    for (const std::size_t bi : b_by_row[a_tile.block_col]) {
      const detail::ExpandedTile& b_tile = b_tiles[bi];
      ++ledger.candidate_products;
      const double product_bound =
          a_tile.frobenius_norm * b_tile.frobenius_norm;
      const auto output_key =
          std::make_pair(a_tile.block_row, b_tile.block_col);
      if (product_bound < eps_filter) {
        ++ledger.dropped_products;
        ledger.dropped_frobenius_bound += product_bound;
        dropped_by_output[output_key] += product_bound;
        continue;
      }
      ++ledger.retained_products;
      retained_by_output[output_key].push_back(RetainedTaskBuild{
          ai,
          bi,
          static_cast<std::uint32_t>(
              std::min(a_tile.col_extent, b_tile.row_extent))});
    }
  }

  ledger.output_tile_bounds.reserve(dropped_by_output.size());
  for (const auto& [tile, bound] : dropped_by_output) {
    ledger.output_tile_bounds.push_back(
        TileErrorBound{tile.first, tile.second, bound});
  }
  ledger.operation_ledger.Add(OperationLedgerEntry{
      OperationKind::kSpGemmFiltered,
      PrecisionDescriptor{NumericFormat::kFloat64, NumericFormat::kFloat64,
                          NumericFormat::kFloat64,
                          DeterminismMode::kDeterministic, false, true,
                          "cuda-fp64-spgemm-reference"},
      ErrorBudget{ErrorMetric::kFrobenius, ledger.dropped_frobenius_bound,
                  "sum ||A_ik||_F ||B_kj||_F for dropped tile products"},
      ledger.dropped_frobenius_bound,
      static_cast<std::uint64_t>(ledger.candidate_products),
      static_cast<std::uint64_t>(ledger.retained_products),
      static_cast<std::uint64_t>(ledger.dropped_products),
      "CUDA FP64 deterministic filtered SpGEMM reference"});

  std::vector<double> a_values;
  std::vector<double> b_values;
  a_values.reserve(a_tiles.size() * tile_cells);
  b_values.reserve(b_tiles.size() * tile_cells);
  for (const detail::ExpandedTile& tile : a_tiles) {
    a_values.insert(a_values.end(), tile.values.begin(), tile.values.end());
  }
  for (const detail::ExpandedTile& tile : b_tiles) {
    b_values.insert(b_values.end(), tile.values.begin(), tile.values.end());
  }

  std::vector<ProductTaskHost> tasks;
  std::vector<OutputTileHost> outputs;
  tasks.reserve(ledger.retained_products);
  outputs.reserve(retained_by_output.size());
  for (const auto& [tile, retained] : retained_by_output) {
    const std::uint32_t task_begin = static_cast<std::uint32_t>(tasks.size());
    for (const RetainedTaskBuild& task : retained) {
      tasks.push_back(ProductTaskHost{
          static_cast<std::uint64_t>(task.a_index * tile_cells),
          static_cast<std::uint64_t>(task.b_index * tile_cells),
          task.k_extent});
    }
    const std::uint32_t task_end = static_cast<std::uint32_t>(tasks.size());
    const std::size_t row0 = tile.first * edge;
    const std::size_t col0 = tile.second * edge;
    outputs.push_back(OutputTileHost{
        static_cast<std::uint32_t>(tile.first),
        static_cast<std::uint32_t>(tile.second),
        static_cast<std::uint32_t>(
            std::min<std::size_t>(edge, a.rows() - row0)),
        static_cast<std::uint32_t>(
            std::min<std::size_t>(edge, b.cols() - col0)),
        task_begin,
        task_end});
  }

  std::vector<double> c_dense(a.rows() * b.cols(), 0.0);
  double kernel_ms = 0.0;
  if (!tasks.empty()) {
    DeviceBuffer<double> d_a;
    DeviceBuffer<double> d_b;
    DeviceBuffer<double> d_c;
    DeviceBuffer<ProductTaskHost> d_tasks;
    DeviceBuffer<OutputTileHost> d_outputs;

    Status status = d_a.CopyFromHost(a_values, "cudaMalloc/cudaMemcpy A");
    if (!status.ok()) return status;
    status = d_b.CopyFromHost(b_values, "cudaMalloc/cudaMemcpy B");
    if (!status.ok()) return status;
    status = d_c.CopyFromHost(c_dense, "cudaMalloc/cudaMemcpy C");
    if (!status.ok()) return status;
    status = d_tasks.CopyFromHost(tasks, "cudaMalloc/cudaMemcpy tasks");
    if (!status.ok()) return status;
    status = d_outputs.CopyFromHost(outputs, "cudaMalloc/cudaMemcpy outputs");
    if (!status.ok()) return status;

    EventHandle start;
    EventHandle stop;
    status = start.Create("cudaEventCreate start");
    if (!status.ok()) return status;
    status = stop.Create("cudaEventCreate stop");
    if (!status.ok()) return status;

    const dim3 block(kBlockEdge, kBlockEdge);
    const dim3 grid((edge + kBlockEdge - 1) / kBlockEdge,
                    (edge + kBlockEdge - 1) / kBlockEdge,
                    static_cast<unsigned int>(outputs.size()));
    const std::size_t smem_bytes =
        static_cast<std::size_t>(edge) * edge * 2 * sizeof(double);
    cudaError_t error = cudaEventRecord(start.get());
    if (error != cudaSuccess) return CudaStatus(error, "cudaEventRecord start");
    FilteredSpGemmOutputTileKernel<<<grid, block, smem_bytes>>>(
        d_a.get(), d_b.get(), d_c.get(), d_tasks.get(), d_outputs.get(),
        static_cast<std::uint32_t>(outputs.size()), edge, b.cols());
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaStatus(error, "FilteredSpGemmOutputTileKernel launch");
    }
    error = cudaEventRecord(stop.get());
    if (error != cudaSuccess) return CudaStatus(error, "cudaEventRecord stop");
    error = cudaEventSynchronize(stop.get());
    if (error != cudaSuccess) {
      return CudaStatus(error, "FilteredSpGemmOutputTileKernel synchronize");
    }
    float elapsed_ms = 0.0F;
    error = cudaEventElapsedTime(&elapsed_ms, start.get(), stop.get());
    if (error != cudaSuccess) return CudaStatus(error, "cudaEventElapsedTime");
    kernel_ms = static_cast<double>(elapsed_ms);

    error = cudaMemcpy(c_dense.data(), d_c.get(), c_dense.size() * sizeof(double),
                       cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return CudaStatus(error, "cudaMemcpy D2H");
  }

  auto product = TileMat::FromDense(a.rows(), b.cols(), c_dense, edge);
  if (!product.ok()) {
    return product.status();
  }
  return CudaSpGemmFilteredResult{product.take_value(), std::move(ledger),
                                  kernel_ms};
}

}  // namespace tides::tile
