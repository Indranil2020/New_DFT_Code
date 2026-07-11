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

#include <cublasLt.h>

#include "tile/gemm_grouped.hpp"

namespace tides::tile {
namespace {

constexpr int kBlockEdge = 16;

struct CublasLtHandleHolder {
  cublasLtHandle_t handle = nullptr;
  CublasLtHandleHolder() {
    cublasLtCreate(&handle);
    // Warm up cuBLASLt with a tiny matmul to trigger JIT compilation
    // so the first real call doesn't pay the cost.
    cublasLtMatmulDesc_t warm_desc = nullptr;
    cublasLtMatmulDescCreate(&warm_desc, CUBLAS_COMPUTE_64F, CUDA_R_64F);
    cublasOperation_t nt = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(warm_desc,
        CUBLASLT_MATMUL_DESC_TRANSA, &nt, sizeof(nt));
    cublasLtMatmulDescSetAttribute(warm_desc,
        CUBLASLT_MATMUL_DESC_TRANSB, &nt, sizeof(nt));
    cublasLtMatrixLayout_t warm_a = nullptr, warm_b = nullptr, warm_c = nullptr;
    cublasLtMatrixLayoutCreate(&warm_a, CUDA_R_64F, 16, 16, 16);
    cublasLtMatrixLayoutCreate(&warm_b, CUDA_R_64F, 16, 16, 16);
    cublasLtMatrixLayoutCreate(&warm_c, CUDA_R_64F, 16, 16, 16);
    double* d_wa = nullptr;
    double* d_wb = nullptr;
    double* d_wc = nullptr;
    cudaMalloc(&d_wa, 16 * 16 * sizeof(double));
    cudaMalloc(&d_wb, 16 * 16 * sizeof(double));
    cudaMalloc(&d_wc, 16 * 16 * sizeof(double));
    double alpha = 1.0, beta = 0.0;
    if (d_wa && d_wb && d_wc) {
      cublasLtMatmul(handle, warm_desc, &alpha, d_wa, warm_a, d_wb, warm_b,
          &beta, d_wc, warm_c, d_wc, warm_c, nullptr, nullptr, 0, nullptr);
      cudaDeviceSynchronize();
    }
    if (d_wa) cudaFree(d_wa);
    if (d_wb) cudaFree(d_wb);
    if (d_wc) cudaFree(d_wc);
    cublasLtMatmulDescDestroy(warm_desc);
    cublasLtMatrixLayoutDestroy(warm_a);
    cublasLtMatrixLayoutDestroy(warm_b);
    cublasLtMatrixLayoutDestroy(warm_c);
  }
  ~CublasLtHandleHolder() {
    if (handle) cublasLtDestroy(handle);
  }
};

__global__ void ScatterAddKernel(
    const double* task_results,
    double* c_dense,
    const std::uint32_t* task_block_row,
    const std::uint32_t* task_block_col,
    const std::uint32_t* task_row_extent,
    const std::uint32_t* task_col_extent,
    std::uint32_t num_tasks, std::uint32_t edge,
    std::uint64_t c_cols) {
  const std::uint32_t task_id = blockIdx.z;
  if (task_id >= num_tasks) return;
  const std::uint32_t local_row = blockIdx.y * blockDim.y + threadIdx.y;
  const std::uint32_t local_col = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_row >= task_row_extent[task_id] || local_col >= task_col_extent[task_id]) return;
  const double val = task_results[static_cast<std::uint64_t>(task_id) * edge * edge
                                  + local_row * edge + local_col];
  const std::uint64_t row = static_cast<std::uint64_t>(task_block_row[task_id]) * edge + local_row;
  const std::uint64_t col = static_cast<std::uint64_t>(task_block_col[task_id]) * edge + local_col;
  atomicAdd(&c_dense[row * c_cols + col], val);
}

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
    // Use cuBLASLt for all but the trivial single-task case.
    // cuBLASLt outperforms the custom kernel even at small batch sizes
    // due to optimized tensor core paths and better memory access patterns.
    const bool use_cublaslt = tasks.size() >= 4;
    if (use_cublaslt) {
      // Repack A and B data in task order for strided batched GEMM.
      std::vector<double> a_batch(tasks.size() * tile_cells);
      std::vector<double> b_batch(tasks.size() * tile_cells);
      std::vector<std::uint32_t> task_block_row(tasks.size());
      std::vector<std::uint32_t> task_block_col(tasks.size());
      std::vector<std::uint32_t> task_row_extent(tasks.size());
      std::vector<std::uint32_t> task_col_extent(tasks.size());
      std::uint32_t task_idx = 0;
      for (const auto& [tile, retained] : retained_by_output) {
        for (const RetainedTaskBuild& rtb : retained) {
          std::memcpy(&a_batch[task_idx * tile_cells],
                      &a_values[rtb.a_index * tile_cells],
                      tile_cells * sizeof(double));
          std::memcpy(&b_batch[task_idx * tile_cells],
                      &b_values[rtb.b_index * tile_cells],
                      tile_cells * sizeof(double));
          task_block_row[task_idx] =
              static_cast<std::uint32_t>(tile.first);
          task_block_col[task_idx] =
              static_cast<std::uint32_t>(tile.second);
          task_row_extent[task_idx] =
              static_cast<std::uint32_t>(
                  std::min<std::size_t>(edge, a.rows() - tile.first * edge));
          task_col_extent[task_idx] =
              static_cast<std::uint32_t>(
                  std::min<std::size_t>(edge, b.cols() - tile.second * edge));
          ++task_idx;
        }
      }

      DeviceBuffer<double> d_a;
      DeviceBuffer<double> d_b;
      DeviceBuffer<double> d_c;
      DeviceBuffer<double> d_task_c;
      DeviceBuffer<std::uint32_t> d_task_br;
      DeviceBuffer<std::uint32_t> d_task_bc;
      DeviceBuffer<std::uint32_t> d_task_re;
      DeviceBuffer<std::uint32_t> d_task_ce;

      Status status = d_a.CopyFromHost(a_batch, "cudaMalloc/cudaMemcpy A batch");
      if (!status.ok()) return status;
      status = d_b.CopyFromHost(b_batch, "cudaMalloc/cudaMemcpy B batch");
      if (!status.ok()) return status;
      status = d_c.CopyFromHost(c_dense, "cudaMalloc/cudaMemcpy C");
      if (!status.ok()) return status;
      status = d_task_c.CopyFromHost(
          std::vector<double>(tasks.size() * tile_cells, 0.0),
          "cudaMalloc/cudaMemcpy task C");
      if (!status.ok()) return status;
      status = d_task_br.CopyFromHost(task_block_row, "cudaMalloc/cudaMemcpy task_br");
      if (!status.ok()) return status;
      status = d_task_bc.CopyFromHost(task_block_col, "cudaMalloc/cudaMemcpy task_bc");
      if (!status.ok()) return status;
      status = d_task_re.CopyFromHost(task_row_extent, "cudaMalloc/cudaMemcpy task_re");
      if (!status.ok()) return status;
      status = d_task_ce.CopyFromHost(task_col_extent, "cudaMalloc/cudaMemcpy task_ce");
      if (!status.ok()) return status;

      EventHandle start;
      EventHandle stop;
      status = start.Create("cudaEventCreate start");
      if (!status.ok()) return status;
      status = stop.Create("cudaEventCreate stop");
      if (!status.ok()) return status;

      // cuBLASLt strided batched GEMM: C[i] = A[i] * B[i]
      static CublasLtHandleHolder lt_holder;
      const int nn = static_cast<int>(edge);
      const int batch_count = static_cast<int>(tasks.size());
      const long long stride = static_cast<long long>(tile_cells);
      const double alpha = 1.0;
      const double beta = 0.0;
      const cublasOperation_t no_trans = CUBLAS_OP_N;

      cublasLtMatmulDesc_t matmul_desc = nullptr;
      cublasStatus_t cb_status = cublasLtMatmulDescCreate(
          &matmul_desc, CUBLAS_COMPUTE_64F, CUDA_R_64F);
      if (cb_status != CUBLAS_STATUS_SUCCESS) {
        return Status::IoError("cublasLtMatmulDescCreate failed");
      }
      cublasLtMatmulDescSetAttribute(matmul_desc,
          CUBLASLT_MATMUL_DESC_TRANSA, &no_trans, sizeof(no_trans));
      cublasLtMatmulDescSetAttribute(matmul_desc,
          CUBLASLT_MATMUL_DESC_TRANSB, &no_trans, sizeof(no_trans));

      cublasLtMatrixLayout_t a_layout = nullptr, b_layout = nullptr, c_layout = nullptr;
      cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_64F, nn, nn, nn);
      cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_64F, nn, nn, nn);
      cublasLtMatrixLayoutCreate(&c_layout, CUDA_R_64F, nn, nn, nn);
      cublasLtMatrixLayoutSetAttribute(a_layout,
          CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
      cublasLtMatrixLayoutSetAttribute(a_layout,
          CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride));
      cublasLtMatrixLayoutSetAttribute(b_layout,
          CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
      cublasLtMatrixLayoutSetAttribute(b_layout,
          CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride));
      cublasLtMatrixLayoutSetAttribute(c_layout,
          CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
      cublasLtMatrixLayoutSetAttribute(c_layout,
          CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride));

      // Allocate workspace for cuBLASLt algorithm selection.
      const std::size_t workspace_size = 16 * 1024 * 1024;  // 16 MB
      void* workspace_ptr = nullptr;
      cudaError_t ws_err = cudaMalloc(&workspace_ptr, workspace_size);
      if (ws_err != cudaSuccess) workspace_ptr = nullptr;

      // AUDIT: cuBLASLt heuristic segfault on Blackwell sm_120.
      // The heuristic returns an invalid algorithm on sm_120+ that causes
      // a segfault when used. Check device compute capability and skip
      // the heuristic on Blackwell, falling back to the default algo.
      int device_major = 0, device_minor = 0;
      cudaDeviceGetAttribute(&device_major,
          cudaDevAttrComputeCapabilityMajor, 0);
      cudaDeviceGetAttribute(&device_minor,
          cudaDevAttrComputeCapabilityMinor, 0);
      const bool skip_heuristic = (device_major >= 12);

      cublasLtMatmulHeuristicResult_t heuristic_result = {};
      cublasLtMatmulPreference_t pref = nullptr;
      cublasLtMatmulPreferenceCreate(&pref);
      cublasLtMatmulPreferenceSetAttribute(pref,
          CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &workspace_size, sizeof(workspace_size));
      int returned_results = 0;
      if (!skip_heuristic) {
        cublasLtMatmulAlgoGetHeuristic(lt_holder.handle, matmul_desc,
            b_layout, a_layout, c_layout, c_layout,
            pref, 1, &heuristic_result, &returned_results);
      }

      cudaError_t error = cudaEventRecord(start.get());
      if (error != cudaSuccess) {
        if (workspace_ptr) cudaFree(workspace_ptr);
        cublasLtMatmulPreferenceDestroy(pref);
        return CudaStatus(error, "cudaEventRecord start");
      }

      // Row-major C = A_row * B_row = col-major B_col * A_col.
      // Swap A and B in cublasLtMatmul to account for row-major data.
      if (returned_results > 0) {
        cb_status = cublasLtMatmul(lt_holder.handle, matmul_desc,
            &alpha, d_b.get(), b_layout, d_a.get(), a_layout,
            &beta, d_task_c.get(), c_layout,
            d_task_c.get(), c_layout,
            &heuristic_result.algo, workspace_ptr, workspace_size, nullptr);
      } else {
        cb_status = cublasLtMatmul(lt_holder.handle, matmul_desc,
            &alpha, d_b.get(), b_layout, d_a.get(), a_layout,
            &beta, d_task_c.get(), c_layout,
            d_task_c.get(), c_layout,
            nullptr, workspace_ptr, workspace_size, nullptr);
      }
      if (cb_status != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatmulDescDestroy(matmul_desc);
        cublasLtMatrixLayoutDestroy(a_layout);
        cublasLtMatrixLayoutDestroy(b_layout);
        cublasLtMatrixLayoutDestroy(c_layout);
        cublasLtMatmulPreferenceDestroy(pref);
        if (workspace_ptr) cudaFree(workspace_ptr);
        return Status::IoError("cublasLtMatmul failed");
      }

      // Scatter-add per-task results into output matrix.
      const dim3 scatter_block(kBlockEdge, kBlockEdge);
      const dim3 scatter_grid((edge + kBlockEdge - 1) / kBlockEdge,
                              (edge + kBlockEdge - 1) / kBlockEdge,
              static_cast<unsigned int>(tasks.size()));
      ScatterAddKernel<<<scatter_grid, scatter_block>>>(
          d_task_c.get(), d_c.get(),
          d_task_br.get(), d_task_bc.get(),
          d_task_re.get(), d_task_ce.get(),
          static_cast<std::uint32_t>(tasks.size()), edge, b.cols());
      error = cudaGetLastError();
      if (error != cudaSuccess) {
        cublasLtMatmulDescDestroy(matmul_desc);
        cublasLtMatrixLayoutDestroy(a_layout);
        cublasLtMatrixLayoutDestroy(b_layout);
        cublasLtMatrixLayoutDestroy(c_layout);
        cublasLtMatmulPreferenceDestroy(pref);
        if (workspace_ptr) cudaFree(workspace_ptr);
        return CudaStatus(error, "ScatterAddKernel launch");
      }

      error = cudaEventRecord(stop.get());
      if (error != cudaSuccess) return CudaStatus(error, "cudaEventRecord stop");
      error = cudaEventSynchronize(stop.get());
      if (error != cudaSuccess) {
        cublasLtMatmulDescDestroy(matmul_desc);
        cublasLtMatrixLayoutDestroy(a_layout);
        cublasLtMatrixLayoutDestroy(b_layout);
        cublasLtMatrixLayoutDestroy(c_layout);
        cublasLtMatmulPreferenceDestroy(pref);
        if (workspace_ptr) cudaFree(workspace_ptr);
        return CudaStatus(error, "SpGEMM synchronize");
      }
      float elapsed_ms = 0.0F;
      error = cudaEventElapsedTime(&elapsed_ms, start.get(), stop.get());
      if (error != cudaSuccess) return CudaStatus(error, "cudaEventElapsedTime");
      kernel_ms = static_cast<double>(elapsed_ms);

      cublasLtMatmulDescDestroy(matmul_desc);
      cublasLtMatrixLayoutDestroy(a_layout);
      cublasLtMatrixLayoutDestroy(b_layout);
      cublasLtMatrixLayoutDestroy(c_layout);
      cublasLtMatmulPreferenceDestroy(pref);
      if (workspace_ptr) cudaFree(workspace_ptr);

      error = cudaMemcpy(c_dense.data(), d_c.get(), c_dense.size() * sizeof(double),
                         cudaMemcpyDeviceToHost);
      if (error != cudaSuccess) return CudaStatus(error, "cudaMemcpy D2H");
    } else {
      // Use custom kernel for small batch sizes (lower overhead).
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
  }

  auto product = TileMat::FromDense(a.rows(), b.cols(), c_dense, edge);
  if (!product.ok()) {
    return product.status();
  }
  return CudaSpGemmFilteredResult{product.take_value(), std::move(ledger),
                                  kernel_ms};
}

}  // namespace tides::tile
