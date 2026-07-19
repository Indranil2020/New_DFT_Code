#pragma once

// GPU memory arena + persistent stream (audit B10).
//
// Every GPU wrapper (XCEvalLdaCuda, rho/vmat/poisson GPU paths) previously
// did cudaMalloc → H2D → kernel → D2H → cudaFree → cudaDeviceSynchronize
// per call. This is the residency defect the whole XC design exists to fix.
//
// This arena provides:
//   1. Persistent device buffers that grow as needed (never shrink mid-session)
//   2. A persistent CUDA stream (no implicit full-device stalls)
//   3. Async H2D/D2H on the stream (overlapped with compute)
//
// Usage:
//   GpuArena& arena = GpuArena::Instance();
//   auto* d_rho = arena.Alloc(N * sizeof(double));
//   arena.H2D(d_rho, rho.data(), N * sizeof(double));
//   // ... launch kernels on arena.Stream() ...
//   arena.Sync();
//   arena.Free(d_rho);  // returns to pool, not cudaFree

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#if defined(__CUDACC__) || defined(TIDES_HAVE_CUDA)
#include <cuda_runtime.h>
#endif

namespace tides::grid {

#if defined(__CUDACC__) || defined(TIDES_HAVE_CUDA)

class GpuArena {
 public:
  static GpuArena& Instance() {
    static GpuArena instance;
    return instance;
  }

  // Allocate device memory. Reuses cached blocks when possible.
  // Returns nullptr if allocation fails.
  void* Alloc(std::size_t bytes) {
    if (bytes == 0) return nullptr;

    // Try to find the smallest cached block of sufficient size.
    std::size_t best_idx = free_blocks_.size();
    std::size_t best_waste = static_cast<std::size_t>(-1);
    for (std::size_t i = 0; i < free_blocks_.size(); ++i) {
      const auto& block = free_blocks_[i];
      if (block.second >= bytes && block.second - bytes < best_waste) {
        best_idx = i;
        best_waste = block.second - bytes;
      }
    }
    if (best_idx < free_blocks_.size()) {
      void* ptr = free_blocks_[best_idx].first;
      free_blocks_.erase(free_blocks_.begin() + best_idx);
      return ptr;
    }

    // No cached block — allocate new.
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) return nullptr;
    sizes_[ptr] = bytes;
    total_allocated_ += bytes;
    return ptr;
  }

  // Return a block to the pool (no cudaFree).
  void Free(void* ptr) {
    if (ptr == nullptr) return;
    auto it = sizes_.find(ptr);
    if (it == sizes_.end()) {
      // Untracked pointer: free immediately.
      cudaFree(ptr);
      return;
    }
    std::size_t bytes = it->second;
    if (free_blocks_.size() < max_cached_blocks_) {
      free_blocks_.push_back({ptr, bytes});
    } else {
      sizes_.erase(it);
      cudaFree(ptr);
    }
  }

  // Async H2D on the persistent stream.
  cudaError_t H2D(void* dst, const void* src, std::size_t bytes) {
    return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream_);
  }

  // Async D2H on the persistent stream.
  cudaError_t D2H(void* dst, const void* src, std::size_t bytes) {
    return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream_);
  }

  // Synchronize the stream.
  cudaError_t Sync() {
    return cudaStreamSynchronize(stream_);
  }

  // Get the persistent stream for kernel launches.
  cudaStream_t Stream() const { return stream_; }

  // Free all cached (Free'd) blocks back to the device, but keep the stream.
  // Call between independent SCF runs to avoid memory accumulation.
  void ReleaseCached() {
    for (auto& [ptr, sz] : free_blocks_) {
      if (ptr) cudaFree(ptr);
      sizes_.erase(ptr);
    }
    free_blocks_.clear();
  }

  // Release all cached blocks and the persistent stream (call at session end).
  void Release() {
    ReleaseCached();
    if (stream_) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  ~GpuArena() { Release(); }

  // Prevent copies.
  GpuArena(const GpuArena&) = delete;
  GpuArena& operator=(const GpuArena&) = delete;

 private:
  GpuArena() : stream_(nullptr), total_allocated_(0) {
    cudaStreamCreate(&stream_);
  }

  cudaStream_t stream_ = nullptr;
  std::size_t total_allocated_ = 0;
  std::unordered_map<void*, std::size_t> sizes_;
  static constexpr std::size_t max_cached_blocks_ = 32;
  std::vector<std::pair<void*, std::size_t>> free_blocks_;
};

#endif  // __CUDACC__ || TIDES_HAVE_CUDA

}  // namespace tides::grid
