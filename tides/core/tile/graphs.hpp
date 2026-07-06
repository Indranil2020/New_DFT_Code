#pragma once

// TIDES CUDA graph utilities.
//
// Provides RAII wrappers for CUDA graph capture and replay, used to eliminate
// kernel launch overhead in SCF loops where the same sequence of GPU kernels
// is executed every iteration.
//
// Usage:
//   CudaGraphCapture graph;
//   graph.Begin();
//   // ... launch kernels ...
//   graph.End();
//   graph.Replay();  // re-execute all captured kernels

#ifdef TIDES_HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include "common/status.hpp"

#include <cstddef>
#include <string>

namespace tides::tile {

#ifdef TIDES_HAVE_CUDA

class CudaGraphCapture {
 public:
  CudaGraphCapture() = default;
  ~CudaGraphCapture() { Release(); }

  CudaGraphCapture(const CudaGraphCapture&) = delete;
  CudaGraphCapture& operator=(const CudaGraphCapture&) = delete;

  // Begin graph capture on the current stream.
  [[nodiscard]] Status Begin() {
    cudaError_t err = cudaStreamBeginCapture(
        cudaStreamPerThread, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess)
      return Status::IoError("cudaStreamBeginCapture: " +
                             std::string(cudaGetErrorString(err)));
    capturing_ = true;
    return Status::Ok();
  }

  // End capture and instantiate the executable graph.
  [[nodiscard]] Status End() {
    if (!capturing_) return Status::InvalidArgument("not capturing");
    cudaError_t err = cudaStreamEndCapture(cudaStreamPerThread, &graph_);
    if (err != cudaSuccess)
      return Status::IoError("cudaStreamEndCapture: " +
                             std::string(cudaGetErrorString(err)));
    capturing_ = false;
    err = cudaGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0);
    if (err != cudaSuccess)
      return Status::IoError("cudaGraphInstantiate: " +
                             std::string(cudaGetErrorString(err)));
    instantiated_ = true;
    return Status::Ok();
  }

  // Replay the captured graph.
  [[nodiscard]] Status Replay() {
    if (!instantiated_)
      return Status::InvalidArgument("graph not instantiated");
    cudaError_t err = cudaGraphLaunch(exec_, cudaStreamPerThread);
    if (err != cudaSuccess)
      return Status::IoError("cudaGraphLaunch: " +
                             std::string(cudaGetErrorString(err)));
    err = cudaStreamSynchronize(cudaStreamPerThread);
    if (err != cudaSuccess)
      return Status::IoError("cudaStreamSynchronize: " +
                             std::string(cudaGetErrorString(err)));
    return Status::Ok();
  }

  void Release() {
    if (exec_) { cudaGraphExecDestroy(exec_); exec_ = nullptr; }
    if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }
    instantiated_ = false;
    capturing_ = false;
  }

  bool IsCapturing() const { return capturing_; }
  bool IsInstantiated() const { return instantiated_; }

 private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
  bool capturing_ = false;
  bool instantiated_ = false;
};

#endif  // TIDES_HAVE_CUDA

}  // namespace tides::tile
