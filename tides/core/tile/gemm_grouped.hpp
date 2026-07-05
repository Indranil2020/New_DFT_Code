#pragma once

#include <cstdint>
#include <vector>

#include "common/status.hpp"
#include "tile/precision.hpp"

namespace tides::tile {

struct CudaGemmProblem {
  std::uint32_t m = 0;
  std::uint32_t k = 0;
  std::uint32_t n = 0;
  double a_scale = 1.0;
  double b_scale = 1.0;
  double epilogue_scale = 1.0;
  std::vector<double> a;
  std::vector<double> b;
};

struct CudaGroupedGemmResult {
  std::vector<std::vector<double>> c_tiles;
  OperationLedger ledger;
  double kernel_ms = 0.0;
};

struct CudaGraphReplayResult {
  std::vector<std::vector<double>> c_tiles;
  OperationLedger ledger;
  std::uint32_t repeats = 0;
  double raw_repeated_kernel_ms = 0.0;
  double raw_repeated_wall_ms = 0.0;
  double graph_setup_ms = 0.0;
  double graph_replay_ms = 0.0;
  double graph_replay_wall_ms = 0.0;
};

class CudaGroupedGemmFp16AccumPlan {
 public:
  CudaGroupedGemmFp16AccumPlan();
  ~CudaGroupedGemmFp16AccumPlan();
  CudaGroupedGemmFp16AccumPlan(const CudaGroupedGemmFp16AccumPlan&) = delete;
  CudaGroupedGemmFp16AccumPlan& operator=(
      const CudaGroupedGemmFp16AccumPlan&) = delete;
  CudaGroupedGemmFp16AccumPlan(CudaGroupedGemmFp16AccumPlan&& other) noexcept;
  CudaGroupedGemmFp16AccumPlan& operator=(
      CudaGroupedGemmFp16AccumPlan&& other) noexcept;

  [[nodiscard]] bool valid() const;

 private:
  struct Impl;
  Impl* impl_ = nullptr;

  explicit CudaGroupedGemmFp16AccumPlan(Impl* impl);

  friend Result<CudaGroupedGemmFp16AccumPlan>
  BuildGroupedGemmFp16AccumCudaPlan(
      const std::vector<CudaGemmProblem>& problems);
  friend Result<CudaGroupedGemmResult> RunGroupedGemmFp16AccumCudaPlan(
      const CudaGroupedGemmFp16AccumPlan& plan);
  friend Result<CudaGraphReplayResult>
  RunGroupedGemmFp16AccumCudaPlanGraphReplay(
      const CudaGroupedGemmFp16AccumPlan& plan, std::uint32_t repeats);
};

[[nodiscard]] bool CudaRuntimeAvailable();
[[nodiscard]] Status CudaRuntimeStatus();

[[nodiscard]] Result<CudaGroupedGemmResult> GroupedGemmFp64Cuda(
    const std::vector<CudaGemmProblem>& problems);

[[nodiscard]] Result<CudaGroupedGemmResult> GroupedGemmFp16AccumCuda(
    const std::vector<CudaGemmProblem>& problems);

[[nodiscard]] Result<CudaGroupedGemmFp16AccumPlan>
BuildGroupedGemmFp16AccumCudaPlan(
    const std::vector<CudaGemmProblem>& problems);

[[nodiscard]] Result<CudaGroupedGemmResult> RunGroupedGemmFp16AccumCudaPlan(
    const CudaGroupedGemmFp16AccumPlan& plan);

[[nodiscard]] Result<CudaGraphReplayResult> GroupedGemmFp64CudaGraphReplay(
    const std::vector<CudaGemmProblem>& problems, std::uint32_t repeats);

[[nodiscard]] Result<CudaGraphReplayResult>
GroupedGemmFp16AccumCudaGraphReplay(
    const std::vector<CudaGemmProblem>& problems, std::uint32_t repeats);

[[nodiscard]] Result<CudaGraphReplayResult>
RunGroupedGemmFp16AccumCudaPlanGraphReplay(
    const CudaGroupedGemmFp16AccumPlan& plan, std::uint32_t repeats);

}  // namespace tides::tile
