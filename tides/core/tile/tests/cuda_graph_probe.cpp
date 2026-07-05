#include "tile/gemm_grouped.hpp"

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

using tides::tile::CudaGemmProblem;
using tides::tile::CudaRuntimeStatus;
using tides::tile::GroupedGemmFp16AccumCudaGraphReplay;
using tides::tile::GroupedGemmFp64CudaGraphReplay;

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

int RunCase(const char* label, const std::vector<CudaGemmProblem>& problems,
            std::uint32_t repeats) {
  auto graph = GroupedGemmFp64CudaGraphReplay(problems, repeats);
  if (!graph.ok()) {
    std::cerr << "GroupedGemmFp64CudaGraphReplay failed: "
              << graph.status().message() << '\n';
    return 1;
  }
  std::cout << "cuda_graph_probe:"
            << " case=" << label
            << " problems=" << problems.size()
            << " repeats=" << graph.value().repeats
            << " raw_repeated_kernel_ms="
            << graph.value().raw_repeated_kernel_ms
            << " raw_repeated_wall_ms="
            << graph.value().raw_repeated_wall_ms
            << " graph_setup_ms=" << graph.value().graph_setup_ms
            << " graph_replay_ms=" << graph.value().graph_replay_ms
            << " graph_replay_wall_ms="
            << graph.value().graph_replay_wall_ms
            << " raw_per_launch_us="
            << 1000.0 * graph.value().raw_repeated_kernel_ms / repeats
            << " graph_per_captured_kernel_us="
            << 1000.0 * graph.value().graph_replay_ms / repeats
            << " launch_count_reduction=" << repeats
            << '\n';
  return 0;
}

int RunMixedCase(const char* label, const std::vector<CudaGemmProblem>& problems,
                 std::uint32_t repeats) {
  auto graph = GroupedGemmFp16AccumCudaGraphReplay(problems, repeats);
  if (!graph.ok()) {
    std::cerr << "GroupedGemmFp16AccumCudaGraphReplay failed: "
              << graph.status().message() << '\n';
    return 1;
  }
  std::cout << "cuda_graph_probe:"
            << " case=" << label
            << " problems=" << problems.size()
            << " repeats=" << graph.value().repeats
            << " raw_repeated_kernel_ms="
            << graph.value().raw_repeated_kernel_ms
            << " raw_repeated_wall_ms="
            << graph.value().raw_repeated_wall_ms
            << " graph_setup_ms=" << graph.value().graph_setup_ms
            << " graph_replay_ms=" << graph.value().graph_replay_ms
            << " graph_replay_wall_ms="
            << graph.value().graph_replay_wall_ms
            << " raw_per_launch_us="
            << 1000.0 * graph.value().raw_repeated_kernel_ms / repeats
            << " graph_per_captured_kernel_us="
            << 1000.0 * graph.value().graph_replay_ms / repeats
            << " launch_count_reduction=" << repeats
            << " mixed_abs_bound="
            << graph.value().ledger.entries().front().budget.bound
            << '\n';
  return 0;
}

}  // namespace

int main() {
  const auto runtime_status = CudaRuntimeStatus();
  if (!runtime_status.ok()) {
    std::cerr << "CUDA runtime/device is not available: "
              << runtime_status.message() << '\n';
    return 77;
  }

  std::vector<CudaGemmProblem> tiny_problems;
  CudaGemmProblem tiny;
  tiny.m = 16;
  tiny.k = 16;
  tiny.n = 16;
  tiny.a_scale = 4.0;
  tiny.b_scale = -0.25;
  tiny.epilogue_scale = 2.0;
  tiny.a = MakeDense(tiny.m, tiny.k, 0xB00000ULL);
  tiny.b = MakeDense(tiny.k, tiny.n, 0xB10000ULL);
  tiny_problems.push_back(std::move(tiny));

  if (RunCase("tiny_launch_bound", tiny_problems, 1000) != 0) {
    return 1;
  }
  if (RunMixedCase("tiny_launch_bound_mixed", tiny_problems, 1000) != 0) {
    return 1;
  }

  std::vector<CudaGemmProblem> problems;
  for (std::size_t i = 0; i < 128; ++i) {
    const std::uint32_t edge = i % 3 == 0 ? 16U : (i % 3 == 1 ? 32U : 64U);
    CudaGemmProblem problem;
    problem.m = edge;
    problem.k = edge;
    problem.n = edge;
    problem.a_scale = 1.0 + 0.015625 * static_cast<double>(i % 31);
    problem.b_scale = (i % 2 == 0) ? 0.5 : -0.5;
    problem.epilogue_scale = (i % 7 == 0) ? -1.5 : 1.0;
    problem.a = MakeDense(edge, edge, 0x900000ULL + i);
    problem.b = MakeDense(edge, edge, 0xA00000ULL + i);
    problems.push_back(std::move(problem));
  }

  if (RunCase("mixed_tile_compute", problems, 200) != 0) {
    return 1;
  }
  return RunMixedCase("mixed_tile_compute_fp16", problems, 200);
}
