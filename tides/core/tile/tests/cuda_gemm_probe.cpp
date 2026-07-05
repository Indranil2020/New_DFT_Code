#include "tile/f64e_reference.hpp"
#include "tile/gemm_grouped.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
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
  const auto planned_start = std::chrono::steady_clock::now();
  auto planned = RunGroupedGemmFp16AccumCudaPlan(mixed_plan.value());
  const auto planned_end = std::chrono::steady_clock::now();
  if (!planned.ok()) {
    std::cerr << "RunGroupedGemmFp16AccumCudaPlan failed: "
              << planned.status().message() << '\n';
    return 1;
  }
  if (planned.value().c_tiles != mixed.value().c_tiles) {
    std::cerr << "planned mixed GEMM differs from one-shot mixed GEMM\n";
    return 1;
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
            << " mixed_kernel_gflops="
            << flops / (mixed.value().kernel_ms * 1.0e6)
            << " plan_build_ms=" << plan_build_ms
            << " planned_total_ms=" << planned_total_ms
            << " planned_kernel_ms=" << planned.value().kernel_ms
            << " planned_kernel_gflops="
            << flops / (planned.value().kernel_ms * 1.0e6)
            << " mixed_abs_bound="
            << mixed.value().ledger.entries().front().budget.bound
            << '\n';
  return 0;
}
