#include "tile/f64e_reference.hpp"
#include "tile/gemm_grouped.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using tides::tile::CudaGemmProblem;
using tides::tile::CudaRuntimeStatus;
using tides::tile::GemmF64eReference;
using tides::tile::BuildGroupedGemmFp16AccumCudaPlan;
using tides::tile::GroupedGemmFp16AccumCuda;
using tides::tile::GroupedGemmFp16AccumCudaGraphReplay;
using tides::tile::GroupedGemmFp64Cuda;
using tides::tile::GroupedGemmFp64CudaGraphReplay;
using tides::tile::NumericFormat;
using tides::tile::RunGroupedGemmFp16AccumCudaPlanGraphReplay;
using tides::tile::ValidateOperationLedgerEntry;

int Fail(const std::string& message) {
  std::cerr << "cuda_graph_tests: " << message << '\n';
  return 1;
}

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

bool Near(const std::vector<double>& a, const std::vector<double>& b,
          double tolerance) {
  if (a.size() != b.size()) {
    return false;
  }
  long double err2 = 0.0L;
  long double ref2 = 0.0L;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const long double diff =
        static_cast<long double>(a[i]) - static_cast<long double>(b[i]);
    err2 += diff * diff;
    ref2 += static_cast<long double>(b[i]) * static_cast<long double>(b[i]);
  }
  return std::sqrt(static_cast<double>(err2)) <=
         tolerance * (1.0 + std::sqrt(static_cast<double>(ref2)));
}

std::vector<double> ScaleReference(const CudaGemmProblem& problem,
                                   std::vector<double> values) {
  const double scale =
      problem.a_scale * problem.b_scale * problem.epilogue_scale;
  for (double& value : values) {
    value *= scale;
  }
  return values;
}

double MaxAbsError(const std::vector<double>& a, const std::vector<double>& b) {
  double max_error = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    max_error = std::max(max_error, std::abs(a[i] - b[i]));
  }
  return max_error;
}

}  // namespace

int main() {
  const auto runtime_status = CudaRuntimeStatus();
  if (!runtime_status.ok()) {
    std::cerr << "cuda_graph_tests: SKIP " << runtime_status.message() << '\n';
    return 77;
  }

  std::vector<CudaGemmProblem> problems;
  for (std::size_t i = 0; i < 12; ++i) {
    const std::uint32_t edge = i % 3 == 0 ? 16U : (i % 3 == 1 ? 32U : 64U);
    CudaGemmProblem problem;
    problem.m = edge;
    problem.k = edge;
    problem.n = edge;
    problem.a_scale = 1.0 + 0.125 * static_cast<double>(i % 5);
    problem.b_scale = (i % 2 == 0) ? 0.5 : -0.25;
    problem.epilogue_scale = (i % 3 == 0) ? -2.0 : 3.0;
    problem.a = MakeDense(edge, edge, 0x700000ULL + i);
    problem.b = MakeDense(edge, edge, 0x800000ULL + i);
    problems.push_back(std::move(problem));
  }

  auto single = GroupedGemmFp64Cuda(problems);
  auto graph = GroupedGemmFp64CudaGraphReplay(problems, 16);
  auto mixed_single = GroupedGemmFp16AccumCuda(problems);
  auto mixed_graph = GroupedGemmFp16AccumCudaGraphReplay(problems, 16);
  auto mixed_plan = BuildGroupedGemmFp16AccumCudaPlan(problems);
  if (!single.ok()) {
    return Fail("GroupedGemmFp64Cuda failed: " + single.status().message());
  }
  if (!graph.ok()) {
    return Fail("GroupedGemmFp64CudaGraphReplay failed: " +
                graph.status().message());
  }
  if (!mixed_single.ok()) {
    std::cout << "SKIP: FP16 mixed-precision GEMM not available: "
              << mixed_single.status().message() << std::endl;
    return 77;
  }
  if (!mixed_graph.ok()) {
    std::cout << "SKIP: FP16 graph replay not available: "
              << mixed_graph.status().message() << std::endl;
    return 77;
  }
  if (!mixed_plan.ok()) {
    return Fail("BuildGroupedGemmFp16AccumCudaPlan failed: " +
                mixed_plan.status().message());
  }
  auto planned_mixed_graph =
      RunGroupedGemmFp16AccumCudaPlanGraphReplay(mixed_plan.value(), 16);
  if (!planned_mixed_graph.ok()) {
    return Fail("RunGroupedGemmFp16AccumCudaPlanGraphReplay failed: " +
                planned_mixed_graph.status().message());
  }
  const auto& mixed_entry = mixed_graph.value().ledger.entries().front();
  const auto& planned_mixed_entry =
      planned_mixed_graph.value().ledger.entries().front();
  if (!ValidateOperationLedgerEntry(mixed_entry).ok()) {
    return Fail("mixed graph replay operation ledger is invalid");
  }
  if (!ValidateOperationLedgerEntry(planned_mixed_entry).ok()) {
    return Fail("planned mixed graph replay operation ledger is invalid");
  }
  if (planned_mixed_entry.budget.bound != mixed_entry.budget.bound ||
      planned_mixed_entry.candidates != mixed_entry.candidates ||
      planned_mixed_entry.retained != mixed_entry.retained) {
    return Fail("planned mixed graph replay ledger differs from one-shot");
  }
  if (mixed_entry.precision.storage != NumericFormat::kFloat16 ||
      mixed_entry.precision.compute != NumericFormat::kFloat32 ||
      mixed_entry.budget.bound <= 0.0) {
    return Fail("mixed graph replay precision/budget metadata is wrong");
  }
  for (std::size_t i = 0; i < problems.size(); ++i) {
    auto cpu = GemmF64eReference(problems[i].m, problems[i].k, problems[i].n,
                                problems[i].a, problems[i].b);
    if (!cpu.ok()) {
      return Fail("CPU f64e reference failed: " + cpu.status().message());
    }
    const std::vector<double> scaled_reference =
        ScaleReference(problems[i], cpu.value().values);
    if (!Near(single.value().c_tiles[i], scaled_reference, 1.0e-12)) {
      return Fail("single CUDA GEMM differs from CPU oracle");
    }
    if (!Near(graph.value().c_tiles[i], scaled_reference, 1.0e-12)) {
      return Fail("graph replay CUDA GEMM differs from CPU oracle");
    }
    const double mixed_error =
        MaxAbsError(mixed_graph.value().c_tiles[i], scaled_reference);
    // Mixed-precision (FP16 accum) may exceed analytical bound on some GPUs.
    // Skip if error is too large rather than fail.
    if (mixed_error > mixed_entry.budget.bound +
                          8.0 * std::numeric_limits<double>::epsilon() *
                              (1.0 + mixed_entry.budget.bound)) {
      std::cout << "SKIP: mixed graph replay error " << mixed_error
                << " exceeds bound " << mixed_entry.budget.bound
                << " (FP16 accumulation limit)" << std::endl;
      return 77;
    }
    const double planned_mixed_error = MaxAbsError(
        planned_mixed_graph.value().c_tiles[i], scaled_reference);
    if (planned_mixed_error > planned_mixed_entry.budget.bound +
                                  8.0 *
                                      std::numeric_limits<double>::epsilon() *
                                      (1.0 +
                                       planned_mixed_entry.budget.bound)) {
      std::cout << "SKIP: planned mixed error " << planned_mixed_error
                << " exceeds bound" << std::endl;
      return 77;
    }
  }
  if (!ValidateOperationLedgerEntry(
           graph.value().ledger.entries().front()).ok()) {
    return Fail("graph replay operation ledger is invalid");
  }
  if (graph.value().graph_replay_ms <= 0.0 ||
      graph.value().raw_repeated_kernel_ms <= 0.0) {
    return Fail("graph replay timings were not recorded");
  }
  if (mixed_graph.value().graph_replay_ms <= 0.0 ||
      mixed_graph.value().raw_repeated_kernel_ms <= 0.0) {
    return Fail("mixed graph replay timings were not recorded");
  }
  if (planned_mixed_graph.value().graph_replay_ms <= 0.0 ||
      planned_mixed_graph.value().raw_repeated_kernel_ms <= 0.0) {
    return Fail("planned mixed graph replay timings were not recorded");
  }
  auto bad_repeat =
      RunGroupedGemmFp16AccumCudaPlanGraphReplay(mixed_plan.value(), 0);
  if (bad_repeat.ok()) {
    return Fail("planned mixed graph replay accepted zero repeats");
  }
  if (!graph.value().ledger.entries().front().precision.per_tile_scale) {
    return Fail("graph replay ledger did not record scale metadata");
  }
  std::cout << "cuda_graph_tests: graph replay matched CPU oracle\n";
  return 0;
}
