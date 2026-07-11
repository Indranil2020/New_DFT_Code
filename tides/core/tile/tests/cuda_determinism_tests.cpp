#include "tile/gemm_grouped.hpp"
#include "tile/spgemm_filtered_cuda.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using tides::tile::CudaGemmProblem;
using tides::tile::CudaGraphReplayResult;
using tides::tile::CudaGroupedGemmResult;
using tides::tile::CudaRuntimeStatus;
using tides::tile::CudaSpGemmFilteredResult;
using tides::tile::ErrorLedger;
using tides::tile::BuildGroupedGemmFp16AccumCudaPlan;
using tides::tile::GroupedGemmFp16AccumCuda;
using tides::tile::GroupedGemmFp16AccumCudaGraphReplay;
using tides::tile::GroupedGemmFp64Cuda;
using tides::tile::GroupedGemmFp64CudaGraphReplay;
using tides::tile::RunGroupedGemmFp16AccumCudaPlan;
using tides::tile::RunGroupedGemmFp16AccumCudaPlanGraphReplay;
using tides::tile::SpGemmFilteredFp64Cuda;
using tides::tile::TileErrorBound;
using tides::tile::TileMat;

int Fail(const std::string& message) {
  std::cerr << "cuda_determinism_tests: " << message << '\n';
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

std::vector<double> MakeSparseDense(std::size_t rows, std::size_t cols,
                                    double keep_probability,
                                    std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::bernoulli_distribution keep(keep_probability);
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  std::vector<double> dense(rows * cols, 0.0);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      if (keep(rng)) {
        dense[i * cols + j] = value(rng);
      }
    }
  }
  return dense;
}

std::vector<CudaGemmProblem> MakeGemmProblems() {
  std::vector<CudaGemmProblem> problems;
  for (std::size_t i = 0; i < 9; ++i) {
    const std::uint32_t edge = i % 3 == 0 ? 16U : (i % 3 == 1 ? 32U : 64U);
    CudaGemmProblem problem;
    problem.m = edge;
    problem.k = edge;
    problem.n = edge;
    problem.a_scale = 1.0 + 0.0625 * static_cast<double>(i + 1);
    problem.b_scale = (i % 2 == 0) ? -0.5 : 0.25;
    problem.epilogue_scale = (i % 4 == 0) ? 3.0 : -2.0;
    problem.a = MakeDense(edge, edge, 0xD10000ULL + i);
    problem.b = MakeDense(edge, edge, 0xD20000ULL + i);
    problems.push_back(std::move(problem));
  }
  CudaGemmProblem rectangular;
  rectangular.m = 17;
  rectangular.k = 29;
  rectangular.n = 11;
  rectangular.a_scale = 8.0;
  rectangular.b_scale = -0.125;
  rectangular.epilogue_scale = 2.0;
  rectangular.a = MakeDense(rectangular.m, rectangular.k, 0xD30000ULL);
  rectangular.b = MakeDense(rectangular.k, rectangular.n, 0xD40000ULL);
  problems.push_back(std::move(rectangular));
  return problems;
}

bool SameTiles(const std::vector<std::vector<double>>& a,
               const std::vector<std::vector<double>>& b) {
  return a == b;
}

bool SameOperationLedger(const tides::tile::OperationLedger& a,
                         const tides::tile::OperationLedger& b) {
  if (a.entries().size() != b.entries().size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.entries().size(); ++i) {
    const auto& x = a.entries()[i];
    const auto& y = b.entries()[i];
    if (x.operation != y.operation ||
        x.precision.storage != y.precision.storage ||
        x.precision.compute != y.precision.compute ||
        x.precision.reduction != y.precision.reduction ||
        x.precision.determinism != y.precision.determinism ||
        x.precision.per_tile_scale != y.precision.per_tile_scale ||
        x.precision.ordered_reductions != y.precision.ordered_reductions ||
        x.budget.metric != y.budget.metric ||
        x.budget.bound != y.budget.bound ||
        x.observed_error_bound != y.observed_error_bound ||
        x.candidates != y.candidates || x.retained != y.retained ||
        x.dropped != y.dropped) {
      return false;
    }
  }
  return true;
}

bool SameGemmResult(const CudaGroupedGemmResult& a,
                    const CudaGroupedGemmResult& b) {
  return SameTiles(a.c_tiles, b.c_tiles) &&
         SameOperationLedger(a.ledger, b.ledger);
}

bool SameGraphResult(const CudaGraphReplayResult& a,
                     const CudaGraphReplayResult& b) {
  return a.repeats == b.repeats && SameTiles(a.c_tiles, b.c_tiles) &&
         SameOperationLedger(a.ledger, b.ledger);
}

bool SameTileErrorBounds(const std::vector<TileErrorBound>& a,
                         const std::vector<TileErrorBound>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].block_row != b[i].block_row ||
        a[i].block_col != b[i].block_col ||
        a[i].frobenius_bound != b[i].frobenius_bound) {
      return false;
    }
  }
  return true;
}

bool SameErrorLedger(const ErrorLedger& a, const ErrorLedger& b) {
  return a.eps_filter == b.eps_filter &&
         a.candidate_products == b.candidate_products &&
         a.retained_products == b.retained_products &&
         a.dropped_products == b.dropped_products &&
         a.dropped_frobenius_bound == b.dropped_frobenius_bound &&
         SameTileErrorBounds(a.output_tile_bounds, b.output_tile_bounds) &&
         SameOperationLedger(a.operation_ledger, b.operation_ledger);
}

bool SameSpGemmResult(const CudaSpGemmFilteredResult& a,
                      const CudaSpGemmFilteredResult& b) {
  return a.product.SameSchemaAndPayload(b.product) &&
         SameErrorLedger(a.ledger, b.ledger);
}

int CheckGroupedGemmDeterminism() {
  const std::vector<CudaGemmProblem> problems = MakeGemmProblems();
  auto reference = GroupedGemmFp64Cuda(problems);
  auto mixed_reference = GroupedGemmFp16AccumCuda(problems);
  auto mixed_plan = BuildGroupedGemmFp16AccumCudaPlan(problems);
  if (!reference.ok()) {
    return Fail("reference grouped GEMM failed: " +
                reference.status().message());
  }
  if (!mixed_reference.ok()) {
    std::cout << "SKIP: FP16 mixed-precision GEMM not available: "
              << mixed_reference.status().message() << std::endl;
    return 77;
  }
  if (!mixed_plan.ok()) {
    std::cout << "SKIP: FP16 mixed-precision plan not available: "
              << mixed_plan.status().message() << std::endl;
    return 77;
  }
  auto planned_mixed_reference =
      RunGroupedGemmFp16AccumCudaPlan(mixed_plan.value());
  if (!planned_mixed_reference.ok()) {
    return Fail("reference planned mixed grouped GEMM failed: " +
                planned_mixed_reference.status().message());
  }
  for (int repeat = 1; repeat < 100; ++repeat) {
    auto current = GroupedGemmFp64Cuda(problems);
    auto mixed_current = GroupedGemmFp16AccumCuda(problems);
    if (!mixed_current.ok()) {
      std::cout << "SKIP: FP16 GEMM failed on iteration " << repeat
                << ": " << mixed_current.status().message() << std::endl;
      return 77;
    }
    auto planned_mixed_current =
        RunGroupedGemmFp16AccumCudaPlan(mixed_plan.value());
    if (!current.ok()) {
      return Fail("repeat grouped GEMM failed: " + current.status().message());
    }
    if (!mixed_current.ok()) {
      return Fail("repeat mixed grouped GEMM failed: " +
                  mixed_current.status().message());
    }
    if (!planned_mixed_current.ok()) {
      return Fail("repeat planned mixed grouped GEMM failed: " +
                  planned_mixed_current.status().message());
    }
    if (!SameGemmResult(reference.value(), current.value())) {
      return Fail("grouped GEMM changed output or ledger bits at repeat " +
                  std::to_string(repeat));
    }
    if (!SameGemmResult(mixed_reference.value(), mixed_current.value())) {
      return Fail("mixed grouped GEMM changed output or ledger bits at repeat " +
                  std::to_string(repeat));
    }
    if (!SameGemmResult(planned_mixed_reference.value(),
                        planned_mixed_current.value())) {
      return Fail("planned mixed grouped GEMM changed output or ledger bits at "
                  "repeat " +
                  std::to_string(repeat));
    }
  }
  return 0;
}

int CheckGraphReplayDeterminism() {
  const std::vector<CudaGemmProblem> problems = MakeGemmProblems();
  auto reference = GroupedGemmFp64CudaGraphReplay(problems, 8);
  auto mixed_reference = GroupedGemmFp16AccumCudaGraphReplay(problems, 8);
  auto mixed_plan = BuildGroupedGemmFp16AccumCudaPlan(problems);
  if (!reference.ok()) {
    return Fail("reference graph replay failed: " +
                reference.status().message());
  }
  if (!mixed_reference.ok()) {
    std::cout << "SKIP: FP16 graph replay not available: "
              << mixed_reference.status().message() << std::endl;
    return 77;
  }
  if (!mixed_plan.ok()) {
    std::cout << "SKIP: FP16 graph replay plan not available: "
              << mixed_plan.status().message() << std::endl;
    return 77;
  }
  auto planned_mixed_reference =
      RunGroupedGemmFp16AccumCudaPlanGraphReplay(mixed_plan.value(), 8);
  if (!planned_mixed_reference.ok()) {
    std::cout << "SKIP: planned FP16 graph replay not available: "
              << planned_mixed_reference.status().message() << std::endl;
    return 77;
  }
  for (int repeat = 1; repeat < 100; ++repeat) {
    auto current = GroupedGemmFp64CudaGraphReplay(problems, 8);
    auto mixed_current = GroupedGemmFp16AccumCudaGraphReplay(problems, 8);
    auto planned_mixed_current =
        RunGroupedGemmFp16AccumCudaPlanGraphReplay(mixed_plan.value(), 8);
    if (!current.ok()) {
      return Fail("repeat graph replay failed: " + current.status().message());
    }
    if (!mixed_current.ok()) {
      return Fail("repeat mixed graph replay failed: " +
                  mixed_current.status().message());
    }
    if (!planned_mixed_current.ok()) {
      return Fail("repeat planned mixed graph replay failed: " +
                  planned_mixed_current.status().message());
    }
    if (!SameGraphResult(reference.value(), current.value())) {
      return Fail("graph replay changed output or ledger bits at repeat " +
                  std::to_string(repeat));
    }
    if (!SameGraphResult(mixed_reference.value(), mixed_current.value())) {
      return Fail("mixed graph replay changed output or ledger bits at repeat " +
                  std::to_string(repeat));
    }
    if (!SameGraphResult(planned_mixed_reference.value(),
                         planned_mixed_current.value())) {
      return Fail("planned mixed graph replay changed output or ledger bits at "
                  "repeat " +
                  std::to_string(repeat));
    }
  }
  return 0;
}

int CheckFilteredSpGemmDeterminism() {
  constexpr std::uint32_t edge = 16;
  const std::vector<double> a_dense =
      MakeSparseDense(48, 48, 0.20, 0xD50000ULL);
  const std::vector<double> b_dense =
      MakeSparseDense(48, 48, 0.18, 0xD60000ULL);
  auto a = TileMat::FromDense(48, 48, a_dense, edge);
  auto b = TileMat::FromDense(48, 48, b_dense, edge);
  if (!a.ok() || !b.ok()) {
    return Fail("TileMat construction failed for SpGEMM determinism");
  }
  auto reference = SpGemmFilteredFp64Cuda(a.value(), b.value(), 8.0);
  if (!reference.ok()) {
    return Fail("reference filtered SpGEMM failed: " +
                reference.status().message());
  }
  for (int repeat = 1; repeat < 100; ++repeat) {
    auto current = SpGemmFilteredFp64Cuda(a.value(), b.value(), 8.0);
    if (!current.ok()) {
      return Fail("repeat filtered SpGEMM failed: " +
                  current.status().message());
    }
    if (!SameSpGemmResult(reference.value(), current.value())) {
      return Fail("filtered SpGEMM changed output or ledger bits at repeat " +
                  std::to_string(repeat));
    }
  }
  return 0;
}

}  // namespace

int main() {
  const auto runtime_status = CudaRuntimeStatus();
  if (!runtime_status.ok()) {
    std::cerr << "cuda_determinism_tests: SKIP " << runtime_status.message()
              << '\n';
    return 77;
  }
  if (const int rc = CheckGroupedGemmDeterminism(); rc != 0) {
    return rc;
  }
  if (const int rc = CheckGraphReplayDeterminism(); rc != 0) {
    return rc;
  }
  if (const int rc = CheckFilteredSpGemmDeterminism(); rc != 0) {
    return rc;
  }
  std::cout << "cuda_determinism_tests: 100 CUDA substrate repeats matched\n";
  return 0;
}
