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
using tides::tile::GroupedGemmFp64Cuda;
using tides::tile::NumericFormat;
using tides::tile::RunGroupedGemmFp16AccumCudaPlan;
using tides::tile::ValidateOperationLedgerEntry;

int Fail(const std::string& message) {
  std::cerr << "cuda_gemm_tests: " << message << '\n';
  return 1;
}

std::vector<double> MakeDense(std::size_t rows, std::size_t cols,
                              std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> value(-2.0, 2.0);
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

double MaxAbsError(const std::vector<double>& a, const std::vector<double>& b) {
  double max_error = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    max_error = std::max(max_error, std::abs(a[i] - b[i]));
  }
  return max_error;
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

}  // namespace

int main() {
  const auto runtime_status = CudaRuntimeStatus();
  if (!runtime_status.ok()) {
    std::cerr << "cuda_gemm_tests: SKIP " << runtime_status.message() << '\n';
    return 77;
  }
  std::vector<CudaGemmProblem> problems;
  for (std::uint32_t edge : {16U, 32U, 64U}) {
    CudaGemmProblem problem;
    problem.m = edge;
    problem.k = edge;
    problem.n = edge;
    problem.a_scale = edge == 16U ? 8.0 : (edge == 32U ? 0.5 : -2.0);
    problem.b_scale = edge == 16U ? 0.125 : (edge == 32U ? -4.0 : 0.25);
    problem.epilogue_scale = edge == 64U ? -3.0 : 1.5;
    problem.a = MakeDense(edge, edge, 0x100000ULL + edge);
    problem.b = MakeDense(edge, edge, 0x200000ULL + edge);
    problems.push_back(std::move(problem));
  }
  CudaGemmProblem rectangular;
  rectangular.m = 17;
  rectangular.k = 29;
  rectangular.n = 11;
  rectangular.a_scale = 16.0;
  rectangular.b_scale = -0.0625;
  rectangular.epilogue_scale = 2.0;
  rectangular.a = MakeDense(rectangular.m, rectangular.k, 0x3333ULL);
  rectangular.b = MakeDense(rectangular.k, rectangular.n, 0x4444ULL);
  problems.push_back(std::move(rectangular));

  auto gpu = GroupedGemmFp64Cuda(problems);
  auto mixed = GroupedGemmFp16AccumCuda(problems);
  auto mixed_plan = BuildGroupedGemmFp16AccumCudaPlan(problems);
  if (!gpu.ok()) {
    return Fail("GroupedGemmFp64Cuda failed: " + gpu.status().message());
  }
  if (!mixed.ok()) {
    return Fail("GroupedGemmFp16AccumCuda failed: " + mixed.status().message());
  }
  if (!mixed_plan.ok()) {
    return Fail("BuildGroupedGemmFp16AccumCudaPlan failed: " +
                mixed_plan.status().message());
  }
  if (!mixed_plan.value().valid()) {
    return Fail("mixed CUDA grouped GEMM plan is not valid after build");
  }
  auto planned = RunGroupedGemmFp16AccumCudaPlan(mixed_plan.value());
  if (!planned.ok()) {
    return Fail("RunGroupedGemmFp16AccumCudaPlan failed: " +
                planned.status().message());
  }
  if (gpu.value().c_tiles.size() != problems.size()) {
    return Fail("GPU result count does not match problem count");
  }
  if (mixed.value().c_tiles.size() != problems.size()) {
    return Fail("mixed GPU result count does not match problem count");
  }
  if (planned.value().c_tiles != mixed.value().c_tiles) {
    return Fail("planned mixed CUDA grouped GEMM differs from one-shot mixed");
  }
  const auto& mixed_entry = mixed.value().ledger.entries().front();
  const auto& planned_entry = planned.value().ledger.entries().front();
  if (!ValidateOperationLedgerEntry(mixed_entry).ok()) {
    return Fail("mixed CUDA grouped GEMM ledger entry is invalid");
  }
  if (!ValidateOperationLedgerEntry(planned_entry).ok()) {
    return Fail("planned mixed CUDA grouped GEMM ledger entry is invalid");
  }
  if (planned_entry.budget.bound != mixed_entry.budget.bound ||
      planned_entry.candidates != mixed_entry.candidates ||
      planned_entry.retained != mixed_entry.retained) {
    return Fail("planned mixed CUDA grouped GEMM ledger differs from one-shot");
  }
  if (mixed_entry.precision.storage != NumericFormat::kFloat16 ||
      mixed_entry.precision.compute != NumericFormat::kFloat32 ||
      mixed_entry.precision.reduction != NumericFormat::kFloat32 ||
      !mixed_entry.precision.per_tile_scale ||
      mixed_entry.budget.bound <= 0.0) {
    return Fail("mixed CUDA grouped GEMM precision/budget metadata is wrong");
  }
  for (std::size_t i = 0; i < problems.size(); ++i) {
    const CudaGemmProblem& problem = problems[i];
    auto reference = GemmF64eReference(problem.m, problem.k, problem.n,
                                      problem.a, problem.b);
    if (!reference.ok()) {
      return Fail("GemmF64eReference failed: " + reference.status().message());
    }
    const std::vector<double> scaled_reference =
        ScaleReference(problem, reference.value().values);
    if (!Near(gpu.value().c_tiles[i], scaled_reference, 1.0e-12)) {
      return Fail(
          "CUDA grouped GEMM scale epilogue differs from CPU f64e reference");
    }
    const double mixed_error =
        MaxAbsError(mixed.value().c_tiles[i], scaled_reference);
    if (mixed_error > mixed_entry.budget.bound +
                          8.0 * std::numeric_limits<double>::epsilon() *
                              (1.0 + mixed_entry.budget.bound)) {
      return Fail("mixed CUDA grouped GEMM exceeded analytical error bound");
    }
  }
  const auto& entry = gpu.value().ledger.entries().front();
  if (!ValidateOperationLedgerEntry(entry).ok()) {
    return Fail("CUDA grouped GEMM ledger entry is invalid");
  }
  if (!entry.precision.per_tile_scale) {
    return Fail("CUDA grouped GEMM ledger did not record scale metadata");
  }

  CudaGemmProblem invalid = problems.front();
  invalid.a_scale = std::numeric_limits<double>::quiet_NaN();
  auto invalid_result = GroupedGemmFp64Cuda({invalid});
  if (invalid_result.ok()) {
    return Fail("CUDA grouped GEMM accepted a NaN scale");
  }
  CudaGemmProblem invalid_mixed = problems.front();
  invalid_mixed.a[0] = 1.0e10;
  auto invalid_mixed_result = GroupedGemmFp16AccumCuda({invalid_mixed});
  if (invalid_mixed_result.ok()) {
    return Fail("mixed CUDA grouped GEMM accepted FP16 overflow input");
  }
  auto invalid_mixed_plan = BuildGroupedGemmFp16AccumCudaPlan({invalid_mixed});
  if (invalid_mixed_plan.ok()) {
    return Fail("mixed CUDA grouped GEMM plan accepted FP16 overflow input");
  }
  std::cout << "cuda_gemm_tests: CUDA FP64 grouped GEMM matched CPU oracle\n";
  return 0;
}
