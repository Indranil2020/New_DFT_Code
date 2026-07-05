#include "tile/f64e_reference.hpp"
#include "tile/reduce_f64e.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using tides::tile::DeterminismMode;
using tides::tile::DotF64eCuda;
using tides::tile::DotF64eReference;
using tides::tile::ErrorMetric;
using tides::tile::NumericFormat;
using tides::tile::OperationKind;
using tides::tile::SumF64eCuda;
using tides::tile::TraceF64eCuda;
using tides::tile::ValidateOperationLedgerEntry;

bool IsCudaRuntimeUnavailable(const tides::Status& status) {
  return status.message().find("cudaGetDeviceCount") != std::string::npos ||
         status.message().find("CUDA") != std::string::npos;
}

int Fail(const std::string& message) {
  std::cerr << "cuda_reduce_f64e_tests: " << message << '\n';
  return 1;
}

std::vector<double> MakeDynamicRange(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> mantissa(0.5, 1.0);
  std::uniform_int_distribution<int> exponent(-9, 9);
  std::bernoulli_distribution sign(0.5);
  std::vector<double> values(n, 0.0);
  for (double& value : values) {
    value = std::ldexp(mantissa(rng), exponent(rng) * 3);
    if (sign(rng)) {
      value = -value;
    }
  }
  return values;
}

bool CheckLedger(const tides::tile::CudaF64eReductionResult& result,
                 OperationKind operation) {
  if (result.ledger.entries().size() != 1) {
    return false;
  }
  const auto& entry = result.ledger.entries().front();
  return ValidateOperationLedgerEntry(entry).ok() &&
         entry.operation == operation &&
         entry.precision.compute == NumericFormat::kFloat64Emulated &&
         entry.precision.reduction == NumericFormat::kFloat64Emulated &&
         entry.precision.determinism == DeterminismMode::kDeterministic &&
         entry.budget.metric == ErrorMetric::kAbsolute &&
         entry.observed_error_bound <=
             entry.budget.bound +
                 8.0 * std::numeric_limits<double>::epsilon() *
                     (1.0 + entry.budget.bound);
}

int CheckDot() {
  const std::vector<double> a = MakeDynamicRange(4099, 0xF64001ULL);
  const std::vector<double> b = MakeDynamicRange(4099, 0xF64002ULL);
  auto gpu = DotF64eCuda(a, b);
  auto cpu = DotF64eReference(a, b);
  if (!gpu.ok()) {
    return Fail("DotF64eCuda failed: " + gpu.status().message());
  }
  if (!cpu.ok()) {
    return Fail("DotF64eReference failed: " + cpu.status().message());
  }
  const double abs_error = std::abs(gpu.value().value - cpu.value().value);
  if (abs_error > gpu.value().analytical_abs_bound +
                      8.0 * std::numeric_limits<double>::epsilon() *
                          (1.0 + std::abs(cpu.value().value))) {
    return Fail("CUDA f64e dot exceeded analytical absolute bound");
  }
  if (!CheckLedger(gpu.value(), OperationKind::kDot)) {
    return Fail("CUDA f64e dot ledger is invalid");
  }
  return 0;
}

int CheckCancellationSum() {
  std::vector<double> values;
  values.reserve(6144);
  for (std::size_t i = 0; i < 2048; ++i) {
    values.push_back(1.0e16);
    values.push_back(1.0);
    values.push_back(-1.0e16);
  }
  auto gpu = SumF64eCuda(values);
  if (!gpu.ok()) {
    return Fail("SumF64eCuda failed: " + gpu.status().message());
  }
  if (std::abs(gpu.value().value - 2048.0) > 1.0e-9) {
    return Fail("CUDA f64e sum did not preserve small cancellation terms");
  }
  if (!CheckLedger(gpu.value(), OperationKind::kReduction)) {
    return Fail("CUDA f64e sum ledger is invalid");
  }
  return 0;
}

int CheckTrace() {
  constexpr std::size_t edge = 258;
  std::vector<double> matrix(edge * edge, 0.0);
  double expected = 0.0;
  for (std::size_t i = 0; i < edge; ++i) {
    double value = 0.0;
    if (i % 3 == 0) {
      value = 1.0e12;
    } else if (i % 3 == 1) {
      value = 0.25;
    } else {
      value = -1.0e12;
    }
    matrix[i * edge + i] = value;
    expected += value;
  }
  auto gpu = TraceF64eCuda(edge, edge, matrix);
  if (!gpu.ok()) {
    return Fail("TraceF64eCuda failed: " + gpu.status().message());
  }
  const double rel_error =
      std::abs(gpu.value().value - expected) / (1.0 + std::abs(expected));
  if (rel_error > 1.0e-13) {
    return Fail("CUDA f64e trace exceeded 1e-13 relative target");
  }
  if (!CheckLedger(gpu.value(), OperationKind::kTrace)) {
    return Fail("CUDA f64e trace ledger is invalid");
  }
  return 0;
}

int CheckValidationFailures() {
  auto mismatch = DotF64eCuda({1.0, 2.0}, {3.0});
  if (mismatch.ok()) {
    return Fail("CUDA f64e dot accepted mismatched vector lengths");
  }
  auto nonfinite = SumF64eCuda({1.0, std::numeric_limits<double>::infinity()});
  if (nonfinite.ok()) {
    return Fail("CUDA f64e sum accepted a non-finite input");
  }
  auto bad_trace = TraceF64eCuda(2, 3, {1.0, 2.0});
  if (bad_trace.ok()) {
    return Fail("CUDA f64e trace accepted a bad payload size");
  }
  auto empty = SumF64eCuda({});
  if (!empty.ok() || empty.value().value != 0.0 ||
      empty.value().ledger.entries().front().candidates != 0) {
    return Fail("CUDA f64e empty sum contract is wrong");
  }
  return 0;
}

}  // namespace

int main() {
  auto smoke = DotF64eCuda({1.0}, {1.0});
  if (!smoke.ok() && IsCudaRuntimeUnavailable(smoke.status())) {
    std::cerr << "cuda_reduce_f64e_tests: SKIP " << smoke.status().message()
              << '\n';
    return 77;
  }
  if (!smoke.ok()) {
    return Fail("CUDA f64e smoke dot failed: " + smoke.status().message());
  }
  if (const int rc = CheckDot(); rc != 0) {
    return rc;
  }
  if (const int rc = CheckCancellationSum(); rc != 0) {
    return rc;
  }
  if (const int rc = CheckTrace(); rc != 0) {
    return rc;
  }
  if (const int rc = CheckValidationFailures(); rc != 0) {
    return rc;
  }
  std::cout << "cuda_reduce_f64e_tests: CUDA f64e reductions passed "
               "analytical cancellation tests\n";
  return 0;
}
