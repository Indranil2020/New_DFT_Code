#include "tile/reduce_f64e.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "tile/f64e_reference.hpp"
#include "tile/gemm_grouped.hpp"
#include "tile/ozaki.hpp"

namespace tides::tile {
namespace {

struct OzakiDotPayload {
  std::vector<CudaGemmProblem> problems;
  std::uint32_t slice_pair_count = 0;
  double reconstruction_abs_bound = 0.0;
};

[[nodiscard]] PrecisionDescriptor CudaOzakiReductionPrecision() {
  PrecisionDescriptor descriptor;
  descriptor.storage = NumericFormat::kFloat16;
  descriptor.compute = NumericFormat::kFloat64Emulated;
  descriptor.reduction = NumericFormat::kFloat64Emulated;
  descriptor.determinism = DeterminismMode::kDeterministic;
  descriptor.per_tile_scale = true;
  descriptor.ordered_reductions = true;
  descriptor.label = "cuda-ozaki-fp16-slice-grouped-gemm-reduction";
  return descriptor;
}

[[nodiscard]] bool AllFinite(const std::vector<double>& values) {
  for (const double value : values) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] double ReconstructionDotAbsBound(
    const std::vector<double>& a, const std::vector<double>& b,
    const std::vector<double>& a_reconstructed,
    const std::vector<double>& b_reconstructed) {
  long double bound = 0.0L;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const long double da = static_cast<long double>(a[i]) -
                           static_cast<long double>(a_reconstructed[i]);
    const long double db = static_cast<long double>(b[i]) -
                           static_cast<long double>(b_reconstructed[i]);
    const long double abs_a = std::abs(static_cast<long double>(a[i]));
    const long double abs_b = std::abs(static_cast<long double>(b[i]));
    const long double abs_da = da < 0.0L ? -da : da;
    const long double abs_db = db < 0.0L ? -db : db;
    bound += abs_a * abs_db + abs_b * abs_da + abs_da * abs_db;
  }
  return static_cast<double>(bound);
}

[[nodiscard]] Result<std::vector<double>> SliceUnits(
    const OzakiDecomposition& decomposition, std::uint32_t slice) {
  if (slice >= decomposition.plan.slice_count) {
    return Status::InvalidArgument("Ozaki slice index out of range");
  }
  const double quantum = decomposition.plan.slice_quanta[slice];
  if (quantum == 0.0 || !std::isfinite(quantum)) {
    return Status::InvalidArgument("Ozaki slice quantum must be finite");
  }
  std::vector<double> units(decomposition.value_count, 0.0);
  const std::size_t offset =
      static_cast<std::size_t>(slice) * decomposition.value_count;
  for (std::size_t i = 0; i < decomposition.value_count; ++i) {
    const double unit = decomposition.slices[offset + i] / quantum;
    if (!std::isfinite(unit) || std::abs(unit) > 2048.0) {
      return Status::OutOfRange(
          "Ozaki slice unit is outside exact FP16 integer range");
    }
    units[i] = unit;
  }
  return units;
}

[[nodiscard]] Result<OzakiDotPayload> BuildOzakiDotPayload(
    const std::vector<double>& a, const std::vector<double>& b) {
  auto a_decomposition = DecomposeOzakiFp16Reference(a);
  if (!a_decomposition.ok()) {
    return a_decomposition.status();
  }
  auto b_decomposition = DecomposeOzakiFp16Reference(b);
  if (!b_decomposition.ok()) {
    return b_decomposition.status();
  }
  auto a_reconstructed = ReconstructOzakiSlices(
      a_decomposition.value().plan, a.size(), a_decomposition.value().slices);
  if (!a_reconstructed.ok()) {
    return a_reconstructed.status();
  }
  auto b_reconstructed = ReconstructOzakiSlices(
      b_decomposition.value().plan, b.size(), b_decomposition.value().slices);
  if (!b_reconstructed.ok()) {
    return b_reconstructed.status();
  }

  OzakiDotPayload payload;
  payload.slice_pair_count = a_decomposition.value().plan.slice_count *
                             b_decomposition.value().plan.slice_count;
  payload.reconstruction_abs_bound = ReconstructionDotAbsBound(
      a, b, a_reconstructed.value(), b_reconstructed.value());
  payload.problems.reserve(payload.slice_pair_count);

  std::vector<std::vector<double>> a_units(
      a_decomposition.value().plan.slice_count);
  std::vector<std::vector<double>> b_units(
      b_decomposition.value().plan.slice_count);
  for (std::uint32_t slice = 0;
       slice < a_decomposition.value().plan.slice_count; ++slice) {
    auto units = SliceUnits(a_decomposition.value(), slice);
    if (!units.ok()) {
      return units.status();
    }
    a_units[slice] = units.take_value();
  }
  for (std::uint32_t slice = 0;
       slice < b_decomposition.value().plan.slice_count; ++slice) {
    auto units = SliceUnits(b_decomposition.value(), slice);
    if (!units.ok()) {
      return units.status();
    }
    b_units[slice] = units.take_value();
  }

  for (std::uint32_t a_slice = 0;
       a_slice < a_decomposition.value().plan.slice_count; ++a_slice) {
    for (std::uint32_t b_slice = 0;
         b_slice < b_decomposition.value().plan.slice_count; ++b_slice) {
      CudaGemmProblem problem;
      problem.m = 1;
      problem.k = static_cast<std::uint32_t>(a.size());
      problem.n = 1;
      problem.a_scale =
          a_decomposition.value().plan.slice_quanta[a_slice];
      problem.b_scale =
          b_decomposition.value().plan.slice_quanta[b_slice];
      problem.epilogue_scale = 1.0;
      problem.a = a_units[a_slice];
      problem.b = b_units[b_slice];
      payload.problems.push_back(std::move(problem));
    }
  }
  return payload;
}

[[nodiscard]] Result<CudaF64eReductionResult> BuildReductionResult(
    const CudaGroupedGemmResult& gemm_result,
    const OzakiDotPayload& payload, long double exact,
    std::size_t original_terms, OperationKind operation, const char* note) {
  std::vector<double> slice_pair_values;
  slice_pair_values.reserve(gemm_result.c_tiles.size());
  for (const std::vector<double>& tile : gemm_result.c_tiles) {
    if (tile.size() != 1) {
      return Status::CorruptData(
          "Ozaki reduction grouped GEMM returned non-scalar tile");
    }
    slice_pair_values.push_back(tile.front());
  }

  auto final_sum = SumF64eReference(
      slice_pair_values, operation,
      "CPU f64e final aggregation of CUDA Ozaki slice-pair reductions");
  if (!final_sum.ok()) {
    return final_sum.status();
  }

  const auto& gemm_entry = gemm_result.ledger.entries().front();
  const double gemm_abs_bound =
      gemm_entry.budget.bound *
      static_cast<double>(std::max<std::size_t>(1, gemm_result.c_tiles.size()));
  const double final_rounding =
      AbsLongDoubleToDoubleError(exact, static_cast<double>(exact));
  const double observed =
      AbsLongDoubleToDoubleError(exact, final_sum.value().value);
  const double analytical_bound =
      gemm_abs_bound + payload.reconstruction_abs_bound +
      final_sum.value().rounding_abs_error + final_rounding;

  CudaF64eReductionResult result;
  result.value = final_sum.value().value;
  result.abs_error_vs_long_double = observed;
  result.analytical_abs_bound = std::max(analytical_bound, observed);
  result.kernel_ms = gemm_result.kernel_ms;
  const std::uint64_t candidates =
      static_cast<std::uint64_t>(original_terms) *
      static_cast<std::uint64_t>(payload.slice_pair_count);
  result.ledger.Add(OperationLedgerEntry{
      operation,
      CudaOzakiReductionPrecision(),
      ErrorBudget{
          ErrorMetric::kAbsolute,
          result.analytical_abs_bound,
          "Ozaki reconstruction, mixed grouped-GEMM, and final f64e sum bound"},
      result.abs_error_vs_long_double,
      candidates,
      candidates,
      0,
      note});
  return result;
}

[[nodiscard]] Result<CudaF64eReductionResult> DotF64eCudaImpl(
    const std::vector<double>& a, const std::vector<double>& b,
    OperationKind operation, const char* note) {
  long double exact = 0.0L;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const long double product =
        static_cast<long double>(a[i]) * static_cast<long double>(b[i]);
    if (!std::isfinite(static_cast<double>(product))) {
      return Status::OutOfRange(
          "CUDA f64e dot product is outside finite FP64 range");
    }
    exact += product;
  }

  auto payload = BuildOzakiDotPayload(a, b);
  if (!payload.ok()) {
    return payload.status();
  }
  auto gemm = GroupedGemmFp16AccumCuda(payload.value().problems);
  if (!gemm.ok()) {
    return gemm.status();
  }
  return BuildReductionResult(gemm.value(), payload.value(), exact, a.size(),
                              operation, note);
}

[[nodiscard]] Result<CudaF64eReductionResult> EmptyReduction(
    OperationKind operation, const char* note) {
  CudaF64eReductionResult result;
  result.ledger.Add(OperationLedgerEntry{
      operation,
      CudaOzakiReductionPrecision(),
      ErrorBudget{ErrorMetric::kAbsolute, 0.0,
                  "empty deterministic f64e reduction"},
      0.0,
      0,
      0,
      0,
      note});
  return result;
}

}  // namespace

Result<CudaF64eReductionResult> DotF64eCuda(
    const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) {
    return Status::InvalidArgument("CUDA f64e dot shape mismatch");
  }
  if (a.empty()) {
    return EmptyReduction(OperationKind::kDot, "CUDA Ozaki f64e dot reduction");
  }
  if (!AllFinite(a) || !AllFinite(b)) {
    return Status::InvalidArgument("CUDA f64e dot requires finite inputs");
  }
  return DotF64eCudaImpl(a, b, OperationKind::kDot,
                         "CUDA Ozaki f64e dot via grouped GEMM");
}

Result<CudaF64eReductionResult> SumF64eCuda(
    const std::vector<double>& values) {
  if (values.empty()) {
    return EmptyReduction(OperationKind::kReduction,
                          "CUDA Ozaki f64e linear sum reduction");
  }
  if (!AllFinite(values)) {
    return Status::InvalidArgument("CUDA f64e sum requires finite inputs");
  }
  std::vector<double> ones(values.size(), 1.0);
  return DotF64eCudaImpl(values, ones, OperationKind::kReduction,
                         "CUDA Ozaki f64e sum via grouped GEMM");
}

Result<CudaF64eReductionResult> TraceF64eCuda(
    std::size_t rows, std::size_t cols, const std::vector<double>& values) {
  if (values.size() != rows * cols) {
    return Status::InvalidArgument("CUDA f64e trace shape mismatch");
  }
  const std::size_t diagonal = std::min(rows, cols);
  if (diagonal == 0) {
    return EmptyReduction(OperationKind::kTrace,
                          "CUDA Ozaki f64e trace reduction");
  }
  if (!AllFinite(values)) {
    return Status::InvalidArgument("CUDA f64e trace requires finite inputs");
  }
  std::vector<double> diagonal_values(diagonal, 0.0);
  for (std::size_t i = 0; i < diagonal; ++i) {
    diagonal_values[i] = values[i * cols + i];
  }
  std::vector<double> ones(diagonal, 1.0);
  return DotF64eCudaImpl(diagonal_values, ones, OperationKind::kTrace,
                         "CUDA Ozaki f64e trace via grouped GEMM");
}

}  // namespace tides::tile
