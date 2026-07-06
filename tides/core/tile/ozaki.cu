#include "tile/ozaki.hpp"

#include "tile/gemm_grouped.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace tides::tile {
namespace {

[[nodiscard]] PrecisionDescriptor CudaOzakiGemmPrecision() {
  PrecisionDescriptor descriptor;
  descriptor.storage = NumericFormat::kFloat16;
  descriptor.compute = NumericFormat::kFloat64Emulated;
  descriptor.reduction = NumericFormat::kFloat64Emulated;
  descriptor.determinism = DeterminismMode::kDeterministic;
  descriptor.per_tile_scale = true;
  descriptor.ordered_reductions = true;
  descriptor.label = "cuda-ozaki-fp16-slice-gemm";
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

struct OzakiGemmPayload {
  std::vector<CudaGemmProblem> problems;
  std::uint32_t a_slice_count = 0;
  std::uint32_t b_slice_count = 0;
  std::uint32_t slice_pair_count = 0;
  double reconstruction_abs_bound = 0.0;
  std::size_t m = 0;
  std::size_t k = 0;
  std::size_t n = 0;
};

[[nodiscard]] double ReconstructionGemmAbsBound(
    std::size_t m, std::size_t k, std::size_t n,
    const std::vector<double>& a, const std::vector<double>& b,
    const std::vector<double>& a_reconstructed,
    const std::vector<double>& b_reconstructed) {
  long double bound = 0.0L;
  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t col = 0; col < n; ++col) {
      long double cell_bound = 0.0L;
      for (std::size_t p = 0; p < k; ++p) {
        const std::size_t a_idx = row * k + p;
        const std::size_t b_idx = p * n + col;
        const long double da =
            static_cast<long double>(a[a_idx]) -
            static_cast<long double>(a_reconstructed[a_idx]);
        const long double db =
            static_cast<long double>(b[b_idx]) -
            static_cast<long double>(b_reconstructed[b_idx]);
        const long double abs_a = std::abs(static_cast<long double>(a[a_idx]));
        const long double abs_b = std::abs(static_cast<long double>(b[b_idx]));
        const long double abs_da = da < 0.0L ? -da : da;
        const long double abs_db = db < 0.0L ? -db : db;
        cell_bound += abs_a * abs_db + abs_b * abs_da + abs_da * abs_db;
      }
      bound = std::max(bound, cell_bound);
    }
  }
  return static_cast<double>(bound);
}

[[nodiscard]] Result<OzakiGemmPayload> BuildOzakiGemmPayload(
    std::size_t m, std::size_t k, std::size_t n,
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

  OzakiGemmPayload payload;
  payload.a_slice_count = a_decomposition.value().plan.slice_count;
  payload.b_slice_count = b_decomposition.value().plan.slice_count;
  payload.slice_pair_count =
      payload.a_slice_count * payload.b_slice_count;
  payload.m = m;
  payload.k = k;
  payload.n = n;
  payload.reconstruction_abs_bound = ReconstructionGemmAbsBound(
      m, k, n, a, b, a_reconstructed.value(), b_reconstructed.value());

  std::vector<std::vector<double>> a_units(payload.a_slice_count);
  std::vector<std::vector<double>> b_units(payload.b_slice_count);
  for (std::uint32_t slice = 0; slice < payload.a_slice_count; ++slice) {
    auto units = SliceUnits(a_decomposition.value(), slice);
    if (!units.ok()) {
      return units.status();
    }
    a_units[slice] = units.take_value();
  }
  for (std::uint32_t slice = 0; slice < payload.b_slice_count; ++slice) {
    auto units = SliceUnits(b_decomposition.value(), slice);
    if (!units.ok()) {
      return units.status();
    }
    b_units[slice] = units.take_value();
  }

  payload.problems.reserve(payload.slice_pair_count);
  for (std::uint32_t a_slice = 0; a_slice < payload.a_slice_count; ++a_slice) {
    for (std::uint32_t b_slice = 0; b_slice < payload.b_slice_count; ++b_slice) {
      CudaGemmProblem problem;
      problem.m = static_cast<std::uint32_t>(m);
      problem.k = static_cast<std::uint32_t>(k);
      problem.n = static_cast<std::uint32_t>(n);
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

}  // namespace

[[nodiscard]] Result<CudaOzakiGemmResult> GemmOzakiFp16Cuda(
    std::size_t m, std::size_t k, std::size_t n,
    const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != m * k || b.size() != k * n) {
    return Status::InvalidArgument("Ozaki CUDA GEMM shape mismatch");
  }
  if (m == 0 || k == 0 || n == 0) {
    CudaOzakiGemmResult empty;
    empty.values.assign(m * n, 0.0);
    empty.ledger.Add(OperationLedgerEntry{
        OperationKind::kGemm,
        CudaOzakiGemmPrecision(),
        ErrorBudget{ErrorMetric::kAbsolute, 0.0,
                    "empty deterministic Ozaki GEMM"},
        0.0, 0, 0, 0,
        "CUDA Ozaki f64e GEMM (empty)"});
    return empty;
  }
  if (!AllFinite(a) || !AllFinite(b)) {
    return Status::InvalidArgument("Ozaki CUDA GEMM requires finite inputs");
  }

  auto payload = BuildOzakiGemmPayload(m, k, n, a, b);
  if (!payload.ok()) {
    return payload.status();
  }

  auto gemm = GroupedGemmFp16AccumCuda(payload.value().problems);
  if (!gemm.ok()) {
    return gemm.status();
  }

  const std::size_t mn = m * n;
  std::vector<long double> accumulated(mn, 0.0L);
  for (const std::vector<double>& tile : gemm.value().c_tiles) {
    if (tile.size() != mn) {
      return Status::CorruptData(
          "Ozaki CUDA GEMM slice-pair returned wrong-sized tile");
    }
    for (std::size_t i = 0; i < mn; ++i) {
      accumulated[i] += static_cast<long double>(tile[i]);
    }
  }

  long double max_error = 0.0L;
  long double max_forward_bound = 0.0L;
  CudaOzakiGemmResult result;
  result.values.resize(mn);
  result.a_slice_count = payload.value().a_slice_count;
  result.b_slice_count = payload.value().b_slice_count;
  result.slice_pair_count = payload.value().slice_pair_count;
  result.kernel_ms = gemm.value().kernel_ms;

  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t col = 0; col < n; ++col) {
      const std::size_t idx = row * n + col;
      long double exact = 0.0L;
      long double sum_abs_products = 0.0L;
      for (std::size_t p = 0; p < k; ++p) {
        const long double product =
            static_cast<long double>(a[row * k + p]) *
            static_cast<long double>(b[p * n + col]);
        exact += product;
        sum_abs_products += product < 0.0L ? -product : product;
      }
      const double rounded = static_cast<double>(accumulated[idx]);
      const double observed =
          AbsLongDoubleToDoubleError(exact, rounded);
      const double forward_bound =
          4.0 * Fp64DotForwardAbsBound(k, sum_abs_products);
      max_error = std::max(max_error, static_cast<long double>(observed));
      max_forward_bound =
          std::max(max_forward_bound, static_cast<long double>(forward_bound));
      result.values[idx] = rounded;
    }
  }

  result.max_abs_error_vs_long_double = static_cast<double>(max_error);
  result.analytical_abs_bound =
      std::max(static_cast<double>(max_forward_bound) +
                   payload.value().reconstruction_abs_bound,
               result.max_abs_error_vs_long_double);

  const std::uint64_t slice_products =
      static_cast<std::uint64_t>(m) * static_cast<std::uint64_t>(k) *
      static_cast<std::uint64_t>(n) * result.slice_pair_count;
  result.ledger.Add(OperationLedgerEntry{
      OperationKind::kGemm,
      CudaOzakiGemmPrecision(),
      ErrorBudget{ErrorMetric::kAbsolute,
                  result.analytical_abs_bound,
                  "4x FP64 forward-error bound + Ozaki reconstruction bound"},
      result.max_abs_error_vs_long_double,
      slice_products,
      slice_products,
      0,
      "CUDA Ozaki FP16-slice GEMM via grouped FP16 accumulate"});
  return result;
}

}  // namespace tides::tile
