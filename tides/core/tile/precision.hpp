#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "common/status.hpp"

namespace tides::tile {

enum class NumericFormat : std::uint32_t {
  kUnknown = 0,
  kFloat64,
  kFloat64Emulated,
  kFloat32,
  kTensorFloat32,
  kBFloat16,
  kFloat16,
  kFloat8,
};

enum class OperationKind : std::uint32_t {
  kUnknown = 0,
  kTileMatCreate,
  kDenseRoundTrip,
  kAxpy,
  kSpGemmFiltered,
  kDot,
  kGemm,
  kTrace,
  kNorm,
  kSerialization,
  kPrecisionTransform,
  kReduction,
  kPoissonSolve,
  kXcFunctional,
};

enum class DeterminismMode : std::uint32_t {
  kDeterministic = 0,
  kFastNonDeterministic,
};

enum class ErrorMetric : std::uint32_t {
  kNone = 0,
  kAbsolute,
  kRelative,
  kFrobenius,
  kUlp,
};

struct PrecisionDescriptor {
  NumericFormat storage = NumericFormat::kFloat64;
  NumericFormat compute = NumericFormat::kFloat64;
  NumericFormat reduction = NumericFormat::kFloat64;
  DeterminismMode determinism = DeterminismMode::kDeterministic;
  bool per_tile_scale = false;
  bool ordered_reductions = true;
  const char* label = "fp64-reference";
};

struct ErrorBudget {
  ErrorMetric metric = ErrorMetric::kNone;
  double bound = 0.0;
  const char* reason = "";
};

struct OperationLedgerEntry {
  OperationKind operation = OperationKind::kUnknown;
  PrecisionDescriptor precision;
  ErrorBudget budget;
  double observed_error_bound = 0.0;
  std::uint64_t candidates = 0;
  std::uint64_t retained = 0;
  std::uint64_t dropped = 0;
  const char* note = "";
};

class OperationLedger {
 public:
  void Add(OperationLedgerEntry entry) { entries_.push_back(std::move(entry)); }

  void Merge(const OperationLedger& other) {
    for (const auto& e : other.entries_) entries_.push_back(e);
  }

  [[nodiscard]] const std::vector<OperationLedgerEntry>& entries() const {
    return entries_;
  }

  [[nodiscard]] std::size_t size() const { return entries_.size(); }

  [[nodiscard]] double TotalObservedErrorBound(ErrorMetric metric) const {
    double total = 0.0;
    for (const OperationLedgerEntry& entry : entries_) {
      if (entry.budget.metric == metric) {
        total += entry.observed_error_bound;
      }
    }
    return total;
  }

  [[nodiscard]] std::uint64_t TotalDropped() const {
    std::uint64_t dropped = 0;
    for (const OperationLedgerEntry& entry : entries_) {
      dropped += entry.dropped;
    }
    return dropped;
  }

 private:
  std::vector<OperationLedgerEntry> entries_;
};

[[nodiscard]] inline PrecisionDescriptor Fp64ReferencePrecision() {
  return PrecisionDescriptor{};
}

[[nodiscard]] inline PrecisionDescriptor MixedTilePrecision() {
  PrecisionDescriptor descriptor;
  descriptor.storage = NumericFormat::kFloat16;
  descriptor.compute = NumericFormat::kFloat32;
  descriptor.reduction = NumericFormat::kFloat64Emulated;
  descriptor.per_tile_scale = true;
  descriptor.ordered_reductions = true;
  descriptor.label = "mixed-tile-f64e-reduction";
  return descriptor;
}

[[nodiscard]] inline PrecisionDescriptor F64eReferencePrecision() {
  PrecisionDescriptor descriptor;
  descriptor.storage = NumericFormat::kFloat64;
  descriptor.compute = NumericFormat::kFloat64Emulated;
  descriptor.reduction = NumericFormat::kFloat64Emulated;
  descriptor.label = "cpu-long-double-f64e-reference";
  return descriptor;
}

[[nodiscard]] inline Status ValidatePrecisionDescriptor(
    const PrecisionDescriptor& descriptor) {
  if (descriptor.storage == NumericFormat::kUnknown ||
      descriptor.compute == NumericFormat::kUnknown ||
      descriptor.reduction == NumericFormat::kUnknown) {
    return Status::InvalidArgument("precision descriptor has unknown format");
  }
  if (descriptor.determinism == DeterminismMode::kDeterministic &&
      !descriptor.ordered_reductions) {
    return Status::InvalidArgument(
        "deterministic precision descriptor requires ordered reductions");
  }
  if (descriptor.per_tile_scale &&
      descriptor.storage != NumericFormat::kFloat64 &&
      descriptor.storage != NumericFormat::kFloat16 &&
      descriptor.storage != NumericFormat::kBFloat16 &&
      descriptor.storage != NumericFormat::kFloat8) {
    return Status::InvalidArgument(
        "per-tile scale is only valid for tiled storage formats");
  }
  return Status::Ok();
}

[[nodiscard]] inline Status ValidateErrorBudget(const ErrorBudget& budget) {
  if (budget.metric == ErrorMetric::kNone && budget.bound != 0.0) {
    return Status::InvalidArgument("no-error metric must have zero bound");
  }
  if (budget.bound < 0.0 || std::isnan(budget.bound)) {
    return Status::InvalidArgument("error budget bound must be non-negative");
  }
  return Status::Ok();
}

[[nodiscard]] inline Status ValidateOperationLedgerEntry(
    const OperationLedgerEntry& entry) {
  if (entry.operation == OperationKind::kUnknown) {
    return Status::InvalidArgument("operation ledger entry has unknown op");
  }
  const Status precision_status =
      ValidatePrecisionDescriptor(entry.precision);
  if (!precision_status.ok()) {
    return precision_status;
  }
  const Status budget_status = ValidateErrorBudget(entry.budget);
  if (!budget_status.ok()) {
    return budget_status;
  }
  if (entry.observed_error_bound < 0.0 ||
      std::isnan(entry.observed_error_bound)) {
    return Status::InvalidArgument(
        "observed error bound must be non-negative");
  }
  if (entry.candidates != entry.retained + entry.dropped) {
    return Status::InvalidArgument(
        "operation ledger candidate count is not conserved");
  }
  return Status::Ok();
}

}  // namespace tides::tile
