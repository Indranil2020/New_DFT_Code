#include "tile/precision.hpp"
#include "tile/spgemm_filtered.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::tile::DeterminismMode;
using tides::tile::ErrorMetric;
using tides::tile::Fp64ReferencePrecision;
using tides::tile::MixedTilePrecision;
using tides::tile::NumericFormat;
using tides::tile::OperationKind;
using tides::tile::OperationLedger;
using tides::tile::OperationLedgerEntry;
using tides::tile::PrecisionDescriptor;
using tides::tile::SpGemmFilteredFp64;
using tides::tile::TileMat;
using tides::tile::ValidateOperationLedgerEntry;
using tides::tile::ValidatePrecisionDescriptor;

int Fail(const std::string& message) {
  std::cerr << "precision_tests: " << message << '\n';
  return 1;
}

int DescriptorValidation() {
  const PrecisionDescriptor fp64 = Fp64ReferencePrecision();
  if (!ValidatePrecisionDescriptor(fp64).ok()) {
    return Fail("FP64 reference descriptor rejected");
  }
  PrecisionDescriptor fp64_scaled = fp64;
  fp64_scaled.per_tile_scale = true;
  if (!ValidatePrecisionDescriptor(fp64_scaled).ok()) {
    return Fail("scaled FP64 reference descriptor rejected");
  }
  const PrecisionDescriptor mixed = MixedTilePrecision();
  if (!ValidatePrecisionDescriptor(mixed).ok()) {
    return Fail("mixed tile descriptor rejected");
  }

  PrecisionDescriptor invalid = fp64;
  invalid.storage = NumericFormat::kUnknown;
  if (ValidatePrecisionDescriptor(invalid).ok()) {
    return Fail("unknown storage descriptor accepted");
  }

  invalid = fp64;
  invalid.storage = NumericFormat::kFloat32;
  invalid.per_tile_scale = true;
  if (ValidatePrecisionDescriptor(invalid).ok()) {
    return Fail("per-tile scaling accepted for unsupported storage");
  }

  invalid = fp64;
  invalid.determinism = DeterminismMode::kDeterministic;
  invalid.ordered_reductions = false;
  if (ValidatePrecisionDescriptor(invalid).ok()) {
    return Fail("deterministic unordered reduction descriptor accepted");
  }
  return 0;
}

int OperationLedgerAggregation() {
  OperationLedger ledger;
  ledger.Add(OperationLedgerEntry{
      OperationKind::kSpGemmFiltered,
      Fp64ReferencePrecision(),
      tides::tile::ErrorBudget{ErrorMetric::kFrobenius, 2.0, "test"},
      1.5,
      7,
      5,
      2,
      "first"});
  ledger.Add(OperationLedgerEntry{
      OperationKind::kSpGemmFiltered,
      Fp64ReferencePrecision(),
      tides::tile::ErrorBudget{ErrorMetric::kFrobenius, 3.0, "test"},
      2.5,
      11,
      8,
      3,
      "second"});
  for (const OperationLedgerEntry& entry : ledger.entries()) {
    if (!ValidateOperationLedgerEntry(entry).ok()) {
      return Fail("valid operation ledger entry rejected");
    }
  }
  if (ledger.TotalDropped() != 5) {
    return Fail("dropped-product aggregation is wrong");
  }
  if (ledger.TotalObservedErrorBound(ErrorMetric::kFrobenius) != 4.0) {
    return Fail("Frobenius bound aggregation is wrong");
  }

  OperationLedgerEntry bad = ledger.entries().front();
  bad.candidates = 99;
  if (ValidateOperationLedgerEntry(bad).ok()) {
    return Fail("candidate-count violation accepted");
  }
  return 0;
}

int SpGemmProducesGenericLedger() {
  std::vector<double> a_dense(32 * 32, 0.0);
  std::vector<double> b_dense(32 * 32, 0.0);
  a_dense[0] = 1.0;
  b_dense[0] = 1.0e-3;
  a_dense[16 * 32 + 16] = 2.0;
  b_dense[16 * 32 + 16] = 2.0;
  auto a = TileMat::FromDense(32, 32, a_dense, 16);
  auto b = TileMat::FromDense(32, 32, b_dense, 16);
  if (!a.ok() || !b.ok()) {
    return Fail("TileMat construction failed");
  }

  auto result = SpGemmFilteredFp64(a.value(), b.value(), 1.0e-2);
  if (!result.ok()) {
    return Fail("SpGemmFilteredFp64 failed: " + result.status().message());
  }
  const auto& op_ledger = result.value().ledger.operation_ledger;
  if (op_ledger.size() != 1) {
    return Fail("SpGEMM did not emit one operation ledger entry");
  }
  const OperationLedgerEntry& entry = op_ledger.entries().front();
  const auto status = ValidateOperationLedgerEntry(entry);
  if (!status.ok()) {
    return Fail("SpGEMM operation ledger invalid: " + status.message());
  }
  if (entry.operation != OperationKind::kSpGemmFiltered ||
      entry.budget.metric != ErrorMetric::kFrobenius) {
    return Fail("SpGEMM operation ledger has wrong operation or metric");
  }
  if (entry.dropped == 0 || entry.observed_error_bound <= 0.0) {
    return Fail("SpGEMM operation ledger did not record dropped work");
  }
  if (op_ledger.TotalObservedErrorBound(ErrorMetric::kFrobenius) !=
      result.value().ledger.dropped_frobenius_bound) {
    return Fail("generic ledger bound disagrees with SpGEMM ledger");
  }
  return 0;
}

}  // namespace

int main() {
  if (const int rc = DescriptorValidation(); rc != 0) {
    return rc;
  }
  if (const int rc = OperationLedgerAggregation(); rc != 0) {
    return rc;
  }
  if (const int rc = SpGemmProducesGenericLedger(); rc != 0) {
    return rc;
  }
  std::cout << "precision_tests: descriptor and operation ledger checks passed\n";
  return 0;
}
