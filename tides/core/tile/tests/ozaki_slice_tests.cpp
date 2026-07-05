#include "tile/ozaki.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using tides::tile::BuildOzakiFp16SlicePlan;
using tides::tile::DecomposeOzakiFp16Reference;
using tides::tile::GemmF64eReference;
using tides::tile::GemmOzakiFp16Reference;
using tides::tile::OperationKind;
using tides::tile::OzakiSlicePlan;
using tides::tile::ReconstructOzakiSlices;
using tides::tile::ValidateOperationLedgerEntry;

int Fail(const std::string& message) {
  std::cerr << "ozaki_slice_tests: " << message << '\n';
  return 1;
}

double UlpAbs(double value) {
  if (value == 0.0) {
    return std::numeric_limits<double>::denorm_min();
  }
  const double next =
      std::nextafter(value, std::numeric_limits<double>::infinity());
  return std::abs(next - value);
}

int DynamicRangeIncreasesSliceCount() {
  auto compact = BuildOzakiFp16SlicePlan({0.5, -1.25, 2.0, 3.0});
  auto broad = BuildOzakiFp16SlicePlan({1.0e9, -1.0e-9, 3.0, -2.0});
  if (!compact.ok() || !broad.ok()) {
    return Fail("valid slice plan construction failed");
  }
  if (compact.value().slice_count != 6) {
    return Fail("compact FP64 mantissa should require six FP16 slices");
  }
  if (broad.value().slice_count <= compact.value().slice_count) {
    return Fail("broad dynamic range did not increase slice count");
  }
  if (broad.value().exponent_span <= compact.value().exponent_span) {
    return Fail("broad dynamic range did not record larger exponent span");
  }
  return 0;
}

int SlicesRespectFp16MantissaContract(const OzakiSlicePlan& plan,
                                      std::size_t value_count,
                                      const std::vector<double>& slices) {
  const double max_units =
      std::ldexp(1.0, static_cast<int>(plan.mantissa_bits_per_slice));
  for (std::uint32_t slice = 0; slice < plan.slice_count; ++slice) {
    const double quantum = plan.slice_quanta[slice];
    for (std::size_t value_index = 0; value_index < value_count;
         ++value_index) {
      const double chunk = slices[static_cast<std::size_t>(slice) *
                                      value_count +
                                  value_index];
      const double units = chunk / quantum;
      const double rounded_units = std::nearbyint(units);
      if (units != rounded_units) {
        return Fail("slice chunk is not an integer quantum multiple");
      }
      if (std::abs(units) > max_units) {
        return Fail("slice chunk exceeds FP16 mantissa unit bound");
      }
    }
  }
  return 0;
}

int ReconstructsAdversarialDynamicRange() {
  const std::vector<double> values = {
      1.0e9,
      -1.0e9,
      1.0e-9,
      -1.0e-9,
      3.141592653589793,
      -2.718281828459045,
      std::ldexp(1.0, -40),
      -std::ldexp(1.0, 42),
      0.0,
  };
  auto decomposition = DecomposeOzakiFp16Reference(values);
  if (!decomposition.ok()) {
    return Fail("Ozaki decomposition failed: " +
                decomposition.status().message());
  }
  if (const int rc = SlicesRespectFp16MantissaContract(
          decomposition.value().plan, values.size(),
          decomposition.value().slices);
      rc != 0) {
    return rc;
  }
  auto reconstructed =
      ReconstructOzakiSlices(decomposition.value().plan, values.size(),
                             decomposition.value().slices);
  if (!reconstructed.ok()) {
    return Fail("Ozaki reconstruction failed: " +
                reconstructed.status().message());
  }
  for (std::size_t i = 0; i < values.size(); ++i) {
    const double error = std::abs(values[i] - reconstructed.value()[i]);
    const double bound = 4.0 * UlpAbs(values[i]);
    if (error > bound) {
      return Fail("Ozaki reconstruction exceeds 4x FP64 ulp bound");
    }
  }
  const auto& entry = decomposition.value().ledger.entries().front();
  if (entry.operation != OperationKind::kPrecisionTransform ||
      !ValidateOperationLedgerEntry(entry).ok()) {
    return Fail("Ozaki decomposition ledger entry is invalid");
  }
  return 0;
}

int RejectsInsufficientSliceBudget() {
  auto plan = BuildOzakiFp16SlicePlan({1.0e9, 1.0e-9}, 2);
  if (plan.ok()) {
    return Fail("insufficient Ozaki max_slices budget was accepted");
  }
  return 0;
}

int ZeroInputHasSingleSlice() {
  auto decomposition = DecomposeOzakiFp16Reference({0.0, -0.0});
  if (!decomposition.ok()) {
    return Fail("zero Ozaki decomposition failed");
  }
  if (decomposition.value().plan.slice_count != 1) {
    return Fail("zero input should use one regularity slice");
  }
  if (decomposition.value().max_reconstruction_abs_error != 0.0) {
    return Fail("zero input reconstructed with nonzero error");
  }
  return 0;
}

std::vector<double> MakeDynamicRangeMatrix(std::size_t rows, std::size_t cols,
                                           std::uint64_t salt) {
  std::vector<double> values(rows * cols, 0.0);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      const std::uint64_t key =
          (i + 3) * 0x9E3779B185EBCA87ULL ^
          (j + 11) * 0xC2B2AE3D27D4EB4FULL ^ salt;
      const double sign = (key & 1U) == 0 ? 1.0 : -1.0;
      if (key % 11 == 0) {
        values[i * cols + j] = sign * 1.0e9;
      } else if (key % 7 == 0) {
        values[i * cols + j] = sign * 1.0e-9;
      } else {
        values[i * cols + j] =
            sign * static_cast<double>((key % 997U) + 1U) / 997.0;
      }
    }
  }
  return values;
}

int GemmMatchesF64eWithinForwardBound() {
  constexpr std::size_t m = 6;
  constexpr std::size_t k = 19;
  constexpr std::size_t n = 5;
  const std::vector<double> a =
      MakeDynamicRangeMatrix(m, k, 0xA0A0A0A0ULL);
  const std::vector<double> b =
      MakeDynamicRangeMatrix(k, n, 0xB0B0B0B0ULL);
  auto ozaki = GemmOzakiFp16Reference(m, k, n, a, b);
  auto reference = GemmF64eReference(m, k, n, a, b);
  if (!ozaki.ok()) {
    return Fail("Ozaki GEMM failed: " + ozaki.status().message());
  }
  if (!reference.ok()) {
    return Fail("f64e GEMM failed: " + reference.status().message());
  }
  if (ozaki.value().a_plan.slice_count <= 6 ||
      ozaki.value().b_plan.slice_count <= 6) {
    return Fail("dynamic-range GEMM did not require extra Ozaki slices");
  }
  if (ozaki.value().slice_pair_count !=
      ozaki.value().a_plan.slice_count * ozaki.value().b_plan.slice_count) {
    return Fail("Ozaki GEMM slice-pair count is inconsistent");
  }
  for (std::size_t i = 0; i < reference.value().values.size(); ++i) {
    const double error =
        std::abs(ozaki.value().values[i] - reference.value().values[i]);
    const double bound = ozaki.value().max_fp64_forward_abs_bound +
                         4.0 * UlpAbs(reference.value().values[i]);
    if (error > bound) {
      return Fail("Ozaki GEMM exceeded FP64 forward-error bound");
    }
  }
  const auto& entry = ozaki.value().ledger.entries().front();
  if (entry.operation != OperationKind::kGemm ||
      !ValidateOperationLedgerEntry(entry).ok()) {
    return Fail("Ozaki GEMM ledger entry is invalid");
  }
  if (entry.observed_error_bound > entry.budget.bound +
                                       4.0 * UlpAbs(entry.budget.bound)) {
    return Fail("Ozaki GEMM observed error exceeded ledger budget");
  }

  auto repeat = GemmOzakiFp16Reference(m, k, n, a, b);
  if (!repeat.ok()) {
    return Fail("repeat Ozaki GEMM failed: " + repeat.status().message());
  }
  if (repeat.value().values != ozaki.value().values ||
      repeat.value().max_abs_error_vs_long_double !=
          ozaki.value().max_abs_error_vs_long_double ||
      repeat.value().max_fp64_forward_abs_bound !=
          ozaki.value().max_fp64_forward_abs_bound ||
      repeat.value().slice_pair_count != ozaki.value().slice_pair_count) {
    return Fail("Ozaki GEMM is not bitwise deterministic");
  }
  return 0;
}

}  // namespace

int main() {
  if (const int rc = DynamicRangeIncreasesSliceCount(); rc != 0) {
    return rc;
  }
  if (const int rc = ReconstructsAdversarialDynamicRange(); rc != 0) {
    return rc;
  }
  if (const int rc = RejectsInsufficientSliceBudget(); rc != 0) {
    return rc;
  }
  if (const int rc = ZeroInputHasSingleSlice(); rc != 0) {
    return rc;
  }
  if (const int rc = GemmMatchesF64eWithinForwardBound(); rc != 0) {
    return rc;
  }
  std::cout << "ozaki_slice_tests: FP16-style slice planning checks passed\n";
  return 0;
}
