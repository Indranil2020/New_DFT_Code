#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "common/status.hpp"
#include "tile/f64e_reference.hpp"
#include "tile/precision.hpp"

namespace tides::tile {

enum class OzakiSliceFormat : std::uint32_t {
  kFp16 = 0,
};

struct OzakiSlicePlan {
  OzakiSliceFormat format = OzakiSliceFormat::kFp16;
  std::uint32_t slice_count = 0;
  std::uint32_t mantissa_bits_per_slice = 10;
  std::uint32_t target_mantissa_bits = 53;
  int min_exponent = 0;
  int max_exponent = 0;
  int exponent_span = 0;
  std::vector<double> slice_quanta;
};

struct OzakiDecomposition {
  OzakiSlicePlan plan;
  std::size_t value_count = 0;
  std::vector<double> slices;
  double max_reconstruction_abs_error = 0.0;
  OperationLedger ledger;
};

struct OzakiGemmResult {
  std::vector<double> values;
  OzakiSlicePlan a_plan;
  OzakiSlicePlan b_plan;
  double max_abs_error_vs_long_double = 0.0;
  double max_fp64_forward_abs_bound = 0.0;
  std::uint32_t slice_pair_count = 0;
  OperationLedger ledger;
};

[[nodiscard]] inline PrecisionDescriptor OzakiFp16ReferencePrecision() {
  PrecisionDescriptor descriptor;
  descriptor.storage = NumericFormat::kFloat16;
  descriptor.compute = NumericFormat::kFloat64Emulated;
  descriptor.reduction = NumericFormat::kFloat64Emulated;
  descriptor.per_tile_scale = true;
  descriptor.ordered_reductions = true;
  descriptor.label = "ozaki-fp16-slice-reference";
  return descriptor;
}

[[nodiscard]] inline std::uint32_t CeilDivU32(std::uint32_t num,
                                              std::uint32_t den) {
  return (num + den - 1) / den;
}

[[nodiscard]] inline Result<OzakiSlicePlan> BuildOzakiFp16SlicePlan(
    const std::vector<double>& values, std::uint32_t max_slices = 16,
    std::uint32_t target_mantissa_bits = 53) {
  if (max_slices == 0) {
    return Status::InvalidArgument("Ozaki max_slices must be nonzero");
  }
  if (target_mantissa_bits == 0) {
    return Status::InvalidArgument(
        "Ozaki target_mantissa_bits must be nonzero");
  }

  bool have_nonzero = false;
  int min_exponent = 0;
  int max_exponent = 0;
  for (const double value : values) {
    if (!std::isfinite(value)) {
      return Status::InvalidArgument("Ozaki slicing requires finite values");
    }
    if (value == 0.0) {
      continue;
    }
    int exponent = 0;
    std::frexp(std::abs(value), &exponent);
    if (!have_nonzero) {
      min_exponent = exponent;
      max_exponent = exponent;
      have_nonzero = true;
    } else {
      min_exponent = std::min(min_exponent, exponent);
      max_exponent = std::max(max_exponent, exponent);
    }
  }

  OzakiSlicePlan plan;
  plan.min_exponent = min_exponent;
  plan.max_exponent = max_exponent;
  plan.exponent_span = have_nonzero ? max_exponent - min_exponent : 0;
  const std::uint32_t span_bits =
      static_cast<std::uint32_t>(std::max(0, plan.exponent_span));
  const std::uint32_t required_bits = target_mantissa_bits + span_bits;
  plan.slice_count = have_nonzero
                         ? CeilDivU32(required_bits,
                                      plan.mantissa_bits_per_slice)
                         : 1;
  if (plan.slice_count > max_slices) {
    return Status::OutOfRange("Ozaki FP16 slice plan exceeds max_slices");
  }

  plan.slice_quanta.reserve(plan.slice_count);
  for (std::uint32_t slice = 0; slice < plan.slice_count; ++slice) {
    const int exponent =
        plan.max_exponent -
        static_cast<int>((slice + 1) * plan.mantissa_bits_per_slice);
    const double quantum = std::ldexp(1.0, exponent);
    if (quantum == 0.0 || !std::isfinite(quantum)) {
      return Status::OutOfRange(
          "Ozaki FP16 slice quantum is outside FP64 range");
    }
    plan.slice_quanta.push_back(quantum);
  }
  return plan;
}

[[nodiscard]] inline Result<std::vector<double>> ReconstructOzakiSlices(
    const OzakiSlicePlan& plan, std::size_t value_count,
    const std::vector<double>& slices) {
  if (plan.slice_count == 0) {
    return Status::InvalidArgument("Ozaki slice plan has zero slices");
  }
  if (plan.slice_quanta.size() != plan.slice_count) {
    return Status::InvalidArgument("Ozaki slice plan quantum count mismatch");
  }
  if (slices.size() != value_count * plan.slice_count) {
    return Status::InvalidArgument("Ozaki slice payload size mismatch");
  }

  std::vector<double> reconstructed(value_count, 0.0);
  for (std::size_t value_index = 0; value_index < value_count; ++value_index) {
    long double sum = 0.0L;
    for (std::uint32_t slice = 0; slice < plan.slice_count; ++slice) {
      sum += static_cast<long double>(
          slices[static_cast<std::size_t>(slice) * value_count + value_index]);
    }
    reconstructed[value_index] = static_cast<double>(sum);
  }
  return reconstructed;
}

[[nodiscard]] inline Result<OzakiDecomposition> DecomposeOzakiFp16Reference(
    const std::vector<double>& values, std::uint32_t max_slices = 16,
    std::uint32_t target_mantissa_bits = 53) {
  auto plan_result =
      BuildOzakiFp16SlicePlan(values, max_slices, target_mantissa_bits);
  if (!plan_result.ok()) {
    return plan_result.status();
  }
  OzakiDecomposition result;
  result.plan = plan_result.take_value();
  result.value_count = values.size();
  result.slices.assign(values.size() * result.plan.slice_count, 0.0);

  const double max_units =
      std::ldexp(1.0, static_cast<int>(result.plan.mantissa_bits_per_slice));
  for (std::size_t value_index = 0; value_index < values.size();
       ++value_index) {
    double residual = values[value_index];
    for (std::uint32_t slice = 0; slice < result.plan.slice_count; ++slice) {
      const double quantum = result.plan.slice_quanta[slice];
      const double units = std::nearbyint(residual / quantum);
      if (!std::isfinite(units) || std::abs(units) > max_units) {
        return Status::OutOfRange(
            "Ozaki FP16 slice unit exceeds mantissa contract");
      }
      const double chunk = units * quantum;
      result.slices[static_cast<std::size_t>(slice) * values.size() +
                    value_index] = chunk;
      residual -= chunk;
    }
  }

  auto reconstructed =
      ReconstructOzakiSlices(result.plan, values.size(), result.slices);
  if (!reconstructed.ok()) {
    return reconstructed.status();
  }
  for (std::size_t i = 0; i < values.size(); ++i) {
    const double error = std::abs(values[i] - reconstructed.value()[i]);
    result.max_reconstruction_abs_error =
        std::max(result.max_reconstruction_abs_error, error);
  }

  const std::uint64_t slice_terms =
      static_cast<std::uint64_t>(values.size()) * result.plan.slice_count;
  result.ledger.Add(OperationLedgerEntry{
      OperationKind::kPrecisionTransform,
      OzakiFp16ReferencePrecision(),
      ErrorBudget{ErrorMetric::kAbsolute,
                  result.max_reconstruction_abs_error,
                  "max absolute reconstruction error after FP16-style slices"},
      result.max_reconstruction_abs_error,
      slice_terms,
      slice_terms,
      0,
      "CPU Ozaki FP16-style operand slicing reference"});
  return result;
}

[[nodiscard]] inline double Fp64DotForwardAbsBound(
    std::size_t k, long double sum_abs_products) {
  constexpr long double unit_roundoff =
      static_cast<long double>(std::numeric_limits<double>::epsilon()) / 2.0L;
  const long double ku = static_cast<long double>(k) * unit_roundoff;
  const long double gamma = ku < 1.0L ? ku / (1.0L - ku) : ku;
  return static_cast<double>(gamma * sum_abs_products);
}

[[nodiscard]] inline Result<OzakiGemmResult> GemmOzakiFp16Reference(
    std::size_t m, std::size_t k, std::size_t n,
    const std::vector<double>& a, const std::vector<double>& b,
    std::uint32_t max_slices = 16) {
  if (a.size() != m * k || b.size() != k * n) {
    return Status::InvalidArgument("Ozaki GEMM shape mismatch");
  }
  auto a_decomposition = DecomposeOzakiFp16Reference(a, max_slices);
  if (!a_decomposition.ok()) {
    return a_decomposition.status();
  }
  auto b_decomposition = DecomposeOzakiFp16Reference(b, max_slices);
  if (!b_decomposition.ok()) {
    return b_decomposition.status();
  }

  OzakiGemmResult result;
  result.values.assign(m * n, 0.0);
  result.a_plan = a_decomposition.value().plan;
  result.b_plan = b_decomposition.value().plan;
  result.slice_pair_count = result.a_plan.slice_count * result.b_plan.slice_count;

  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t col = 0; col < n; ++col) {
      long double sliced_sum = 0.0L;
      for (std::uint32_t a_slice = 0; a_slice < result.a_plan.slice_count;
           ++a_slice) {
        const std::size_t a_slice_offset =
            static_cast<std::size_t>(a_slice) * a.size();
        for (std::uint32_t b_slice = 0; b_slice < result.b_plan.slice_count;
             ++b_slice) {
          const std::size_t b_slice_offset =
              static_cast<std::size_t>(b_slice) * b.size();
          for (std::size_t p = 0; p < k; ++p) {
            const double a_chunk =
                a_decomposition.value().slices[a_slice_offset + row * k + p];
            const double b_chunk =
                b_decomposition.value().slices[b_slice_offset + p * n + col];
            sliced_sum += static_cast<long double>(a_chunk) *
                          static_cast<long double>(b_chunk);
          }
        }
      }

      long double exact_sum = 0.0L;
      long double sum_abs_products = 0.0L;
      for (std::size_t p = 0; p < k; ++p) {
        const long double product = static_cast<long double>(a[row * k + p]) *
                                    static_cast<long double>(b[p * n + col]);
        exact_sum += product;
        sum_abs_products += product < 0.0L ? -product : product;
      }
      const double rounded = static_cast<double>(sliced_sum);
      const double observed_error =
          AbsLongDoubleToDoubleError(exact_sum, rounded);
      const double forward_bound =
          4.0 * Fp64DotForwardAbsBound(k, sum_abs_products);
      result.max_abs_error_vs_long_double =
          std::max(result.max_abs_error_vs_long_double, observed_error);
      result.max_fp64_forward_abs_bound =
          std::max(result.max_fp64_forward_abs_bound, forward_bound);
      result.values[row * n + col] = rounded;
    }
  }

  const std::uint64_t slice_products =
      static_cast<std::uint64_t>(m) * static_cast<std::uint64_t>(k) *
      static_cast<std::uint64_t>(n) * result.slice_pair_count;
  result.ledger.Add(OperationLedgerEntry{
      OperationKind::kGemm,
      OzakiFp16ReferencePrecision(),
      ErrorBudget{ErrorMetric::kAbsolute,
                  result.max_fp64_forward_abs_bound,
                  "4x FP64 deterministic dot-product forward-error bound"},
      result.max_abs_error_vs_long_double,
      slice_products,
      slice_products,
      0,
      "CPU Ozaki FP16-slice GEMM reference"});
  return result;
}

}  // namespace tides::tile
