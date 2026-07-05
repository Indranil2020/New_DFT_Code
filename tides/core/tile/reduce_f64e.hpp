#pragma once

#include <cstddef>
#include <vector>

#include "common/status.hpp"
#include "tile/precision.hpp"

namespace tides::tile {

struct CudaF64eReductionResult {
  double value = 0.0;
  double abs_error_vs_long_double = 0.0;
  double analytical_abs_bound = 0.0;
  double kernel_ms = 0.0;
  OperationLedger ledger;
};

[[nodiscard]] Result<CudaF64eReductionResult> DotF64eCuda(
    const std::vector<double>& a, const std::vector<double>& b);

[[nodiscard]] Result<CudaF64eReductionResult> SumF64eCuda(
    const std::vector<double>& values);

[[nodiscard]] Result<CudaF64eReductionResult> TraceF64eCuda(
    std::size_t rows, std::size_t cols, const std::vector<double>& values);

}  // namespace tides::tile
