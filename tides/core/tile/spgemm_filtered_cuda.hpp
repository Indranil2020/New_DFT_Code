#pragma once

#include "common/status.hpp"
#include "tile/spgemm_filtered.hpp"

namespace tides::tile {

struct CudaSpGemmFilteredResult {
  TileMat product;
  ErrorLedger ledger;
  double kernel_ms = 0.0;
};

[[nodiscard]] Result<CudaSpGemmFilteredResult> SpGemmFilteredFp64Cuda(
    const TileMat& a, const TileMat& b, double eps_filter);

}  // namespace tides::tile
