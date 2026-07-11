#pragma once

// Compatibility shim for the legacy GPU XC interface.
// The production XC engine is in xc/xc_engine.hpp (device-resident XcEval).
// This header preserves the old XCEvalLdaCuda/XCEvalPbeCuda signatures
// used by nao_driver.hpp for the GPU dispatch fallback path.
// When CUDA is enabled, the implementation in xc.cu provided these;
// now xc.cu is deleted, so cuda_stubs.cpp provides stubs that return
// Unimplemented, causing callers to fall through to the fused Tier-0 engine.

#include <cstddef>
#include <vector>

#include "common/status.hpp"
#include "grid/dual_grid.hpp"
#include "tile/precision.hpp"

namespace tides::grid {

struct XCGpuResult {
  std::vector<double> vxc;
  std::vector<double> eps_xc;
  double xc_energy = 0.0;
  double kernel_ms = 0.0;
  std::size_t n_points = 0;
  tides::tile::OperationLedger ledger;
};

[[nodiscard]] bool XCCudaAvailable();

[[nodiscard]] Result<XCGpuResult> XCEvalLdaCuda(
    const UniformGrid3D& grid,
    const std::vector<double>& rho,
    double zeta);

[[nodiscard]] Result<XCGpuResult> XCEvalPbeCuda(
    const UniformGrid3D& grid,
    const std::vector<double>& rho);

}  // namespace tides::grid
