#pragma once

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

}  // namespace tides::grid
