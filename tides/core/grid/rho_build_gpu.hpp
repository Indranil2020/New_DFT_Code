#pragma once

#include <cstddef>
#include <vector>

#include "common/status.hpp"
#include "grid/dual_grid.hpp"
#include "tile/precision.hpp"

namespace tides::grid {

struct RhoBuildGpuResult {
  std::vector<double> rho;
  double integral = 0.0;
  double kernel_ms = 0.0;
  std::size_t n_points = 0;
  std::size_t n_orbitals = 0;
  tides::tile::OperationLedger ledger;
};

[[nodiscard]] bool RhoBuildCudaAvailable();

[[nodiscard]] Result<RhoBuildGpuResult> RhoBuildCuda(
    const UniformGrid3D& grid,
    const std::vector<std::vector<double>>& orbitals,
    const std::vector<double>& occupations);

}  // namespace tides::grid
