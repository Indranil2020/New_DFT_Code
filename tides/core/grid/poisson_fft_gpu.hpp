#pragma once

#include <cstddef>
#include <vector>

#include "common/status.hpp"
#include "grid/dual_grid.hpp"
#include "tile/precision.hpp"

namespace tides::grid {

struct PoissonFftGpuResult {
  std::vector<double> V;
  double hartree_energy = 0.0;
  double kernel_ms = 0.0;
  std::size_t grid_size = 0;
  tides::tile::OperationLedger ledger;
};

[[nodiscard]] bool PoissonFftCudaAvailable();

[[nodiscard]] Result<PoissonFftGpuResult> PoissonFftCuda(
    const UniformGrid3D& grid,
    const std::vector<double>& rho);

}  // namespace tides::grid
