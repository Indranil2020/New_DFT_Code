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

// GPU free-space Poisson solver via zero-padded cuFFT convolution.
// Solves nabla^2 V = -4*pi*rho for isolated (free BC) systems on GPU.
// Doubles the grid, zero-pads rho, builds 1/|r| kernel, FFT convolution.
[[nodiscard]] Result<PoissonFftGpuResult> PoissonFreeCuda(
    const UniformGrid3D& grid,
    const std::vector<double>& rho);

}  // namespace tides::grid
