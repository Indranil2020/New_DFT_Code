#pragma once

#include <cstddef>
#include <vector>

#include "grid/dual_grid.hpp"
#include "common/status.hpp"
#include "tile/precision.hpp"

namespace tides::grid {

struct VmatGpuResult {
  std::vector<double> H;  // n_orb x n_orb Hamiltonian tile
  double kernel_ms = 0.0;
  std::size_t n_points = 0;
  std::size_t n_orb = 0;
  tides::tile::OperationLedger ledger;
};

[[nodiscard]] bool VmatCudaAvailable();

// GPU adjoint: H_ij = integral v(r) * phi_i(r) * phi_j(r) d^3r
//
// @param grid    Uniform 3D grid
// @param orbitals  Vector of orbitals, each of size grid.total_points()
// @param v       Potential on the grid (size grid.total_points())
// @returns       H matrix (n_orb x n_orb, symmetric, row-major)
[[nodiscard]] Result<VmatGpuResult> VmatBuildCuda(
    const UniformGrid3D& grid,
    const std::vector<std::vector<double>>& orbitals,
    const std::vector<double>& v);

}  // namespace tides::grid
