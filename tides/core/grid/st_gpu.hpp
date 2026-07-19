#pragma once

#include <cstddef>
#include <vector>

#include "common/status.hpp"

namespace tides::grid {

// Compute overlap (S) and kinetic (T) matrices from real-space grid orbitals
// on the GPU using cuBLAS.  phi_flat is [n_basis][n_points] row-major and
// grad_flat is [3][n_basis][n_points] row-major.  Returns S and T as
// n_basis x n_basis row-major matrices.
struct StGpuResult {
  std::vector<double> S;
  std::vector<double> T;
  Status status;
};

StGpuResult BuildStFromGridGpu(const std::vector<double>& phi_flat,
                               const std::vector<double>& grad_flat,
                               std::size_t n, std::size_t np_total,
                               double dv);

}  // namespace tides::grid
