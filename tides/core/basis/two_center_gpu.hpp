#pragma once

#include <cstddef>
#include <vector>

#include "basis/two_center_integrals.hpp"
#include "common/status.hpp"
#include "tile/precision.hpp"

namespace tides::basis {

struct TwoCenterGpuResult {
  std::vector<double> S;
  std::vector<double> T;
  double kernel_ms = 0.0;
  std::size_t n_pairs = 0;
  tides::tile::OperationLedger ledger;
};

[[nodiscard]] bool TwoCenterCudaAvailable();

[[nodiscard]] Result<TwoCenterGpuResult> AssembleTwoCenterCuda(
    const std::vector<double>& positions,
    const std::vector<int>& atomic_numbers,
    const std::vector<int>& l_per_atom,
    const std::vector<int>& basis_offsets,
    std::size_t n_basis,
    const CubicSpline& s_spline,
    const CubicSpline& t_spline);

}  // namespace tides::basis
