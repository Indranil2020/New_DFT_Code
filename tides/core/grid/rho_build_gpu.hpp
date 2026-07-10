#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/status.hpp"
#include "grid/dual_grid.hpp"
#include "tile/precision.hpp"

#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>
#else
struct CUstream_st;
using cudaStream_t = CUstream_st*;
#endif

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

// Borrowed device-resident NAO product views.  All basis/grid planes use
// point_stride; no allocation, copy, or synchronization occurs in the call.
struct RhoGradientDeviceIn {
  const double* density_matrix = nullptr;  // [nbasis][nbasis], row-major
  const double* phi = nullptr;             // [nbasis][point_stride]
  const double* grad_phi = nullptr;        // [3][nbasis][point_stride]
  std::int64_t nbasis = 0;
  std::int64_t np = 0;
  std::int64_t point_stride = 0;
};

// rho_g = sum_mn P_mn phi_m,g phi_n,g
// grad_a rho_g = sum_mn P_mn [(d_a phi_m,g) phi_n,g + phi_m,g (d_a phi_n,g)].
// Outputs are [point_stride] and [3][point_stride], ready for XcGridIn.
[[nodiscard]] Status BuildRhoGradientDevice(const RhoGradientDeviceIn& input,
                                             double* rho, double* grad,
                                             cudaStream_t stream);

}  // namespace tides::grid
