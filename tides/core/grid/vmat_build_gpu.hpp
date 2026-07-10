#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "grid/dual_grid.hpp"
#include "common/status.hpp"
#include "tile/precision.hpp"

#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>
#else
struct CUstream_st;
using cudaStream_t = CUstream_st*;
#endif

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

// Borrowed device-resident GGA adjoint inputs.  wv_rho and wv_grad already
// include quadrature weights, so this path must not multiply by dv again.
struct GgaVmatDeviceIn {
  const double* phi = nullptr;       // [nbasis][point_stride]
  const double* grad_phi = nullptr;  // [3][nbasis][point_stride]
  const double* wv_rho = nullptr;    // [point_stride]
  const double* wv_grad = nullptr;   // [3][point_stride]
  std::int64_t nbasis = 0;
  std::int64_t np = 0;
  std::int64_t point_stride = 0;
};

// V_mn = sum_g [wv_rho phi_m phi_n + sum_a wv_grad_a
// ((d_a phi_m) phi_n + phi_m (d_a phi_n))].  Writes [nbasis][nbasis]
// row-major on device without allocating or synchronizing.
[[nodiscard]] Status BuildGgaVmatDevice(const GgaVmatDeviceIn& input,
                                        double* vmat, cudaStream_t stream);

}  // namespace tides::grid
