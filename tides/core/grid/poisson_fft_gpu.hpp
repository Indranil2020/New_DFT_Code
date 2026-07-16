#pragma once

#include <cstddef>
#include <vector>

#include "common/status.hpp"
#include "grid/dual_grid.hpp"
#include "tile/precision.hpp"

#ifdef TIDES_HAVE_CUDA
#include <cuda_runtime.h>
#include <cufft.h>
#endif

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

#ifdef TIDES_HAVE_CUDA
// Device-resident free-space Poisson solver.
// Accepts rho as a device pointer, outputs V as a device pointer.
// Uses the caller's CUDA stream. Caches cuFFT plans and the 1/|r| kernel
// across calls for the same grid dimensions.
// d_rho: device pointer to rho (np_total doubles)
// d_V_out: device pointer to receive V (np_total doubles), must be pre-allocated
// Returns hartree_energy in result (computed on device).
struct PoissonFreeDeviceResult {
  double hartree_energy = 0.0;
  double kernel_ms = 0.0;
  std::size_t grid_size = 0;
};

class PoissonFreeDeviceCache {
 public:
  static PoissonFreeDeviceCache& Instance() {
    static PoissonFreeDeviceCache inst;
    return inst;
  }

  // Solve free-space Poisson with device-resident rho and V.
  // d_rho: device pointer (np_total doubles)
  // d_V_out: device pointer (np_total doubles, pre-allocated)
  // stream: caller's CUDA stream
  Result<PoissonFreeDeviceResult> Solve(
      const UniformGrid3D& grid,
      const double* d_rho,
      double* d_V_out,
      cudaStream_t stream);

  // Release cached resources.
  void Release();

  ~PoissonFreeDeviceCache();

 private:
  PoissonFreeDeviceCache() = default;
  PoissonFreeDeviceCache(const PoissonFreeDeviceCache&) = delete;
  PoissonFreeDeviceCache& operator=(const PoissonFreeDeviceCache&) = delete;

  bool initialized_ = false;
  std::size_t cached_n0_ = 0, cached_n1_ = 0, cached_n2_ = 0;
  std::size_t m0_ = 0, m1_ = 0, m2_ = 0, M_ = 0;

  cufftHandle plan_fwd_ = 0;
  cufftHandle plan_inv_ = 0;
  cufftDoubleComplex* d_g_ = nullptr;      // 1/|r| kernel (device)
  cufftDoubleComplex* d_rho_pad_ = nullptr; // zero-padded rho (device)
  cufftDoubleComplex* d_V_pad_ = nullptr;   // zero-padded V (device)
  double* d_energy_ = nullptr;              // scratch for hartree energy
};
#endif

}  // namespace tides::grid
