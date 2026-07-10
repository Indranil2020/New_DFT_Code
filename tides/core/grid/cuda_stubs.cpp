// CPU-only stubs for CUDA grid functions.
// Linked when TIDES_HAVE_CUDA is NOT defined, so that the NaoDriver
// (which calls these functions directly) can compile and link without CUDA.
// All stubs return false/unavailable, causing the NaoDriver to fall back
// to the CPU code paths.

#include "grid/xc_gpu.hpp"
#include "grid/vmat_build_gpu.hpp"
#include "grid/poisson_fft_gpu.hpp"
#include "grid/rho_build_gpu.hpp"
#include "common/status.hpp"

namespace tides::grid {

// XC GPU stubs
bool XCCudaAvailable() { return false; }

Result<XCGpuResult> XCEvalLdaCuda(
    const UniformGrid3D& /*grid*/,
    const std::vector<double>& /*rho*/,
    double /*zeta*/) {
  return Status::Unimplemented("CUDA not compiled — XC GPU stub");
}

Result<XCGpuResult> XCEvalPbeCuda(
    const UniformGrid3D& /*grid*/,
    const std::vector<double>& /*rho*/) {
  return Status::Unimplemented("CUDA not compiled — XC GPU stub");
}

// Vmat GPU stubs
bool VmatCudaAvailable() { return false; }

Result<VmatGpuResult> VmatBuildCuda(
    const UniformGrid3D& /*grid*/,
    const std::vector<std::vector<double>>& /*orbitals*/,
    const std::vector<double>& /*v*/) {
  return Status::Unimplemented("CUDA not compiled — Vmat GPU stub");
}

// Rho GPU stubs
bool RhoBuildCudaAvailable() { return false; }

Result<RhoBuildGpuResult> RhoBuildCuda(
    const UniformGrid3D& /*grid*/,
    const std::vector<std::vector<double>>& /*orbitals*/,
    const std::vector<double>& /*occupations*/) {
  return Status::Unimplemented("CUDA not compiled — Rho GPU stub");
}

// Poisson GPU stubs
bool PoissonFftCudaAvailable() { return false; }

Result<PoissonFftGpuResult> PoissonFftCuda(
    const UniformGrid3D& /*grid*/,
    const std::vector<double>& /*rho*/) {
  return Status::Unimplemented("CUDA not compiled — Poisson GPU stub");
}

}  // namespace tides::grid
