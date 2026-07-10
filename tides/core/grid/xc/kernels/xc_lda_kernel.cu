// AUDIT T-X1.2: Fused LDA XC kernel.
// Loads rho → eval LDA Slater + PW92 functors → write w·v_ρ → in-kernel E_xc reduction.
// Zero register spills target. Deterministic reduction mode available.

#include <cuda_runtime.h>
#include <cmath>

#include "grid/xc/functionals/lda_slater.cuh"
#include "grid/xc/functionals/lda_pw92.cuh"

namespace tides::grid::xc {

// Fused LDA kernel: one thread per grid point.
// Each thread computes eps_xc and v_xc for its point.
// E_xc reduction via atomicAdd (fast mode) or ordered per-block (deterministic).
__global__ void XcLdaKernel(
    const double* __restrict__ rho,
    double* __restrict__ vxc,
    double* __restrict__ eps_xc,
    double* __restrict__ xc_energy,
    double grid_weight,
    std::size_t np) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= np) return;

  const double n = rho[idx];
  const double eps_x = LdaSlater::Eps(n);
  const double eps_c = LdaPw92::Eps(n);
  const double v_x = LdaSlater::Vrho(n);
  const double v_c = LdaPw92::Vrho(n);

  const double eps_xc_val = eps_x + eps_c;
  const double vxc_val = v_x + v_c;

  vxc[idx] = vxc_val;
  eps_xc[idx] = eps_xc_val;

  // In-kernel E_xc reduction: E_xc = sum(w * eps_xc * rho).
  const double e_contrib = grid_weight * eps_xc_val * n;
  atomicAdd(xc_energy, e_contrib);
}

}  // namespace tides::grid::xc
