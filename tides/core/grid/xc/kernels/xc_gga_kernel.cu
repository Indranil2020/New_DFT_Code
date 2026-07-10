// AUDIT T-X1.2: Fused GGA XC kernel.
// Loads rho + ∇ρ → compute σ = |∇ρ|^2 in registers → eval PBE functors
// → write w·v_ρ and w·v_σ∇ρ (the GGA adjoint term).
// The -2∇·(v_σ∇ρ) term lives in vmat_build (audit T-X1.4), not here.

#include <cuda_runtime.h>
#include <cmath>

#include "grid/xc/functionals/lda_slater.cuh"
#include "grid/xc/functionals/lda_pw92.cuh"
#include "grid/xc/functionals/gga_pbe.cuh"

namespace tides::grid::xc {

// Fused GGA kernel: one thread per grid point.
// Computes PBE exchange + correlation from rho and grad_rho.
// Outputs v_rho (LDA-like potential) and v_sigma (GGA derivative w.r.t. |grad_rho|^2).
__global__ void XcGgaKernel(
    const double* __restrict__ rho,
    const double* __restrict__ grad_rho_x,
    const double* __restrict__ grad_rho_y,
    const double* __restrict__ grad_rho_z,
    double* __restrict__ vxc,        // v_rho = d(E_xc)/d(rho)
    double* __restrict__ vsigma,     // v_sigma = d(E_xc)/d(sigma)
    double* __restrict__ eps_xc,
    double* __restrict__ xc_energy,
    double grid_weight,
    std::size_t np) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= np) return;

  const double n = rho[idx];
  if (n < 1e-14) {
    vxc[idx] = 0.0;
    vsigma[idx] = 0.0;
    eps_xc[idx] = 0.0;
    return;
  }

  // Compute sigma = |grad_rho|^2 in registers.
  const double gx = grad_rho_x[idx];
  const double gy = grad_rho_y[idx];
  const double gz = grad_rho_z[idx];
  const double sigma = gx * gx + gy * gy + gz * gz;

  // PBE Exchange.
  auto x_res = PbeX::VrhoVsigma(n, sigma);
  const double eps_x = PbeX::Eps(n, sigma);

  // PBE Correlation.
  auto c_res = PbeC::VrhoVsigma(n, sigma);
  const double eps_c = PbeC::Eps(n, sigma);

  // Combine.
  vxc[idx] = x_res.v_rho + c_res.v_rho;
  vsigma[idx] = x_res.v_sigma + c_res.v_sigma;
  eps_xc[idx] = eps_x + eps_c;

  // In-kernel E_xc reduction.
  const double e_contrib = grid_weight * (eps_x + eps_c) * n;
  atomicAdd(xc_energy, e_contrib);
}

}  // namespace tides::grid::xc
