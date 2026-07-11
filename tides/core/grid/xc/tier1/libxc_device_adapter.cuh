#pragma once

// T-X4.1: SoA <-> interleaved layout adapter for Tier-1 libxc maple2c device calls.
//
// TIDES XC kernels use SoA (Structure-of-Arrays) layout for coalesced GPU memory
// access: rho[nspin][point_stride], grad[nspin][3][point_stride], etc.
//
// libxc maple2c functions expect interleaved (AoS) layout:
//   rho[2*ip], sigma[3*ip], tau[2*ip] for nspin=2
//   rho[ip], sigma[ip], tau[ip]       for nspin=1
//
// This adapter provides device-side conversion functions to bridge the two layouts.
// In the current Tier-1 implementation, each .cu file handles the conversion inline
// (constructing rho_in[2], sigma_in[3], tau_in[2] from SoA planes before calling
// func_vxc). This header documents the convention and provides reusable helpers
// for future Tier-1 functionals that want to call libxc device functions directly.
//
// Convention:
//   - SoA input:  XcGridIn (rho, grad, tau, w with point_stride)
//   - SoA output: XcGridOut (wv_rho, wv_grad, wv_tau with point_stride)
//   - Interleaved is only used transiently in registers, never materialized in memory.

#include "grid/xc/xc_engine.hpp"

#include <cuda_runtime.h>

namespace tides::grid::xc::tier1 {

// Convert SoA density planes to interleaved rho[2] for a single grid point.
// For nspin=1: rho_in[0] = rho[point], rho_in[1] = 0 (unused by libxc).
// For nspin=2: rho_in[0] = rho[point], rho_in[1] = rho[point_stride + point].
__device__ inline void SoaToInterleavedRho(
    const double* __restrict__ rho, std::int64_t point, std::int64_t stride,
    int nspin, double rho_out[2]) {
  rho_out[0] = rho[point];
  rho_out[1] = (nspin == 2) ? rho[stride + point] : 0.0;
}

// Convert SoA gradient planes to interleaved sigma[3] for a single grid point.
// For nspin=1: sigma_out[0] = |grad|^2, sigma_out[1..2] = 0.
// For nspin=2: sigma_out[0] = |grad_up|^2, sigma_out[1] = grad_up·grad_down,
//              sigma_out[2] = |grad_down|^2.
__device__ inline void SoaToInterleavedSigma(
    const double* __restrict__ grad, std::int64_t point, std::int64_t stride,
    int nspin, double sigma_out[3]) {
  const double gx_up = grad[point];
  const double gy_up = grad[stride + point];
  const double gz_up = grad[2 * stride + point];
  if (nspin == 1) {
    sigma_out[0] = gx_up * gx_up + gy_up * gy_up + gz_up * gz_up;
    sigma_out[1] = 0.0;
    sigma_out[2] = 0.0;
  } else {
    const double gx_dn = grad[3 * stride + point];
    const double gy_dn = grad[4 * stride + point];
    const double gz_dn = grad[5 * stride + point];
    sigma_out[0] = gx_up * gx_up + gy_up * gy_up + gz_up * gz_up;
    sigma_out[1] = gx_up * gx_dn + gy_up * gy_dn + gz_up * gz_dn;
    sigma_out[2] = gx_dn * gx_dn + gy_dn * gy_dn + gz_dn * gz_dn;
  }
}

// Convert SoA tau planes to interleaved tau[2] for a single grid point.
__device__ inline void SoaToInterleavedTau(
    const double* __restrict__ tau, std::int64_t point, std::int64_t stride,
    int nspin, double tau_out[2]) {
  tau_out[0] = tau[point];
  tau_out[1] = (nspin == 2) ? tau[stride + point] : 0.0;
}

// Convert interleaved vrho[2] back to SoA wv_rho planes for a single grid point.
__device__ inline void InterleavedToSoaVrho(
    double* __restrict__ wv_rho, std::int64_t point, std::int64_t stride,
    int nspin, double weight, const double vrho[2]) {
  wv_rho[point] = weight * vrho[0];
  if (nspin == 2) {
    wv_rho[stride + point] = weight * vrho[1];
  }
}

// Convert interleaved vsigma[3] back to SoA wv_grad planes for a single grid point.
// Applies the chain rule: wv_grad = 2 * weight * vsigma * grad.
__device__ inline void InterleavedToSoaVgrad(
    double* __restrict__ wv_grad, std::int64_t point, std::int64_t stride,
    int nspin, double weight, const double vsigma[3],
    const double* __restrict__ grad) {
  const double gx_up = grad[point];
  const double gy_up = grad[stride + point];
  const double gz_up = grad[2 * stride + point];
  if (nspin == 1) {
    const double wv0 = 2.0 * weight * vsigma[0];
    wv_grad[point] = wv0 * gx_up;
    wv_grad[stride + point] = wv0 * gy_up;
    wv_grad[2 * stride + point] = wv0 * gz_up;
  } else {
    const double gx_dn = grad[3 * stride + point];
    const double gy_dn = grad[4 * stride + point];
    const double gz_dn = grad[5 * stride + point];
    const double wv0 = 2.0 * weight * vsigma[0];
    const double wv1 = weight * vsigma[1];
    const double wv2 = 2.0 * weight * vsigma[2];
    wv_grad[point] = wv0 * gx_up + wv1 * gx_dn;
    wv_grad[stride + point] = wv0 * gy_up + wv1 * gy_dn;
    wv_grad[2 * stride + point] = wv0 * gz_up + wv1 * gz_dn;
    wv_grad[3 * stride + point] = wv1 * gx_up + wv2 * gx_dn;
    wv_grad[4 * stride + point] = wv1 * gy_up + wv2 * gy_dn;
    wv_grad[5 * stride + point] = wv1 * gz_up + wv2 * gz_dn;
  }
}

// Convert interleaved vtau[2] back to SoA wv_tau planes for a single grid point.
__device__ inline void InterleavedToSoaVtau(
    double* __restrict__ wv_tau, std::int64_t point, std::int64_t stride,
    int nspin, double weight, const double vtau[2]) {
  wv_tau[point] = weight * vtau[0];
  if (nspin == 2) {
    wv_tau[stride + point] = weight * vtau[1];
  }
}

}  // namespace tides::grid::xc::tier1
