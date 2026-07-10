// AUDIT T-X0.3: XC Engine host dispatch.
// Routes XcSpec to Tier-0 (fused device kernels), Tier-1 (libxc device), or Tier-2 (CPU).
// Currently implements Tier-0 for LDA-PW92 and PBE.

#include "grid/xc/xc_engine.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>

#ifdef TIDES_HAVE_CUDA
#include <cuda_runtime.h>
#include "grid/xc/kernels/xc_lda_kernel.cu"
#include "grid/xc/kernels/xc_gga_kernel.cu"
#endif

namespace tides::grid::xc {

std::string XcFunctionalName(XcFunctionalId id) {
  switch (id) {
    case XcFunctionalId::kLdaPw92:    return "LDA-PW92";
    case XcFunctionalId::kLdaVwn5:    return "SVWN5";
    case XcFunctionalId::kPbe:        return "PBE";
    case XcFunctionalId::kPbesol:     return "PBEsol";
    case XcFunctionalId::kRevPbe:     return "revPBE";
    case XcFunctionalId::kRpbe:       return "RPBE";
    case XcFunctionalId::kBlyp:       return "BLYP";
    case XcFunctionalId::kPbe0Local:  return "PBE0(local)";
    case XcFunctionalId::kB3lypLocal: return "B3LYP(local)";
    case XcFunctionalId::kHse06Local: return "HSE06(local)";
    case XcFunctionalId::kTpss:       return "TPSS";
    case XcFunctionalId::kR2scan:     return "r2SCAN";
    case XcFunctionalId::kScan:       return "SCAN";
    case XcFunctionalId::kWb97xLocal: return "wB97X(local)";
    case XcFunctionalId::kM062xLocal: return "M06-2X(local)";
  }
  return "unknown";
}

bool IsTier0(XcFunctionalId id) {
  switch (id) {
    case XcFunctionalId::kLdaPw92:
    case XcFunctionalId::kPbe:
    case XcFunctionalId::kPbesol:
    case XcFunctionalId::kRevPbe:
      return true;
    default:
      return false;
  }
}

#ifdef TIDES_HAVE_CUDA

// CUDA helper: allocate device, copy, launch, copy back.
static bool CudaXcEvalLda(const XcGridIn& in, XcGridOut& out, std::string& error_msg) {
  const std::size_t np = in.np;
  const std::size_t bytes = np * sizeof(double);

  double *d_rho = nullptr, *d_vxc = nullptr, *d_eps = nullptr, *d_energy = nullptr;
  cudaMalloc(&d_rho, bytes);
  cudaMalloc(&d_vxc, bytes);
  cudaMalloc(&d_eps, bytes);
  cudaMalloc(&d_energy, sizeof(double));
  if (!d_rho || !d_vxc || !d_eps || !d_energy) {
    error_msg = "cudaMalloc failed in CudaXcEvalLda";
    if (d_rho) cudaFree(d_rho);
    if (d_vxc) cudaFree(d_vxc);
    if (d_eps) cudaFree(d_eps);
    if (d_energy) cudaFree(d_energy);
    return false;
  }

  cudaMemset(d_energy, 0, sizeof(double));
  cudaMemcpy(d_rho, in.rho, bytes, cudaMemcpyHostToDevice);

  const int block = 256;
  const int grid = (static_cast<int>(np) + block - 1) / block;

  auto t0 = std::chrono::steady_clock::now();
  XcLdaKernel<<<grid, block>>>(d_rho, d_vxc, d_eps, d_energy,
                                 in.grid_weight, np);
  cudaDeviceSynchronize();
  auto t1 = std::chrono::steady_clock::now();
  out.kernel_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  cudaMemcpy(out.vxc, d_vxc, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(out.eps_xc, d_eps, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(&out.xc_energy, d_energy, sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_rho);
  cudaFree(d_vxc);
  cudaFree(d_eps);
  cudaFree(d_energy);
  return true;
}

static bool CudaXcEvalGga(const XcGridIn& in, XcGridOut& out, std::string& error_msg) {
  const std::size_t np = in.np;
  const std::size_t bytes = np * sizeof(double);

  double *d_rho = nullptr, *d_gx = nullptr, *d_gy = nullptr, *d_gz = nullptr;
  double *d_vxc = nullptr, *d_vsigma = nullptr, *d_eps = nullptr, *d_energy = nullptr;

  cudaMalloc(&d_rho, bytes);
  cudaMalloc(&d_gx, bytes);
  cudaMalloc(&d_gy, bytes);
  cudaMalloc(&d_gz, bytes);
  cudaMalloc(&d_vxc, bytes);
  cudaMalloc(&d_vsigma, bytes);
  cudaMalloc(&d_eps, bytes);
  cudaMalloc(&d_energy, sizeof(double));

  if (!d_rho || !d_vxc || !d_eps || !d_energy) {
    error_msg = "cudaMalloc failed in CudaXcEvalGga";
    return false;
  }

  cudaMemset(d_energy, 0, sizeof(double));
  cudaMemcpy(d_rho, in.rho, bytes, cudaMemcpyHostToDevice);
  if (in.grad_rho_x) cudaMemcpy(d_gx, in.grad_rho_x, bytes, cudaMemcpyHostToDevice);
  if (in.grad_rho_y) cudaMemcpy(d_gy, in.grad_rho_y, bytes, cudaMemcpyHostToDevice);
  if (in.grad_rho_z) cudaMemcpy(d_gz, in.grad_rho_z, bytes, cudaMemcpyHostToDevice);

  const int block = 256;
  const int grid = (static_cast<int>(np) + block - 1) / block;

  auto t0 = std::chrono::steady_clock::now();
  XcGgaKernel<<<grid, block>>>(d_rho, d_gx, d_gy, d_gz,
                                 d_vxc, d_vsigma, d_eps, d_energy,
                                 in.grid_weight, np);
  cudaDeviceSynchronize();
  auto t1 = std::chrono::steady_clock::now();
  out.kernel_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  cudaMemcpy(out.vxc, d_vxc, bytes, cudaMemcpyDeviceToHost);
  if (out.vsigma) cudaMemcpy(out.vsigma, d_vsigma, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(out.eps_xc, d_eps, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(&out.xc_energy, d_energy, sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_rho); cudaFree(d_gx); cudaFree(d_gy); cudaFree(d_gz);
  cudaFree(d_vxc); cudaFree(d_vsigma); cudaFree(d_eps); cudaFree(d_energy);
  return true;
}

#endif // TIDES_HAVE_CUDA

// CPU fallback for Tier-0 functionals (also used when CUDA is not available).
static bool CpuXcEvalLda(const XcGridIn& in, XcGridOut& out) {
  // Include functors for CPU path.
#include "grid/xc/functionals/lda_slater.cuh"
#include "grid/xc/functionals/lda_pw92.cuh"

  auto t0 = std::chrono::steady_clock::now();
  double energy = 0.0;
  for (std::size_t i = 0; i < in.np; ++i) {
    const double n = std::max(0.0, in.rho[i]);
    const double eps_x = LdaSlater::Eps(n);
    const double eps_c = LdaPw92::Eps(n);
    const double v_x = LdaSlater::Vrho(n);
    const double v_c = LdaPw92::Vrho(n);
    out.vxc[i] = v_x + v_c;
    out.eps_xc[i] = eps_x + eps_c;
    energy += in.grid_weight * (eps_x + eps_c) * n;
  }
  auto t1 = std::chrono::steady_clock::now();
  out.kernel_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  out.xc_energy = energy;
  return true;
}

static bool CpuXcEvalGga(const XcGridIn& in, XcGridOut& out) {
#include "grid/xc/functionals/gga_pbe.cuh"

  auto t0 = std::chrono::steady_clock::now();
  double energy = 0.0;
  for (std::size_t i = 0; i < in.np; ++i) {
    const double n = std::max(0.0, in.rho[i]);
    if (n < 1e-14) {
      out.vxc[i] = 0.0;
      if (out.vsigma) out.vsigma[i] = 0.0;
      out.eps_xc[i] = 0.0;
      continue;
    }
    const double gx = in.grad_rho_x ? in.grad_rho_x[i] : 0.0;
    const double gy = in.grad_rho_y ? in.grad_rho_y[i] : 0.0;
    const double gz = in.grad_rho_z ? in.grad_rho_z[i] : 0.0;
    const double sigma = gx * gx + gy * gy + gz * gz;

    auto x_res = PbeX::VrhoVsigma(n, sigma);
    auto c_res = PbeC::VrhoVsigma(n, sigma);
    const double eps_x = PbeX::Eps(n, sigma);
    const double eps_c = PbeC::Eps(n, sigma);

    out.vxc[i] = x_res.v_rho + c_res.v_rho;
    if (out.vsigma) out.vsigma[i] = x_res.v_sigma + c_res.v_sigma;
    out.eps_xc[i] = eps_x + eps_c;
    energy += in.grid_weight * (eps_x + eps_c) * n;
  }
  auto t1 = std::chrono::steady_clock::now();
  out.kernel_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  out.xc_energy = energy;
  return true;
}

bool XcEval(const XcSpec& spec, const XcGridIn& in, XcGridOut& out,
            std::string& error_msg) {
  if (in.np == 0) {
    error_msg = "XcEval: np = 0";
    return false;
  }
  if (!in.rho || !out.vxc || !out.eps_xc) {
    error_msg = "XcEval: null pointers in input/output";
    return false;
  }

  if (!IsTier0(spec.id)) {
    error_msg = "XcEval: functional " + XcFunctionalName(spec.id) +
                " not yet implemented in Tier-0";
    return false;
  }

  // Route by family.
  if (spec.family == XcFamily::kLda) {
#ifdef TIDES_HAVE_CUDA
    return CudaXcEvalLda(in, out, error_msg);
#else
    return CpuXcEvalLda(in, out);
#endif
  } else if (spec.family == XcFamily::kGga) {
    if (!in.grad_rho_x || !in.grad_rho_y || !in.grad_rho_z) {
      error_msg = "XcEval: GGA requires grad_rho_x/y/z";
      return false;
    }
#ifdef TIDES_HAVE_CUDA
    return CudaXcEvalGga(in, out, error_msg);
#else
    return CpuXcEvalGga(in, out);
#endif
  }

  error_msg = "XcEval: family " + std::to_string(static_cast<int>(spec.family)) +
              " not supported";
  return false;
}

}  // namespace tides::grid::xc
