// Fallback implementations for tier0/tier1 XC functions when the libxc
// maple2c submodule is not available.  The Tier-0 polarized path uses the
// installed libxc library on the host and copies the resulting weighted
// potentials back to the device so the rest of the device pipeline
// (BuildRhoGradientDevice / BuildGgaVmatDevice) can remain unchanged.
#include "common/status.hpp"
#include "grid/xc/xc_engine.hpp"

#include <cuda_runtime.h>
#include <xc.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace tides::grid::xc::tier0 {

namespace {

[[nodiscard]] Status CudaStatus(cudaError_t error, const char* context) {
  if (error == cudaSuccess) return Status::Ok();
  return Status::IoError(std::string(context) + ": " + cudaGetErrorString(error));
}

struct LibxcComponent {
  int id = 0;
  double coeff = 0.0;
  bool is_gga = false;
};

// Map a single-coefficient Tier-0 functional to the libxc components that
// implement it.  B3LYP and PBE0 are expanded into their local parts; exact
// exchange is not included here because the device XcSpec path is local-only.
bool BuildComponents(const XcSpec& spec, std::vector<LibxcComponent>& out) {
  if (spec.terms.size() != 1 || spec.terms[0].coefficient != 1.0) return false;
  switch (spec.terms[0].functional) {
    case Functional::kLdaPw92:
      out = {{XC_LDA_X, 1.0, false}, {XC_LDA_C_PW, 1.0, false}};
      return true;
    case Functional::kSvwn5:
      out = {{XC_LDA_X, 1.0, false}, {XC_LDA_C_VWN, 1.0, false}};
      return true;
    case Functional::kPbe:
      out = {{XC_GGA_X_PBE, 1.0, true}, {XC_GGA_C_PBE, 1.0, true}};
      return true;
    case Functional::kPbeSol:
      out = {{XC_GGA_X_PBE_SOL, 1.0, true}, {XC_GGA_C_PBE_SOL, 1.0, true}};
      return true;
    case Functional::kRevPbe:
      out = {{XC_GGA_X_PBE_R, 1.0, true}, {XC_GGA_C_PBE, 1.0, true}};
      return true;
    case Functional::kRpbe:
      out = {{XC_GGA_X_RPBE, 1.0, true}, {XC_GGA_C_PBE, 1.0, true}};
      return true;
    case Functional::kBlyp:
      out = {{XC_GGA_X_B88, 1.0, true}, {XC_GGA_C_LYP, 1.0, true}};
      return true;
    case Functional::kB3lyp:
      out = {{XC_LDA_X, 0.08, false},
             {XC_GGA_X_B88, 0.72, true},
             {XC_LDA_C_VWN, 0.19, false},
             {XC_GGA_C_LYP, 0.81, true}};
      return true;
    case Functional::kPbe0:
      out = {{XC_GGA_X_PBE, 0.75, true}, {XC_GGA_C_PBE, 1.0, true}};
      return true;
    default:
      return false;
  }
}

}  // namespace
[[nodiscard]] Status LaunchTier0Pol(const XcSpec& spec,
                                    const XcGridIn& input,
                                    XcGridOut& output,
                                    cudaStream_t stream) {
  if (spec.nspin != 2) {
    return Status::InvalidArgument(
        "LaunchTier0Pol fallback requires nspin == 2");
  }

  const bool is_gga = (spec.family != Family::kLda);
  const std::int64_t np = input.np;
  const std::int64_t stride = input.point_stride;

  if (np < 0 || stride < np) {
    return Status::InvalidArgument(
        "LaunchTier0Pol requires np >= 0 and point_stride >= np");
  }
  if (input.rho == nullptr || input.w == nullptr ||
      output.wv_rho == nullptr || output.exc_per_system == nullptr) {
    return Status::InvalidArgument(
        "LaunchTier0Pol requires non-null rho, weights, wv_rho, and exc_per_system");
  }
  if (is_gga && (input.grad == nullptr || output.wv_grad == nullptr)) {
    return Status::InvalidArgument(
        "GGA LaunchTier0Pol requires non-null grad and wv_grad");
  }

  std::vector<LibxcComponent> components;
  if (!BuildComponents(spec, components)) {
    return Status::Unimplemented(
        "LaunchTier0Pol fallback: functional not supported");
  }

  if (np == 0) {
    return CudaStatus(cudaMemsetAsync(output.exc_per_system, 0, sizeof(double),
                                      stream),
                      "cudaMemsetAsync exc_per_system");
  }

  // Download the strided device arrays computed by BuildRhoGradientDevice.
  std::vector<double> rho_strided(2 * stride, 0.0);
  std::vector<double> w_strided(stride, 0.0);
  std::vector<double> grad_strided(is_gga ? 6 * stride : 0, 0.0);

  cudaError_t err =
      cudaMemcpyAsync(rho_strided.data(), input.rho,
                      2 * stride * sizeof(double), cudaMemcpyDeviceToHost,
                      stream);
  if (err != cudaSuccess) return CudaStatus(err, "cudaMemcpyAsync rho");
  err = cudaMemcpyAsync(w_strided.data(), input.w, stride * sizeof(double),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) return CudaStatus(err, "cudaMemcpyAsync weights");
  if (is_gga) {
    err = cudaMemcpyAsync(grad_strided.data(), input.grad,
                          6 * stride * sizeof(double), cudaMemcpyDeviceToHost,
                          stream);
    if (err != cudaSuccess) return CudaStatus(err, "cudaMemcpyAsync grad");
  }
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) return CudaStatus(err, "cudaStreamSynchronize download");

  // Build contiguous libxc arrays (interleaved spin, per-point).
  std::vector<double> rho_lxc(2 * np, 0.0);
  std::vector<double> sigma_lxc(is_gga ? 3 * np : 0, 0.0);
  for (std::int64_t i = 0; i < np; ++i) {
    double ru = std::max(0.0, rho_strided[i]);
    double rd = std::max(0.0, rho_strided[stride + i]);
    rho_lxc[2 * i] = ru;
    rho_lxc[2 * i + 1] = rd;
    if (is_gga) {
      double gxu = grad_strided[i];
      double gyu = grad_strided[stride + i];
      double gzu = grad_strided[2 * stride + i];
      double gxd = grad_strided[3 * stride + i];
      double gyd = grad_strided[4 * stride + i];
      double gzd = grad_strided[5 * stride + i];
      sigma_lxc[3 * i] = gxu * gxu + gyu * gyu + gzu * gzu;
      sigma_lxc[3 * i + 1] = gxu * gxd + gyu * gyd + gzu * gzd;
      sigma_lxc[3 * i + 2] = gxd * gxd + gyd * gyd + gzd * gzd;
    }
  }

  std::vector<double> exc_total(np, 0.0);
  std::vector<double> vrho_total(2 * np, 0.0);
  std::vector<double> vsigma_total(is_gga ? 3 * np : 0, 0.0);

  std::vector<double> exc_comp(np, 0.0);
  std::vector<double> vrho_comp(2 * np, 0.0);
  std::vector<double> vsigma_comp(is_gga ? 3 * np : 0, 0.0);

  // Accumulate each libxc component with its coefficient.
  for (const auto& comp : components) {
    xc_func_type func;
    if (xc_func_init(&func, comp.id, XC_POLARIZED) != 0) {
      return Status::Unimplemented(
          "LaunchTier0Pol fallback: libxc init failed for functional id " +
          std::to_string(comp.id));
    }
    if (comp.is_gga) {
      xc_gga_exc_vxc(&func, static_cast<std::size_t>(np), rho_lxc.data(),
                     sigma_lxc.data(), exc_comp.data(), vrho_comp.data(),
                     vsigma_comp.data());
    } else {
      xc_lda_exc_vxc(&func, static_cast<std::size_t>(np), rho_lxc.data(),
                     exc_comp.data(), vrho_comp.data());
    }
    xc_func_end(&func);

    for (std::int64_t i = 0; i < np; ++i) {
      exc_total[i] += comp.coeff * exc_comp[i];
      vrho_total[2 * i] += comp.coeff * vrho_comp[2 * i];
      vrho_total[2 * i + 1] += comp.coeff * vrho_comp[2 * i + 1];
    }
    if (is_gga) {
      for (std::int64_t i = 0; i < 3 * np; ++i) {
        vsigma_total[i] += comp.coeff * vsigma_comp[i];
      }
    }
  }

  // Convert back to the strided device layout and compute the XC energy.
  std::vector<double> wv_rho_strided(2 * stride, 0.0);
  std::vector<double> wv_grad_strided(is_gga ? 6 * stride : 0, 0.0);
  double exc_system = 0.0;

  for (std::int64_t i = 0; i < np; ++i) {
    double w = w_strided[i];
    double ru = rho_lxc[2 * i];
    double rd = rho_lxc[2 * i + 1];
    wv_rho_strided[i] = w * vrho_total[2 * i];
    wv_rho_strided[stride + i] = w * vrho_total[2 * i + 1];
    exc_system += w * (ru + rd) * exc_total[i];

    if (is_gga) {
      double gxu = grad_strided[i];
      double gyu = grad_strided[stride + i];
      double gzu = grad_strided[2 * stride + i];
      double gxd = grad_strided[3 * stride + i];
      double gyd = grad_strided[4 * stride + i];
      double gzd = grad_strided[5 * stride + i];
      double v0 = vsigma_total[3 * i];
      double v1 = vsigma_total[3 * i + 1];
      double v2 = vsigma_total[3 * i + 2];

      wv_grad_strided[i] = w * (2.0 * v0 * gxu + v1 * gxd);
      wv_grad_strided[stride + i] = w * (2.0 * v0 * gyu + v1 * gyd);
      wv_grad_strided[2 * stride + i] = w * (2.0 * v0 * gzu + v1 * gzd);
      wv_grad_strided[3 * stride + i] = w * (v1 * gxu + 2.0 * v2 * gxd);
      wv_grad_strided[4 * stride + i] = w * (v1 * gyu + 2.0 * v2 * gyd);
      wv_grad_strided[5 * stride + i] = w * (v1 * gzu + 2.0 * v2 * gzd);
    }
  }

  err = cudaMemcpyAsync(output.wv_rho, wv_rho_strided.data(),
                        2 * stride * sizeof(double), cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) return CudaStatus(err, "cudaMemcpyAsync wv_rho");
  if (is_gga) {
    err = cudaMemcpyAsync(output.wv_grad, wv_grad_strided.data(),
                          6 * stride * sizeof(double), cudaMemcpyHostToDevice,
                          stream);
    if (err != cudaSuccess) return CudaStatus(err, "cudaMemcpyAsync wv_grad");
  }
  err = cudaMemcpyAsync(output.exc_per_system, &exc_system, sizeof(double),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) return CudaStatus(err, "cudaMemcpyAsync exc_per_system");

  return CudaStatus(cudaStreamSynchronize(stream),
                    "cudaStreamSynchronize LaunchTier0Pol");
}
}  // namespace tides::grid::xc::tier0

namespace tides::grid::xc::tier1 {
Status LaunchMggaTpss(const XcGridIn& input, XcGridOut& output,
                      cudaStream_t stream, bool deterministic, int nspin) {
  return Status::Unimplemented("mgga_tpss not available");
}
Status LaunchMggaScan(const XcGridIn& input, XcGridOut& output,
                      cudaStream_t stream, bool deterministic, int nspin) {
  return Status::Unimplemented("mgga_scan not available");
}
Status LaunchMggaR2scan(const XcGridIn& input, XcGridOut& output,
                        cudaStream_t stream, bool deterministic, int nspin) {
  return Status::Unimplemented("mgga_r2scan not available");
}
Status LaunchMggaM06_2x(const XcGridIn& input, XcGridOut& output,
                        cudaStream_t stream, bool deterministic, int nspin) {
  return Status::Unimplemented("mgga_m06_2x not available");
}
Status LaunchRshHse06(const XcGridIn& input, XcGridOut& output,
                      cudaStream_t stream, bool deterministic, int nspin) {
  return Status::Unimplemented("rsh_hse06 not available");
}
Status LaunchRshWb97x(const XcGridIn& input, XcGridOut& output,
                      cudaStream_t stream, bool deterministic, int nspin) {
  return Status::Unimplemented("rsh_wb97x not available");
}
}  // namespace tides::grid::xc::tier1
