#include "grid/xc/tier0/tier0_pol.hpp"
#include "grid/xc/tier0/tier0_pol_kernel.cuh"
#include "grid/xc/tier1/util.h"
#include "grid/xc/functionals/common.cuh"
#include "common/status.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <cstdint>
#include <string>

namespace tides::grid::xc::tier0 {

// ===========================================================================
// LDA maple2c namespaces
// ===========================================================================

namespace lda_x {
typedef struct { double alpha; } lda_x_params;
#define XC_DONT_COMPILE_EXC
#define XC_DONT_COMPILE_FXC
#define XC_DONT_COMPILE_KXC
#define XC_DONT_COMPILE_LXC
#include "lda_exc/lda_x.c"
}  // namespace lda_x

namespace lda_c_pw {
typedef struct {
  double pp[3], a[3], alpha1[3];
  double beta1[3], beta2[3], beta3[3], beta4[3];
  double fz20;
} lda_c_pw_params;
#define XC_DONT_COMPILE_EXC
#define XC_DONT_COMPILE_FXC
#define XC_DONT_COMPILE_KXC
#define XC_DONT_COMPILE_LXC
#include "lda_exc/lda_c_pw.c"
}  // namespace lda_c_pw

namespace lda_c_vwn {
#define XC_DONT_COMPILE_EXC
#define XC_DONT_COMPILE_FXC
#define XC_DONT_COMPILE_KXC
#define XC_DONT_COMPILE_LXC
#include "lda_exc/lda_c_vwn.c"
}  // namespace lda_c_vwn

// ===========================================================================
// GGA maple2c namespaces
// ===========================================================================

namespace gga_x_pbe {
typedef struct { double kappa, mu, lambda; } gga_x_pbe_params;
#define XC_DONT_COMPILE_EXC
#define XC_DONT_COMPILE_FXC
#define XC_DONT_COMPILE_KXC
#define XC_DONT_COMPILE_LXC
#include "gga_exc/gga_x_pbe.c"
}  // namespace gga_x_pbe

namespace gga_c_pbe {
typedef struct { double beta, gamma, BB; } gga_c_pbe_params;
#define XC_DONT_COMPILE_EXC
#define XC_DONT_COMPILE_FXC
#define XC_DONT_COMPILE_KXC
#define XC_DONT_COMPILE_LXC
#include "gga_exc/gga_c_pbe.c"
}  // namespace gga_c_pbe

namespace gga_x_rpbe {
typedef struct { double rpbe_kappa, rpbe_mu; } gga_x_rpbe_params;
#define XC_DONT_COMPILE_EXC
#define XC_DONT_COMPILE_FXC
#define XC_DONT_COMPILE_KXC
#define XC_DONT_COMPILE_LXC
#include "gga_exc/gga_x_rpbe.c"
}  // namespace gga_x_rpbe

namespace gga_x_b88 {
typedef struct { double beta, gamma; } gga_x_b88_params;
#define XC_DONT_COMPILE_EXC
#define XC_DONT_COMPILE_FXC
#define XC_DONT_COMPILE_KXC
#define XC_DONT_COMPILE_LXC
#include "gga_exc/gga_x_b88.c"
}  // namespace gga_x_b88

namespace gga_c_lyp {
typedef struct { double a, b, c, d; } gga_c_lyp_params;
#define XC_DONT_COMPILE_EXC
#define XC_DONT_COMPILE_FXC
#define XC_DONT_COMPILE_KXC
#define XC_DONT_COMPILE_LXC
#include "gga_exc/gga_c_lyp.c"
}  // namespace gga_c_lyp

// ===========================================================================
// Helper: set up xc_func_type for polarized evaluation
// ===========================================================================

namespace {

__device__ void SetupLdaFunc(xc_func_type& func, xc_func_info_type& info,
                              void* params_ptr) {
  info.flags = XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC;
  info.dens_threshold = 1e-15;
  func.info = &info;
  func.nspin = 2;
  func.dens_threshold = 1e-15;
  func.zeta_threshold = DBL_EPSILON;
  func.sigma_threshold = 1e-20;
  func.tau_threshold = 1e-20;
  func.dim = {2, 1, 1, 1, 1, 2, 1, 1, 1};
  func.params = params_ptr;
}

__device__ void SetupGgaFunc(xc_func_type& func, xc_func_info_type& info,
                              void* params_ptr) {
  info.flags = XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC;
  info.dens_threshold = 1e-15;
  func.info = &info;
  func.nspin = 2;
  func.dens_threshold = 1e-15;
  func.zeta_threshold = DBL_EPSILON;
  func.sigma_threshold = 1e-20;
  func.tau_threshold = 1e-20;
  func.dim = {2, 3, 1, 1, 1, 2, 3, 1, 1};
  func.params = params_ptr;
}

// ===========================================================================
// LDA polarized functors
// ===========================================================================

// LDA-X (Slater exchange) polarized
struct LdaXPolFunctor {
  __device__ LdaPolEvaluation operator()(const double rho[2]) const {
    lda_x::lda_x_params params = {1.0};
    xc_func_info_type info = {};
    xc_func_type func = {};
    SetupLdaFunc(func, info, &params);

    double zk = 0.0;
    double vrho[2] = {};
    xc_lda_out_params out = {};
    out.zk = &zk;
    out.vrho = vrho;

    lda_x::func_vxc_pol(&func, 0, rho, &out);

    LdaPolEvaluation result;
    result.eps = zk;
    result.vrho[0] = vrho[0];
    result.vrho[1] = vrho[1];
    return result;
  }
};

// LDA-C-PW (Perdew-Wang 1992) polarized
struct LdaCpwPolFunctor {
  __device__ LdaPolEvaluation operator()(const double rho[2]) const {
    lda_c_pw::lda_c_pw_params params = {
      {1.0, 1.0, 1.0},                          // pp
      {0.031091, 0.015545, 0.016887},           // a
      {0.21370, 0.20548, 0.11125},              // alpha1
      {7.5957, 14.1189, 10.357},                // beta1
      {3.5876, 6.1977, 3.6231},                 // beta2
      {1.6382, 3.3662, 0.88026},                // beta3
      {0.49294, 0.62517, 0.49671},              // beta4
      1.709921                                  // fz20
    };
    xc_func_info_type info = {};
    xc_func_type func = {};
    SetupLdaFunc(func, info, &params);

    double zk = 0.0;
    double vrho[2] = {};
    xc_lda_out_params out = {};
    out.zk = &zk;
    out.vrho = vrho;

    lda_c_pw::func_vxc_pol(&func, 0, rho, &out);

    LdaPolEvaluation result;
    result.eps = zk;
    result.vrho[0] = vrho[0];
    result.vrho[1] = vrho[1];
    return result;
  }
};

// LDA-C-VWN5 polarized
struct LdaCvwnPolFunctor {
  __device__ LdaPolEvaluation operator()(const double rho[2]) const {
    int dummy_params = 0;  // VWN doesn't use params, but func needs non-null
    xc_func_info_type info = {};
    xc_func_type func = {};
    SetupLdaFunc(func, info, &dummy_params);

    double zk = 0.0;
    double vrho[2] = {};
    xc_lda_out_params out = {};
    out.zk = &zk;
    out.vrho = vrho;

    lda_c_vwn::func_vxc_pol(&func, 0, rho, &out);

    LdaPolEvaluation result;
    result.eps = zk;
    result.vrho[0] = vrho[0];
    result.vrho[1] = vrho[1];
    return result;
  }
};

// LDA-PW92 = Slater exchange + PW correlation
struct LdaPw92PolFunctor {
  __device__ LdaPolEvaluation operator()(const double rho[2]) const {
    LdaPolEvaluation ex = LdaXPolFunctor{}(rho);
    LdaPolEvaluation ec = LdaCpwPolFunctor{}(rho);
    return {ex.eps + ec.eps, {ex.vrho[0] + ec.vrho[0], ex.vrho[1] + ec.vrho[1]}};
  }
};

// SVWN5 = Slater exchange + VWN5 correlation
struct Svwn5PolFunctor {
  __device__ LdaPolEvaluation operator()(const double rho[2]) const {
    LdaPolEvaluation ex = LdaXPolFunctor{}(rho);
    LdaPolEvaluation ec = LdaCvwnPolFunctor{}(rho);
    return {ex.eps + ec.eps, {ex.vrho[0] + ec.vrho[0], ex.vrho[1] + ec.vrho[1]}};
  }
};

// ===========================================================================
// GGA polarized functors
// ===========================================================================

// GGA-X-PBE polarized (parameterized by kappa and mu)
struct GgaXPbePolFunctor {
  double kappa_, mu_;
  __device__ GgaXPbePolFunctor(double kappa = 0.8040, double mu = MU_PBE)
      : kappa_(kappa), mu_(mu) {}
  __device__ GgaPolEvaluation operator()(const double rho[2], const double sigma[3]) const {
    gga_x_pbe::gga_x_pbe_params params = {kappa_, mu_, 0.0};
    xc_func_info_type info = {};
    xc_func_type func = {};
    SetupGgaFunc(func, info, &params);

    double zk = 0.0;
    double vrho[2] = {};
    double vsigma[3] = {};
    xc_gga_out_params out = {};
    out.zk = &zk;
    out.vrho = vrho;
    out.vsigma = vsigma;

    gga_x_pbe::func_vxc_pol(&func, 0, rho, sigma, &out);

    GgaPolEvaluation result;
    result.eps = zk;
    result.vrho[0] = vrho[0]; result.vrho[1] = vrho[1];
    result.vsigma[0] = vsigma[0]; result.vsigma[1] = vsigma[1]; result.vsigma[2] = vsigma[2];
    return result;
  }
};

// GGA-C-PBE polarized (parameterized by beta, gamma, BB)
struct GgaCPbePolFunctor {
  double beta_, gamma_, BB_;
  __device__ GgaCPbePolFunctor(double beta = 0.06672455060314922,
                               double gamma = 0.031090690869654895034,
                               double BB = 1.0)
      : beta_(beta), gamma_(gamma), BB_(BB) {}
  __device__ GgaPolEvaluation operator()(const double rho[2], const double sigma[3]) const {
    gga_c_pbe::gga_c_pbe_params params = {beta_, gamma_, BB_};
    xc_func_info_type info = {};
    xc_func_type func = {};
    SetupGgaFunc(func, info, &params);

    double zk = 0.0;
    double vrho[2] = {};
    double vsigma[3] = {};
    xc_gga_out_params out = {};
    out.zk = &zk;
    out.vrho = vrho;
    out.vsigma = vsigma;

    gga_c_pbe::func_vxc_pol(&func, 0, rho, sigma, &out);

    GgaPolEvaluation result;
    result.eps = zk;
    result.vrho[0] = vrho[0]; result.vrho[1] = vrho[1];
    result.vsigma[0] = vsigma[0]; result.vsigma[1] = vsigma[1]; result.vsigma[2] = vsigma[2];
    return result;
  }
};

// GGA-X-RPBE polarized
struct GgaXRpbePolFunctor {
  __device__ GgaPolEvaluation operator()(const double rho[2], const double sigma[3]) const {
    gga_x_rpbe::gga_x_rpbe_params params = {0.8040, MU_PBE};
    xc_func_info_type info = {};
    xc_func_type func = {};
    SetupGgaFunc(func, info, &params);

    double zk = 0.0;
    double vrho[2] = {};
    double vsigma[3] = {};
    xc_gga_out_params out = {};
    out.zk = &zk;
    out.vrho = vrho;
    out.vsigma = vsigma;

    gga_x_rpbe::func_vxc_pol(&func, 0, rho, sigma, &out);

    GgaPolEvaluation result;
    result.eps = zk;
    result.vrho[0] = vrho[0]; result.vrho[1] = vrho[1];
    result.vsigma[0] = vsigma[0]; result.vsigma[1] = vsigma[1]; result.vsigma[2] = vsigma[2];
    return result;
  }
};

// GGA-X-B88 polarized
struct GgaXB88PolFunctor {
  __device__ GgaPolEvaluation operator()(const double rho[2], const double sigma[3]) const {
    gga_x_b88::gga_x_b88_params params = {0.0042, 6.0};
    xc_func_info_type info = {};
    xc_func_type func = {};
    SetupGgaFunc(func, info, &params);

    double zk = 0.0;
    double vrho[2] = {};
    double vsigma[3] = {};
    xc_gga_out_params out = {};
    out.zk = &zk;
    out.vrho = vrho;
    out.vsigma = vsigma;

    gga_x_b88::func_vxc_pol(&func, 0, rho, sigma, &out);

    GgaPolEvaluation result;
    result.eps = zk;
    result.vrho[0] = vrho[0]; result.vrho[1] = vrho[1];
    result.vsigma[0] = vsigma[0]; result.vsigma[1] = vsigma[1]; result.vsigma[2] = vsigma[2];
    return result;
  }
};

// GGA-C-LYP polarized
struct GgaCLypPolFunctor {
  __device__ GgaPolEvaluation operator()(const double rho[2], const double sigma[3]) const {
    gga_c_lyp::gga_c_lyp_params params = {0.04918, 0.132, 0.2533, 0.349};
    xc_func_info_type info = {};
    xc_func_type func = {};
    SetupGgaFunc(func, info, &params);

    double zk = 0.0;
    double vrho[2] = {};
    double vsigma[3] = {};
    xc_gga_out_params out = {};
    out.zk = &zk;
    out.vrho = vrho;
    out.vsigma = vsigma;

    gga_c_lyp::func_vxc_pol(&func, 0, rho, sigma, &out);

    GgaPolEvaluation result;
    result.eps = zk;
    result.vrho[0] = vrho[0]; result.vrho[1] = vrho[1];
    result.vsigma[0] = vsigma[0]; result.vsigma[1] = vsigma[1]; result.vsigma[2] = vsigma[2];
    return result;
  }
};

// ===========================================================================
// Composite GGA polarized functors
// ===========================================================================

// PBE = PBE-X + PBE-C
struct PbePolFunctor {
  __device__ GgaPolEvaluation operator()(const double rho[2], const double sigma[3]) const {
    GgaPolEvaluation ex = GgaXPbePolFunctor{}(rho, sigma);
    GgaPolEvaluation ec = GgaCPbePolFunctor{}(rho, sigma);
    return {ex.eps + ec.eps,
            {ex.vrho[0] + ec.vrho[0], ex.vrho[1] + ec.vrho[1]},
            {ex.vsigma[0] + ec.vsigma[0], ex.vsigma[1] + ec.vsigma[1], ex.vsigma[2] + ec.vsigma[2]}};
  }
};

// PBEsol = PBE-X(sol) + PBE-C(sol)
struct PbeSolPolFunctor {
  __device__ GgaPolEvaluation operator()(const double rho[2], const double sigma[3]) const {
    GgaPolEvaluation ex = GgaXPbePolFunctor{0.804, MU_GE}(rho, sigma);
    GgaPolEvaluation ec = GgaCPbePolFunctor{0.046, 0.031090690869654895034, 1.0}(rho, sigma);
    return {ex.eps + ec.eps,
            {ex.vrho[0] + ec.vrho[0], ex.vrho[1] + ec.vrho[1]},
            {ex.vsigma[0] + ec.vsigma[0], ex.vsigma[1] + ec.vsigma[1], ex.vsigma[2] + ec.vsigma[2]}};
  }
};

// revPBE = PBE-X(rev) + PBE-C
struct RevPbePolFunctor {
  __device__ GgaPolEvaluation operator()(const double rho[2], const double sigma[3]) const {
    GgaPolEvaluation ex = GgaXPbePolFunctor{1.245, MU_PBE}(rho, sigma);
    GgaPolEvaluation ec = GgaCPbePolFunctor{}(rho, sigma);
    return {ex.eps + ec.eps,
            {ex.vrho[0] + ec.vrho[0], ex.vrho[1] + ec.vrho[1]},
            {ex.vsigma[0] + ec.vsigma[0], ex.vsigma[1] + ec.vsigma[1], ex.vsigma[2] + ec.vsigma[2]}};
  }
};

// RPBE = RPBE-X + PBE-C
struct RpbePolFunctor {
  __device__ GgaPolEvaluation operator()(const double rho[2], const double sigma[3]) const {
    GgaPolEvaluation ex = GgaXRpbePolFunctor{}(rho, sigma);
    GgaPolEvaluation ec = GgaCPbePolFunctor{}(rho, sigma);
    return {ex.eps + ec.eps,
            {ex.vrho[0] + ec.vrho[0], ex.vrho[1] + ec.vrho[1]},
            {ex.vsigma[0] + ec.vsigma[0], ex.vsigma[1] + ec.vsigma[1], ex.vsigma[2] + ec.vsigma[2]}};
  }
};

// BLYP = B88-X + LYP-C
struct BlypPolFunctor {
  __device__ GgaPolEvaluation operator()(const double rho[2], const double sigma[3]) const {
    GgaPolEvaluation ex = GgaXB88PolFunctor{}(rho, sigma);
    GgaPolEvaluation ec = GgaCLypPolFunctor{}(rho, sigma);
    return {ex.eps + ec.eps,
            {ex.vrho[0] + ec.vrho[0], ex.vrho[1] + ec.vrho[1]},
            {ex.vsigma[0] + ec.vsigma[0], ex.vsigma[1] + ec.vsigma[1], ex.vsigma[2] + ec.vsigma[2]}};
  }
};

// B3LYP = 0.08*LDA-X + 0.72*B88-X + 0.19*VWN5-C + 0.81*LYP-C
struct B3lypPolFunctor {
  __device__ GgaPolEvaluation operator()(const double rho[2], const double sigma[3]) const {
    LdaPolEvaluation lda_x = LdaXPolFunctor{}(rho);
    LdaPolEvaluation vwn = LdaCvwnPolFunctor{}(rho);
    GgaPolEvaluation b88 = GgaXB88PolFunctor{}(rho, sigma);
    GgaPolEvaluation lyp = GgaCLypPolFunctor{}(rho, sigma);

    GgaPolEvaluation result;
    result.eps = 0.08 * lda_x.eps + 0.72 * b88.eps + 0.19 * vwn.eps + 0.81 * lyp.eps;
    result.vrho[0] = 0.08 * lda_x.vrho[0] + 0.72 * b88.vrho[0] + 0.19 * vwn.vrho[0] + 0.81 * lyp.vrho[0];
    result.vrho[1] = 0.08 * lda_x.vrho[1] + 0.72 * b88.vrho[1] + 0.19 * vwn.vrho[1] + 0.81 * lyp.vrho[1];
    result.vsigma[0] = 0.72 * b88.vsigma[0] + 0.81 * lyp.vsigma[0];
    result.vsigma[1] = 0.72 * b88.vsigma[1] + 0.81 * lyp.vsigma[1];
    result.vsigma[2] = 0.72 * b88.vsigma[2] + 0.81 * lyp.vsigma[2];
    return result;
  }
};

// PBE0 = 0.75*PBE-X + 1.0*PBE-C
struct Pbe0PolFunctor {
  __device__ GgaPolEvaluation operator()(const double rho[2], const double sigma[3]) const {
    GgaPolEvaluation ex = GgaXPbePolFunctor{}(rho, sigma);
    GgaPolEvaluation ec = GgaCPbePolFunctor{}(rho, sigma);
    return {0.75 * ex.eps + 1.0 * ec.eps,
            {0.75 * ex.vrho[0] + 1.0 * ec.vrho[0], 0.75 * ex.vrho[1] + 1.0 * ec.vrho[1]},
            {0.75 * ex.vsigma[0] + 1.0 * ec.vsigma[0],
             0.75 * ex.vsigma[1] + 1.0 * ec.vsigma[1],
             0.75 * ex.vsigma[2] + 1.0 * ec.vsigma[2]}};
  }
};

// ===========================================================================
// Launch helpers
// ===========================================================================

constexpr int kThreads = 256;

[[nodiscard]] Status CudaStatus(cudaError_t error, const char* context) {
  if (error == cudaSuccess) return Status::Ok();
  return Status::IoError(std::string(context) + ": " + cudaGetErrorString(error));
}

template <typename PolFunctor>
[[nodiscard]] Status LaunchLdaPol(const XcGridIn& input, XcGridOut& output,
                                  cudaStream_t stream, bool deterministic) {
  const std::int64_t required_blocks = (input.np + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(std::min<std::int64_t>(required_blocks, 65535));
  LdaPolKernel<PolFunctor><<<blocks, kThreads, 0, stream>>>(
      input.rho, input.w, output.wv_rho, output.exc_per_system,
      input.np, input.point_stride, !deterministic);
  Status status = CudaStatus(cudaGetLastError(), "LdaPolKernel launch");
  if (!status.ok() || !deterministic) return status;
  LdaPolDeterministicEnergyKernel<PolFunctor><<<1, 1, 0, stream>>>(
      input.rho, input.w, output.exc_per_system, input.np, input.point_stride);
  return CudaStatus(cudaGetLastError(), "LdaPolDeterministicEnergyKernel launch");
}

template <typename PolFunctor>
[[nodiscard]] Status LaunchGgaPol(const XcGridIn& input, XcGridOut& output,
                                  cudaStream_t stream, bool deterministic) {
  const std::int64_t required_blocks = (input.np + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(std::min<std::int64_t>(required_blocks, 65535));
  GgaPolKernel<PolFunctor><<<blocks, kThreads, 0, stream>>>(
      input.rho, input.grad, input.w, output.wv_rho, output.wv_grad,
      output.exc_per_system, input.np, input.point_stride, !deterministic);
  Status status = CudaStatus(cudaGetLastError(), "GgaPolKernel launch");
  if (!status.ok() || !deterministic) return status;
  GgaPolDeterministicEnergyKernel<PolFunctor><<<1, 1, 0, stream>>>(
      input.rho, input.grad, input.w, output.exc_per_system, input.np, input.point_stride);
  return CudaStatus(cudaGetLastError(), "GgaPolDeterministicEnergyKernel launch");
}

}  // namespace

// ===========================================================================
// Dispatch
// ===========================================================================

Status LaunchTier0Pol(const XcSpec& spec, const XcGridIn& input,
                      XcGridOut& output, cudaStream_t stream) {
  switch (spec.terms[0].functional) {
    // LDA functionals
    case Functional::kLdaPw92:
      return LaunchLdaPol<LdaPw92PolFunctor>(input, output, stream, spec.deterministic);
    case Functional::kSvwn5:
      return LaunchLdaPol<Svwn5PolFunctor>(input, output, stream, spec.deterministic);
    // GGA functionals
    case Functional::kPbe:
      return LaunchGgaPol<PbePolFunctor>(input, output, stream, spec.deterministic);
    case Functional::kPbeSol:
      return LaunchGgaPol<PbeSolPolFunctor>(input, output, stream, spec.deterministic);
    case Functional::kRevPbe:
      return LaunchGgaPol<RevPbePolFunctor>(input, output, stream, spec.deterministic);
    case Functional::kRpbe:
      return LaunchGgaPol<RpbePolFunctor>(input, output, stream, spec.deterministic);
    case Functional::kBlyp:
      return LaunchGgaPol<BlypPolFunctor>(input, output, stream, spec.deterministic);
    case Functional::kB3lyp:
      return LaunchGgaPol<B3lypPolFunctor>(input, output, stream, spec.deterministic);
    case Functional::kPbe0:
      return LaunchGgaPol<Pbe0PolFunctor>(input, output, stream, spec.deterministic);
    default:
      return Status::Unimplemented("Functional not yet implemented in Tier-0 polarized");
  }
}

}  // namespace tides::grid::xc::tier0
