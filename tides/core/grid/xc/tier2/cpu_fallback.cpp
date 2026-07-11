// T-X4.3: Tier-2 synchronous CPU libxc fallback for exotic/mixed XC specs.
//
// Scope: nspin=1, nsys=1, FP64.  Copies device inputs to host, evaluates with
// libxc, and uploads weighted potentials and integrated energy.

#include "grid/xc/tier2/cpu_fallback.hpp"
#include "grid/libxc_wrapper.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tides::grid::xc::tier2 {

namespace {

using tides::grid::kLibxc_GGA_C_LYP;
using tides::grid::kLibxc_GGA_C_PBE;
using tides::grid::kLibxc_GGA_C_PBE_SOL;
using tides::grid::kLibxc_GGA_X_B88;
using tides::grid::kLibxc_GGA_X_PBE;
using tides::grid::kLibxc_GGA_X_PBE_R;
using tides::grid::kLibxc_GGA_X_PBE_SOL;
using tides::grid::kLibxc_GGA_X_RPBE;
using tides::grid::kLibxc_HYB_GGA_XC_HSE06;
using tides::grid::kLibxc_HYB_GGA_XC_WB97X;
using tides::grid::kLibxc_HYB_MGGA_X_M06_2X;
using tides::grid::kLibxc_LDA_C_PW;
using tides::grid::kLibxc_LDA_C_VWN;
using tides::grid::kLibxc_LDA_X;
using tides::grid::kLibxc_MGGA_C_M06_2X;
using tides::grid::kLibxc_MGGA_C_R2SCAN;
using tides::grid::kLibxc_MGGA_C_SCAN;
using tides::grid::kLibxc_MGGA_C_TPSS;
using tides::grid::kLibxc_MGGA_X_R2SCAN;
using tides::grid::kLibxc_MGGA_X_SCAN;
using tides::grid::kLibxc_MGGA_X_TPSS;
using tides::grid::LibxcFunctional;

struct LibxcSubterm {
  int id;
  double coefficient;
};

std::vector<LibxcSubterm> FunctionalLibxcTerms(Functional functional) {
  switch (functional) {
    case Functional::kLdaPw92:
      return {{kLibxc_LDA_X, 1.0}, {kLibxc_LDA_C_PW, 1.0}};
    case Functional::kSvwn5:
      return {{kLibxc_LDA_X, 1.0}, {kLibxc_LDA_C_VWN, 1.0}};
    case Functional::kPbe:
      return {{kLibxc_GGA_X_PBE, 1.0}, {kLibxc_GGA_C_PBE, 1.0}};
    case Functional::kPbeSol:
      return {{kLibxc_GGA_X_PBE_SOL, 1.0}, {kLibxc_GGA_C_PBE_SOL, 1.0}};
    case Functional::kRevPbe:
      return {{kLibxc_GGA_X_PBE_R, 1.0}, {kLibxc_GGA_C_PBE, 1.0}};
    case Functional::kRpbe:
      return {{kLibxc_GGA_X_RPBE, 1.0}, {kLibxc_GGA_C_PBE, 1.0}};
    case Functional::kBlyp:
      return {{kLibxc_GGA_X_B88, 1.0}, {kLibxc_GGA_C_LYP, 1.0}};
    case Functional::kB3lyp:
      return {{kLibxc_LDA_X, 0.08}, {kLibxc_GGA_X_B88, 0.72},
              {kLibxc_LDA_C_VWN, 0.19}, {kLibxc_GGA_C_LYP, 0.81}};
    case Functional::kPbe0:
      return {{kLibxc_GGA_X_PBE, 0.75}, {kLibxc_GGA_C_PBE, 1.0}};
    case Functional::kHse06:
      return {{kLibxc_HYB_GGA_XC_HSE06, 1.0}};
    case Functional::kWb97x:
      return {{kLibxc_HYB_GGA_XC_WB97X, 1.0}};
    case Functional::kTpss:
      return {{kLibxc_MGGA_X_TPSS, 1.0}, {kLibxc_MGGA_C_TPSS, 1.0}};
    case Functional::kScan:
      return {{kLibxc_MGGA_X_SCAN, 1.0}, {kLibxc_MGGA_C_SCAN, 1.0}};
    case Functional::kR2scan:
      return {{kLibxc_MGGA_X_R2SCAN, 1.0}, {kLibxc_MGGA_C_R2SCAN, 1.0}};
    case Functional::kM06_2x:
      return {{kLibxc_HYB_MGGA_X_M06_2X, 1.0}, {kLibxc_MGGA_C_M06_2X, 1.0}};
    default:
      return {};
  }
}

[[nodiscard]] Status CudaStatus(cudaError_t error, const char* context) {
  if (error == cudaSuccess) return Status::Ok();
  return Status::IoError(std::string(context) + ": " + cudaGetErrorString(error));
}

}  // namespace

Status LaunchCpuFallback(const XcSpec& spec, const XcGridIn& input,
                         XcGridOut& output, cudaStream_t stream) {
  if (spec.nspin != 1) {
    return Status::Unimplemented(
        "Tier-2 CPU fallback supports nspin=1 only");
  }
  if (input.nsys != 1) {
    return Status::Unimplemented(
        "Tier-2 CPU fallback supports nsys=1 only");
  }
  if (spec.precision != PrecisionPolicy::kFloat64) {
    return Status::Unimplemented(
        "Tier-2 CPU fallback supports FP64 only");
  }
  if (spec.terms.empty()) {
    return Status::InvalidArgument(
        "Tier-2 CPU fallback requires at least one XC term");
  }
  const std::int64_t np = input.np;
  if (np < 0) {
    return Status::InvalidArgument(
        "Tier-2 CPU fallback requires np >= 0");
  }
  if (input.point_stride < np) {
    return Status::InvalidArgument(
        "Tier-2 CPU fallback requires point_stride >= np");
  }
  if (input.rho == nullptr || input.w == nullptr ||
      output.wv_rho == nullptr || output.exc_per_system == nullptr) {
    return Status::InvalidArgument(
        "Tier-2 CPU fallback requires non-null rho, weights, wv_rho, and exc");
  }

  const bool need_grad = (spec.family != Family::kLda);
  const bool need_tau = (spec.family == Family::kMgga);
  if (need_grad && input.grad == nullptr) {
    return Status::InvalidArgument(
        "Tier-2 CPU fallback GGA/RSH/mGGA requires grad");
  }
  if (need_tau && input.tau == nullptr) {
    return Status::InvalidArgument(
        "Tier-2 CPU fallback mGGA requires tau");
  }
  if (need_grad && output.wv_grad == nullptr) {
    return Status::InvalidArgument(
        "Tier-2 CPU fallback GGA/RSH/mGGA requires wv_grad");
  }
  if (need_tau && output.wv_tau == nullptr) {
    return Status::InvalidArgument(
        "Tier-2 CPU fallback mGGA requires wv_tau");
  }

  // Wait for the upstream device work producing rho/grad/tau to complete.
  Status status = CudaStatus(cudaStreamSynchronize(stream),
                             "Tier-2 CPU fallback stream sync");
  if (!status.ok()) return status;

  std::vector<double> host_rho(static_cast<std::size_t>(np));
  std::vector<double> host_weights(static_cast<std::size_t>(np));
  std::vector<double> host_grad(need_grad ? static_cast<std::size_t>(3 * np) : 0);
  std::vector<double> host_tau(need_tau ? static_cast<std::size_t>(np) : 0);

  cudaError_t err = cudaMemcpy(host_rho.data(), input.rho,
                                static_cast<std::size_t>(np) * sizeof(double),
                                cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return CudaStatus(err, "cpu_fallback D2H rho");
  err = cudaMemcpy(host_weights.data(), input.w,
                   static_cast<std::size_t>(np) * sizeof(double),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) return CudaStatus(err, "cpu_fallback D2H weights");
  if (need_grad) {
    for (int component = 0; component < 3; ++component) {
      err = cudaMemcpy(host_grad.data() + component * np,
                       input.grad + component * input.point_stride,
                       static_cast<std::size_t>(np) * sizeof(double),
                       cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) return CudaStatus(err, "cpu_fallback D2H grad");
    }
  }
  if (need_tau) {
    err = cudaMemcpy(host_tau.data(), input.tau,
                     static_cast<std::size_t>(np) * sizeof(double),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return CudaStatus(err, "cpu_fallback D2H tau");
  }

  std::vector<double> host_sigma(need_grad ? static_cast<std::size_t>(np) : 0, 0.0);
  if (need_grad) {
    for (std::int64_t point = 0; point < np; ++point) {
      const double gx = host_grad[point];
      const double gy = host_grad[np + point];
      const double gz = host_grad[2 * np + point];
      host_sigma[point] = gx * gx + gy * gy + gz * gz;
    }
  }

  std::vector<double> host_lapl(need_tau ? static_cast<std::size_t>(np) : 0, 0.0);
  std::vector<double> host_eps(static_cast<std::size_t>(np), 0.0);
  std::vector<double> host_vrho(static_cast<std::size_t>(np), 0.0);
  std::vector<double> host_vsigma(need_grad ? static_cast<std::size_t>(np) : 0, 0.0);
  std::vector<double> host_vtau(need_tau ? static_cast<std::size_t>(np) : 0, 0.0);

  for (const auto& term : spec.terms) {
    const auto subterms = FunctionalLibxcTerms(term.functional);
    if (subterms.empty()) {
      return Status::Unimplemented(
          "Tier-2 CPU fallback does not support this Functional");
    }
    for (const auto& subterm : subterms) {
      LibxcFunctional func;
      if (!func.Init(subterm.id, 1)) {
        return Status::IoError(
            "Tier-2 CPU fallback failed to initialize libxc functional");
      }
      const int family = func.Family();
      const double scale = term.coefficient * subterm.coefficient;
      if (family == XC_FAMILY_LDA) {
        const auto res = func.EvalLDA(host_rho, static_cast<std::size_t>(np));
        for (std::int64_t point = 0; point < np; ++point) {
          host_eps[point] += scale * res.eps_xc[point];
          host_vrho[point] += scale * res.vrho[point];
        }
      } else if (family == XC_FAMILY_GGA || family == XC_FAMILY_HYB_GGA) {
        if (spec.family == Family::kLda) {
          return Status::Unimplemented(
              "Tier-2 CPU fallback cannot place GGA subterm in LDA family");
        }
        const auto res = func.EvalGGA(host_rho, host_sigma,
                                      static_cast<std::size_t>(np));
        for (std::int64_t point = 0; point < np; ++point) {
          host_eps[point] += scale * res.eps_xc[point];
          host_vrho[point] += scale * res.vrho[point];
          host_vsigma[point] += scale * res.vsigma[point];
        }
      } else if (family == XC_FAMILY_MGGA || family == XC_FAMILY_HYB_MGGA) {
        if (spec.family != Family::kMgga) {
          return Status::Unimplemented(
              "Tier-2 CPU fallback can place mGGA subterm only in mGGA family");
        }
        const auto res = func.EvalMGGA(host_rho, host_sigma, host_lapl,
                                       host_tau, static_cast<std::size_t>(np));
        for (std::int64_t point = 0; point < np; ++point) {
          host_eps[point] += scale * res.eps_xc[point];
          host_vrho[point] += scale * res.vrho[point];
          host_vsigma[point] += scale * res.vsigma[point];
          host_vtau[point] += scale * res.vtau[point];
        }
      } else {
        return Status::Unimplemented(
            "Tier-2 CPU fallback unsupported libxc family");
      }
    }
  }

  std::vector<double> host_wv_rho(static_cast<std::size_t>(np), 0.0);
  std::vector<double> host_wv_grad(need_grad ? static_cast<std::size_t>(3 * np) : 0);
  std::vector<double> host_wv_tau(need_tau ? static_cast<std::size_t>(np) : 0);
  double energy = 0.0;
  for (std::int64_t point = 0; point < np; ++point) {
    const double weight = host_weights[point];
    const double density = host_rho[point];
    host_wv_rho[point] = weight * host_vrho[point];
    if (need_grad) {
      const double two_w_vsigma = 2.0 * weight * host_vsigma[point];
      host_wv_grad[point] = two_w_vsigma * host_grad[point];
      host_wv_grad[np + point] = two_w_vsigma * host_grad[np + point];
      host_wv_grad[2 * np + point] = two_w_vsigma * host_grad[2 * np + point];
    }
    if (need_tau) {
      host_wv_tau[point] = weight * host_vtau[point];
    }
    energy += weight * density * host_eps[point];
  }

  err = cudaMemcpy(output.wv_rho, host_wv_rho.data(),
                   static_cast<std::size_t>(np) * sizeof(double),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) return CudaStatus(err, "cpu_fallback H2D wv_rho");
  if (need_grad) {
    for (int component = 0; component < 3; ++component) {
      err = cudaMemcpy(output.wv_grad + component * input.point_stride,
                       host_wv_grad.data() + component * np,
                       static_cast<std::size_t>(np) * sizeof(double),
                       cudaMemcpyHostToDevice);
      if (err != cudaSuccess) return CudaStatus(err, "cpu_fallback H2D wv_grad");
    }
  }
  if (need_tau) {
    err = cudaMemcpy(output.wv_tau, host_wv_tau.data(),
                     static_cast<std::size_t>(np) * sizeof(double),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return CudaStatus(err, "cpu_fallback H2D wv_tau");
  }
  err = cudaMemcpy(output.exc_per_system, &energy, sizeof(double),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) return CudaStatus(err, "cpu_fallback H2D exc");

  return CudaStatus(cudaStreamSynchronize(stream),
                    "Tier-2 CPU fallback output sync");
}

}  // namespace tides::grid::xc::tier2
