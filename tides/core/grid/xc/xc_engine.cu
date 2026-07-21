#include "grid/xc/xc_arena.hpp"
#include "grid/xc/xc_engine.hpp"
#include "grid/xc/functional_dispatch.hpp"
#include "grid/xc/kernels/xc_gga_kernel.hpp"
#include "grid/xc/tier2/cpu_fallback.hpp"

#ifdef TIDES_HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tides::grid::xc {
namespace {

#ifdef TIDES_HAVE_CUDA

constexpr std::size_t kDeviceAlignment = 256;
constexpr std::size_t kPointPadding = 512;

[[nodiscard]] Status CudaStatus(cudaError_t error, const char* context) {
  if (error == cudaSuccess) return Status::Ok();
  return Status::IoError(std::string(context) + ": " + cudaGetErrorString(error));
}

[[nodiscard]] bool IsAligned(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % kDeviceAlignment == 0;
}

[[nodiscard]] bool IsTier0Functional(const XcSpec& spec) {
  if (spec.terms.size() != 1 || spec.terms[0].coefficient != 1.0) {
    return false;
  }
  switch (spec.terms[0].functional) {
    case Functional::kLdaPw92:
    case Functional::kSvwn5:
      return spec.family == Family::kLda;
    case Functional::kPbe:
    case Functional::kPbeSol:
    case Functional::kRevPbe:
    case Functional::kRpbe:
    case Functional::kBlyp:
    case Functional::kB3lyp:
    case Functional::kPbe0:
      return spec.family == Family::kGga;
    default:
      return false;
  }
}

[[nodiscard]] bool IsTier1Functional(const XcSpec& spec) {
  if (spec.terms.size() != 1 || spec.terms[0].coefficient != 1.0) {
    return false;
  }
  switch (spec.terms[0].functional) {
    case Functional::kTpss:
    case Functional::kScan:
    case Functional::kR2scan:
    case Functional::kM06_2x:
      return spec.family == Family::kMgga;
    case Functional::kHse06:
    case Functional::kWb97x:
      return spec.family == Family::kRsh;
    default:
      return false;
  }
}

[[nodiscard]] bool IsSupportedFp64(const XcSpec& spec) {
  if (spec.precision == PrecisionPolicy::kFloat64) {
    if (spec.nspin != 1 && spec.nspin != 2) return false;
    return IsTier0Functional(spec) || IsTier1Functional(spec);
  }
  // T-X4.4: FP32 mid-SCF path supports Tier-0 LDA/GGA only.
  // mGGA/RSH remain FP64-only due to α/erfc hazard complexity.
  if (spec.precision == PrecisionPolicy::kFloat32MidScf) {
    if (spec.nspin != 1) return false;  // FP32 path is unpolarized only for now
    return IsTier0Functional(spec);
  }
  return false;
}

#endif  // TIDES_HAVE_CUDA

}  // namespace

Status XcEval(const XcSpec& spec, const XcGridIn& input, XcGridOut& output,
              cudaStream_t stream) {
#ifdef TIDES_HAVE_CUDA
  if (!IsSupportedFp64(spec)) {
    if (spec.nspin == 1 && input.nsys == 1 &&
        spec.precision == PrecisionPolicy::kFloat64) {
      return tier2::LaunchCpuFallback(spec, input, output, stream);
    }
    return Status::Unimplemented(
        "XC engine supports FP64 LDA/GGA (Tier-0) and mGGA/RSH (Tier-1) "
        "functionals with nspin=1 or nspin=2. Unsupported functionals require "
        "Tier-2 CPU fallback (T-X4.3), which is only available for nspin=1, "
        "nsys=1, and FP64.");
  }
  if (input.np < 0 || input.point_stride < input.np || input.nsys < 1) {
    return Status::InvalidArgument(
        "Tier-0 requires nsys >= 1, np >= 0, and point_stride >= np");
  }
  if (output.exc_per_system == nullptr || !IsAligned(output.exc_per_system)) {
    return Status::InvalidArgument("exc_per_system must be a 256-byte-aligned device pointer");
  }
  if (input.np == 0) {
    return CudaStatus(cudaMemsetAsync(output.exc_per_system, 0,
                      static_cast<std::size_t>(input.nsys) * sizeof(double), stream),
                      "cudaMemsetAsync exc_per_system");
  }
  if (input.rho == nullptr || input.w == nullptr || output.wv_rho == nullptr ||
      !IsAligned(input.rho) || !IsAligned(input.w) || !IsAligned(output.wv_rho)) {
    return Status::InvalidArgument(
        "rho, weights, and wv_rho must be non-null 256-byte-aligned device pointers");
  }
  const bool needs_grad = (spec.family != Family::kLda);
  if (needs_grad &&
      (input.grad == nullptr || output.wv_grad == nullptr ||
       !IsAligned(input.grad) || !IsAligned(output.wv_grad))) {
    return Status::InvalidArgument(
        "GGA functionals require 256-byte-aligned grad and wv_grad device pointers");
  }
  const bool needs_tau = (spec.family == Family::kMgga);
  if (needs_tau &&
      (input.tau == nullptr || output.wv_tau == nullptr ||
       !IsAligned(input.tau) || !IsAligned(output.wv_tau))) {
    return Status::InvalidArgument(
        "mGGA functionals require 256-byte-aligned tau and wv_tau device pointers");
  }

  cudaError_t error = cudaMemsetAsync(output.exc_per_system, 0,
                      static_cast<std::size_t>(input.nsys) * sizeof(double), stream);
  if (error != cudaSuccess) return CudaStatus(error, "cudaMemsetAsync exc_per_system");
  return LaunchXcFunctional(spec, input, output, stream);
#else
  (void)spec; (void)input; (void)output; (void)stream;
  return Status::Unimplemented("XcEval device-resident path requires CUDA");
#endif
}

class XcArena::Impl {
 public:
  std::size_t capacity = 0;
  int nspin = 0;
  int nsys = 0;
  bool has_grad = false;
  bool has_tau = false;
  double* rho = nullptr;
  double* weights = nullptr;
  double* grad = nullptr;
  double* tau = nullptr;
  double* wv_rho = nullptr;
  double* wv_grad = nullptr;
  double* wv_tau = nullptr;
  double* exc_per_system = nullptr;
  std::int64_t* sys_offsets = nullptr;
};

XcArena::XcArena() : impl_(std::make_unique<Impl>()) {}
XcArena::XcArena(XcArena&&) noexcept = default;
XcArena& XcArena::operator=(XcArena&&) noexcept = default;

XcArena::~XcArena() {
  if (impl_ != nullptr) (void)Release(nullptr);
}

Status XcArena::Reserve(std::size_t np, int nspin, bool need_grad, bool need_tau,
                        int nsys, cudaStream_t stream) {
#ifdef TIDES_HAVE_CUDA
  if (np == 0 || nspin <= 0 || nsys <= 0) {
    return Status::InvalidArgument("XcArena requires positive shape dimensions");
  }
  if (impl_->capacity >= np && impl_->nspin == nspin && impl_->nsys == nsys &&
      (!need_grad || impl_->has_grad) && (!need_tau || impl_->has_tau)) {
    return Status::Ok();
  }

  Status release_status = Release(stream);
  if (!release_status.ok()) return release_status;
  const std::size_t padded_np = ((np + kPointPadding - 1) / kPointPadding) * kPointPadding;
  const auto allocate = [stream](void** pointer, std::size_t bytes,
                                 const char* name) -> Status {
    return CudaStatus(cudaMallocAsync(pointer, bytes, stream), name);
  };
  const auto cleanup = [this, stream]() { (void)Release(stream); };

  Status status = allocate(reinterpret_cast<void**>(&impl_->rho),
                           static_cast<std::size_t>(nspin) * padded_np * sizeof(double),
                           "cudaMallocAsync xc rho");
  if (!status.ok()) return status;
  status = allocate(reinterpret_cast<void**>(&impl_->weights), padded_np * sizeof(double),
                    "cudaMallocAsync xc weights");
  if (!status.ok()) { cleanup(); return status; }
  status = allocate(reinterpret_cast<void**>(&impl_->wv_rho),
                    static_cast<std::size_t>(nspin) * padded_np * sizeof(double),
                    "cudaMallocAsync xc wv_rho");
  if (!status.ok()) { cleanup(); return status; }
  status = allocate(reinterpret_cast<void**>(&impl_->exc_per_system),
                    static_cast<std::size_t>(nsys) * sizeof(double),
                    "cudaMallocAsync xc energy");
  if (!status.ok()) { cleanup(); return status; }
  status = allocate(reinterpret_cast<void**>(&impl_->sys_offsets),
                    static_cast<std::size_t>(nsys + 1) * sizeof(std::int64_t),
                    "cudaMallocAsync xc system offsets");
  if (!status.ok()) { cleanup(); return status; }
  // Initialize sys_offsets to {0, padded_np, 2*padded_np, ...} as default.
  {
    std::vector<std::int64_t> host_offsets(nsys + 1);
    for (int s = 0; s <= nsys; ++s)
      host_offsets[s] = static_cast<std::int64_t>(s) * padded_np;
    cudaError_t err = cudaMemcpyAsync(impl_->sys_offsets, host_offsets.data(),
                                      (nsys + 1) * sizeof(std::int64_t),
                                      cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { cleanup(); return CudaStatus(err, "cudaMemcpyAsync sys_offsets init"); }
  }
  if (need_grad) {
    status = allocate(reinterpret_cast<void**>(&impl_->grad),
                      static_cast<std::size_t>(nspin) * 3 * padded_np * sizeof(double),
                      "cudaMallocAsync xc grad");
    if (!status.ok()) { cleanup(); return status; }
    status = allocate(reinterpret_cast<void**>(&impl_->wv_grad),
                      static_cast<std::size_t>(nspin) * 3 * padded_np * sizeof(double),
                      "cudaMallocAsync xc wv_grad");
    if (!status.ok()) { cleanup(); return status; }
  }
  if (need_tau) {
    status = allocate(reinterpret_cast<void**>(&impl_->tau),
                      static_cast<std::size_t>(nspin) * padded_np * sizeof(double),
                      "cudaMallocAsync xc tau");
    if (!status.ok()) { cleanup(); return status; }
    status = allocate(reinterpret_cast<void**>(&impl_->wv_tau),
                      static_cast<std::size_t>(nspin) * padded_np * sizeof(double),
                      "cudaMallocAsync xc wv_tau");
    if (!status.ok()) { cleanup(); return status; }
  }
  impl_->capacity = padded_np;
  impl_->nspin = nspin;
  impl_->nsys = nsys;
  impl_->has_grad = need_grad;
  impl_->has_tau = need_tau;
  return Status::Ok();
#else
  (void)np; (void)nspin; (void)need_grad; (void)need_tau; (void)nsys; (void)stream;
  return Status::Unimplemented("XcArena requires CUDA");
#endif
}

Status XcArena::Release(cudaStream_t stream) {
  if (impl_ == nullptr) return Status::Ok();
#ifdef TIDES_HAVE_CUDA
  Status first_error = Status::Ok();
  const auto free_pointer = [&first_error, stream](auto*& pointer, const char* name) {
    if (pointer == nullptr) return;
    const Status status = CudaStatus(cudaFreeAsync(pointer, stream), name);
    if (status.ok()) {
      pointer = nullptr;
    } else if (first_error.ok()) {
      first_error = status;
    }
  };
  free_pointer(impl_->rho, "cudaFreeAsync xc rho");
  free_pointer(impl_->weights, "cudaFreeAsync xc weights");
  free_pointer(impl_->grad, "cudaFreeAsync xc grad");
  free_pointer(impl_->tau, "cudaFreeAsync xc tau");
  free_pointer(impl_->wv_rho, "cudaFreeAsync xc wv_rho");
  free_pointer(impl_->wv_grad, "cudaFreeAsync xc wv_grad");
  free_pointer(impl_->wv_tau, "cudaFreeAsync xc wv_tau");
  free_pointer(impl_->exc_per_system, "cudaFreeAsync xc energy");
  free_pointer(impl_->sys_offsets, "cudaFreeAsync xc system offsets");
  if (!first_error.ok()) return first_error;
#else
  (void)stream;
#endif
  impl_->capacity = 0;
  impl_->nspin = 0;
  impl_->nsys = 0;
  impl_->has_grad = false;
  impl_->has_tau = false;
  return Status::Ok();
}

std::size_t XcArena::capacity() const { return impl_->capacity; }
double* XcArena::rho() const { return impl_->rho; }
double* XcArena::weights() const { return impl_->weights; }
double* XcArena::grad() const { return impl_->grad; }
double* XcArena::tau() const { return impl_->tau; }
double* XcArena::wv_rho() const { return impl_->wv_rho; }
double* XcArena::wv_grad() const { return impl_->wv_grad; }
double* XcArena::wv_tau() const { return impl_->wv_tau; }
double* XcArena::exc_per_system() const { return impl_->exc_per_system; }
std::int64_t* XcArena::sys_offsets() const { return impl_->sys_offsets; }

// ---- Host-oriented convenience API (for GTO molecule driver) ----

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
    case XcFunctionalId::kRpbe:
    case XcFunctionalId::kBlyp:
    case XcFunctionalId::kB3lypLocal:
    case XcFunctionalId::kPbe0Local:
      return true;
    default:
      return false;
  }
}

bool XcEvalHost(const HostXcSpec& spec, const HostXcGridIn& in,
                HostXcGridOut& out, std::string& error_msg) {
  if (in.np == 0) {
    error_msg = "XcEvalHost: np = 0";
    return false;
  }
  if (!in.rho || !out.vxc || !out.eps_xc) {
    error_msg = "XcEvalHost: null pointers in input/output";
    return false;
  }
  if (!IsTier0(spec.id)) {
    error_msg = "XcEvalHost: functional " + XcFunctionalName(spec.id) +
                " not yet implemented in Tier-0";
    return false;
  }

  auto t0 = std::chrono::steady_clock::now();
  double energy = 0.0;

  if (spec.family == XcFamily::kLda) {
    for (std::size_t i = 0; i < in.np; ++i) {
      const double n = std::max(0.0, in.rho[i]);
      if (n < 1e-14) {
        out.vxc[i] = 0.0;
        out.eps_xc[i] = 0.0;
        continue;
      }
      const auto eval = LdaSlater::Eval(n) + LdaPw92::Eval(n);
      out.vxc[i] = eval.vrho;
      out.eps_xc[i] = eval.eps;
      energy += in.grid_weight * eval.eps * n;
    }
  } else if (spec.family == XcFamily::kGga) {
    if (!in.grad_rho_x || !in.grad_rho_y || !in.grad_rho_z) {
      error_msg = "XcEvalHost: GGA requires grad_rho_x/y/z";
      return false;
    }
    for (std::size_t i = 0; i < in.np; ++i) {
      const double n = std::max(0.0, in.rho[i]);
      if (n < 1e-14) {
        out.vxc[i] = 0.0;
        if (out.vsigma) out.vsigma[i] = 0.0;
        out.eps_xc[i] = 0.0;
        continue;
      }
      const double gx = in.grad_rho_x[i];
      const double gy = in.grad_rho_y[i];
      const double gz = in.grad_rho_z[i];
      const double sigma = gx * gx + gy * gy + gz * gz;
      const auto eval = GgaPbeStandard::Eval(n, sigma);
      out.vxc[i] = eval.vrho;
      if (out.vsigma) out.vsigma[i] = eval.vsigma;
      out.eps_xc[i] = eval.eps;
      energy += in.grid_weight * eval.eps * n;
    }
  } else {
    error_msg = "XcEvalHost: family " + std::to_string(static_cast<int>(spec.family)) +
                " not supported";
    return false;
  }

  auto t1 = std::chrono::steady_clock::now();
  out.kernel_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  out.xc_energy = energy;
  return true;
}

}  // namespace tides::grid::xc
