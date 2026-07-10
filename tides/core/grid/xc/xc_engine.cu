#include "grid/xc/xc_arena.hpp"
#include "grid/xc/xc_engine.hpp"
#include "grid/xc/functionals/lda_pw92.cuh"
#include "grid/xc/functionals/lda_slater.cuh"
#include "grid/xc/kernels/xc_gga_kernel.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

namespace tides::grid::xc {
namespace {

constexpr std::size_t kDeviceAlignment = 256;
constexpr std::size_t kPointPadding = 512;

[[nodiscard]] Status CudaStatus(cudaError_t error, const char* context) {
  if (error == cudaSuccess) return Status::Ok();
  return Status::IoError(std::string(context) + ": " + cudaGetErrorString(error));
}

[[nodiscard]] bool IsAligned(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % kDeviceAlignment == 0;
}

[[nodiscard]] bool IsLdaPw92(const XcSpec& spec) {
  return spec.family == Family::kLda && spec.nspin == 1 &&
         spec.precision == PrecisionPolicy::kFloat64 && spec.terms.size() == 1 &&
         spec.terms[0].functional == Functional::kLdaPw92 &&
         spec.terms[0].coefficient == 1.0;
}

[[nodiscard]] bool IsPbe(const XcSpec& spec) {
  return spec.family == Family::kGga && spec.nspin == 1 &&
         spec.precision == PrecisionPolicy::kFloat64 && spec.terms.size() == 1 &&
         spec.terms[0].functional == Functional::kPbe &&
         spec.terms[0].coefficient == 1.0;
}

__global__ void LdaPw92Kernel(const double* __restrict__ rho,
                              const double* __restrict__ weights,
                              double* __restrict__ wv_rho,
                              double* __restrict__ exc,
                              std::int64_t np, bool fast_reduction) {
  double local_energy = 0.0;
  for (std::int64_t point = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x;
       point < np;
       point += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const double density = rho[point];
    const double eps = LdaSlater::Eps(density) + LdaPw92::EpsCorrelation(density);
    const double vrho = LdaSlater::V(density) + LdaPw92::VCorrelation(density);
    wv_rho[point] = weights[point] * vrho;
    if (fast_reduction) local_energy += weights[point] * density * eps;
  }
  if (!fast_reduction) return;
  for (int offset = 16; offset > 0; offset /= 2)
    local_energy += __shfl_down_sync(0xffffffff, local_energy, offset);
  if ((threadIdx.x & 31) == 0) atomicAdd(exc, local_energy);
}

__global__ void LdaPw92DeterministicEnergyKernel(const double* __restrict__ rho,
                                                  const double* __restrict__ weights,
                                                  double* __restrict__ exc,
                                                  std::int64_t np) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  double energy = 0.0;
  for (std::int64_t point = 0; point < np; ++point) {
    const double density = rho[point];
    const double eps = LdaSlater::Eps(density) + LdaPw92::EpsCorrelation(density);
    energy += weights[point] * density * eps;
  }
  exc[0] = energy;
}

}  // namespace

Status XcEval(const XcSpec& spec, const XcGridIn& input, XcGridOut& output,
              cudaStream_t stream) {
  const bool is_lda_pw92 = IsLdaPw92(spec);
  const bool is_pbe = IsPbe(spec);
  if (!is_lda_pw92 && !is_pbe) {
    return Status::Unimplemented(
        "Tier-0 supports only unpolarized LDA-PW92 or PBE in FP64");
  }
  if (input.np < 0 || input.point_stride < input.np || input.nsys != 1) {
    return Status::InvalidArgument(
        "Tier-0 requires one system, np >= 0, and point_stride >= np");
  }
  if (output.exc_per_system == nullptr || !IsAligned(output.exc_per_system)) {
    return Status::InvalidArgument("exc_per_system must be a 256-byte-aligned device pointer");
  }
  if (input.np == 0) {
    return CudaStatus(cudaMemsetAsync(output.exc_per_system, 0, sizeof(double), stream),
                      "cudaMemsetAsync exc_per_system");
  }
  if (input.rho == nullptr || input.w == nullptr || output.wv_rho == nullptr ||
      !IsAligned(input.rho) || !IsAligned(input.w) || !IsAligned(output.wv_rho)) {
    return Status::InvalidArgument(
        "rho, weights, and wv_rho must be non-null 256-byte-aligned device pointers");
  }
  if (is_pbe &&
      (input.grad == nullptr || output.wv_grad == nullptr ||
       !IsAligned(input.grad) || !IsAligned(output.wv_grad))) {
    return Status::InvalidArgument(
        "PBE requires 256-byte-aligned grad and wv_grad device pointers");
  }

  cudaError_t error = cudaMemsetAsync(output.exc_per_system, 0, sizeof(double), stream);
  if (error != cudaSuccess) return CudaStatus(error, "cudaMemsetAsync exc_per_system");
  if (is_pbe) return LaunchPbeGgaKernel(input, output, stream, spec.deterministic);
  constexpr int kThreads = 256;
  const std::int64_t required_blocks = (input.np + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(std::min<std::int64_t>(required_blocks, 65535));
  LdaPw92Kernel<<<blocks, kThreads, 0, stream>>>(input.rho, input.w, output.wv_rho,
                                                   output.exc_per_system, input.np,
                                                   !spec.deterministic);
  Status status = CudaStatus(cudaGetLastError(), "LdaPw92Kernel launch");
  if (!status.ok() || !spec.deterministic) return status;
  LdaPw92DeterministicEnergyKernel<<<1, 1, 0, stream>>>(
      input.rho, input.w, output.exc_per_system, input.np);
  return CudaStatus(cudaGetLastError(), "LdaPw92DeterministicEnergyKernel launch");
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
}

Status XcArena::Release(cudaStream_t stream) {
  if (impl_ == nullptr) return Status::Ok();
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

}  // namespace tides::grid::xc
