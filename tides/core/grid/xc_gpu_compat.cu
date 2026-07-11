// Compatibility wrapper for the legacy XCEvalLdaCuda interface.
// Re-implemented as a thin call into the new XC engine (T-X1.6 retirement path).
// XCEvalPbeCuda is NOT re-implemented — it was the anti-pattern (CPU libxc + PCIe)
// and is deleted per T-X1.2. The stub in cuda_stubs.cpp returns Unimplemented.

#include "grid/xc_gpu.hpp"
#include "grid/xc/xc_arena.hpp"
#include "grid/xc/xc_engine.hpp"
#include "grid/xc/functional_dispatch.hpp"

#include <cuda_runtime.h>

#include <vector>

namespace tides::grid {

bool XCCudaAvailable() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess) return false;
  return device_count > 0;
}

Result<XCGpuResult> XCEvalLdaCuda(
    const UniformGrid3D& grid,
    const std::vector<double>& rho,
    double /*zeta*/) {
  XCGpuResult result;
  const std::size_t N = grid.total_points();
  if (N == 0 || rho.size() != N) {
    return Status::InvalidArgument("XCEvalLdaCuda: grid/rho size mismatch");
  }

  const auto [h0, h1, h2] = grid.h;
  const double dv = h0 * h1 * h2;
  std::vector<double> weights(N, dv);

  cudaStream_t stream = nullptr;
  xc::XcArena arena;
  auto reserve_status = arena.Reserve(N, 1, false, false, 1, stream);
  if (!reserve_status.ok()) return reserve_status;

  cudaMemcpyAsync(arena.rho(), rho.data(), N * sizeof(double),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(arena.weights(), weights.data(), N * sizeof(double),
                  cudaMemcpyHostToDevice, stream);

  xc::XcSpec spec;
  spec.family = xc::Family::kLda;
  spec.nspin = 1;
  spec.terms = {{xc::Functional::kLdaPw92, 1.0}};
  spec.precision = xc::PrecisionPolicy::kFloat64;

  xc::XcGridIn input{arena.rho(), nullptr, nullptr, arena.weights(),
                     static_cast<std::int64_t>(N),
                     static_cast<std::int64_t>(arena.capacity()),
                     1, arena.sys_offsets()};
  xc::XcGridOut output{arena.wv_rho(), nullptr, nullptr, arena.exc_per_system()};

  auto eval_status = xc::XcEval(spec, input, output, stream);
  if (!eval_status.ok()) {
    (void)arena.Release(stream);
    return eval_status;
  }
  cudaStreamSynchronize(stream);

  result.vxc.resize(N, 0.0);
  result.eps_xc.resize(N, 0.0);
  cudaMemcpyAsync(result.vxc.data(), arena.wv_rho(), N * sizeof(double),
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&result.xc_energy, arena.exc_per_system(), sizeof(double),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  for (std::size_t i = 0; i < N; ++i) {
    const double n = rho[i] > 0.0 ? rho[i] : 0.0;
    const auto eval = xc::LdaSlater::Eval(n) + xc::LdaPw92::Eval(n);
    result.eps_xc[i] = eval.eps;
    result.vxc[i] = (dv > 0.0) ? result.vxc[i] / dv : 0.0;
  }

  result.n_points = N;
  (void)arena.Release(stream);
  return result;
}

}  // namespace tides::grid
