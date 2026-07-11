#pragma once

// Test-only helper for evaluating the LDA-PW92 XC functional on a host
// UniformGrid3D using the device-resident XcEval engine.  This is a diagnostic
// convenience, not a production code path: it allocates an XcArena, copies the
// host density and weights to device, launches XcEval, and reads back V_xc and
// the integrated energy.  It does not bypass the XcEngine contract.

#include <chrono>
#include <vector>

#include "grid/dual_grid.hpp"
#include "grid/xc/functional_dispatch.hpp"
#include "grid/xc/xc_arena.hpp"
#include "grid/xc/xc_engine.hpp"

#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>
#else
struct CUstream_st;
using cudaStream_t = CUstream_st*;
#endif

namespace tides::grid::xc {

struct HostXcResult {
  std::vector<double> vxc;     // V_xc(r) on the grid
  std::vector<double> eps_xc;  // eps_xc(r) from CPU reference functor
  double xc_energy = 0.0;      // integrated E_xc
  double kernel_ms = 0.0;      // end-to-end device time (ms)
};

// Evaluate LDA-PW92 on a host grid through the XcEval engine.
// rho must have size grid.total_points() and be non-negative.
inline HostXcResult RunLdaXcOnHostGrid(const UniformGrid3D& grid,
                                       const std::vector<double>& rho,
                                       cudaStream_t stream = nullptr) {
  HostXcResult result;
  const std::size_t N = grid.total_points();
  if (N == 0 || rho.size() != N) return result;

  const auto [h0, h1, h2] = grid.h;
  const double dv = h0 * h1 * h2;

  std::vector<double> weights(N, dv);

  XcArena arena;
  if (!arena.Reserve(static_cast<std::size_t>(N), 1, false, false, 1, stream).ok()) return result;

  cudaMemcpyAsync(arena.rho(), rho.data(), N * sizeof(double),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(arena.weights(), weights.data(), N * sizeof(double),
                  cudaMemcpyHostToDevice, stream);

  XcSpec spec;
  spec.family = Family::kLda;
  spec.nspin = 1;
  spec.terms = {{Functional::kLdaPw92, 1.0}};
  spec.precision = PrecisionPolicy::kFloat64;
  spec.deterministic = false;

  XcGridIn input{arena.rho(), nullptr, nullptr, arena.weights(),
                 static_cast<std::int64_t>(N),
                 static_cast<std::int64_t>(arena.capacity()),
                 1, arena.sys_offsets()};
  XcGridOut output{arena.wv_rho(), nullptr, nullptr, arena.exc_per_system()};

  const auto t0 = std::chrono::steady_clock::now();
  if (!XcEval(spec, input, output, stream).ok()) {
    cudaStreamSynchronize(stream);
    return result;
  }
  cudaStreamSynchronize(stream);
  const auto t1 = std::chrono::steady_clock::now();

  result.kernel_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::vector<double> wv_rho(N);
  cudaMemcpyAsync(wv_rho.data(), arena.wv_rho(), N * sizeof(double),
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&result.xc_energy, arena.exc_per_system(), sizeof(double),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  result.vxc.resize(N, 0.0);
  result.eps_xc.resize(N, 0.0);
  for (std::size_t i = 0; i < N; ++i) {
    const double n = std::max(0.0, rho[i]);
    const auto eval = LdaPw92Functor::Eval(n);
    result.eps_xc[i] = eval.eps;
    result.vxc[i] = (dv > 0.0) ? wv_rho[i] / dv : 0.0;
  }

  (void)arena.Release(stream);
  return result;
}

}  // namespace tides::grid::xc
