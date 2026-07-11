// T-X2.4: R0 batched layout + CUDA graph capture.
//
// Tests multi-system concatenated grid support and CUDA graph capture
// of the XC evaluation pipeline. Measures launch overhead reduction
// from graph replay vs. individual kernel launches.
//
// Acceptance: 1000 × 30-atom batch: one graph launch per sweep;
//             ≥5× launch-overhead reduction vs loop.
//
// Usage: tides_xc_batched_graph

#include "grid/xc/xc_arena.hpp"
#include "grid/xc/xc_engine.hpp"
#include "grid/xc/kernels/xc_gga_kernel.hpp"
#include "grid/xc/functional_dispatch.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using tides::grid::xc::Family;
using tides::grid::xc::Functional;
using tides::grid::xc::PrecisionPolicy;
using tides::grid::xc::XcArena;
using tides::grid::xc::XcEval;
using tides::grid::xc::XcGridIn;
using tides::grid::xc::XcGridOut;
using tides::grid::xc::XcSpec;
using tides::grid::xc::LaunchXcFunctional;

int Fail(const char* msg) {
  std::fprintf(stderr, "xc_batched_graph: %s\n", msg);
  return 1;
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::printf("SKIP: CUDA not available\n");
    return 77;
  }

  // Simulate 1000 × 30-atom batch: each "system" has a small number of
  // grid points to isolate launch overhead from kernel compute time.
  constexpr int kNsys = 1000;
  constexpr int kPointsPerSys = 10;
  const std::int64_t total_np = static_cast<std::int64_t>(kNsys) * kPointsPerSys;
  const std::int64_t stride = ((total_np + 511) / 512) * 512;

  std::printf("=== T-X2.4: Batched Layout + CUDA Graph Capture ===\n");
  std::printf("Systems: %d, points/system: %d, total points: %ld\n\n",
              kNsys, kPointsPerSys, static_cast<long>(total_np));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  XcArena arena;
  auto reserve_status = arena.Reserve(static_cast<std::size_t>(total_np), 1,
                                      true, false, kNsys, stream);
  if (!reserve_status.ok()) return Fail(reserve_status.message().c_str());

  // Fill density/gradient/weights
  std::vector<double> rho(total_np);
  std::vector<double> weights(total_np);
  std::vector<double> grad(stride * 3, 0.0);
  for (int s = 0; s < kNsys; ++s) {
    for (int p = 0; p < kPointsPerSys; ++p) {
      const std::int64_t idx = static_cast<std::int64_t>(s) * kPointsPerSys + p;
      rho[idx] = 0.05 + 0.001 * static_cast<double>(p % 100);
      weights[idx] = 1.0 / static_cast<double>(kPointsPerSys);
      const double g = 0.02 * static_cast<double>(p % 30);
      grad[idx] = g;
      grad[stride + idx] = 2.0 * g;
      grad[2 * stride + idx] = 3.0 * g;
    }
  }

  // System offsets
  std::vector<std::int64_t> sys_offsets(kNsys + 1);
  for (int s = 0; s <= kNsys; ++s) {
    sys_offsets[s] = static_cast<std::int64_t>(s) * kPointsPerSys;
  }

  cudaMemcpyAsync(arena.rho(), rho.data(), total_np * sizeof(double),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(arena.weights(), weights.data(), total_np * sizeof(double),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(arena.grad(), grad.data(), stride * 3 * sizeof(double),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(arena.sys_offsets(), sys_offsets.data(),
                  (kNsys + 1) * sizeof(std::int64_t),
                  cudaMemcpyHostToDevice, stream);

  XcSpec spec;
  spec.family = Family::kGga;
  spec.nspin = 1;
  spec.terms = {{Functional::kPbe, 1.0}};
  spec.precision = PrecisionPolicy::kFloat64;
  spec.deterministic = false;

  XcGridIn input{arena.rho(), arena.grad(), nullptr, arena.weights(),
                 total_np, stride, kNsys, arena.sys_offsets()};
  XcGridOut output{arena.wv_rho(), arena.wv_grad(), nullptr,
                   arena.exc_per_system()};

  // --- Test 1: Multi-system correctness ---
  // Run via XcEval and check per-system energies sum to total.
  auto status = XcEval(spec, input, output, stream);
  if (!status.ok()) return Fail(status.message().c_str());

  std::vector<double> exc_host(kNsys);
  cudaMemcpyAsync(exc_host.data(), arena.exc_per_system(),
                  kNsys * sizeof(double), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  double total_energy = 0.0;
  double min_e = exc_host[0], max_e = exc_host[0];
  for (int s = 0; s < kNsys; ++s) {
    total_energy += exc_host[s];
    min_e = std::min(min_e, exc_host[s]);
    max_e = std::max(max_e, exc_host[s]);
  }
  std::printf("Multi-system correctness:\n");
  std::printf("  Total E_xc = %.10f Ha\n", total_energy);
  std::printf("  Per-system range: [%.10f, %.10f] Ha\n", min_e, max_e);
  std::printf("  All systems non-zero: %s\n",
              (min_e != 0.0 && max_e != 0.0) ? "YES" : "NO");

  // Verify per-system energies are approximately equal (same density pattern)
  // Use a loose tolerance since fast (non-deterministic) reduction accumulates
  // in different order per system.
  bool per_sys_ok = true;
  for (int s = 1; s < kNsys; ++s) {
    if (std::abs(exc_host[s] - exc_host[0]) > 1e-6) {
      per_sys_ok = false;
      break;
    }
  }
  std::printf("  Per-system energies match (same density pattern): %s\n\n",
              per_sys_ok ? "YES" : "NO");

  // --- Test 2: Per-system loop vs batched CUDA graph ---
  // The plan's 5x target: compare launching 1000 individual XC evaluations
  // (one per system) vs one batched evaluation captured in a CUDA graph.

  // First, correctness check with deterministic mode.
  spec.deterministic = true;
  auto det_status = XcEval(spec, input, output, stream);
  if (!det_status.ok()) return Fail(det_status.message().c_str());
  std::vector<double> exc_det(kNsys);
  cudaMemcpyAsync(exc_det.data(), arena.exc_per_system(),
                  kNsys * sizeof(double), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // Switch to non-deterministic for timing (fast kernel path).
  spec.deterministic = false;

  // --- Path A: Per-system loop (1000 individual launches) ---
  // Each system is evaluated separately — this is the "without batching" path.
  constexpr int kWarmup = 3;
  constexpr int kIterSweep = 10;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup
  for (int iter = 0; iter < kWarmup; ++iter) {
    for (int s = 0; s < kNsys; ++s) {
      XcGridIn single_in{arena.rho() + s * kPointsPerSys,
                         arena.grad() + s * kPointsPerSys,
                         nullptr,
                         arena.weights() + s * kPointsPerSys,
                         kPointsPerSys, stride, 1, nullptr};
      XcGridOut single_out{arena.wv_rho() + s * kPointsPerSys,
                           arena.wv_grad() + s * kPointsPerSys,
                           nullptr,
                           arena.exc_per_system() + s};
      LaunchXcFunctional(spec, single_in, single_out, stream);
    }
  }
  cudaStreamSynchronize(stream);

  cudaEventRecord(start, stream);
  for (int iter = 0; iter < kIterSweep; ++iter) {
    for (int s = 0; s < kNsys; ++s) {
      XcGridIn single_in{arena.rho() + s * kPointsPerSys,
                         arena.grad() + s * kPointsPerSys,
                         nullptr,
                         arena.weights() + s * kPointsPerSys,
                         kPointsPerSys, stride, 1, nullptr};
      XcGridOut single_out{arena.wv_rho() + s * kPointsPerSys,
                           arena.wv_grad() + s * kPointsPerSys,
                           nullptr,
                           arena.exc_per_system() + s};
      LaunchXcFunctional(spec, single_in, single_out, stream);
    }
  }
  cudaEventRecord(stop, stream);
  cudaStreamSynchronize(stream);

  float loop_ms = 0.0f;
  cudaEventElapsedTime(&loop_ms, start, stop);
  const double loop_per_sweep = static_cast<double>(loop_ms) / kIterSweep;

  // --- Path B: Batched graph replay (1 graph launch per sweep) ---
  // Capture one batched XcEval call into a CUDA graph.
  cudaGraph_t graph;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  LaunchXcFunctional(spec, input, output, stream);
  cudaStreamEndCapture(stream, &graph);

  cudaGraphExec_t graph_exec;
  cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
  cudaGraphDestroy(graph);

  // Warmup
  for (int iter = 0; iter < kWarmup; ++iter) {
    cudaGraphLaunch(graph_exec, stream);
  }
  cudaStreamSynchronize(stream);

  cudaEventRecord(start, stream);
  for (int iter = 0; iter < kIterSweep; ++iter) {
    cudaGraphLaunch(graph_exec, stream);
  }
  cudaEventRecord(stop, stream);
  cudaStreamSynchronize(stream);

  float graph_ms = 0.0f;
  cudaEventElapsedTime(&graph_ms, start, stop);
  const double graph_per_sweep = static_cast<double>(graph_ms) / kIterSweep;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  const double speedup = loop_per_sweep / graph_per_sweep;
  std::printf("Per-system loop vs batched graph:\n");
  std::printf("  Per-system loop:  %.4f ms/sweep (%d sys × %d iter)\n",
              loop_per_sweep, kNsys, kIterSweep);
  std::printf("  Batched graph:    %.4f ms/sweep (%d iter)\n",
              graph_per_sweep, kIterSweep);
  std::printf("  Speedup:          %.2fx\n", speedup);
  std::printf("  Target:           >= 5x launch-overhead reduction\n\n");

  // Verify graph replay produces same total energy (deterministic baseline)
  std::vector<double> exc_graph(kNsys);
  spec.deterministic = true;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  LaunchXcFunctional(spec, input, output, stream);
  cudaStreamEndCapture(stream, &graph);
  cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
  cudaGraphDestroy(graph);
  cudaGraphLaunch(graph_exec, stream);
  cudaMemcpyAsync(exc_graph.data(), arena.exc_per_system(),
                  kNsys * sizeof(double), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  bool graph_correct = true;
  for (int s = 0; s < kNsys; ++s) {
    if (std::abs(exc_graph[s] - exc_det[s]) > 1e-12) {
      graph_correct = false;
      break;
    }
  }
  std::printf("  Graph result matches deterministic baseline: %s\n",
              graph_correct ? "YES" : "NO");

  cudaGraphExecDestroy(graph_exec);
  arena.Release(stream);
  cudaStreamDestroy(stream);

  const bool pass = per_sys_ok && graph_correct && (speedup >= 5.0);
  std::printf("\n=== Result: %s ===\n", pass ? "PASS" : "FAIL");
  std::printf("  Multi-system correctness: %s\n", per_sys_ok ? "OK" : "FAIL");
  std::printf("  Graph correctness:        %s\n", graph_correct ? "OK" : "FAIL");
  std::printf("  Speedup >= 5x:            %s (%.2fx)\n",
              speedup >= 5.0 ? "OK" : "FAIL", speedup);
  return pass ? 0 : 1;
}
