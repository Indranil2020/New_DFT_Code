// T-X1.5: Roofline benchmark for LDA and PBE XC kernels.
//
// Measures Gpt/s (grid points per second) for the fast (non-deterministic)
// kernel path and compares against the MEASURED HBM roofline.
//
// Memory traffic per grid point (FP64):
//   LDA:  read rho(8B) + read weight(8B) + write wv_rho(8B) = 24B
//   GGA:  read rho(8B) + read grad(3*8B) + read weight(8B)
//         + write wv_rho(8B) + write wv_grad(3*8B) = 48B
//
// The roofline is: Gpt/s_peak = min(measured_HBM_BW / bytes_per_point,
//                                    peak_FLOPS / flops_per_point)
//
// LDA-PW92: ~30 FLOPs/point (cbrt + polynomial), 24B/point → AI ~1.3 FLOP/B
// PBE:      ~80 FLOPs/point (cbrt + enhancement factor), 48B/point → AI ~1.7 FLOP/B
// On consumer GPUs these are compute-bound, not memory-bound.
//
// Usage: tides_xc_bench_roofline
// Exit 0 if >=60% of measured roofline, 1 otherwise.

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

struct BenchResult {
  double kernel_ms;
  double gpt_s;
  double mem_roofline_gpt_s;
  double compute_roofline_gpt_s;
  double roofline_gpt_s;
  double fraction;
  bool compute_bound;
};

// Measure actual HBM bandwidth using cudaMemcpy (D2D copy).
double MeasureHbmBandwidthGBps(std::int64_t n_elements, int n_warmup,
                               int n_iter) {
  double* d_src = nullptr;
  double* d_dst = nullptr;
  const std::size_t bytes = static_cast<std::size_t>(n_elements) * sizeof(double);
  cudaMalloc(&d_src, bytes);
  cudaMalloc(&d_dst, bytes);
  cudaMemset(d_src, 1, bytes);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  for (int i = 0; i < n_warmup; ++i) {
    cudaMemcpyAsync(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice, stream);
  }
  cudaStreamSynchronize(stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, stream);
  for (int i = 0; i < n_iter; ++i) {
    cudaMemcpyAsync(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice, stream);
  }
  cudaEventRecord(stop, stream);
  cudaStreamSynchronize(stream);

  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // D2D copy reads src + writes dst = 2 * bytes per iteration
  const double total_bytes = 2.0 * static_cast<double>(bytes) * n_iter;
  const double seconds = static_cast<double>(elapsed_ms) / 1e3;
  const double gbps = total_bytes / seconds / 1e9;

  cudaFree(d_src);
  cudaFree(d_dst);
  cudaStreamDestroy(stream);

  return gbps;
}

BenchResult RunBench(Functional func, Family family, std::int64_t np,
                     std::int64_t stride, int n_warmup, int n_iter,
                     double measured_hbm_gbps) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  XcArena arena;
  arena.Reserve(static_cast<std::size_t>(np), 1, family != Family::kLda,
                false, 1, stream);

  std::vector<double> rho(np, 0.0);
  std::vector<double> weights(np, 0.0);
  std::vector<double> grad(stride * 3, 0.0);
  for (std::int64_t i = 0; i < np; ++i) {
    rho[i] = 0.1 + 0.01 * static_cast<double>(i % 1000);
    weights[i] = 1.0 / static_cast<double>(np);
  }
  for (std::int64_t i = 0; i < np; ++i) {
    const double g = 0.05 * static_cast<double>(i % 50);
    grad[i] = g;
    grad[stride + i] = 2.0 * g;
    grad[2 * stride + i] = 3.0 * g;
  }

  cudaMemcpyAsync(arena.rho(), rho.data(), np * sizeof(double),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(arena.weights(), weights.data(), np * sizeof(double),
                  cudaMemcpyHostToDevice, stream);
  if (family != Family::kLda) {
    cudaMemcpyAsync(arena.grad(), grad.data(), stride * 3 * sizeof(double),
                    cudaMemcpyHostToDevice, stream);
  }

  XcSpec spec;
  spec.family = family;
  spec.nspin = 1;
  spec.terms = {{func, 1.0}};
  spec.precision = PrecisionPolicy::kFloat64;
  spec.deterministic = false;

  XcGridIn input{arena.rho(),
                 family != Family::kLda ? arena.grad() : nullptr,
                 nullptr, arena.weights(), np, stride, 1, arena.sys_offsets()};
  XcGridOut output{arena.wv_rho(),
                   family != Family::kLda ? arena.wv_grad() : nullptr,
                   nullptr, arena.exc_per_system()};

  // Measure kernel directly via LaunchXcFunctional (skips XcEval's per-call
  // cudaMemsetAsync and validation overhead).
  for (int i = 0; i < n_warmup; ++i) {
    LaunchXcFunctional(spec, input, output, stream);
  }
  cudaStreamSynchronize(stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, stream);
  for (int i = 0; i < n_iter; ++i) {
    LaunchXcFunctional(spec, input, output, stream);
  }
  cudaEventRecord(stop, stream);
  cudaStreamSynchronize(stream);

  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  const double kernel_ms = static_cast<double>(elapsed_ms) / n_iter;
  const double gpt_s = static_cast<double>(np) / (kernel_ms * 1e-3) / 1e9;

  const double bytes_per_point = (family == Family::kLda) ? 24.0 : 48.0;
  const double flops_per_point = (family == Family::kLda) ? 30.0 : 80.0;
  const double mem_roofline_gpt_s = measured_hbm_gbps / bytes_per_point;

  // Compute roofline: peak FP64 FLOPS / flops_per_point
  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  // FP64 throughput: 1/64 of FP32 on consumer GPUs (GeForce), 1/2 on datacenter
  const double fp64_ratio = (prop.major >= 7 && prop.minor >= 0) ? 1.0/64.0 : 0.5;
  const double peak_fp64_gflops = static_cast<double>(prop.multiProcessorCount) *
      static_cast<double>(prop.clockRate) * 1e3 * fp64_ratio * 32.0 / 1e9;
  const double compute_roofline_gpt_s = peak_fp64_gflops / flops_per_point;

  const double roofline_gpt_s = std::min(mem_roofline_gpt_s, compute_roofline_gpt_s);
  const bool compute_bound = compute_roofline_gpt_s < mem_roofline_gpt_s;
  const double fraction = gpt_s / roofline_gpt_s;

  arena.Release(stream);
  cudaStreamDestroy(stream);

  return {kernel_ms, gpt_s, mem_roofline_gpt_s, compute_roofline_gpt_s,
          roofline_gpt_s, fraction, compute_bound};
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::printf("SKIP: CUDA not available\n");
    return 77;
  }

  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  std::printf("=== T-X1.5: XC Kernel Roofline Benchmark ===\n");
  std::printf("GPU: %s\n", prop.name);
  std::printf("SM count: %d, Clock: %.0f MHz\n\n", prop.multiProcessorCount,
              prop.clockRate / 1e3);

  // Measure actual HBM bandwidth with a copy kernel (8M doubles = 64MB)
  constexpr int kBwWarmup = 5;
  constexpr int kBwIter = 50;
  constexpr std::int64_t kBwElements = 8 * 1024 * 1024;
  const double measured_hbm_gbps =
      MeasureHbmBandwidthGBps(kBwElements, kBwWarmup, kBwIter);
  std::printf("Measured HBM BW: %.1f GB/s (copy kernel, %ldM doubles)\n\n",
              measured_hbm_gbps, static_cast<long>(kBwElements / 1024 / 1024));

  constexpr int kWarmup = 5;
  constexpr int kIter = 50;

  struct BenchConfig {
    const char* name;
    Functional func;
    Family family;
    std::int64_t np;
  };

  const BenchConfig configs[] = {
      {"LDA-PW92   1M", Functional::kLdaPw92, Family::kLda, 1000000},
      {"LDA-PW92  10M", Functional::kLdaPw92, Family::kLda, 10000000},
      {"PBE         1M", Functional::kPbe,     Family::kGga, 1000000},
      {"PBE        10M", Functional::kPbe,     Family::kGga, 10000000},
  };

  const double target_fraction = 0.60;
  bool all_pass = true;

  std::printf("%-16s  %10s  %10s  %10s  %10s  %10s  %8s  %s\n",
              "Kernel", "np", "kernel_ms", "Gpt/s", "MemRoof", "CmpRoof", "Roof%", "Status");
  std::printf("%-16s  %10s  %10s  %10s  %10s  %10s  %8s  %s\n",
              "------", "--", "---------", "-----", "-------", "-------", "-----", "------");

  for (const auto& cfg : configs) {
    const std::int64_t stride = ((cfg.np + 511) / 512) * 512;
    auto r = RunBench(cfg.func, cfg.family, cfg.np, stride, kWarmup, kIter,
                      measured_hbm_gbps);
    const double pct = r.fraction * 100.0;
    const bool pass = r.fraction >= target_fraction;
    if (!pass) all_pass = false;
    std::printf("%-16s  %10ld  %10.3f  %10.3f  %10.3f  %10.3f  %7.1f%%  %s%s\n",
                cfg.name, static_cast<long>(cfg.np), r.kernel_ms,
                r.gpt_s, r.mem_roofline_gpt_s, r.compute_roofline_gpt_s,
                pct, pass ? "PASS" : "FAIL",
                r.compute_bound ? " (cmp)" : " (mem)");
  }

  std::printf("\nTarget: >= %.0f%% of HBM roofline\n", target_fraction * 100);
  std::printf("Result: %s\n\n", all_pass ? "ALL PASS" : "SOME FAIL");

  return all_pass ? 0 : 1;
}
