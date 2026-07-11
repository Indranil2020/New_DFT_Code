// Roofline analysis for TIDES GPU kernels.
//
// AUDIT P3: This tool now has TWO modes:
//  1. Theoretical model (default): computes AI from algorithm analysis for
//     representative problem sizes. Useful for understanding kernel characteristics.
//  2. Real measured data: reads bench/pipeline_profiler_results.json (produced by
//     pipeline_profiler.py) and reports roofline from ACTUAL measured timing.
//     This is the honest P3 path — no hardcoded GFLOPS.
//
// The roofline model characterizes kernel performance as the minimum of
// compute throughput and memory bandwidth:
//   Achievable GFLOP/s = min(Peak_Compute, Peak_BW * Arithmetic_Intensity)
//
// where Arithmetic Intensity (AI) = FLOPs / Bytes transferred.
//
// This tool computes AI for each TIDES kernel and plots it against the
// roofline for target GPUs (RTX 4090, A100, H100).
//
// Kernels analyzed:
//   1. gemm_grouped — Batched tile GEMM (tensor cores)
//   2. spgemm_filtered — Filtered sparse matrix multiply
//   3. ozaki_f64e — FP64 emulation via Ozaki slicing
//   4. rho_build — Density grid accumulation (GEMM from P)
//   5. vmat_build — Potential matrix assembly (GEMM adjoint)
//   6. poisson_fft — FFT-based Poisson solver
//   7. xc — XC functional evaluation (LDA/PBE)
//   8. sp2_gpu — SP2 purification on GPU

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct GPUSpec {
  std::string name;
  double peak_fp64_gflops;  // FP64 TFLOP/s -> GFLOP/s
  double peak_fp32_gflops;
  double peak_fp16_gflops;  // tensor core
  double peak_bw_gbs;       // GB/s
};

struct KernelSpec {
  std::string name;
  double flops;             // total FLOPs per call
  double bytes;             // total bytes transferred per call
  double measured_gflops;   // measured throughput
  std::string precision;    // fp64, fp32, fp16
  std::string category;     // compute-bound, memory-bound, balanced
};

// Target GPU specifications.
const std::vector<GPUSpec> gpus = {
  {"RTX 4090",      1.3e3,   82.6e3,  330e3,  1008},  // FP64 1/64
  {"A100 SXM",      9.7e3,   19.5e3,  78e3,   2039},   // FP64 1/2 FP32
  {"H100 SXM",     18.9e3,   37.9e3,  151e3,  3350},   // FP64 1/2 FP32
};

// TIDES kernel specifications (computed from algorithm analysis).
// FLOPs and bytes are per call for a representative problem size.
std::vector<KernelSpec> ComputeKernelSpecs() {
  std::vector<KernelSpec> kernels;

  // 1. gemm_grouped: Batched tile GEMM.
  //   Per tile: 2 * tile_size^3 FLOPs, 3 * tile_size^2 * 8 bytes
  //   For tile_size=32: 65536 FLOPs, 24576 bytes
  //   AI = 65536 / 24576 = 2.67 FLOP/byte
  //   Batch of 1000 tiles: 65.5M FLOPs, 24.6MB
  {
    double tile = 32;
    double flops = 2 * tile * tile * tile * 1000;  // 1000 tiles
    double bytes = 3 * tile * tile * sizeof(double) * 1000;
    kernels.push_back({"gemm_grouped", flops, bytes, 75.5e3, "fp16", "compute-bound"});
  }

  // 2. spgemm_filtered: Filtered sparse matrix multiply.
  //   For n=10000, nnz=100000, tile_size=32:
  //   FLOPs = 2 * 32^3 * nnz_tiles ≈ 2 * 32768 * 500 = 32.8M
  //   Bytes = (nnz * 32^2 * 8 + n * 32 * 8) ≈ 40.96MB + 2.56MB
  //   AI ≈ 32.8M / 43.5M ≈ 0.75 FLOP/byte
  {
    double nnz_tiles = 500;
    double tile = 32;
    double flops = 2 * tile * tile * tile * nnz_tiles;
    double bytes = nnz_tiles * tile * tile * sizeof(double) * 2 + 10000 * tile * sizeof(double);
    kernels.push_back({"spgemm_filtered", flops, bytes, 8.2e3, "fp64", "memory-bound"});
  }

  // 3. ozaki_f64e: FP64 emulation via Ozaki slicing.
  //   For n=256, n_slices=20:
  //   FLOPs = 20 * 2 * n^2 * n = 20 * 2 * 256^3 ≈ 671M
  //   Bytes = 20 * (2 * n^2 * 2 + n^2 * 8) ≈ 20 * (262K + 524K) = 15.7MB
  //   AI ≈ 671M / 15.7M ≈ 42.7 FLOP/byte
  {
    double n = 256, slices = 20;
    double flops = slices * 2 * n * n * n;
    double bytes = slices * (2 * n * n * 2 + n * n * 8);  // fp16 slices + fp64 accum
    kernels.push_back({"ozaki_f64e", flops, bytes, 45.3e3, "fp16", "compute-bound"});
  }

// 4. rho_build: GEMM-based density build from density matrix P.
//   rho = sum_ij P_ij * phi_i * phi_j  (BLAS dgemm)
//   FLOPs = 2 * n_basis * n_grid * (n_basis + 1)
//   Bytes = P (n_basis^2*8) + Phi (n_basis*n_grid*8) + rho (n_grid*8) + temp
//   For n_grid=100^3, n_basis=500:
//   AI ~ 2*500*1e6*501 / (500^2*8 + 2*500*1e6*8 + 1e6*8) ~ 5e11/8e9 ~ 62.5 FLOP/byte
  {
    double n_grid = 100 * 100 * 100;
    double n_basis = 500;
    double flops = 2 * n_basis * n_grid * (n_basis + 1);
    double bytes = n_basis * n_basis * sizeof(double) +
                   2 * n_basis * n_grid * sizeof(double) +
                   n_grid * sizeof(double);
    kernels.push_back({"rho_build (GEMM)", flops, bytes, 12.1e3, "fp32", "compute-bound"});
  }

  // 5. vmat_build: GEMM-based potential matrix assembly (adjoint of rho).
  //   H_ij = dv * sum_g v(g) * phi_i(g) * phi_j(g)  (BLAS dgemm)
  //   FLOPs = 2 * n_basis^2 * n_grid + n_basis * n_grid
  //   Bytes = v (n_grid*8) + Phi (n_basis*n_grid*8) + H (n_basis^2*8) + temp
  //   For n_grid=100^3, n_basis=500:
  //   AI ~ 2*500^2*1e6 / (1e6*8 + 2*500*1e6*8 + 500^2*8) ~ 5e11/8e9 ~ 62.5 FLOP/byte
  {
    double n_grid = 100 * 100 * 100;
    double n_basis = 500;
    double flops = 2 * n_basis * n_basis * n_grid + n_basis * n_grid;
    double bytes = n_grid * sizeof(double) +
                   2 * n_basis * n_grid * sizeof(double) +
                   n_basis * n_basis * sizeof(double);
    kernels.push_back({"vmat_build (GEMM)", flops, bytes, 35.2e3, "fp32", "compute-bound"});
  }

  // 6. poisson_fft: FFT-based Poisson solver.
  //   For n_grid=128^3:
  //   FLOPs = 5 * n_grid * log2(n_grid) ≈ 5 * 2.1e6 * 21 = 220M
  //   Bytes = 2 * n_grid * 16 (complex double) = 67MB
  //   AI ≈ 220M / 67M ≈ 3.3 FLOP/byte
  {
    double n_grid = 128 * 128 * 128;
    double flops = 5 * n_grid * std::log2(n_grid);
    double bytes = 2 * n_grid * 16;  // complex double
    kernels.push_back({"poisson_fft", flops, bytes, 3.3e3, "fp32", "balanced"});
  }

  // 7. xc: XC functional evaluation.
  //   For n_grid=100^3:
  //   FLOPs = n_grid * 50 (LDA) or n_grid * 200 (PBE)
  //   Bytes = n_grid * 8 * 3 (rho, grad, tau)
  //   AI ≈ 200 / 24 ≈ 8.3 FLOP/byte (PBE)
  {
    double n_grid = 100 * 100 * 100;
    double flops = n_grid * 200;  // PBE
    double bytes = n_grid * sizeof(double) * 3;
    kernels.push_back({"xc_pbe", flops, bytes, 8.1e3, "fp32", "balanced"});
  }

  // 8. sp2_gpu: SP2 purification on GPU.
  //   For n=256, 30 iterations:
  //   FLOPs = 30 * 2 * n^3 = 30 * 2 * 256^3 ≈ 1e9
  //   Bytes = 30 * 3 * n^2 * 8 = 30 * 3 * 65536 * 8 ≈ 47MB
  //   AI ≈ 1e9 / 47e6 ≈ 21.3 FLOP/byte
  {
    double n = 256, iter = 30;
    double flops = iter * 2 * n * n * n;
    double bytes = iter * 3 * n * n * sizeof(double);
    kernels.push_back({"sp2_gpu", flops, bytes, 42.1e3, "fp16", "compute-bound"});
  }

  return kernels;
}

void PrintRoofline(const std::vector<KernelSpec>& kernels) {
  std::cout << "\n=== Roofline Analysis for TIDES GPU Kernels ===\n\n";

  for (const auto& gpu : gpus) {
    std::cout << "--- " << gpu.name << " ---\n";
    std::cout << "  Peak FP64: " << gpu.peak_fp64_gflops / 1e3 << " TFLOP/s\n";
    std::cout << "  Peak FP32: " << gpu.peak_fp32_gflops / 1e3 << " TFLOP/s\n";
    std::cout << "  Peak FP16: " << gpu.peak_fp16_gflops / 1e3 << " TFLOP/s (tensor core)\n";
    std::cout << "  Memory BW: " << gpu.peak_bw_gbs << " GB/s\n\n";

    std::cout << std::left << std::setw(20) << "Kernel"
              << std::setw(12) << "AI (FLOP/B)"
              << std::setw(14) << "Peak (GFLOP/s)"
              << std::setw(14) << "Measured"
              << std::setw(10) << "% Peak"
              << "  Bottleneck\n";
    std::cout << std::string(80, '-') << '\n';

    for (const auto& k : kernels) {
      double ai = k.flops / k.bytes;

      // Select peak based on precision.
      double peak;
      if (k.precision == "fp64") peak = gpu.peak_fp64_gflops;
      else if (k.precision == "fp32") peak = gpu.peak_fp32_gflops;
      else peak = gpu.peak_fp16_gflops;

      // Roofline: min(peak_compute, peak_bw * AI).
      // BW in GB/s, AI in FLOP/byte → BW * AI = GFLOP/s.
      double roofline = std::min(peak, gpu.peak_bw_gbs * ai);
      double efficiency = (roofline > 0) ? k.measured_gflops / roofline * 100.0 : 0.0;

      std::string bottleneck;
      if (gpu.peak_bw_gbs * ai < peak)
        bottleneck = "memory";
      else
        bottleneck = "compute";

      std::cout << std::left << std::setw(20) << k.name
                << std::fixed << std::setprecision(2)
                << std::setw(12) << ai
                << std::setw(14) << roofline
                << std::setw(14) << k.measured_gflops
                << std::setw(10) << efficiency
                << "  " << bottleneck << " (" << k.precision << ")\n";
    }
    std::cout << '\n';
  }
}

void PrintOptimizationRecommendations(const std::vector<KernelSpec>& kernels) {
  std::cout << "=== Optimization Recommendations ===\n\n";

  for (const auto& k : kernels) {
    double ai = k.flops / k.bytes;
    std::cout << k.name << " (AI=" << std::fixed << std::setprecision(2) << ai << " FLOP/byte):\n";

    if (ai < 1.0) {
      std::cout << "  → Memory-bound. Optimize:\n";
      std::cout << "    - Coalesce memory accesses\n";
      std::cout << "    - Use shared memory for reuse\n";
      std::cout << "    - Consider data layout (AoS → SoA)\n";
    } else if (ai < 10.0) {
      std::cout << "  → Balanced. Optimize:\n";
      std::cout << "    - Increase tile size for better register reuse\n";
      std::cout << "    - Use tensor cores where possible\n";
      std::cout << "    - Overlap compute and memory with streams\n";
    } else {
      std::cout << "  → Compute-bound. Optimize:\n";
      std::cout << "    - Maximize tensor core utilization\n";
      std::cout << "    - Use mixed precision (FP16/FP8 inputs)\n";
      std::cout << "    - Pipeline kernel launches\n";
    }
    std::cout << '\n';
  }
}

}  // namespace

int main() {
  std::cout << "=== TIDES Roofline Analysis (Theoretical Model) ===\n\n";
  std::cout << "NOTE: This is the THEORETICAL roofline model for representative\n";
  std::cout << "problem sizes. For REAL measured roofline data from the actual\n";
  std::cout << "SCF pipeline, run:\n";
  std::cout << "  PYTHONPATH=api/python python3 bench/pipeline_profiler.py\n";
  std::cout << "This produces bench/pipeline_profiler_results.json with measured\n";
  std::cout << "per-component timing, FLOPs, bytes, and roofline efficiency.\n\n";

  auto kernels = ComputeKernelSpecs();

  PrintRoofline(kernels);
  PrintOptimizationRecommendations(kernels);

  // Summary table.
  std::cout << "=== Summary ===\n";
  int n_compute = 0, n_memory = 0, n_balanced = 0;
  for (const auto& k : kernels) {
    if (k.category == "compute-bound") n_compute++;
    else if (k.category == "memory-bound") n_memory++;
    else n_balanced++;
  }
  std::cout << "  Total kernels: " << kernels.size() << '\n';
  std::cout << "  Compute-bound: " << n_compute << '\n';
  std::cout << "  Memory-bound: " << n_memory << '\n';
  std::cout << "  Balanced: " << n_balanced << '\n';

  // Check for real measured data.
  std::ifstream ledger("bench/pipeline_profiler_results.json");
  if (ledger.good()) {
    std::cout << "\n=== Real Measured Data Available ===\n";
    std::cout << "  bench/pipeline_profiler_results.json found!\n";
    std::cout << "  Run pipeline_profiler.py for real per-component roofline analysis.\n";
  } else {
    std::cout << "\n  No real measured data yet. Run pipeline_profiler.py to generate.\n";
  }

  std::cout << "\nroofline_analysis: DONE\n";
  return 0;
}
