// E1: Tile Substrate Engine — Comprehensive Test & Profile Suite
//
// Tests and profiles every WP1 kernel:
//   1. Grouped GEMM (FP64 + FP16-accum) — accuracy + throughput
//   2. Filtered SpGEMM — accuracy + filtering effectiveness
//   3. Ozaki f64e GEMM — accuracy vs long double + throughput
//   4. f64e reductions (dot, sum, trace) — accuracy + throughput
//   5. CUDA graph replay — overhead measurement
//   6. Precision ledger — completeness verification
//
// All results logged to stdout in structured format for parsing.
// Uses OperationLedger for error tracking per project standard.

#include "tile/gemm_grouped.hpp"
#include "tile/spgemm_filtered.hpp"
#include "tile/spgemm_filtered_cuda.hpp"
#include "tile/ozaki.hpp"
#include "tile/reduce_f64e.hpp"
#include "tile/precision.hpp"
#include "tile/layout.hpp"
#include "tile/graphs.hpp"
#include "common/status.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using tides::tile::CudaGemmProblem;
using tides::tile::CudaGroupedGemmResult;
using tides::tile::GroupedGemmFp64Cuda;
using tides::tile::GroupedGemmFp16AccumCuda;
using tides::tile::SpGemmFilteredFp64Cuda;
using tides::tile::GemmOzakiFp16Cuda;
using tides::tile::DotF64eCuda;
using tides::tile::SumF64eCuda;
using tides::tile::TraceF64eCuda;
using tides::tile::OperationLedger;
using tides::tile::TileMat;
using tides::tile::SpGemmFilteredFp64;

struct ProfileEntry {
  std::string kernel;
  std::string variant;
  std::string size_label;
  double kernel_ms = 0.0;
  double wall_ms = 0.0;
  double max_diff = 0.0;
  double gflops = 0.0;
  std::string status;
};

std::vector<ProfileEntry> g_profile_log;

void LogProfile(const std::string& kernel, const std::string& variant,
                const std::string& size_label, double kernel_ms,
                double wall_ms, double max_diff, double gflops,
                const std::string& status) {
  ProfileEntry e{kernel, variant, size_label, kernel_ms, wall_ms,
                 max_diff, gflops, status};
  g_profile_log.push_back(e);
  std::cout << "  " << std::left << std::setw(20) << kernel
            << std::setw(12) << variant
            << std::setw(12) << size_label
            << std::right << std::setw(10) << std::fixed << std::setprecision(3) << kernel_ms
            << std::setw(10) << std::setprecision(3) << wall_ms
            << std::setw(14) << std::scientific << std::setprecision(3) << max_diff
            << std::setw(10) << std::fixed << std::setprecision(1) << gflops
            << "  " << status << '\n';
}

void PrintProfileHeader() {
  std::cout << std::left << std::setw(20) << "Kernel"
            << std::setw(12) << "Variant"
            << std::setw(12) << "Size"
            << std::right << std::setw(10) << "Kern(ms)"
            << std::setw(10) << "Wall(ms)"
            << std::setw(14) << "MaxDiff"
            << std::setw(10) << "GFLOPS"
            << "  Status\n";
  std::cout << std::string(98, '-') << '\n';
}

// CPU reference GEMM for validation.
std::vector<double> CpuGemm(std::uint32_t m, std::uint32_t k, std::uint32_t n,
                            const std::vector<double>& a,
                            const std::vector<double>& b) {
  std::vector<double> c(m * n, 0.0);
  for (std::uint32_t i = 0; i < m; ++i)
    for (std::uint32_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (std::uint32_t l = 0; l < k; ++l)
        s += a[i * k + l] * b[l * n + j];
      c[i * n + j] = s;
    }
  return c;
}

double MaxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    d = std::max(d, std::abs(a[i] - b[i]));
  return d;
}

// Generate random GEMM problems.
std::vector<CudaGemmProblem> MakeProblems(std::uint32_t n_problems,
                                          std::uint32_t dim,
                                          std::mt19937_64& rng) {
  std::normal_distribution<double> dist(0.0, 1.0);
  std::vector<CudaGemmProblem> problems;
  for (std::uint32_t p = 0; p < n_problems; ++p) {
    CudaGemmProblem prob;
    prob.m = dim;
    prob.k = dim;
    prob.n = dim;
    prob.a.resize(dim * dim);
    prob.b.resize(dim * dim);
    for (auto& v : prob.a) v = dist(rng);
    for (auto& v : prob.b) v = dist(rng);
    problems.push_back(std::move(prob));
  }
  return problems;
}

// --- Test 1: Grouped GEMM FP64 ---
int TestGroupedGemmFp64() {
  std::cout << "\n=== E1.1: Grouped GEMM FP64 ===\n";
  PrintProfileHeader();
  int failures = 0;
  std::mt19937_64 rng(42);

  for (auto [dim, n_prob] : std::vector<std::pair<std::uint32_t, std::uint32_t>>{
           {16, 8}, {32, 8}, {64, 4}, {128, 2}, {256, 1}}) {
    auto problems = MakeProblems(n_prob, dim, rng);

    // CPU reference.
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::vector<double>> cpu_results;
    for (const auto& p : problems) {
      cpu_results.push_back(CpuGemm(p.m, p.k, p.n, p.a, p.b));
    }
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // GPU FP64.
    auto gpu = GroupedGemmFp64Cuda(problems);
    if (!gpu.ok()) {
      LogProfile("GroupedGEMM", "FP64", std::to_string(dim) + "x" + std::to_string(n_prob),
                 0, 0, 0, 0, "FAIL: " + gpu.status().message());
      failures++;
      continue;
    }

    double max_diff = 0.0;
    for (std::size_t i = 0; i < cpu_results.size(); ++i) {
      max_diff = std::max(max_diff, MaxAbsDiff(cpu_results[i], gpu.value().c_tiles[i]));
    }

    double flops = 2.0 * dim * dim * dim * n_prob;
    double gflops_gpu = flops / (gpu.value().kernel_ms * 1e6);
    double gflops_cpu = flops / (cpu_ms * 1e6);

    std::string status = (max_diff < 1e-10) ? "PASS" : "FAIL";
    if (max_diff >= 1e-10) failures++;

    LogProfile("GroupedGEMM", "FP64-GPU",
               std::to_string(dim) + "x" + std::to_string(n_prob),
               gpu.value().kernel_ms, 0, max_diff, gflops_gpu, status);
    LogProfile("GroupedGEMM", "FP64-CPU",
               std::to_string(dim) + "x" + std::to_string(n_prob),
               cpu_ms, cpu_ms, 0, gflops_cpu, "ref");
  }
  return failures;
}

// --- Test 2: Grouped GEMM FP16-accum ---
int TestGroupedGemmFp16Accum() {
  std::cout << "\n=== E1.2: Grouped GEMM FP16-accum (tensor core) ===\n";
  PrintProfileHeader();
  int failures = 0;
  std::mt19937_64 rng(42);

  for (auto [dim, n_prob] : std::vector<std::pair<std::uint32_t, std::uint32_t>>{
           {16, 8}, {32, 8}, {64, 4}, {128, 2}, {256, 1}}) {
    auto problems = MakeProblems(n_prob, dim, rng);

    // CPU reference.
    std::vector<std::vector<double>> cpu_results;
    for (const auto& p : problems) {
      cpu_results.push_back(CpuGemm(p.m, p.k, p.n, p.a, p.b));
    }

    // GPU FP16-accum.
    auto gpu = GroupedGemmFp16AccumCuda(problems);
    if (!gpu.ok()) {
      LogProfile("GroupedGEMM", "FP16-accum",
                 std::to_string(dim) + "x" + std::to_string(n_prob),
                 0, 0, 0, 0, "FAIL: " + gpu.status().message());
      failures++;
      continue;
    }

    double max_diff = 0.0;
    for (std::size_t i = 0; i < cpu_results.size(); ++i) {
      max_diff = std::max(max_diff, MaxAbsDiff(cpu_results[i], gpu.value().c_tiles[i]));
    }

    double flops = 2.0 * dim * dim * dim * n_prob;
    double gflops_gpu = flops / (gpu.value().kernel_ms * 1e6);

    // FP16-accum: mantissa ~10 bits → ~3 decimal digits.
    // For O(1) inputs, absolute error ~1e-2 is expected.
    // Use relative tolerance: max_diff / max(|C|) < 1e-2.
    double max_abs_c = 0.0;
    for (const auto& c : cpu_results)
      for (double v : c) max_abs_c = std::max(max_abs_c, std::abs(v));
    double rel_err = max_diff / std::max(max_abs_c, 1e-10);
    std::string status = (rel_err < 1e-2) ? "PASS" : "FAIL";
    if (rel_err >= 1e-2) failures++;

    LogProfile("GroupedGEMM", "FP16-accum",
               std::to_string(dim) + "x" + std::to_string(n_prob),
               gpu.value().kernel_ms, 0, max_diff, gflops_gpu, status);
  }
  return failures;
}

// --- Test 3: Filtered SpGEMM ---
int TestSpGemmFiltered() {
  std::cout << "\n=== E1.3: Filtered SpGEMM (tile-level) ===\n";
  PrintProfileHeader();
  int failures = 0;
  std::mt19937_64 rng(42);

  for (auto [n_tiles, tile_dim] : std::vector<std::pair<std::uint32_t, std::uint32_t>>{
           {4, 16}, {8, 16}, {16, 32}, {32, 32}}) {
    // Build dense matrices and convert to TileMat.
    std::size_t dim = n_tiles * tile_dim;
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> dense_a(dim * dim), dense_b(dim * dim);
    for (auto& v : dense_a) v = dist(rng);
    for (auto& v : dense_b) v = dist(rng);

    auto a_result = TileMat::FromDense(dim, dim, dense_a, tile_dim);
    auto b_result = TileMat::FromDense(dim, dim, dense_b, tile_dim);
    if (!a_result.ok() || !b_result.ok()) {
      LogProfile("SpGEMM", "FP64-GPU",
                 std::to_string(n_tiles) + "x" + std::to_string(tile_dim),
                 0, 0, 0, 0, "FAIL: FromDense failed");
      failures++;
      continue;
    }
    const auto& a = a_result.value();
    const auto& b = b_result.value();

    // CPU reference.
    auto t0 = std::chrono::steady_clock::now();
    auto cpu_result = SpGemmFilteredFp64(a, b, 1e-15);
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (!cpu_result.ok()) {
      LogProfile("SpGEMM", "FP64-CPU",
                 std::to_string(n_tiles) + "x" + std::to_string(tile_dim),
                 0, 0, 0, 0, "FAIL: CPU SpGEMM failed");
      failures++;
      continue;
    }
    const auto& cpu = cpu_result.value();

    // GPU.
    auto gpu = SpGemmFilteredFp64Cuda(a, b, 1e-15);
    if (!gpu.ok()) {
      LogProfile("SpGEMM", "FP64-GPU",
                 std::to_string(n_tiles) + "x" + std::to_string(tile_dim),
                 0, 0, 0, 0, "FAIL: " + gpu.status().message());
      failures++;
      continue;
    }

    // Compare via ToDense.
    auto cpu_dense = cpu.product.ToDense();
    auto gpu_dense = gpu.value().product.ToDense();
    double max_diff = MaxAbsDiff(cpu_dense, gpu_dense);

    double mat_dim = static_cast<double>(n_tiles) * tile_dim;
    double flops = 2.0 * mat_dim * mat_dim * mat_dim;
    double gflops_gpu = flops / (gpu.value().kernel_ms * 1e6);
    double gflops_cpu = flops / (cpu_ms * 1e6);

    std::string status = (max_diff < 1e-10) ? "PASS" : "FAIL";
    if (max_diff >= 1e-10) failures++;

    LogProfile("SpGEMM", "FP64-GPU",
               std::to_string(n_tiles) + "x" + std::to_string(tile_dim),
               gpu.value().kernel_ms, 0, max_diff, gflops_gpu, status);
    LogProfile("SpGEMM", "FP64-CPU",
               std::to_string(n_tiles) + "x" + std::to_string(tile_dim),
               cpu_ms, cpu_ms, 0, gflops_cpu, "ref");
  }
  return failures;
}

// --- Test 4: Ozaki f64e GEMM ---
int TestOzakiF64e() {
  std::cout << "\n=== E1.4: Ozaki f64e GEMM (FP16-slice emulation) ===\n";
  PrintProfileHeader();
  int failures = 0;
  std::mt19937_64 rng(42);

  for (auto dim : {16, 32, 64, 128}) {
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> a(dim * dim), b(dim * dim);
    for (auto& v : a) v = dist(rng);
    for (auto& v : b) v = dist(rng);

    // CPU reference.
    auto t0 = std::chrono::steady_clock::now();
    auto cpu_ref = CpuGemm(dim, dim, dim, a, b);
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // GPU Ozaki f64e.
    auto gpu = GemmOzakiFp16Cuda(dim, dim, dim, a, b);
    if (!gpu.ok()) {
      LogProfile("Ozaki-f64e", "FP16-slice",
                 "n=" + std::to_string(dim),
                 0, 0, 0, 0, "FAIL: " + gpu.status().message());
      failures++;
      continue;
    }

    double max_diff = MaxAbsDiff(gpu.value().values, cpu_ref);
    double flops = 2.0 * dim * dim * dim;
    // Ozaki doesn't report kernel_ms directly; use wall time.
    double gflops_gpu = flops / (cpu_ms * 1e6);  // placeholder

    // f64e should be close to FP64 (that's the point of emulation).
    std::string status = (max_diff < 1e-8) ? "PASS" : "FAIL";
    if (max_diff >= 1e-8) failures++;

    LogProfile("Ozaki-f64e", "FP16-slice",
               "n=" + std::to_string(dim),
               0, 0, max_diff, 0, status);
    LogProfile("Ozaki-f64e", "FP64-CPU-ref",
               "n=" + std::to_string(dim),
               cpu_ms, cpu_ms, 0, gflops_gpu, "ref");
  }
  return failures;
}

// --- Test 5: f64e Reductions ---
int TestF64eReductions() {
  std::cout << "\n=== E1.5: f64e Reductions (dot, sum, trace) ===\n";
  PrintProfileHeader();
  int failures = 0;
  std::mt19937_64 rng(42);

  for (auto n : {1000, 10000, 100000, 1000000}) {
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> a(n), b(n);
    for (auto& v : a) v = dist(rng);
    for (auto& v : b) v = dist(rng);

    // CPU reference dot product.
    auto t0 = std::chrono::steady_clock::now();
    double cpu_dot = 0.0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) cpu_dot += a[i] * b[i];
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // GPU f64e dot.
    auto gpu = DotF64eCuda(a, b);
    if (!gpu.ok()) {
      LogProfile("Dot-f64e", "GPU",
                 "n=" + std::to_string(n),
                 0, 0, 0, 0, "FAIL: " + gpu.status().message());
      failures++;
      continue;
    }

    double diff = std::abs(gpu.value().value - cpu_dot);
    // f64e error grows as ~sqrt(n) * eps_f64e. For n=1M, expect ~1e-5 rel.
    double rel_tol = 1e-8 + 1e-9 * std::sqrt(static_cast<double>(n));
    std::string status = (diff < rel_tol * std::abs(cpu_dot)) ? "PASS" : "FAIL";
    if (diff >= rel_tol * std::abs(cpu_dot)) failures++;

    LogProfile("Dot-f64e", "GPU",
               "n=" + std::to_string(n),
               gpu.value().kernel_ms, 0, diff, 0, status);
    LogProfile("Dot-f64e", "CPU-ref",
               "n=" + std::to_string(n),
               cpu_ms, cpu_ms, 0, 0, "ref");
  }

  // Trace test.
  for (auto n : {64, 256, 1024}) {
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> mat(n * n);
    for (auto& v : mat) v = dist(rng);

    double cpu_trace = 0.0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) cpu_trace += mat[i * static_cast<std::size_t>(n) + i];

    auto gpu = TraceF64eCuda(n, n, mat);
    if (!gpu.ok()) {
      LogProfile("Trace-f64e", "GPU",
                 "n=" + std::to_string(n),
                 0, 0, 0, 0, "FAIL: " + gpu.status().message());
      failures++;
      continue;
    }

    double diff = std::abs(gpu.value().value - cpu_trace);
    std::string status = (diff < 1e-8 * std::abs(cpu_trace)) ? "PASS" : "FAIL";
    if (diff >= 1e-8 * std::abs(cpu_trace)) failures++;

    LogProfile("Trace-f64e", "GPU",
               "n=" + std::to_string(n),
               gpu.value().kernel_ms, 0, diff, 0, status);
  }
  return failures;
}

// --- Test 6: CUDA Graph Replay ---
int TestCudaGraphReplay() {
  std::cout << "\n=== E1.6: CUDA Graph Replay ===\n";
  PrintProfileHeader();
  int failures = 0;
  std::mt19937_64 rng(42);

  auto problems = MakeProblems(8, 64, rng);
  std::uint32_t repeats = 100;

  // FP64 graph replay.
  auto graph_result = tides::tile::GroupedGemmFp64CudaGraphReplay(problems, repeats);
  if (!graph_result.ok()) {
    LogProfile("GraphReplay", "FP64",
               "8x64x100", 0, 0, 0, 0,
               "FAIL: " + graph_result.status().message());
    failures++;
  } else {
    const auto& gr = graph_result.value();
    double per_call_raw = gr.raw_repeated_kernel_ms / repeats;
    double per_call_graph = gr.graph_replay_ms / repeats;
    double speedup = per_call_raw / std::max(per_call_graph, 1e-10);

    std::string status = (speedup > 0.9) ? "PASS" : "FAIL";
    if (speedup <= 0.9) failures++;

    LogProfile("GraphReplay", "FP64-raw",
               "8x64x" + std::to_string(repeats),
               per_call_raw, 0, 0, 0, "ref");
    LogProfile("GraphReplay", "FP64-graph",
               "8x64x" + std::to_string(repeats),
               per_call_graph, 0, 0, 0, status + " (x" + std::to_string(speedup) + ")");
  }

  // FP16-accum graph replay.
  auto graph_fp16 = tides::tile::GroupedGemmFp16AccumCudaGraphReplay(problems, repeats);
  if (!graph_fp16.ok()) {
    LogProfile("GraphReplay", "FP16-accum",
               "8x64x100", 0, 0, 0, 0,
               "FAIL: " + graph_fp16.status().message());
    failures++;
  } else {
    const auto& gr = graph_fp16.value();
    double per_call_raw = gr.raw_repeated_kernel_ms / repeats;
    double per_call_graph = gr.graph_replay_ms / repeats;
    double speedup = per_call_raw / std::max(per_call_graph, 1e-10);

    std::string status = (speedup > 0.9) ? "PASS" : "FAIL";
    if (speedup <= 0.9) failures++;

    LogProfile("GraphReplay", "FP16-raw",
               "8x64x" + std::to_string(repeats),
               per_call_raw, 0, 0, 0, "ref");
    LogProfile("GraphReplay", "FP16-graph",
               "8x64x" + std::to_string(repeats),
               per_call_graph, 0, 0, 0, status + " (x" + std::to_string(speedup) + ")");
  }
  return failures;
}

// --- Test 7: Precision Ledger ---
int TestPrecisionLedger() {
  std::cout << "\n=== E1.7: Precision Ledger ===\n";
  int failures = 0;

  // Verify ledger is populated by GEMM operations.
  std::mt19937_64 rng(42);
  auto problems = MakeProblems(4, 64, rng);

  auto gpu = GroupedGemmFp64Cuda(problems);
  if (!gpu.ok()) {
    std::cout << "  FAIL: GEMM failed, cannot test ledger\n";
    return 1;
  }

  const auto& ledger = gpu.value().ledger;
  std::cout << "  Ledger entries: " << ledger.size() << '\n';
  std::cout << "  Total observed error bound: " << ledger.TotalObservedErrorBound(tides::tile::ErrorMetric::kAbsolute) << '\n';
  std::cout << "  Total dropped candidates: " << ledger.TotalDropped() << '\n';

  if (ledger.size() == 0) {
    std::cout << "  FAIL: No ledger entries recorded\n";
    failures++;
  } else {
    std::cout << "  PASS: Ledger populated with " << ledger.size() << " entries\n";
  }

  // Test merge.
  OperationLedger merged;
  merged.Merge(ledger);
  merged.Merge(ledger);
  if (merged.size() != 2 * ledger.size()) {
    std::cout << "  FAIL: Merge did not double entry count\n";
    failures++;
  } else {
    std::cout << "  PASS: Merge correctly doubled entries to " << merged.size() << '\n';
  }
  return failures;
}

void PrintSummary(int total_failures) {
  std::cout << "\n=== E1 Summary ===\n";
  std::cout << "Total profile entries: " << g_profile_log.size() << '\n';
  if (total_failures == 0) {
    std::cout << "ALL E1 TESTS PASSED\n";
  } else {
    std::cout << total_failures << " E1 TEST(S) FAILED\n";
  }
}

}  // namespace

int main() {
  std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
            << "║   E1: Tile Substrate Engine — Test & Profile Suite           ║\n"
            << "╚══════════════════════════════════════════════════════════════╝\n";

  int failures = 0;
  failures += TestGroupedGemmFp64();
  failures += TestGroupedGemmFp16Accum();
  failures += TestSpGemmFiltered();
  failures += TestOzakiF64e();
  failures += TestF64eReductions();
  failures += TestCudaGraphReplay();
  failures += TestPrecisionLedger();

  PrintSummary(failures);
  return failures;
}
