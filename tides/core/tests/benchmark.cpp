// P4: Profiling and benchmarking — CPU vs GPU kernel comparison.
//
// Benchmarks all major kernels at multiple problem sizes, reporting
// CPU time, GPU kernel time, and speedup factor.

#include "grid/rho_build.hpp"
#include "grid/rho_build_gpu.hpp"
#include "grid/vmat_build.hpp"
#include "grid/vmat_build_gpu.hpp"
#include "grid/xc.hpp"
#include "grid/xc/tests/xc_test_utils.hpp"
#include "grid/dual_grid.hpp"
#include "solvers/sp2_submatrix/sp2.hpp"
#include "solvers/sp2_submatrix/sp2_gpu.hpp"
#include "solvers/dense/batched_eig.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

using tides::grid::UniformGrid3D;
using tides::grid::RhoBuilder;
using tides::grid::VmatBuilder;
using tides::grid::XCGridEvaluator;
using tides::grid::xc::RunLdaXcOnHostGrid;
using tides::solvers::SP2Purification;
using tides::solvers::BatchedDenseEig;

double MaxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    d = std::max(d, std::abs(a[i] - b[i]));
  return d;
}

void PrintHeader(const std::string& kernel) {
  std::cout << "\n=== " << kernel << " ===\n";
  std::cout << std::left << std::setw(12) << "Size"
            << std::right << std::setw(14) << "CPU (ms)"
            << std::setw(14) << "GPU (ms)"
            << std::setw(12) << "Speedup"
            << std::setw(14) << "Max Diff" << '\n';
  std::cout << std::string(66, '-') << '\n';
}

void PrintRow(const std::string& label, double cpu_ms, double gpu_ms,
              double diff) {
  std::cout << std::left << std::setw(12) << label
            << std::right << std::setw(14) << std::fixed << std::setprecision(3) << cpu_ms
            << std::setw(14) << std::setprecision(3) << gpu_ms
            << std::setw(12) << std::setprecision(2) << (cpu_ms / std::max(gpu_ms, 1e-10))
            << std::setw(14) << std::scientific << std::setprecision(3) << diff << '\n';
}

// --- Rho Build Benchmark ---
void BenchRhoBuild() {
  PrintHeader("Rho Build (density from orbitals)");
  for (int n_per_axis : {16, 24, 32, 48, 64}) {
    UniformGrid3D grid;
    grid.n = {static_cast<std::size_t>(n_per_axis),
              static_cast<std::size_t>(n_per_axis),
              static_cast<std::size_t>(n_per_axis)};
    grid.h = {0.3, 0.3, 0.3};
    grid.origin = {-2.4, -2.4, -2.4};

    auto orb0 = RhoBuilder::GaussianOrbital(grid, 1.0, {0, 0, 0});
    auto orb1 = RhoBuilder::GaussianOrbital(grid, 1.5, {0.3, 0, 0});
    std::vector<std::vector<double>> orbitals = {orb0, orb1};
    std::vector<double> occ = {2.0, 2.0};

    // CPU.
    auto t0 = std::chrono::steady_clock::now();
    auto rho_cpu = RhoBuilder::BuildFromOrbitals(grid, orbitals, occ);
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // GPU.
    auto rho_gpu = tides::grid::RhoBuildCuda(grid, orbitals, occ);
    double gpu_ms = rho_gpu.ok() ? rho_gpu.value().kernel_ms : -1.0;
    double diff = rho_gpu.ok() ? MaxAbsDiff(rho_gpu.value().rho, rho_cpu) : -1.0;

    PrintRow(std::to_string(n_per_axis) + "^3", cpu_ms, gpu_ms, diff);
  }
}

// --- Vmat Build Benchmark ---
void BenchVmatBuild() {
  PrintHeader("Vmat Build (v->H adjoint)");
  for (int n_per_axis : {16, 24, 32, 48, 64}) {
    UniformGrid3D grid;
    grid.n = {static_cast<std::size_t>(n_per_axis),
              static_cast<std::size_t>(n_per_axis),
              static_cast<std::size_t>(n_per_axis)};
    grid.h = {0.3, 0.3, 0.3};
    grid.origin = {-2.4, -2.4, -2.4};
    const std::size_t N = grid.total_points();

    auto orb0 = RhoBuilder::GaussianOrbital(grid, 1.0, {0, 0, 0});
    auto orb1 = RhoBuilder::GaussianOrbital(grid, 1.5, {0.3, 0, 0});
    std::vector<std::vector<double>> orbitals = {orb0, orb1};

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> v(N);
    for (auto& x : v) x = dist(rng);

    auto t0 = std::chrono::steady_clock::now();
    auto H_cpu = VmatBuilder::BuildHmat(grid, orbitals, v);
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto H_gpu = tides::grid::VmatBuildCuda(grid, orbitals, v);
    double gpu_ms = H_gpu.ok() ? H_gpu.value().kernel_ms : -1.0;
    double diff = H_gpu.ok() ? MaxAbsDiff(H_gpu.value().H, H_cpu) : -1.0;

    PrintRow(std::to_string(n_per_axis) + "^3", cpu_ms, gpu_ms, diff);
  }
}

// --- XC Evaluation Benchmark ---
void BenchXC() {
  PrintHeader("XC Evaluation (LDA)");
  for (int n_per_axis : {16, 24, 32, 48, 64}) {
    UniformGrid3D grid;
    grid.n = {static_cast<std::size_t>(n_per_axis),
              static_cast<std::size_t>(n_per_axis),
              static_cast<std::size_t>(n_per_axis)};
    grid.h = {0.3, 0.3, 0.3};
    grid.origin = {-2.4, -2.4, -2.4};
    const std::size_t N = grid.total_points();

    std::vector<double> rho(N);
    for (std::size_t i = 0; i < N; ++i) {
      const auto [ix, iy, iz] = grid.unflatten(i);
      const auto [x, y, z] = grid.coord(ix, iy, iz);
      rho[i] = 0.1 + 0.5 * std::exp(-(x*x + y*y + z*z));
    }

    auto t0 = std::chrono::steady_clock::now();
    auto xc_cpu = XCGridEvaluator::EvaluateLDA(grid, rho);
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto xc_gpu = RunLdaXcOnHostGrid(grid, rho);
    double gpu_ms = xc_gpu.kernel_ms;
    double diff = MaxAbsDiff(xc_gpu.vxc, xc_cpu.vxc);

    PrintRow(std::to_string(n_per_axis) + "^3", cpu_ms, gpu_ms, diff);
  }
}

// --- SP2 Purification Benchmark ---
void BenchSP2() {
  PrintHeader("SP2 Purification (density matrix)");
  for (int n : {20, 50, 100, 200, 500}) {
    const std::size_t n_occ = static_cast<std::size_t>(n) / 2;
    const double gap = 2.0;

    std::mt19937_64 rng(42);
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<double> lam(n);
    for (int i = 0; i < n; ++i) {
      if (static_cast<std::size_t>(i) < n_occ)
        lam[i] = -2.0 + static_cast<double>(i) / n_occ;
      else
        lam[i] = -1.0 + gap + static_cast<double>(i - n_occ) / (n - n_occ);
    }
    std::vector<double> Q(n * n);
    for (auto& v : Q) v = g(rng);
    for (std::size_t j = 0; j < static_cast<std::size_t>(n); ++j) {
      for (std::size_t k = 0; k < j; ++k) {
        double dot = 0.0;
        for (int i = 0; i < n; ++i)
          dot += Q[i * n + j] * Q[i * n + k];
        for (int i = 0; i < n; ++i)
          Q[i * n + j] -= dot * Q[i * n + k];
      }
      double nrm = 0.0;
      for (int i = 0; i < n; ++i) nrm += Q[i * n + j] * Q[i * n + j];
      nrm = std::sqrt(nrm);
      for (int i = 0; i < n; ++i) Q[i * n + j] /= nrm;
    }
    std::vector<double> H(n * n, 0.0), S(n * n, 0.0);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        double s = 0.0;
        for (int k = 0; k < n; ++k)
          s += Q[i * n + k] * lam[k] * Q[j * n + k];
        H[i * n + j] = s;
      }
    for (int i = 0; i < n; ++i) S[i * n + i] = 1.0;

    auto ref = BatchedDenseEig::SolveGeneralized(n, H, S);
    if (!ref.ok) continue;
    const double mu = 0.5 * (ref.eigenvalues[n_occ - 1] + ref.eigenvalues[n_occ]);
    const double n_e = static_cast<double>(n_occ);

    auto t0 = std::chrono::steady_clock::now();
    auto cpu = SP2Purification::Purify(n, H, S, n_e, mu,
                                       ref.eigenvalues[0],
                                       ref.eigenvalues[n - 1]);
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto gpu = tides::solvers::SP2PurifyCuda(n, H, S, n_e, mu,
                                             ref.eigenvalues[0],
                                             ref.eigenvalues[n - 1]);
    double gpu_ms = gpu.ok() ? gpu.value().kernel_ms : -1.0;
    double diff = gpu.ok() ? MaxAbsDiff(gpu.value().P, cpu.P) : -1.0;

    PrintRow("n=" + std::to_string(n), cpu_ms, gpu_ms, diff);
  }
}

}  // namespace

int main() {
  std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
            << "║          TIDES P4: CPU vs GPU Benchmark Suite                ║\n"
            << "╚══════════════════════════════════════════════════════════════╝\n";

  BenchRhoBuild();
  BenchVmatBuild();
  BenchXC();
  BenchSP2();

  std::cout << "\n=== Benchmark Summary ===\n"
            << "All benchmarks completed. Speedup = CPU_ms / GPU_ms.\n"
            << "Note: GPU times include H2D/D2H transfer for first call.\n"
            << "For small problems, CPU may be faster due to transfer overhead.\n";
  return 0;
}
