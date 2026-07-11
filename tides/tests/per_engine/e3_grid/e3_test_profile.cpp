// E3: Grid Engine — Comprehensive Test & Profile Suite
//
// Tests and profiles every WP3 kernel:
//   1. Dual-grid index mapping + halo
//   2. Rho build (CPU + GPU) — density from orbitals
//   3. Vmat build (CPU + GPU) — adjoint map v→H
//   4. Poisson FFT (CPU + GPU) — Hartree potential
//   5. XC evaluation (CPU + GPU) — LDA
//   6. Adjointness test — <rho|V> consistency

#include "grid/dual_grid.hpp"
#include "grid/rho_build.hpp"
#include "grid/rho_build_gpu.hpp"
#include "grid/vmat_build.hpp"
#include "grid/vmat_build_gpu.hpp"
#include "grid/poisson.hpp"
#include "grid/poisson_fft_gpu.hpp"
#include "grid/xc.hpp"
#include "grid/xc/tests/xc_test_utils.hpp"
#include "common/status.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using tides::grid::UniformGrid3D;
using tides::grid::BoundaryCondition;
using tides::grid::RhoBuilder;
using tides::grid::VmatBuilder;
using tides::grid::PoissonSolver;
using tides::grid::XCGridEvaluator;
using tides::grid::xc::RunLdaXcOnHostGrid;

struct ProfileEntry {
  std::string kernel;
  std::string variant;
  std::string size_label;
  double time_ms = 0.0;
  double error = 0.0;
  std::string status;
};

std::vector<ProfileEntry> g_log;

void Log(const std::string& kernel, const std::string& variant,
         const std::string& size_label, double time_ms, double error,
         const std::string& status) {
  ProfileEntry e{kernel, variant, size_label, time_ms, error, status};
  g_log.push_back(e);
  std::cout << "  " << std::left << std::setw(18) << kernel
            << std::setw(12) << variant
            << std::setw(14) << size_label
            << std::right << std::setw(10) << std::fixed << std::setprecision(3) << time_ms
            << std::setw(14) << std::scientific << std::setprecision(3) << error
            << "  " << status << '\n';
}

void PrintHeader() {
  std::cout << std::left << std::setw(18) << "Kernel"
            << std::setw(12) << "Variant"
            << std::setw(14) << "Size"
            << std::right << std::setw(10) << "Time(ms)"
            << std::setw(14) << "Error"
            << "  Status\n";
  std::cout << std::string(82, '-') << '\n';
}

double MaxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    d = std::max(d, std::abs(a[i] - b[i]));
  return d;
}

UniformGrid3D MakeGrid(int n, double h = 0.2) {
  UniformGrid3D g;
  g.n = {static_cast<std::size_t>(n), static_cast<std::size_t>(n), static_cast<std::size_t>(n)};
  g.h = {h, h, h};
  g.origin = {0.0, 0.0, 0.0};
  g.bc = {BoundaryCondition::kFree, BoundaryCondition::kFree, BoundaryCondition::kFree};
  return g;
}

// --- Test 1: Dual-grid index mapping ---
int TestDualGrid() {
  std::cout << "\n=== E3.1: Dual-grid Index Mapping ===\n";
  PrintHeader();
  int failures = 0;

  for (auto nc : {32, 48, 64}) {
    UniformGrid3D grid = MakeGrid(nc);
    auto t0 = std::chrono::steady_clock::now();
    double err = 0.0;
    for (std::size_t i = 0; i < grid.total_points(); ++i) {
      auto [ix, iy, iz] = grid.unflatten(i);
      std::size_t back = grid.flatten(ix, iy, iz);
      err = std::max(err, static_cast<double>(std::abs(static_cast<long>(back) - static_cast<long>(i))));
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::string status = (err < 1e-15) ? "PASS" : "FAIL";
    if (err >= 1e-15) failures++;
    Log("DualGrid", "flatten",
        std::to_string(nc) + "^3",
        ms, err, status);
  }
  return failures;
}

// --- Test 2: Rho build (CPU + GPU) ---
int TestRhoBuild() {
  std::cout << "\n=== E3.2: Rho Build (CPU + GPU) ===\n";
  PrintHeader();
  int failures = 0;
  std::mt19937_64 rng(42);

  for (auto [n_grid, n_orb] : std::vector<std::pair<int, int>>{
           {16, 4}, {24, 8}, {32, 16}, {48, 32}}) {
    UniformGrid3D grid = MakeGrid(n_grid);
    std::size_t N = grid.total_points();
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<std::vector<double>> orbitals(n_orb);
    for (int k = 0; k < n_orb; ++k) {
      orbitals[k].resize(N);
      for (auto& v : orbitals[k]) v = dist(rng);
    }
    std::vector<double> occ(n_orb, 2.0);

    // CPU rho build.
    auto t0 = std::chrono::steady_clock::now();
    auto cpu_rho = RhoBuilder::BuildFromOrbitals(grid, orbitals, occ);
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // GPU rho build.
    auto t2 = std::chrono::steady_clock::now();
    auto gpu = tides::grid::RhoBuildCuda(grid, orbitals, occ);
    auto t3 = std::chrono::steady_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    if (!gpu.ok()) {
      Log("RhoBuild", "GPU",
          std::to_string(n_grid) + "^3x" + std::to_string(n_orb),
          gpu_ms, 0, "FAIL: " + gpu.status().message());
      failures++;
      continue;
    }

    double diff = MaxAbsDiff(cpu_rho, gpu.value().rho);
    std::string status = (diff < 1e-10) ? "PASS" : "FAIL";
    if (diff >= 1e-10) failures++;

    Log("RhoBuild", "CPU",
        std::to_string(n_grid) + "^3x" + std::to_string(n_orb),
        cpu_ms, 0, "ref");
    Log("RhoBuild", "GPU",
        std::to_string(n_grid) + "^3x" + std::to_string(n_orb),
        gpu_ms, diff, status);
  }
  return failures;
}

// --- Test 3: Vmat build (CPU + GPU) ---
int TestVmatBuild() {
  std::cout << "\n=== E3.3: Vmat Build (CPU + GPU) ===\n";
  PrintHeader();
  int failures = 0;
  std::mt19937_64 rng(42);

  for (auto [n_grid, n_orb] : std::vector<std::pair<int, int>>{
           {16, 4}, {24, 8}, {32, 16}, {48, 32}}) {
    UniformGrid3D grid = MakeGrid(n_grid);
    std::size_t N = grid.total_points();
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> v_grid(N);
    for (auto& v : v_grid) v = dist(rng);

    std::vector<std::vector<double>> orbitals(n_orb);
    for (int k = 0; k < n_orb; ++k) {
      orbitals[k].resize(N);
      for (auto& v : orbitals[k]) v = dist(rng);
    }

    // CPU vmat build.
    auto t0 = std::chrono::steady_clock::now();
    auto cpu_H = VmatBuilder::BuildHmat(grid, orbitals, v_grid);
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // GPU vmat build.
    auto t2 = std::chrono::steady_clock::now();
    auto gpu = tides::grid::VmatBuildCuda(grid, orbitals, v_grid);
    auto t3 = std::chrono::steady_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    if (!gpu.ok()) {
      Log("VmatBuild", "GPU",
          std::to_string(n_grid) + "^3x" + std::to_string(n_orb),
          gpu_ms, 0, "FAIL: " + gpu.status().message());
      failures++;
      continue;
    }

    double diff = MaxAbsDiff(cpu_H, gpu.value().H);
    std::string status = (diff < 1e-10) ? "PASS" : "FAIL";
    if (diff >= 1e-10) failures++;

    Log("VmatBuild", "CPU",
        std::to_string(n_grid) + "^3x" + std::to_string(n_orb),
        cpu_ms, 0, "ref");
    Log("VmatBuild", "GPU",
        std::to_string(n_grid) + "^3x" + std::to_string(n_orb),
        gpu_ms, diff, status);
  }
  return failures;
}

// --- Test 4: Poisson FFT (CPU + GPU) ---
int TestPoissonFFT() {
  std::cout << "\n=== E3.4: Poisson FFT (CPU + GPU) ===\n";
  PrintHeader();
  int failures = 0;

  // NOTE: CPU Poisson uses naive O(N^2) DFT for periodic BC.
  // n=16 -> N=4096 -> 16M ops (fast). n=32 -> N=32768 -> 1B ops (~10s).
  // Larger sizes would hang the CPU reference. GPU uses cuFFT (O(N log N)).
  for (auto n : {16, 32}) {
    UniformGrid3D grid = MakeGrid(n);
    std::size_t N = grid.total_points();
    std::vector<double> rho(N, 0.0);

    // Gaussian charge distribution centered in grid.
    double sigma = static_cast<double>(n) * grid.h[0] / 6.0;
    double center = static_cast<double>(n) * grid.h[0] / 2.0;
    for (std::size_t i = 0; i < N; ++i) {
      auto [ix, iy, iz] = grid.unflatten(i);
      auto [x, y, z] = grid.coord(ix, iy, iz);
      double dx = x - center, dy = y - center, dz = z - center;
      rho[i] = std::exp(-(dx*dx + dy*dy + dz*dz) / (2 * sigma * sigma));
    }

    // CPU Poisson (periodic for FFT comparison).
    UniformGrid3D grid_p = grid;
    grid_p.bc = {BoundaryCondition::kPeriodic, BoundaryCondition::kPeriodic, BoundaryCondition::kPeriodic};

    auto t0 = std::chrono::steady_clock::now();
    auto cpu_v = PoissonSolver::Solve(grid_p, rho);
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // GPU Poisson.
    auto t2 = std::chrono::steady_clock::now();
    auto gpu = tides::grid::PoissonFftCuda(grid_p, rho);
    auto t3 = std::chrono::steady_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    if (!gpu.ok()) {
      Log("Poisson", "GPU",
          std::to_string(n) + "^3",
          gpu_ms, 0, "FAIL: " + gpu.status().message());
      failures++;
      continue;
    }

    double diff = MaxAbsDiff(cpu_v, gpu.value().V);
    std::string status = (diff < 1e-8) ? "PASS" : "FAIL";
    if (diff >= 1e-8) failures++;

    Log("Poisson", "CPU",
        std::to_string(n) + "^3",
        cpu_ms, 0, "ref");
    Log("Poisson", "GPU",
        std::to_string(n) + "^3",
        gpu_ms, diff, status);
  }
  return failures;
}

// --- Test 5: XC evaluation (CPU + GPU) ---
int TestXCEvaluation() {
  std::cout << "\n=== E3.5: XC Evaluation (CPU + GPU) ===\n";
  PrintHeader();
  int failures = 0;
  std::mt19937_64 rng(42);

  for (auto n : {16, 24, 32, 48}) {
    UniformGrid3D grid = MakeGrid(n);
    std::size_t N = grid.total_points();

    // Generate random positive densities.
    std::uniform_real_distribution<double> dist(1e-4, 1.0);
    std::vector<double> rho(N);
    for (auto& v : rho) v = dist(rng);

    // CPU XC.
    auto t0 = std::chrono::steady_clock::now();
    auto cpu_xc = XCGridEvaluator::EvaluateLDA(grid, rho);
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // GPU XC.
    auto t2 = std::chrono::steady_clock::now();
    auto gpu = RunLdaXcOnHostGrid(grid, rho);
    auto t3 = std::chrono::steady_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    const double cpu_energy = XCGridEvaluator::XCEnergy(grid, cpu_xc, rho);
    double diff_energy = std::abs(cpu_energy - gpu.xc_energy);
    double diff_vxc = MaxAbsDiff(cpu_xc.vxc, gpu.vxc);
    double diff = std::max(diff_energy, diff_vxc);

    std::string status = (diff < 1e-10) ? "PASS" : "FAIL";
    if (diff >= 1e-10) failures++;

    Log("XC-LDA", "CPU",
        std::to_string(n) + "^3",
        cpu_ms, 0, "ref");
    Log("XC-LDA", "GPU",
        std::to_string(n) + "^3",
        gpu_ms, diff, status);
  }
  return failures;
}

// --- Test 6: Adjointness <rho|V> consistency ---
int TestAdjointness() {
  std::cout << "\n=== E3.6: Adjointness (rho→V→H consistency) ===\n";
  PrintHeader();
  int failures = 0;
  std::mt19937_64 rng(42);

  int n = 16;
  int n_orb = 4;
  UniformGrid3D grid = MakeGrid(n);
  std::size_t N = grid.total_points();
  std::normal_distribution<double> dist(0.0, 1.0);

  std::vector<std::vector<double>> orbitals(n_orb);
  for (int k = 0; k < n_orb; ++k) {
    orbitals[k].resize(N);
    for (auto& v : orbitals[k]) v = dist(rng);
  }

  // Random density matrix P and potential w.
  std::vector<double> P(n_orb * n_orb);
  for (auto& v : P) v = dist(rng);
  for (int i = 0; i < n_orb; ++i)
    for (int j = i; j < n_orb; ++j)
      P[j * n_orb + i] = P[i * n_orb + j];

  std::vector<double> w(N);
  for (auto& v : w) v = dist(rng);

  double adj_err = VmatBuilder::CheckAdjointness(grid, orbitals, P, w);

  std::string status = (adj_err < 1e-12) ? "PASS" : "FAIL";
  if (adj_err >= 1e-12) failures++;
  Log("Adjointness", "<AP,w>=<P,ATw>",
      "16^3x4",
      0, adj_err, status);

  // Also check H symmetry.
  auto H = VmatBuilder::BuildHmat(grid, orbitals, w);
  double sym_err = 0.0;
  for (int i = 0; i < n_orb; ++i)
    for (int j = 0; j < n_orb; ++j)
      sym_err = std::max(sym_err, std::abs(H[i * n_orb + j] - H[j * n_orb + i]));

  status = (sym_err < 1e-14) ? "PASS" : "FAIL";
  if (sym_err >= 1e-14) failures++;
  Log("Adjointness", "H-symmetry",
      "16^3x4",
      0, sym_err, status);

  return failures;
}

void PrintSummary(int total_failures) {
  std::cout << "\n=== E3 Summary ===\n";
  std::cout << "Total profile entries: " << g_log.size() << '\n';
  if (total_failures == 0) {
    std::cout << "ALL E3 TESTS PASSED\n";
  } else {
    std::cout << total_failures << " E3 TEST(S) FAILED\n";
  }
}

}  // namespace

int main() {
  std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
            << "║   E3: Grid Engine — Test & Profile Suite                     ║\n"
            << "╚══════════════════════════════════════════════════════════════╝\n";

  int failures = 0;
  failures += TestDualGrid();
  failures += TestRhoBuild();
  failures += TestVmatBuild();
  failures += TestPoissonFFT();
  failures += TestXCEvaluation();
  failures += TestAdjointness();

  PrintSummary(failures);
  return failures;
}
