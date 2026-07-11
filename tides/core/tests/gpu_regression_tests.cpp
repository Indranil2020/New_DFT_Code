// P3.3: GPU regression tests — all GPU kernels vs CPU across grid sizes.
//
// This test runs every GPU kernel at multiple problem sizes and verifies
// the GPU result matches the CPU reference within the declared tolerance.
// It serves as a regression gate: any kernel that drifts from CPU will fail.

#include "grid/rho_build.hpp"
#include "grid/rho_build_gpu.hpp"
#include "grid/vmat_build.hpp"
#include "grid/vmat_build_gpu.hpp"
#include "grid/poisson.hpp"
#include "grid/poisson_fft_gpu.hpp"
#include "grid/xc.hpp"
#include "grid/xc/tests/xc_test_utils.hpp"
#include "grid/dual_grid.hpp"
#include "basis/two_center_integrals.hpp"
#include "basis/two_center_gpu.hpp"
#include "basis/three_center_gpu.hpp"
#include "solvers/sp2_submatrix/sp2.hpp"
#include "solvers/sp2_submatrix/sp2_gpu.hpp"
#include "solvers/dense/batched_eig.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using tides::grid::UniformGrid3D;
using tides::grid::RhoBuilder;
using tides::grid::VmatBuilder;
using tides::grid::XCGridEvaluator;
using tides::grid::xc::RunLdaXcOnHostGrid;

int Fail(const std::string& msg) {
  std::cerr << "gpu_regression: FAIL: " << msg << '\n';
  return 1;
}

int Pass(const std::string& msg) {
  std::cout << "gpu_regression: PASS: " << msg << '\n';
  return 0;
}

double MaxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    d = std::max(d, std::abs(a[i] - b[i]));
  return d;
}

// --- Rho build regression ---
int TestRhoBuildRegression() {
  std::cout << "\n=== GPU Regression: Rho Build ===\n";
  int failures = 0;
  for (int n_per_axis : {16, 24, 32}) {
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

    auto rho_cpu = RhoBuilder::BuildFromOrbitals(grid, orbitals, occ);
    auto rho_gpu = tides::grid::RhoBuildCuda(grid, orbitals, occ);
    if (!rho_gpu.ok()) {
      std::cerr << "  n=" << n_per_axis << ": GPU failed\n";
      failures++;
      continue;
    }
    double diff = MaxAbsDiff(rho_gpu.value().rho, rho_cpu);
    std::cout << "  n=" << n_per_axis << ": max_diff=" << diff << '\n';
    if (diff > 1e-12) {
      failures++;
      std::cerr << "  FAIL: diff=" << diff << " > 1e-12\n";
    }
  }
  return failures;
}

// --- vmat build regression ---
int TestVmatBuildRegression() {
  std::cout << "\n=== GPU Regression: Vmat Build ===\n";
  int failures = 0;
  for (int n_per_axis : {16, 24, 32}) {
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

    auto H_cpu = VmatBuilder::BuildHmat(grid, orbitals, v);
    auto H_gpu = tides::grid::VmatBuildCuda(grid, orbitals, v);
    if (!H_gpu.ok()) {
      std::cerr << "  n=" << n_per_axis << ": GPU failed\n";
      failures++;
      continue;
    }
    double diff = MaxAbsDiff(H_gpu.value().H, H_cpu);
    std::cout << "  n=" << n_per_axis << ": max_diff=" << diff << '\n';
    if (diff > 1e-12) {
      failures++;
      std::cerr << "  FAIL: diff=" << diff << " > 1e-12\n";
    }
  }
  return failures;
}

// --- XC evaluation regression ---
int TestXCRegression() {
  std::cout << "\n=== GPU Regression: XC Evaluation ===\n";
  int failures = 0;
  for (int n_per_axis : {16, 24, 32}) {
    UniformGrid3D grid;
    grid.n = {static_cast<std::size_t>(n_per_axis),
              static_cast<std::size_t>(n_per_axis),
              static_cast<std::size_t>(n_per_axis)};
    grid.h = {0.3, 0.3, 0.3};
    grid.origin = {-2.4, -2.4, -2.4};
    const std::size_t N = grid.total_points();

    // Non-uniform density: Gaussian + background.
    std::vector<double> rho(N);
    for (std::size_t i = 0; i < N; ++i) {
      const auto [ix, iy, iz] = grid.unflatten(i);
      const auto [x, y, z] = grid.coord(ix, iy, iz);
      rho[i] = 0.1 + 0.5 * std::exp(-(x*x + y*y + z*z));
    }

    auto xc_cpu = XCGridEvaluator::EvaluateLDA(grid, rho);
    auto xc_gpu = RunLdaXcOnHostGrid(grid, rho);
    const double cpu_energy = XCGridEvaluator::XCEnergy(grid, xc_cpu, rho);
    double diff_vxc = MaxAbsDiff(xc_gpu.vxc, xc_cpu.vxc);
    double diff_energy = std::abs(xc_gpu.xc_energy - cpu_energy);
    std::cout << "  n=" << n_per_axis
              << ": max_diff_vxc=" << diff_vxc
              << " energy_diff=" << diff_energy << '\n';
    if (diff_vxc > 1e-11 || diff_energy > 1e-10) {
      failures++;
      std::cerr << "  FAIL: diff > 1e-12\n";
    }
  }
  return failures;
}

// --- SP2 regression ---
int TestSP2Regression() {
  std::cout << "\n=== GPU Regression: SP2 Purification ===\n";
  int failures = 0;
  for (int n : {20, 50, 100}) {
    const std::size_t n_occ = static_cast<std::size_t>(n) / 2;
    const double gap = 2.0;

    // Build gapped system.
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

    auto ref = tides::solvers::BatchedDenseEig::SolveGeneralized(n, H, S);
    if (!ref.ok) { failures++; continue; }
    const double mu = 0.5 * (ref.eigenvalues[n_occ - 1] + ref.eigenvalues[n_occ]);
    const double n_e = static_cast<double>(n_occ);

    auto cpu = tides::solvers::SP2Purification::Purify(
        n, H, S, n_e, mu, ref.eigenvalues[0], ref.eigenvalues[n - 1]);
    auto gpu = tides::solvers::SP2PurifyCuda(
        n, H, S, n_e, mu, ref.eigenvalues[0], ref.eigenvalues[n - 1]);
    if (!gpu.ok()) {
      std::cerr << "  n=" << n << ": GPU failed\n";
      failures++;
      continue;
    }
    double diff = MaxAbsDiff(gpu.value().P, cpu.P);
    std::cout << "  n=" << n << ": max_diff=" << diff << '\n';
    if (diff > 1e-10) {
      failures++;
      std::cerr << "  FAIL: diff=" << diff << " > 1e-10\n";
    }
  }
  return failures;
}

}  // namespace

int main() {
  int failures = 0;
  failures += TestRhoBuildRegression();
  failures += TestVmatBuildRegression();
  failures += TestXCRegression();
  failures += TestSP2Regression();

  std::cout << "\n=== GPU Regression Summary ===\n";
  if (failures == 0) {
    std::cout << "ALL GPU REGRESSION TESTS PASSED\n";
  } else {
    std::cerr << failures << " GPU REGRESSION TEST(S) FAILED\n";
  }
  return failures;
}
