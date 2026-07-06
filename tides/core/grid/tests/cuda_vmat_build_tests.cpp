// T3.3: GPU v->H adjoint map tests — vs CPU reference.
//
// Validates that the CUDA vmat_build kernel produces H_ij matching the CPU
// VmatBuilder::BuildHmat within <=1e-12, and that adjointness holds.

#include "grid/vmat_build.hpp"
#include "grid/vmat_build_gpu.hpp"
#include "grid/rho_build.hpp"
#include "grid/dual_grid.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

namespace {

using tides::grid::UniformGrid3D;
using tides::grid::VmatBuilder;
using tides::grid::VmatCudaAvailable;
using tides::grid::VmatBuildCuda;
using tides::grid::VmatGpuResult;
using tides::grid::RhoBuilder;

UniformGrid3D MakeTestGrid() {
  UniformGrid3D grid;
  grid.n = {16, 16, 16};
  grid.h = {0.3, 0.3, 0.3};
  grid.origin = {-2.4, -2.4, -2.4};
  return grid;
}

double MaxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    d = std::max(d, std::abs(a[i] - b[i]));
  return d;
}

int TestVmatVsCpu() {
  auto grid = MakeTestGrid();
  const std::size_t N = grid.total_points();

  // Two Gaussian orbitals.
  auto orb0 = RhoBuilder::GaussianOrbital(grid, 1.0, {0, 0, 0});
  auto orb1 = RhoBuilder::GaussianOrbital(grid, 1.5, {0.3, 0, 0});
  std::vector<std::vector<double>> orbitals = {orb0, orb1};
  const std::size_t n_orb = 2;

  // Random potential.
  std::mt19937_64 rng(123);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> v(N);
  for (auto& x : v) x = dist(rng);

  // CPU reference.
  auto H_cpu = VmatBuilder::BuildHmat(grid, orbitals, v);

  // GPU.
  auto gpu_result = VmatBuildCuda(grid, orbitals, v);
  if (!gpu_result.ok()) {
    std::cerr << "VmatBuildCuda failed: " << gpu_result.status().message()
              << '\n';
    return 1;
  }

  const double diff = MaxAbsDiff(gpu_result.value().H, H_cpu);

  std::cout << "vmat_vs_cpu: n_orb=" << n_orb << " N=" << N
            << " kernel_ms=" << gpu_result.value().kernel_ms
            << " max_diff=" << diff << '\n';

  if (diff > 1e-12) {
    std::cerr << "FAIL: max_diff=" << diff << " > 1e-12\n";
    return 1;
  }
  return 0;
}

int TestVmatLarger() {
  // 4 orbitals on a 24^3 grid.
  UniformGrid3D grid;
  grid.n = {24, 24, 24};
  grid.h = {0.2, 0.2, 0.2};
  grid.origin = {-2.4, -2.4, -2.4};
  const std::size_t N = grid.total_points();

  auto orb0 = RhoBuilder::GaussianOrbital(grid, 0.8, {0, 0, 0});
  auto orb1 = RhoBuilder::GaussianOrbital(grid, 1.2, {0.5, 0, 0});
  auto orb2 = RhoBuilder::GaussianOrbital(grid, 1.0, {0, 0.5, 0});
  auto orb3 = RhoBuilder::GaussianOrbital(grid, 1.5, {0, 0, 0.5});
  std::vector<std::vector<double>> orbitals = {orb0, orb1, orb2, orb3};
  const std::size_t n_orb = 4;

  std::mt19937_64 rng(456);
  std::uniform_real_distribution<double> dist(-2.0, 2.0);
  std::vector<double> v(N);
  for (auto& x : v) x = dist(rng);

  auto H_cpu = VmatBuilder::BuildHmat(grid, orbitals, v);

  auto gpu_result = VmatBuildCuda(grid, orbitals, v);
  if (!gpu_result.ok()) {
    std::cerr << "VmatBuildCuda failed: " << gpu_result.status().message()
              << '\n';
    return 1;
  }

  const double diff = MaxAbsDiff(gpu_result.value().H, H_cpu);

  std::cout << "vmat_larger: n_orb=" << n_orb << " N=" << N
            << " kernel_ms=" << gpu_result.value().kernel_ms
            << " max_diff=" << diff << '\n';

  if (diff > 1e-12) {
    std::cerr << "FAIL: max_diff=" << diff << " > 1e-12\n";
    return 1;
  }
  return 0;
}

int TestAdjointness() {
  // Verify adjointness: <A P, w> == <P, A^T w> using GPU v->H.
  auto grid = MakeTestGrid();
  const std::size_t N = grid.total_points();

  auto orb0 = RhoBuilder::GaussianOrbital(grid, 1.0, {0, 0, 0});
  auto orb1 = RhoBuilder::GaussianOrbital(grid, 1.5, {0.3, 0, 0});
  std::vector<std::vector<double>> orbitals = {orb0, orb1};
  const std::size_t n_orb = 2;

  std::mt19937_64 rng(789);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  const auto [h0, h1, h2] = grid.h;
  const double dv = h0 * h1 * h2;

  double max_err = 0.0;
  for (int trial = 0; trial < 100; ++trial) {
    std::vector<double> P(n_orb * n_orb, 0.0);
    P[0] = dist(rng); P[3] = dist(rng);
    P[1] = P[2] = dist(rng);

    std::vector<double> w(N);
    for (auto& x : w) x = dist(rng);

    // Forward: rho_P = A(P) using CPU.
    auto rho_P = VmatBuilder::BuildRho(grid, orbitals, P);
    double AP_w = 0.0;
    for (std::size_t g = 0; g < N; ++g)
      AP_w += rho_P[g] * w[g] * dv;

    // Adjoint: H_w = A^T(w) using GPU.
    auto gpu_H = VmatBuildCuda(grid, orbitals, w);
    if (!gpu_H.ok()) {
      std::cerr << "VmatBuildCuda failed in adjointness test\n";
      return 1;
    }
    double P_ATw = 0.0;
    for (std::size_t i = 0; i < n_orb; ++i)
      for (std::size_t j = 0; j < n_orb; ++j)
        P_ATw += P[i * n_orb + j] * gpu_H.value().H[i * n_orb + j];

    max_err = std::max(max_err, std::fabs(AP_w - P_ATw));
  }

  std::cout << "adjointness_gpu: max_err over 100 pairs = " << max_err << '\n';
  if (max_err > 1e-12) {
    std::cerr << "FAIL: adjointness error " << max_err << " > 1e-12\n";
    return 1;
  }
  return 0;
}

}  // namespace

int main() {
  if (!VmatCudaAvailable()) {
    std::cout << "SKIP: CUDA runtime not available\n";
    return 77;
  }

  int failures = 0;
  failures += TestVmatVsCpu();
  failures += TestVmatLarger();
  failures += TestAdjointness();

  if (failures == 0) {
    std::cout << "All GPU vmat_build tests passed.\n";
  } else {
    std::cerr << failures << " GPU vmat_build test(s) failed.\n";
  }
  return failures;
}
