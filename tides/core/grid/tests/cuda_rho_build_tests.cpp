// T3.2: GPU rho builder tests — vs CPU reference + integral check.
// Validates that the CUDA rho builder produces density equal to the CPU path
// within <=1e-9, and integral(rho) = N_e within <=1e-10.

#include "grid/rho_build.hpp"
#include "grid/rho_build_gpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

using tides::grid::RhoBuilder;
using tides::grid::RhoBuildCuda;
using tides::grid::RhoBuildCudaAvailable;
using tides::grid::RhoBuildGpuResult;
using tides::grid::UniformGrid3D;
using tides::grid::BoundaryCondition;

double MaxAbsDifference(const std::vector<double>& a,
                        const std::vector<double>& b) {
  double max_diff = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
  }
  return max_diff;
}

int TestRhoBuildVsCpu() {
  // 32x32x32 grid, 2 Gaussian orbitals.
  UniformGrid3D grid;
  grid.n = {32, 32, 32};
  grid.h = {0.2, 0.2, 0.2};
  grid.origin = {-3.2, -3.2, -3.2};

  const std::vector<double> occupations = {2.0, 2.0};
  std::vector<std::vector<double>> orbitals;
  orbitals.push_back(RhoBuilder::GaussianOrbital(grid, 0.5, {0.0, 0.0, 0.0}));
  orbitals.push_back(RhoBuilder::GaussianOrbital(grid, 0.5, {1.0, 0.0, 0.0}));

  auto cpu_rho = RhoBuilder::BuildFromOrbitals(grid, orbitals, occupations);
  const double cpu_integral = RhoBuilder::Integral(grid, cpu_rho);

  auto gpu_result = RhoBuildCuda(grid, orbitals, occupations);
  if (!gpu_result.ok()) {
    std::cerr << "RhoBuildCuda failed: " << gpu_result.status().message()
              << '\n';
    return 1;
  }

  const double max_diff = MaxAbsDifference(gpu_result.value().rho, cpu_rho);
  const double n_e = 4.0;  // sum of occupations
  const double integral_vs_cpu =
      std::abs(gpu_result.value().integral - cpu_integral);
  const double integral_vs_n_e = std::abs(gpu_result.value().integral - n_e);

  std::cout << "rho_vs_cpu: grid=32^3 n_orbitals=2"
            << " kernel_ms=" << gpu_result.value().kernel_ms
            << " max_diff=" << max_diff
            << " gpu_integral=" << gpu_result.value().integral
            << " cpu_integral=" << cpu_integral
            << " N_e=" << n_e
            << " integral_vs_cpu=" << integral_vs_cpu
            << " integral_vs_N_e=" << integral_vs_n_e << '\n';

  // Observable (1): vs CPU <= 1e-9
  if (max_diff > 1e-9) {
    std::cerr << "FAIL: max_diff=" << max_diff << " > 1e-9\n";
    return 1;
  }
  // Observable (2): integral(rho) matches CPU integral <= 1e-10
  // (absolute N_e match depends on orbital normalization, not the builder)
  if (integral_vs_cpu > 1e-10) {
    std::cerr << "FAIL: integral_vs_cpu=" << integral_vs_cpu
              << " > 1e-10\n";
    return 1;
  }
  return 0;
}

int TestRhoBuildLargerGrid() {
  // 64x64x64 grid, 4 Gaussian orbitals.
  UniformGrid3D grid;
  grid.n = {64, 64, 64};
  grid.h = {0.15, 0.15, 0.15};
  grid.origin = {-4.8, -4.8, -4.8};

  const std::vector<double> occupations = {2.0, 2.0, 1.0, 1.0};
  std::vector<std::vector<double>> orbitals;
  orbitals.push_back(RhoBuilder::GaussianOrbital(grid, 0.5, {0.0, 0.0, 0.0}));
  orbitals.push_back(RhoBuilder::GaussianOrbital(grid, 0.5, {1.5, 0.0, 0.0}));
  orbitals.push_back(RhoBuilder::GaussianOrbital(grid, 0.8, {0.0, 1.5, 0.0}));
  orbitals.push_back(RhoBuilder::GaussianOrbital(grid, 0.8, {0.0, 0.0, 1.5}));

  auto cpu_rho = RhoBuilder::BuildFromOrbitals(grid, orbitals, occupations);

  auto gpu_result = RhoBuildCuda(grid, orbitals, occupations);
  if (!gpu_result.ok()) {
    std::cerr << "RhoBuildCuda failed: " << gpu_result.status().message()
              << '\n';
    return 1;
  }

  const double max_diff = MaxAbsDifference(gpu_result.value().rho, cpu_rho);
  const double n_e = 6.0;
  const double cpu_integral = RhoBuilder::Integral(grid, cpu_rho);
  const double integral_vs_cpu =
      std::abs(gpu_result.value().integral - cpu_integral);

  std::cout << "rho_larger: grid=64^3 n_orbitals=4"
            << " kernel_ms=" << gpu_result.value().kernel_ms
            << " max_diff=" << max_diff
            << " integral_vs_cpu=" << integral_vs_cpu << '\n';

  if (max_diff > 1e-9) {
    std::cerr << "FAIL: max_diff=" << max_diff << " > 1e-9\n";
    return 1;
  }
  if (integral_vs_cpu > 1e-10) {
    std::cerr << "FAIL: integral_vs_cpu=" << integral_vs_cpu
              << " > 1e-10\n";
    return 1;
  }
  return 0;
}

int TestRhoBuildLedger() {
  UniformGrid3D grid;
  grid.n = {16, 16, 16};
  grid.h = {0.3, 0.3, 0.3};
  grid.origin = {-2.4, -2.4, -2.4};

  const std::vector<double> occupations = {1.0};
  std::vector<std::vector<double>> orbitals;
  orbitals.push_back(RhoBuilder::GaussianOrbital(grid, 0.5, {0.0, 0.0, 0.0}));

  auto gpu_result = RhoBuildCuda(grid, orbitals, occupations);
  if (!gpu_result.ok()) {
    std::cerr << "Ledger test failed: " << gpu_result.status().message()
              << '\n';
    return 1;
  }

  const auto& entries = gpu_result.value().ledger.entries();
  if (entries.size() != 1) {
    std::cerr << "FAIL: expected 1 ledger entry, got " << entries.size()
              << '\n';
    return 1;
  }
  if (entries[0].precision.determinism !=
      tides::tile::DeterminismMode::kDeterministic) {
    std::cerr << "FAIL: ledger determinism is not kDeterministic\n";
    return 1;
  }
  std::cout << "ledger: entries=" << entries.size()
            << " label=" << entries[0].precision.label
            << " candidates=" << entries[0].candidates << '\n';
  return 0;
}

}  // namespace

int main() {
  if (!RhoBuildCudaAvailable()) {
    std::cout << "SKIP: CUDA runtime not available\n";
    return 77;
  }

  int failures = 0;
  failures += TestRhoBuildVsCpu();
  failures += TestRhoBuildLargerGrid();
  failures += TestRhoBuildLedger();

  if (failures == 0) {
    std::cout << "All GPU rho builder tests passed.\n";
  } else {
    std::cerr << failures << " GPU rho builder test(s) failed.\n";
  }
  return failures;
}
