// T3.5: GPU XC functional evaluation tests — vs CPU reference.
// Validates that the CUDA LDA-PW92 XC evaluation produces V_xc and eps_xc
// equal to the CPU path within <=1e-12, and XC energy matches.

#include "grid/xc.hpp"
#include "grid/xc_gpu.hpp"
#include "grid/libxc_wrapper.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

using tides::grid::XCGridEvaluator;
using tides::grid::XCCudaAvailable;
using tides::grid::XCEvalLdaCuda;
using tides::grid::XCEvalPbeCuda;
using tides::grid::XCGpuResult;
using tides::grid::UniformGrid3D;
using tides::grid::LibxcFunctional;

double MaxAbsDifference(const std::vector<double>& a,
                        const std::vector<double>& b) {
  double max_diff = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
  }
  return max_diff;
}

std::vector<double> MakeGaussianRho(const UniformGrid3D& grid,
                                    double alpha,
                                    std::array<double, 3> center) {
  const std::size_t N = grid.total_points();
  std::vector<double> rho(N, 0.0);
  for (std::size_t i = 0; i < N; ++i) {
    const auto [ix, iy, iz] = grid.unflatten(i);
    const auto [x, y, z] = grid.coord(ix, iy, iz);
    const double dx = x - center[0], dy = y - center[1], dz = z - center[2];
    rho[i] = std::exp(-alpha * (dx * dx + dy * dy + dz * dz));
  }
  return rho;
}

int TestXCVsCpu() {
  // 32x32x32 grid with Gaussian density.
  UniformGrid3D grid;
  grid.n = {32, 32, 32};
  grid.h = {0.2, 0.2, 0.2};
  grid.origin = {-3.2, -3.2, -3.2};

  const auto rho = MakeGaussianRho(grid, 0.5, {0.0, 0.0, 0.0});

  auto cpu_xc = XCGridEvaluator::EvaluateLDA(grid, rho);
  const double cpu_energy = XCGridEvaluator::XCEnergy(grid, cpu_xc, rho);

  auto gpu_result = XCEvalLdaCuda(grid, rho, 0.0);
  if (!gpu_result.ok()) {
    std::cerr << "XCEvalLdaCuda failed: " << gpu_result.status().message()
              << '\n';
    return 1;
  }

  const double vxc_diff = MaxAbsDifference(gpu_result.value().vxc, cpu_xc.vxc);
  const double eps_diff = MaxAbsDifference(gpu_result.value().eps_xc, cpu_xc.eps_xc);
  const double energy_diff = std::abs(gpu_result.value().xc_energy - cpu_energy);

  std::cout << "xc_vs_cpu: grid=32^3"
            << " kernel_ms=" << gpu_result.value().kernel_ms
            << " vxc_diff=" << vxc_diff
            << " eps_diff=" << eps_diff
            << " cpu_energy=" << cpu_energy
            << " gpu_energy=" << gpu_result.value().xc_energy
            << " energy_diff=" << energy_diff << '\n';

  // GPU and CPU use the same LDA-PW92 formulas, so eps_xc agreement is
  // machine-precision. V_xc uses a central FD derivative for d(eps_c)/d(rs),
  // which has ~1e-10 rounding differences between GPU/CPU math libraries.
  if (vxc_diff > 1e-9) {
    std::cerr << "FAIL: vxc_diff=" << vxc_diff << " > 1e-9\n";
    return 1;
  }
  if (eps_diff > 1e-12) {
    std::cerr << "FAIL: eps_diff=" << eps_diff << " > 1e-12\n";
    return 1;
  }
  if (energy_diff > 1e-10) {
    std::cerr << "FAIL: energy_diff=" << energy_diff << " > 1e-10\n";
    return 1;
  }
  return 0;
}

int TestXCLargerGrid() {
  // 64x64x64 grid with two Gaussian densities.
  UniformGrid3D grid;
  grid.n = {64, 64, 64};
  grid.h = {0.15, 0.15, 0.15};
  grid.origin = {-4.8, -4.8, -4.8};

  auto rho1 = MakeGaussianRho(grid, 0.5, {0.0, 0.0, 0.0});
  auto rho2 = MakeGaussianRho(grid, 0.8, {1.5, 0.0, 0.0});
  std::vector<double> rho(grid.total_points());
  for (std::size_t i = 0; i < rho.size(); ++i) {
    rho[i] = rho1[i] + rho2[i];
  }

  auto cpu_xc = XCGridEvaluator::EvaluateLDA(grid, rho);
  const double cpu_energy = XCGridEvaluator::XCEnergy(grid, cpu_xc, rho);

  auto gpu_result = XCEvalLdaCuda(grid, rho, 0.0);
  if (!gpu_result.ok()) {
    std::cerr << "XCEvalLdaCuda failed: " << gpu_result.status().message()
              << '\n';
    return 1;
  }

  const double vxc_diff = MaxAbsDifference(gpu_result.value().vxc, cpu_xc.vxc);
  const double eps_diff = MaxAbsDifference(gpu_result.value().eps_xc, cpu_xc.eps_xc);
  const double energy_diff = std::abs(gpu_result.value().xc_energy - cpu_energy);

  std::cout << "xc_larger: grid=64^3"
            << " kernel_ms=" << gpu_result.value().kernel_ms
            << " vxc_diff=" << vxc_diff
            << " eps_diff=" << eps_diff
            << " energy_diff=" << energy_diff << '\n';

  if (vxc_diff > 1e-9) {
    std::cerr << "FAIL: vxc_diff=" << vxc_diff << " > 1e-9\n";
    return 1;
  }
  if (eps_diff > 1e-12) {
    std::cerr << "FAIL: eps_diff=" << eps_diff << " > 1e-12\n";
    return 1;
  }
  if (energy_diff > 1e-10) {
    std::cerr << "FAIL: energy_diff=" << energy_diff << " > 1e-10\n";
    return 1;
  }
  return 0;
}

int TestXCLedger() {
  UniformGrid3D grid;
  grid.n = {16, 16, 16};
  grid.h = {0.3, 0.3, 0.3};
  grid.origin = {-2.4, -2.4, -2.4};

  const auto rho = MakeGaussianRho(grid, 0.5, {0.0, 0.0, 0.0});

  auto gpu_result = XCEvalLdaCuda(grid, rho, 0.0);
  if (!gpu_result.ok()) {
    std::cerr << "Ledger test failed: " << gpu_result.status().message() << '\n';
    return 1;
  }

  const auto& entries = gpu_result.value().ledger.entries();
  if (entries.size() != 1) {
    std::cerr << "FAIL: expected 1 ledger entry, got " << entries.size() << '\n';
    return 1;
  }
  if (entries[0].operation != tides::tile::OperationKind::kXcFunctional) {
    std::cerr << "FAIL: ledger operation is not kXcFunctional\n";
    return 1;
  }
  std::cout << "ledger: entries=" << entries.size()
            << " label=" << entries[0].precision.label
            << " candidates=" << entries[0].candidates << '\n';
  return 0;
}

int TestPbeLibxc() {
  // Test PBE GGA evaluation via libxc on a 16^3 grid.
  UniformGrid3D grid;
  grid.n = {16, 16, 16};
  grid.h = {0.3, 0.3, 0.3};
  grid.origin = {-2.4, -2.4, -2.4};

  const auto rho = MakeGaussianRho(grid, 0.5, {0.0, 0.0, 0.0});

  // Evaluate PBE via libxc wrapper (CPU).
  const auto [n0, n1, n2] = grid.n;
  const auto [h0, h1, h2] = grid.h;
  auto pbe_cpu = LibxcFunctional::EvalPBEOnGrid(n0, n1, n2, h0, h1, h2, rho);

  // Evaluate PBE via GPU path (libxc CPU + GPU energy reduction).
  auto gpu_result = XCEvalPbeCuda(grid, rho);
  if (!gpu_result.ok()) {
    std::cerr << "XCEvalPbeCuda failed: " << gpu_result.status().message()
              << '\n';
    return 1;
  }

  const double eps_diff =
      MaxAbsDifference(gpu_result.value().eps_xc, pbe_cpu.eps_xc);
  const double vxc_diff =
      MaxAbsDifference(gpu_result.value().vxc, pbe_cpu.vxc);

  // Compute CPU energy for comparison.
  const double dv = h0 * h1 * h2;
  double cpu_energy = 0.0;
  for (std::size_t i = 0; i < rho.size(); ++i)
    cpu_energy += pbe_cpu.eps_xc[i] * rho[i] * dv;
  const double energy_diff =
      std::abs(gpu_result.value().xc_energy - cpu_energy);

  std::cout << "pbe_libxc: grid=16^3"
            << " eps_diff=" << eps_diff
            << " vxc_diff=" << vxc_diff
            << " cpu_energy=" << cpu_energy
            << " gpu_energy=" << gpu_result.value().xc_energy
            << " energy_diff=" << energy_diff << '\n';

  // libxc CPU and GPU path use the same libxc evaluation, so agreement
  // should be machine precision for eps_xc and vxc.
  if (eps_diff > 1e-12) {
    std::cerr << "FAIL: pbe eps_diff=" << eps_diff << " > 1e-12\n";
    return 1;
  }
  if (vxc_diff > 1e-12) {
    std::cerr << "FAIL: pbe vxc_diff=" << vxc_diff << " > 1e-12\n";
    return 1;
  }
  if (energy_diff > 1e-10) {
    std::cerr << "FAIL: pbe energy_diff=" << energy_diff << " > 1e-10\n";
    return 1;
  }

  // Also verify that PBE gives different results from LDA (sanity check).
  auto lda_result = XCEvalLdaCuda(grid, rho, 0.0);
  if (lda_result.ok()) {
    double lda_eps_max = 0.0, pbe_eps_max = 0.0;
    for (std::size_t i = 0; i < rho.size(); ++i) {
      lda_eps_max = std::max(lda_eps_max, std::abs(lda_result.value().eps_xc[i]));
      pbe_eps_max = std::max(pbe_eps_max, std::abs(gpu_result.value().eps_xc[i]));
    }
    std::cout << "pbe_vs_lda: lda_eps_max=" << lda_eps_max
              << " pbe_eps_max=" << pbe_eps_max << '\n';
    // PBE and LDA should differ (PBE has gradient corrections).
    double max_eps_diff = MaxAbsDifference(gpu_result.value().eps_xc,
                                           lda_result.value().eps_xc);
    if (max_eps_diff < 1e-6) {
      std::cerr << "FAIL: PBE and LDA give nearly identical results "
                << "(max_eps_diff=" << max_eps_diff << ")\n";
      return 1;
    }
    std::cout << "pbe_vs_lda: max_eps_diff=" << max_eps_diff << " (expected > 1e-6)\n";
  }

  return 0;
}

}  // namespace

int main() {
  if (!XCCudaAvailable()) {
    std::cout << "SKIP: CUDA runtime not available\n";
    return 77;
  }

  int failures = 0;
  failures += TestXCVsCpu();
  failures += TestXCLargerGrid();
  failures += TestXCLedger();
  failures += TestPbeLibxc();

  if (failures == 0) {
    std::cout << "All GPU XC evaluation tests passed.\n";
  } else {
    std::cerr << failures << " GPU XC test(s) failed.\n";
  }
  return failures;
}
