// Collinear spin XC tests.
//
// Validates:
//   - Spin-polarized LDA energy matches unpolarized at zeta=0
//   - Fully polarized (rho_down=0) exchange energy > unpolarized
//   - Spin-polarized PBE runs and gives reasonable results
//   - Zeta computation is correct
//   - V_xc_up != V_xc_down for polarized densities

#include "grid/collinear_spin_xc.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::grid::BoundaryCondition;
using tides::grid::CollinearSpinXC;
using tides::grid::LibxcFunctional;
using tides::grid::SpinXCResult;
using tides::grid::UniformGrid3D;

int Fail(const std::string& msg) {
  std::cerr << "FAIL: " << msg << '\n';
  return 1;
}

// Gaussian density on grid.
std::vector<double> GaussianRho(const UniformGrid3D& grid, double alpha,
                                 std::array<double, 3> center) {
  std::vector<double> rho(grid.total_points(), 0.0);
  const double norm = std::pow(alpha / M_PI, 1.5);
  for (std::size_t i = 0; i < grid.total_points(); ++i) {
    const auto [ix, iy, iz] = grid.unflatten(i);
    const auto r = grid.coord(ix, iy, iz);
    const double dx = r[0] - center[0];
    const double dy = r[1] - center[1];
    const double dz = r[2] - center[2];
    rho[i] = norm * std::exp(-alpha * (dx * dx + dy * dy + dz * dz));
  }
  return rho;
}

// Test 1: Unpolarized limit — spin-polarized LDA with equal spins = unpolarized.
int TestUnpolarizedLimit() {
  std::cout << "\n=== Collinear spin XC: Unpolarized limit ===\n";
  UniformGrid3D grid;
  grid.n = {16, 16, 16};
  grid.h = {0.3, 0.3, 0.3};
  grid.origin = {-2.4, -2.4, -2.4};
  grid.bc = {BoundaryCondition::kFree, BoundaryCondition::kFree,
             BoundaryCondition::kFree};

  auto rho = GaussianRho(grid, 2.0, {0.0, 0.0, 0.0});
  std::vector<double> rho_half(rho.size());
  for (std::size_t i = 0; i < rho.size(); ++i)
    rho_half[i] = 0.5 * rho[i];

  auto spin_res = CollinearSpinXC::EvaluateLDA(grid, rho_half, rho_half);
  if (!spin_res.ok) return Fail("unpol limit: LDA eval failed");

  auto unpol = LibxcFunctional::EvalLDAOnGrid(rho);
  const auto [h0, h1, h2] = grid.h;
  const double dv = h0 * h1 * h2;
  double unpol_E = 0.0;
  for (std::size_t i = 0; i < rho.size(); ++i)
    unpol_E += unpol.eps_xc[i] * rho[i] * dv;

  double err = std::fabs(spin_res.xc_energy - unpol_E);
  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  spin_E=" << spin_res.xc_energy << " unpol_E=" << unpol_E
            << " |err|=" << err << '\n';
  if (err > 1e-10) return Fail("unpol limit: energy mismatch > 1e-10");

  std::cout << "PASS\n";
  return 0;
}

// Test 2: Fully polarized exchange energy > unpolarized.
int TestPolarizedExchange() {
  std::cout << "\n=== Collinear spin XC: Polarized > unpolarized exchange ===\n";
  UniformGrid3D grid;
  grid.n = {16, 16, 16};
  grid.h = {0.3, 0.3, 0.3};
  grid.origin = {-2.4, -2.4, -2.4};

  auto rho = GaussianRho(grid, 2.0, {0.0, 0.0, 0.0});

  // Unpolarized: rho_up = rho_down = rho/2
  std::vector<double> rho_half(rho.size());
  for (std::size_t i = 0; i < rho.size(); ++i) rho_half[i] = 0.5 * rho[i];
  auto unpol = CollinearSpinXC::EvaluateLDA(grid, rho_half, rho_half);

  // Fully polarized: rho_up = rho, rho_down = 0
  std::vector<double> rho_zero(rho.size(), 0.0);
  auto pol = CollinearSpinXC::EvaluateLDA(grid, rho, rho_zero);

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  unpol_E=" << unpol.xc_energy << " pol_E=" << pol.xc_energy
            << " diff=" << (pol.xc_energy - unpol.xc_energy) << '\n';

  // Fully polarized exchange energy should be more negative (stronger X).
  // The exchange scales as ((1+zeta)^(4/3) + (1-zeta)^(4/3)) / 2.
  // At zeta=1: factor = 2^(4/3)/2 ≈ 1.26 vs zeta=0: factor = 1.
  // So pol exchange should be ~26% more negative. Correlation is less affected.
  if (pol.xc_energy >= unpol.xc_energy)
    return Fail("polarized XC should be more negative than unpolarized");

  // Check the ratio is reasonable (exchange dominates at low density).
  double ratio = pol.xc_energy / unpol.xc_energy;
  std::cout << "  ratio=" << ratio << " (expected ~1.2-1.3)\n";
  if (ratio < 1.1 || ratio > 1.4)
    return Fail("polarized/unpolarized ratio out of expected range");

  std::cout << "PASS\n";
  return 0;
}

// Test 3: V_xc_up != V_xc_down for polarized density.
int TestSpinSplitPotential() {
  std::cout << "\n=== Collinear spin XC: Spin-split potential ===\n";
  UniformGrid3D grid;
  grid.n = {8, 8, 8};
  grid.h = {0.4, 0.4, 0.4};
  grid.origin = {-1.6, -1.6, -1.6};

  auto rho = GaussianRho(grid, 2.0, {0.0, 0.0, 0.0});

  // Polarized: rho_up = 0.7*rho, rho_down = 0.3*rho
  std::vector<double> rho_up(rho.size()), rho_down(rho.size());
  for (std::size_t i = 0; i < rho.size(); ++i) {
    rho_up[i] = 0.7 * rho[i];
    rho_down[i] = 0.3 * rho[i];
  }

  auto res = CollinearSpinXC::EvaluateLDA(grid, rho_up, rho_down);
  if (!res.ok) return Fail("spin split: LDA eval failed");

  // At the center (highest density), V_xc_up should differ from V_xc_down.
  std::size_t center = grid.flatten(4, 4, 4);
  double v_up = res.vxc_up[center];
  double v_down = res.vxc_down[center];
  double diff = std::fabs(v_up - v_down);
  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  V_xc_up=" << v_up << " V_xc_down=" << v_down
            << " |diff|=" << diff << '\n';

  if (diff < 1e-6) return Fail("spin split: V_xc_up == V_xc_down for polarized density");

  // At unpolarized (equal spins), V_xc_up should equal V_xc_down.
  std::vector<double> rho_half(rho.size());
  for (std::size_t i = 0; i < rho.size(); ++i) rho_half[i] = 0.5 * rho[i];
  auto res2 = CollinearSpinXC::EvaluateLDA(grid, rho_half, rho_half);
  double v_up2 = res2.vxc_up[center];
  double v_down2 = res2.vxc_down[center];
  double diff2 = std::fabs(v_up2 - v_down2);
  std::cout << "  unpol: V_xc_up=" << v_up2 << " V_xc_down=" << v_down2
            << " |diff|=" << diff2 << '\n';
  if (diff2 > 1e-10) return Fail("spin split: V_xc_up != V_xc_down at zeta=0");

  std::cout << "PASS\n";
  return 0;
}

// Test 4: Zeta computation.
int TestZeta() {
  std::cout << "\n=== Collinear spin XC: Zeta ===\n";
  std::vector<double> rho_up = {1.0, 0.5, 0.0, 2.0};
  std::vector<double> rho_down = {0.0, 0.5, 1.0, 2.0};

  auto zeta = CollinearSpinXC::ComputeZeta(rho_up, rho_down);

  // Expected: 1.0, 0.0, -1.0, 0.0
  if (std::fabs(zeta[0] - 1.0) > 1e-14) return Fail("zeta[0] should be 1.0");
  if (std::fabs(zeta[1] - 0.0) > 1e-14) return Fail("zeta[1] should be 0.0");
  if (std::fabs(zeta[2] - (-1.0)) > 1e-14) return Fail("zeta[2] should be -1.0");
  if (std::fabs(zeta[3] - 0.0) > 1e-14) return Fail("zeta[3] should be 0.0");

  std::cout << "  zeta = {1.0, 0.0, -1.0, 0.0} correct\n";
  std::cout << "PASS\n";
  return 0;
}

// Test 5: Spin-polarized PBE runs.
int TestPBE() {
  std::cout << "\n=== Collinear spin XC: PBE ===\n";
  UniformGrid3D grid;
  grid.n = {8, 8, 8};
  grid.h = {0.4, 0.4, 0.4};
  grid.origin = {-1.6, -1.6, -1.6};

  auto rho = GaussianRho(grid, 2.0, {0.0, 0.0, 0.0});

  std::vector<double> rho_up(rho.size()), rho_down(rho.size());
  for (std::size_t i = 0; i < rho.size(); ++i) {
    rho_up[i] = 0.6 * rho[i];
    rho_down[i] = 0.4 * rho[i];
  }

  auto res = CollinearSpinXC::EvaluatePBE(grid, rho_up, rho_down);
  if (!res.ok) return Fail("PBE: eval failed");

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  PBE spin E=" << res.xc_energy << '\n';
  if (res.xc_energy >= 0.0) return Fail("PBE: energy should be negative");
  if (res.vxc_up.size() != grid.total_points())
    return Fail("PBE: wrong vxc_up size");

  // Check potentials are negative (attractive) near center.
  std::size_t center = grid.flatten(4, 4, 4);
  if (res.vxc_up[center] >= 0.0) return Fail("PBE: V_xc_up should be negative");
  if (res.vxc_down[center] >= 0.0) return Fail("PBE: V_xc_down should be negative");

  std::cout << "PASS\n";
  return 0;
}

}  // namespace

int main() {
  if (TestUnpolarizedLimit()) return 1;
  if (TestPolarizedExchange()) return 1;
  if (TestSpinSplitPotential()) return 1;
  if (TestZeta()) return 1;
  if (TestPBE()) return 1;
  std::cout << "\ncollinear_spin_xc_tests: ALL GREEN\n";
  return 0;
}
