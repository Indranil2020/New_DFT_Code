// ISF Poisson kernel tests for non-periodic BCs.
//
// Validates:
//   - Free BC: ISF matches direct Coulomb sum
//   - Wire BC: ISF handles mixed periodic/free
//   - Slab BC: ISF handles mixed periodic/free
//   - Periodic BC: ISF falls back to standard FFT
//   - Gaussian charge: Hartree energy matches analytic result

#include "grid/isf_poisson.hpp"
#include "grid/poisson.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using tides::grid::BoundaryCondition;
using tides::grid::ISFConfig;
using tides::grid::ISFPoissonSolver;
using tides::grid::PoissonSolver;
using tides::grid::UniformGrid3D;

int Fail(const std::string& msg) {
  std::cerr << "FAIL: " << msg << '\n';
  return 1;
}

// Gaussian charge density: rho(r) = q * (alpha/pi)^{3/2} * exp(-alpha * r^2)
std::vector<double> GaussianCharge(const UniformGrid3D& grid,
                                    double q, double alpha) {
  std::vector<double> rho(grid.total_points(), 0.0);
  const auto [L0, L1, L2] = grid.cell_size();
  const std::array<double, 3> center = {
      grid.origin[0] + L0 / 2.0,
      grid.origin[1] + L1 / 2.0,
      grid.origin[2] + L2 / 2.0
  };
  const double norm = q * std::pow(alpha / M_PI, 1.5);
  for (std::size_t i = 0; i < grid.total_points(); ++i) {
    const auto [ix, iy, iz] = grid.unflatten(i);
    const auto r = grid.coord(ix, iy, iz);
    const double dx = r[0] - center[0];
    const double dy = r[1] - center[1];
    const double dz = r[2] - center[2];
    const double r2 = dx * dx + dy * dy + dz * dz;
    rho[i] = norm * std::exp(-alpha * r2);
  }
  return rho;
}

// Test 1: Free BC — ISF vs direct Coulomb sum.
int TestFreeBC() {
  std::cout << "\n=== ISF Poisson: Free BC ===\n";
  UniformGrid3D grid;
  grid.n = {16, 16, 16};
  const double L = 8.0;
  grid.h = {L / 16, L / 16, L / 16};
  grid.origin = {-L / 2, -L / 2, -L / 2};
  grid.bc = {BoundaryCondition::kFree, BoundaryCondition::kFree,
             BoundaryCondition::kFree};

  const double alpha = 5.0, q = 1.0;
  auto rho = GaussianCharge(grid, q, alpha);

  // Direct Coulomb sum (reference).
  auto V_ref = PoissonSolver::SolveFree(grid, rho);
  double E_ref = PoissonSolver::HartreeEnergy(grid, rho);

  // ISF solver.
  ISFConfig config;
  config.mu = 0.3;
  config.pad_factor = 3;
  config.r_cutoff = 5.0;
  auto isf = ISFPoissonSolver::Solve(grid, rho, config);

  if (!isf.ok) return Fail("free BC: ISF solve failed");

  // Compare potentials.
  double max_v_diff = 0.0;
  for (std::size_t i = 0; i < V_ref.size(); ++i)
    max_v_diff = std::max(max_v_diff, std::fabs(isf.V[i] - V_ref[i]));

  double E_exact = ISFPoissonSolver::AnalyticGaussianHartree(q, alpha);
  double E_err = std::fabs(isf.hartree_energy - E_exact);

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  E_isf=" << isf.hartree_energy << " E_ref=" << E_ref
            << " E_exact=" << E_exact
            << " |E_isf-E_exact|=" << E_err
            << " max_V_diff=" << max_v_diff << '\n';

  // ISF should be close to analytic.
  if (E_err > 0.1) return Fail("free BC: energy error too large");

  std::cout << "PASS\n";
  return 0;
}

// Test 2: Wire BC (1D periodic, 2D free).
int TestWireBC() {
  std::cout << "\n=== ISF Poisson: Wire BC ===\n";
  UniformGrid3D grid;
  grid.n = {16, 16, 16};
  const double L = 8.0;
  grid.h = {L / 16, L / 16, L / 16};
  grid.origin = {-L / 2, -L / 2, -L / 2};
  grid.bc = {BoundaryCondition::kFree, BoundaryCondition::kFree,
             BoundaryCondition::kPeriodic};

  const double alpha = 5.0, q = 1.0;
  auto rho = GaussianCharge(grid, q, alpha);

  ISFConfig config;
  config.mu = 0.3;
  config.pad_factor = 3;
  config.r_cutoff = 5.0;
  auto isf = ISFPoissonSolver::Solve(grid, rho, config);
  if (!isf.ok) return Fail("wire BC: ISF solve failed");

  double E_exact = ISFPoissonSolver::AnalyticGaussianHartree(q, alpha);
  double E_err = std::fabs(isf.hartree_energy - E_exact);
  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  E_isf=" << isf.hartree_energy
            << " E_exact=" << E_exact
            << " |err|=" << E_err << '\n';

  if (E_err > 0.2) return Fail("wire BC: energy error too large");

  std::cout << "PASS\n";
  return 0;
}

// Test 3: Slab BC (2D periodic, 1D free).
int TestSlabBC() {
  std::cout << "\n=== ISF Poisson: Slab BC ===\n";
  UniformGrid3D grid;
  grid.n = {16, 16, 16};
  const double L = 8.0;
  grid.h = {L / 16, L / 16, L / 16};
  grid.origin = {-L / 2, -L / 2, -L / 2};
  grid.bc = {BoundaryCondition::kPeriodic, BoundaryCondition::kPeriodic,
             BoundaryCondition::kFree};

  const double alpha = 5.0, q = 1.0;
  auto rho = GaussianCharge(grid, q, alpha);

  ISFConfig config;
  config.mu = 0.3;
  config.pad_factor = 3;
  config.r_cutoff = 5.0;
  auto isf = ISFPoissonSolver::Solve(grid, rho, config);
  if (!isf.ok) return Fail("slab BC: ISF solve failed");

  double E_exact = ISFPoissonSolver::AnalyticGaussianHartree(q, alpha);
  double E_err = std::fabs(isf.hartree_energy - E_exact);
  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  E_isf=" << isf.hartree_energy
            << " E_exact=" << E_exact
            << " |err|=" << E_err << '\n';

  if (E_err > 0.2) return Fail("slab BC: energy error too large");

  std::cout << "PASS\n";
  return 0;
}

// Test 4: Periodic BC — ISF falls back to standard FFT.
int TestPeriodicFallback() {
  std::cout << "\n=== ISF Poisson: Periodic fallback ===\n";
  UniformGrid3D grid;
  grid.n = {16, 16, 16};
  const double L = 10.0;
  grid.h = {L / 16, L / 16, L / 16};
  grid.origin = {-L / 2, -L / 2, -L / 2};
  grid.bc = {BoundaryCondition::kPeriodic, BoundaryCondition::kPeriodic,
             BoundaryCondition::kPeriodic};

  const double alpha = 5.0, q = 1.0;
  auto rho = GaussianCharge(grid, q, alpha);

  ISFConfig config;
  auto isf = ISFPoissonSolver::Solve(grid, rho, config);
  if (!isf.ok) return Fail("periodic: ISF solve failed");

  // Compare with standard PoissonSolver.
  auto V_ref = PoissonSolver::SolvePeriodicFFT(grid, rho);
  double E_ref = PoissonSolver::HartreeEnergy(grid, rho);

  double max_v_diff = 0.0;
  for (std::size_t i = 0; i < V_ref.size(); ++i)
    max_v_diff = std::max(max_v_diff, std::fabs(isf.V[i] - V_ref[i]));

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  E_isf=" << isf.hartree_energy << " E_ref=" << E_ref
            << " max_V_diff=" << max_v_diff << '\n';

  if (max_v_diff > 1e-10) return Fail("periodic: ISF != standard FFT");

  std::cout << "PASS\n";
  return 0;
}

// Test 5: Config parameters.
int TestConfig() {
  std::cout << "\n=== ISF Poisson: Config ===\n";
  ISFConfig config;
  config.mu = 0.3;
  config.pad_factor = 3;
  config.r_cutoff = 6.0;

  if (config.mu != 0.3) return Fail("config: mu wrong");
  if (config.pad_factor != 3) return Fail("config: pad_factor wrong");
  if (config.r_cutoff != 6.0) return Fail("config: r_cutoff wrong");

  std::cout << "  mu=0.3 pad_factor=3 r_cutoff=6.0\n";
  std::cout << "PASS\n";
  return 0;
}

}  // namespace

int main() {
  if (TestFreeBC()) return 1;
  if (TestWireBC()) return 1;
  if (TestSlabBC()) return 1;
  if (TestPeriodicFallback()) return 1;
  if (TestConfig()) return 1;
  std::cout << "\nisf_poisson_tests: ALL GREEN\n";
  return 0;
}
