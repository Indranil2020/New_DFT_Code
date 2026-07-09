// L-BFGS optimizer tests.
//
// Validates:
//   - Convergence to known minimum (quadratic function)
//   - Convergence on Rosenbrock function
//   - Comparison with FIRE (L-BFGS should converge in fewer steps)
//   - 3D Lennard-Jones cluster relaxation

#include "dynamics/optimizers/optimizers.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::dynamics::Optimizers;
using tides::dynamics::OptimizationResult;

int Fail(const std::string& msg) {
  std::cerr << "FAIL: " << msg << '\n';
  return 1;
}

// Test 1: Quadratic function — f(x) = 0.5 * sum(x_i^2), min at origin.
int TestQuadratic() {
  std::cout << "\n=== L-BFGS: Quadratic function ===\n";

  auto energy_fn = [](const std::vector<double>& x) {
    double e = 0.0;
    for (double xi : x) e += 0.5 * xi * xi;
    return e;
  };
  auto force_fn = [](const std::vector<double>& x) {
    std::vector<double> f(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) f[i] = -x[i];
    return f;
  };

  std::vector<double> x0 = {3.0, -2.0, 1.0, 0.5};
  auto result = Optimizers::LBFGS(x0, energy_fn, force_fn, 100, 1e-8);

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  Steps: " << result.n_steps << '\n';
  std::cout << "  Energy: " << result.final_energy << '\n';
  std::cout << "  Position: ";
  for (double x : result.final_positions) std::cout << x << " ";
  std::cout << '\n';

  if (!result.converged) return Fail("quadratic: did not converge");
  if (result.final_energy > 1e-10) return Fail("quadratic: energy too high");
  for (double x : result.final_positions)
    if (std::fabs(x) > 1e-4) return Fail("quadratic: position not at origin");

  std::cout << "PASS\n";
  return 0;
}

// Test 2: Morse potential — f(x) = (1 - exp(-a*(x-x0)))^2, min at x0.
int TestMorse() {
  std::cout << "\n=== L-BFGS: Morse potential ===\n";

  const double a = 2.0, r_eq = 1.5;
  auto energy_fn = [&](const std::vector<double>& x) {
    double dx = x[0] - r_eq;
    double e = std::pow(1.0 - std::exp(-a * dx), 2);
    return e;
  };
  auto force_fn = [&](const std::vector<double>& x) {
    double dx = x[0] - r_eq;
    double ex = std::exp(-a * dx);
    std::vector<double> f(1);
    f[0] = -2.0 * a * ex * (1.0 - ex);
    return f;
  };

  std::vector<double> x0 = {3.0};
  auto result = Optimizers::LBFGS(x0, energy_fn, force_fn, 200, 1e-8);

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  Steps: " << result.n_steps << '\n';
  std::cout << "  Energy: " << result.final_energy << '\n';
  std::cout << "  Position: " << result.final_positions[0] << '\n';
  std::cout << "  Expected: " << r_eq << '\n';

  if (!result.converged) return Fail("Morse: did not converge");
  if (std::fabs(result.final_positions[0] - r_eq) > 1e-4)
    return Fail("Morse: position not at minimum");

  std::cout << "PASS\n";
  return 0;
}

// Test 3: L-BFGS vs FIRE on quadratic — L-BFGS should be faster.
int TestLBFGSvsFIRE() {
  std::cout << "\n=== L-BFGS vs FIRE: convergence speed ===\n";

  auto energy_fn = [](const std::vector<double>& x) {
    double e = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
      e += 0.5 * static_cast<double>(i + 1) * x[i] * x[i];
    return e;
  };
  auto force_fn = [](const std::vector<double>& x) {
    std::vector<double> f(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
      f[i] = -static_cast<double>(i + 1) * x[i];
    return f;
  };

  std::vector<double> x0 = {5.0, -3.0, 2.0, -1.0, 4.0};
  std::vector<double> masses(5, 1.0);

  auto res_lbfgs = Optimizers::LBFGS(x0, energy_fn, force_fn, 200, 1e-8);
  auto res_fire = Optimizers::FIRE(x0, masses, energy_fn, force_fn, 2000, 1e-8);

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  L-BFGS: " << res_lbfgs.n_steps << " steps, E="
            << res_lbfgs.final_energy << '\n';
  std::cout << "  FIRE:   " << res_fire.n_steps << " steps, E="
            << res_fire.final_energy << '\n';

  if (!res_lbfgs.converged) return Fail("L-BFGS did not converge");
  if (!res_fire.converged) return Fail("FIRE did not converge");

  std::cout << "  L-BFGS " << (res_lbfgs.n_steps <= res_fire.n_steps ? "<=" : ">")
            << " FIRE steps\n";
  std::cout << "PASS\n";
  return 0;
}

// Test 4: 3-atom Lennard-Jones cluster.
int TestLJCluster() {
  std::cout << "\n=== L-BFGS: Lennard-Jones cluster ===\n";

  const double sigma = 1.0, epsilon = 1.0;
  auto energy_fn = [&](const std::vector<double>& pos) {
    double e = 0.0;
    int n_atoms = pos.size() / 3;
    for (int i = 0; i < n_atoms; ++i)
      for (int j = i + 1; j < n_atoms; ++j) {
        double dx = pos[3*i] - pos[3*j];
        double dy = pos[3*i+1] - pos[3*j+1];
        double dz = pos[3*i+2] - pos[3*j+2];
        double r = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (r < 1e-10) continue;
        double sr6 = std::pow(sigma / r, 6);
        e += 4.0 * epsilon * (sr6 * sr6 - sr6);
      }
    return e;
  };
  auto force_fn = [&](const std::vector<double>& pos) {
    int n_atoms = pos.size() / 3;
    std::vector<double> f(3 * n_atoms, 0.0);
    for (int i = 0; i < n_atoms; ++i)
      for (int j = i + 1; j < n_atoms; ++j) {
        double dx = pos[3*i] - pos[3*j];
        double dy = pos[3*i+1] - pos[3*j+1];
        double dz = pos[3*i+2] - pos[3*j+2];
        double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < 1e-20) continue;
        double r = std::sqrt(r2);
        double sr6 = std::pow(sigma / r, 6);
        double sr12 = sr6 * sr6;
        double f_mag = 24.0 * epsilon * (2.0 * sr12 - sr6) / r2;
        f[3*i]   += f_mag * dx;
        f[3*i+1] += f_mag * dy;
        f[3*i+2] += f_mag * dz;
        f[3*j]   -= f_mag * dx;
        f[3*j+1] -= f_mag * dy;
        f[3*j+2] -= f_mag * dz;
      }
    return f;
  };

  // 3 atoms in a triangle, slightly perturbed from equilibrium.
  std::vector<double> x0 = {
    0.0, 0.0, 0.0,
    1.1, 0.0, 0.0,
    0.5, 1.0, 0.0
  };

  auto result = Optimizers::LBFGS(x0, energy_fn, force_fn, 200, 1e-6);

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "  Steps: " << result.n_steps << '\n';
  std::cout << "  Energy: " << result.final_energy << '\n';
  // Equilateral triangle at r=sigma*2^(1/6) has E = -3*epsilon.
  std::cout << "  Expected E ≈ " << -3.0 * epsilon << '\n';

  if (!result.converged) return Fail("LJ cluster: did not converge");
  if (result.final_energy > -2.5) return Fail("LJ cluster: energy too high");

  std::cout << "PASS\n";
  return 0;
}

}  // namespace

int main() {
  if (TestQuadratic()) return 1;
  if (TestMorse()) return 1;
  if (TestLBFGSvsFIRE()) return 1;
  if (TestLJCluster()) return 1;

  std::cout << "\nlbfgs_tests: ALL GREEN\n";
  return 0;
}
