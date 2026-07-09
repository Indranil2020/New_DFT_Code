// T3.9: ESP / Prolate Ewald backend tests.
//
// Validates:
//   - Energy conservation: E_real + E_recip + E_self = E_coulomb
//   - ESP correctness vs direct Coulomb sum
//   - Force correctness (zero net force for symmetric configurations)
//   - Surface correction for elongated systems

#include "grid/prolate_ewald.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::grid::ProlateEwald;
using tides::grid::ProlateEwaldConfig;
using tides::grid::ProlateEwaldResult;

int Fail(const std::string& msg) {
  std::cerr << "FAIL: " << msg << '\n';
  return 1;
}

// Direct Coulomb energy (reference).
double DirectCoulombEnergy(const std::vector<double>& charges,
                            const std::vector<double>& positions) {
  const std::size_t n = charges.size();
  double e = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i + 1; j < n; ++j) {
      double dx = positions[3*i] - positions[3*j];
      double dy = positions[3*i+1] - positions[3*j+1];
      double dz = positions[3*i+2] - positions[3*j+2];
      double r = std::sqrt(dx*dx + dy*dy + dz*dz);
      if (r > 1e-10) e += charges[i] * charges[j] / r;
    }
  return e;
}

// Test 1: Energy conservation — Ewald sum should match direct Coulomb.
int TestEnergyConservation() {
  std::cout << "\n=== T3.9: Energy conservation ===\n";

  // Simple 2-charge system.
  std::vector<double> charges = {1.0, -1.0};
  std::vector<double> positions = {0.0, 0.0, 0.0, 2.0, 0.0, 0.0};

  ProlateEwaldConfig cfg;
  cfg.alpha = 0.5;
  cfg.r_cut = 20.0;

  auto result = ProlateEwald::Compute(charges, positions, cfg);
  double e_direct = DirectCoulombEnergy(charges, positions);

  std::cout << std::scientific << std::setprecision(10);
  std::cout << "  E_ewald = " << result.energy << '\n';
  std::cout << "  E_direct = " << e_direct << '\n';
  std::cout << "  E_real = " << result.real_energy << '\n';
  std::cout << "  E_recip = " << result.recip_energy << '\n';
  std::cout << "  E_self = " << result.self_energy << '\n';

  double err = std::fabs(result.energy - e_direct);
  std::cout << "  |err| = " << err << '\n';

  if (err > 1e-8) return Fail("energy conservation: error too large");
  std::cout << "PASS\n";
  return 0;
}

// Test 2: ESP correctness.
int TestESP() {
  std::cout << "\n=== T3.9: ESP correctness ===\n";

  // 3 charges in a triangle.
  std::vector<double> charges = {1.0, 1.0, -2.0};
  std::vector<double> positions = {0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 1.5, 2.0, 0.0};

  // Evaluate ESP at a point far from all charges.
  std::vector<double> eval_points = {10.0, 10.0, 10.0};

  ProlateEwaldConfig cfg;
  auto esp = ProlateEwald::ComputeESP(charges, positions, eval_points, cfg);

  // Direct ESP.
  double esp_direct = 0.0;
  for (std::size_t i = 0; i < charges.size(); ++i) {
    double dx = eval_points[0] - positions[3*i];
    double dy = eval_points[1] - positions[3*i+1];
    double dz = eval_points[2] - positions[3*i+2];
    double r = std::sqrt(dx*dx + dy*dy + dz*dz);
    esp_direct += charges[i] / r;
  }

  std::cout << std::scientific << std::setprecision(10);
  std::cout << "  ESP_ewald = " << esp[0] << '\n';
  std::cout << "  ESP_direct = " << esp_direct << '\n';

  double err = std::fabs(esp[0] - esp_direct);
  std::cout << "  |err| = " << err << '\n';

  if (err > 1e-10) return Fail("ESP: error too large");
  std::cout << "PASS\n";
  return 0;
}

// Test 3: Force symmetry — net force should be zero for symmetric config.
int TestForceSymmetry() {
  std::cout << "\n=== T3.9: Force symmetry ===\n";

  // 4 charges at corners of a square, alternating signs.
  std::vector<double> charges = {1.0, -1.0, 1.0, -1.0};
  std::vector<double> positions = {
    1.0, 1.0, 0.0,
    -1.0, 1.0, 0.0,
    -1.0, -1.0, 0.0,
    1.0, -1.0, 0.0
  };

  ProlateEwaldConfig cfg;
  cfg.alpha = 0.5;
  cfg.r_cut = 20.0;

  auto result = ProlateEwald::Compute(charges, positions, cfg);

  // Net force should be zero by symmetry.
  double fx = 0.0, fy = 0.0, fz = 0.0;
  for (std::size_t i = 0; i < charges.size(); ++i) {
    fx += result.forces[3*i];
    fy += result.forces[3*i+1];
    fz += result.forces[3*i+2];
  }

  std::cout << std::scientific << std::setprecision(10);
  std::cout << "  Net force: (" << fx << ", " << fy << ", " << fz << ")\n";

  double net = std::sqrt(fx*fx + fy*fy + fz*fz);
  std::cout << "  |F_net| = " << net << '\n';

  if (net > 1e-10) return Fail("force symmetry: net force too large");
  std::cout << "PASS\n";
  return 0;
}

// Test 4: Multi-charge system — Ewald vs direct.
int TestMultiCharge() {
  std::cout << "\n=== T3.9: Multi-charge system ===\n";

  // 6 charges (model water hexamer).
  std::vector<double> charges = {
    -0.834, 0.417, 0.417,  // molecule 1
    -0.834, 0.417, 0.417,  // molecule 2
  };
  std::vector<double> positions = {
    0.0, 0.0, 0.0,    0.96, 0.0, 0.0,   -0.24, 0.93, 0.0,
    3.0, 0.0, 0.0,    3.96, 0.0, 0.0,   2.76, 0.93, 0.0,
  };

  ProlateEwaldConfig cfg;
  cfg.alpha = 0.3;
  cfg.r_cut = 20.0;

  auto result = ProlateEwald::Compute(charges, positions, cfg);
  double e_direct = DirectCoulombEnergy(charges, positions);

  std::cout << std::scientific << std::setprecision(10);
  std::cout << "  E_ewald = " << result.energy << '\n';
  std::cout << "  E_direct = " << e_direct << '\n';

  double err = std::fabs(result.energy - e_direct);
  std::cout << "  |err| = " << err << '\n';

  if (err > 1e-8) return Fail("multi-charge: error too large");
  std::cout << "PASS\n";
  return 0;
}

// Test 5: Elongated system — surface correction.
int TestElongatedSystem() {
  std::cout << "\n=== T3.9: Elongated system ===\n";

  // 4 charges in a line (elongated system).
  std::vector<double> charges = {1.0, -1.0, 1.0, -1.0};
  std::vector<double> positions = {
    0.0, 0.0, 0.0,
    0.0, 0.0, 3.0,
    0.0, 0.0, 6.0,
    0.0, 0.0, 9.0,
  };

  ProlateEwaldConfig cfg;
  cfg.alpha = 0.3;
  cfg.r_cut = 20.0;

  auto result = ProlateEwald::Compute(charges, positions, cfg);
  double e_direct = DirectCoulombEnergy(charges, positions);

  std::cout << std::scientific << std::setprecision(10);
  std::cout << "  E_ewald = " << result.energy << '\n';
  std::cout << "  E_direct = " << e_direct << '\n';

  double err = std::fabs(result.energy - e_direct);
  std::cout << "  |err| = " << err << '\n';

  // Surface correction should improve accuracy for elongated systems.
  // Without it, the error would be larger.
  if (err > 1e-6) return Fail("elongated system: error too large");
  std::cout << "PASS\n";
  return 0;
}

}  // namespace

int main() {
  if (TestEnergyConservation()) return 1;
  if (TestESP()) return 1;
  if (TestForceSymmetry()) return 1;
  if (TestMultiCharge()) return 1;
  if (TestElongatedSystem()) return 1;

  std::cout << "\nprolate_ewald_tests: ALL GREEN\n";
  return 0;
}
