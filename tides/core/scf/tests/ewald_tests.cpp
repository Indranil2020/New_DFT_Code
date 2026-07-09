// Ewald summation tests for periodic ion-ion electrostatics.
//
// Validates:
//   - NaCl-like crystal: Ewald energy converges with shells
//   - Simple cubic: matches known Madelung constant
//   - Neutral system: Ewald matches direct sum at large cell
//   - Charged system: background correction applied
//   - Energy components are individually correct

#include "scf/ewald.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::scf::EwaldResult;
using tides::scf::EwaldSum;

int Fail(const std::string& msg) {
  std::cerr << "FAIL: " << msg << '\n';
  return 1;
}

// Test 1: Simple cubic — two charges in a box.
// At large box, Ewald should approach the direct Coulomb sum.
int TestLargeBoxLimit() {
  std::cout << "\n=== Ewald: Large box -> direct sum ===\n";
  // Two opposite charges in a large box.
  std::vector<double> pos = {0.0, 0.0, 0.0, 2.0, 0.0, 0.0};
  std::vector<double> q = {1.0, -1.0};

  // Direct sum (free BC).
  double E_direct = EwaldSum::DirectCoulomb(pos, q);
  std::cout << std::scientific << std::setprecision(8);
  std::cout << "  Direct Coulomb: " << E_direct << '\n';

  // Ewald in a large box (L=50 Bohr).
  std::array<double, 9> cell = {
    50.0, 0.0, 0.0,
    0.0, 50.0, 0.0,
    0.0, 0.0, 50.0
  };
  auto result = EwaldSum::Compute(pos, q, cell, 0.0, 0, 0);
  if (!result.ok) return Fail("large box: Ewald failed");

  std::cout << "  Ewald (L=50): " << result.energy
            << " (real=" << result.real_space
            << " recip=" << result.reciprocal
            << " self=" << result.self_correction << ")\n";

  double err = std::fabs(result.energy - E_direct);
  std::cout << "  |err|=" << err << '\n';
  // At L=50, image contributions are ~1/L = 0.02, so tolerance is loose.
  if (err > 0.05) return Fail("large box: Ewald != direct sum");

  std::cout << "PASS\n";
  return 0;
}

// Test 2: NaCl crystal — Madelung constant.
// NaCl: simple cubic lattice with alternating charges.
// Madelung constant alpha_M = 1.74756 for NaCl structure.
// E_per_ion-pair = -alpha_M * q^2 / r_nn (r_nn = nearest-neighbor distance).
int TestNaClMadelung() {
  std::cout << "\n=== Ewald: NaCl Madelung constant ===\n";
  // NaCl: two interpenetrating FCC lattices.
  // Simple model: 2 atoms in cubic cell at (0,0,0) and (a/2, a/2, a/2).
  // This gives the correct Madelung constant for the CsCl structure
  // (alpha_M = 1.76267 for CsCl, close to NaCl's 1.74756).
  const double a = 4.0;  // lattice constant (Bohr)
  std::vector<double> pos = {0.0, 0.0, 0.0, a / 2, a / 2, a / 2};
  std::vector<double> q = {1.0, -1.0};

  std::array<double, 9> cell = {
    a, 0.0, 0.0,
    0.0, a, 0.0,
    0.0, 0.0, a
  };

  auto result = EwaldSum::Compute(pos, q, cell, 0.0, 0, 0);
  if (!result.ok) return Fail("NaCl: Ewald failed");

  // Expected: E = -alpha_M * q^2 / r_nn where r_nn = a*sqrt(3)/2 (body diagonal/2).
  const double r_nn = a * std::sqrt(3.0) / 2.0;
  const double alpha_M_CsCl = 1.76267;  // CsCl Madelung constant
  const double E_expected = -alpha_M_CsCl / r_nn;

  std::cout << std::scientific << std::setprecision(8);
  std::cout << "  E_ewald=" << result.energy
            << " E_expected=" << E_expected
            << " r_nn=" << r_nn << '\n';
  std::cout << "  real=" << result.real_space
            << " recip=" << result.reciprocal
            << " self=" << result.self_correction << '\n';

  double err = std::fabs(result.energy - E_expected);
  std::cout << "  |err|=" << err << '\n';
  if (err > 1e-4) return Fail("NaCl: Madelung constant mismatch");

  std::cout << "PASS\n";
  return 0;
}

// Test 3: Convergence with Ewald parameters.
int TestConvergence() {
  std::cout << "\n=== Ewald: Convergence ===\n";
  // Simple cubic with 2 charges.
  const double a = 5.0;
  std::vector<double> pos = {0.0, 0.0, 0.0, 2.5, 2.5, 2.5};
  std::vector<double> q = {1.0, -1.0};
  std::array<double, 9> cell = {
    a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a
  };

  // Run with increasing shells and check convergence.
  double prev_energy = 0.0;
  for (int n_shell : {2, 4, 6, 8}) {
    auto result = EwaldSum::Compute(pos, q, cell, 0.5, n_shell, n_shell);
    if (!result.ok) return Fail("convergence: failed at n=" + std::to_string(n_shell));
    double delta = std::fabs(result.energy - prev_energy);
    std::cout << std::scientific << std::setprecision(8);
    std::cout << "  n_shell=" << n_shell << " E=" << result.energy
              << " delta=" << delta << '\n';
    if (n_shell >= 4 && delta > 1e-6)
      return Fail("convergence: not converging at n=" + std::to_string(n_shell));
    prev_energy = result.energy;
  }

  std::cout << "PASS\n";
  return 0;
}

// Test 4: Neutral system — no background correction needed.
int TestNeutralSystem() {
  std::cout << "\n=== Ewald: Neutral system ===\n";
  std::vector<double> pos = {0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.5};
  std::vector<double> q = {1.0, -1.0, 1.0, -1.0};
  std::array<double, 9> cell = {
    6.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 6.0
  };

  auto result = EwaldSum::Compute(pos, q, cell);
  if (!result.ok) return Fail("neutral: Ewald failed");

  std::cout << std::scientific << std::setprecision(8);
  std::cout << "  E=" << result.energy
            << " background=" << result.background << '\n';

  // For a neutral system, background should be ~0.
  if (std::fabs(result.background) > 1e-10)
    return Fail("neutral: background should be zero");

  std::cout << "PASS\n";
  return 0;
}

// Test 5: Charged system — background correction applied.
int TestChargedSystem() {
  std::cout << "\n=== Ewald: Charged system ===\n";
  std::vector<double> pos = {0.0, 0.0, 0.0, 1.5, 0.0, 0.0};
  std::vector<double> q = {1.0, 1.0};  // total charge = +2
  std::array<double, 9> cell = {
    5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0
  };

  auto result = EwaldSum::Compute(pos, q, cell, 0.5, 4, 4);
  if (!result.ok) return Fail("charged: Ewald failed");

  std::cout << std::scientific << std::setprecision(8);
  std::cout << "  E=" << result.energy
            << " background=" << result.background << '\n';

  // Background should be negative (removes divergent self-energy).
  if (result.background >= 0.0)
    return Fail("charged: background should be negative for like charges");

  std::cout << "PASS\n";
  return 0;
}

// Test 6: Self-correction is correct.
int TestSelfCorrection() {
  std::cout << "\n=== Ewald: Self correction ===\n";
  // Single charge: Ewald should give only self + background.
  std::vector<double> pos = {0.0, 0.0, 0.0};
  std::vector<double> q = {1.0};
  std::array<double, 9> cell = {
    5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0
  };

  double alpha = 0.5;
  auto result = EwaldSum::Compute(pos, q, cell, alpha, 4, 4);
  if (!result.ok) return Fail("self: Ewald failed");

  // For a single charge, real and recip should cancel (no pairs).
  // E = self + background.
  double expected_self = -alpha / std::sqrt(M_PI);
  double expected_bg = -0.5 * M_PI / (alpha * alpha * 125.0);  // V=125

  std::cout << std::scientific << std::setprecision(8);
  std::cout << "  self=" << result.self_correction
            << " expected=" << expected_self << '\n';
  std::cout << "  bg=" << result.background
            << " expected=" << expected_bg << '\n';
  std::cout << "  real=" << result.real_space
            << " recip=" << result.reciprocal << '\n';

  if (std::fabs(result.self_correction - expected_self) > 1e-10)
    return Fail("self: correction mismatch");
  if (std::fabs(result.background - expected_bg) > 1e-8)
    return Fail("self: background mismatch");

  std::cout << "PASS\n";
  return 0;
}

}  // namespace

int main() {
  if (TestLargeBoxLimit()) return 1;
  if (TestNaClMadelung()) return 1;
  if (TestConvergence()) return 1;
  if (TestNeutralSystem()) return 1;
  if (TestChargedSystem()) return 1;
  if (TestSelfCorrection()) return 1;
  std::cout << "\newald_tests: ALL GREEN\n";
  return 0;
}
