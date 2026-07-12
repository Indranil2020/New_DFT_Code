// BSSE / Counterpoise correction tests.
//
// Tests:
//   1. BSSE correction on He2 dimer with model energy
//   2. BSSE is positive (artificial stabilization)
//   3. Corrected energy > total energy (removes artificial lowering)
//   4. Monomer energies are finite

#include "basis/bsse.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::basis::BSSECorrection;
using tides::basis::CounterpoiseResult;

int Fail(const std::string& msg) {
  std::cerr << "bsse_tests: FAIL - " << msg << '\n';
  return 1;
}

// Simple model energy function: E = -sum_A Z_A + 0.1 * overlap_proxy
// where overlap_proxy increases with the number of atoms (simulating
// basis set superposition). For ghost atoms (Z=0), the energy is
// less negative but the basis functions still contribute.
// This simulates the BSSE effect: a monomer in the dimer basis has
// a lower energy than in its own basis.
double ModelEnergy(const std::vector<int>& Z, const std::vector<double>& positions) {
  const std::size_t n = Z.size();
  if (n == 0) return 0.0;

  // Nuclear attraction + electron count: E ~ -Z^2/2 per atom
  double E = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    E += -0.5 * static_cast<double>(Z[i]) * static_cast<double>(Z[i]);
  }

  // Simulated "basis set" effect: more atoms → more basis functions →
  // lower energy (variational principle). Even ghost atoms (Z=0)
  // contribute basis functions.
  E -= 0.05 * static_cast<double>(n);  // each atom's basis lowers energy

  return E;
}

int TestBSSEHe2() {
  std::cout << "\n=== BSSE Test 1: He2 dimer ===\n";
  std::vector<int> Z = {2, 2};
  std::vector<double> pos = {0.0, 0.0, 0.0, 3.0, 0.0, 0.0};

  auto res = BSSECorrection::ComputeDimer(Z, pos, 1, ModelEnergy);

  std::cout << "  Total energy (uncorrected) = " << res.total_energy << "\n";
  std::cout << "  BSSE correction            = " << res.bsse_correction << "\n";
  std::cout << "  Corrected energy            = " << res.corrected_energy << "\n";
  std::cout << "  E_He(isolated) = " << res.monomer_energies_isolated[0] << "\n";
  std::cout << "  E_He(full)     = " << res.monomer_energies_full[0] << "\n";

  if (!res.ok)
    return Fail("BSSE computation failed: " + res.error);

  if (!std::isfinite(res.bsse_correction))
    return Fail("BSSE correction is not finite");

  // BSSE should be positive (monomer in own basis has higher energy than
  // in full basis, so E_own - E_full > 0).
  if (res.bsse_correction < 0.0)
    return Fail("BSSE should be positive, got " +
                std::to_string(res.bsse_correction));

  // Corrected energy should be higher than total (less negative).
  if (res.corrected_energy < res.total_energy)
    return Fail("Corrected energy should be >= total energy");

  std::cout << "  PASS\n";
  return 0;
}

int TestBSSEMonomerEnergies() {
  std::cout << "\n=== BSSE Test 2: Monomer energies ===\n";
  std::vector<int> Z = {2, 2};
  std::vector<double> pos = {0.0, 0.0, 0.0, 3.0, 0.0, 0.0};

  auto res = BSSECorrection::ComputeDimer(Z, pos, 1, ModelEnergy);

  // Each monomer should have finite energy.
  for (std::size_t i = 0; i < res.monomer_energies_isolated.size(); ++i) {
    if (!std::isfinite(res.monomer_energies_isolated[i]))
      return Fail("Monomer isolated energy " + std::to_string(i) + " not finite");
    if (!std::isfinite(res.monomer_energies_full[i]))
      return Fail("Monomer full energy " + std::to_string(i) + " not finite");
  }

  // Monomer in full basis should have lower energy (more basis functions).
  for (std::size_t i = 0; i < res.monomer_energies_full.size(); ++i) {
    if (res.monomer_energies_full[i] > res.monomer_energies_isolated[i]) {
      return Fail("Monomer " + std::to_string(i) +
                  " full energy should be <= isolated energy");
    }
  }

  std::cout << "  PASS\n";
  return 0;
}

int TestBSSEWithFragments() {
  std::cout << "\n=== BSSE Test 3: Multi-fragment ===\n";
  // H2O...H2 complex: 3 fragments.
  std::vector<int> Z = {8, 1, 1, 1, 1};
  std::vector<double> pos = {
    0.0, 0.0, 0.0,    // O
    1.0, 0.0, 0.0,    // H1 (of H2O)
    -1.0, 0.0, 0.0,   // H2 (of H2O)
    3.0, 0.0, 0.0,    // H1 (of H2)
    4.0, 0.0, 0.0,    // H2 (of H2)
  };

  std::vector<std::vector<std::size_t>> fragments = {
    {0, 1, 2},  // H2O
    {3},         // H atom
    {4},         // H atom
  };

  auto res = BSSECorrection::Compute(Z, pos, fragments, ModelEnergy);

  std::cout << "  Total energy = " << res.total_energy << "\n";
  std::cout << "  BSSE         = " << res.bsse_correction << "\n";
  std::cout << "  Corrected    = " << res.corrected_energy << "\n";

  if (!res.ok)
    return Fail("Multi-fragment BSSE failed: " + res.error);

  if (!std::isfinite(res.bsse_correction))
    return Fail("BSSE not finite");

  // BSSE should be positive.
  if (res.bsse_correction < 0.0)
    return Fail("BSSE should be positive");

  std::cout << "  PASS\n";
  return 0;
}

int TestBSSEEmptyFragments() {
  std::cout << "\n=== BSSE Test 4: Error handling ===\n";
  std::vector<int> Z = {1, 1};
  std::vector<double> pos = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};

  auto res = BSSECorrection::ComputeDimer(Z, pos, 1, ModelEnergy);

  if (!res.ok)
    return Fail("Valid BSSE computation failed");

  // Test empty fragments.
  std::vector<std::vector<std::size_t>> empty_frags;
  auto res2 = BSSECorrection::Compute(Z, pos, empty_frags, ModelEnergy);

  if (res2.ok)
    return Fail("Empty fragments should fail");

  std::cout << "  PASS\n";
  return 0;
}

}  // namespace

int main() {
  std::cout << "=== BSSE / Counterpoise Correction Tests ===\n";
  int failures = 0;
  failures += TestBSSEHe2();
  failures += TestBSSEMonomerEnergies();
  failures += TestBSSEWithFragments();
  failures += TestBSSEEmptyFragments();

  if (failures == 0) {
    std::cout << "\nALL BSSE TESTS PASSED\n";
    return 0;
  }
  std::cout << "\n" << failures << " TEST(S) FAILED\n";
  return failures;
}
