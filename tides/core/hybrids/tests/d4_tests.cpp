// D4 dispersion correction tests.
//
// Tests:
//   1. D4 energy is finite and negative (attractive) for a dimer
//   2. D4 with zero charges reduces to D3 limit
//   3. D4 with nonzero charges differs from D3
//   4. Forces are finite and satisfy Newton's 3rd law
//   5. Pair energy matches ComputeEnergy for a dimer

#include "hybrids/d4_dispersion.hpp"
#include "hybrids/d3_dispersion.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::hybrids::D4Dispersion;
using tides::hybrids::D4Result;
using tides::hybrids::D3Dispersion;
using tides::hybrids::DispersionResult;

int Fail(const std::string& msg) {
  std::cerr << "d4_tests: FAIL - " << msg << '\n';
  return 1;
}

int TestD4EnergyFinite() {
  std::cout << "\n=== D4 Test 1: Energy is finite and negative ===\n";
  std::vector<int> Z = {1, 1};
  std::vector<double> pos = {0.0, 0.0, 0.0, 3.0, 0.0, 0.0};

  auto res = D4Dispersion::ComputeEnergy(Z, pos);

  std::cout << "  D4 energy = " << res.energy << " Ha\n";
  std::cout << "  charges = [" << res.charges[0] << ", " << res.charges[1] << "]\n";

  if (!std::isfinite(res.energy))
    return Fail("D4 energy is not finite");
  if (res.energy >= 0.0)
    return Fail("D4 energy should be negative (attractive), got " +
                std::to_string(res.energy));

  std::cout << "  PASS\n";
  return 0;
}

int TestD4ZeroChargesEqualsD3() {
  std::cout << "\n=== D4 Test 2: Zero charges = D3 limit ===\n";
  std::vector<int> Z = {6, 8};
  std::vector<double> pos = {0.0, 0.0, 0.0, 2.5, 0.0, 0.0};

  // D4 with explicit zero charges.
  std::vector<double> zero_charges = {0.0, 0.0};
  auto d4_res = D4Dispersion::ComputeEnergy(Z, pos, zero_charges);

  // D3 reference.
  auto d3_res = D3Dispersion::ComputeEnergy(Z, pos);

  std::cout << "  D4 (q=0) energy = " << d4_res.energy << " Ha\n";
  std::cout << "  D3 energy       = " << d3_res.energy << " Ha\n";
  std::cout << "  difference       = " << std::fabs(d4_res.energy - d3_res.energy) << "\n";

  if (std::fabs(d4_res.energy - d3_res.energy) > 1e-10)
    return Fail("D4 with zero charges should equal D3, diff = " +
                std::to_string(d4_res.energy - d3_res.energy));

  std::cout << "  PASS\n";
  return 0;
}

int TestD4NonzeroChargesDiffers() {
  std::cout << "\n=== D4 Test 3: Nonzero charges differ from D3 ===\n";
  std::vector<int> Z = {6, 8};
  std::vector<double> pos = {0.0, 0.0, 0.0, 2.5, 0.0, 0.0};

  // D4 with charges.
  std::vector<double> charges = {0.3, -0.3};  // C slightly positive, O negative
  auto d4_charged = D4Dispersion::ComputeEnergy(Z, pos, charges);
  auto d4_zero = D4Dispersion::ComputeEnergy(Z, pos, {0.0, 0.0});

  std::cout << "  D4 (q=[0.3,-0.3]) energy = " << d4_charged.energy << " Ha\n";
  std::cout << "  D4 (q=[0,0])      energy = " << d4_zero.energy << " Ha\n";
  std::cout << "  difference = " << std::fabs(d4_charged.energy - d4_zero.energy) << "\n";

  if (std::fabs(d4_charged.energy - d4_zero.energy) < 1e-12)
    return Fail("D4 with nonzero charges should differ from zero-charge D4");

  std::cout << "  PASS\n";
  return 0;
}

int TestD4Forces() {
  std::cout << "\n=== D4 Test 4: Forces finite and Newton 3rd law ===\n";
  std::vector<int> Z = {6, 8};
  std::vector<double> pos = {0.0, 0.0, 0.0, 2.5, 0.0, 0.0};

  auto res = D4Dispersion::ComputeEnergy(Z, pos);

  // Check forces are finite.
  for (double f : res.forces) {
    if (!std::isfinite(f))
      return Fail("D4 force is not finite");
  }

  // Newton's 3rd law: F_A = -F_B.
  double net_fx = res.forces[0] + res.forces[3];
  double net_fy = res.forces[1] + res.forces[4];
  double net_fz = res.forces[2] + res.forces[5];

  std::cout << "  net force = [" << net_fx << ", " << net_fy << ", " << net_fz << "]\n";

  if (std::fabs(net_fx) > 1e-10 || std::fabs(net_fy) > 1e-10 ||
      std::fabs(net_fz) > 1e-10)
    return Fail("Newton's 3rd law violated");

  std::cout << "  PASS\n";
  return 0;
}

int TestD4PairEnergy() {
  std::cout << "\n=== D4 Test 5: PairEnergy matches ComputeEnergy ===\n";
  std::vector<int> Z = {1, 1};
  std::vector<double> pos = {0.0, 0.0, 0.0, 4.0, 0.0, 0.0};

  auto res = D4Dispersion::ComputeEnergy(Z, pos, {0.0, 0.0});
  double pair = D4Dispersion::PairEnergy(1, 1, 4.0, 0.0, 0.0);

  std::cout << "  ComputeEnergy = " << res.energy << "\n";
  std::cout << "  PairEnergy    = " << pair << "\n";

  if (std::fabs(res.energy - pair) > 1e-12)
    return Fail("PairEnergy does not match ComputeEnergy");

  std::cout << "  PASS\n";
  return 0;
}

int TestD4EEQCharges() {
  std::cout << "\n=== D4 Test 6: EEQ charges are reasonable ===\n";
  // H2O: O should be negative, H should be positive.
  std::vector<int> Z = {8, 1, 1};
  std::vector<double> pos = {
    0.0, 0.0, 0.0,           // O
    0.0, 1.0, 0.0,           // H1
    0.0, -1.0, 0.0           // H2 (linear for simplicity)
  };

  auto res = D4Dispersion::ComputeEnergy(Z, pos);
  std::cout << "  charges = [" << res.charges[0] << ", "
            << res.charges[1] << ", " << res.charges[2] << "]\n";

  // Sum of charges should be ~0 (charge neutrality).
  double q_sum = 0.0;
  for (double q : res.charges) q_sum += q;
  if (std::fabs(q_sum) > 1e-6)
    return Fail("EEQ charges not neutral (sum = " + std::to_string(q_sum) + ")");

  // O should be negative (more electronegative).
  if (res.charges[0] >= 0.0)
    return Fail("O should have negative charge, got " +
                std::to_string(res.charges[0]));

  std::cout << "  PASS\n";
  return 0;
}

}  // namespace

int main() {
  std::cout << "=== D4 Dispersion Tests ===\n";
  int failures = 0;
  failures += TestD4EnergyFinite();
  failures += TestD4ZeroChargesEqualsD3();
  failures += TestD4NonzeroChargesDiffers();
  failures += TestD4Forces();
  failures += TestD4PairEnergy();
  failures += TestD4EEQCharges();

  if (failures == 0) {
    std::cout << "\nALL D4 TESTS PASSED\n";
    return 0;
  }
  std::cout << "\n" << failures << " TEST(S) FAILED\n";
  return failures;
}
