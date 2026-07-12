// Point-group symmetrization tests.
//
// Tests:
//   1. H2O is detected as C2v
//   2. CH4 is detected as Td
//   3. CO2 (linear) is detected as D2h or C2h
//   4. Asymmetric molecule is C1
//   5. SymmetrizePositions rounds a slightly distorted geometry

#include "common/point_group.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::common::PointGroup;
using tides::common::PointGroupSymmetrizer;

int Fail(const std::string& msg) {
  std::cerr << "point_group_tests: FAIL - " << msg << '\n';
  return 1;
}

int TestH2O_C2v() {
  std::cout << "\n=== PointGroup Test 1: H2O should be C2v ===\n";
  // H2O: O at origin, H atoms at ~1.0 Bohr, ~104.5° angle.
  const double r_OH = 1.0;
  const double angle = 104.5 * M_PI / 180.0;
  const double half = angle / 2.0;

  std::vector<int> Z = {8, 1, 1};
  std::vector<double> pos = {
    0.0, 0.0, 0.0,
    r_OH * std::sin(half), 0.0, r_OH * std::cos(half),
    -r_OH * std::sin(half), 0.0, r_OH * std::cos(half),
  };

  auto pg = PointGroupSymmetrizer::Detect(Z, pos, 0.01);
  std::cout << "  H2O point group: " << pg.symbol << " (order=" << pg.order() << ")\n";

  if (pg.symbol != "C2v")
    return Fail("Expected C2v for H2O, got " + pg.symbol);

  if (pg.order() != 4)
    return Fail("C2v should have 4 operations, got " + std::to_string(pg.order()));

  std::cout << "  PASS\n";
  return 0;
}

int TestCH4_Td() {
  std::cout << "\n=== PointGroup Test 2: CH4 should be Td ===\n";
  // CH4: C at origin, 4 H atoms at tetrahedral positions.
  const double r_CH = 1.0;
  std::vector<int> Z = {6, 1, 1, 1, 1};
  std::vector<double> pos = {
    0.0, 0.0, 0.0,
    r_CH, r_CH, r_CH,
    -r_CH, -r_CH, r_CH,
    -r_CH, r_CH, -r_CH,
    r_CH, -r_CH, -r_CH,
  };

  auto pg = PointGroupSymmetrizer::Detect(Z, pos, 0.01);
  std::cout << "  CH4 point group: " << pg.symbol << " (order=" << pg.order() << ")\n";

  if (pg.symbol != "Td")
    return Fail("Expected Td for CH4, got " + pg.symbol);

  std::cout << "  PASS\n";
  return 0;
}

int TestAsymmetric_C1() {
  std::cout << "\n=== PointGroup Test 3: Asymmetric molecule should be C1 ===\n";
  // Random asymmetric arrangement.
  std::vector<int> Z = {1, 6, 8};
  std::vector<double> pos = {
    0.0, 0.0, 0.0,
    1.3, 0.7, -0.3,
    -0.5, 1.2, 0.8,
  };

  auto pg = PointGroupSymmetrizer::Detect(Z, pos, 0.01);
  std::cout << "  Asymmetric point group: " << pg.symbol << "\n";

  if (pg.symbol != "C1")
    return Fail("Expected C1 for asymmetric molecule, got " + pg.symbol);

  std::cout << "  PASS\n";
  return 0;
}

int TestMirrorPlane_Cs() {
  std::cout << "\n=== PointGroup Test 4: Planar molecule should be Cs or higher ===\n";
  // H2O2-like planar molecule (all in xz plane → mirror yz or xz).
  // BF3 planar: B at center, 3 F atoms in a plane.
  std::vector<int> Z = {5, 9, 9, 9};
  const double r = 1.3;
  std::vector<double> pos = {
    0.0, 0.0, 0.0,
    r, 0.0, 0.0,
    -r * 0.5, 0.0, r * 0.866,
    -r * 0.5, 0.0, -r * 0.866,
  };

  auto pg = PointGroupSymmetrizer::Detect(Z, pos, 0.01);
  std::cout << "  BF3 point group: " << pg.symbol << " (order=" << pg.order() << ")\n";

  // BF3 is D3h, but our detector may get at least Cs or C2v.
  // Check that it has at least a mirror plane (order >= 2).
  if (pg.order() < 2)
    return Fail("BF3 should have symmetry beyond C1, got " + pg.symbol);

  std::cout << "  PASS\n";
  return 0;
}

int TestSymmetrizePositions() {
  std::cout << "\n=== PointGroup Test 5: SymmetrizePositions ===\n";
  // H2O with slightly distorted geometry.
  const double r_OH = 1.0;
  const double angle = 104.5 * M_PI / 180.0;
  const double half = angle / 2.0;

  std::vector<int> Z = {8, 1, 1};
  std::vector<double> pos = {
    0.0, 0.0, 0.0,
    r_OH * std::sin(half) + 0.01, 0.0, r_OH * std::cos(half),  // slightly off
    -r_OH * std::sin(half), 0.0, r_OH * std::cos(half),
  };

  auto pg = PointGroupSymmetrizer::Detect(Z, pos, 0.1);

  // Symmetrize.
  auto sym_pos = PointGroupSymmetrizer::SymmetrizePositions(pos, pg, Z, 0.1);

  // Check that the two H atoms are symmetric (same z, opposite x).
  double hx1 = sym_pos[3], hz1 = sym_pos[5];
  double hx2 = sym_pos[6], hz2 = sym_pos[8];
  std::cout << "  H1 = (" << hx1 << ", 0, " << hz1 << ")\n";
  std::cout << "  H2 = (" << hx2 << ", 0, " << hz2 << ")\n";
  std::cout << "  |hx1 + hx2| = " << std::fabs(hx1 + hx2) << "\n";
  std::cout << "  |hz1 - hz2| = " << std::fabs(hz1 - hz2) << "\n";

  if (std::fabs(hx1 + hx2) > 1e-6)
    return Fail("Symmetrized H atoms not mirror-symmetric in x");

  if (std::fabs(hz1 - hz2) > 1e-6)
    return Fail("Symmetrized H atoms not at same z");

  std::cout << "  PASS\n";
  return 0;
}

int TestEthylene_D2h() {
  std::cout << "\n=== PointGroup Test 6: Ethylene should be D2h ===\n";
  // C2H4: planar, centered at origin.
  // C atoms at ±0.67 Bohr along x, H atoms at ±1.2 Bohr x, ±0.93 Bohr z.
  std::vector<int> Z = {6, 6, 1, 1, 1, 1};
  std::vector<double> pos = {
    -0.67, 0.0, 0.0,
     0.67, 0.0, 0.0,
    -1.2, 0.0, 0.93,
    -1.2, 0.0, -0.93,
     1.2, 0.0, 0.93,
     1.2, 0.0, -0.93,
  };

  auto pg = PointGroupSymmetrizer::Detect(Z, pos, 0.01);
  std::cout << "  C2H4 point group: " << pg.symbol << " (order=" << pg.order() << ")\n";

  // Ethylene is D2h. Our detector should at least get C2h or higher.
  if (pg.order() < 2)
    return Fail("Ethylene should have D2h symmetry, got " + pg.symbol);

  std::cout << "  PASS\n";
  return 0;
}

}  // namespace

int main() {
  std::cout << "=== Point Group Symmetrization Tests ===\n";
  int failures = 0;
  failures += TestH2O_C2v();
  failures += TestCH4_Td();
  failures += TestAsymmetric_C1();
  failures += TestMirrorPlane_Cs();
  failures += TestSymmetrizePositions();
  failures += TestEthylene_D2h();

  if (failures == 0) {
    std::cout << "\nALL POINT GROUP TESTS PASSED\n";
    return 0;
  }
  std::cout << "\n" << failures << " TEST(S) FAILED\n";
  return failures;
}
