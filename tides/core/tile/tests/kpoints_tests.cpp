// K-point (Monkhorst-Pack) sampling tests.
#include "tile/kpoints.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {
using tides::tile::KPoint;
using tides::tile::KPointGrid;
using tides::tile::KPointSampler;

int Fail(const std::string& msg) {
  std::cerr << "kpoints_tests: FAIL — " << msg << '\n';
  return 1;
}

int TestMonkhorstPackGrid() {
  std::cout << "\n=== K-Points: Monkhorst-Pack Grid (2x2x2) ===\n";
  std::array<int, 3> dims = {2, 2, 2};
  std::array<std::array<double, 3>, 3> recip{{
    {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}
  }};
  auto grid = KPointSampler::GenerateMonkhorstPack(dims, recip, false, false);

  std::cout << "  Full grid: " << grid.kpoints.size() << " k-points\n";
  for (const auto& kp : grid.kpoints)
    std::cout << "    k = (" << kp.kvec[0] << ", " << kp.kvec[1]
              << ", " << kp.kvec[2] << ") w=" << kp.weight << "\n";

  // Without time-reversal: 2*2*2 = 8 k-points.
  if (grid.kpoints.size() != 8)
    return Fail("Expected 8 k-points without TR, got " +
                std::to_string(grid.kpoints.size()));
  // Weights should sum to 1.
  double wsum = 0.0;
  for (const auto& kp : grid.kpoints) wsum += kp.weight;
  if (std::fabs(wsum - 1.0) > 1e-12)
    return Fail("Weights don't sum to 1: " + std::to_string(wsum));
  std::cout << "  PASS\n";
  return 0;
}

int TestTimeReversal() {
  std::cout << "\n=== K-Points: Time-Reversal Symmetry (2x2x2) ===\n";
  std::array<int, 3> dims = {2, 2, 2};
  std::array<std::array<double, 3>, 3> recip{{
    {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}
  }};
  auto grid = KPointSampler::GenerateMonkhorstPack(dims, recip, false, true);

  std::cout << "  With TR: " << grid.kpoints.size() << " irreducible k-points\n";
  // With TR for 2x2x2: k and -k are equivalent. The grid has points at
  // ±0.25, so there are 4 unique pairs → 4 irreducible points.
  if (grid.kpoints.size() != 4)
    return Fail("Expected 4 irreducible k-points with TR, got " +
                std::to_string(grid.kpoints.size()));
  // Weights should still sum to 1.
  double wsum = 0.0;
  for (const auto& kp : grid.kpoints) wsum += kp.weight;
  if (std::fabs(wsum - 1.0) > 1e-12)
    return Fail("Weights don't sum to 1 with TR: " + std::to_string(wsum));
  std::cout << "  PASS\n";
  return 0;
}

int TestGammaCentered() {
  std::cout << "\n=== K-Points: Gamma-Centered (3x3x3) ===\n";
  std::array<int, 3> dims = {3, 3, 3};
  std::array<std::array<double, 3>, 3> recip{{
    {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}
  }};
  auto grid = KPointSampler::GenerateMonkhorstPack(dims, recip, true, false);

  // Gamma-centered: k=0 should be in the grid.
  bool has_gamma = false;
  for (const auto& kp : grid.kpoints) {
    if (std::fabs(kp.kvec[0]) < 1e-12 &&
        std::fabs(kp.kvec[1]) < 1e-12 &&
        std::fabs(kp.kvec[2]) < 1e-12)
      has_gamma = true;
  }
  std::cout << "  Has Gamma point: " << (has_gamma ? "yes" : "no") << "\n";
  if (!has_gamma)
    return Fail("Gamma-centered grid should include k=0");
  std::cout << "  PASS\n";
  return 0;
}

int TestFoldToBZ() {
  std::cout << "\n=== K-Points: Fold to BZ ===\n";
  auto k = KPointSampler::FoldToBZ({0.6, -0.7, 1.5});
  std::cout << "  (0.6, -0.7, 1.5) → (" << k[0] << ", " << k[1] << ", " << k[2] << ")\n";
  // 0.6 → -0.4, -0.7 → -0.7 (already in [-0.5, 0.5)? No: -0.7 → 0.3), 1.5 → -0.5
  if (k[0] > 0.5 || k[0] < -0.5)
    return Fail("kx not in [-0.5, 0.5): " + std::to_string(k[0]));
  if (k[1] > 0.5 || k[1] < -0.5)
    return Fail("ky not in [-0.5, 0.5): " + std::to_string(k[1]));
  if (k[2] > 0.5 || k[2] < -0.5)
    return Fail("kz not in [-0.5, 0.5): " + std::to_string(k[2]));
  std::cout << "  PASS\n";
  return 0;
}

int TestBlochPhase() {
  std::cout << "\n=== K-Points: Bloch Phase Transform ===\n";
  std::size_t n = 2;
  std::vector<double> H = {1.0, 0.5, 0.5, 2.0};
  KPoint kp;
  kp.kvec = {0.0, 0.0, 0.0};
  kp.weight = 1.0;
  std::array<std::array<double, 3>, 3> lat{{
    {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}
  }};
  auto H_k = KPointSampler::BlochPhaseTransform(H, n, kp, lat);

  // At Gamma, H(k) should equal H (all real).
  std::cout << "  H(Gamma) = [[" << H_k[0].real() << ", " << H_k[1].real()
            << "], [" << H_k[2].real() << ", " << H_k[3].real() << "]]\n";
  for (std::size_t i = 0; i < n * n; ++i) {
    if (std::fabs(H_k[i].imag()) > 1e-12)
      return Fail("Gamma-point H should be purely real");
    if (std::fabs(H_k[i].real() - H[i]) > 1e-12)
      return Fail("Gamma-point H should equal real H");
  }
  std::cout << "  PASS\n";
  return 0;
}

}  // namespace

int main() {
  std::cout << "=== K-Point Sampling Tests ===\n";
  int failures = 0;
  failures += TestMonkhorstPackGrid();
  failures += TestTimeReversal();
  failures += TestGammaCentered();
  failures += TestFoldToBZ();
  failures += TestBlochPhase();
  if (failures == 0) std::cout << "\nALL K-POINT TESTS PASSED\n";
  else std::cout << "\n" << failures << " TEST(S) FAILED\n";
  return failures;
}
