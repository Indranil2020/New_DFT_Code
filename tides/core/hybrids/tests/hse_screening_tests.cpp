// T7.4: Short-range HSE screening tests.
//
// Validates:
//   - Screened Coulomb SR + LR = 1/r
//   - Tile screening drops pairs beyond r_cut
//   - Exchange energy with screening < exchange energy without screening
//   - Screening stats correctly count active/screened pairs

#include "hybrids/hse_screening.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::hybrids::BuildScreenedExchange;
using tides::hybrids::ComputeScreeningStats;
using tides::hybrids::HSEParams;
using tides::hybrids::ScreenedCoulomb;
using tides::hybrids::ScreenedExchangeEnergy;
using tides::hybrids::ScreeningStats;

int Fail(const std::string& msg) {
  std::cerr << "FAIL: " << msg << '\n';
  return 1;
}

// T7.4a: SR + LR = 1/r (partition of Coulomb operator).
int TestCoulombPartition() {
  std::cout << "\n=== T7.4a: Screened Coulomb partition ===\n";
  ScreenedCoulomb sc(0.11);
  double max_err = 0.0;
  for (double r : {0.5, 1.0, 2.0, 5.0, 10.0, 20.0}) {
    double sr = sc.SR(r);
    double lr = sc.LR(r);
    double full = sc.Full(r);
    double err = std::fabs(sr + lr - full);
    max_err = std::max(max_err, err);
    std::cout << "  r=" << r << " SR=" << sr << " LR=" << lr
              << " SR+LR=" << sr + lr << " 1/r=" << full
              << " err=" << err << '\n';
  }
  if (max_err > 1e-12) return Fail("T7.4a: SR+LR != 1/r");
  std::cout << "T7.4a: GREEN\n";
  return 0;
}

// T7.4b: Tile screening drops pairs beyond r_cut.
int TestTileScreening() {
  std::cout << "\n=== T7.4b: Tile screening ===\n";
  // 3 atoms at positions: (0,0,0), (1,0,0), (10,0,0)
  // 1 basis function per atom (n=3).
  std::vector<double> positions = {0, 0, 0,  1, 0, 0,  10, 0, 0};
  std::vector<std::size_t> centers = {0, 1, 2};

  // Simple density matrix (all pairs non-zero).
  std::vector<double> P = {1, 0.5, 0.3,  0.5, 1, 0.2,  0.3, 0.2, 1};

  // Without screening (r_cut = 0): all pairs active.
  HSEParams params_no_cut;
  params_no_cut.r_cut_screen = 0.0;
  auto K_full = BuildScreenedExchange(3, P, centers, positions, params_no_cut);

  // With screening at r_cut = 5.0: pair (0,2) and (1,2) should be zero.
  HSEParams params_cut;
  params_cut.r_cut_screen = 5.0;
  auto K_screened = BuildScreenedExchange(3, P, centers, positions, params_cut);

  // Check that K_screened[0][2] = 0 and K_screened[1][2] = 0
  // (atoms 0 and 2 are 10 Bohr apart, beyond r_cut=5).
  if (std::fabs(K_screened[0 * 3 + 2]) > 1e-15)
    return Fail("T7.4b: K[0][2] should be zero (screened)");
  if (std::fabs(K_screened[2 * 3 + 0]) > 1e-15)
    return Fail("T7.4b: K[2][0] should be zero (screened)");
  if (std::fabs(K_screened[1 * 3 + 2]) > 1e-15)
    return Fail("T7.4b: K[1][2] should be zero (screened)");
  if (std::fabs(K_screened[2 * 3 + 1]) > 1e-15)
    return Fail("T7.4b: K[2][1] should be zero (screened)");

  // K[0][1] should be non-zero (atoms 0 and 1 are 1 Bohr apart).
  if (std::fabs(K_screened[0 * 3 + 1]) < 1e-10)
    return Fail("T7.4b: K[0][1] should be non-zero (within r_cut)");

  // K[0][0] should be non-zero (self-interaction, r=0).
  if (std::fabs(K_screened[0 * 3 + 0]) < 1e-10)
    return Fail("T7.4b: K[0][0] should be non-zero (self)");

  // Full (no cut) should have non-zero K[0][2].
  if (std::fabs(K_full[0 * 3 + 2]) < 1e-10)
    return Fail("T7.4b: K_full[0][2] should be non-zero (no screening)");

  std::cout << "  K_screened[0][1]=" << K_screened[0 * 3 + 1]
            << " K_screened[0][2]=" << K_screened[0 * 3 + 2] << '\n';
  std::cout << "T7.4b: GREEN\n";
  return 0;
}

// T7.4c: Screened exchange energy < unscreened exchange energy.
int TestExchangeEnergy() {
  std::cout << "\n=== T7.4c: Exchange energy comparison ===\n";
  // 4 atoms in a line: 0, 2, 4, 6 Bohr apart.
  std::vector<double> positions = {0, 0, 0,  2, 0, 0,  4, 0, 0,  6, 0, 0};
  std::vector<std::size_t> centers = {0, 1, 2, 3};
  std::vector<double> P(16, 0.0);
  for (std::size_t i = 0; i < 4; ++i) P[i * 4 + i] = 1.0;
  for (std::size_t i = 0; i < 3; ++i) {
    P[i * 4 + (i + 1)] = 0.3;
    P[(i + 1) * 4 + i] = 0.3;
  }

  // Unscreened (r_cut = 0).
  HSEParams params_full;
  auto K_full = BuildScreenedExchange(4, P, centers, positions, params_full);
  double E_full = ScreenedExchangeEnergy(4, P, K_full);

  // Screened at r_cut = 3.0 (drops pairs beyond 3 Bohr).
  HSEParams params_cut;
  params_cut.r_cut_screen = 3.0;
  auto K_cut = BuildScreenedExchange(4, P, centers, positions, params_cut);
  double E_cut = ScreenedExchangeEnergy(4, P, K_cut);

  std::cout << "  E_x(full)=" << E_full << " E_x(screened)=" << E_cut << '\n';
  if (std::fabs(E_cut) > std::fabs(E_full))
    return Fail("T7.4c: |E_x(screened)| should be <= |E_x(full)|");
  if (std::fabs(E_full) < 1e-10)
    return Fail("T7.4c: E_x(full) should be non-zero");

  std::cout << "T7.4c: GREEN\n";
  return 0;
}

// T7.4d: Screening stats.
int TestScreeningStats() {
  std::cout << "\n=== T7.4d: Screening stats ===\n";
  // 5 atoms: at 0, 1, 2, 10, 20 Bohr.
  std::vector<double> positions = {0, 0, 0,  1, 0, 0,  2, 0, 0,
                                    10, 0, 0,  20, 0, 0};
  auto stats = ComputeScreeningStats(positions, 5.0);
  std::cout << "  n_atoms=5 n_pairs=" << stats.n_pairs_total
            << " active=" << stats.n_pairs_active
            << " screened=" << stats.n_pairs_screened
            << " fraction=" << stats.fraction_screened << '\n';

  // 5 atoms -> C(5,2) = 10 pairs.
  if (stats.n_pairs_total != 10) return Fail("T7.4d: wrong pair count");
  // Pairs within 5 Bohr: (0,1)=1, (0,2)=2, (1,2)=1 -> 3 active.
  if (stats.n_pairs_active != 3) return Fail("T7.4d: wrong active count");
  if (stats.n_pairs_screened != 7) return Fail("T7.4d: wrong screened count");
  if (std::fabs(stats.fraction_screened - 0.7) > 1e-10)
    return Fail("T7.4d: wrong fraction");

  std::cout << "T7.4d: GREEN\n";
  return 0;
}

}  // namespace

int main() {
  if (TestCoulombPartition()) return 1;
  if (TestTileScreening()) return 1;
  if (TestExchangeEnergy()) return 1;
  if (TestScreeningStats()) return 1;
  std::cout << "\nhse_screening_tests: ALL GREEN\n";
  return 0;
}
