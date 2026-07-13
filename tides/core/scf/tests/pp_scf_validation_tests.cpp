// Pseudopotential SCF validation tests: end-to-end NaoDriver with PPs.
//
// These tests exercise the pseudopotential path of NaoDriver::Run using
// trivial (all-electron-equivalent) pseudopotentials, where v_local = -Z/r
// and there are no KB channels. This isolates the PP SCF machinery from
// the quality of any particular PP — the results should match all-electron.
//
// Tests:
//   1. H atom with PP (Z_valence=1) — SCF converges, energy ~ all-electron H
//   2. He atom with PP (Z_valence=2) — SCF converges, energy ~ all-electron He
//
// A "trivial" pseudopotential has Z_valence = Z (no core frozen),
// v_local(r) = -Z/r (same as all-electron nuclear attraction),
// and no KB channels (no nonlocal projectors). This should give identical
// results to the all-electron calculation, providing a clean SCF validation.

#include "scf/nao_driver.hpp"
#include "basis/pseudo/pseudopotential.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::scf::NaoDriver;
using tides::scf::NaoDriverResult;
using tides::basis::Pseudopotential;

int Fail(const std::string& msg) {
  std::cerr << "pp_scf_validation_tests: FAIL — " << msg << '\n';
  return 1;
}

// Create a trivial pseudopotential: v_local = -Z/r, no KB channels.
// This should give the same result as all-electron.
Pseudopotential MakeTrivialPP(int Z) {
  Pseudopotential pp;
  pp.Z_valence = Z;
  pp.element = (Z == 1) ? "H" : (Z == 2) ? "He" : "X";
  pp.format = "UPF2";  // trivial PP, format tag for provenance
  // Radial grid: 0.01 to 20.0 Bohr, 500 points
  const int n_r = 500;
  pp.r_grid.resize(n_r);
  pp.v_local.resize(n_r);
  for (int i = 0; i < n_r; ++i) {
    double r = 0.01 + 20.0 * static_cast<double>(i) / static_cast<double>(n_r - 1);
    pp.r_grid[i] = r;
    pp.v_local[i] = -static_cast<double>(Z) / r;
  }
  // No KB channels — trivial PP has no nonlocal part.
  pp.l_max = 0;
  pp.rcut = pp.r_grid.back();
  return pp;
}

// Test 1: H atom with PP (Z_valence=1)
// Validates that the PP SCF path converges and gives an energy consistent
// with the all-electron result (trivial PP: v_local = -1/r, no core).
int TestHAtomWithPP() {
  std::cout << "\n=== Test 1: H atom with PP (Z_valence=1) ===\n";

  std::vector<int> Z = {1};
  std::vector<double> pos = {0.0, 0.0, 0.0};

  // Run without PP (all-electron reference)
  auto res_ae = NaoDriver::Run(Z, pos, 0.2, 4.0, 100, 1e-6);
  std::cout << "  All-electron: E=" << res_ae.scf.energy
            << " converged=" << res_ae.scf.converged << "\n";

  // Run with trivial PP (should match all-electron)
  auto pp = MakeTrivialPP(1);
  std::vector<Pseudopotential> pps = {pp};
  auto res_pp = NaoDriver::Run(Z, pos, 0.2, 4.0, 100, 1e-6, &pps);
  std::cout << "  With PP:      E=" << res_pp.scf.energy
            << " converged=" << res_pp.scf.converged << "\n";
  std::cout << "  n_electrons (PP) = " << res_pp.n_electrons << "\n";

  if (!res_ae.scf.converged)
    return Fail("H all-electron SCF did not converge");
  if (!res_pp.scf.converged)
    return Fail("H PP SCF did not converge");

  // Verify Z_valence was used: PP path should see 1 electron.
  if (res_pp.n_electrons != 1)
    return Fail("H PP should have 1 electron, got " +
                std::to_string(res_pp.n_electrons));

  double diff = std::fabs(res_ae.scf.energy - res_pp.scf.energy);
  std::cout << "  |E_ae - E_pp| = " << diff << " Ha\n";

  // Trivial PP should match all-electron closely.
  if (diff > 0.05) {
    return Fail("H PP energy differs from all-electron by " +
                std::to_string(diff) + " > 0.05 Ha");
  }

  // Sanity: H atom energy should be negative (bound state).
  if (res_pp.scf.energy > 0.0)
    return Fail("H PP energy should be negative, got " +
                std::to_string(res_pp.scf.energy));

  std::cout << "  PASS (PP matches all-electron, diff=" << diff << ")\n";
  return 0;
}

// Test 2: He atom with PP (Z_valence=2)
// Validates the PP SCF path with 2 valence electrons. The trivial PP
// (v_local = -2/r) should match the all-electron He result.
int TestHeAtomWithPP() {
  std::cout << "\n=== Test 2: He atom with PP (Z_valence=2) ===\n";

  std::vector<int> Z = {2};
  std::vector<double> pos = {0.0, 0.0, 0.0};

  // Run without PP (all-electron reference)
  auto res_ae = NaoDriver::Run(Z, pos, 0.2, 4.0, 100, 1e-6);
  std::cout << "  All-electron: E=" << res_ae.scf.energy
            << " converged=" << res_ae.scf.converged << "\n";

  // Run with trivial PP (should match all-electron)
  auto pp = MakeTrivialPP(2);
  std::vector<Pseudopotential> pps = {pp};
  auto res_pp = NaoDriver::Run(Z, pos, 0.2, 4.0, 100, 1e-6, &pps);
  std::cout << "  With PP:      E=" << res_pp.scf.energy
            << " converged=" << res_pp.scf.converged << "\n";
  std::cout << "  n_electrons (PP) = " << res_pp.n_electrons << "\n";

  if (!res_ae.scf.converged)
    return Fail("He all-electron SCF did not converge");
  if (!res_pp.scf.converged)
    return Fail("He PP SCF did not converge");

  // Verify Z_valence was used: PP path should see 2 electrons.
  if (res_pp.n_electrons != 2)
    return Fail("He PP should have 2 electrons, got " +
                std::to_string(res_pp.n_electrons));

  double diff = std::fabs(res_ae.scf.energy - res_pp.scf.energy);
  std::cout << "  |E_ae - E_pp| = " << diff << " Ha\n";

  // Trivial PP should match all-electron closely.
  if (diff > 0.05) {
    return Fail("He PP energy differs from all-electron by " +
                std::to_string(diff) + " > 0.05 Ha");
  }

  // Sanity: He atom energy should be negative (bound state).
  if (res_pp.scf.energy > 0.0)
    return Fail("He PP energy should be negative, got " +
                std::to_string(res_pp.scf.energy));

  std::cout << "  PASS (PP matches all-electron, diff=" << diff << ")\n";
  return 0;
}

}  // namespace

int main() {
  std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
            << "║  PP SCF Validation Tests — Real Pseudopotential SCF         ║\n"
            << "╚══════════════════════════════════════════════════════════════╝\n";

  int failures = 0;
  failures += TestHAtomWithPP();
  failures += TestHeAtomWithPP();

  std::cout << "\n=== Summary ===\n";
  if (failures == 0) {
    std::cout << "ALL PP SCF VALIDATION TESTS PASSED\n";
  } else {
    std::cout << failures << " TEST(S) FAILED\n";
  }
  return failures;
}
