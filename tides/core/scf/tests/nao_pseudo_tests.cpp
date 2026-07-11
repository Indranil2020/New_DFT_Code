// Pseudopotential SCF tests: validate NaoDriver with pseudopotentials.
//
// Tests:
//   1. H atom with trivial PP (Z_val=1, v_local=-1/r, no KB) — matches all-electron
//   2. H2 with trivial PP — matches all-electron
//
// A "trivial" pseudopotential has Z_valence = Z (no core frozen),
// v_local(r) = -Z/r (same as all-electron nuclear attraction),
// and no KB channels (no nonlocal projectors). This should give
// identical results to the all-electron calculation.

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
  std::cerr << "nao_pseudo_tests: FAIL — " << msg << '\n';
  return 1;
}

// Create a trivial pseudopotential: v_local = -Z/r, no KB channels.
// This should give the same result as all-electron.
Pseudopotential MakeTrivialPP(int Z) {
  Pseudopotential pp;
  pp.Z_valence = Z;
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
  return pp;
}

int TestHAtomWithPP() {
  std::cout << "\n=== Test 1: H atom with trivial PP ===\n";

  std::vector<int> Z = {1};
  std::vector<double> pos = {0.0, 0.0, 0.0};

  // Run without PP (all-electron)
  auto res_ae = NaoDriver::Run(Z, pos, 0.2, 4.0, 100, 1e-6);
  std::cout << "  All-electron: E=" << res_ae.scf.energy
            << " converged=" << res_ae.scf.converged << "\n";

  // Run with trivial PP (should match all-electron)
  auto pp = MakeTrivialPP(1);
  std::vector<Pseudopotential> pps = {pp};
  auto res_pp = NaoDriver::Run(Z, pos, 0.2, 4.0, 100, 1e-6, &pps);
  std::cout << "  With PP:      E=" << res_pp.scf.energy
            << " converged=" << res_pp.scf.converged << "\n";

  if (!res_ae.scf.converged) return Fail("H all-electron SCF did not converge");
  if (!res_pp.scf.converged) return Fail("H PP SCF did not converge");

  double diff = std::fabs(res_ae.scf.energy - res_pp.scf.energy);
  std::cout << "  |E_ae - E_pp| = " << diff << " Ha\n";

  if (diff > 0.05) {
    return Fail("H PP energy differs from all-electron by " +
                std::to_string(diff) + " > 0.05 Ha");
  }

  std::cout << "  PASS (PP matches all-electron, diff=" << diff << ")\n";
  return 0;
}

int TestH2WithPP() {
  std::cout << "\n=== Test 2: H2 with trivial PP ===\n";

  std::vector<int> Z = {1, 1};
  std::vector<double> pos = {0.0, 0.0, 0.0, 1.4, 0.0, 0.0};

  // All-electron
  auto res_ae = NaoDriver::Run(Z, pos, 0.2, 4.0, 100, 1e-6);
  std::cout << "  All-electron: E=" << res_ae.scf.energy
            << " converged=" << res_ae.scf.converged << "\n";

  // With trivial PP
  auto pp = MakeTrivialPP(1);
  std::vector<Pseudopotential> pps = {pp, pp};
  auto res_pp = NaoDriver::Run(Z, pos, 0.2, 4.0, 100, 1e-6, &pps);
  std::cout << "  With PP:      E=" << res_pp.scf.energy
            << " converged=" << res_pp.scf.converged << "\n";

  if (!res_ae.scf.converged) return Fail("H2 all-electron SCF did not converge");
  if (!res_pp.scf.converged) return Fail("H2 PP SCF did not converge");

  double diff = std::fabs(res_ae.scf.energy - res_pp.scf.energy);
  std::cout << "  |E_ae - E_pp| = " << diff << " Ha\n";

  if (diff > 0.1) {
    return Fail("H2 PP energy differs from all-electron by " +
                std::to_string(diff) + " > 0.1 Ha");
  }

  std::cout << "  PASS (PP matches all-electron, diff=" << diff << ")\n";
  return 0;
}

}  // namespace

int main() {
  std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
            << "║  NAO Pseudopotential Tests — PP SCF Validation              ║\n"
            << "╚══════════════════════════════════════════════════════════════╝\n";

  int failures = 0;
  failures += TestHAtomWithPP();
  failures += TestH2WithPP();

  std::cout << "\n=== Summary ===\n";
  if (failures == 0) {
    std::cout << "ALL PSEUDOPOTENTIAL TESTS PASSED\n";
  } else {
    std::cout << failures << " TEST(S) FAILED\n";
  }
  return failures;
}
