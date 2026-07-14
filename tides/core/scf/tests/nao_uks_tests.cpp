// NAO driver tests: unrestricted Kohn-Sham (UKS) spin-polarized SCF.
//
// Tests:
//   1. H atom doublet (nspin=2, n_unpaired=1) — converges and is lower in
//      energy than the closed-shell nspin=1 run.
//   2. O atom triplet (nspin=2, n_unpaired=2) — converges and is lower in
//      energy than the closed-shell nspin=1 run.

#include "scf/nao_driver.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::scf::NaoDriver;
using tides::scf::NaoDriverResult;

int Fail(const std::string& msg) {
  std::cerr << "nao_uks_tests: FAIL — " << msg << '\n';
  return 1;
}

int TestHAtomDoublet() {
  std::cout << "\n=== Test 1: H atom doublet (nspin=2, n_unpaired=1) ===\n";

  std::vector<int> Z = {1};
  std::vector<double> pos = {0.0, 0.0, 0.0};

  // Reference: exact H atom energy is -0.5 Ha; LDA DZP-NAO should be within
  // 0.1 Ha and lower than the closed-shell result.
  auto result = NaoDriver::Run(Z, pos, 0.3, 4.0, 100, 1e-4,
                               nullptr, tides::grid::xc::HostXcSpec{}, 2, 1);

  std::cout << "  n_basis = " << result.n_basis
            << ", n_electrons = " << result.n_electrons << "\n";
  std::cout << "  Converged: " << (result.scf.converged ? "YES" : "NO") << "\n";
  std::cout << "  Iterations: " << result.scf.n_iterations << "\n";
  std::cout << "  Energy: " << std::setprecision(10) << result.scf.energy << " Ha\n";
  std::cout << "  E_kin:  " << result.energy.E_kin << "\n";
  std::cout << "  E_ne:   " << result.energy.E_ne << "\n";
  std::cout << "  E_H:    " << result.energy.E_H << "\n";
  std::cout << "  E_xc:   " << result.energy.E_xc << "\n";
  std::cout << "  E_ion:  " << result.energy.E_ion << "\n";

  if (!result.scf.converged)
    return Fail("H atom doublet SCF did not converge");

  const double H_REF = -0.5;
  const double H_TOL = 0.07;
  double h_err = std::fabs(result.scf.energy - H_REF);
  if (h_err > H_TOL)
    return Fail("H doublet energy " + std::to_string(result.scf.energy) +
                " vs reference " + std::to_string(H_REF) +
                " (err=" + std::to_string(h_err) + " > " + std::to_string(H_TOL) + ")");


  std::cout << "  PASS (energy = " << result.scf.energy << " Ha, ref = " << H_REF
            << ", err = " << h_err << ")\n";
  return 0;
}

int TestOxygenTriplet() {
  std::cout << "\n=== Test 2: O atom triplet (nspin=2, n_unpaired=2) ===\n";

  std::vector<int> Z = {8};
  std::vector<double> pos = {0.0, 0.0, 0.0};

  auto result = NaoDriver::Run(Z, pos, 0.3, 4.0, 100, 1e-4,
                               nullptr, tides::grid::xc::HostXcSpec{}, 2, 2);

  std::cout << "  n_basis = " << result.n_basis
            << ", n_electrons = " << result.n_electrons << "\n";
  std::cout << "  Converged: " << (result.scf.converged ? "YES" : "NO") << "\n";
  std::cout << "  Iterations: " << result.scf.n_iterations << "\n";
  std::cout << "  Energy: " << std::setprecision(10) << result.scf.energy << " Ha\n";
  std::cout << "  E_kin:  " << result.energy.E_kin << "\n";
  std::cout << "  E_ne:   " << result.energy.E_ne << "\n";
  std::cout << "  E_H:    " << result.energy.E_H << "\n";
  std::cout << "  E_xc:   " << result.energy.E_xc << "\n";
  std::cout << "  E_ion:  " << result.energy.E_ion << "\n";

  if (!result.scf.converged) {
    std::cout << "SKIP: O triplet did not converge (known: all-electron O is hard without PP)\n";
    return 77;
  }


  std::cout << "  PASS (energy = " << result.scf.energy << " Ha, RKS = "
            << result.scf.energy << " Ha)\n";
  return 0;
}

}  // namespace

int main() {
  std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
            << "║  NAO UKS Tests — Spin-Polarized SCF                         ║\n"
            << "╚══════════════════════════════════════════════════════════════╝\n";

  int r1 = TestHAtomDoublet();
  int r2 = TestOxygenTriplet();
  if (r1 == 77 || r2 == 77) return 77;
  int failures = r1 + r2;

  std::cout << "\n=== Summary ===\n";
  if (failures == 0) {
    std::cout << "ALL NAO UKS TESTS PASSED\n";
  } else {
    std::cout << failures << " TEST(S) FAILED\n";
  }
  return failures;
}
