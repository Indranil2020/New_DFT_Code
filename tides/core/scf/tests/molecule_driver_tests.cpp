// Molecule driver tests: end-to-end SCF on real molecules.
//
// Tests:
//   1. H2 (STO-3G) — 2 basis functions, 2 electrons
//   2. He (STO-3G) — 1 basis function, 2 electrons
//   3. H2O (STO-3G) — 7 basis functions, 10 electrons
//   4. Overlap matrix vs known values
//
// Reference energies (STO-3G, RKS-LDA):
//   H2:   ~-0.97 Ha (bond length 1.4 Bohr)
//   He:   ~-2.86 Ha
//   H2O:  ~-74.96 Ha (O-H = 1.81 Bohr, HOH = 104.5°)

#include "scf/molecule_driver.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using tides::scf::MoleculeDriver;
using tides::scf::MoleculeDriverResult;
using tides::scf::GTOMolecule;
using tides::scf::GTOIntegrals;

int Fail(const std::string& msg) {
  std::cerr << "molecule_driver_tests: FAIL — " << msg << '\n';
  return 1;
}

int TestOverlap() {
  std::cout << "\n=== Test 1: Overlap Matrix (H2 STO-3G) ===\n";

  // H2 at R = 1.4 Bohr.
  std::vector<int> Z = {1, 1};
  std::vector<double> pos = {0.0, 0.0, 0.0, 1.4, 0.0, 0.0};
  auto mol = MoleculeDriver::BuildMolecule(Z, pos);

  auto S = GTOIntegrals::Overlap(mol);

  std::cout << "  n_basis = " << mol.n_basis << "\n";
  std::cout << "  S = [[" << S[0] << ", " << S[1] << "],\n";
  std::cout << "       [" << S[2] << ", " << S[3] << "]]\n";

  // S should be ~[[1.0, 0.65], [0.65, 1.0]] for H2 at 1.4 Bohr.
  if (std::fabs(S[0] - 1.0) > 1e-6) return Fail("S[0,0] != 1.0");
  if (std::fabs(S[3] - 1.0) > 1e-6) return Fail("S[1,1] != 1.0");
  if (std::fabs(S[1] - S[2]) > 1e-10) return Fail("S not symmetric");
  if (S[1] < 0.5 || S[1] > 0.8) return Fail("S[0,1] out of expected range");

  std::cout << "  PASS (S[0,1] = " << S[1] << ")\n";
  return 0;
}

int TestH2() {
  std::cout << "\n=== Test 2: H2 SCF (STO-3G, R=1.4 Bohr) ===\n";

  std::vector<int> Z = {1, 1};
  std::vector<double> pos = {0.0, 0.0, 0.0, 1.4, 0.0, 0.0};
  auto mol = MoleculeDriver::BuildMolecule(Z, pos);

  std::cout << "  n_basis = " << mol.n_basis << ", n_electrons = 2\n";
  std::cout << "  Running SCF...\n";

  auto result = MoleculeDriver::Run(mol, 0.3, 4.0, 100, 1e-6);

  std::cout << "  Converged: " << (result.scf.converged ? "YES" : "NO") << "\n";
  std::cout << "  Iterations: " << result.scf.n_iterations << "\n";
  std::cout << "  Energy: " << std::setprecision(10) << result.scf.energy << " Ha\n";
  std::cout << "  E_kin:  " << result.energy.E_kin << "\n";
  std::cout << "  E_ne:   " << result.energy.E_ne << "\n";
  std::cout << "  E_H:    " << result.energy.E_H << "\n";
  std::cout << "  E_xc:   " << result.energy.E_xc << "\n";
  std::cout << "  E_ion:  " << result.energy.E_ion << "\n";
  std::cout << "  Wall time: " << result.wall_time_ms << " ms\n";
  std::cout << "  Grid: " << result.grid_n[0] << "x" << result.grid_n[1]
            << "x" << result.grid_n[2] << "\n";

  if (!result.scf.converged)
    return Fail("SCF did not converge");

  // H2 STO-3G LDA energy should be roughly -0.9 to -1.1 Ha.
  // The grid-based V_ext and V_H introduce some error vs analytic,
  // so we use a generous tolerance.
  if (result.scf.energy > -0.5 || result.scf.energy < -2.0)
    return Fail("H2 energy out of range: " + std::to_string(result.scf.energy));

  std::cout << "  PASS (energy = " << result.scf.energy << " Ha)\n";
  return 0;
}

int TestHe() {
  std::cout << "\n=== Test 3: He SCF (STO-3G) ===\n";

  std::vector<int> Z = {2};
  std::vector<double> pos = {0.0, 0.0, 0.0};
  auto mol = MoleculeDriver::BuildMolecule(Z, pos);

  std::cout << "  n_basis = " << mol.n_basis << ", n_electrons = 2\n";
  std::cout << "  Running SCF...\n";

  auto result = MoleculeDriver::Run(mol, 0.25, 4.0, 100, 1e-6);

  std::cout << "  Converged: " << (result.scf.converged ? "YES" : "NO") << "\n";
  std::cout << "  Iterations: " << result.scf.n_iterations << "\n";
  std::cout << "  Energy: " << std::setprecision(10) << result.scf.energy << " Ha\n";
  std::cout << "  Wall time: " << result.wall_time_ms << " ms\n";
  std::cout << "  Grid: " << result.grid_n[0] << "x" << result.grid_n[1]
            << "x" << result.grid_n[2] << "\n";

  if (!result.scf.converged)
    return Fail("SCF did not converge");

  // He STO-3G LDA energy should be roughly -2.7 to -3.0 Ha.
  if (result.scf.energy > -2.0 || result.scf.energy < -4.0)
    return Fail("He energy out of range: " + std::to_string(result.scf.energy));

  std::cout << "  PASS (energy = " << result.scf.energy << " Ha)\n";
  return 0;
}

int TestH2O() {
  std::cout << "\n=== Test 4: H2O SCF (STO-3G) ===\n";

  // H2O geometry: O at origin, H atoms at ~1.81 Bohr, 104.5° angle.
  const double r_OH = 1.81;  // Bohr
  const double angle = 104.5 * M_PI / 180.0;
  const double half_angle = angle / 2.0;

  std::vector<int> Z = {8, 1, 1};
  std::vector<double> pos = {
    0.0, 0.0, 0.0,                                    // O
    r_OH * std::sin(half_angle), 0.0, r_OH * std::cos(half_angle),  // H1
    -r_OH * std::sin(half_angle), 0.0, r_OH * std::cos(half_angle), // H2
  };
  auto mol = MoleculeDriver::BuildMolecule(Z, pos);

  std::cout << "  n_basis = " << mol.n_basis << ", n_electrons = 10\n";
  std::cout << "  Running SCF (this may take a moment)...\n";

  auto result = MoleculeDriver::Run(mol, 0.3, 4.0, 150, 1e-6);

  std::cout << "  Converged: " << (result.scf.converged ? "YES" : "NO") << "\n";
  std::cout << "  Iterations: " << result.scf.n_iterations << "\n";
  std::cout << "  Energy: " << std::setprecision(10) << result.scf.energy << " Ha\n";
  std::cout << "  E_kin:  " << result.energy.E_kin << "\n";
  std::cout << "  E_ne:   " << result.energy.E_ne << "\n";
  std::cout << "  E_H:    " << result.energy.E_H << "\n";
  std::cout << "  E_xc:   " << result.energy.E_xc << "\n";
  std::cout << "  E_ion:  " << result.energy.E_ion << "\n";
  std::cout << "  Wall time: " << result.wall_time_ms << " ms\n";
  std::cout << "  Grid: " << result.grid_n[0] << "x" << result.grid_n[1]
            << "x" << result.grid_n[2] << "\n";

  if (!result.scf.converged)
    return Fail("SCF did not converge for H2O");

  // H2O STO-3G LDA energy should be roughly -70 to -78 Ha.
  // Grid-based integrals introduce error, so generous tolerance.
  if (result.scf.energy > -60.0 || result.scf.energy < -85.0)
    return Fail("H2O energy out of range: " + std::to_string(result.scf.energy));

  std::cout << "  PASS (energy = " << result.scf.energy << " Ha)\n";
  return 0;
}

}  // namespace

int main() {
  std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
            << "║  Molecule Driver Tests — End-to-End SCF                     ║\n"
            << "╚══════════════════════════════════════════════════════════════╝\n";

  int failures = 0;
  failures += TestOverlap();
  failures += TestH2();
  failures += TestHe();
  // H2O is slower (larger grid); run it but don't fail the whole suite
  // if the grid is too coarse for good accuracy.
  failures += TestH2O();

  std::cout << "\n=== Summary ===\n";
  if (failures == 0) {
    std::cout << "ALL MOLECULE DRIVER TESTS PASSED\n";
  } else {
    std::cout << failures << " TEST(S) FAILED\n";
  }
  return failures;
}
