// Mermin finite-temperature DFT tests.
#include "scf/mermin.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {
using tides::scf::MerminDFT;
using tides::scf::MerminResult;

int Fail(const std::string& msg) {
  std::cerr << "mermin_tests: FAIL — " << msg << '\n';
  return 1;
}

int TestFermiLevelSearch() {
  std::cout << "\n=== Mermin: Fermi Level Search ===\n";
  // 4 orbitals, 2 electrons.
  std::vector<double> evals = {-2.0, -1.0, 0.0, 1.0};
  double n_electrons = 2.0;
  double kT = 0.05;  // Hartree

  double mu = MerminDFT::FindFermiLevel(evals, n_electrons, kT);
  std::cout << "  Fermi level: " << mu << " (expected ~ -0.5)\n";

  // Check that occupations sum to n_electrons.
  auto res = MerminDFT::Compute(evals, n_electrons, kT);
  std::cout << "  Occupations: [";
  for (double f : res.occupations) std::cout << f << " ";
  std::cout << "]\n";
  std::cout << "  Sum of occupations: " << res.nelectrons_check
            << " (expected " << n_electrons << ")\n";

  if (std::fabs(res.nelectrons_check - n_electrons) > 1e-8)
    return Fail("Occupation sum mismatch: " +
                std::to_string(res.nelectrons_check) + " vs " +
                std::to_string(n_electrons));
  // Fermi level should be between HOMO (-1.0) and LUMO (0.0).
  if (mu < -1.5 || mu > 0.5)
    return Fail("Fermi level out of expected range: " + std::to_string(mu));
  std::cout << "  PASS\n";
  return 0;
}

int TestT0Limit() {
  std::cout << "\n=== Mermin: T=0 Limit ===\n";
  std::vector<double> evals = {-2.0, -1.0, 0.0, 1.0};
  double n_electrons = 2.0;
  double kT = 0.0;  // zero temperature

  auto res = MerminDFT::Compute(evals, n_electrons, kT);
  std::cout << "  Occupations: [";
  for (double f : res.occupations) std::cout << f << " ";
  std::cout << "]\n";
  std::cout << "  Entropy: " << res.electronic_entropy << "\n";

  // At T=0, first 2 orbitals fully occupied, rest empty.
  if (std::fabs(res.occupations[0] - 1.0) > 1e-12 ||
      std::fabs(res.occupations[1] - 1.0) > 1e-12 ||
      std::fabs(res.occupations[2] - 0.0) > 1e-12 ||
      std::fabs(res.occupations[3] - 0.0) > 1e-12)
    return Fail("T=0 occupations should be [1, 1, 0, 0]");
  // Entropy should be zero at T=0.
  if (std::fabs(res.electronic_entropy) > 1e-12)
    return Fail("Entropy should be 0 at T=0");
  std::cout << "  PASS\n";
  return 0;
}

int TestEntropy() {
  std::cout << "\n=== Mermin: Entropy at Finite T ===\n";
  std::vector<double> evals = {-1.0, 0.0};
  double kT = 0.1;

  auto res = MerminDFT::Compute(evals, 1.0, kT);
  std::cout << "  Occupations: [" << res.occupations[0] << ", "
            << res.occupations[1] << "]\n";
  std::cout << "  Entropy: " << res.electronic_entropy << "\n";

  // Entropy should be positive (thermal excitations increase disorder).
  if (res.electronic_entropy < 0.0)
    return Fail("Entropy should be non-negative");
  // At half-filling with kT comparable to gap, entropy should be ~ln(2).
  if (res.electronic_entropy > 1.0)
    return Fail("Entropy unexpectedly large: " +
                std::to_string(res.electronic_entropy));
  std::cout << "  PASS\n";
  return 0;
}

int TestMerminCorrectedEnergy() {
  std::cout << "\n=== Mermin: Corrected Energy ===\n";
  std::vector<double> evals = {-2.0, -1.0, 0.0, 1.0};
  double kT = 0.05;

  auto res = MerminDFT::Compute(evals, 2.0, kT);
  double E_0K = -3.0;  // ground state energy
  double E_T = MerminDFT::MerminCorrectedEnergy(E_0K, res, kT);
  std::cout << "  E(0K): " << E_0K << "\n";
  std::cout << "  E(T): " << E_T << "\n";
  std::cout << "  -kT*S: " << -kT * res.electronic_entropy << "\n";

  // Corrected energy should be lower (entropy stabilization).
  if (E_T > E_0K)
    return Fail("Mermin free energy should be <= E_0K");
  std::cout << "  PASS\n";
  return 0;
}

}  // namespace

int main() {
  std::cout << "=== Mermin Finite-Temperature Tests ===\n";
  int failures = 0;
  failures += TestFermiLevelSearch();
  failures += TestT0Limit();
  failures += TestEntropy();
  failures += TestMerminCorrectedEnergy();
  if (failures == 0) std::cout << "\nALL MERMIN TESTS PASSED\n";
  else std::cout << "\n" << failures << " TEST(S) FAILED\n";
  return failures;
}
