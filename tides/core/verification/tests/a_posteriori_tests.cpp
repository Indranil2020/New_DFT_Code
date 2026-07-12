// A-posteriori error control tests.
#include "verification/a_posteriori_error.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {
using tides::verification::APosterioriErrorControl;
using tides::verification::ErrorBounds;

int Fail(const std::string& msg) {
  std::cerr << "a_posteriori_tests: FAIL — " << msg << '\n';
  return 1;
}

int TestExactDiagonalization() {
  std::cout << "\n=== Error Control: Exact Diagonalization (zero residual) ===\n";
  // A diagonal Hamiltonian with exact eigenpairs → zero commutator.
  std::size_t n = 4;
  std::vector<double> H(n * n, 0.0), S(n * n, 0.0), P(n * n, 0.0);
  std::vector<double> evals = {-2.0, -1.0, 0.0, 1.0};
  std::vector<double> evecs(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    H[i * n + i] = evals[i];
    S[i * n + i] = 1.0;
    evecs[i * n + i] = 1.0;  // identity eigenvectors
    P[i * n + i] = 1.0;      // fully occupied
  }

  auto bounds = APosterioriErrorControl::Compute(n, 2, H, S, P, evals, evecs);
  std::cout << "  SCF residual: " << bounds.scf_residual_norm << "\n";
  std::cout << "  Energy bound: " << bounds.energy_error_bound << "\n";
  std::cout << "  Eigenvalue residuals: [";
  for (double r : bounds.eigenvalue_residuals) std::cout << r << " ";
  std::cout << "]\n";

  if (bounds.scf_residual_norm > 1e-12)
    return Fail("SCF residual should be ~0 for exact diagonal H");
  if (bounds.energy_error_bound > 1e-12)
    return Fail("Energy bound should be ~0 for exact diagonal H");
  std::cout << "  PASS\n";
  return 0;
}

int TestPerturbedSystem() {
  std::cout << "\n=== Error Control: Perturbed System (nonzero residual) ===\n";
  std::size_t n = 4;
  // Off-diagonal H → nonzero commutator with diagonal P.
  std::vector<double> H = {
    -2, 0.1, 0, 0,
    0.1, -1, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 1
  };
  std::vector<double> S(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) S[i * n + i] = 1.0;
  std::vector<double> P(n * n, 0.0);
  // P that mixes blocks 0-2 → nonzero commutator with H.
  P[0] = 1.0; P[2] = 0.1; P[8] = 0.1;

  std::vector<double> evals = {-2.0, -1.0, 0.0, 1.0};  // approximate
  std::vector<double> evecs(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) evecs[i * n + i] = 1.0;

  auto bounds = APosterioriErrorControl::Compute(n, 2, H, S, P, evals, evecs);
  std::cout << "  SCF residual: " << bounds.scf_residual_norm << "\n";
  std::cout << "  Energy bound: " << bounds.energy_error_bound << "\n";
  std::cout << "  Force bound: " << bounds.force_error_bound << "\n";

  if (bounds.scf_residual_norm < 1e-10)
    return Fail("SCF residual should be nonzero for off-diagonal H with diagonal P");
  if (bounds.energy_error_bound <= 0.0)
    return Fail("Energy bound should be positive for perturbed system");
  std::cout << "  PASS\n";
  return 0;
}

int TestConvergenceCheck() {
  std::cout << "\n=== Error Control: Convergence Check ===\n";
  ErrorBounds bounds;
  bounds.energy_error_bound = 1e-10;
  if (!APosterioriErrorControl::EnergyConverged(bounds, 1e-8))
    return Fail("Should be converged at 1e-10 < 1e-8");

  bounds.energy_error_bound = 1e-4;
  if (APosterioriErrorControl::EnergyConverged(bounds, 1e-8))
    return Fail("Should NOT be converged at 1e-4 > 1e-8");
  std::cout << "  PASS\n";
  return 0;
}

int TestRecommendIterations() {
  std::cout << "\n=== Error Control: Recommend Iterations ===\n";
  std::vector<ErrorBounds> history;
  // Simulate geometric convergence: 1e-2, 1e-4, 1e-6.
  ErrorBounds b1; b1.scf_residual_norm = 1e-2;
  ErrorBounds b2; b2.scf_residual_norm = 1e-4;
  ErrorBounds b3; b3.scf_residual_norm = 1e-6;
  history = {b1, b2, b3};

  int n_rec = APosterioriErrorControl::RecommendIterations(history, 1e-10);
  std::cout << "  Current residual: 1e-6, target: 1e-10, recommended: " << n_rec << "\n";
  if (n_rec <= 0)
    return Fail("Should recommend >0 iterations to reach 1e-10 from 1e-6");
  std::cout << "  PASS\n";
  return 0;
}

}  // namespace

int main() {
  std::cout << "=== A-Posteriori Error Control Tests ===\n";
  int failures = 0;
  failures += TestExactDiagonalization();
  failures += TestPerturbedSystem();
  failures += TestConvergenceCheck();
  failures += TestRecommendIterations();
  if (failures == 0) std::cout << "\nALL A-POSTERIORI TESTS PASSED\n";
  else std::cout << "\n" << failures << " TEST(S) FAILED\n";
  return failures;
}
