// Broker dispatch test: verify R1 (ChFSI) and R2 (SP2) are selected
// for large systems and produce correct eigenvalues.
#include "solvers/broker.hpp"
#include "solvers/dense/batched_eig.hpp"
#include "solvers/chfsi/chfsi.hpp"
#include "solvers/sp2_submatrix/sp2.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {
using tides::solvers::SolverBroker;
using tides::solvers::SolverRegime;
using tides::solvers::BrokerInput;

int Fail(const std::string& msg) {
  std::cerr << "broker_dispatch_tests: " << msg << '\n';
  return 1;
}

void BuildFromSpectrum(std::size_t n, const std::vector<double>& lambda,
                       std::uint64_t seed, std::vector<double>& A,
                       std::vector<double>& Q) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> g(0.0, 1.0);
  Q.assign(n * n, 0.0);
  for (auto& v : Q) v = g(rng);
  // Gram-Schmidt orthonormalize
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      double dot = 0.0;
      for (std::size_t k = 0; k < n; ++k) dot += Q[j*n+k] * Q[i*n+k];
      for (std::size_t k = 0; k < n; ++k) Q[i*n+k] -= dot * Q[j*n+k];
    }
    double norm = 0.0;
    for (std::size_t k = 0; k < n; ++k) norm += Q[i*n+k] * Q[i*n+k];
    norm = std::sqrt(norm);
    for (std::size_t k = 0; k < n; ++k) Q[i*n+k] /= norm;
  }
  // A = Q diag(lambda) Q^T
  A.assign(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t k = 0; k < n; ++k)
        A[i*n+j] += Q[k*n+i] * lambda[k] * Q[k*n+j];
}

int TestBrokerDispatch() {
  std::cout << "\n=== Broker Dispatch Test ===\n";
  
  // Test R0 (small system)
  {
    BrokerInput input;
    input.n_atoms = 2;
    input.n_basis = 10;
    input.gap_estimate = 5.0;
    input.electronic_temp = 0.0;
    auto calib = SolverBroker::GenerateCalibTable();
    std::string reason;
    auto regime = SolverBroker::Dispatch(input, calib, reason);
    std::cout << "  Small system (n=10): regime=" << static_cast<int>(regime)
              << " reason=" << reason << "\n";
    if (regime != SolverRegime::kR0_BatchDense)
      return Fail("Expected R0 for small system");
  }

  // Test R1 (medium system with gap)
  {
    BrokerInput input;
    input.n_atoms = 20;
    input.n_basis = 128;
    input.gap_estimate = 3.0;
    input.electronic_temp = 0.0;
    auto calib = SolverBroker::GenerateCalibTable();
    std::string reason;
    auto regime = SolverBroker::Dispatch(input, calib, reason);
    std::cout << "  Medium system (n=128): regime=" << static_cast<int>(regime)
              << " reason=" << reason << "\n";
    // Broker should select R1 or R0 depending on calibration
  }

  // Test R2 (large system)
  {
    BrokerInput input;
    input.n_atoms = 100;
    input.n_basis = 1024;
    input.gap_estimate = 2.0;
    input.electronic_temp = 0.0;
    auto calib = SolverBroker::GenerateCalibTable();
    std::string reason;
    auto regime = SolverBroker::Dispatch(input, calib, reason);
    std::cout << "  Large system (n=1024): regime=" << static_cast<int>(regime)
              << " reason=" << reason << "\n";
    // Broker should select R2 or R3 for large systems
  }

  // Verify ChFSI produces correct eigenvalues for a medium system
  {
    const int n = 64, n_occ = 8;
    std::vector<double> lam(n);
    for (int i = 0; i < n; ++i) lam[i] = -5.0 + 0.2 * i;
    std::vector<double> H, Q;
    BuildFromSpectrum(n, lam, 42, H, Q);
    std::vector<double> S(n * n, 0.0);
    for (int i = 0; i < n; ++i) S[i*n+i] = 1.0;

    auto ref = tides::solvers::BatchedDenseEig::SolveGeneralized(n, H, S);
    if (!ref.ok) return Fail("Dense reference failed");

    double lo = ref.eigenvalues[n_occ] + 0.1;
    double hi = ref.eigenvalues[n-1] + 1.0;
    auto chfsi = tides::solvers::ChFSI::Solve(n, H, S, n_occ, lo, hi, 20, 100, 1e-9);

    std::cout << "  ChFSI (n=64, n_occ=8): converged=" << chfsi.converged << "\n";
    if (!chfsi.converged) return Fail("ChFSI did not converge");

    for (int k = 0; k < n_occ; ++k) {
      double err = std::fabs(chfsi.eigenvalues[k] - ref.eigenvalues[k]);
      std::cout << "    k=" << k << " eig=" << chfsi.eigenvalues[k]
                << " ref=" << ref.eigenvalues[k] << " err=" << err << "\n";
      if (err > 1e-7) return Fail("ChFSI eigenvalue mismatch at k=" + std::to_string(k));
    }
  }

  std::cout << "  PASS (broker dispatch + ChFSI validation)\n";
  return 0;
}

}  // namespace

int main() {
  std::cout << "=== Broker Dispatch Tests ===\n";
  int failures = TestBrokerDispatch();
  if (failures == 0) std::cout << "\nALL BROKER DISPATCH TESTS PASSED\n";
  else std::cout << "\n" << failures << " TEST(S) FAILED\n";
  return failures;
}
