// Broker dispatch test: verify R1 (ChFSI) and R2 (SP2) are selected
// for large systems and produce correct eigenvalues.
#include "solvers/broker.hpp"
#include "solvers/dense/batched_eig.hpp"
#include "solvers/chfsi/chfsi.hpp"
#include "solvers/sp2_submatrix/sp2.hpp"
#include "solvers/foe_sq/foe.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {
using tides::solvers::SolverBroker;
using tides::solvers::SolverRegime;
using tides::solvers::BrokerInput;
using tides::solvers::BrokerRunner;
using tides::solvers::BrokerSolveResult;

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

// Compute ||P*P - P||_F (Frobenius norm of idempotency residual).
double IdempotencyError(std::size_t n, const std::vector<double>& P) {
  double err = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double pp = 0.0;
      for (std::size_t k = 0; k < n; ++k)
        pp += P[i * n + k] * P[k * n + j];
      double diff = pp - P[i * n + j];
      err += diff * diff;
    }
  }
  return std::sqrt(err);
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

  // Test R1 (medium system with gap, n_basis > 200)
  {
    BrokerInput input;
    input.n_atoms = 50;
    input.n_basis = 256;  // > 200 triggers R1
    input.gap_estimate = 3.0;
    input.electronic_temp = 0.0;
    auto calib = SolverBroker::GenerateCalibTable();
    std::string reason;
    auto regime = SolverBroker::Dispatch(input, calib, reason);
    std::cout << "  Medium system (n_basis=256): regime=" << static_cast<int>(regime)
              << " reason=" << reason << "\n";
    if (regime != SolverRegime::kR1_ChFSI)
      return Fail("Expected R1 for medium system (n_basis=256)");
  }

  // Test R2 (large system, n_basis > 2000)
  {
    BrokerInput input;
    input.n_atoms = 500;
    input.n_basis = 3000;  // > 2000 triggers R2
    input.gap_estimate = 2.0;
    input.electronic_temp = 0.0;
    auto calib = SolverBroker::GenerateCalibTable();
    std::string reason;
    auto regime = SolverBroker::Dispatch(input, calib, reason);
    std::cout << "  Large system (n_basis=3000): regime=" << static_cast<int>(regime)
              << " reason=" << reason << "\n";
    if (regime != SolverRegime::kR2_SP2)
      return Fail("Expected R2 for large system (n_basis=3000)");
  }

  // Test R3 (metallic large system)
  {
    BrokerInput input;
    input.n_atoms = 500;
    input.n_basis = 1000;
    input.gap_estimate = 0.01;  // metallic (gap < 0.1 eV)
    input.electronic_temp = 0.0;
    auto calib = SolverBroker::GenerateCalibTable();
    std::string reason;
    auto regime = SolverBroker::Dispatch(input, calib, reason);
    std::cout << "  Metallic system (n_basis=1000, gap=0.01): regime=" << static_cast<int>(regime)
              << " reason=" << reason << "\n";
    if (regime != SolverRegime::kR3_FOE_SQ)
      return Fail("Expected R3 for metallic large system");
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

  // Test BrokerRunner::Solve for R0 (small system) — verify P is idempotent: P^2 ≈ P.
  {
    std::cout << "\n  --- BrokerRunner R0 execution test (idempotency) ---\n";
    const std::size_t n = 12, n_occ = 3;
    std::vector<double> lam(n);
    for (std::size_t i = 0; i < n; ++i) lam[i] = -5.0 + 0.3 * static_cast<double>(i);
    std::vector<double> H, Q;
    BuildFromSpectrum(n, lam, 123, H, Q);
    std::vector<double> S(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) S[i * n + i] = 1.0;

    BrokerInput input;
    input.n_atoms = 3;
    input.n_basis = n;  // small → R0
    input.gap_estimate = 5.0;
    input.electronic_temp = 0.0;

    BrokerSolveResult res = BrokerRunner::Solve(input, n, n_occ, H, S, 1e-12);
    std::cout << "    regime_used=R" << static_cast<int>(res.regime_used)
              << " converged=" << res.converged
              << " reason=" << res.reason << "\n";

    if (res.regime_used != SolverRegime::kR0_BatchDense)
      return Fail("BrokerRunner expected R0 for small system");

    if (!res.converged)
      return Fail("BrokerRunner R0 did not converge");

    if (res.P.size() != n * n)
      return Fail("BrokerRunner R0 produced wrong P size");

    // Check idempotency: P^2 ≈ P (P is a projector onto the occupied subspace).
    double idem_err = IdempotencyError(n, res.P);
    std::cout << "    ||P^2 - P||_F = " << idem_err << "\n";
    if (idem_err > 1e-9)
      return Fail("BrokerRunner R0 P not idempotent (||P^2-P||_F=" +
                  std::to_string(idem_err) + ")");

    // Check trace(P) = n_occ (for S=I, tr(P) = number of occupied orbitals).
    double tr = 0.0;
    for (std::size_t i = 0; i < n; ++i) tr += res.P[i * n + i];
    std::cout << "    tr(P) = " << tr << " (expected " << n_occ << ")\n";
    if (std::fabs(tr - static_cast<double>(n_occ)) > 1e-9)
      return Fail("BrokerRunner R0 tr(P) mismatch (got " +
                  std::to_string(tr) + ", expected " +
                  std::to_string(n_occ) + ")");

    // Check eigenvalues against the reference.
    auto ref = tides::solvers::BatchedDenseEig::SolveGeneralized(n, H, S);
    if (!ref.ok) return Fail("Dense reference failed for BrokerRunner test");
    for (std::size_t k = 0; k < n_occ; ++k) {
      double err = std::fabs(res.eigenvalues[k] - ref.eigenvalues[k]);
      std::cout << "    k=" << k << " eig=" << res.eigenvalues[k]
                << " ref=" << ref.eigenvalues[k] << " err=" << err << "\n";
      if (err > 1e-9)
        return Fail("BrokerRunner R0 eigenvalue mismatch at k=" + std::to_string(k));
    }

    std::cout << "    PASS (R0 idempotency + trace + eigenvalues)\n";
  }

  // Test BrokerRunner::Solve for R2 (SP2) — verify P is approximately idempotent.
  {
    std::cout << "\n  --- BrokerRunner R2 execution test (SP2 idempotency) ---\n";
    const std::size_t n = 16, n_occ = 4;
    std::vector<double> lam(n);
    for (std::size_t i = 0; i < n; ++i) lam[i] = -5.0 + 0.3 * static_cast<double>(i);
    std::vector<double> H, Q;
    BuildFromSpectrum(n, lam, 777, H, Q);
    std::vector<double> S(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) S[i * n + i] = 1.0;

    BrokerInput input;
    input.n_atoms = 500;
    input.n_basis = 3000;  // large → R2
    input.gap_estimate = 2.0;
    input.electronic_temp = 0.0;
    input.user_override = true;
    input.forced_regime = SolverRegime::kR2_SP2;

    BrokerSolveResult res = BrokerRunner::Solve(input, n, n_occ, H, S, 1e-10);
    std::cout << "    regime_used=R" << static_cast<int>(res.regime_used)
              << " converged=" << res.converged
              << " reason=" << res.reason << "\n";

    if (res.regime_used != SolverRegime::kR2_SP2)
      return Fail("BrokerRunner expected R2 for forced regime");

    if (res.P.size() != n * n)
      return Fail("BrokerRunner R2 produced wrong P size");

    // SP2 should produce an approximately idempotent P (||P^2 - P|| small).
    double idem_err = IdempotencyError(n, res.P);
    std::cout << "    ||P^2 - P||_F = " << idem_err << "\n";
    if (idem_err > 1e-5)
      return Fail("BrokerRunner R2 P not approximately idempotent (||P^2-P||_F=" +
                  std::to_string(idem_err) + ")");

    std::cout << "    PASS (R2 SP2 approximate idempotency)\n";
  }

  std::cout << "  PASS (broker dispatch + ChFSI validation + R0 execution)\n";
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
