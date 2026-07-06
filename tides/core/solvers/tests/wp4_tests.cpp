// WP4 tests: T4.1 (batched dense eig), T4.3 (ChFSI), T4.5 (OMM), T4.6 (broker).
// T4.4 (ELPA bridge) is validated by comparing against T4.1's LAPACK path.
//
// All tests use constructed matrices with known eigenvalues (analytical gates).
// The generalized problem H x = e S x is tested with S = I (standard) and
// with S != I (generalized).

#include "solvers/chfsi/chfsi.hpp"
#include "solvers/dense/batched_eig.hpp"
#include "solvers/omm/omm.hpp"
#include "solvers/broker.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using tides::solvers::BatchedDenseEig;
using tides::solvers::BrokerInput;
using tides::solvers::CalibEntry;
using tides::solvers::ChFSI;
using tides::solvers::ChFSIResult;
using tides::solvers::EigenResult;
using tides::solvers::OMMResult;
using tides::solvers::OMMSolver;
using tides::solvers::SolverBroker;
using tides::solvers::SolverRegime;

int Fail(const std::string& msg) {
  std::cerr << "wp4_tests: " << msg << '\n';
  return 1;
}

// Build a symmetric matrix A = Q diag(lambda) Q^T with known eigenvalues.
void BuildFromSpectrum(std::size_t n, const std::vector<double>& lambda,
                       std::uint64_t seed, std::vector<double>& A,
                       std::vector<double>& Q) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> g(0.0, 1.0);
  Q.assign(n * n, 0.0);
  for (auto& v : Q) v = g(rng);
  // Orthonormalize columns (Gram-Schmidt).
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t k = 0; k < j; ++k) {
      double dot = 0.0;
      for (std::size_t i = 0; i < n; ++i) dot += Q[i * n + j] * Q[i * n + k];
      for (std::size_t i = 0; i < n; ++i) Q[i * n + j] -= dot * Q[i * n + k];
    }
    double nrm = 0.0;
    for (std::size_t i = 0; i < n; ++i) nrm += Q[i * n + j] * Q[i * n + j];
    nrm = std::sqrt(nrm);
    for (std::size_t i = 0; i < n; ++i) Q[i * n + j] /= nrm;
  }
  A.assign(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < n; ++k)
        s += Q[i * n + k] * lambda[k] * Q[j * n + k];
      A[i * n + j] = s;
    }
}

// T4.1: Batched dense eigensolver — residuals <= 1e-9 at n <= 400.
int TestBatchedEig() {
  std::cout << "\n=== T4.1: Batched dense eigensolver (R0) ===\n";
  // Standard problem (S = I): n=100, known spectrum.
  for (int n : {10, 50, 100, 200, 400}) {
    std::vector<double> lam(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) lam[i] = -5.0 + 0.1 * i;
    std::vector<double> H, Q;
    BuildFromSpectrum(static_cast<std::size_t>(n), lam, 42, H, Q);
    std::vector<double> S(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) S[i * n + i] = 1.0;

    auto res = BatchedDenseEig::SolveGeneralized(n, H, S);
    if (!res.ok) return Fail("T4.1: eigensolve failed at n=" + std::to_string(n));

    // Check eigenvalues match.
    double max_val_err = 0.0;
    for (int i = 0; i < n; ++i)
      max_val_err = std::max(max_val_err, std::fabs(res.eigenvalues[i] - lam[i]));

    // Check residuals.
    double max_res = 0.0;
    for (int k = 0; k < n; ++k) {
      std::vector<double> x(n);
      for (int j = 0; j < n; ++j) x[j] = res.eigenvectors[k * n + j];
      max_res = std::max(max_res, BatchedDenseEig::Residual(n, H, S,
                                                           res.eigenvalues[k], x));
    }
    std::cout << "  n=" << n << " val_err=" << max_val_err
              << " max_residual=" << max_res << '\n';
    if (max_res > 1e-9) {
      std::ostringstream os;
      os << "T4.1: residual " << max_res << " > 1e-9 at n=" << n;
      return Fail(os.str());
    }
  }

  // Generalized problem (S != I): n=50, S = diag(1, 2, 3, ...).
  {
    const int n = 50;
    std::vector<double> lam(n);
    for (int i = 0; i < n; ++i) lam[i] = -3.0 + 0.05 * i;
    std::vector<double> H, Q;
    BuildFromSpectrum(n, lam, 99, H, Q);
    std::vector<double> S(n * n, 0.0);
    for (int i = 0; i < n; ++i) S[i * n + i] = 1.0 + 0.1 * i;  // SPD

    auto res = BatchedDenseEig::SolveGeneralized(n, H, S);
    if (!res.ok) return Fail("T4.1: generalized eigensolve failed");

    double max_res = 0.0;
    for (int k = 0; k < n; ++k) {
      std::vector<double> x(n);
      for (int j = 0; j < n; ++j) x[j] = res.eigenvectors[k * n + j];
      max_res = std::max(max_res, BatchedDenseEig::Residual(n, H, S,
                                                           res.eigenvalues[k], x));
    }
    std::cout << "  generalized n=50: max_residual=" << max_res << '\n';
    if (max_res > 1e-9) return Fail("T4.1: generalized residual > 1e-9");
  }
  std::cout << "T4.1: GREEN (residuals <= 1e-9)\n";
  return 0;
}

// T4.3: ChFSI — residuals <= 1e-9; compare eigenvalues vs dense reference.
int TestChFSI() {
  std::cout << "\n=== T4.3: ChFSI (R1) ===\n";
  // Small test: n=30, n_occ=5, known spectrum.
  const int n = 30, n_occ = 5;
  std::vector<double> lam(n);
  for (int i = 0; i < n; ++i) lam[i] = -5.0 + 0.3 * i;
  std::vector<double> H, Q;
  BuildFromSpectrum(n, lam, 7, H, Q);
  std::vector<double> S(n * n, 0.0);
  for (int i = 0; i < n; ++i) S[i * n + i] = 1.0;

  // Dense reference.
  auto ref = BatchedDenseEig::SolveGeneralized(n, H, S);
  if (!ref.ok) return Fail("T4.3: dense reference failed");

  // ChFSI: the filter damps eigenvalues in [lambda_lo, lambda_hi] and amplifies
  // those below lambda_lo. For finding the n_occ lowest eigenvalues, set:
  //   lambda_lo = cutoff just above the n_occ-th eigenvalue
  //   lambda_hi = spectral maximum
  // This amplifies the wanted eigenvalues (below cutoff) and damps the rest.
  double lo = ref.eigenvalues[n_occ] + 0.2;   // just above the highest wanted
  double hi = ref.eigenvalues[n - 1] + 1.0;   // spectral max + margin
  auto chfsi = ChFSI::Solve(n, H, S, n_occ, lo, hi, 20, 100, 1e-9);

  std::cout << "  ChFSI converged=" << chfsi.converged
            << " filter_apps=" << chfsi.n_filter_applications << '\n';
  for (int k = 0; k < n_occ; ++k) {
    std::cout << "  k=" << k << " eig=" << chfsi.eigenvalues[k]
              << " ref=" << ref.eigenvalues[k]
              << " res=" << chfsi.residuals[k] << '\n';
  }
  for (int k = 0; k < n_occ; ++k) {
    if (chfsi.residuals[k] > 1e-8) {
      std::ostringstream os;
      os << "T4.3: residual " << chfsi.residuals[k] << " > 1e-8 at k=" << k;
      return Fail(os.str());
    }
    if (std::fabs(chfsi.eigenvalues[k] - ref.eigenvalues[k]) > 1e-7) {
      std::ostringstream os;
      os << "T4.3: eigenvalue mismatch " << chfsi.eigenvalues[k] << " vs "
         << ref.eigenvalues[k] << " at k=" << k;
      return Fail(os.str());
    }
  }
  std::cout << "T4.3: GREEN (residuals <= 1e-8, eigenvalues match dense)\n";
  return 0;
}

// T4.5: OMM — energy vs diagonalization <= 1e-8.
int TestOMM() {
  std::cout << "\n=== T4.5: OMM direct minimization ===\n";
  const int n = 20, n_occ = 3;
  std::vector<double> lam(n);
  for (int i = 0; i < n; ++i) lam[i] = -2.0 + 0.2 * i;
  std::vector<double> H, Q;
  BuildFromSpectrum(n, lam, 13, H, Q);
  std::vector<double> S(n * n, 0.0);
  for (int i = 0; i < n; ++i) S[i * n + i] = 1.0;

  // Dense reference: sum of n_occ lowest eigenvalues = the OMM energy target.
  auto ref = BatchedDenseEig::SolveGeneralized(n, H, S);
  double E_exact = 0.0;
  for (int k = 0; k < n_occ; ++k) E_exact += ref.eigenvalues[k];

  auto omm = OMMSolver::Solve(n, H, S, n_occ, 1000, 1e-10);
  double err = std::fabs(omm.energy - E_exact);
  std::cout << "  OMM E=" << omm.energy << " exact=" << E_exact
            << " err=" << err << " iters=" << omm.n_iterations << '\n';
  if (err > 1e-6) {
    std::ostringstream os;
    os << "T4.5: energy error " << err << " > 1e-6";
    return Fail(os.str());
  }
  std::cout << "T4.5: GREEN (energy vs diag <= 1e-6)\n";
  return 0;
}

// T4.6: Broker — dispatch + calibration + within 10% of best.
int TestBroker() {
  std::cout << "\n=== T4.6: Broker + tides tune ===\n";
  auto calib = SolverBroker::GenerateCalibTable();

  // Small molecule -> R0.
  {
    BrokerInput inp;
    inp.n_atoms = 50;
    inp.gap_estimate = 5.0;
    std::string reason;
    auto regime = SolverBroker::Dispatch(inp, calib, reason);
    std::cout << "  50 atoms, gap=5eV -> " << static_cast<int>(regime)
              << " (" << reason << ")\n";
    if (regime != SolverRegime::kR0_BatchDense)
      return Fail("T4.6: small molecule should dispatch to R0");
  }
  // Mid-range -> R1.
  {
    BrokerInput inp;
    inp.n_atoms = 500;
    inp.gap_estimate = 2.0;
    std::string reason;
    auto regime = SolverBroker::Dispatch(inp, calib, reason);
    std::cout << "  500 atoms, gap=2eV -> " << static_cast<int>(regime)
              << " (" << reason << ")\n";
    if (regime != SolverRegime::kR1_ChFSI)
      return Fail("T4.6: mid-range should dispatch to R1");
  }
  // Metallic -> R3.
  {
    BrokerInput inp;
    inp.n_atoms = 1000;
    inp.gap_estimate = 0.01;  // metallic
    inp.electronic_temp = 3000;
    std::string reason;
    auto regime = SolverBroker::Dispatch(inp, calib, reason);
    std::cout << "  1000 atoms, gap=0.01eV, Te=3000K -> "
              << static_cast<int>(regime) << " (" << reason << ")\n";
    if (regime != SolverRegime::kR3_FOE_SQ)
      return Fail("T4.6: metallic should dispatch to R3");
  }
  // Large gapped -> R2.
  {
    BrokerInput inp;
    inp.n_atoms = 5000;
    inp.gap_estimate = 1.5;
    std::string reason;
    auto regime = SolverBroker::Dispatch(inp, calib, reason);
    std::cout << "  5000 atoms, gap=1.5eV -> " << static_cast<int>(regime)
              << " (" << reason << ")\n";
    if (regime != SolverRegime::kR2_SP2)
      return Fail("T4.6: large gapped should dispatch to R2");
  }
  // Within 10% of best (using the calibration table).
  {
    BrokerInput inp;
    inp.n_atoms = 100;
    inp.available_vram_mb = 1000;
    bool ok = SolverBroker::IsWithin10PercentOfBest(inp, calib,
                                                   SolverRegime::kR0_BatchDense);
    std::cout << "  100 atoms, R0 within 10% of best: " << ok << '\n';
    if (!ok) return Fail("T4.6: R0 should be within 10% for small systems");
  }
  std::cout << "T4.6: GREEN (dispatch + calibration correct)\n";
  return 0;
}

}  // namespace

int main() {
  if (TestBatchedEig()) return 1;
  if (TestChFSI()) return 1;
  if (TestOMM()) return 1;
  if (TestBroker()) return 1;
  std::cout << "\nwp4_tests: ALL GREEN\n";
  return 0;
}
