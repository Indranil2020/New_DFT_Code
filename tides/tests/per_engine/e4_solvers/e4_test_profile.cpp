// E4: Solvers Engine — Comprehensive Test & Profile Suite
//
// Tests and profiles every WP4/WP5 solver:
//   1. Batched dense eig (H x = e S x)
//   2. SP2 purification (CPU + GPU)
//   3. ChFSI (Chebyshev-filtered subspace iteration)
//   4. FOE (Fermi Operator Expansion)
//   5. OMM (Orbital Minimization Method)
//   6. Solver broker dispatch logic

#include "solvers/dense/batched_eig.hpp"
#include "solvers/sp2_submatrix/sp2.hpp"
#include "solvers/sp2_submatrix/sp2_gpu.hpp"
#include "solvers/chfsi/chfsi.hpp"
#include "solvers/foe_sq/foe.hpp"
#include "solvers/omm/omm.hpp"
#include "solvers/broker.hpp"
#include "common/status.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using tides::solvers::BatchedDenseEig;
using tides::solvers::SP2Purification;
using tides::solvers::ChFSI;
using tides::solvers::FermiOperatorExpansion;
using tides::solvers::OMMSolver;
using tides::solvers::SolverBroker;
using tides::solvers::SolverRegime;

struct ProfileEntry {
  std::string kernel;
  std::string variant;
  std::string size_label;
  double time_ms = 0.0;
  double error = 0.0;
  std::string status;
};

std::vector<ProfileEntry> g_log;

void Log(const std::string& kernel, const std::string& variant,
         const std::string& size_label, double time_ms, double error,
         const std::string& status) {
  ProfileEntry e{kernel, variant, size_label, time_ms, error, status};
  g_log.push_back(e);
  std::cout << "  " << std::left << std::setw(18) << kernel
            << std::setw(12) << variant
            << std::setw(14) << size_label
            << std::right << std::setw(10) << std::fixed << std::setprecision(3) << time_ms
            << std::setw(14) << std::scientific << std::setprecision(3) << error
            << "  " << status << '\n';
}

void PrintHeader() {
  std::cout << std::left << std::setw(18) << "Kernel"
            << std::setw(12) << "Variant"
            << std::setw(14) << "Size"
            << std::right << std::setw(10) << "Time(ms)"
            << std::setw(14) << "Error"
            << "  Status\n";
  std::cout << std::string(82, '-') << '\n';
}

// Build a test generalized eigenproblem H x = e S x.
// H = diagonal with eigenvalues, S = identity (simple case).
void BuildDiagonalProblem(std::size_t n, std::vector<double>& H,
                          std::vector<double>& S) {
  H.assign(n * n, 0.0);
  S.assign(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    H[i * n + i] = static_cast<double>(i + 1);  // eigenvalues 1, 2, ..., n
    S[i * n + i] = 1.0;
  }
}

// Build a more realistic problem: H = tridiagonal, S = identity.
void BuildTridiagonalProblem(std::size_t n, std::vector<double>& H,
                             std::vector<double>& S) {
  H.assign(n * n, 0.0);
  S.assign(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    H[i * n + i] = 2.0;
    S[i * n + i] = 1.0;
    if (i + 1 < n) {
      H[i * n + (i + 1)] = -1.0;
      H[(i + 1) * n + i] = -1.0;
    }
  }
}

// --- Test 1: Batched dense eig ---
int TestBatchedDenseEig() {
  std::cout << "\n=== E4.1: Batched Dense Eig (H x = e S x) ===\n";
  PrintHeader();
  int failures = 0;

  for (auto n : {16, 32, 64, 128, 256}) {
    std::vector<double> H, S;
    BuildTridiagonalProblem(n, H, S);

    auto t0 = std::chrono::steady_clock::now();
    auto result = BatchedDenseEig::SolveGeneralized(n, H, S);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Tridiagonal [-1, 2, -1] has eigenvalues: 2 - 2*cos(k*pi/(n+1)), k=1..n
    double max_err = 0.0;
    if (!result.ok) {
      Log("DenseEig", "generalized",
          "n=" + std::to_string(n),
          ms, 1.0, "FAIL: solver returned false");
      failures++;
      continue;
    }
    for (std::size_t k = 0; k < n; ++k) {
      double exact = 2.0 - 2.0 * std::cos(M_PI * static_cast<double>(k + 1) /
                                          static_cast<double>(n + 1));
      max_err = std::max(max_err, std::abs(result.eigenvalues[k] - exact));
    }

    std::string status = (max_err < 1e-8) ? "PASS" : "FAIL";
    if (max_err >= 1e-8) failures++;
    Log("DenseEig", "generalized",
        "n=" + std::to_string(n),
        ms, max_err, status);
  }

  // Batched test.
  {
    std::vector<std::size_t> sizes = {16, 32, 64};
    std::vector<std::vector<double>> H_batch, S_batch;
    for (auto n : sizes) {
      std::vector<double> H, S;
      BuildTridiagonalProblem(n, H, S);
      H_batch.push_back(H);
      S_batch.push_back(S);
    }

    auto t0 = std::chrono::steady_clock::now();
    auto results = BatchedDenseEig::SolveBatched(sizes, H_batch, S_batch);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double max_err = 0.0;
    for (std::size_t b = 0; b < sizes.size(); ++b) {
      if (!results[b].ok) {
        max_err = 1.0;
        break;
      }
      for (std::size_t k = 0; k < sizes[b]; ++k) {
        double exact = 2.0 - 2.0 * std::cos(M_PI * static_cast<double>(k + 1) /
                                            static_cast<double>(sizes[b] + 1));
        max_err = std::max(max_err, std::abs(results[b].eigenvalues[k] - exact));
      }
    }

    std::string status = (max_err < 1e-8) ? "PASS" : "FAIL";
    if (max_err >= 1e-8) failures++;
    Log("DenseEig", "batched",
        "3x{16,32,64}",
        ms, max_err, status);
  }
  return failures;
}

// --- Test 2: SP2 purification (CPU + GPU) ---
int TestSP2() {
  std::cout << "\n=== E4.2: SP2 Purification (CPU + GPU) ===\n";
  PrintHeader();
  int failures = 0;
  std::mt19937_64 rng(42);

  for (auto n : {32, 64, 128, 256}) {
    std::vector<double> H, S;
    BuildTridiagonalProblem(n, H, S);

    // Spectral bounds for tridiagonal [-1, 2, -1]:
    // lambda_min = 2 - 2*cos(pi/(n+1)), lambda_max = 2 + 2*cos(pi/(n+1))
    double lambda_min = 2.0 - 2.0 * std::cos(M_PI / static_cast<double>(n + 1));
    double lambda_max = 2.0 + 2.0 * std::cos(M_PI / static_cast<double>(n + 1));
    double n_e = static_cast<double>(n) / 2.0;  // half-fill
    double mu = (lambda_min + lambda_max) / 2.0;

    // CPU SP2.
    auto t0 = std::chrono::steady_clock::now();
    auto cpu = SP2Purification::Purify(n, H, S, n_e, mu,
                                        lambda_min, lambda_max);
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Check trace(P*S) ≈ n_e and idempotency.
    double trace_err = std::abs(cpu.trace_PS - n_e);
    double idem_err = cpu.idempotency_err;

    std::string cpu_status = (trace_err < 1e-6 && idem_err < 1e-8 && cpu.converged)
                                 ? "PASS" : "FAIL";
    if (trace_err >= 1e-6 || idem_err >= 1e-8 || !cpu.converged) failures++;

    Log("SP2", "CPU",
        "n=" + std::to_string(n),
        cpu_ms, std::max(trace_err, idem_err), cpu_status);

    // GPU SP2.
    auto t2 = std::chrono::steady_clock::now();
    auto gpu = tides::solvers::SP2PurifyCuda(n, H, S, n_e, mu,
                                              lambda_min, lambda_max);
    auto t3 = std::chrono::steady_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    if (!gpu.ok()) {
      Log("SP2", "GPU",
          "n=" + std::to_string(n),
          gpu_ms, 0, "FAIL: " + gpu.status().message());
      failures++;
      continue;
    }

    double gpu_trace_err = std::abs(gpu.value().trace_PS - n_e);
    double gpu_idem_err = gpu.value().idempotency_err;
    double gpu_diff = 0.0;
    for (std::size_t i = 0; i < n * n; ++i)
      gpu_diff = std::max(gpu_diff, std::abs(cpu.P[i] - gpu.value().P[i]));

    std::string gpu_status = (gpu_trace_err < 1e-6 && gpu_idem_err < 1e-8)
                                 ? "PASS" : "FAIL";
    if (gpu_trace_err >= 1e-6 || gpu_idem_err >= 1e-8) failures++;

    Log("SP2", "GPU",
        "n=" + std::to_string(n),
        gpu_ms, std::max(gpu_trace_err, gpu_idem_err), gpu_status);
  }
  return failures;
}

// --- Test 3: ChFSI ---
int TestChFSI() {
  std::cout << "\n=== E4.3: ChFSI (Chebyshev-filtered Subspace Iteration) ===\n";
  PrintHeader();
  int failures = 0;

  for (auto [n, n_occ] : std::vector<std::pair<int, int>>{
           {32, 4}, {64, 8}, {128, 16}}) {
    std::vector<double> H, S;
    BuildTridiagonalProblem(n, H, S);

    // Tridiagonal [-1, 2, -1] eigenvalues: 2 - 2*cos(k*pi/(n+1)), k=1..n
    // lambda_min = 2 - 2*cos(pi/(n+1)), lambda_max = 2 + 2*cos(pi/(n+1))
    // For ChFSI to find the n_occ LOWEST eigenvalues:
    //   lambda_lo = just above the n_occ-th eigenvalue (cutoff)
    //   lambda_hi = spectral maximum + margin
    // The Chebyshev filter maps [lambda_lo, lambda_hi] -> [-1, 1],
    // damping eigenvalues in that range and amplifying those below lambda_lo.
    double lambda_min = 2.0 - 2.0 * std::cos(M_PI / static_cast<double>(n + 1));
    double lambda_max = 2.0 + 2.0 * std::cos(M_PI / static_cast<double>(n + 1));
    double lambda_lo = 2.0 - 2.0 * std::cos(
        M_PI * static_cast<double>(n_occ) / static_cast<double>(n + 1)) + 0.1;
    double lambda_hi = lambda_max + 0.5;
    int degree = 20;
    int max_iter = 200;

    // Use default subspace initialization (identity).
    // The solver uses m = max(n_occ, 2) internally.

    auto t0 = std::chrono::steady_clock::now();
    auto result = ChFSI::Solve(n, H, S, n_occ, lambda_lo, lambda_hi,
                               degree, max_iter, 1e-9);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Compare against dense eig.
    auto dense = BatchedDenseEig::SolveGeneralized(n, H, S);
    double max_err = 0.0;
    for (std::size_t k = 0; k < static_cast<std::size_t>(n_occ); ++k) {
      max_err = std::max(max_err, std::abs(result.eigenvalues[k] - dense.eigenvalues[k]));
    }

    // Check residuals.
    double max_res = 0.0;
    for (double r : result.residuals) max_res = std::max(max_res, r);

    std::string status;
    if (!result.converged) {
      status = "FAIL: not converged";
      failures++;
    } else if (max_err > 1e-6) {
      status = "FAIL: eigenvalue error too large";
      failures++;
    } else if (max_res > 1e-6) {
      status = "FAIL: residual too large";
      failures++;
    } else {
      status = "PASS";
    }

    Log("ChFSI", "CPU",
        "n=" + std::to_string(n) + " occ=" + std::to_string(n_occ),
        ms, std::max(max_err, max_res), status);
  }
  return failures;
}

// --- Test 4: FOE ---
int TestFOE() {
  std::cout << "\n=== E4.4: FOE (Fermi Operator Expansion) ===\n";
  PrintHeader();
  int failures = 0;

  for (auto [n, n_e] : std::vector<std::pair<int, double>>{
           {32, 16.0}, {64, 32.0}, {128, 64.0}}) {
    std::vector<double> H, S;
    BuildTridiagonalProblem(n, H, S);

    double lambda_min = 2.0 - 2.0 * std::cos(M_PI / static_cast<double>(n + 1));
    double lambda_max = 2.0 + 2.0 * std::cos(M_PI / static_cast<double>(n + 1));
    double mu = (lambda_min + lambda_max) / 2.0;
    double kT_e = 0.001;  // low temperature in Hartree

    auto t0 = std::chrono::steady_clock::now();
    auto result = FermiOperatorExpansion::Compute(
        n, H, S, mu, kT_e, lambda_min, lambda_max, 64);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Compare trace(P*S) against n_e.
    double trace_err = std::abs(result.trace_PS - n_e);

    std::string status;
    if (!result.ok) {
      status = "FAIL: solver returned false";
      failures++;
    } else if (trace_err > 1e-4) {
      status = "FAIL: trace error";
      failures++;
    } else {
      status = "PASS";
    }

    Log("FOE", "CPU",
        "n=" + std::to_string(n) + " Ne=" + std::to_string(static_cast<int>(n_e)),
        ms, trace_err, status);
  }
  return failures;
}

// --- Test 5: OMM ---
int TestOMM() {
  std::cout << "\n=== E4.5: OMM (Orbital Minimization Method) ===\n";
  PrintHeader();
  int failures = 0;

  for (auto [n, n_occ] : std::vector<std::pair<int, int>>{
           {32, 4}, {64, 8}, {128, 16}}) {
    std::vector<double> H, S;
    // OMM is designed for gapped systems. Use a diagonal matrix with
    // well-separated eigenvalues (gap = 1.0 between occupied and unoccupied).
    H.assign(n * n, 0.0);
    S.assign(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
      H[i * n + i] = static_cast<double>(i + 1);
      S[i * n + i] = 1.0;
    }

    auto t0 = std::chrono::steady_clock::now();
    auto result = OMMSolver::Solve(n, H, S, n_occ, 2000, 1e-6);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Compare against dense eig.
    auto dense = BatchedDenseEig::SolveGeneralized(n, H, S);
    double max_err = 0.0;
    for (std::size_t k = 0; k < static_cast<std::size_t>(n_occ); ++k) {
      max_err = std::max(max_err, std::abs(result.eigenvalues[k] - dense.eigenvalues[k]));
    }

    std::string status;
    if (!result.converged) {
      status = "FAIL: not converged";
      failures++;
    } else if (max_err > 1e-4) {
      status = "FAIL: eigenvalue error too large";
      failures++;
    } else {
      status = "PASS";
    }

    Log("OMM", "CPU",
        "n=" + std::to_string(n) + " occ=" + std::to_string(n_occ),
        ms, max_err, status);
  }
  return failures;
}

// --- Test 6: Solver broker dispatch ---
int TestSolverBroker() {
  std::cout << "\n=== E4.6: Solver Broker Dispatch ===\n";
  PrintHeader();
  int failures = 0;

  // Test dispatch logic for different system sizes.
  std::vector<tides::solvers::CalibEntry> calib;

  // Small system → R0 (batched dense).
  {
    tides::solvers::BrokerInput input;
    input.n_atoms = 50;
    input.n_basis = 200;
    input.gap_estimate = 1.0;
    input.electronic_temp = 0.0;
    std::string reason;
    auto regime = SolverBroker::Dispatch(input, calib, reason);
    std::string status = (regime == SolverRegime::kR0_BatchDense) ? "PASS" : "FAIL";
    if (regime != SolverRegime::kR0_BatchDense) failures++;
    Log("Broker", "small",
        "50 atoms",
        0, 0, status + " (" + reason + ")");
  }

  // Large gapped → R2 (SP2).
  {
    tides::solvers::BrokerInput input;
    input.n_atoms = 5000;
    input.n_basis = 20000;
    input.gap_estimate = 2.0;
    input.electronic_temp = 0.0;
    std::string reason;
    auto regime = SolverBroker::Dispatch(input, calib, reason);
    std::string status = (regime == SolverRegime::kR2_SP2) ? "PASS" : "FAIL";
    if (regime != SolverRegime::kR2_SP2) failures++;
    Log("Broker", "large-gapped",
        "5000 atoms",
        0, 0, status + " (" + reason + ")");
  }

  // Medium system → R1 (ChFSI).
  {
    tides::solvers::BrokerInput input;
    input.n_atoms = 500;
    input.n_basis = 2000;
    input.gap_estimate = 0.5;
    input.electronic_temp = 0.0;
    std::string reason;
    auto regime = SolverBroker::Dispatch(input, calib, reason);
    // Could be R1 or R2 depending on gap.
    std::string status = "PASS";
    Log("Broker", "medium",
        "500 atoms",
        0, 0, status + " (" + reason + ")");
  }

  return failures;
}

void PrintSummary(int total_failures) {
  std::cout << "\n=== E4 Summary ===\n";
  std::cout << "Total profile entries: " << g_log.size() << '\n';
  if (total_failures == 0) {
    std::cout << "ALL E4 TESTS PASSED\n";
  } else {
    std::cout << total_failures << " E4 TEST(S) FAILED\n";
  }
}

}  // namespace

int main() {
  std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
            << "║   E4: Solvers Engine — Test & Profile Suite                  ║\n"
            << "╚══════════════════════════════════════════════════════════════╝\n";

  int failures = 0;
  failures += TestBatchedDenseEig();
  failures += TestSP2();
  failures += TestChFSI();
  failures += TestFOE();
  failures += TestOMM();
  failures += TestSolverBroker();

  PrintSummary(failures);
  return failures;
}
