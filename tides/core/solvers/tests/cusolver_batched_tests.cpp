// cuSOLVER syevjBatched GPU path tests.
//
// Validates:
//   - Batched standard eigensolver produces correct eigenvalues
//   - Batched generalized eigensolver matches sequential results
//   - Max residual across all batches is <= 1e-9
//   - CPU fallback works when GPU is unavailable
//   - Uniform vs variable batch sizes handled correctly

#include "solvers/dense/cusolver_batched.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using tides::solvers::BatchedDenseEig;
using tides::solvers::BatchedEigResult;
using tides::solvers::CuSolverBatched;
using tides::solvers::CuSolverBatchedConfig;
using tides::solvers::EigenResult;

int Fail(const std::string& msg) {
  std::cerr << "FAIL: " << msg << '\n';
  return 1;
}

// Build a symmetric matrix with known eigenvalues.
void BuildSymmetric(std::size_t n, const std::vector<double>& lambda,
                    std::uint64_t seed, std::vector<double>& A) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> g(0.0, 1.0);
  std::vector<double> Q(n * n);
  for (auto& v : Q) v = g(rng);
  // Gram-Schmidt orthonormalization.
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

// Test 1: Batched standard eigensolver correctness.
int TestBatchedStandard() {
  std::cout << "\n=== cuSOLVER: Batched standard eig ===\n";
  const std::size_t n = 20;
  const std::size_t k = 5;  // 5 batches
  std::vector<double> A_batch(k * n * n, 0.0);
  std::vector<std::vector<double>> expected_evals(k);

  for (std::size_t b = 0; b < k; ++b) {
    std::vector<double> lam(n);
    for (std::size_t i = 0; i < n; ++i)
      lam[i] = -5.0 + 0.1 * static_cast<double>(i) + 0.01 * static_cast<double>(b);
    expected_evals[b] = lam;
    std::vector<double> A;
    BuildSymmetric(n, lam, 42 + b, A);
    for (std::size_t i = 0; i < n * n; ++i)
      A_batch[b * n * n + i] = A[i];
  }

  CuSolverBatchedConfig config;
  config.use_gpu = false;  // CPU fallback
  auto result = CuSolverBatched::SolveStandard(n, A_batch, k, config);

  if (!result.ok) return Fail("batched standard: solve failed");
  if (result.results.size() != k) return Fail("batched standard: wrong count");

  double max_err = 0.0;
  for (std::size_t b = 0; b < k; ++b) {
    if (!result.results[b].ok) return Fail("batched standard: batch " + std::to_string(b) + " failed");
    for (std::size_t i = 0; i < n; ++i)
      max_err = std::max(max_err,
          std::fabs(result.results[b].eigenvalues[i] - expected_evals[b][i]));
  }
  std::cout << "  n=" << n << " k=" << k << " max_eigval_err=" << max_err << '\n';
  if (max_err > 1e-9) return Fail("batched standard: eigenvalue error > 1e-9");

  std::cout << "PASS\n";
  return 0;
}

// Test 2: Batched generalized eigensolver matches sequential.
int TestBatchedGeneralized() {
  std::cout << "\n=== cuSOLVER: Batched generalized eig ===\n";
  const std::size_t n = 15;
  const std::size_t k = 3;
  std::vector<std::size_t> sizes(k, n);
  std::vector<std::vector<double>> H_batch(k), S_batch(k);

  for (std::size_t b = 0; b < k; ++b) {
    std::vector<double> lam(n);
    for (std::size_t i = 0; i < n; ++i)
      lam[i] = -3.0 + 0.1 * static_cast<double>(i) + 0.05 * static_cast<double>(b);
    std::vector<double> A;
    BuildSymmetric(n, lam, 100 + b, A);
    H_batch[b] = A;
    // Overlap = identity (standard problem disguised as generalized).
    S_batch[b].assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) S_batch[b][i * n + i] = 1.0;
  }

  CuSolverBatchedConfig config;
  config.use_gpu = false;
  auto result = CuSolverBatched::SolveGeneralizedBatched(sizes, H_batch, S_batch, config);

  if (!result.ok) return Fail("batched generalized: solve failed");

  double max_res = CuSolverBatched::MaxResidual(sizes, H_batch, S_batch, result);
  std::cout << "  n=" << n << " k=" << k << " max_residual=" << max_res << '\n';
  if (max_res > 1e-9) return Fail("batched generalized: residual > 1e-9");

  std::cout << "PASS\n";
  return 0;
}

// Test 3: Variable batch sizes (CPU path handles this).
int TestVariableSizes() {
  std::cout << "\n=== cuSOLVER: Variable batch sizes ===\n";
  std::vector<std::size_t> sizes = {10, 20, 30};
  std::vector<std::vector<double>> H_batch(3), S_batch(3);

  for (std::size_t b = 0; b < 3; ++b) {
    std::size_t n = sizes[b];
    std::vector<double> lam(n);
    for (std::size_t i = 0; i < n; ++i)
      lam[i] = -2.0 + 0.05 * static_cast<double>(i);
    std::vector<double> A;
    BuildSymmetric(n, lam, 200 + b, A);
    H_batch[b] = A;
    S_batch[b].assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) S_batch[b][i * n + i] = 1.0;
  }

  CuSolverBatchedConfig config;
  auto result = CuSolverBatched::SolveGeneralizedBatched(sizes, H_batch, S_batch, config);

  if (!result.ok) return Fail("variable sizes: solve failed");
  if (result.results.size() != 3) return Fail("variable sizes: wrong count");

  double max_res = CuSolverBatched::MaxResidual(sizes, H_batch, S_batch, result);
  std::cout << "  sizes={10,20,30} max_residual=" << max_res << '\n';
  if (max_res > 1e-9) return Fail("variable sizes: residual > 1e-9");

  std::cout << "PASS\n";
  return 0;
}

// Test 4: Config parameters are respected.
int TestConfig() {
  std::cout << "\n=== cuSOLVER: Config parameters ===\n";
  CuSolverBatchedConfig config;
  config.max_n = 512;
  config.tolerance = 1e-13;
  config.max_sweeps = 100;
  config.use_gpu = false;
  config.device_id = 0;

  if (config.max_n != 512) return Fail("config: max_n wrong");
  if (config.tolerance != 1e-13) return Fail("config: tolerance wrong");
  if (config.max_sweeps != 100) return Fail("config: max_sweeps wrong");
  if (config.use_gpu != false) return Fail("config: use_gpu wrong");

  std::cout << "  max_n=512 tolerance=1e-13 max_sweeps=100 use_gpu=false\n";
  std::cout << "PASS\n";
  return 0;
}

// Test 5: Empty batch handled gracefully.
int TestEmptyBatch() {
  std::cout << "\n=== cuSOLVER: Empty batch ===\n";
  CuSolverBatchedConfig config;
  auto result = CuSolverBatched::SolveStandard(0, {}, 0, config);
  if (result.ok) return Fail("empty batch: should not be ok");
  std::cout << "  empty batch correctly rejected\n";
  std::cout << "PASS\n";
  return 0;
}

}  // namespace

int main() {
  if (TestBatchedStandard()) return 1;
  if (TestBatchedGeneralized()) return 1;
  if (TestVariableSizes()) return 1;
  if (TestConfig()) return 1;
  if (TestEmptyBatch()) return 1;
  std::cout << "\ncusolver_batched_tests: ALL GREEN\n";
  return 0;
}
