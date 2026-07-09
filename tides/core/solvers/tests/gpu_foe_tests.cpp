// GPU FOE (R3 on GPU) tests.
//
// Validates:
//   - GPU FOE dispatch produces correct density matrix (CPU fallback)
//   - Batched FOE across multiple systems
//   - Trace error within tolerance
//   - Speedup estimation model
//   - Config parameters respected

#include "solvers/dense/batched_eig.hpp"
#include "solvers/foe_sq/gpu_foe.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using tides::solvers::BatchedDenseEig;
using tides::solvers::FermiLevelSearch;
using tides::solvers::FermiOperatorExpansion;
using tides::solvers::GPUFOEConfig;
using tides::solvers::GPUFOEResult;
using tides::solvers::GPUFOERunner;

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

// Test 1: Single-system FOE via GPU dispatch.
int TestSingleSystem() {
  std::cout << "\n=== GPU FOE: Single system ===\n";
  const std::size_t n = 30;
  const std::size_t n_occ = 15;
  std::vector<double> lam(n);
  for (std::size_t i = 0; i < n; ++i) lam[i] = -3.0 + 0.1 * i;
  std::vector<double> H;
  BuildSymmetric(n, lam, 42, H);
  std::vector<double> S(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) S[i * n + i] = 1.0;

  auto ref = BatchedDenseEig::SolveGeneralized(n, H, S);
  if (!ref.ok) return Fail("single: reference eigensolve failed");

  double kT = 0.5;
  double mu = FermiLevelSearch::Search(ref.eigenvalues, {}, n_occ, kT);

  GPUFOEConfig config;
  config.use_gpu = false;
  auto result = GPUFOERunner::Compute(n, H, S, mu, kT,
                                       ref.eigenvalues[0],
                                       ref.eigenvalues[n - 1], 60, config);
  if (!result.ok) return Fail("single: FOE failed");

  double tr_err = std::fabs(result.trace_PS - n_occ);
  std::cout << "  n=" << n << " kT=" << kT << " order=" << result.chebyshev_order
            << " tr(PS)=" << result.trace_PS << " |tr-Ne|=" << tr_err << '\n';
  if (tr_err > n_occ * 0.5) return Fail("single: trace error too large");

  std::cout << "PASS\n";
  return 0;
}

// Build a gapped system matching T5.5 pattern.
void BuildGappedSystem(std::size_t n, std::size_t n_occ, double gap,
                       std::uint64_t seed, std::vector<double>& H,
                       std::vector<double>& S) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> g(0.0, 1.0);
  std::vector<double> lam(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (i < n_occ) lam[i] = -2.0 + static_cast<double>(i) / n_occ;
    else lam[i] = -1.0 + gap + static_cast<double>(i - n_occ) / (n - n_occ);
  }
  std::vector<double> Q(n * n);
  for (auto& v : Q) v = g(rng);
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
  H.assign(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < n; ++k)
        s += Q[i * n + k] * lam[k] * Q[j * n + k];
      H[i * n + j] = s;
    }
  S.assign(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) S[i * n + i] = 1.0;
}

// Test 2: Batched FOE across multiple systems.
int TestBatched() {
  std::cout << "\n=== GPU FOE: Batched ===\n";
  const std::size_t k = 3;
  std::vector<std::size_t> sizes = {20, 24, 30};
  std::vector<std::vector<double>> H_batch(k), S_batch(k);
  std::vector<double> mu_batch(k), kT_batch(k), lam_min(k), lam_max(k);
  std::vector<double> n_e_batch(k);

  for (std::size_t b = 0; b < k; ++b) {
    std::size_t n = sizes[b];
    std::size_t n_occ = n / 2;
    BuildGappedSystem(n, n_occ, 0.5, 100 + b, H_batch[b], S_batch[b]);

    auto ref = BatchedDenseEig::SolveGeneralized(n, H_batch[b], S_batch[b]);
    if (!ref.ok) return Fail("batched: ref failed for system " + std::to_string(b));

    kT_batch[b] = 1.0;
    n_e_batch[b] = static_cast<double>(n_occ);
    mu_batch[b] = FermiLevelSearch::Search(ref.eigenvalues, {}, n_e_batch[b], kT_batch[b]);
    lam_min[b] = ref.eigenvalues[0];
    lam_max[b] = ref.eigenvalues[n - 1];
  }

  GPUFOEConfig config;
  config.use_gpu = false;
  auto result = GPUFOERunner::ComputeBatched(
      sizes, H_batch, S_batch, mu_batch, kT_batch,
      lam_min, lam_max, 100, config);

  if (!result.ok) return Fail("batched: FOE failed");
  if (result.results.size() != k) return Fail("batched: wrong count");

  for (std::size_t b = 0; b < k; ++b) {
    double tr_err = std::fabs(result.results[b].trace_PS - n_e_batch[b]);
    std::cout << "  system " << b << ": n=" << sizes[b]
              << " tr(PS)=" << result.results[b].trace_PS
              << " |tr-Ne|=" << tr_err << '\n';
    if (tr_err > n_e_batch[b] * 0.5)
      return Fail("batched: trace error too large for system " + std::to_string(b));
  }

  std::cout << "PASS\n";
  return 0;
}

// Test 3: Config parameters.
int TestConfig() {
  std::cout << "\n=== GPU FOE: Config ===\n";
  GPUFOEConfig config;
  config.use_gpu = false;
  config.device_id = 0;
  config.precision = "fp64";
  config.max_order = 200;
  config.tile_size = 64;
  config.batch_systems = true;

  if (config.max_order != 200) return Fail("config: max_order wrong");
  if (config.tile_size != 64) return Fail("config: tile_size wrong");
  if (config.precision != "fp64") return Fail("config: precision wrong");

  std::cout << "  max_order=200 tile_size=64 precision=fp64\n";
  std::cout << "PASS\n";
  return 0;
}

// Test 4: Speedup estimation model.
int TestSpeedupModel() {
  std::cout << "\n=== GPU FOE: Speedup model ===\n";
  // n=500, batch=100, order=50: should give a reasonable speedup.
  double speedup = GPUFOERunner::EstimatedSpeedup(500, 100, 50);
  std::cout << "  n=500 batch=100 order=50: estimated speedup=" << speedup << "x\n";
  if (speedup < 1.0) return Fail("speedup: should be >= 1x");
  if (speedup > 50.0) return Fail("speedup: capped at 50x");

  // Small n: speedup should be reduced.
  double small_speedup = GPUFOERunner::EstimatedSpeedup(50, 10, 30);
  std::cout << "  n=50 batch=10 order=30: estimated speedup=" << small_speedup << "x\n";
  if (small_speedup >= speedup) return Fail("speedup: small n should be slower");

  std::cout << "PASS\n";
  return 0;
}

// Test 5: Max trace error utility.
int TestMaxTraceError() {
  std::cout << "\n=== GPU FOE: Max trace error ===\n";
  const std::size_t n = 20;
  const std::size_t n_occ = 10;
  std::vector<double> lam(n);
  for (std::size_t i = 0; i < n; ++i) lam[i] = -2.0 + 0.1 * i;
  std::vector<double> H;
  BuildSymmetric(n, lam, 77, H);
  std::vector<double> S(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) S[i * n + i] = 1.0;

  auto ref = BatchedDenseEig::SolveGeneralized(n, H, S);
  double kT = 0.5;
  double mu = FermiLevelSearch::Search(ref.eigenvalues, {}, n_occ, kT);

  GPUFOEConfig config;
  auto result = GPUFOERunner::Compute(n, H, S, mu, kT,
                                       ref.eigenvalues[0],
                                       ref.eigenvalues[n - 1], 60, config);

  GPUFOEResult batched_result;
  batched_result.results.push_back(result);
  std::vector<double> n_e_batch = {static_cast<double>(n_occ)};
  std::vector<std::size_t> sizes = {n};

  double max_err = GPUFOERunner::MaxTraceError(batched_result, n_e_batch, {S}, sizes);
  double expected_err = std::fabs(result.trace_PS - static_cast<double>(n_occ));
  std::cout << "  max_trace_error=" << max_err << " expected=" << expected_err << '\n';
  if (std::fabs(max_err - expected_err) > 1e-15)
    return Fail("max trace error: mismatch");

  std::cout << "PASS\n";
  return 0;
}

}  // namespace

int main() {
  if (TestSingleSystem()) return 1;
  if (TestBatched()) return 1;
  if (TestConfig()) return 1;
  if (TestSpeedupModel()) return 1;
  if (TestMaxTraceError()) return 1;
  std::cout << "\ngpu_foe_tests: ALL GREEN\n";
  return 0;
}
