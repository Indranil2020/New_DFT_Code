// T5.3: GPU SP2 purification tests — vs CPU reference.
//
// Validates that the CUDA SP2 purification produces the same density matrix
// as the CPU SP2Purification::Purify within <=1e-10, and that idempotency
// and trace conditions are met.

#include "solvers/sp2_submatrix/sp2.hpp"
#include "solvers/sp2_submatrix/sp2_gpu.hpp"
#include "solvers/dense/batched_eig.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

namespace {

using tides::solvers::SP2Purification;
using tides::solvers::SP2Result;
using tides::solvers::SP2CudaAvailable;
using tides::solvers::SP2PurifyCuda;
using tides::solvers::SP2GpuResult;
using tides::solvers::SP2PurifyBatchCuda;
using tides::solvers::SP2BatchGpuResult;
using tides::solvers::BatchedDenseEig;

// Build a gapped Hamiltonian with known spectrum (same as wp5_tests).
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

double MaxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    d = std::max(d, std::abs(a[i] - b[i]));
  return d;
}

int TestSP2VsCpu() {
  for (int n : {20, 50, 100}) {
    const std::size_t n_occ = static_cast<std::size_t>(n) / 2;
    const double gap = 2.0;
    std::vector<double> H, S;
    BuildGappedSystem(n, n_occ, gap, 42, H, S);
    const double n_e = static_cast<double>(n_occ);

    // Get spectral bounds from dense eigensolve (same as wp5_tests).
    auto ref = BatchedDenseEig::SolveGeneralized(n, H, S);
    if (!ref.ok) {
      std::cerr << "Dense eigensolve failed at n=" << n << '\n';
      return 1;
    }
    const double mu = 0.5 * (ref.eigenvalues[n_occ - 1] + ref.eigenvalues[n_occ]);
    const double lam_min = ref.eigenvalues[0];
    const double lam_max = ref.eigenvalues[n - 1];

    // CPU reference.
    auto cpu = SP2Purification::Purify(n, H, S, n_e, mu, lam_min, lam_max);

    // GPU.
    auto gpu = SP2PurifyCuda(n, H, S, n_e, mu, lam_min, lam_max);
    if (!gpu.ok()) {
      std::cerr << "SP2PurifyCuda failed: " << gpu.status().message() << '\n';
      return 1;
    }

    const double diff = MaxAbsDiff(gpu.value().P, cpu.P);
    const double trace_err = std::fabs(gpu.value().trace_PS - n_e);
    const double idem_err = gpu.value().idempotency_err;

    std::cout << "sp2_vs_cpu: n=" << n
              << " iters_cpu=" << cpu.n_iterations
              << " iters_gpu=" << gpu.value().n_iterations
              << " kernel_ms=" << gpu.value().kernel_ms
              << " max_diff=" << diff
              << " |tr(PS)-Ne|=" << trace_err
              << " idem_err=" << idem_err << '\n';

    if (diff > 1e-10) {
      std::cerr << "FAIL: max_diff=" << diff << " > 1e-10 at n=" << n << '\n';
      return 1;
    }
    if (trace_err > 1e-8) {
      std::cerr << "FAIL: trace_err=" << trace_err << " > 1e-8\n";
      return 1;
    }
  }
  return 0;
}

int TestSP2Idempotency() {
  const int n = 50;
  const std::size_t n_occ = 25;
  const double gap = 2.0;
  std::vector<double> H, S;
  BuildGappedSystem(n, n_occ, gap, 99, H, S);
  const double n_e = static_cast<double>(n_occ);

  auto ref = BatchedDenseEig::SolveGeneralized(n, H, S);
  if (!ref.ok) {
    std::cerr << "Dense eigensolve failed\n";
    return 1;
  }
  const double mu = 0.5 * (ref.eigenvalues[n_occ - 1] + ref.eigenvalues[n_occ]);
  const double lam_min = ref.eigenvalues[0];
  const double lam_max = ref.eigenvalues[n - 1];

  auto gpu = SP2PurifyCuda(n, H, S, n_e, mu, lam_min, lam_max);
  if (!gpu.ok()) {
    std::cerr << "SP2PurifyCuda failed: " << gpu.status().message() << '\n';
    return 1;
  }

  // Check ||P^2 - P||_F explicitly.
  const auto& P = gpu.value().P;
  std::vector<double> P2(n * n, 0.0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double s = 0.0;
      for (int k = 0; k < n; ++k)
        s += P[i * n + k] * P[k * n + j];
      P2[i * n + j] = s;
    }
  double idem = 0.0;
  for (int i = 0; i < n * n; ++i)
    idem += (P2[i] - P[i]) * (P2[i] - P[i]);
  idem = std::sqrt(idem);

  std::cout << "sp2_idempotency: n=" << n
            << " ||P^2-P||_F=" << idem
            << " converged=" << gpu.value().converged
            << " iters=" << gpu.value().n_iterations << '\n';

  if (idem > 1e-8) {
    std::cerr << "FAIL: idempotency " << idem << " > 1e-8\n";
    return 1;
  }
  return 0;
}

int TestSP2Batch() {
  // Purify 3 blocks of different sizes simultaneously.
  std::vector<std::size_t> block_sizes = {10, 20, 15};
  std::vector<std::vector<double>> H_blocks(3), S_blocks(3);
  std::vector<double> n_e_values, mu_values, lam_mins, lam_maxs;

  for (int b = 0; b < 3; ++b) {
    const int n = static_cast<int>(block_sizes[b]);
    const std::size_t n_occ = block_sizes[b] / 2;
    BuildGappedSystem(n, n_occ, 2.0, 42 + b * 100, H_blocks[b], S_blocks[b]);
    n_e_values.push_back(static_cast<double>(n_occ));

    auto ref = BatchedDenseEig::SolveGeneralized(n, H_blocks[b], S_blocks[b]);
    if (!ref.ok) {
      std::cerr << "Dense eigensolve failed for block " << b << '\n';
      return 1;
    }
    mu_values.push_back(0.5 * (ref.eigenvalues[n_occ - 1] + ref.eigenvalues[n_occ]));
    lam_mins.push_back(ref.eigenvalues[0]);
    lam_maxs.push_back(ref.eigenvalues[n - 1]);
  }

  auto gpu = SP2PurifyBatchCuda(block_sizes, H_blocks, S_blocks,
                                n_e_values, mu_values,
                                lam_mins, lam_maxs);
  if (!gpu.ok()) {
    std::cerr << "SP2PurifyBatchCuda failed: " << gpu.status().message() << '\n';
    return 1;
  }

  int failures = 0;
  for (int b = 0; b < 3; ++b) {
    const int n = static_cast<int>(block_sizes[b]);
    auto cpu = SP2Purification::Purify(n, H_blocks[b], S_blocks[b],
                                       n_e_values[b], mu_values[b],
                                       lam_mins[b], lam_maxs[b]);
    const double diff = MaxAbsDiff(gpu.value().P_blocks[b], cpu.P);
    std::cout << "sp2_batch[" << b << "]: n=" << n
              << " max_diff=" << diff
              << " iters=" << gpu.value().n_iterations[b]
              << " converged=" << gpu.value().converged[b] << '\n';
    if (diff > 1e-10) {
      std::cerr << "FAIL: batch block " << b << " max_diff=" << diff << '\n';
      failures++;
    }
  }

  std::cout << "sp2_batch: kernel_ms=" << gpu.value().kernel_ms << '\n';
  return failures;
}

}  // namespace

int main() {
  if (!SP2CudaAvailable()) {
    std::cout << "SKIP: CUDA runtime not available\n";
    return 77;
  }

  int failures = 0;
  failures += TestSP2VsCpu();
  failures += TestSP2Idempotency();
  failures += TestSP2Batch();

  if (failures == 0) {
    std::cout << "All GPU SP2 tests passed.\n";
  } else {
    std::cerr << failures << " GPU SP2 test(s) failed.\n";
  }
  return failures;
}
