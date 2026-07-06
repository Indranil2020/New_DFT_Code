#pragma once

#include <cstddef>
#include <vector>

#include "common/status.hpp"
#include "tile/precision.hpp"

namespace tides::solvers {

struct SP2GpuResult {
  std::vector<double> P;      // density matrix (row-major, n x n)
  double trace_PS = 0.0;
  double idempotency_err = 0.0;
  int n_iterations = 0;
  bool converged = false;
  double kernel_ms = 0.0;
  tides::tile::OperationLedger ledger;
};

[[nodiscard]] bool SP2CudaAvailable();

// GPU SP2 purification using batched cuBLAS GEMM for matrix products.
// Same algorithm as CPU SP2Purification::Purify but with GPU-accelerated
// matrix multiplication (X*S*X products).
//
// @param n            Matrix dimension
// @param H            Hamiltonian (n x n, row-major)
// @param S            Overlap (n x n, row-major)
// @param n_e          Number of electrons
// @param mu           Fermi level
// @param lambda_min   Spectral lower bound
// @param lambda_max   Spectral upper bound
// @param max_iter     Max SP2 iterations
// @param tol          Idempotency tolerance
[[nodiscard]] Result<SP2GpuResult> SP2PurifyCuda(
    std::size_t n, const std::vector<double>& H,
    const std::vector<double>& S, double n_e, double mu,
    double lambda_min, double lambda_max,
    int max_iter = 40, double tol = 1e-12);

// Batched GPU SP2 for submatrix method: purify multiple submatrices
// simultaneously using grouped GEMM.
struct SP2BatchGpuResult {
  std::vector<std::vector<double>> P_blocks;  // one per submatrix
  std::vector<double> idempotency_errs;
  std::vector<double> trace_PS_values;
  std::vector<int> n_iterations;
  std::vector<bool> converged;
  double kernel_ms = 0.0;
  tides::tile::OperationLedger ledger;
};

[[nodiscard]] Result<SP2BatchGpuResult> SP2PurifyBatchCuda(
    const std::vector<std::size_t>& block_sizes,
    const std::vector<std::vector<double>>& H_blocks,
    const std::vector<std::vector<double>>& S_blocks,
    const std::vector<double>& n_e_values,
    const std::vector<double>& mu_values,
    const std::vector<double>& lambda_mins,
    const std::vector<double>& lambda_maxs,
    int max_iter = 40, double tol = 1e-12);

}  // namespace tides::solvers
