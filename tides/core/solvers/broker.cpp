// broker.cpp — BrokerRunner implementation (regime dispatch in product path).
//
// This translation unit provides the BrokerRunner utility that executes the
// solver chosen by the SolverBroker. It is the single entry point for
// "dispatch and solve": given a BrokerInput + matrices H, S + n_occ, it
// selects the optimal regime and runs the corresponding solver.
//
// Regime routing:
//   R0 (kR0_BatchDense) → BatchedDenseEig::SolveGeneralized
//   R1 (kR1_ChFSI)      → ChFSI::Solve
//   R2 (kR2_SP2)         → SP2Purification::Purify
//   R3 (kR3_FOE_SQ)      → FermiOperatorExpansion::Compute

#include "solvers/broker.hpp"
#include "solvers/dense/batched_eig.hpp"
#include "solvers/chfsi/chfsi.hpp"
#include "solvers/sp2_submatrix/sp2.hpp"
#include "solvers/foe_sq/foe.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace tides::solvers {

BrokerSolveResult BrokerRunner::Solve(
    const BrokerInput& input,
    std::size_t n, std::size_t n_occ,
    const std::vector<double>& H,
    const std::vector<double>& S,
    double tol) {

  BrokerSolveResult result;
  if (n == 0 || n_occ == 0 || n_occ > n) {
    result.reason = "invalid dimensions";
    return result;
  }

  // 1. Dispatch to choose the regime.
  auto calib = SolverBroker::GenerateCalibTable();
  result.regime_used = SolverBroker::Dispatch(input, calib, result.reason);

  // Helper: build density matrix P = sum_k C_k C_k^T from column-major
  // eigenvectors where evec[k*n + j] = component j of eigenvector k.
  auto build_P = [&](const std::vector<double>& evec) {
    result.P.assign(n * n, 0.0);
    for (std::size_t k = 0; k < n_occ; ++k)
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
          result.P[i * n + j] += evec[k * n + i] * evec[k * n + j];
  };

  // 2. Route to the correct solver.
  if (result.regime_used == SolverRegime::kR0_BatchDense) {
    // R0: dense eigensolve via LAPACK dsygv_ (S^{-1/2} orthogonalization).
    auto eig = BatchedDenseEig::SolveGeneralized(n, H, S);
    if (!eig.ok) {
      result.reason = "R0 dense eigensolve failed";
      return result;
    }
    result.eigenvalues = eig.eigenvalues;
    build_P(eig.eigenvectors);
    result.converged = true;

  } else if (result.regime_used == SolverRegime::kR1_ChFSI) {
    // R1: Chebyshev-filtered subspace iteration.
    auto bounds = ChFSI::SpectralBounds(n, H, S);
    double lo = bounds.first;
    double hi = bounds.second;
    auto chfsi = ChFSI::Solve(n, H, S, n_occ, lo, hi, 20, 100, tol);
    if (chfsi.converged) {
      result.eigenvalues = chfsi.eigenvalues;
      build_P(chfsi.eigenvectors);
      result.converged = true;
    } else {
      // Fallback to R0 dense eigensolve.
      auto eig = BatchedDenseEig::SolveGeneralized(n, H, S);
      if (eig.ok) {
        result.eigenvalues = eig.eigenvalues;
        build_P(eig.eigenvectors);
        result.converged = true;
        result.reason += " (ChFSI did not converge; fell back to R0 dense)";
      } else {
        result.reason = "R1 ChFSI failed and R0 fallback also failed";
      }
    }

  } else if (result.regime_used == SolverRegime::kR2_SP2) {
    // R2: SP2 density-matrix purification.
    // Estimate spectral bounds from diagonal of H (widened by 10%).
    double lambda_min = 1e30, lambda_max = -1e30;
    for (std::size_t i = 0; i < n; ++i) {
      lambda_min = std::min(lambda_min, H[i * n + i]);
      lambda_max = std::max(lambda_max, H[i * n + i]);
    }
    double sw = lambda_max - lambda_min;
    lambda_min -= 0.1 * sw;
    lambda_max += 0.1 * sw;

    double mu = 0.5 * (lambda_min + lambda_max);
    double n_e = static_cast<double>(n_occ);

    auto sp2 = SP2Purification::Purify(n, H, S, n_e, mu,
                                       lambda_min, lambda_max, 40, tol);
    result.P = sp2.P;
    result.converged = sp2.converged;
    if (!result.converged) {
      result.reason += " (SP2 did not converge)";
    }

  } else {
    // R3: Fermi-operator expansion.
    double lambda_min = 1e30, lambda_max = -1e30;
    for (std::size_t i = 0; i < n; ++i) {
      lambda_min = std::min(lambda_min, H[i * n + i]);
      lambda_max = std::max(lambda_max, H[i * n + i]);
    }
    double sw = lambda_max - lambda_min;
    lambda_min -= 0.1 * sw;
    lambda_max += 0.1 * sw;

    double mu = 0.5 * (lambda_min + lambda_max);
    double kT = input.electronic_temp * 3.1668e-6;
    if (kT <= 0.0) kT = 0.01;

    auto foe = FermiOperatorExpansion::Compute(
        n, H, S, mu, kT, lambda_min, lambda_max);
    result.P = foe.P;
    result.converged = foe.ok;
    if (!result.converged) {
      result.reason += " (FOE failed)";
    }
  }

  return result;
}

}  // namespace tides::solvers
