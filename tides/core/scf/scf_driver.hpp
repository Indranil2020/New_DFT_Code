#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

#include "solvers/dense/batched_eig.hpp"
#include "solvers/broker.hpp"

// BLAS symmetric rank-k update for density matrix construction.
extern "C" {
void dsyrk_(const char* uplo, const char* trans, const int* n, const int* k,
            const double* alpha, const double* a, const int* lda,
            const double* beta, double* c, const int* ldc);
}

namespace tides::scf {

// SCF driver with Pulay / simple mixing (per WP6 T6.1).
//
// The SCF loop:
//   1. Given density matrix P, build the effective potential V_eff[P].
//   2. Build the Hamiltonian H[P] and overlap S (fixed for NAOs).
//   3. Solve the generalized eigenproblem H x = e S x (via SolverBroker).
//   4. Occupy the n_occ lowest orbitals -> new density P_new.
//   5. Mix: P_next = alpha * P_new + (1-alpha) * P_old (simple) or Pulay.
//   6. Check convergence (energy or density change < tol).
//
// For the CPU reference, the Hamiltonian build is a callback (the caller
// provides H(P) given the current density). This separates the SCF logic
// from the physics of H construction (WP2/WP3), making it testable.
//
// Observable (T6.1): converges gauntlet within documented iteration budget.

struct SCFResult {
  double energy = 0.0;
  std::vector<double> P;             // converged density matrix
  std::vector<double> eigenvalues;   // orbital eigenvalues
  std::vector<double> eigenvectors;  // orbital coefficients
  int n_iterations = 0;
  bool converged = false;
  std::vector<double> energy_history;  // per-iteration energies
};

class SCFDriver {
 public:
  // Run SCF. The caller provides:
  //   n:        matrix dimension (basis size)
  //   n_occ:    number of occupied orbitals (spin-paired)
  //   S:        overlap matrix (fixed, n x n row-major)
  //   build_H:  callback: given P (n x n), returns H (n x n)
  //   energy:   callback: given P and eigenvalues, returns total energy
  //   P_init:   initial density (empty = identity guess)
  //   max_iter: maximum SCF iterations
  //   tol:      energy convergence target
  //   mixing:   0=simple, 1=Pulay(Anderson)
  //   alpha:    mixing parameter (0 < alpha <= 1)
  //
  // AUDIT B5/B7: energy_fn now receives eigenvalues from the SCF loop's
  // eigensolve, eliminating the redundant re-diagonalization. The caller
  // must not call build_H inside energy_fn — the (P, H) pair is consistent.
  static SCFResult Run(
      std::size_t n, std::size_t n_occ, const std::vector<double>& S,
      const std::function<std::vector<double>(const std::vector<double>&)>& build_H,
      const std::function<double(const std::vector<double>&, const std::vector<double>&)>& energy_fn,
      const std::vector<double>& P_init = {},
      int max_iter = 100, double tol = 1e-10,
      int mixing = 1, double alpha = 0.3) {
    SCFResult res;
    if (n == 0 || n_occ == 0 || n_occ > n) return res;

    // Initialize density.
    std::vector<double> P;
    if (P_init.size() == n * n) {
      P = P_init;
    } else {
      P.assign(n * n, 0.0);
      for (std::size_t i = 0; i < n; ++i) P[i * n + i] = 0.5;  // diagonal guess
    }

    // Pulay/DIIS history.
    const int pulay_depth = 8;
    std::vector<std::vector<double>> P_history;   // P at each step
    std::vector<std::vector<double>> F_history;    // P_new (output) at each step

    double E_prev = 1e30;
    for (int iter = 0; iter < max_iter; ++iter) {
      res.n_iterations = iter + 1;

      // Build H from current P.
      auto H = build_H(P);

      // AUDIT C5: Use SolverBroker to dispatch the eigensolve.
      // For small molecular systems (<=200 atoms), broker selects R0
      // (batched dense eig). For larger systems, it would select R1/R2/R3.
      // Currently only R0 is fully wired; other regimes fall back to R0.
      tides::solvers::BrokerInput broker_in;
      broker_in.n_basis = n;
      broker_in.n_atoms = n;  // conservative: assume ~1 basis fn/atom for broker
      broker_in.gap_estimate = 1.0;  // assume gapped (molecular)
      auto calib = tides::solvers::SolverBroker::GenerateCalibTable();
      std::string broker_reason;
      auto regime = tides::solvers::SolverBroker::Dispatch(broker_in, calib,
                                                            broker_reason);

      tides::solvers::EigenResult eig;
      if (regime == tides::solvers::SolverRegime::kR0_BatchDense) {
        eig = tides::solvers::BatchedDenseEig::SolveGeneralized(n, H, S);
      } else {
        // R1/R2/R3 not yet wired for single-system SCF; fall back to R0.
        eig = tides::solvers::BatchedDenseEig::SolveGeneralized(n, H, S);
      }
      if (!eig.ok) return res;

      // Occupy n_occ lowest orbitals -> P_new = C_occ @ C_occ^T.
      // Use BLAS dsyrk for O(n^2 * n_occ) symmetric rank-k update.
      std::vector<double> P_new(n * n, 0.0);
      {
        int nn = static_cast<int>(n);
        int kk = static_cast<int>(n_occ);
        double alpha_blas = 1.0;
        double beta_blas = 0.0;
        char uplo = 'L';
        char trans = 'N';
        // eigenvectors are column-major: evec[k*n + j] = component j of eigvector k
        // dsyrk with trans='N' computes C = alpha * A * A^T + beta * C
        // A is n x kk (first kk columns of eigenvectors), lda = n
        dsyrk_(&uplo, &trans, &nn, &kk, &alpha_blas,
               eig.eigenvectors.data(), &nn,
               &beta_blas, P_new.data(), &nn);
        // Symmetrize: copy lower triangle to upper
        for (std::size_t i = 0; i < n; ++i)
          for (std::size_t j = i + 1; j < n; ++j)
            P_new[i * n + j] = P_new[j * n + i];
      }

      // Compute energy from the same (P_new, H) pair — no re-diagonalization.
      // AUDIT B5/B7: eigenvalues come from the SCF loop's eigensolve.
      double E = energy_fn(P_new, eig.eigenvalues);
      res.energy_history.push_back(E);

      // Convergence check.
      if (std::fabs(E - E_prev) < tol) {
        res.converged = true;
        res.energy = E;
        res.P = P_new;
        res.eigenvalues = eig.eigenvalues;
        res.eigenvectors = eig.eigenvectors;
        return res;
      }
      E_prev = E;

      // Mixing.
      std::vector<double> P_next(n * n, 0.0);
      if (mixing == 1 && static_cast<int>(P_history.size()) >= 2) {
        // Real DIIS/Pulay: find coefficients c_j minimizing ||sum c_j R_j||^2
        // subject to sum c_j = 1, where R_j = F_j - P_j (residual).
        // Then P_next = sum c_j F_j.
        const int m = std::min(pulay_depth, static_cast<int>(P_history.size()));
        const int mstart = static_cast<int>(P_history.size()) - m;

        // Build the DIIS matrix B[i][j] = <R_i, R_j> (Frobenius inner product).
        // B is m x m. We augment with the constraint sum c = 1 via Lagrange.
        // Add small diagonal regularization to prevent ill-conditioning.
        std::vector<std::vector<double>> B(m + 1, std::vector<double>(m + 1, 0.0));
        for (int i = 0; i < m; ++i) {
          const auto& Ri = F_history[mstart + i];
          const auto& Pi = P_history[mstart + i];
          for (int j = i; j < m; ++j) {
            const auto& Rj = F_history[mstart + j];
            const auto& Pj = P_history[mstart + j];
            double dot = 0.0;
            for (std::size_t idx = 0; idx < n * n; ++idx)
              dot += (Ri[idx] - Pi[idx]) * (Rj[idx] - Pj[idx]);
            B[i][j] = dot;
            B[j][i] = dot;
          }
          B[i][m] = 1.0;
          B[m][i] = 1.0;
        }
        B[m][m] = 0.0;
        // Regularize: add small lambda to diagonal of the residual block.
        double bmax = 0.0;
        for (int i = 0; i < m; ++i)
          bmax = std::max(bmax, std::fabs(B[i][i]));
        double lambda = 1e-10 * bmax;
        for (int i = 0; i < m; ++i)
          B[i][i] += lambda;

        // Solve B * c = [0, ..., 0, 1]^T via Gaussian elimination with pivoting.
        std::vector<double> rhs(m + 1, 0.0);
        rhs[m] = 1.0;
        std::vector<double> c = rhs;
        // Forward elimination.
        for (int col = 0; col <= m; ++col) {
          int piv = col;
          for (int row = col + 1; row <= m; ++row)
            if (std::fabs(B[row][col]) > std::fabs(B[piv][col])) piv = row;
          if (piv != col) {
            std::swap(B[piv], B[col]);
            std::swap(c[piv], c[col]);
          }
          if (std::fabs(B[col][col]) < 1e-30) continue;
          for (int row = col + 1; row <= m; ++row) {
            double factor = B[row][col] / B[col][col];
            for (int k2 = col; k2 <= m; ++k2)
              B[row][k2] -= factor * B[col][k2];
            c[row] -= factor * c[col];
          }
        }
        // Back substitution.
        std::vector<double> coeffs(m + 1, 0.0);
        for (int row = m; row >= 0; --row) {
          double sum = c[row];
          for (int k2 = row + 1; k2 <= m; ++k2)
            sum -= B[row][k2] * coeffs[k2];
          if (std::fabs(B[row][row]) > 1e-30)
            coeffs[row] = sum / B[row][row];
        }

        // Extrapolate: P_next = sum_{j=0}^{m-1} coeffs[j] * F_history[mstart+j].
        for (int j = 0; j < m; ++j) {
          const auto& Fj = F_history[mstart + j];
          for (std::size_t idx = 0; idx < n * n; ++idx)
            P_next[idx] += coeffs[j] * Fj[idx];
        }

        // Validate DIIS coefficients: if any |c_j| > 10 or sum != ~1, reject.
        bool diis_ok = true;
        double csum = 0.0;
        for (int j = 0; j < m; ++j) {
          csum += coeffs[j];
          if (std::fabs(coeffs[j]) > 10.0) { diis_ok = false; break; }
        }
        if (std::fabs(csum - 1.0) > 0.5) diis_ok = false;

        // Check result for NaN/inf.
        double pmax = 0.0;
        for (std::size_t idx = 0; idx < n * n; ++idx)
          pmax = std::max(pmax, std::fabs(P_next[idx]));
        if (!std::isfinite(pmax) || pmax > 1e6) diis_ok = false;

        if (!diis_ok) {
          // Fall back to simple mixing with Kerker damping.
          double res_norm = 0.0;
          for (std::size_t idx = 0; idx < n * n; ++idx) {
            double d = P_new[idx] - P[idx];
            res_norm += d * d;
          }
          double rms = std::sqrt(res_norm / static_cast<double>(n * n));
          double eff_a = std::max(alpha / (1.0 + rms), 0.05);
          for (std::size_t idx = 0; idx < n * n; ++idx)
            P_next[idx] = eff_a * P_new[idx] + (1.0 - eff_a) * P[idx];
        }
      } else {
        // Simple mixing with Kerker-style damping using RMS residual.
        double res_norm = 0.0;
        for (std::size_t idx = 0; idx < n * n; ++idx) {
          double d = P_new[idx] - P[idx];
          res_norm += d * d;
        }
        double rms = std::sqrt(res_norm / static_cast<double>(n * n));
        double eff_alpha = std::max(alpha / (1.0 + rms), 0.05);
        for (std::size_t idx = 0; idx < n * n; ++idx)
          P_next[idx] = eff_alpha * P_new[idx] + (1.0 - eff_alpha) * P[idx];
      }

      // Update history.
      P_history.push_back(P);
      F_history.push_back(P_new);
      if (static_cast<int>(P_history.size()) > pulay_depth) {
        P_history.erase(P_history.begin());
        F_history.erase(F_history.begin());
      }

      P = P_next;
      res.energy = E;
      res.P = P;
      res.eigenvalues = eig.eigenvalues;
      res.eigenvectors = eig.eigenvectors;
    }
    return res;
  }
};

}  // namespace tides::scf
