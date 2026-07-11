#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <vector>

#include "solvers/dense/batched_eig.hpp"
#include "solvers/broker.hpp"
#include "solvers/chfsi/chfsi.hpp"
#include "solvers/sp2_submatrix/sp2.hpp"
#include "solvers/foe_sq/foe.hpp"

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
      int mixing = 1, double alpha = 0.3,
      const solvers::BrokerInput* broker_input = nullptr) {
    SCFResult res;
    if (n == 0 || n_occ == 0 || n_occ > n) return res;

    // --- S^{-1/2} with eigenvalue filtering (Löwdin orthogonalization) ---
    // Diagonalize S = V Λ V^T, filter eigenvalues below threshold, and form
    // X = V Λ^{-1/2} V^T (truncated to retained subspace). This handles
    // near-linear dependence in NAO basis sets (small S eigenvalues).
    const double s_filter = 1e-8;  // relative threshold: discard evals < s_filter * max_eval
    std::vector<double> S_copy(S);
    std::vector<double> s_eval(n, 0.0);
    std::vector<double> s_evec(n * n, 0.0);  // column-major eigenvectors
    {
      int nn = static_cast<int>(n);
      char jobz = 'V';
      char uplo = 'L';
      int lda = nn;
      int lwork = -1;
      double wkopt = 0.0;
      int info = 0;
      dsyev_(&jobz, &uplo, &nn, S_copy.data(), &lda, s_eval.data(), &wkopt, &lwork, &info);
      if (info != 0) return res;
      lwork = static_cast<int>(wkopt);
      std::vector<double> work(static_cast<std::size_t>(lwork));
      dsyev_(&jobz, &uplo, &nn, S_copy.data(), &lda, s_eval.data(), work.data(), &lwork, &info);
      if (info != 0) return res;
      // S_copy now holds eigenvectors as columns (column-major).
      s_evec = S_copy;
    }
    double s_max = s_eval[n - 1];  // ascending order
    double s_thresh = s_filter * s_max;
    std::size_t n_retained = 0;
    for (std::size_t i = 0; i < n; ++i)
      if (s_eval[i] > s_thresh) ++n_retained;
    // Build X = V_retained Λ^{-1/2} (n x n_retained, column-major)
    // We store X as row-major n x n_retained for convenience.
    // s_evec is column-major: evec[j + i*nn] = component j of eigvec i.
    // X[i * n_retained + k] = s_evec[j + (skip+k)*nn] / sqrt(s_eval[skip+k])
    std::size_t skip = n - n_retained;  // number of discarded small eigenvalues
    std::vector<double> X(n * n_retained, 0.0);  // row-major: X[i * n_retained + k]
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t k = 0; k < n_retained; ++k) {
        // s_evec is column-major: column (skip+k), row i
        X[i * n_retained + k] = s_evec[i + (skip + k) * n] / std::sqrt(s_eval[skip + k]);
      }
    }
    // For convenience also store X^T (n_retained x n, row-major)
    std::vector<double> Xt(n_retained * n, 0.0);  // Xt[k * n + i] = X[i * n_retained + k]
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t k = 0; k < n_retained; ++k)
        Xt[k * n + i] = X[i * n_retained + k];

    // Initialize density.
    std::vector<double> P;
    if (P_init.size() == n * n) {
      P = P_init;
    } else {
      P.assign(n * n, 0.0);
      // Improved initial guess: uniform filling of lowest diagonal elements.
      // This is closer to the converged density than uniform diagonal.
      const double p_init =
          (n > 0) ? static_cast<double>(n_occ) / static_cast<double>(n) : 0.0;
      for (std::size_t i = 0; i < n; ++i) P[i * n + i] = p_init;  // diagonal guess
    }

    // --- Broker dispatch: select solver regime ---
    solvers::SolverRegime regime = solvers::SolverRegime::kR0_BatchDense;
    std::string broker_reason;
    if (broker_input != nullptr) {
      auto calib = solvers::SolverBroker::GenerateCalibTable();
      regime = solvers::SolverBroker::Dispatch(*broker_input, calib, broker_reason);
      std::cout << "[SCFDriver] broker: " << broker_reason
                << " -> regime R" << static_cast<int>(regime) << std::endl;
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

      // --- R2/R3: density-matrix solvers (no eigendecomposition) ---
      if (regime == solvers::SolverRegime::kR2_SP2 ||
          regime == solvers::SolverRegime::kR3_FOE_SQ) {
        // Estimate spectral bounds from diagonal of H.
        double lambda_min = 1e30, lambda_max = -1e30;
        for (std::size_t i = 0; i < n; ++i) {
          lambda_min = std::min(lambda_min, H[i * n + i]);
          lambda_max = std::max(lambda_max, H[i * n + i]);
        }
        // Widen bounds by 10% for safety.
        double sw = lambda_max - lambda_min;
        lambda_min -= 0.1 * sw;
        lambda_max += 0.1 * sw;

        // Fermi level: place in the gap (estimated at mid-spectrum).
        double mu = 0.5 * (lambda_min + lambda_max);
        double n_e = static_cast<double>(n_occ);

        std::vector<double> P_new;
        if (regime == solvers::SolverRegime::kR2_SP2) {
          auto sp2_res = solvers::SP2Purification::Purify(
              n, H, S, n_e, mu, lambda_min, lambda_max);
          P_new = sp2_res.P;
        } else {
          double kT = broker_input ? broker_input->electronic_temp * 3.1668e-6 : 0.01;
          if (kT <= 0) kT = 0.01;
          auto foe_res = solvers::FermiOperatorExpansion::Compute(
              n, H, S, mu, kT, lambda_min, lambda_max);
          P_new = foe_res.P;
        }

        // Compute approximate eigenvalues via Rayleigh quotient for energy_fn.
        // eps_k ≈ <P_k|H|P_k> / <P_k|S|P_k> — use diagonal of H as proxy.
        std::vector<double> approx_evals(n);
        for (std::size_t i = 0; i < n; ++i)
          approx_evals[i] = H[i * n + i];

        double E = energy_fn(P_new, approx_evals);
        res.energy_history.push_back(E);

        if (std::fabs(E - E_prev) < tol) {
          res.converged = true;
          res.energy = E;
          res.P = P_new;
          res.eigenvalues = approx_evals;
          return res;
        }
        E_prev = E;

        // Simple mixing for R2/R3 (DIIS on density matrix).
        std::vector<double> P_next(n * n, 0.0);
        double res_norm = 0.0;
        for (std::size_t idx = 0; idx < n * n; ++idx) {
          double d = P_new[idx] - P[idx];
          res_norm += d * d;
        }
        double rms = std::sqrt(res_norm / static_cast<double>(n * n));
        double eff_alpha = std::max(alpha / (1.0 + rms), 0.05);
        for (std::size_t idx = 0; idx < n * n; ++idx)
          P_next[idx] = eff_alpha * P_new[idx] + (1.0 - eff_alpha) * P[idx];

        P_history.push_back(P);
        F_history.push_back(P_new);
        if (static_cast<int>(P_history.size()) > pulay_depth) {
          P_history.erase(P_history.begin());
          F_history.erase(F_history.begin());
        }

        P = P_next;
        res.energy = E;
        res.P = P;
        res.eigenvalues = approx_evals;
        continue;
      }

      // --- R0/R1: eigensolve-based solvers ---
      // Step 1: tmp = H X  (n x n_retained, row-major)
      std::vector<double> tmp(n * n_retained, 0.0);
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n_retained; ++j) {
          double s = 0.0;
          for (std::size_t k = 0; k < n; ++k)
            s += H[i * n + k] * X[k * n_retained + j];
          tmp[i * n_retained + j] = s;
        }
      // Step 2: H' = X^T tmp (n_retained x n_retained, row-major)
      std::vector<double> Hp(n_retained * n_retained, 0.0);
      for (std::size_t k = 0; k < n_retained; ++k)
        for (std::size_t l = 0; l < n_retained; ++l) {
          double s = 0.0;
          for (std::size_t i = 0; i < n; ++i)
            s += Xt[k * n + i] * tmp[i * n_retained + l];
          Hp[k * n_retained + l] = s;
        }
      // Solve standard eigenproblem H' y = e y (n_retained x n_retained)
      std::vector<double> Hp_work = Hp;
      std::vector<double> y_eval(n_retained, 0.0);
      std::vector<double> y_evec(n_retained * n_retained, 0.0);

      if (regime == solvers::SolverRegime::kR1_ChFSI && n_retained > 20) {
        // R1: Chebyshev-filtered subspace iteration.
        // Estimate spectral window from Gershgorin bounds.
        double lo = 1e30, hi = -1e30;
        for (std::size_t i = 0; i < n_retained; ++i) {
          double diag = Hp[i * n_retained + i];
          double radius = 0.0;
          for (std::size_t j = 0; j < n_retained; ++j)
            if (j != i) radius += std::fabs(Hp[i * n_retained + j]);
          lo = std::min(lo, diag - radius);
          hi = std::max(hi, diag + radius);
        }
        auto chfsi_res = solvers::ChFSI::Solve(
            n_retained, Hp, std::vector<double>(n_retained * n_retained, 0.0),
            n_occ, lo, hi, 12, 50, 1e-9);
        if (chfsi_res.converged) {
          y_eval = chfsi_res.eigenvalues;
          y_evec.assign(n_retained * n_retained, 0.0);
          for (std::size_t k = 0; k < n_occ; ++k)
            for (std::size_t j = 0; j < n_retained; ++j)
              y_evec[j + k * n_retained] = chfsi_res.eigenvectors[k * n_retained + j];
        } else {
          // Fall back to dense eig.
          regime = solvers::SolverRegime::kR0_BatchDense;
        }
      }

      if (regime != solvers::SolverRegime::kR1_ChFSI || n_retained <= 20) {
        // R0: dense eigensolve via LAPACK dsyev_.
        int nn2 = static_cast<int>(n_retained);
        char jobz = 'V';
        char uplo = 'L';
        int lda = nn2;
        int lwork = -1;
        double wkopt = 0.0;
        int info = 0;
        dsyev_(&jobz, &uplo, &nn2, Hp_work.data(), &lda, y_eval.data(), &wkopt, &lwork, &info);
        if (info != 0) return res;
        lwork = static_cast<int>(wkopt);
        std::vector<double> work(static_cast<std::size_t>(lwork));
        dsyev_(&jobz, &uplo, &nn2, Hp_work.data(), &lda, y_eval.data(), work.data(), &lwork, &info);
        if (info != 0) return res;
        y_evec = Hp_work;  // column-major eigenvectors
      }
      // Back-transform: C = X y (n x n_retained)
      // y_evec is column-major n_retained x n_retained: y_evec[j + k*n_retained] = comp j of evec k
      // C[i, k] = sum_j X[i, j] * y_evec[j + k*n_retained]
      std::vector<double> C_evec(n * n_retained, 0.0);  // row-major
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < n_retained; ++k) {
          double s = 0.0;
          for (std::size_t j = 0; j < n_retained; ++j)
            s += X[i * n_retained + j] * y_evec[j + k * n_retained];
          C_evec[i * n_retained + k] = s;
        }
      // Pack into EigenResult format (same as SolveGeneralized: evec[k*n + j])
      tides::solvers::EigenResult eig;
      eig.eigenvalues = y_eval;
      eig.eigenvectors.resize(n * n_retained, 0.0);
      for (std::size_t k = 0; k < n_retained; ++k)
        for (std::size_t i = 0; i < n; ++i)
          eig.eigenvectors[k * n + i] = C_evec[i * n_retained + k];
      eig.ok = true;

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
        // Symmetrize: dsyrk_ uplo='L' writes column-major lower = row-major
        // upper. Copy row-major upper (computed) to row-major lower (zeros).
        for (std::size_t i = 0; i < n; ++i)
          for (std::size_t j = i + 1; j < n; ++j)
            P_new[j * n + i] = P_new[i * n + j];
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
          // Adaptive alpha: increase mixing as SCF converges (residual shrinks).
          double res_norm = 0.0;
          for (std::size_t idx = 0; idx < n * n; ++idx) {
            double d = P_new[idx] - P[idx];
            res_norm += d * d;
          }
          double rms = std::sqrt(res_norm / static_cast<double>(n * n));
          // Start conservative, increase as residual shrinks.
          double eff_a = std::min(std::max(alpha / (1.0 + rms), 0.05), 0.8);
          for (std::size_t idx = 0; idx < n * n; ++idx)
            P_next[idx] = eff_a * P_new[idx] + (1.0 - eff_a) * P[idx];
        }
      } else {
        // Simple mixing with Kerker-style damping using RMS residual.
        // Adaptive: increase alpha as residual decreases.
        double res_norm = 0.0;
        for (std::size_t idx = 0; idx < n * n; ++idx) {
          double d = P_new[idx] - P[idx];
          res_norm += d * d;
        }
        double rms = std::sqrt(res_norm / static_cast<double>(n * n));
        double eff_alpha = std::min(std::max(alpha / (1.0 + rms), 0.05), 0.8);
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
