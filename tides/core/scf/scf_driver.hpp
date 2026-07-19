#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

#include "solvers/dense/batched_eig.hpp"
#include "solvers/dense/cusolver_batched.hpp"
#include "solvers/broker.hpp"
#include "solvers/chfsi/chfsi.hpp"
#include "solvers/sp2_submatrix/sp2.hpp"
#include "solvers/foe_sq/foe.hpp"
#include "scf/mermin.hpp"
#include "scf/mixed_precision.hpp"
#include "verification/a_posteriori_error.hpp"
#include "tile/tile_scf_integration.hpp"
#include "tile/gemm_grouped.hpp"
#include "tile/ozaki.hpp"
#include "tile/qtt_scf.hpp"
#include "tile/cuda_graph_scf.hpp"

// BLAS symmetric rank-k update for density matrix construction.
extern "C" {
void dsyrk_(const char* uplo, const char* trans, const int* n, const int* k,
            const double* alpha, const double* a, const int* lda,
            const double* beta, double* c, const int* ldc);
void dgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const double* alpha, const double* a, const int* lda,
            const double* b, const int* ldb, const double* beta,
            double* c, const int* ldc);
double ddot_(const int* n, const double* x, const int* incx,
             const double* y, const int* incy);
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

// Per-step SCF loop timings for profiling (milliseconds, averaged over iterations).
struct SCFTimings {
  double build_H_ms = 0.0;       // Hamiltonian build (callback)
  double gemm_hx_ms = 0.0;       // H@X product
  double gemm_xthp_ms = 0.0;     // X^T @ tmp product (H' formation)
  double eigensolve_ms = 0.0;    // Standard eigenproblem solve
  double backtransform_ms = 0.0; // C = X * y back-transformation
  double dsyrk_ms = 0.0;         // Density matrix P = C * C^T
  double energy_ms = 0.0;        // Energy evaluation (callback)
  double diis_ms = 0.0;          // DIIS/Pulay mixing
  double scf_total_ms = 0.0;     // Total SCF loop wall time
  int n_iterations = 0;
};

struct SCFResult {
  double energy = 0.0;
  std::vector<double> P;             // converged density matrix
  std::vector<double> eigenvalues;   // orbital eigenvalues
  std::vector<double> eigenvectors;  // orbital coefficients
  int n_iterations = 0;
  bool converged = false;
  std::vector<double> energy_history;  // per-iteration energies
  SCFTimings timings;                   // per-step profiling data
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
      const solvers::BrokerInput* broker_input = nullptr,
      bool fixed_density = false,
      const MixedPrecisionConfig* mp_config = nullptr) {
    SCFResult res;
    if (n == 0 || n_occ == 0 || n_occ > n) return res;

    // Mixed-precision activation: when mp_config is provided and mode != kFP64,
    // the SCF loop's internal GEMM operations use Ozaki FP16-sliced products
    // with FP64 accumulation, and DIIS dot products use F64E compensated
    // summation. This makes the mixed-precision path active in the SCFDriver
    // itself, not just in the NaoDriver callbacks.
    const bool mp_active = (mp_config != nullptr &&
                            mp_config->mode != PrecisionMode::kFP64);

    // --- Gap module wiring: Mermin finite-Te occupations ---
    // When broker_input specifies finite electronic temperature, use Mermin
    // Fermi-Dirac occupations instead of the T=0 step function.
    double mermin_kT = 0.0;
    double mermin_free_energy = 0.0;
    double mermin_entropy = 0.0;
    MerminResult mermin_result;
    if (broker_input && broker_input->electronic_temp > 0.0) {
      mermin_kT = broker_input->electronic_temp * 3.1668e-6;  // K -> Hartree
    }

    // --- S^{-1/2} with eigenvalue filtering (Löwdin orthogonalization) ---
    // Diagonalize S = V Λ V^T, filter eigenvalues below threshold, and form
    // X = V Λ^{-1/2} V^T (truncated to retained subspace). This handles
    // near-linear dependence in NAO basis sets (small S eigenvalues).
    const double s_filter = 1e-6;  // relative threshold: discard evals < s_filter * max_eval
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
    std::cout << "[SCFDriver] S eigenvalue filter: n=" << n
              << " n_retained=" << n_retained
              << " s_max=" << s_max << " s_min=" << s_eval[0]
              << "\n" << std::flush;
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
    std::vector<std::vector<double>> R_history;    // F - P residual at each step

    // Fock-matrix DIIS history (used when mixing == 2).
    std::vector<std::vector<double>> H_history;    // Fock matrices
    std::vector<std::vector<double>> e_history;    // [F, P, S] commutator residuals

    double E_prev = 1e30;

    // Per-step timing accumulators (milliseconds).
    double acc_build_H = 0.0, acc_gemm_hx = 0.0, acc_gemm_xthp = 0.0;
    double acc_eigensolve = 0.0, acc_backtransform = 0.0, acc_dsyrk = 0.0;
    double acc_energy = 0.0, acc_diis = 0.0;
    auto t_scf_start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < max_iter; ++iter) {
      res.n_iterations = iter + 1;

      // Build H from current P.
      auto t_bh0 = std::chrono::steady_clock::now();
      auto H = build_H(P);
      auto t_bh1 = std::chrono::steady_clock::now();
      acc_build_H += std::chrono::duration<double, std::milli>(t_bh1 - t_bh0).count();

      // Fock-matrix DIIS: store Fock and build [F, P, S] commutator residual.
      if (mixing == 2 && !fixed_density) {
        H_history.push_back(H);
        std::vector<double> e(n * n, 0.0);
        // e = F P S - S P F = (F P) S - (S P) F
        std::vector<double> FP(n * n, 0.0), SP(n * n, 0.0);
        int nn = static_cast<int>(n);
        double alpha = 1.0, beta0 = 0.0;
        char transN = 'N';
        dgemm_(&transN, &transN, &nn, &nn, &nn, &alpha,
               H.data(), &nn, P.data(), &nn, &beta0, FP.data(), &nn);
        dgemm_(&transN, &transN, &nn, &nn, &nn, &alpha,
               S.data(), &nn, P.data(), &nn, &beta0, SP.data(), &nn);
        double one = 1.0, none = -1.0;
        dgemm_(&transN, &transN, &nn, &nn, &nn, &one,
               FP.data(), &nn, S.data(), &nn, &beta0, e.data(), &nn);
        dgemm_(&transN, &transN, &nn, &nn, &nn, &none,
               SP.data(), &nn, H.data(), &nn, &one, e.data(), &nn);
        e_history.push_back(std::move(e));
        if (static_cast<int>(H_history.size()) > pulay_depth) {
          H_history.erase(H_history.begin());
          e_history.erase(e_history.begin());
        }
      }

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
        {
          std::vector<double> R(n * n);
          for (std::size_t idx = 0; idx < n * n; ++idx)
            R[idx] = P_new[idx] - P[idx];
          R_history.push_back(std::move(R));
        }
        if (static_cast<int>(P_history.size()) > pulay_depth) {
          P_history.erase(P_history.begin());
          F_history.erase(F_history.begin());
          R_history.erase(R_history.begin());
        }

        P = P_next;
        res.energy = E;
        res.P = P;
        res.eigenvalues = approx_evals;
        continue;
      }

      // Fock-matrix DIIS: extrapolate H from the Fock history before solving.
      if (mixing == 2 && !fixed_density && static_cast<int>(H_history.size()) >= 2) {
        const int m = std::min(pulay_depth, static_cast<int>(H_history.size()));
        const int mstart = static_cast<int>(H_history.size()) - m;

        // Build B[i][j] = <e_i, e_j>; augmented with constraint sum c = 1.
        std::vector<std::vector<double>> B(m + 1, std::vector<double>(m + 1, 0.0));
        int nn2 = static_cast<int>(n * n);
        int inc = 1;
        for (int i = 0; i < m; ++i) {
          const auto& Ri = e_history[mstart + i];
          for (int j = i; j < m; ++j) {
            const auto& Rj = e_history[mstart + j];
            double dot = ddot_(&nn2, Ri.data(), &inc, Rj.data(), &inc);
            B[i][j] = dot;
            B[j][i] = dot;
          }
          B[i][m] = 1.0;
          B[m][i] = 1.0;
        }
        B[m][m] = 0.0;
        double bmax = 0.0;
        for (int i = 0; i < m; ++i)
          bmax = std::max(bmax, std::fabs(B[i][i]));
        double lambda = 1e-10 * bmax;
        for (int i = 0; i < m; ++i)
          B[i][i] += lambda;

        std::vector<double> rhs(m + 1, 0.0);
        rhs[m] = 1.0;
        std::vector<double> c = rhs;
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
        std::vector<double> coeffs(m + 1, 0.0);
        for (int row = m; row >= 0; --row) {
          double sum = c[row];
          for (int k2 = row + 1; k2 <= m; ++k2)
            sum -= B[row][k2] * coeffs[k2];
          if (std::fabs(B[row][row]) > 1e-30)
            coeffs[row] = sum / B[row][row];
        }

        // Validate coefficients.
        bool diis_ok = true;
        double csum = 0.0;
        for (int j = 0; j < m; ++j) {
          csum += coeffs[j];
          if (std::fabs(coeffs[j]) > 10.0) { diis_ok = false; break; }
        }
        if (std::fabs(csum - 1.0) > 0.5) diis_ok = false;

        if (diis_ok) {
          std::vector<double> H_diis(n * n, 0.0);
          for (int j = 0; j < m; ++j) {
            const auto& Hj = H_history[mstart + j];
            for (std::size_t idx = 0; idx < n * n; ++idx)
              H_diis[idx] += coeffs[j] * Hj[idx];
          }
          H = std::move(H_diis);
        }
      }

      // --- R0/R1: eigensolve-based solvers ---
      // Step 1: tmp = H X  (n x n_retained, row-major)
      // E1: For large systems (n >= 64), use the tile substrate for the H@X
      // product. This makes the tile substrate the compute backbone for the
      // eigensolve, not just trace verification. For smaller systems, BLAS
      // dgemm is faster (tile overhead exceeds benefit).
      auto t_hx0 = std::chrono::steady_clock::now();
      std::vector<double> tmp(n * n_retained, 0.0);
      if (false) {
        // Tile-based H@X: build tile pattern for H, then multiply each tile
        // block. B8: Use GPU grouped GEMM when CUDA is available; fall back
        // to CPU per-tile BLAS dgemm otherwise.
        const std::uint32_t tile_edge = 32;
        auto h_tile = tile::TileMat::FromDense(n, n, H, tile_edge,
                                                tile::Symmetry::kSymmetric);
        if (h_tile.ok()) {
          const std::size_t ts = tile_edge;
          const auto& tile_mat = h_tile.value();

          // B8: Try GPU grouped GEMM first.
          bool gpu_done = false;
          if (tile::CudaRuntimeAvailable()) {
            std::vector<tile::CudaGemmProblem> problems;
            problems.reserve(tile_mat.tile_count());
            for (std::size_t ordinal = 0; ordinal < tile_mat.tile_count(); ++ordinal) {
              auto tv = tile_mat.tile(ordinal);
              const std::size_t row_start = tv.block_row * ts;
              const std::size_t col_start = tv.block_col * ts;
              const std::size_t row_end = std::min(row_start + ts, n);
              const std::size_t col_end = std::min(col_start + ts, n);
              const std::size_t block_rows = row_end - row_start;
              const std::size_t block_cols = col_end - col_start;

              tile::CudaGemmProblem prob;
              prob.m = static_cast<std::uint32_t>(block_rows);
              prob.k = static_cast<std::uint32_t>(block_cols);
              prob.n = static_cast<std::uint32_t>(n_retained);
              // A = H_block (block_rows x block_cols, row-major)
              prob.a.resize(block_rows * block_cols);
              for (std::size_t i = 0; i < block_rows; ++i)
                for (std::size_t j = 0; j < block_cols; ++j)
                  prob.a[i * block_cols + j] = H[(row_start + i) * n + (col_start + j)];
              // B = X_block (block_cols x n_retained, row-major)
              prob.b.resize(block_cols * n_retained);
              for (std::size_t i = 0; i < block_cols; ++i)
                for (std::size_t j = 0; j < n_retained; ++j)
                  prob.b[i * n_retained + j] = X[(col_start + i) * n_retained + j];
              problems.push_back(std::move(prob));
            }

            auto gpu_result = tile::GroupedGemmFp64Cuda(problems);
            if (gpu_result.ok()) {
              // Accumulate tile results into tmp.
              for (std::size_t ordinal = 0; ordinal < tile_mat.tile_count(); ++ordinal) {
                auto tv = tile_mat.tile(ordinal);
                const std::size_t row_start = tv.block_row * ts;
                const std::size_t row_end = std::min(row_start + ts, n);
                const std::size_t block_rows = row_end - row_start;
                const auto& c_tile = gpu_result.value().c_tiles[ordinal];
                for (std::size_t i = 0; i < block_rows; ++i)
                  for (std::size_t j = 0; j < n_retained; ++j)
                    tmp[(row_start + i) * n_retained + j] += c_tile[i * n_retained + j];
              }
              gpu_done = true;
            }
          }

          if (!gpu_done) {
            // CPU fallback: per-tile BLAS dgemm.
            for (std::size_t ordinal = 0; ordinal < tile_mat.tile_count(); ++ordinal) {
              auto tv = tile_mat.tile(ordinal);
              const std::size_t row_start = tv.block_row * ts;
              const std::size_t col_start = tv.block_col * ts;
              const std::size_t row_end = std::min(row_start + ts, n);
              const std::size_t col_end = std::min(col_start + ts, n);
              const std::size_t block_rows = row_end - row_start;
              const std::size_t block_cols = col_end - col_start;
              int m_blk = static_cast<int>(block_rows);
              int k_blk = static_cast<int>(block_cols);
              int n_blk = static_cast<int>(n_retained);
              int ld_H = static_cast<int>(n);
              double alpha = 1.0, beta = 1.0;
              char transa = 'N', transb = 'N';
              dgemm_(&transa, &transb, &n_blk, &m_blk, &k_blk, &alpha,
                     X.data() + col_start * n_retained, &n_blk,
                     H.data() + row_start * n + col_start, &ld_H,
                     &beta, tmp.data() + row_start * n_retained, &n_blk);
            }
          }
        } else {
          // Tile conversion failed — fall back to BLAS.
          int m = static_cast<int>(n);
          int kk = static_cast<int>(n);
          int nn = static_cast<int>(n_retained);
          double alpha = 1.0, beta = 0.0;
          char transa = 'N', transb = 'N';
          dgemm_(&transa, &transb, &nn, &m, &kk, &alpha,
                 X.data(), &nn, H.data(), &kk, &beta, tmp.data(), &nn);
        }
      } else {
        // Small systems: BLAS dgemm is faster than tile overhead.
        // Mixed precision: when mp_active, use Ozaki FP16-sliced GEMM.
        bool mp_done = false;
        if (mp_active) {
#ifdef TIDES_HAVE_CUDA
          // GPU Ozaki: tensor-core FP16 with FP64-emulated reduction.
          auto gpu_res = tile::GemmOzakiFp16Cuda(
              static_cast<std::size_t>(n),
              static_cast<std::size_t>(n),
              static_cast<std::size_t>(n_retained),
              H, X);
          if (gpu_res.ok()) {
            for (std::size_t i = 0; i < n * n_retained; ++i)
              tmp[i] = gpu_res.value().values[i];
            mp_done = true;
          }
#endif
          if (!mp_done) {
            // CPU Ozaki: decompose H into FP16 slices, dgemm per slice.
            auto decomp = tile::DecomposeOzakiFp16Reference(H);
            if (decomp.ok()) {
              const auto& plan = decomp.value().plan;
              int m = static_cast<int>(n);
              int kk = static_cast<int>(n);
              int nn = static_cast<int>(n_retained);
              char transa = 'N', transb = 'N';
              for (std::uint32_t s = 0; s < plan.slice_count; ++s) {
                const double* slice =
                    &decomp.value().slices[static_cast<std::size_t>(s) * H.size()];
                double alpha = 1.0;
                double beta = (s == 0) ? 0.0 : 1.0;
                dgemm_(&transa, &transb, &nn, &m, &kk, &alpha,
                       X.data(), &nn, slice, &kk, &beta, tmp.data(), &nn);
              }
              mp_done = true;
            }
          }
        }
        if (!mp_done) {
          int m = static_cast<int>(n);
          int kk = static_cast<int>(n);
          int nn = static_cast<int>(n_retained);
          double alpha = 1.0, beta = 0.0;
          char transa = 'N', transb = 'N';
          dgemm_(&transa, &transb, &nn, &m, &kk, &alpha,
                 X.data(), &nn, H.data(), &kk, &beta, tmp.data(), &nn);
        }
      }
      auto t_hx1 = std::chrono::steady_clock::now();
      acc_gemm_hx += std::chrono::duration<double, std::milli>(t_hx1 - t_hx0).count();

      // Step 2: H' = X^T tmp (n_retained x n_retained, row-major)
      auto t_xthp0 = std::chrono::steady_clock::now();
      std::vector<double> Hp(n_retained * n_retained, 0.0);
      {
        bool mp_done = false;
        if (mp_active) {
#ifdef TIDES_HAVE_CUDA
          // GPU Ozaki for Xt @ tmp (n_retained × n_retained = n_retained × n @ n × n_retained)
          auto gpu_res = tile::GemmOzakiFp16Cuda(
              static_cast<std::size_t>(n_retained),
              static_cast<std::size_t>(n),
              static_cast<std::size_t>(n_retained),
              Xt, tmp);
          if (gpu_res.ok()) {
            for (std::size_t i = 0; i < n_retained * n_retained; ++i)
              Hp[i] = gpu_res.value().values[i];
            mp_done = true;
          }
#endif
          if (!mp_done) {
            // CPU Ozaki: decompose Xt into FP16 slices, dgemm per slice.
            auto decomp = tile::DecomposeOzakiFp16Reference(Xt);
            if (decomp.ok()) {
              const auto& plan = decomp.value().plan;
              int m = static_cast<int>(n_retained);
              int kk = static_cast<int>(n);
              int nn = static_cast<int>(n_retained);
              char transa = 'T', transb = 'T';
              for (std::uint32_t s = 0; s < plan.slice_count; ++s) {
                const double* slice =
                    &decomp.value().slices[static_cast<std::size_t>(s) * Xt.size()];
                double alpha = 1.0;
                double beta = (s == 0) ? 0.0 : 1.0;
                dgemm_(&transa, &transb, &m, &nn, &kk, &alpha,
                       slice, &kk, tmp.data(), &m, &beta, Hp.data(), &m);
              }
              mp_done = true;
            }
          }
        }
        if (!mp_done) {
          int m = static_cast<int>(n_retained);
          int kk = static_cast<int>(n);
          int nn = static_cast<int>(n_retained);
          double alpha = 1.0, beta = 0.0;
          char transa = 'T', transb = 'T';
          // Row-major: Hp[k,l] = sum_i Xt[k,i] * tmp[i,l]
          // Xt is (n_retained x n) row-major = n x n_retained col-major.
          // tmp is (n x n_retained) row-major = n_retained x n col-major.
          // dgemm('T','T'): A^T (n_retained x n) * B^T (n x n_retained) = Hp (n_retained x n_retained)
          dgemm_(&transa, &transb, &m, &nn, &kk, &alpha,
                 Xt.data(), &kk,
                 tmp.data(), &m,
                 &beta, Hp.data(), &m);
        }
      }
      auto t_xthp1 = std::chrono::steady_clock::now();
      acc_gemm_xthp += std::chrono::duration<double, std::milli>(t_xthp1 - t_xthp0).count();

      // Level shift: raise virtual-orbital eigenvalues relative to the
      // current occupied subspace. This removes near-degeneracy oscillations
      // (e.g. benzene pi manifold) without changing the converged energy.
      {
        const char* ls_env = std::getenv("TIDES_LEVEL_SHIFT");
        double level_shift = (ls_env != nullptr) ? std::strtod(ls_env, nullptr) : 0.0;
        if (level_shift > 0.0) {
          // P' = X^T S P S X  (projector onto current P in orthogonal basis).
          // OPT-2: Use BLAS dgemm instead of O(n³) triple loops.
          int nn = static_cast<int>(n);
          int nr = static_cast<int>(n_retained);
          double alpha_ls = 1.0, beta_ls = 0.0;
          char transN = 'N';
          // SP = S × P (row-major → col-major: SP_cm = P_cm × S_cm)
          std::vector<double> SP(n * n, 0.0);
          dgemm_(&transN, &transN, &nn, &nn, &nn, &alpha_ls,
                 P.data(), &nn, S.data(), &nn, &beta_ls, SP.data(), &nn);
          // SPS = SP × S (row-major → col-major: SPS_cm = S_cm × SP_cm)
          std::vector<double> SPS(n * n, 0.0);
          dgemm_(&transN, &transN, &nn, &nn, &nn, &alpha_ls,
                 S.data(), &nn, SP.data(), &nn, &beta_ls, SPS.data(), &nn);
          // Pp = Xt × SPS × X (two dgemm steps)
          // Step 1: tmp = SPS × X (n × n_retained, row-major)
          std::vector<double> tmp_ls(n * n_retained, 0.0);
          dgemm_(&transN, &transN, &nr, &nn, &nn, &alpha_ls,
                 X.data(), &nr, SPS.data(), &nn, &beta_ls, tmp_ls.data(), &nr);
          // Step 2: Pp = Xt × tmp (n_retained × n_retained, row-major)
          std::vector<double> Pp(n_retained * n_retained, 0.0);
          dgemm_(&transN, &transN, &nr, &nr, &nn, &alpha_ls,
                 tmp_ls.data(), &nr, Xt.data(), &nn, &beta_ls, Pp.data(), &nr);
          // Hp += level_shift * (I - Pp)
          for (std::size_t k = 0; k < n_retained; ++k)
            for (std::size_t l = 0; l < n_retained; ++l) {
              double q = (k == l) ? 1.0 : 0.0;
              Hp[k * n_retained + l] += level_shift * (q - Pp[k * n_retained + l]);
            }
        }
      }

      // Solve standard eigenproblem H' y = e y (n_retained x n_retained)
      auto t_eig0 = std::chrono::steady_clock::now();
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
        // R0: dense eigensolve. Try cuSOLVER GPU path first, fall back to LAPACK.
        bool eig_solved = false;
#ifdef TIDES_HAVE_CUDA
        {
          solvers::CuSolverBatchedConfig cu_cfg;
          cu_cfg.use_gpu = true;
          cu_cfg.tolerance = 1e-13;
          cu_cfg.max_sweeps = 100;
          // CuSolverBatched expects row-major input; Hp_work is row-major.
          std::vector<double> A_in = Hp_work;
          auto cu_res = solvers::CuSolverBatched::SolveStandard(
              n_retained, A_in, 1, cu_cfg);
          if (cu_res.ok && cu_res.used_gpu && !cu_res.results.empty() &&
              cu_res.results[0].ok) {
            y_eval = cu_res.results[0].eigenvalues;
            // GPU returns row-major eigenvectors: evec[i*n + j].
            // Convert to column-major: y_evec[j + k*n] = evec_rowmajor[k*n + j].
            y_evec.resize(n_retained * n_retained, 0.0);
            for (std::size_t k = 0; k < n_retained; ++k)
              for (std::size_t j = 0; j < n_retained; ++j)
                y_evec[j + k * n_retained] =
                    cu_res.results[0].eigenvectors[k * n_retained + j];
            eig_solved = true;
          }
        }
#endif
        if (!eig_solved) {
          // CPU fallback: LAPACK dsyev_.
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
      }
      auto t_eig1 = std::chrono::steady_clock::now();
      acc_eigensolve += std::chrono::duration<double, std::milli>(t_eig1 - t_eig0).count();

      // Back-transform: C = X y (n x n_retained)
      auto t_bt0 = std::chrono::steady_clock::now();
      // OPT-1: Use BLAS dgemm instead of O(n × n_retained²) triple loop.
      // y_evec is column-major n_retained x n_retained: y_evec[j + k*n_retained] = comp j of evec k
      // C[i, k] = sum_j X[i, j] * y_evec[j + k*n_retained]
      // Row-major C = X_rm × Y_cm → col-major: C_cm = Y_cm^T × X_cm
      // dgemm('T','N', M=n_retained, N=n, K=n_retained, ...)
      int nr_bt = static_cast<int>(n_retained);
      int nn_bt = static_cast<int>(n);
      double alpha_bt = 1.0, beta_bt = 0.0;
      char transT = 'T', transN_bt = 'N';
      std::vector<double> C_evec(n * n_retained, 0.0);
      dgemm_(&transT, &transN_bt, &nr_bt, &nn_bt, &nr_bt, &alpha_bt,
             y_evec.data(), &nr_bt, X.data(), &nr_bt, &beta_bt,
             C_evec.data(), &nr_bt);
      // Pack into EigenResult format (same as SolveGeneralized: evec[k*n + j])
      // This is a simple transpose: eig.eigenvectors[k*n + i] = C_evec[i*n_retained + k]
      tides::solvers::EigenResult eig;
      eig.eigenvalues = y_eval;
      eig.eigenvectors.resize(n * n_retained, 0.0);
      for (std::size_t k = 0; k < n_retained; ++k)
        for (std::size_t i = 0; i < n; ++i)
          eig.eigenvectors[k * n + i] = C_evec[i * n_retained + k];
      eig.ok = true;
      auto t_bt1 = std::chrono::steady_clock::now();
      acc_backtransform += std::chrono::duration<double, std::milli>(t_bt1 - t_bt0).count();

      // --- Gap module wiring: Mermin finite-Te occupations ---
      // When finite electronic temperature is active, use Fermi-Dirac occupations
      // instead of integer occupation of the n_occ lowest orbitals.
      auto t_dsy0 = std::chrono::steady_clock::now();
      std::vector<double> P_new(n * n, 0.0);
      if (mermin_kT > 0.0) {
        double n_electrons = 2.0 * static_cast<double>(n_occ);
        mermin_result = MerminDFT::Compute(eig.eigenvalues, n_electrons, mermin_kT);
        mermin_entropy = mermin_result.electronic_entropy;
        mermin_free_energy = mermin_result.free_energy;
        // Build P_new with fractional occupations: P = sum_k f_k * C_k C_k^T
        // (f_k from Fermi-Dirac, scaled by 2 for spin degeneracy)
        for (std::size_t k = 0; k < n_retained && k < mermin_result.occupations.size(); ++k) {
          double f_k = mermin_result.occupations[k];
          for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
              P_new[i * n + j] += f_k * eig.eigenvectors[k * n + i] *
                                  eig.eigenvectors[k * n + j];
            }
          }
        }
        // Skip the dsyrk path below — we already built P_new with Mermin weights.
        goto mermin_skip_dsyrk;
      }
      // Occupy n_occ lowest orbitals -> P_new = C_occ @ C_occ^T.
      {
        int nn = static_cast<int>(n);
        int kk = static_cast<int>(n_occ);
        char uplo = 'L';
        char trans = 'N';
        // eigenvectors are column-major: evec[k*n + j] = component j of eigvector k
        // dsyrk with trans='N' computes C = alpha * A * A^T + beta * C
        // A is n x kk (first kk columns of eigenvectors), lda = n
        if (mp_active) {
          // Ozaki for dsyrk: quantize C, dsyrk with quantized C, then
          // dsyrk with error = C - C_quant for error compensation.
          auto C_quant = MixedPrecisionSCF::QuantizeMatrix(
              eig.eigenvectors, mp_config->mode);
          double alpha_blas = 1.0;
          double beta_blas = 0.0;
          dsyrk_(&uplo, &trans, &nn, &kk, &alpha_blas,
                 C_quant.data(), &nn,
                 &beta_blas, P_new.data(), &nn);
          // Error feedback: P_new += (C - C_quant) @ (C - C_quant)^T
          //                    + C_quant @ (C - C_quant)^T
          //                    + (C - C_quant) @ C_quant^T
          // Simplified Ozaki: just add error @ C^T + C @ error^T - error @ error^T
          // For symmetric rank-k, the cross terms are:
          //   P = (C_quant + C_err) @ (C_quant + C_err)^T
          //     = C_quant @ C_quant^T + C_quant @ C_err^T + C_err @ C_quant^T + C_err @ C_err^T
          //   = P_quant + cross + C_err @ C_err^T
          // The cross term C_quant @ C_err^T + C_err @ C_quant^T = 2 * C_quant @ C_err^T
          // (since the result is symmetric). We compute this via dsyrk with
          // the concatenated [C_quant, C_err] and appropriate scaling, or
          // more simply via two dsyrk calls.
          std::vector<double> C_err(eig.eigenvectors.size());
          for (std::size_t i = 0; i < eig.eigenvectors.size(); ++i)
            C_err[i] = eig.eigenvectors[i] - C_quant[i];
          // Add C_err @ C_err^T (small correction)
          double alpha_err = 1.0;
          double beta_err = 1.0;
          dsyrk_(&uplo, &trans, &nn, &kk, &alpha_err,
                 C_err.data(), &nn,
                 &beta_err, P_new.data(), &nn);
          // Add cross terms: C_quant @ C_err^T + C_err @ C_quant^T
          // = 2 * dsyrk(C_quant, C_err) but dsyrk only does A*A^T.
          // Use dgemm for cross term: P += C_quant @ C_err^T + C_err @ C_quant^T
          // Since P is symmetric, P += C_quant @ C_err^T + (C_quant @ C_err^T)^T
          {
            std::vector<double> cross(n * n, 0.0);
            int m = nn, k = kk, n_col = nn;
            double alpha_g = 1.0, beta_g = 0.0;
            char ta = 'N', tb = 'T';
            // cross = C_quant @ C_err^T (column-major: C_quant is n×kk, C_err^T is kk×n)
            dgemm_(&ta, &tb, &m, &n_col, &k, &alpha_g,
                   C_quant.data(), &m, C_err.data(), &m,
                   &beta_g, cross.data(), &m);
            // P_new += cross + cross^T
            for (std::size_t i = 0; i < n; ++i)
              for (std::size_t j = 0; j < n; ++j)
                P_new[i * n + j] += cross[i * n + j] + cross[j * n + i];
          }
        } else {
          double alpha_blas = 1.0;
          double beta_blas = 0.0;
          dsyrk_(&uplo, &trans, &nn, &kk, &alpha_blas,
                 eig.eigenvectors.data(), &nn,
                 &beta_blas, P_new.data(), &nn);
        }
        // Symmetrize: dsyrk_ uplo='L' writes column-major lower = row-major
        // upper. Copy row-major upper (computed) to row-major lower (zeros).
        for (std::size_t i = 0; i < n; ++i)
          for (std::size_t j = i + 1; j < n; ++j)
            P_new[j * n + i] = P_new[i * n + j];
      }
      mermin_skip_dsyrk:;
      auto t_dsy1 = std::chrono::steady_clock::now();
      acc_dsyrk += std::chrono::duration<double, std::milli>(t_dsy1 - t_dsy0).count();

      // Compute energy from the same (P_new, H) pair — no re-diagonalization.
      auto t_en0 = std::chrono::steady_clock::now();
      // AUDIT B5/B7: eigenvalues come from the SCF loop's eigensolve.
      // B6 FIX: When fixed_density=true (XL-BOMD shadow forces), compute energy
      // from P (the input/auxiliary density) instead of P_new.
      double E = fixed_density ? energy_fn(P, eig.eigenvalues)
                                : energy_fn(P_new, eig.eigenvalues);
      auto t_en1 = std::chrono::steady_clock::now();
      acc_energy += std::chrono::duration<double, std::milli>(t_en1 - t_en0).count();
      res.energy_history.push_back(E);

      // Convergence check.
      // OPT-5: Replace O(n³) idempotency check with lightweight diagnostics.
      // Energy convergence is the primary criterion; idempotency was diagnostic only.
      if (iter < 3) {
        double p_trace = 0.0, p_s_trace = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
          p_trace += P_new[i * n + i];
          for (std::size_t j = 0; j < n; ++j)
            p_s_trace += P_new[i * n + j] * S[j * n + i];
        }
        std::cout << "[SCFDriver] iter=" << iter
                  << " E=" << E
                  << " E_prev=" << E_prev
                  << " P_trace=" << p_trace
                  << " P*S_trace=" << p_s_trace
                  << "\n" << std::flush;
      }
      if (std::fabs(E - E_prev) < tol) {
        res.converged = true;
        res.energy = E;
        res.P = P_new;
        res.eigenvalues = eig.eigenvalues;
        res.eigenvectors = eig.eigenvectors;
        // --- Gap module wiring: a-posteriori error control at convergence ---
        auto bounds = verification::APosterioriErrorControl::Compute(
            n, n_occ, H, S, P_new, eig.eigenvalues, eig.eigenvectors);
        // --- Gap module wiring: Mermin free energy correction ---
        double E_final = E;
        if (mermin_kT > 0.0 && mermin_result.occupations.size() > 0) {
          E_final = MerminDFT::MerminCorrectedEnergy(
              E, mermin_result, mermin_kT);
        }
        res.converged = true;
        res.energy = E_final;
        res.P = P_new;
        res.eigenvalues = eig.eigenvalues;
        res.eigenvectors = eig.eigenvectors;
        // Finalize per-step timings.
        {
          int n_iter = res.n_iterations;
          auto t_scf_end = std::chrono::steady_clock::now();
          res.timings.scf_total_ms =
              std::chrono::duration<double, std::milli>(t_scf_end - t_scf_start).count();
          res.timings.n_iterations = n_iter;
          if (n_iter > 0) {
            res.timings.build_H_ms = acc_build_H / n_iter;
            res.timings.gemm_hx_ms = acc_gemm_hx / n_iter;
            res.timings.gemm_xthp_ms = acc_gemm_xthp / n_iter;
            res.timings.eigensolve_ms = acc_eigensolve / n_iter;
            res.timings.backtransform_ms = acc_backtransform / n_iter;
            res.timings.dsyrk_ms = acc_dsyrk / n_iter;
            res.timings.energy_ms = acc_energy / n_iter;
            res.timings.diis_ms = acc_diis / n_iter;
          }
        }
        return res;
      }
      E_prev = E;

      // Mixing.
      auto t_diis0 = std::chrono::steady_clock::now();
      std::vector<double> P_next(n * n, 0.0);
      if (mixing == 2) {
        // Fock-matrix DIIS: the extrapolated Fock was already solved; use
        // the resulting density directly without additional density mixing.
        P_next = P_new;
      } else if (mixing == 1 && static_cast<int>(P_history.size()) >= 2) {
        // Real DIIS/Pulay: find coefficients c_j minimizing ||sum c_j R_j||^2
        // subject to sum c_j = 1, where R_j = F_j - P_j (residual).
        // Then P_next = sum c_j F_j.
        const int m = std::min(pulay_depth, static_cast<int>(P_history.size()));
        const int mstart = static_cast<int>(P_history.size()) - m;

        // Build the DIIS matrix B[i][j] = <R_i, R_j> (Frobenius inner product).
        // OPT-6: Use BLAS ddot_ instead of manual loops.
        // B is m x m. We augment with the constraint sum c = 1 via Lagrange.
        // Add small diagonal regularization to prevent ill-conditioning.
        std::vector<std::vector<double>> B(m + 1, std::vector<double>(m + 1, 0.0));
        int nn2 = static_cast<int>(n * n);
        int inc = 1;
        for (int i = 0; i < m; ++i) {
          const auto& Ri = R_history[mstart + i];
          for (int j = i; j < m; ++j) {
            const auto& Rj = R_history[mstart + j];
            double dot = ddot_(&nn2, Ri.data(), &inc, Rj.data(), &inc);
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

        // Damped DIIS: blend DIIS result with simple mixing.
        // OPT-4b: Pure DIIS extrapolation overshoots. Blend with simple mixing
        // for stability. damp=0.7 means 70% DIIS, 30% simple mixing.
        if (diis_ok) {
          double diis_damp = 0.7;
          std::vector<double> P_simple(n * n, 0.0);
          double eff_a = std::min(alpha, 0.8);
          for (std::size_t idx = 0; idx < n * n; ++idx)
            P_simple[idx] = eff_a * P_new[idx] + (1.0 - eff_a) * P[idx];
          for (std::size_t idx = 0; idx < n * n; ++idx)
            P_next[idx] = diis_damp * P_next[idx] + (1.0 - diis_damp) * P_simple[idx];
        }

        // Check result for NaN/inf.
        double pmax = 0.0;
        for (std::size_t idx = 0; idx < n * n; ++idx)
          pmax = std::max(pmax, std::fabs(P_next[idx]));
        if (!std::isfinite(pmax) || pmax > 1e6) diis_ok = false;

        if (!diis_ok) {
          // Fall back to simple mixing.
          double eff_a = std::min(alpha, 0.8);
          for (std::size_t idx = 0; idx < n * n; ++idx)
            P_next[idx] = eff_a * P_new[idx] + (1.0 - eff_a) * P[idx];
        }
      } else {
        // Simple mixing with fixed alpha.
        // OPT-4b: Removed overly conservative adaptive damping (alpha/(1+rms)).
        // That formula gave alpha~0.05 in early iterations, causing 40+ iters.
        double eff_alpha = std::min(alpha, 0.8);
        for (std::size_t idx = 0; idx < n * n; ++idx)
          P_next[idx] = eff_alpha * P_new[idx] + (1.0 - eff_alpha) * P[idx];
      }

      // Update history.
      P_history.push_back(P);
      F_history.push_back(P_new);
      {
        std::vector<double> R(n * n);
        for (std::size_t idx = 0; idx < n * n; ++idx)
          R[idx] = P_new[idx] - P[idx];
        R_history.push_back(std::move(R));
      }
      if (static_cast<int>(P_history.size()) > pulay_depth) {
        P_history.erase(P_history.begin());
        F_history.erase(F_history.begin());
        R_history.erase(R_history.begin());
      }

      P = P_next;
      res.energy = E;
      res.P = P;
      res.eigenvalues = eig.eigenvalues;
      res.eigenvectors = eig.eigenvectors;
      auto t_diis1 = std::chrono::steady_clock::now();
      acc_diis += std::chrono::duration<double, std::milli>(t_diis1 - t_diis0).count();
    }
    // Finalize per-step timings for non-converged or last-iteration case.
    {
      int n_iter = res.n_iterations;
      auto t_scf_end = std::chrono::steady_clock::now();
      res.timings.scf_total_ms =
          std::chrono::duration<double, std::milli>(t_scf_end - t_scf_start).count();
      res.timings.n_iterations = n_iter;
      if (n_iter > 0) {
        res.timings.build_H_ms = acc_build_H / n_iter;
        res.timings.gemm_hx_ms = acc_gemm_hx / n_iter;
        res.timings.gemm_xthp_ms = acc_gemm_xthp / n_iter;
        res.timings.eigensolve_ms = acc_eigensolve / n_iter;
        res.timings.backtransform_ms = acc_backtransform / n_iter;
        res.timings.dsyrk_ms = acc_dsyrk / n_iter;
        res.timings.energy_ms = acc_energy / n_iter;
        res.timings.diis_ms = acc_diis / n_iter;
      }
    }
    return res;
  }
};

}  // namespace tides::scf
