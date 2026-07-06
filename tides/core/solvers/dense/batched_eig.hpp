#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "common/status.hpp"

// Fortran LAPACK prototypes (already declared in WP2 dense_sym_eig.hpp, but
// re-declared here for the WP4 translation unit to be self-contained).
extern "C" {
void dsyev_(const char* jobz, const char* uplo, const int* n, double* a,
            const int* lda, double* w, double* work, const int* lwork, int* info);
}

namespace tides::solvers {

// Generalized eigenpair result for H x = e S x.
struct EigenResult {
  std::vector<double> eigenvalues;   // ascending
  std::vector<double> eigenvectors;  // column-major: vec[k*n + j] = component j of eigenvalue k
  bool ok = false;
};

// Batched dense symmetric eigensolver (R0 regime). Solves many small
// independent generalized eigenproblems H_k x = e S_k x simultaneously.
// This is the CPU reference; the GPU path uses cuSOLVER syevjBatched.
//
// The generalized problem H x = e S x is reduced to standard form via S^{-1/2}
// (computed via eigendecomposition of S, since S is SPD for NAO bases). The
// standard problem H' y = e y is then solved with dsyev_, and x = S^{-1/2} y.
//
// Observable (T4.1): residuals ||H x - e S x|| <= 1e-9 at n <= 400.
class BatchedDenseEig {
 public:
  // Solve a single generalized eigenproblem H x = e S x.
  // H and S are symmetric (row-major, lower triangle used); S must be SPD.
  // Returns n eigenpairs sorted ascending.
  static EigenResult SolveGeneralized(std::size_t n,
                                      const std::vector<double>& H,
                                      const std::vector<double>& S) {
    if (n == 0 || H.size() != n * n || S.size() != n * n) return {};
    if (n == 1) {
      return {{S[0] > 0 ? H[0] / S[0] : 0.0}, {1.0}, true};
    }

    // Step 1: Eigendecompose S = U s U^T, then S^{-1/2} = U s^{-1/2} U^T.
    std::vector<double> S_work = S;  // dsyev destroys input
    std::vector<double> s_eval(n), s_evec(n * n);
    if (!SolveSymmetric(n, S_work, s_eval, s_evec)) return {};

    // Check S is SPD (all eigenvalues > 0).
    for (double e : s_eval)
      if (e <= 0) return {};  // not SPD — fail

    // S^{-1/2} = U diag(1/sqrt(s)) U^T  (row-major).
    std::vector<double> S_inv_sqrt(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
          const double sk = s_evec[k * n + i] * s_evec[k * n + j];
          s += sk / std::sqrt(s_eval[k]);
        }
        S_inv_sqrt[i * n + j] = s;
      }

    // Step 2: H' = S^{-1/2} H S^{-1/2} (standard form).
    std::vector<double> Hprime(n * n, 0.0);
    // tmp = S^{-1/2} H
    std::vector<double> tmp(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (std::size_t k = 0; k < n; ++k)
          s += S_inv_sqrt[i * n + k] * H[k * n + j];
        tmp[i * n + j] = s;
      }
    // H' = tmp S^{-1/2}
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (std::size_t k = 0; k < n; ++k)
          s += tmp[i * n + k] * S_inv_sqrt[j * n + k];
        Hprime[i * n + j] = s;
      }

    // Symmetrize (roundoff hygiene).
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = i + 1; j < n; ++j) {
        double avg = 0.5 * (Hprime[i * n + j] + Hprime[j * n + i]);
        Hprime[i * n + j] = avg;
        Hprime[j * n + i] = avg;
      }

    // Step 3: Solve H' y = e y.
    std::vector<double> evals(n), evecs_yp(n * n);
    if (!SolveSymmetric(n, Hprime, evals, evecs_yp)) return {};

    // Step 4: x = S^{-1/2} y.
    EigenResult res;
    res.eigenvalues = evals;
    res.eigenvectors.resize(n * n);
    for (std::size_t k = 0; k < n; ++k)
      for (std::size_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i)
          s += S_inv_sqrt[j * n + i] * evecs_yp[k * n + i];
        res.eigenvectors[k * n + j] = s;
      }
    res.ok = true;
    return res;
  }

  // Solve a standard symmetric eigenproblem A x = e x (no overlap matrix).
  // Uses LAPACK dsyev_. Row-major input (lower triangle).
  static bool SolveSymmetric(std::size_t n, std::vector<double>& A,
                             std::vector<double>& evals,
                             std::vector<double>& evecs) {
    evals.assign(n, 0.0);
    evecs.assign(n * n, 0.0);
    if (n == 0) return true;
    if (n == 1) { evals[0] = A[0]; evecs[0] = 1.0; return true; }

#ifdef TIDES_HAVE_LAPACK
    // LAPACK expects column-major; for symmetric matrices row-major == col-major.
    char jobz = 'V';
    char uplo = 'L';
    int nn = static_cast<int>(n);
    int lda = nn;
    int info = 0;
    std::vector<double> w(n);
    int lwork = -1;
    double wkopt = 0.0;
    dsyev_(&jobz, &uplo, &nn, A.data(), &lda, w.data(), &wkopt, &lwork, &info);
    if (info != 0) return false;
    lwork = static_cast<int>(wkopt);
    std::vector<double> work(static_cast<std::size_t>(lwork));
    dsyev_(&jobz, &uplo, &nn, A.data(), &lda, w.data(), work.data(), &lwork, &info);
    if (info != 0) return false;
    evals = w;
    // A now holds eigenvectors as COLUMNS in column-major layout. Column k
    // (eigenvector k) has component j at A[k*n + j] (i.e. A[j + k*lda]).
    // Our convention: evecs[k*n + j] = component j of eigenvalue k.
    for (std::size_t k = 0; k < n; ++k)
      for (std::size_t j = 0; j < n; ++j)
        evecs[k * n + j] = A[k * n + j];
    return true;
#else
    return false;  // no LAPACK fallback in this minimal version
#endif
  }

  // Batched solve: k independent problems, each of size n_k. Returns all
  // eigenpairs. This is the R0 production path (GPU uses syevjBatched).
  static std::vector<EigenResult> SolveBatched(
      const std::vector<std::size_t>& sizes,
      const std::vector<std::vector<double>>& H_batch,
      const std::vector<std::vector<double>>& S_batch) {
    std::vector<EigenResult> results;
    if (H_batch.size() != S_batch.size() || H_batch.size() != sizes.size())
      return results;
    results.reserve(sizes.size());
    for (std::size_t b = 0; b < sizes.size(); ++b)
      results.push_back(SolveGeneralized(sizes[b], H_batch[b], S_batch[b]));
    return results;
  }

  // Compute residual ||H x - e S x|| for a generalized eigenpair.
  static double Residual(std::size_t n, const std::vector<double>& H,
                         const std::vector<double>& S, double e,
                         const std::vector<double>& x) {
    double max_r = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      double Hx = 0.0, Sx = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        Hx += H[i * n + j] * x[j];
        Sx += S[i * n + j] * x[j];
      }
      max_r = std::max(max_r, std::fabs(Hx - e * Sx));
    }
    return max_r;
  }
};

}  // namespace tides::solvers
