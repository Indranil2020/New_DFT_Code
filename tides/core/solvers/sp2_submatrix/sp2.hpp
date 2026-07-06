#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace tides::solvers {

// SP2 density-matrix purification (Niklasson, PRL 2002).
//
// The spectral projector P = theta(mu*S - H) is built by a fixed polynomial
// recursion of X^2 terms. Given a normalized matrix X0 (so that the spectrum
// of the generalized problem maps to [0,1]), the SP2 iteration:
//   X_{k+1} = 2 X_k - X_k^2     (if trace(X_k^2) > trace(X_k))  -> push up
//   X_{k+1} = X_k^2             (else)                           -> push down
// converges to the projector P with eigenvalues at 0 (unoccupied) and 1
// (occupied). The decision is based on whether the mid-spectrum is above or
// below 0.5.
//
// For the generalized problem H x = e S x, we work in the S-orthogonal basis:
// X0 = (mu*S - H) normalized so its S-eigenvalues are in [0,1]. The
// normalization uses spectral bounds (a few Lanczos steps or known bounds).
//
// Observable (T5.1): ||P^2 - P||_F <= 1e-10; tr(P S) = N_e <= 1e-10.
//
// For the CPU reference, H and S are dense matrices (small proxies); the GPU
// path uses WP1 TileMat filtered SpGEMM for the matrix products.

struct SP2Result {
  std::vector<double> P;      // density matrix (row-major, n x n)
  double trace_PS = 0.0;      // tr(P S) = N_e
  double idempotency_err = 0.0;  // ||P^2 - P||_F
  int n_iterations = 0;
  bool converged = false;
};

class SP2Purification {
 public:
  // Purify the density matrix for the generalized problem H x = e S x.
  //   n:       matrix dimension
  //   H, S:    symmetric matrices (row-major); S must be SPD
  //   n_e:     number of electrons (target trace)
  //   mu:      Fermi level (placed in the gap; if 0, estimated via bisection)
  //   lambda_min, lambda_max: spectral bounds of the generalized problem
  //   max_iter: maximum SP2 iterations (typically 30-40)
  //   tol:     idempotency convergence target
  static SP2Result Purify(std::size_t n, const std::vector<double>& H,
                         const std::vector<double>& S, double n_e,
                         double mu, double lambda_min, double lambda_max,
                         int max_iter = 40, double tol = 1e-12) {
    SP2Result res;
    if (n == 0 || H.size() != n * n || S.size() != n * n) return res;
    if (lambda_max <= lambda_min) return res;

    // X0 = 0.5 * (I - (H - mu*S) / (scale/2)) maps the spectrum so that
    // eigenvalues below mu -> X0 > 0.5 (occupied) and above mu -> X0 < 0.5.
    // The SP2 recursion pushes X0 -> projector P with the gap at 0.5.
    const double scale_half = (lambda_max - lambda_min) / 2.0;

    // If mu is not provided (0), bisect to find mu so tr(X0) = n_e.
    double mu_eff = mu;
    if (mu_eff == 0.0) {
      double mu_lo_shift = lambda_min, mu_hi_shift = lambda_max;
      for (int biter = 0; biter < 60; ++biter) {
        mu_eff = 0.5 * (mu_lo_shift + mu_hi_shift);
        double tr_X = 0.0;
        for (std::size_t i = 0; i < n; ++i)
          tr_X += 0.5 * (1.0 - (H[i * n + i] - mu_eff) / scale_half);
        if (tr_X < n_e) mu_lo_shift = mu_eff;
        else mu_hi_shift = mu_eff;
      }
      mu_eff = 0.5 * (mu_lo_shift + mu_hi_shift);
    }

    // Build X0 with the Fermi level.
    std::vector<double> X(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        X[i * n + j] = 0.5 * (S[i * n + j] -
                              (H[i * n + j] - mu_eff * S[i * n + j]) / scale_half);

    // Step 2: SP2 recursion. The decision (X^2 vs 2X-X^2) is based on whether
    // the mid-spectrum is above or below 0.5. We use tr(X^2) vs tr(X) as the
    // proxy (Niklasson's original criterion): if tr(X^2) < tr(X) the spectrum
    // is concentrated below 0.5 -> use X^2 to push down; else use 2X-X^2 to push up.
    for (int iter = 0; iter < max_iter; ++iter) {
      res.n_iterations = iter + 1;

      // X^2 in the S-metric: X2 = X * X (standard matrix product; for the
      // generalized problem the purification is X_{k+1} = X_k S X_k or
      // 2 X_k - X_k S X_k, but with S-orthonormalization the S factor is
      // S-congruent product: X2 = X S X.
      auto X2 = MatMulS(n, X, S, X);

      // Trace-based SP2 decision (robust variant): compare tr(X) to the
      // target n_occ. If tr(X) > n_occ, too many occupied -> push down (X^2).
      // If tr(X) < n_occ, too few -> push up (2X - X^2).
      double tr_X = TraceS(n, X, S);
      double tr_X2 = TraceS(n, X2, S);
      (void)tr_X2;

      std::vector<double> X_next(n * n, 0.0);
      if (tr_X > n_e) {
        X_next = X2;
      } else {
        for (std::size_t i = 0; i < n * n; ++i)
          X_next[i] = 2.0 * X[i] - X2[i];
      }

      // Check convergence: ||X^2 - X||_F -> 0 (idempotency).
      // Use the S-metric: ||X S X - X||.
      double err = 0.0;
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
          double diff = X2[i * n + j] - X[i * n + j];
          err += diff * diff;
        }
      res.idempotency_err = std::sqrt(err);

      X = X_next;

      if (res.idempotency_err < tol) {
        res.converged = true;
        break;
      }
    }

    // Step 3: P = X (the purified density matrix). Shift so that the
    // occupied states have eigenvalue 1 and unoccupied have 0.
    res.P = X;

    // Step 4: Compute trace(P S) = N_e.
    res.trace_PS = TraceS(n, res.P, S);

    return res;
  }

  // Compute the Frobenius norm ||A||_F.
  static double FrobeniusNorm(std::size_t n, const std::vector<double>& A) {
    double s = 0.0;
    for (std::size_t i = 0; i < n * n; ++i) s += A[i] * A[i];
    return std::sqrt(s);
  }

  // Compute tr(A B) (used for tr(P S)).
  static double TraceAB(std::size_t n, const std::vector<double>& A,
                        const std::vector<double>& B) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        s += A[i * n + j] * B[j * n + i];
    return s;
  }

  // Compute tr(A S) = sum_ij A_ij S_ji (the S-metric trace).
  static double TraceS(std::size_t n, const std::vector<double>& A,
                       const std::vector<double>& S) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        s += A[i * n + j] * S[i * n + j];
    return s;
  }

 private:
  // Matrix product C = A * B (standard, n x n).
  static std::vector<double> MatMul(std::size_t n, const std::vector<double>& A,
                                    const std::vector<double>& B) {
    std::vector<double> C(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (std::size_t k = 0; k < n; ++k)
          s += A[i * n + k] * B[k * n + j];
        C[i * n + j] = s;
      }
    return C;
  }

  // S-congruent product: C = A * S * B (for the generalized SP2).
  static std::vector<double> MatMulS(std::size_t n, const std::vector<double>& A,
                                     const std::vector<double>& S,
                                     const std::vector<double>& B) {
    // tmp = S * B
    std::vector<double> tmp(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (std::size_t k = 0; k < n; ++k)
          s += S[i * n + k] * B[k * n + j];
        tmp[i * n + j] = s;
      }
    // C = A * tmp
    std::vector<double> C(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (std::size_t k = 0; k < n; ++k)
          s += A[i * n + k] * tmp[k * n + j];
        C[i * n + j] = s;
      }
    return C;
  }
};

}  // namespace tides::solvers
