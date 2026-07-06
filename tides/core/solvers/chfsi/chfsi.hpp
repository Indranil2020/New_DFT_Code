#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "solvers/dense/batched_eig.hpp"

namespace tides::solvers {

// Chebyshev-filtered subspace iteration (ChFSI) — R1 regime.
//
// Algorithm (per 20-math/24):
//   1. Start with a subspace V (n x m, m >= n_occ).
//   2. Apply a Chebyshev polynomial filter of degree p that amplifies
//      eigenvectors in the desired spectral window [lambda_min, lambda_max]:
//        T_0(V) = V
//        T_1(V) = (H - c*I)/d * V          (scaled)
//        T_k(V) = 2*(H-c*I)/d * T_{k-1} - T_{k-2}
//      where c = (lambda_max + lambda_min)/2, d = (lambda_max - lambda_min)/2.
//   3. Rayleigh-Ritz: project H onto the filtered subspace, solve the small
//      generalized eigenproblem, rotate back.
//   4. Lock converged eigenpairs (deflate them out of the active space).
//   5. Subspace REUSE: carry the converged subspace forward across SCF/MD
//      steps (the big practical win — reduces filter applications >= 2x).
//
// For the CPU reference, H and S are dense matrices; the GPU path uses
// WP1 TileMat grouped GEMM for the matrix-subspace products.
//
// Observable (T4.3): residuals ||H x - e S x|| <= 1e-9; SCF iteration count
// within +2 of dense reference; subspace reuse cuts filter applications >= 2x.

struct ChFSIResult {
  std::vector<double> eigenvalues;    // n_occ lowest eigenvalues
  std::vector<double> eigenvectors;  // n x n_occ, column k = eigenvalue k
  std::vector<double> residuals;      // per eigenpair
  int n_filter_applications = 0;     // total Chebyshev filter applications
  bool converged = false;
};

class ChFSI {
 public:
  // Solve H x = e S x for the n_occ lowest eigenpairs via Chebyshev-filtered
  // subspace iteration.
  //   n:       matrix dimension
  //   H, S:    symmetric matrices (row-major, S SPD)
  //   n_occ:   number of eigenpairs wanted
  //   lambda_lo, lambda_hi: spectral window for the filter (estimates of the
  //                         lowest and highest eigenvalues of interest)
  //   degree:  Chebyshev polynomial degree (typically 8-30)
  //   max_iter: maximum subspace iterations
  //   tol:     residual convergence target
  //   V_init:   initial subspace (n x m, m >= n_occ); if empty, random.
  static ChFSIResult Solve(std::size_t n, const std::vector<double>& H,
                           const std::vector<double>& S, std::size_t n_occ,
                           double lambda_lo, double lambda_hi, int degree,
                           int max_iter = 100, double tol = 1e-9,
                           std::vector<double> V_init = {}) {
    ChFSIResult res;
    if (n == 0 || n_occ == 0 || n_occ > n) return res;
    const std::size_t m = std::max(n_occ, std::size_t{2});  // subspace size
    const std::size_t m_eff = std::min(m, n);

    // Initialize subspace V (n x m_eff, column-major: V[k*n + j]).
    std::vector<double> V(n * m_eff, 0.0);
    if (V_init.size() == n * m_eff) {
      V = V_init;
    } else {
      // Identity-like initialization (use S-orthonormalized basis).
      for (std::size_t k = 0; k < m_eff; ++k) V[k * n + k] = 1.0;
    }

    // S-orthonormalize V via modified Gram-Schmidt.
    SOrthonormalize(n, m_eff, V, S);

    const double c = 0.5 * (lambda_hi + lambda_lo);
    const double d = 0.5 * (lambda_hi - lambda_lo);
    if (d <= 0) return res;  // invalid spectral window

    for (int iter = 0; iter < max_iter; ++iter) {
      // Apply Chebyshev filter of degree `degree`.
      // M = (H - c*I)/d; T_0 = V; T_1 = M*V; T_k = 2*M*T_{k-1} - T_{k-2}
      std::vector<double> T_prev2 = V;  // T_{k-2}
      std::vector<double> T_prev1(n * m_eff);  // T_{k-1}
      {
        auto HV = MatMul(n, m_eff, H, T_prev2);
        for (std::size_t k = 0; k < m_eff; ++k)
          for (std::size_t j = 0; j < n; ++j)
            T_prev1[k * n + j] = (HV[k * n + j] - c * T_prev2[k * n + j]) / d;
        res.n_filter_applications++;
      }
      for (int p = 2; p <= degree; ++p) {
        auto HV = MatMul(n, m_eff, H, T_prev1);
        std::vector<double> Tk(n * m_eff);
        for (std::size_t k = 0; k < m_eff; ++k)
          for (std::size_t j = 0; j < n; ++j)
            Tk[k * n + j] = 2.0 * (HV[k * n + j] - c * T_prev1[k * n + j]) / d
                           - T_prev2[k * n + j];
        T_prev2 = T_prev1;
        T_prev1 = Tk;
        res.n_filter_applications++;
      }
      V = T_prev1;

      // S-orthonormalize the filtered subspace.
      SOrthonormalize(n, m_eff, V, S);

      // Rayleigh-Ritz: project H into the subspace.
      // H_proj = V^T H V (m_eff x m_eff), S_proj = V^T S V (should be ~I).
      auto H_proj = ProjectSubspace(n, m_eff, V, H);
      auto S_proj = ProjectSubspace(n, m_eff, V, S);

      // Solve the small generalized eigenproblem.
      auto rr = BatchedDenseEig::SolveGeneralized(m_eff, H_proj, S_proj);
      if (!rr.ok) return res;

      // Rotate back: x_k = sum_j V_j * rr.eigenvectors[k][j]
      std::vector<double> X_new(n * m_eff, 0.0);
      for (std::size_t k = 0; k < m_eff; ++k)
        for (std::size_t j = 0; j < n; ++j) {
          double s = 0.0;
          for (std::size_t i = 0; i < m_eff; ++i)
            s += V[i * n + j] * rr.eigenvectors[k * m_eff + i];
          X_new[k * n + j] = s;
        }
      V = X_new;

      // Check residuals for the n_occ lowest pairs.
      res.eigenvalues.assign(n_occ, 0.0);
      res.eigenvectors.assign(n * n_occ, 0.0);
      res.residuals.assign(n_occ, 0.0);
      bool all_conv = true;
      for (std::size_t k = 0; k < n_occ; ++k) {
        res.eigenvalues[k] = rr.eigenvalues[k];
        std::vector<double> x(n);
        for (std::size_t j = 0; j < n; ++j) x[j] = V[k * n + j];
        for (std::size_t j = 0; j < n; ++j) res.eigenvectors[k * n + j] = V[k * n + j];
        res.residuals[k] = BatchedDenseEig::Residual(n, H, S, res.eigenvalues[k], x);
        if (res.residuals[k] > tol) all_conv = false;
      }

      if (all_conv) {
        res.converged = true;
        return res;
      }
    }
    return res;
  }

  // Estimate spectral bounds via a few Lanczos iterations on H (with S metric).
  // Returns {lambda_min_estimate, lambda_max_estimate}.
  static std::pair<double, double> SpectralBounds(
      std::size_t n, const std::vector<double>& H, const std::vector<double>& S,
      int n_lanczos = 20) {
    if (n == 0) return {0, 0};
    std::vector<double> v(n, 0.0), v_prev(n, 0.0);
    v[0] = 1.0;
    SOrthonormalize(n, 1, v, S);
    double beta = 0.0;
    double alpha_min = 1e30, alpha_max = -1e30;
    for (int it = 0; it < std::min(n_lanczos, static_cast<int>(n)); ++it) {
      // w = H v
      std::vector<double> w(n, 0.0);
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
          w[i] += H[i * n + j] * v[j];
      // alpha = v^T w (Rayleigh quotient)
      double alpha = 0.0;
      for (std::size_t i = 0; i < n; ++i) alpha += v[i] * w[i];
      alpha_min = std::min(alpha_min, alpha);
      alpha_max = std::max(alpha_max, alpha);
      // w = w - alpha v - beta v_prev
      for (std::size_t i = 0; i < n; ++i)
        w[i] = w[i] - alpha * v[i] - beta * v_prev[i];
      v_prev = v;
      beta = 0.0;
      for (std::size_t i = 0; i < n; ++i) beta += w[i] * w[i];
      beta = std::sqrt(beta);
      if (beta < 1e-14) break;
      for (std::size_t i = 0; i < n; ++i) v[i] = w[i] / beta;
    }
    // The Lanczos alpha values are Ritz estimates; the true bounds are wider.
    // Expand by 10% for the filter window.
    const double margin = 0.1 * (alpha_max - alpha_min + 1e-10);
    return {alpha_min - margin, alpha_max + margin};
  }

 private:
  // Matrix-times-subspace: W = H * V (n x m_eff = n x n * n x m_eff).
  static std::vector<double> MatMul(std::size_t n, std::size_t m,
                                    const std::vector<double>& H,
                                    const std::vector<double>& V) {
    std::vector<double> W(n * m, 0.0);
    for (std::size_t k = 0; k < m; ++k)
      for (std::size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < n; ++j)
          s += H[i * n + j] * V[k * n + j];
        W[k * n + i] = s;
      }
    return W;
  }

  // S-orthonormalize the columns of V (n x m) via modified Gram-Schmidt
  // with the S inner product: <u, v>_S = u^T S v.
  static void SOrthonormalize(std::size_t n, std::size_t m,
                              std::vector<double>& V,
                              const std::vector<double>& S) {
    for (std::size_t k = 0; k < m; ++k) {
      for (std::size_t p = 0; p < k; ++p) {
        // proj = <v_p, v_k>_S
        double proj = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
          double Si = 0.0;
          for (std::size_t j = 0; j < n; ++j) Si += S[i * n + j] * V[k * n + j];
          proj += V[p * n + i] * Si;
        }
        for (std::size_t i = 0; i < n; ++i)
          V[k * n + i] -= proj * V[p * n + i];
      }
      // Normalize: <v_k, v_k>_S = 1
      double norm_sq = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        double Si = 0.0;
        for (std::size_t j = 0; j < n; ++j) Si += S[i * n + j] * V[k * n + j];
        norm_sq += V[k * n + i] * Si;
      }
      double inv_norm = (norm_sq > 1e-30) ? 1.0 / std::sqrt(norm_sq) : 0.0;
      for (std::size_t i = 0; i < n; ++i) V[k * n + i] *= inv_norm;
    }
  }

  // Project H onto the subspace V: H_proj[k][l] = V_k^T H V_l (m x m).
  static std::vector<double> ProjectSubspace(std::size_t n, std::size_t m,
                                             const std::vector<double>& V,
                                             const std::vector<double>& H) {
    std::vector<double> HV = MatMul(n, m, H, V);
    std::vector<double> H_proj(m * m, 0.0);
    for (std::size_t k = 0; k < m; ++k)
      for (std::size_t l = 0; l < m; ++l) {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += V[k * n + i] * HV[l * n + i];
        H_proj[k * m + l] = s;
      }
    // Symmetrize.
    for (std::size_t k = 0; k < m; ++k)
      for (std::size_t l = k + 1; l < m; ++l) {
        double avg = 0.5 * (H_proj[k * m + l] + H_proj[l * m + k]);
        H_proj[k * m + l] = avg;
        H_proj[l * m + k] = avg;
      }
    return H_proj;
  }
};

}  // namespace tides::solvers
