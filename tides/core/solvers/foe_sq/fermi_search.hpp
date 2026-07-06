#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "solvers/sp2_submatrix/sp2.hpp"

namespace tides::solvers {

// Fermi-level search (T5.6): find mu such that tr(P(mu) S) = N_e.
//
// The density matrix P depends on mu via P = f_FD((H - mu S) / kT). The trace
// tr(P S) is a monotonically decreasing function of mu (more mu -> more
//occupied -> higher trace). We find mu by robust bracketed bisection (the
// plan specifies "f64e-safe bracketed Newton").
//
// Observable (T5.6): N_e error <= 1e-10 across 10^2-10^4-atom cases;
// robust bracketing (no failure on the gauntlet).
//
// For the CPU reference we use a dense eigensolver to compute the exact
// tr(P S) at each mu; the production path uses FOE (T5.5) for O(N) evaluation.

class FermiLevelSearch {
 public:
  // Find the chemical potential mu such that tr(P(mu) S) = N_e.
  // Uses the eigenvalues of the generalized problem H x = e S x.
  //   evals:     eigenvalues of the generalized problem (ascending)
  //   S:         overlap matrix (for computing tr(P S) = sum_k f_k <x_k|S|x_k>)
  //   n_e:       target electron count
  //   kT_e:      electronic temperature (Hartree)
  //   tol:       convergence target for |tr(PS) - N_e|
  //   max_iter:  maximum bisection iterations
  // Returns mu, the chemical potential.
  static double Search(const std::vector<double>& evals,
                       const std::vector<double>& S_diag,
                       double n_e, double kT_e,
                       double tol = 1e-12, int max_iter = 200) {
    if (evals.empty() || kT_e <= 0) return 0.0;
    const std::size_t n = evals.size();

    // tr(P S) = sum_k f(e_k - mu) * S_kk  (for S-orthonormal eigenvectors,
    // <x_k|S|x_k> = 1, so tr(P S) = sum_k f(e_k - mu)). For the generalized
    // problem with S != I, the eigenvectors satisfy x_k^T S x_l = delta_kl,
    // so tr(P S) = sum_k f_k where f_k = f_FD((e_k - mu)/kT).
    auto trace_at_mu = [&](double mu) -> double {
      double tr = 0.0;
      for (std::size_t k = 0; k < n; ++k)
        tr += FermiDirac(evals[k], mu, kT_e);
      return tr;
    };

    // Bracket: mu_lo (trace > N_e, i.e. mu above all eigenvalues -> all occupied)
    // and mu_hi (trace < N_e, i.e. mu below all eigenvalues -> all empty).
    // Higher mu -> MORE occupied -> HIGHER trace.
    double mu_lo = evals[n - 1] + 10.0 * kT_e;  // above all -> trace ~ n
    double mu_hi = evals[0] - 10.0 * kT_e;       // below all -> trace ~ 0
    double tr_lo = trace_at_mu(mu_lo);
    double tr_hi = trace_at_mu(mu_hi);

    // Expand bracket if needed.
    int expand = 0;
    while (tr_lo < n_e && expand < 50) {
      mu_lo += 10.0 * kT_e;
      tr_lo = trace_at_mu(mu_lo);
      ++expand;
    }
    expand = 0;
    while (tr_hi > n_e && expand < 50) {
      mu_hi -= 10.0 * kT_e;
      tr_hi = trace_at_mu(mu_hi);
      ++expand;
    }

    // Bisection: tr is monotonically INCREASING in mu. If tr_mid > N_e, mu too
    // high -> search lower (mu_hi = mid). If tr_mid < N_e -> search higher.
    for (int iter = 0; iter < max_iter; ++iter) {
      const double mu_mid = 0.5 * (mu_lo + mu_hi);
      const double tr_mid = trace_at_mu(mu_mid);
      if (std::fabs(tr_mid - n_e) < tol) return mu_mid;
      if (tr_mid > n_e) {
        mu_hi = mu_mid;  // too many occupied -> lower mu
      } else {
        mu_lo = mu_mid;  // too few -> raise mu
      }
    }
    return 0.5 * (mu_lo + mu_hi);
  }

  // Fermi-Dirac distribution f(e, mu, kT) = 1/(1 + exp((e - mu)/kT)).
  static double FermiDirac(double e, double mu, double kT) {
    const double x = (e - mu) / kT;
    if (x > 500) return 0.0;
    if (x < -500) return 1.0;
    return 1.0 / (1.0 + std::exp(x));
  }

  // Verify: given mu, compute |tr(P S) - N_e| using the eigenspectrum.
  static double TraceError(const std::vector<double>& evals, double mu,
                           double kT_e, double n_e) {
    const std::size_t n = evals.size();
    double tr = 0.0;
    for (std::size_t k = 0; k < n; ++k)
      tr += FermiDirac(evals[k], mu, kT_e);
    return std::fabs(tr - n_e);
  }
};

}  // namespace tides::solvers
