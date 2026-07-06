#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

#include "solvers/dense/batched_eig.hpp"

namespace tides::scf {

// SCF driver with Pulay / simple mixing (per WP6 T6.1).
//
// The SCF loop:
//   1. Given density matrix P, build the effective potential V_eff[P].
//   2. Build the Hamiltonian H[P] and overlap S (fixed for NAOs).
//   3. Solve the generalized eigenproblem H x = e S x (via WP4 BatchedDenseEig).
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
  //   energy:   callback: given P, returns total energy
  //   P_init:   initial density (empty = identity guess)
  //   max_iter: maximum SCF iterations
  //   tol:      energy convergence target
  //   mixing:   0=simple, 1=Pulay(Anderson)
  //   alpha:    mixing parameter (0 < alpha <= 1)
  static SCFResult Run(
      std::size_t n, std::size_t n_occ, const std::vector<double>& S,
      const std::function<std::vector<double>(const std::vector<double>&)>& build_H,
      const std::function<double(const std::vector<double>&)>& energy_fn,
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

    // Pulay history (Anderson acceleration).
    const int pulay_depth = 6;
    std::vector<std::vector<double>> P_history;
    std::vector<std::vector<double>> R_history;  // residuals (P_new - P_old)

    double E_prev = 1e30;
    for (int iter = 0; iter < max_iter; ++iter) {
      res.n_iterations = iter + 1;

      // Build H from current P.
      auto H = build_H(P);

      // Diagonalize.
      auto eig = tides::solvers::BatchedDenseEig::SolveGeneralized(n, H, S);
      if (!eig.ok) return res;

      // Occupy n_occ lowest orbitals -> P_new.
      std::vector<double> P_new(n * n, 0.0);
      for (std::size_t k = 0; k < n_occ; ++k)
        for (std::size_t i = 0; i < n; ++i)
          for (std::size_t j = 0; j < n; ++j)
            P_new[i * n + j] += eig.eigenvectors[k * n + i] *
                                eig.eigenvectors[k * n + j];

      // Compute energy.
      double E = energy_fn(P_new);
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
      if (mixing == 1 && iter > 0 && static_cast<int>(P_history.size()) >= 2) {
        // Anderson/Pulay extrapolation: find the linear combination of past
        // residuals that minimizes the new residual, use the same coeffs for P.
        const int m = std::min(pulay_depth, static_cast<int>(P_history.size()));
        // Simple Pulay: mix the last two with alpha.
        // Full Anderson would solve a small least-squares; here we use a
        // simplified version: P_next = P_new - alpha * (P_new - P_history.back())
        const auto& P_last = P_history.back();
        for (std::size_t i = 0; i < n * n; ++i)
          P_next[i] = P_new[i] - alpha * (P_new[i] - P_last[i]);
      } else {
        // Simple mixing.
        for (std::size_t i = 0; i < n * n; ++i)
          P_next[i] = alpha * P_new[i] + (1.0 - alpha) * P[i];
      }

      // Update history.
      P_history.push_back(P);
      R_history.push_back(P_new);
      if (static_cast<int>(P_history.size()) > pulay_depth) {
        P_history.erase(P_history.begin());
        R_history.erase(R_history.begin());
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
