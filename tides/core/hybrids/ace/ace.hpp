#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

#include "hybrids/isdf/isdf.hpp"

namespace tides::hybrids {

// ACE (Adaptively Compressed Exchange) + hybrid SCF — T7.3.
//
// ACE compresses the exact exchange matrix V_x into a low-rank form:
//   V_x ≈ V_tilde = U * Lambda * U^T
// where U is a set of "compressed" orbitals (projections of the occupied
// orbitals onto the exchange operator) and Lambda is a small matrix.
// After construction (once per SCF iteration), applying V_tilde costs
// like a semilocal term (O(N^2) instead of O(N^4)).
//
// The hybrid SCF adds the exact exchange to the semilocal Hamiltonian:
//   H_hybrid = H_PBE + alpha * (V_x - V_x^PBE)
// where alpha is the fraction of exact exchange (0.25 for PBE0, screened for HSE).
//
// Observable (T7.3): PBE0 H2O and benzene vs PySCF <=0.1 mHa.

struct ACEResult {
  std::vector<double> V_x;       // compressed exchange matrix (n x n)
  double exchange_energy = 0.0;  // E_x = -0.5 * Tr(P V_x)
  std::size_t rank = 0;
  bool ok = false;
};

class ACE {
 public:
  // Build the ACE-compressed exchange operator.
  //   P:       density matrix (n x n, row-major)
  //   K_exact: exact exchange matrix K_{ij} = (ij|kl) P_{kl} (n x n, row-major)
  //   n:       matrix dimension
  //   n_occ:   number of occupied orbitals
  //   alpha:   fraction of exact exchange (0.25 for PBE0)
  // Returns V_x = alpha * K_exact (the ACE compression for the CPU reference
  // is identity — we use the full K; the rank is n_occ).
  static ACEResult Build(const std::vector<double>& P,
                         const std::vector<double>& K_exact,
                         std::size_t n, std::size_t n_occ, double alpha) {
    ACEResult res;
    res.V_x.assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        res.V_x[i * n + j] = alpha * K_exact[i * n + j];

    // Exchange energy: E_x = -0.5 * Tr(P K) * alpha.
    double Ex = 0.0;
    for (std::size_t i = 0; i < n * n; ++i) Ex += P[i] * K_exact[i];
    res.exchange_energy = -0.5 * alpha * Ex;
    res.rank = n_occ;
    res.ok = true;
    return res;
  }

  // Compute the exact exchange matrix K_{ij} = sum_{kl} P_{kl} (ij|kl).
  // For the CPU reference, the 2-electron integrals (ij|kl) are provided
  // as a callback (the caller computes them via the grid or a fitting basis).
  // K_{ij} = sum_{kl} P_{kl} * eri(i,j,k,l)
  static std::vector<double> ComputeK(
      const std::vector<double>& P, std::size_t n,
      const std::function<double(std::size_t, std::size_t, std::size_t, std::size_t)>& eri) {
    std::vector<double> K(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        double k = 0.0;
        for (std::size_t kl = 0; kl < n; ++kl)
          for (std::size_t l = 0; l < n; ++l)
            k += P[kl * n + l] * eri(i, j, kl, l);
        K[i * n + j] = k;
      }
    return K;
  }

  // PBE0 hybrid energy: E_PBE0 = E_PBE + alpha * (E_x_exact - E_x_PBE).
  // For the CPU reference, E_x_exact and E_x_PBE are provided.
  static double PBE0Energy(double E_PBE, double Ex_exact, double Ex_PBE,
                           double alpha = 0.25) {
    return E_PBE + alpha * (Ex_exact - Ex_PBE);
  }
};

}  // namespace tides::hybrids
