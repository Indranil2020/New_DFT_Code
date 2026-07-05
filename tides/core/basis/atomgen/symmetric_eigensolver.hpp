#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace tides::atomgen {

// Self-contained symmetric eigensolver using the classical cyclic Jacobi
// method (Golub & Van Loan, Algorithm 8.5.1) with the numerically stable
// tangent rotation formula. No external dependencies (matches the WP1
// zero-dependency C++20 ethos).
//
// Jacobi is backward stable and produces eigenpairs accurate to ~eps_machine
// (eigenvectors to ~eps/gap). Cost is O(n^3) per sweep with ~6-10 sweeps for
// full convergence; adequate for the modest grid sizes (n <= ~1000) used by
// the radial atomic solver.
//
// All matrices are row-major. Eigenvector k is eigenvectors[k*n + j]
// (component j).
class SymmetricEigensolver {
 public:
  // Compute eigenvalues AND eigenvectors. A is destroyed.
  static void Solve(std::vector<double>& a, std::size_t n,
                    std::vector<double>& eigenvalues,
                    std::vector<double>& eigenvectors) {
    eigenvalues.assign(n, 0.0);
    eigenvectors.assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) eigenvectors[i * n + i] = 1.0;
    if (n <= 1) {
      if (n == 1) eigenvalues[0] = a[0];
      return;
    }

    const std::size_t max_sweeps = 100;
    for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
      double off = 0.0;
      for (std::size_t p = 0; p < n; ++p)
        for (std::size_t q = p + 1; q < n; ++q) off += a[p * n + q] * a[p * n + q];
      if (off <= 1e-300) break;
      for (std::size_t p = 0; p < n - 1; ++p) {
        for (std::size_t q = p + 1; q < n; ++q) {
          Rotate(a, eigenvectors, n, p, q);
        }
      }
    }

    for (std::size_t i = 0; i < n; ++i) eigenvalues[i] = a[i * n + i];
    SortAscending(eigenvalues, eigenvectors, n);
  }

  // Eigenvalues only (no vectors). A is destroyed.
  static void ValuesOnly(std::vector<double>& a, std::size_t n,
                         std::vector<double>& eigenvalues) {
    eigenvalues.assign(n, 0.0);
    if (n == 0) return;
    if (n == 1) { eigenvalues[0] = a[0]; return; }
    std::vector<double> dummy;
    const std::size_t max_sweeps = 100;
    for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
      double off = 0.0;
      for (std::size_t p = 0; p < n; ++p)
        for (std::size_t q = p + 1; q < n; ++q) off += a[p * n + q] * a[p * n + q];
      if (off <= 1e-300) break;
      for (std::size_t p = 0; p < n - 1; ++p)
        for (std::size_t q = p + 1; q < n; ++q) Rotate(a, dummy, n, p, q);
    }
    for (std::size_t i = 0; i < n; ++i) eigenvalues[i] = a[i * n + i];
    std::sort(eigenvalues.begin(), eigenvalues.end());
  }

 private:
  // Apply one Jacobi rotation that annihilates a[p*n+q] (p<q). If v is
  // non-empty, accumulate the rotation into the eigenvector matrix.
  static void Rotate(std::vector<double>& a, std::vector<double>& v,
                     std::size_t n, std::size_t p, std::size_t q) {
    const double apq = a[p * n + q];
    if (apq == 0.0) return;
    const double app = a[p * n + p];
    const double aqq = a[q * n + q];
    const double tau = (aqq - app) / (2.0 * apq);
    double t;
    if (tau >= 0.0) {
      t = 1.0 / (tau + std::sqrt(1.0 + tau * tau));
    } else {
      t = -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
    }
    const double c = 1.0 / std::sqrt(1.0 + t * t);
    const double s = t * c;

    // Update columns p and q of A.
    for (std::size_t i = 0; i < n; ++i) {
      const double aip = a[i * n + p];
      const double aiq = a[i * n + q];
      a[i * n + p] = c * aip - s * aiq;
      a[i * n + q] = s * aip + c * aiq;
    }
    // Update rows p and q of A.
    for (std::size_t j = 0; j < n; ++j) {
      const double apj = a[p * n + j];
      const double aqj = a[q * n + j];
      a[p * n + j] = c * apj - s * aqj;
      a[q * n + j] = s * apj + c * aqj;
    }
    // Force exact zero of the annihilated element (roundoff hygiene).
    a[p * n + q] = 0.0;
    a[q * n + p] = 0.0;

    // Accumulate rotation: V <- V J (only if vectors are requested).
    if (!v.empty()) {
      for (std::size_t i = 0; i < n; ++i) {
        const double vip = v[i * n + p];
        const double viq = v[i * n + q];
        v[i * n + p] = c * vip - s * viq;
        v[i * n + q] = s * vip + c * viq;
      }
    }
  }

  static void SortAscending(std::vector<double>& d, std::vector<double>& v,
                            std::size_t n) {
    std::vector<std::size_t> idx(n);
    for (std::size_t i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
              [&](std::size_t a, std::size_t b) { return d[a] < d[b]; });
    std::vector<double> d2(n), v2(n * n);
    for (std::size_t i = 0; i < n; ++i) {
      d2[i] = d[idx[i]];
      for (std::size_t k = 0; k < n; ++k) v2[k * n + i] = v[k * n + idx[i]];
    }
    d = d2;
    v = v2;
  }
};

}  // namespace tides::atomgen
