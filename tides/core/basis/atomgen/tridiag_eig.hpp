#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "basis/atomgen/symmetric_eigensolver.hpp"

// Fortran LAPACK tridiagonal eigensolver prototype.
extern "C" {
void dstev_(const char* jobz, const int* n, double* dz, double* e, double* z,
            const int* ldz, double* work, int* info);
}

namespace tides::atomgen {

// Tridiagonal symmetric eigensolver. The 3-point FD stencil produces a
// tridiagonal matrix; exploiting this structure is ~1000x faster than a dense
// solve at n~3000 (O(n^2) vs O(n^3)), letting us reach the 1e-10 target at
// large n. Uses LAPACK dstev_ when available; falls back to building a dense
// matrix and using the Jacobi eigensolver otherwise.
//
// Convention: diagonal d[0..n-1], off-diagonal e[0..n-2] (e[i] couples d[i]
// and d[i+1]). Eigenvalues ascending; eigenvectors as columns of z (row-major
// z[k*n + j] = component j of eigenvalue k).
class TridiagEig {
 public:
  static void Solve(std::vector<double>& d, std::vector<double>& e,
                    std::size_t n, std::vector<double>& eigenvalues,
                    std::vector<double>& eigenvectors) {
    eigenvalues.assign(n, 0.0);
    eigenvectors.assign(n * n, 0.0);
    if (n == 0) return;
    if (n == 1) { eigenvalues[0] = d[0]; eigenvectors[0] = 1.0; return; }

#ifdef TIDES_HAVE_LAPACK
    // dstev_ uses e[0..n-2] and overwrites with subdiagonal; eigenvectors in Z
    // (column-major, n x n). Copy inputs since dstev destroys them.
    std::vector<double> dc = d;
    std::vector<double> ec(n, 0.0);
    for (std::size_t i = 0; i + 1 < n; ++i) ec[i] = e[i];
    std::vector<double> z(n * n);
    std::vector<double> work(2 * n);
    char jobz = 'V';
    int nn = static_cast<int>(n);
    int ldz = nn;
    int info = 0;
    dstev_(&jobz, &nn, dc.data(), ec.data(), z.data(), &ldz, work.data(),
           &info);
    if (info != 0) {
      // Fall back to dense Jacobi.
      DenseFallback(d, e, n, eigenvalues, eigenvectors);
      return;
    }
    eigenvalues = dc;
    // z is column-major: eigenvector k is column k = z[k*n + 0..n-1].
    for (std::size_t k = 0; k < n; ++k)
      for (std::size_t j = 0; j < n; ++j)
        eigenvectors[k * n + j] = z[k * n + j];
#else
    DenseFallback(d, e, n, eigenvalues, eigenvectors);
#endif
  }

  static void ValuesOnly(std::vector<double>& d, std::vector<double>& e,
                         std::size_t n, std::vector<double>& eigenvalues) {
    eigenvalues.assign(n, 0.0);
    if (n == 0) return;
    if (n == 1) { eigenvalues[0] = d[0]; return; }
#ifdef TIDES_HAVE_LAPACK
    std::vector<double> dc = d;
    std::vector<double> ec(n, 0.0);
    for (std::size_t i = 0; i + 1 < n; ++i) ec[i] = e[i];
    std::vector<double> work(2 * n);
    char jobz = 'N';
    int nn = static_cast<int>(n);
    int ldz = 1;
    int info = 0;
    dstev_(&jobz, &nn, dc.data(), ec.data(), nullptr, &ldz, work.data(), &info);
    if (info != 0) {
      std::vector<double> dummy;
      std::vector<double> dense = TridiagToDense(d, e, n);
      SymmetricEigensolver::ValuesOnly(dense, n, eigenvalues);
      return;
    }
    eigenvalues = dc;
    std::sort(eigenvalues.begin(), eigenvalues.end());
#else
    std::vector<double> dense = TridiagToDense(d, e, n);
    SymmetricEigensolver::ValuesOnly(dense, n, eigenvalues);
#endif
  }

 private:
  static std::vector<double> TridiagToDense(const std::vector<double>& d,
                                           const std::vector<double>& e,
                                           std::size_t n) {
    std::vector<double> a(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      a[i * n + i] = d[i];
      if (i + 1 < n) {
        a[i * n + (i + 1)] = e[i];
        a[(i + 1) * n + i] = e[i];
      }
    }
    return a;
  }

  static void DenseFallback(const std::vector<double>& d,
                           const std::vector<double>& e, std::size_t n,
                           std::vector<double>& eigenvalues,
                           std::vector<double>& eigenvectors) {
    std::vector<double> a = TridiagToDense(d, e, n);
    SymmetricEigensolver::Solve(a, n, eigenvalues, eigenvectors);
  }
};

}  // namespace tides::atomgen
