#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "basis/atomgen/symmetric_eigensolver.hpp"

// Fortran LAPACK prototype. Symbol convention: trailing underscore (gfortran/
// Intel Fortran). The prototype uses C linkage; parameters passed by address
// as Fortran expects.
extern "C" {
void dsyev_(const char* jobz, const char* uplo, const int* n, double* a,
            const int* lda, double* w, double* work, const int* lwork,
            int* info);
}

namespace tides::atomgen {

// Dense symmetric eigensolver dispatcher. Uses LAPACK dsyev_ when available
// (fast, ~100x the Jacobi at these sizes); falls back to the self-contained
// Jacobi eigensolver otherwise. Same contract as SymmetricEigensolver:
// row-major symmetric A (destroyed), ascending eigenvalues, eigenvectors as
// columns (eigenvectors[k*n + j] = component j of eigenvalue k).
class DenseSymEig {
 public:
  static void Solve(std::vector<double>& a, std::size_t n,
                    std::vector<double>& eigenvalues,
                    std::vector<double>& eigenvectors) {
#ifdef TIDES_HAVE_LAPACK
    SolveLapack(a, n, eigenvalues, eigenvectors);
#else
    SymmetricEigensolver::Solve(a, n, eigenvalues, eigenvectors);
#endif
  }

  static void ValuesOnly(std::vector<double>& a, std::size_t n,
                         std::vector<double>& eigenvalues) {
#ifdef TIDES_HAVE_LAPACK
    ValuesOnlyLapack(a, n, eigenvalues);
#else
    SymmetricEigensolver::ValuesOnly(a, n, eigenvalues);
#endif
  }

 private:
#ifdef TIDES_HAVE_LAPACK
  // LAPACK dsyev_ Fortran interface. JOBZ='V' for vectors, 'N' for values only.
  // UPLO='L' (lower triangle stored, but we keep full symmetric storage so it's
  // fine). Column-major layout: A[i][j] -> a_cm[j*n + i].
  static void SolveLapack(std::vector<double>& a, std::size_t n,
                          std::vector<double>& eigenvalues,
                          std::vector<double>& eigenvectors) {
    eigenvalues.assign(n, 0.0);
    eigenvectors.assign(n * n, 0.0);
    if (n == 0) return;
    if (n == 1) { eigenvalues[0] = a[0]; eigenvectors[0] = 1.0; return; }

    // Transpose row-major -> column-major in place (matrix is symmetric, so we
    // can just copy; but to be safe against any asymmetry, copy the full array).
    std::vector<double> acm(n * n);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        acm[j * n + i] = a[i * n + j];

    char jobz = 'V';
    char uplo = 'L';
    int nn = static_cast<int>(n);
    int lda = nn;
    int info = 0;
    std::vector<double> w(n);
    int lwork = -1;
    double wkopt = 0.0;
    dsyev_(&jobz, &uplo, &nn, acm.data(), &lda, w.data(), &wkopt, &lwork, &info);
    lwork = static_cast<int>(wkopt);
    std::vector<double> work(static_cast<std::size_t>(lwork));
    dsyev_(&jobz, &uplo, &nn, acm.data(), &lda, w.data(), work.data(), &lwork, &info);

    if (info != 0) {
      // LAPACK failure -> fall back to Jacobi (fail loud in the fallback, but
      // never silently produce garbage).
      std::vector<double> a2 = a;
      SymmetricEigensolver::Solve(a2, n, eigenvalues, eigenvectors);
      return;
    }
    eigenvalues = w;
    // acm now holds eigenvectors as COLUMNS (column-major). Eigenvector k is
    // acm[k*n + 0..n-1]. Convert to our row-major convention
    // (eigenvectors[k*n + j] = component j of eigenvalue k).
    for (std::size_t k = 0; k < n; ++k)
      for (std::size_t j = 0; j < n; ++j)
        eigenvectors[k * n + j] = acm[k * n + j];
  }

  static void ValuesOnlyLapack(std::vector<double>& a, std::size_t n,
                               std::vector<double>& eigenvalues) {
    eigenvalues.assign(n, 0.0);
    if (n == 0) return;
    if (n == 1) { eigenvalues[0] = a[0]; return; }
    char jobz = 'N';
    char uplo = 'L';
    int nn = static_cast<int>(n);
    int lda = nn;
    int info = 0;
    std::vector<double> w(n);
    int lwork = -1;
    double wkopt = 0.0;
    dsyev_(&jobz, &uplo, &nn, a.data(), &lda, w.data(), &wkopt, &lwork, &info);
    lwork = static_cast<int>(wkopt);
    std::vector<double> work(static_cast<std::size_t>(lwork));
    dsyev_(&jobz, &uplo, &nn, a.data(), &lda, w.data(), work.data(), &lwork, &info);
    if (info != 0) {
      std::vector<double> a2 = a;
      SymmetricEigensolver::ValuesOnly(a2, n, eigenvalues);
      return;
    }
    eigenvalues = w;
    std::sort(eigenvalues.begin(), eigenvalues.end());
  }
#endif
};

}  // namespace tides::atomgen
