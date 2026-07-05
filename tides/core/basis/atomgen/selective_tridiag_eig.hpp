#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "basis/atomgen/symmetric_eigensolver.hpp"
#include "basis/atomgen/tridiag_eig.hpp"

// Fortran LAPACK prototypes.
extern "C" {
void dstev_(const char* jobz, const int* n, double* dz, double* e, double* z,
            const int* ldz, double* work, int* info);
// Selective tridiagonal eigensolver: computes il..iu-th eigenpairs.
void dstevx_(const char* jobz, const char* range, const int* n, double* d,
             double* e, double* vl, double* vu, const int* il, const int* iu,
             double* abstol, int* m, double* w, double* z, const int* ldz,
             double* work, int* iwork, int* ifail, int* info);
}

namespace tides::atomgen {

// Selective tridiagonal eigensolver: computes only the `num_states` lowest
// eigenpairs (il=1..num_states) via LAPACK dstevx_ (bisection + inverse
// iteration). O(n * num_states) — fast even at n~50000 for a handful of states.
// Falls back to the full TridiagEig when LAPACK is unavailable.
class SelectiveTridiagEig {
 public:
  // Returns the `num_states` lowest eigenpairs. Eigenvalues ascending in w;
  // eigenvectors[k*n + j] = component j of eigenvalue k.
  static void Solve(std::vector<double>& d, std::vector<double>& e,
                    std::size_t n, std::size_t num_states,
                    std::vector<double>& eigenvalues,
                    std::vector<double>& eigenvectors) {
    num_states = std::min(num_states, n);
    eigenvalues.assign(num_states, 0.0);
    eigenvectors.assign(num_states * n, 0.0);
    if (n == 0 || num_states == 0) return;
    if (n == 1) {
      eigenvalues[0] = d[0];
      eigenvectors[0] = 1.0;
      return;
    }

#ifdef TIDES_HAVE_LAPACK
    std::vector<double> dc = d;
    std::vector<double> ec(n, 0.0);
    for (std::size_t i = 0; i + 1 < n; ++i) ec[i] = e[i];
    char jobz = 'V';
    char range = 'I';  // index range
    int nn = static_cast<int>(n);
    double vl = 0.0, vu = 0.0;
    int il = 1;
    int iu = static_cast<int>(num_states);
    double abstol = 0.0;  // use default
    int m = 0;
    std::vector<double> w(num_states);
    std::vector<double> z(static_cast<std::size_t>(num_states) * n);
    int ldz = nn;
    std::vector<double> work(5 * n);
    std::vector<int> iwork(5 * n);
    std::vector<int> ifail(n);
    int info = 0;
    dstevx_(&jobz, &range, &nn, dc.data(), ec.data(), &vl, &vu, &il, &iu,
            &abstol, &m, w.data(), z.data(), &ldz, work.data(), iwork.data(),
            ifail.data(), &info);
    if (info != 0 || static_cast<std::size_t>(m) != num_states) {
      // Fall back to full tridiagonal solve.
      std::vector<double> wfull, vfull;
      TridiagEig::Solve(d, e, n, wfull, vfull);
      for (std::size_t k = 0; k < num_states; ++k) {
        eigenvalues[k] = wfull[k];
        for (std::size_t j = 0; j < n; ++j)
          eigenvectors[k * n + j] = vfull[k * n + j];
      }
      return;
    }
    for (std::size_t k = 0; k < num_states; ++k) eigenvalues[k] = w[k];
    // z is column-major: eigenvector k is column k = z[k*n + 0..n-1].
    for (std::size_t k = 0; k < num_states; ++k)
      for (std::size_t j = 0; j < n; ++j)
        eigenvectors[k * n + j] = z[k * n + j];
#else
    std::vector<double> wfull, vfull;
    TridiagEig::Solve(d, e, n, wfull, vfull);
    for (std::size_t k = 0; k < num_states; ++k) {
      eigenvalues[k] = wfull[k];
      for (std::size_t j = 0; j < n; ++j)
        eigenvectors[k * n + j] = vfull[k * n + j];
    }
#endif
  }
};

}  // namespace tides::atomgen
