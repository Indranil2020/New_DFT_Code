#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "basis/atomgen/radial_solver.hpp"
#include "basis/atomgen/selective_tridiag_eig.hpp"

// Fortran LAPACK generalized symmetric eigenvalue solver prototype.
extern "C" {
void dsygv_(const int* itype, const char* jobz, const char* uplo,
            const int* n, double* A, const int* lda, double* B,
            const int* ldb, double* w, double* work, const int* lwork,
            int* info);
}

namespace tides::atomgen {

// Numerov solver for the radial Schrodinger/Kohn-Sham equation.
//
// Solves: -1/2 P''(r) + [V(r) + l(l+1)/(2r^2)] P(r) = eps P(r)
// using the 4th-order Numerov finite-difference method on a uniform grid.
//
// The Numerov discretization yields a generalized tridiagonal eigenvalue
// problem (-A) P = E B P where:
//   A_ii = -2 - (5h^2/3)*V_eff_i,  A_i,i±1 = 1 - (h^2/6)*V_eff_{i±1}
//   B_ii = 5h^2/3,                 B_i,i±1 = h^2/6
// with E = eigenvalue (directly) and V_eff = V + l(l+1)/(2r^2).
//
// For l>0: the centrifugal barrier suppresses the wavefunction near r=0,
// giving ~5x better accuracy than standard FD at the same grid size.
// For l=0: the Coulomb singularity at r=0 degrades Numerov to O(h),
// so we fall back to the standard 3-point FD solver (O(h^1.5)).
//
// For m <= 2000: uses LAPACK dsygv_ (dense generalized eigensolver).
// For m > 2000: falls back to SolveUniform.
class NumerovSolver {
 public:
  static std::vector<RadialState> Solve(
      double r_min, double r_max, std::size_t n_r, int l,
      const std::vector<double>& V_at_grid, std::size_t num_states) {
    if (n_r < 6 || V_at_grid.size() != n_r) return {};

    // For l=0, the Coulomb singularity makes Numerov worse than standard FD.
    // Fall back to the uniform solver which has O(h^1.5) convergence.
    if (l == 0) {
      return RadialSolver::SolveUniform(r_min, r_max, n_r, l, V_at_grid,
                                        num_states);
    }

    const std::size_t m = n_r - 2;
    if (m > 2000) {
      return RadialSolver::SolveUniform(r_min, r_max, n_r, l, V_at_grid,
                                        num_states);
    }

    const double h = (r_max - r_min) / static_cast<double>(n_r - 1);
    const double h2 = h * h;

    std::vector<double> r(n_r);
    for (std::size_t i = 0; i < n_r; ++i)
      r[i] = r_min + h * static_cast<double>(i);

    // V_eff = V(r) + l(l+1)/(2r^2)
    std::vector<double> veff(n_r, 0.0);
    for (std::size_t i = 0; i < n_r; ++i) {
      double centrif = (r[i] > 0.0)
                           ? static_cast<double>(l * (l + 1)) / (2.0 * r[i] * r[i])
                           : 0.0;
      veff[i] = V_at_grid[i] + centrif;
    }

    // Numerov: A P = -E B P  =>  (-A) P = E B P
    // A_ii = -2 - (5h^2/3)*veff_i,  A_i,i±1 = 1 - (h^2/6)*veff_{i±1}
    // B_ii = 5h^2/3,  B_i,i±1 = h^2/6
    // (-A)_ii = 2 + (5h^2/3)*veff_i,  (-A)_i,i±1 = -1 + (h^2/6)*veff_{i±1}
    const double five_h2_3 = 5.0 * h2 / 3.0;
    const double h2_6 = h2 / 6.0;

    std::vector<double> Adiag(m), Aoff(m > 0 ? m - 1 : 0);
    std::vector<double> Bdiag(m), Boff(m > 0 ? m - 1 : 0);
    for (std::size_t p = 0; p < m; ++p) {
      const std::size_t gi = p + 1;
      Adiag[p] = 2.0 + five_h2_3 * veff[gi];
      Bdiag[p] = five_h2_3;
      if (p + 1 < m) {
        const std::size_t gp = p + 2;
        Aoff[p] = -1.0 + h2_6 * veff[gp];
        Boff[p] = h2_6;
      }
    }

    // Pack into dense matrices for dsygv_
    std::vector<double> A(m * m, 0.0), B(m * m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
      A[i * m + i] = Adiag[i];
      B[i * m + i] = Bdiag[i];
      if (i + 1 < m) {
        A[i * m + (i + 1)] = Aoff[i];
        A[(i + 1) * m + i] = Aoff[i];
        B[i * m + (i + 1)] = Boff[i];
        B[(i + 1) * m + i] = Boff[i];
      }
    }

#ifdef TIDES_HAVE_LAPACK
    int itype = 1;
    char jobz = 'V';
    char uplo = 'U';
    int nn = static_cast<int>(m);
    int lda = nn, ldb = nn;
    std::vector<double> w(m);
    int lwork = -1;
    double wkopt;
    int info = 0;
    dsygv_(&itype, &jobz, &uplo, &nn, A.data(), &lda, B.data(), &ldb,
           w.data(), &wkopt, &lwork, &info);
    if (info != 0) return {};
    lwork = static_cast<int>(wkopt);
    std::vector<double> work(lwork);
    dsygv_(&itype, &jobz, &uplo, &nn, A.data(), &lda, B.data(), &ldb,
           w.data(), work.data(), &lwork, &info);
    if (info != 0) return {};

    const std::size_t count = std::min(num_states, m);
    std::vector<RadialState> states(count);
    for (std::size_t k = 0; k < count; ++k) {
      states[k].epsilon = w[k];  // E = eigenvalue directly
      states[k].r_grid = r;
      states[k].P.assign(n_r, 0.0);
      for (std::size_t p = 0; p < m; ++p)
        states[k].P[p + 1] = A[k * m + p];
      // Normalize: integral |P|^2 dr = 1
      double norm2 = 0.0;
      for (std::size_t i = 0; i + 1 < n_r; ++i)
        norm2 += 0.5 * (states[k].P[i] * states[k].P[i] +
                        states[k].P[i + 1] * states[k].P[i + 1]) * h;
      const double inv_norm = (norm2 > 0.0) ? 1.0 / std::sqrt(norm2) : 0.0;
      states[k].R.assign(n_r, 0.0);
      for (std::size_t i = 0; i < n_r; ++i) {
        states[k].P[i] *= inv_norm;
        states[k].R[i] = (r[i] > 0.0) ? states[k].P[i] / r[i] : 0.0;
      }
    }
    std::sort(states.begin(), states.end(),
              [](const RadialState& a, const RadialState& b) {
                return a.epsilon < b.epsilon;
              });
    return states;
#else
    return {};
#endif
  }

  static std::vector<RadialState> SolveHydrogenic(
      int Z, int l, std::size_t num_states, double r_max = 60.0,
      std::size_t n_r = 4000) {
    std::vector<double> r(n_r);
    const double h = r_max / static_cast<double>(n_r - 1);
    for (std::size_t i = 0; i < n_r; ++i) r[i] = h * static_cast<double>(i);
    auto V = RadialSolver::CoulombPotential(r, Z);
    return Solve(0.0, r_max, n_r, l, V, num_states);
  }
};

}  // namespace tides::atomgen
