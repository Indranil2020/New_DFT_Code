#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "basis/atomgen/selective_tridiag_eig.hpp"

namespace tides::atomgen {

// Radial Kohn-Sham / Schrodinger solver.
//
// Solves the standard radial equation for P(r) = r R(r):
//   -1/2 P''(r) + [V(r) + l(l+1)/(2 r^2)] P(r) = eps P(r)
// with P(0) = P(r_max) = 0 (Dirichlet). Validated against the exact hydrogenic
// eigenvalues eps_n = -Z^2/(2 n^2) (see tests/hydrogenic_tests.cpp) and the
// independent Python FP64 oracle.
//
// Discretization: 2nd-order 3-point central FD for -d^2/dr^2 on a uniform
// r-grid, restricted to interior points. The resulting TRIDIAGONAL symmetric
// eigenproblem is solved with LAPACK dstev_ (O(n^2), ~1000x faster than dense
// at n~3000), letting us reach the 1e-10 target at n~16000.
struct RadialState {
  double epsilon = 0.0;        // eigenvalue in Hartree
  std::vector<double> R;       // normalized radial function, integral |R|^2 r^2 dr = 1
  std::vector<double> P;       // P = r R, on the grid
  std::vector<double> r_grid;  // the grid points
};

class RadialSolver {
 public:
  // Solve for the `num_states` lowest states of angular momentum `l` given a
  // potential V sampled at the grid points. V must NOT include the centrifugal
  // term (added internally). Returns states sorted by ascending epsilon.
  static std::vector<RadialState> SolveUniform(
      double r_min, double r_max, std::size_t n_r, int l,
      const std::vector<double>& V_at_grid, std::size_t num_states) {
    if (n_r < 4 || V_at_grid.size() != n_r) return {};
    const double h = (r_max - r_min) / static_cast<double>(n_r - 1);
    std::vector<double> r(n_r);
    for (std::size_t i = 0; i < n_r; ++i)
      r[i] = r_min + h * static_cast<double>(i);

    // Interior points: indices 1..n_r-2 (Dirichlet P=0 at 0 and r_max).
    const std::size_t m = n_r - 2;
    // Tridiagonal: diagonal d[0..m-1], off-diagonal e[0..m-2].
    // H = -1/2 d^2/dr^2 + diag(V + centrif).
    // -1/2 d^2/dr^2 with 3-point stencil: diagonal +1/h^2, off-diagonal -0.5/h^2.
    const double inv_h2 = 1.0 / (h * h);
    std::vector<double> d(m), e(m > 0 ? m - 1 : 0);
    for (std::size_t p = 0; p < m; ++p) {
      const std::size_t g = p + 1;
      const double rg = r[g];
      const double centrif = (l > 0)
                                 ? static_cast<double>(l * (l + 1)) / (2.0 * rg * rg)
                                 : 0.0;
      d[p] = inv_h2 + V_at_grid[g] + centrif;
      if (p + 1 < m) e[p] = -0.5 * inv_h2;
    }

    std::vector<double> evals, evecs;
    SelectiveTridiagEig::Solve(d, e, m, num_states, evals, evecs);

    const std::size_t count = std::min(num_states, m);
    std::vector<RadialState> states(count);
    for (std::size_t k = 0; k < count; ++k) {
      states[k].epsilon = evals[k];
      states[k].r_grid = r;
      states[k].P.assign(n_r, 0.0);
      for (std::size_t p = 0; p < m; ++p) {
        const std::size_t g = p + 1;
        // evecs[k * m + p]: eigenvalue k, component p (interior index).
        states[k].P[g] = evecs[k * m + p];
      }
      // Normalize P: integral |P|^2 dr = 1  =>  integral |R|^2 r^2 dr = 1.
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
    return states;
  }

  // Bare-nucleus Coulomb potential V(r) = -Z/r.
  static std::vector<double> CoulombPotential(
      const std::vector<double>& r, int Z) {
    std::vector<double> V(r.size());
    for (std::size_t i = 0; i < r.size(); ++i)
      V[i] = (r[i] > 0.0) ? -static_cast<double>(Z) / r[i] : 0.0;
    return V;
  }

  // Convenience: solve hydrogenic (bare Coulomb) on a uniform grid.
  static std::vector<RadialState> SolveHydrogenic(
      int Z, int l, std::size_t num_states, double r_max = 60.0,
      std::size_t n_r = 8000) {
    std::vector<double> r(n_r);
    const double h = r_max / static_cast<double>(n_r - 1);
    for (std::size_t i = 0; i < n_r; ++i) r[i] = h * static_cast<double>(i);
    auto V = CoulombPotential(r, Z);
    return SolveUniform(0.0, r_max, n_r, l, V, num_states);
  }
};

}  // namespace tides::atomgen
