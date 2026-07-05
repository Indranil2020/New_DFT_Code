// Convergence probe for the log-grid radial solver on the hydrogenic problem.
// For a given grid spacing h, measures the eigenvalue error vs the exact
// epsilon_n = -Z^2/(2 n^2). This diagnoses the achieved order of accuracy and
// the h needed to reach the 1e-10 Ha target, by sweeping h.

#include "basis/atomgen/dense_sym_eig.hpp"
#include "basis/atomgen/radial_grid.hpp"
#include "basis/atomgen/radial_solver.hpp"

#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>

namespace {

using tides::atomgen::DenseSymEig;
using tides::atomgen::LogGrid;
using tides::atomgen::RadialSolver;
using tides::atomgen::RadialState;

// Patch the solver to use the dense eigensolver (via a local re-solve using
// LAPACK) by reconstructing the same operator. To avoid duplicating the
// assembly, we expose it by calling Solve with the Jacobi path disabled only
// if TIDES_HAVE_LAPACK is defined. Since radial_solver.hpp already calls the
// self-contained Jacobi, this probe builds the operator directly to use the
// chosen DenseSymEig path and to also test an alternative boundary treatment.

std::vector<RadialState> SolveWith(const LogGrid& grid, int Z, int l,
                                   const std::vector<double>& V,
                                   std::size_t num_states,
                                   bool robin_boundary) {
  const std::size_t n = grid.size();
  const std::size_t m = (n >= 2) ? n - 2 : 0;
  if (m == 0) return {};
  const double h = grid.h();
  const auto& r = grid.r();

  static constexpr double kStencil[9] = {
      1.0 / 560.0,  -8.0 / 315.0, 1.0 / 5.0,    -8.0 / 5.0,
      205.0 / 72.0, -8.0 / 5.0,   1.0 / 5.0,    -8.0 / 315.0,
      1.0 / 560.0,
  };
  const double inv_h2 = 1.0 / (h * h);

  std::vector<double> hp(m * m, 0.0);
  for (std::size_t p = 0; p < m; ++p) {
    const std::size_t g = p + 1;
    const double Wg = 0.25 + static_cast<double>(l * (l + 1)) +
                      2.0 * r[g] * r[g] * V[g];
    for (int s = 0; s < 9; ++s) {
      const int offset = s - 4;
      const long gp = static_cast<long>(g) + offset;
      if (gp < 1 || gp > static_cast<long>(n - 2)) continue;
      const std::size_t q = static_cast<std::size_t>(gp) - 1;
      const double kin = kStencil[s] * inv_h2;
      const double entry = kin + (offset == 0 ? Wg : 0.0);
      hp[p * m + q] += entry / (r[g] * r[static_cast<std::size_t>(gp)]);
    }
  }
  // Robin-style correction at the right boundary: account for the wavefunction
  // not being exactly zero at r_max by adding a one-sided 2nd-derivative
  // correction. The simplest robust approach tested here is to enlarge the
  // domain; robin_boundary toggles a trial correction term on the last row.
  if (robin_boundary && m >= 3) {
    // Approximate psi'(x_max) relation psi'' ~ -lambda*psi at the boundary by
    // absorbing the omitted boundary term into the diagonal of the last
    // interior row (a first-order correction; tests whether it helps).
    const std::size_t g_last = (m - 1) + 1;
    const double rl = r[g_last];
    const double rl1 = r[g_last - 1];
    const double w_corr = 2.0 * rl * rl * V[g_last] + 0.25 + l * (l + 1);
    hp[(m - 1) * m + (m - 1)] += (0.5 * inv_h2) * (rl1 / rl) * (1.0 / (rl * rl));
    (void)w_corr;
  }
  for (std::size_t p = 0; p < m; ++p)
    for (std::size_t q = p + 1; q < m; ++q) {
      double avg = 0.5 * (hp[p * m + q] + hp[q * m + p]);
      hp[p * m + q] = avg;
      hp[q * m + p] = avg;
    }

  std::vector<double> evals, evecs;
  DenseSymEig::Solve(hp, m, evals, evecs);

  std::size_t count = std::min(num_states, m);
  std::vector<RadialState> states(count);
  const double sqrt_h = std::sqrt(h);
  for (std::size_t k = 0; k < count; ++k) {
    states[k].epsilon = evals[k] / 2.0;
    states[k].R.assign(n, 0.0);
    for (std::size_t p = 0; p < m; ++p) {
      const std::size_t g = p + 1;
      const double z = evecs[p * m + k];
      states[k].R[g] = z / (sqrt_h * std::pow(r[g], 1.5));
    }
  }
  (void)Z;
  return states;
}

void Run(int Z, int l, const std::vector<double>& expected,
         const std::string& label) {
  std::cout << "\n=== " << label << " (Z=" << Z << " l=" << l << ") ===\n";
  std::cout << std::setw(8) << "h" << std::setw(14) << "1s_err"
            << std::setw(14) << "2s_err" << std::setw(14) << "3s_err"
            << std::setw(12) << "order" << std::setw(12) << "time_ms"
            << '\n';
  double prev_err = 0.0;
  double prev_h = 0.0;
  for (double h : {0.08, 0.04, 0.02, 0.015, 0.0125}) {
    const double x_min = -14.0, x_max = 5.0;
    std::size_t npts = static_cast<std::size_t>((x_max - x_min) / h) + 1;
    LogGrid grid(x_min, x_max, npts);
    std::vector<double> V =
        tides::atomgen::RadialSolver::CoulombPotential(grid, Z);

    auto t0 = std::chrono::steady_clock::now();
    auto states = SolveWith(grid, Z, l, V, expected.size(), false);
    auto t1 = std::chrono::steady_clock::now();
    double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    double e1 = std::fabs(states[0].epsilon - expected[0]);
    double e2 = (states.size() > 1) ? std::fabs(states[1].epsilon - expected[1]) : 0.0;
    double e3 = (states.size() > 2) ? std::fabs(states[2].epsilon - expected[2]) : 0.0;
    double order = 0.0;
    if (prev_err > 0 && e1 > 0 && prev_h > 0) {
      order = std::log(prev_err / e1) / std::log(prev_h / h);
    }
    std::cout << std::setw(8) << h << std::scientific << std::setprecision(3)
              << std::setw(14) << e1 << std::setw(14) << e2 << std::setw(14)
              << e3 << std::defaultfloat << std::setprecision(3)
              << std::setw(12) << order << std::setw(12) << ms << '\n';
    prev_err = e1;
    prev_h = h;
  }
}

}  // namespace

int main() {
  Run(1, 0, {-0.5, -0.125, -1.0 / 18.0}, "H_dirichlet_l0");
  Run(2, 0, {-2.0, -0.5}, "He_dirichlet_l0");
  return 0;
}
