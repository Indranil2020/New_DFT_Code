#pragma once

// Atomic LDA solver: self-consistent radial Kohn-Sham for a spherically
// symmetric atom. Produces the total LDA energy needed by T2.1 observable (2)
// (Ne LDA total energy vs reference) and underpins T2.2 NAO generation (the
// confined-atom solver is this solver with an added confinement potential).
//
// Physics (Hartree atomic units):
//   V_eff(r) = -Z/r + V_H(r) + V_xc(r)
//   V_H(r) = (4 pi / r) int_0^r n(r') r'^2 dr' + 4 pi int_r^inf n(r') r' dr'
//   n(r) = sum_{nl occ} 2(2l+1) |R_nl(r)|^2   (closed-shell, spin-paired)
//   E = sum_occ eps_i + int [V_ext - V_H/2 - V_xc + eps_xc] n 4 pi r^2 dr
//
// SCF: linear mixing of the density (Pulay would converge faster but linear
// suffices for these small atoms at the accuracy we need). The radial KS solve
// reuses RadialSolver::SolveUniform with the current V_eff.

#include <cmath>
#include <cstddef>
#include <vector>

#include "basis/atomgen/lda_xc.hpp"
#include "basis/atomgen/radial_solver.hpp"

namespace tides::atomgen {

struct AtomConfig {
  int Z = 0;
  // Occupation per (n,l) shell: each entry is {n_principal, l, occupancy}
  // (occupation includes spin; closed shell = 2(2l+1)).
  struct Shell { int n; int l; int occ; };
  std::vector<Shell> shells;
};

struct AtomicResult {
  double total_energy = 0.0;
  std::vector<double> eigenvalues;  // per occupied shell
  std::vector<double> density;     // n(r) on the grid
  std::vector<double> r_grid;
  int n_scf_iter = 0;
  bool converged = false;
};

class AtomicLDA {
 public:
  // Hartree potential V_H(r) for a spherical density n(r):
  //   V_H(r) = (4 pi / r) int_0^r n(r') r'^2 dr' + 4 pi int_r^inf n(r') r' dr'
  // Computed via cumulative integrals (trapezoidal). The boundary condition
  // V_H(inf) = 0 sets the integration constant. Public so tests/probes can
  // validate it independently.
  static std::vector<double> HartreePotential(const std::vector<double>& r,
                                             const std::vector<double>& n) {
    const std::size_t N = r.size();
    std::vector<double> VH(N, 0.0);
    if (N < 2) return VH;
    const double h = r[1] - r[0];
    // U(r) = 4 pi int_0^r n(r') r'^2 dr' (cumulative).
    std::vector<double> U(N, 0.0);
    for (std::size_t i = 1; i < N; ++i)
      U[i] = U[i - 1] + 0.5 * 4.0 * M_PI *
                            (n[i - 1] * r[i - 1] * r[i - 1] +
                             n[i] * r[i] * r[i]) * h;
    // W(r) = 4 pi int_r^inf n(r') r' dr' (cumulative from the right).
    std::vector<double> W(N, 0.0);
    for (std::size_t i = N - 1; i-- > 0;)
      W[i] = W[i + 1] + 0.5 * 4.0 * M_PI *
                            (n[i] * r[i] + n[i + 1] * r[i + 1]) * h;
    for (std::size_t i = 1; i < N; ++i)
      VH[i] = U[i] / r[i] + W[i];
    // At r=0 the regular limit is V_H(0) = 4 pi int_0^inf n(r') r' dr' = W[0].
    VH[0] = W[0];
    return VH;
  }
  // Solve the atomic LDA problem. r_max and n_r set the radial grid; alpha is
  // the linear mixing parameter; tol the energy convergence target.
  static AtomicResult Solve(const AtomConfig& cfg, double r_max = 50.0,
                            std::size_t n_r = 4000, double alpha = 0.35,
                            double tol = 1e-9, int max_iter = 200) {
    AtomicResult res;
    res.r_grid.resize(n_r);
    const double h = r_max / static_cast<double>(n_r - 1);
    for (std::size_t i = 0; i < n_r; ++i) res.r_grid[i] = h * static_cast<double>(i);

    // Initial potential: bare nucleus (hydrogenic guess).
    std::vector<double> V(n_r);
    for (std::size_t i = 0; i < n_r; ++i)
      V[i] = (res.r_grid[i] > 0.0) ? -static_cast<double>(cfg.Z) / res.r_grid[i] : 0.0;

    double prev_energy = 1e30;
    res.density.assign(n_r, 0.0);
    std::vector<double> V_eff = V;

    for (int iter = 0; iter < max_iter; ++iter) {
      res.n_scf_iter = iter + 1;

      // Solve KS for each l, collect occupied states and build density.
      // Determine l_max from the config.
      int l_max = 0;
      for (const auto& sh : cfg.shells) l_max = std::max(l_max, sh.l);

      // Build density from scratch each iteration (simple Anderson-free SCF).
      std::vector<double> n_new(n_r, 0.0);
      // occ_eigs stores each occupied eigenvalue weighted by its occupation
      // (number of electrons), so sum(occ_eigs) = sum_i f_i eps_i (the KS
      // sum that enters the total energy).
      std::vector<double> occ_eigs;
      std::vector<int> max_n_per_l(l_max + 1, 0);
      for (const auto& sh : cfg.shells)
        max_n_per_l[sh.l] = std::max(max_n_per_l[sh.l], sh.n);

      for (int l = 0; l <= l_max; ++l) {
        if (max_n_per_l[l] == 0) continue;
        const std::size_t nstates = static_cast<std::size_t>(max_n_per_l[l] - l);
        auto states = RadialSolver::SolveUniform(
            0.0, r_max, n_r, l, V_eff, nstates);
        for (std::size_t k = 0; k < states.size(); ++k) {
          const int n_principal = l + static_cast<int>(k) + 1;
          int occ = 0;
          for (const auto& sh : cfg.shells)
            if (sh.n == n_principal && sh.l == l) { occ = sh.occ; break; }
          if (occ == 0) continue;
          // Push the eigenvalue `occ` times so sum(occ_eigs) = sum f_i eps_i.
          for (int e = 0; e < occ; ++e) occ_eigs.push_back(states[k].epsilon);
          // Physical 3D density for a closed shell:
          //   rho(r) = (occ/(4 pi)) |R(r)|^2,  so integral rho 4pi r^2 dr = occ.
          // R is normalized to integral |R|^2 r^2 dr = 1.
          const double fac = static_cast<double>(occ) / (4.0 * M_PI);
          for (std::size_t i = 0; i < n_r; ++i)
            n_new[i] += fac * states[k].R[i] * states[k].R[i];
        }
      }

      // Mix density.
      for (std::size_t i = 0; i < n_r; ++i)
        res.density[i] = (1.0 - alpha) * res.density[i] + alpha * n_new[i];

      // Build new V_eff from the mixed density.
      const std::vector<double> V_H = HartreePotential(res.r_grid, res.density);
      std::vector<double> V_xc(n_r, 0.0), eps_xc(n_r, 0.0);
      for (std::size_t i = 0; i < n_r; ++i) {
        const double n = std::max(0.0, res.density[i]);
        V_xc[i] = LdaXC::VXC(n, 0.0);
        eps_xc[i] = LdaXC::EpsXC(n, 0.0);
      }
      for (std::size_t i = 0; i < n_r; ++i)
        V_eff[i] = ((res.r_grid[i] > 0.0)
                        ? -static_cast<double>(cfg.Z) / res.r_grid[i] : 0.0) +
                   V_H[i] + V_xc[i];

      // Total energy.
      const double E = TotalEnergy(res.r_grid, res.density, V_H, eps_xc, V_eff,
                                   occ_eigs, cfg.Z);

      if (std::fabs(E - prev_energy) < tol) {
        res.converged = true;
        res.total_energy = E;
        res.eigenvalues = occ_eigs;
        return res;
      }
      prev_energy = E;
      res.total_energy = E;
      res.eigenvalues = occ_eigs;
    }
    return res;
  }

  // Total LDA energy. With KS eigenvalues from the FULL effective potential
  // (V_ext + V_H + V_xc, as our RadialSolver uses), the standard formula is
  //   E = sum_occ eps_i + int [-V_H/2 - V_xc + eps_xc] n 4pi r^2 dr
  // The V_ext term cancels between the kinetic (T_s = sum eps - int V_eff) and
  // the nucleus-electron energy. V_xc is recovered from V_eff = V_ext+V_H+V_xc.
  static double TotalEnergy(const std::vector<double>& r,
                            const std::vector<double>& n,
                            const std::vector<double>& VH,
                            const std::vector<double>& eps_xc,
                            const std::vector<double>& V_eff,
                            const std::vector<double>& occ_eigs, int Z) {
    const std::size_t N = r.size();
    if (N < 2) return 0.0;
    const double h = r[1] - r[0];
    double sum_eps = 0.0;
    for (double e : occ_eigs) sum_eps += e;
    double integ = 0.0;
    for (std::size_t i = 0; i + 1 < N; ++i) {
      const double w0 = 4.0 * M_PI * r[i] * r[i];
      const double w1 = 4.0 * M_PI * r[i + 1] * r[i + 1];
      // V_xc = V_eff - V_ext - V_H, with V_ext = -Z/r (guarded at r=0).
      const double Vext0 = (r[i] > 0.0) ? -static_cast<double>(Z) / r[i] : 0.0;
      const double Vext1 = (r[i + 1] > 0.0) ? -static_cast<double>(Z) / r[i + 1] : 0.0;
      const double Vxc0 = V_eff[i] - Vext0 - VH[i];
      const double Vxc1 = V_eff[i + 1] - Vext1 - VH[i + 1];
      const double f0 = (-0.5 * VH[i] - Vxc0 + eps_xc[i]) * n[i] * w0;
      const double f1 = (-0.5 * VH[i + 1] - Vxc1 + eps_xc[i + 1]) * n[i + 1] * w1;
      integ += 0.5 * (f0 + f1) * h;
    }
    return sum_eps + integ;
  }
};

}  // namespace tides::atomgen
