#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

namespace tides::forces {

// Analytic forces (T6.3) — the GA1 evidence gate.
//
// Force on atom I: F_I = -dE/dR_I.
// Components (per 10-physics/16):
//   F_HF  = -Tr(P dH/dR)                  (Hellmann-Feynman, the dominant term)
//   F_Pul = -Tr(dP/dR H) + ...            (Pulay, from basis movement; NAOs move)
//   F_grid = -d(E_xc + E_H_grid)/dR       (grid terms)
//   F_disp = -d(E_disp)/dR                (dispersion)
//   F_ion  = -d(E_ion)/dR                 (ion-ion Ewald)
//
// For the CPU reference, we implement:
//   (1) The Hellmann-Feynman force F_HF = -Tr(P dH/dR) analytically (given
//       dH/dR from WP2's derivative streams T2.6).
//   (2) 5-point FD validation: F_FD = -(E(R+2h) - 8E(R+h) + 8E(R-h) - E(R-2h))/(12h).
//
// Observable (T6.3): 5-point FD <= 1e-6 Ha/Bohr (FP64 path).

struct ForceResult {
  std::vector<double> forces;  // 3 * n_atoms, F_Ix, F_Iy, F_Iz
  bool fd_validated = false;
  double max_fd_error = 0.0;  // max |F_analytic - F_fd| over all atoms/components
};

class AnalyticForces {
 public:
  // Compute Hellmann-Feynman forces: F_I = -Tr(P dH/dR_I).
  //   P:       density matrix (n x n)
  //   dH_dR:   array of dH/dR for each atom and component (3*n_atoms matrices)
  //   n:       matrix dimension
  //   n_atoms: number of atoms
  static std::vector<double> HellmannFeynman(
      const std::vector<double>& P,
      const std::vector<std::vector<double>>& dH_dR,
      std::size_t n, std::size_t n_atoms) {
    std::vector<double> forces(3 * n_atoms, 0.0);
    for (std::size_t a = 0; a < n_atoms; ++a)
      for (int c = 0; c < 3; ++c) {
        const std::size_t idx = 3 * a + c;
        if (idx >= dH_dR.size()) continue;
        const auto& dH = dH_dR[idx];
        // F = -Tr(P dH/dR) = -sum_ij P_ij dH_ji = -sum_ij P_ij dH_ij (symmetric)
        double f = 0.0;
        for (std::size_t i = 0; i < n * n; ++i) f += P[i] * dH[i];
        forces[idx] = -f;
      }
    return forces;
  }

  // 5-point central FD force: F = -(E(R+2h) - 8E(R+h) + 8E(R-h) - E(R-2h))/(12h).
  //   energy_fn: given positions (3*n_atoms), returns E
  //   positions: current positions (3*n_atoms)
  //   atom_idx: which atom to displace
  //   component: 0=x, 1=y, 2=z
  //   h:         FD step
  static double FD5Force(const std::function<double(const std::vector<double>&)>& energy_fn,
                         std::vector<double> positions, std::size_t atom_idx,
                         int component, double h = 0.001) {
    const std::size_t idx = 3 * atom_idx + component;
    std::vector<double> p2 = positions, p1 = positions, m1 = positions, m2 = positions;
    p2[idx] += 2.0 * h;
    p1[idx] += h;
    m1[idx] -= h;
    m2[idx] -= 2.0 * h;
    double E2 = energy_fn(p2);
    double E1 = energy_fn(p1);
    double Em1 = energy_fn(m1);
    double Em2 = energy_fn(m2);
    return (E2 - 8.0 * E1 + 8.0 * Em1 - Em2) / (12.0 * h);
  }

  // Validate analytic forces against FD for all atoms/components.
  static ForceResult Validate(
      const std::vector<double>& analytic_forces,
      const std::function<double(const std::vector<double>&)>& energy_fn,
      const std::vector<double>& positions, std::size_t n_atoms,
      double h = 0.001, double tol = 1e-6) {
    ForceResult res;
    res.forces = analytic_forces;
    res.max_fd_error = 0.0;
    for (std::size_t a = 0; a < n_atoms; ++a)
      for (int c = 0; c < 3; ++c) {
        double fd = FD5Force(energy_fn, positions, a, c, h);
        double err = std::fabs(analytic_forces[3 * a + c] - fd);
        res.max_fd_error = std::max(res.max_fd_error, err);
      }
    res.fd_validated = (res.max_fd_error < tol);
    return res;
  }
};

}  // namespace tides::forces
