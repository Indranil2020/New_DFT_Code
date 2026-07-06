#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace tides::scf {

// Total energy assembly (T6.2): E = E_kin + E_ne + E_H + E_xc + E_ion.
//
// For the CPU reference, the energy components are computed from the density
// matrix P and the Hamiltonian/overlap matrices. The key observable is
// component-wise match vs PySCF <= 1e-6 Ha/atom.
//
// All sums use FP64 (the production path uses f64e for the critical reductions
// per 33-precision-policy).

struct EnergyComponents {
  double E_kin = 0.0;    // kinetic: Tr(T P) = sum_k f_k eps_k - Tr(V_eff P)
  double E_ne = 0.0;     // nucleus-electron attraction
  double E_H = 0.0;      // Hartree (electron-electron Coulomb)
  double E_xc = 0.0;     // exchange-correlation
  double E_ion = 0.0;    // ion-ion repulsion (Ewald / direct)
  double E_total = 0.0;  // sum of all components
};

class EnergyAssembly {
 public:
  // Compute the total energy from the density matrix and orbital eigenvalues.
  //   sum_eps:   sum of occupied eigenvalues (sum_k f_k eps_k)
  //   P:         density matrix (n x n, row-major)
  //   H:         Hamiltonian (n x n, row-major; includes V_ext + V_H + V_xc)
  //   S:         overlap matrix (n x n, row-major)
  //   V_H:       Hartree potential matrix (n x n, = integral V_H(r) phi_i phi_j)
  //   V_xc:      XC potential matrix (n x n)
  //   eps_xc:    XC energy density matrix (n x n, = integral eps_xc(r) phi_i phi_j)
  //   V_ext:     external (nucleus-electron) potential matrix (n x n)
  //   E_ion:     ion-ion energy (precomputed)
  static EnergyComponents Compute(double sum_eps, const std::vector<double>& P,
                                  const std::vector<double>& V_H,
                                  const std::vector<double>& V_xc,
                                  const std::vector<double>& eps_xc_mat,
                                  const std::vector<double>& V_ext,
                                  const std::vector<double>& S, std::size_t n,
                                  double E_ion = 0.0) {
    EnergyComponents e;
    // tr(A B) = sum_ij A_ij B_ji = sum_ij A_ij B_ij (for symmetric matrices).
    auto trace = [&](const std::vector<double>& A, const std::vector<double>& B) {
      double s = 0.0;
      for (std::size_t i = 0; i < n * n; ++i) s += A[i] * B[i];
      return s;
    };

    // E_ne = Tr(P V_ext)
    e.E_ne = trace(P, V_ext);
    // E_H = 0.5 * Tr(P V_H)
    e.E_H = 0.5 * trace(P, V_H);
    // E_xc = Tr(P eps_xc_mat)  (NOT Tr(P V_xc); eps_xc is the energy density)
    e.E_xc = trace(P, eps_xc_mat);
    // E_kin = sum_eps - Tr(P (V_ext + V_H + V_xc))
    //       = sum_eps - E_ne - 2*E_H - Tr(P V_xc)
    e.E_kin = sum_eps - e.E_ne - 2.0 * e.E_H - trace(P, V_xc);
    // E_ion
    e.E_ion = E_ion;
    // Total
    e.E_total = e.E_kin + e.E_ne + e.E_H + e.E_xc + e.E_ion;
    return e;
  }

  // Ewald ion-ion energy for a set of point charges (periodic).
  // For the CPU reference: direct sum (free BC) or simple Ewald (periodic).
  // This is a stub that returns 0 for molecular (free BC) systems.
  static double EwaldIonIon(const std::vector<double>& positions,
                            const std::vector<double>& charges,
                            bool periodic = false, double alpha = 0.0) {
    if (!periodic) {
      // Direct Coulomb sum: E = 0.5 * sum_{i!=j} Z_i Z_j / |r_i - r_j|
      const std::size_t n_atoms = charges.size();
      double E = 0.0;
      for (std::size_t i = 0; i < n_atoms; ++i)
        for (std::size_t j = i + 1; j < n_atoms; ++j) {
          const double dx = positions[3 * i] - positions[3 * j];
          const double dy = positions[3 * i + 1] - positions[3 * j + 1];
          const double dz = positions[3 * i + 2] - positions[3 * j + 2];
          const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
          if (r > 1e-10) E += charges[i] * charges[j] / r;
        }
      return E;
    }
    return 0.0;  // periodic Ewald is a separate implementation
  }
};

}  // namespace tides::scf
