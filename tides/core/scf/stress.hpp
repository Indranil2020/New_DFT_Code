#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

namespace tides::scf {

// Stress tensor (T6.4): sigma_ab = (1/V) dE/de_ab, where e_ab is the strain.
//
// For the CPU reference, we compute the stress via 5-point FD on the energy
// as a function of cell strain. The analytic stress (from the force
// expression, via the virial) is the production path.
//
// Observable (T6.4): FD vs cell strain <= 1e-6 Ha (FP64 path) on Si and NaCl.

class StressTensor {
 public:
  // Compute the stress tensor via FD on the energy.
  //   energy_fn: given the strain tensor (3x3, flattened), returns E(strained).
  //   V:          cell volume.
  //   h:          FD step for the strain.
  static std::vector<double> ComputeFD(
      const std::function<double(const std::vector<double>&)>& energy_fn,
      double V, double h = 1e-5) {
    std::vector<double> stress(9, 0.0);  // 3x3, row-major
    // For each strain component e_ab:
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        // 5-point FD: dE/de_ab = (-E(+2h) + 8E(+h) - 8E(-h) + E(-2h)) / (12h)
        std::vector<double> eps_p2(9, 0), eps_p1(9, 0), eps_m1(9, 0), eps_m2(9, 0);
        eps_p2[a * 3 + b] = 2.0 * h;
        eps_p1[a * 3 + b] = h;
        eps_m1[a * 3 + b] = -h;
        eps_m2[a * 3 + b] = -2.0 * h;
        double E_p2 = energy_fn(eps_p2);
        double E_p1 = energy_fn(eps_p1);
        double E_m1 = energy_fn(eps_m1);
        double E_m2 = energy_fn(eps_m2);
        stress[a * 3 + b] = (-E_p2 + 8.0 * E_p1 - 8.0 * E_m1 + E_m2) / (12.0 * h * V);
      }
    return stress;
  }

  // Virial stress: sigma_ab = (1/V) sum_i F_i_a * r_i_b  (kinetic + potential).
  // This is the analytic path (computed from forces and positions).
  static std::vector<double> VirialStress(
      const std::vector<double>& forces, const std::vector<double>& positions,
      double V) {
    std::vector<double> stress(9, 0.0);
    const std::size_t n_atoms = forces.size() / 3;
    for (std::size_t i = 0; i < n_atoms; ++i)
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
          stress[a * 3 + b] += forces[3 * i + a] * positions[3 * i + b];
    for (auto& s : stress) s /= V;
    return stress;
  }
};

}  // namespace tides::scf
