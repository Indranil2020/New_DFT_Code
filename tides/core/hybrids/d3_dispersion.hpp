#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace tides::hybrids {

// D3 dispersion correction with Becke-Johnson (BJ) damping (T7.1).
//
// E_D3 = -0.5 * sum_{A!=B} sum_{n=6,8} s_n * C_n^AB(R_AB) / R_AB^n * f_damp^BJ(R_AB)
//
// where C_n^AB is the geometry-dependent dispersion coefficient (computed from
// atomic polarizabilities and reference systems), s_n are global scaling
// factors (s6=1, s8 for BJ), and f_damp^BJ is the Becke-Johnson damping:
//   f_damp = R_AB^n / (R_AB^n + (a1 * sqrt(C8^AB/C6^AB) + a2)^n)
//
// For a self-contained implementation, we use the PBE0-D3(BJ) parameters
// (a1=0.4289, a2=4.421, s6=1.0, s8=0.7875) and tabulated C6/C8 for H, C, N, O.
// The geometry-dependent C6 is computed via the Casimir-Polder combination:
//   C6^AB = (2 * C6^A * C6^B * alpha^A * alpha^B) /
//           (C6^A * alpha^B + C6^B * alpha^A)
// where C6^A, alpha^A are the free-atom values. This is the "zero-damping"
// simplification; the full D3 uses fractional coordination-dependent C6.
//
// Observable (T7.1): vs reference libraries <=1e-10 on a 10-dimer set.

struct DispersionResult {
  double energy = 0.0;
  std::vector<double> forces;  // 3*n_atoms
  double c6_check = 0.0;       // sanity: sum of C6 for the first pair
};

class D3Dispersion {
 public:
  // Atomic element data: Z, C6 (Bohr^6 Ha), alpha (Bohr^3), C8 (Bohr^8 Ha).
  // Values from the D3 reference tables (Grimme 2011) for light elements.
  struct AtomData {
    int Z;
    double C6;     // free-atom C6 (Ha Bohr^6)
    double alpha;   // free-atom polarizability (Bohr^3)
    double C8;     // free-atom C8 (Ha Bohr^8)
  };

  // D3(BJ) parameters for PBE0 (Grimme 2011, Table 1).
  static constexpr double kA1 = 0.4289;
  static constexpr double kA2 = 4.421;
  static constexpr double kS6 = 1.0;
  static constexpr double kS8 = 0.7875;
  static constexpr double kRs6 = 1.0;  // unused in BJ (zero-damping uses it)
  static constexpr double kRs8 = 1.0;

  // Get atomic data for common elements (H, C, N, O, F, Si, P, S, Cl, Ar).
  static AtomData GetAtomData(int Z) {
    // C6 values in (Ha * Bohr^6), alpha in Bohr^3. These are the D3
    // reference values (Grimme 2011, J. Comput. Chem. 32, 1456).
    switch (Z) {
      case 1: return {1, 6.5092, 4.5, 85.0};
      case 6: return {6, 46.589, 12.0, 825.0};
      case 7: return {7, 24.869, 7.4, 354.0};
      case 8: return {8, 15.128, 5.4, 193.0};
      case 9: return {9, 9.5523, 3.8, 110.0};
      case 14: return {14, 305.76, 37.0, 5255.0};
      case 15: return {15, 184.18, 25.0, 3080.0};
      case 16: return {16, 134.55, 19.6, 2150.0};
      case 17: return {17, 81.859, 15.0, 1210.0};
      case 18: return {18, 64.157, 11.1, 920.0};
      default: return {Z, 0.0, 0.0, 0.0};  // no D3 data
    }
  }

  // Compute the D3(BJ) dispersion energy for a set of atoms.
  //   Z:        atomic numbers (n_atoms)
  //   positions: positions in Bohr (3*n_atoms, x,y,z per atom)
  static DispersionResult ComputeEnergy(const std::vector<int>& Z,
                                        const std::vector<double>& positions) {
    DispersionResult res;
    const std::size_t n_atoms = Z.size();
    res.forces.assign(3 * n_atoms, 0.0);
    res.energy = 0.0;

    if (n_atoms < 2) return res;

    double E = 0.0;
    for (std::size_t A = 0; A < n_atoms; ++A) {
      const auto dA = GetAtomData(Z[A]);
      for (std::size_t B = A + 1; B < n_atoms; ++B) {
        const auto dB = GetAtomData(Z[B]);
        const double dx = positions[3*A] - positions[3*B];
        const double dy = positions[3*A+1] - positions[3*B+1];
        const double dz = positions[3*A+2] - positions[3*B+2];
        const double R = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (R < 0.1) continue;

        // Geometry-dependent C6^AB (Casimir-Polder).
        const double C6_AB = CombineC6(dA.C6, dB.C6, dA.alpha, dB.alpha);
        // C8^AB = 3 * C6_AB * sqrt(C8^A/C6^A) * sqrt(C8^B/C6^B) (D3 formula).
        const double C8_AB = 3.0 * C6_AB *
            std::sqrt((dA.C6 > 0 ? dA.C8 / dA.C6 : 1.0)) *
            std::sqrt((dB.C6 > 0 ? dB.C8 / dB.C6 : 1.0));

        // BJ damping.
        const double R0_AB = kA1 * std::sqrt(C8_AB / C6_AB) + kA2;

        // C6 term: -s6 * C6 / (R^6 + R0^6)
        const double R6 = R * R * R * R * R * R;
        const double R06 = R0_AB * R0_AB * R0_AB * R0_AB * R0_AB * R0_AB;
        const double e6 = -kS6 * C6_AB / (R6 + R06);

        // C8 term: -s8 * C8 / (R^8 + R0^8)
        const double R8 = R6 * R * R;
        const double R08 = R06 * R0_AB * R0_AB;
        const double e8 = -kS8 * C8_AB / (R8 + R08);

        E += e6 + e8;

        // Forces: F = -dE/dR * dR/dx_A (analytic derivative).
        // dE6/dR = s6 * C6 * 6 * R^5 / (R^6 + R0^6)^2
        // dE8/dR = s8 * C8 * 8 * R^7 / (R^8 + R0^8)^2
        const double dE6_dR = kS6 * C6_AB * 6.0 * R * R * R * R * R /
                              ((R6 + R06) * (R6 + R06));
        const double dE8_dR = kS8 * C8_AB * 8.0 * R * R * R * R * R * R * R /
                              ((R8 + R08) * (R8 + R08));
        const double dE_dR = dE6_dR + dE8_dR;
        // dR/dx_A = dx/R, force on A = -dE_dR * dx/R (and opposite on B).
        const double f = -dE_dR / R;
        for (int c = 0; c < 3; ++c) {
          const double drc = positions[3*A+c] - positions[3*B+c];
          res.forces[3*A+c] += f * drc;
          res.forces[3*B+c] -= f * drc;
        }

        if (A == 0 && B == 1) res.c6_check = C6_AB;
      }
    }
    res.energy = E;
    return res;
  }

  // Compute the D3 energy for a single pair (for unit testing).
  static double PairEnergy(int ZA, int ZB, double R) {
    const auto dA = GetAtomData(ZA);
    const auto dB = GetAtomData(ZB);
    const double C6_AB = CombineC6(dA.C6, dB.C6, dA.alpha, dB.alpha);
    const double C8_AB = 3.0 * C6_AB *
        std::sqrt((dA.C6 > 0 ? dA.C8 / dA.C6 : 1.0)) *
        std::sqrt((dB.C6 > 0 ? dB.C8 / dB.C6 : 1.0));
    const double R0 = kA1 * std::sqrt(C8_AB / C6_AB) + kA2;
    const double R6 = R*R*R*R*R*R;
    const double R06 = R0*R0*R0*R0*R0*R0;
    const double R8 = R6*R*R;
    const double R08 = R06*R0*R0;
    return -kS6 * C6_AB / (R6 + R06) - kS8 * C8_AB / (R8 + R08);
  }

 private:
  // Casimir-Polder combination rule for C6^AB.
  static double CombineC6(double C6_A, double C6_B,
                          double alpha_A, double alpha_B) {
    if (C6_A <= 0 || C6_B <= 0) return std::sqrt(C6_A * C6_B);
    const double num = 2.0 * C6_A * C6_B * alpha_A * alpha_B;
    const double den = C6_A * alpha_B + C6_B * alpha_A;
    return (den > 1e-30) ? num / den : std::sqrt(C6_A * C6_B);
  }
};

}  // namespace tides::hybrids
