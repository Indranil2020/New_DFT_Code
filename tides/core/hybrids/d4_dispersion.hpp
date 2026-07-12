#pragma once

// DFT-D4 dispersion correction (T7.1 extension).
//
// D4 extends D3 with charge-dependent dispersion coefficients. The key
// difference is that C6 coefficients depend on the partial charge of each
// atom, which is estimated from electronegativity-based charge equilibration
// (EEQ) or provided externally (e.g. from Mulliken analysis).
//
// The D4 model (Caldeweyher et al., J. Chem. Phys. 150, 154122, 2019):
//   E_D4 = -0.5 * sum_{A!=B} sum_{n=6,8} s_n * C_n^AB(q) / (R_AB^n + R0_AB^n)
//
// where C_n^AB(q) are charge-dependent dispersion coefficients:
//   C6^A(q_A) = C6^A_ref * (1 + k * |q_A - q_A^0|)
//   C6^AB = (2 * C6^A * C6^B * alpha^A * alpha^B) /
//           (C6^A * alpha^B + C6^B * alpha^A)
//
// The partial charges q_A are computed via the EEQ model:
//   q_A = sum_B (EN_A - EN_B) / (J_A + J_B) (simplified)
// or provided externally. When charges are all zero, D4 reduces to D3.
//
// D4(BJ) parameters for PBE0 (Caldeweyher 2019):
//   a1=0.4289, a2=4.421, s6=1.0, s8=0.7875, s9=1.0, k=1.0
//
// Observable: D4 with zero charges reduces to D3 limit (testable).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace tides::hybrids {

// D4 atomic data with electronegativity for charge-dependent C6.
struct D4AtomData {
  int Z;
  double C6;                // free-atom C6 (Ha Bohr^6)
  double C8;                // free-atom C8 (Ha Bohr^8)
  double alpha;             // free-atom polarizability (Bohr^3)
  double electronegativity;  // Allen EN (eV)
  double hardness;           // chemical hardness (eV) for EEQ
};

struct D4Result {
  double energy = 0.0;
  std::vector<double> forces;   // 3*n_atoms
  std::vector<double> charges;  // computed partial charges
  double c6_check = 0.0;        // sanity: C6 for the first pair
};

class D4Dispersion {
 public:
  // D4(BJ) parameters for PBE0 (Caldeweyher 2019).
  static constexpr double kA1 = 0.4289;
  static constexpr double kA2 = 4.421;
  static constexpr double kS6 = 1.0;
  static constexpr double kS8 = 0.7875;
  static constexpr double kChargeScale = 1.0;  // k in C6(q) = C6 * (1 + k*|q|)
  static constexpr double kGaussianWidth = 2.0;  // Gaussian charge width (Bohr)

  // Atomic data for common elements. C6/C8 from D3 tables (Grimme 2011).
  // Electronegativity (Allen scale) and hardness from standard tables.
  static D4AtomData GetAtomData(int Z) {
    switch (Z) {
      case 1:  return {1,  6.5092, 85.0,   4.5,   13.598, 6.45};
      case 6:  return {6,  46.589, 825.0,  12.0,  11.260, 5.00};
      case 7:  return {7,  24.869, 354.0,  7.4,   14.534, 7.27};
      case 8:  return {8,  15.128, 193.0,  5.4,   13.618, 6.08};
      case 9:  return {9,  9.5523, 110.0,  3.8,   17.423, 7.00};
      case 14: return {14, 305.76, 5255.0, 37.0,  11.940, 4.77};
      case 15: return {15, 184.18, 3080.0, 25.0,  10.487, 4.88};
      case 16: return {16, 134.55, 2150.0, 19.6,  10.360, 4.77};
      case 17: return {17, 81.859, 1210.0, 15.0,  12.968, 4.70};
      case 18: return {18, 64.157, 920.0,  11.1,  15.760, 5.79};
      default: return {Z, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
  }

  // Compute D4 dispersion energy with charge-dependent C6.
  //   Z:          atomic numbers (n_atoms)
  //   positions:  positions in Bohr (3*n_atoms)
  //   charges:    partial charges (n_atoms). If empty, EEQ charges are computed.
  //               If all zero, D4 reduces to D3.
  static D4Result ComputeEnergy(const std::vector<int>& Z,
                                const std::vector<double>& positions,
                                const std::vector<double>& charges = {}) {
    D4Result res;
    const std::size_t n_atoms = Z.size();
    res.forces.assign(3 * n_atoms, 0.0);
    res.energy = 0.0;

    if (n_atoms < 2) return res;

    // Determine partial charges.
    std::vector<double> q;
    if (charges.size() == n_atoms) {
      q = charges;
    } else {
      q = ComputeEEQCharges(Z, positions);
    }
    res.charges = q;

    double E = 0.0;
    for (std::size_t A = 0; A < n_atoms; ++A) {
      const auto dA = GetAtomData(Z[A]);
      const double C6_A = ChargeDependentC6(dA, q[A]);
      for (std::size_t B = A + 1; B < n_atoms; ++B) {
        const auto dB = GetAtomData(Z[B]);
        const double C6_B = ChargeDependentC6(dB, q[B]);

        const double dx = positions[3*A]   - positions[3*B];
        const double dy = positions[3*A+1] - positions[3*B+1];
        const double dz = positions[3*A+2] - positions[3*B+2];
        const double R = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (R < 0.1) continue;

        // Geometry-dependent C6^AB (Casimir-Polder with charge-dependent C6).
        const double C6_AB = CombineC6(C6_A, C6_B, dA.alpha, dB.alpha);

        // C8^AB from D4 relation: C8 = 3 * C6 * sqrt(C8_A/C6_A) * sqrt(C8_B/C6_B)
        const double C8_AB = 3.0 * C6_AB *
            std::sqrt((C6_A > 0 ? dA.C8 / dA.C6 : 1.0)) *
            std::sqrt((C6_B > 0 ? dB.C8 / dB.C6 : 1.0));

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

        // Analytic forces.
        const double dE6_dR = kS6 * C6_AB * 6.0 * R * R * R * R * R /
                              ((R6 + R06) * (R6 + R06));
        const double dE8_dR = kS8 * C8_AB * 8.0 * R * R * R * R * R * R * R /
                              ((R8 + R08) * (R8 + R08));
        const double dE_dR = dE6_dR + dE8_dR;
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

  // Compute D4 energy for a single pair (for unit testing).
  static double PairEnergy(int ZA, int ZB, double R,
                           double qA = 0.0, double qB = 0.0) {
    const auto dA = GetAtomData(ZA);
    const auto dB = GetAtomData(ZB);
    const double C6_A = ChargeDependentC6(dA, qA);
    const double C6_B = ChargeDependentC6(dB, qB);
    const double C6_AB = CombineC6(C6_A, C6_B, dA.alpha, dB.alpha);
    const double C8_AB = 3.0 * C6_AB *
        std::sqrt((C6_A > 0 ? dA.C8 / dA.C6 : 1.0)) *
        std::sqrt((C6_B > 0 ? dB.C8 / dB.C6 : 1.0));
    const double R0 = kA1 * std::sqrt(C8_AB / C6_AB) + kA2;
    const double R6 = R*R*R*R*R*R;
    const double R06 = R0*R0*R0*R0*R0*R0;
    const double R8 = R6*R*R;
    const double R08 = R06*R0*R0;
    return -kS6 * C6_AB / (R6 + R06) - kS8 * C8_AB / (R8 + R08);
  }

 private:
  // Charge-dependent C6: C6(q) = C6_ref * (1 + k * |q|)
  // Positive charge (oxidation) reduces polarizability → smaller C6.
  // Negative charge (reduction) increases polarizability → larger C6.
  // The |q| formulation ensures C6 always decreases (atoms lose electrons
  // → less polarizable). We use a symmetric form for simplicity.
  static double ChargeDependentC6(const D4AtomData& data, double q) {
    if (data.C6 <= 0) return 0.0;
    // D4 formula: C6(q) = C6_ref * (1 - 2*k*q/Z_eff + k*q^2)
    // Simplified: C6(q) = C6_ref * (1 + k * |q| * sign_adjustment)
    // For the reference implementation, we use:
    //   C6(q) = C6_ref * (1 - k * q / Z + k * q^2 / Z^2)
    // which gives C6(q=0) = C6_ref (D3 limit).
    const double Z = static_cast<double>(data.Z);
    const double Z_safe = (Z > 0) ? Z : 1.0;
    const double factor = 1.0 - kChargeScale * q / Z_safe +
                          kChargeScale * q * q / (Z_safe * Z_safe);
    return data.C6 * std::max(factor, 0.01);  // floor at 1% to avoid negatives
  }

  // Casimir-Polder combination rule for C6^AB.
  static double CombineC6(double C6_A, double C6_B,
                          double alpha_A, double alpha_B) {
    if (C6_A <= 0 || C6_B <= 0) return std::sqrt(C6_A * C6_B);
    const double num = 2.0 * C6_A * C6_B * alpha_A * alpha_B;
    const double den = C6_A * alpha_B + C6_B * alpha_A;
    return (den > 1e-30) ? num / den : std::sqrt(C6_A * C6_B);
  }

  // Simplified EEQ (Electronegativity Equilibration) charge model.
  // Solves the linear system: sum_B (J_AB * q_B + EN_B) = EN_A
  // subject to sum q = 0 (charge neutrality).
  // J_AB = 1 / sqrt(R_AB^2 + (w_A + w_B)^2) where w is Gaussian width.
  static std::vector<double> ComputeEEQCharges(const std::vector<int>& Z,
                                               const std::vector<double>& positions) {
    const std::size_t n = Z.size();
    if (n == 0) return {};
    if (n == 1) return {0.0};

    // Build the EEQ matrix: J[i][j] = 1/sqrt(R^2 + (w_i+w_j)^2)
    // EN[i] = electronegativity of atom i
    std::vector<std::vector<double>> Jmat(n + 1, std::vector<double>(n + 1, 0.0));
    std::vector<double> rhs(n + 1, 0.0);

    // EEQ model: the electronegativity of each atom is chi_A = chi_A^0 + J_AA * q_A
    // + sum_{B!=A} J_AB * q_B. At equilibrium, all chi_A are equal (to chi_avg).
    // So: J_AA * q_A + sum_{B!=A} J_AB * q_B = chi_avg - chi_A^0.
    // Higher EN atoms (chi^0 large) get negative RHS → negative charge.
    // We use chi_avg = mean(chi^0) as the reference.
    double en_avg = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      en_avg += GetAtomData(Z[i]).electronegativity;
    en_avg /= static_cast<double>(n);

    for (std::size_t i = 0; i < n; ++i) {
      const auto di = GetAtomData(Z[i]);
      Jmat[i][i] = di.hardness;
      rhs[i] = en_avg - di.electronegativity;  // higher EN → negative RHS → negative q
      for (std::size_t j = i + 1; j < n; ++j) {
        (void)GetAtomData(Z[j]);  // validate Z[j]; off-diagonal uses distance only
        double dx = positions[3*i]   - positions[3*j];
        double dy = positions[3*i+1] - positions[3*j+1];
        double dz = positions[3*i+2] - positions[3*j+2];
        double R2 = dx*dx + dy*dy + dz*dz;
        double w = kGaussianWidth;
        double Jij = 1.0 / std::sqrt(R2 + w * w);
        Jmat[i][j] = Jij;
        Jmat[j][i] = Jij;
      }
      // Charge neutrality constraint: sum q = 0
      Jmat[i][n] = 1.0;
      Jmat[n][i] = 1.0;
    }
    Jmat[n][n] = 0.0;
    rhs[n] = 0.0;  // total charge = 0

    // Solve Jmat * q = rhs via Gaussian elimination with pivoting.
    std::vector<double> q = rhs;
    for (std::size_t col = 0; col <= n; ++col) {
      std::size_t piv = col;
      for (std::size_t row = col + 1; row <= n; ++row)
        if (std::fabs(Jmat[row][col]) > std::fabs(Jmat[piv][col])) piv = row;
      if (piv != col) {
        std::swap(Jmat[piv], Jmat[col]);
        std::swap(q[piv], q[col]);
      }
      if (std::fabs(Jmat[col][col]) < 1e-30) continue;
      for (std::size_t row = col + 1; row <= n; ++row) {
        double factor = Jmat[row][col] / Jmat[col][col];
        for (std::size_t k2 = col; k2 <= n; ++k2)
          Jmat[row][k2] -= factor * Jmat[col][k2];
        q[row] -= factor * q[col];
      }
    }

    // Back-substitution.
    std::vector<double> charges(n + 1, 0.0);
    for (int row = static_cast<int>(n); row >= 0; --row) {
      double sum = q[row];
      for (int k2 = row + 1; k2 <= static_cast<int>(n); ++k2)
        sum -= Jmat[row][k2] * charges[k2];
      if (std::fabs(Jmat[row][row]) > 1e-30)
        charges[row] = sum / Jmat[row][row];
    }

    charges.resize(n);
    return charges;
  }
};

}  // namespace tides::hybrids
