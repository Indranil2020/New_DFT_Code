#pragma once

// Ewald summation for periodic ion-ion electrostatics.
//
// For a periodic system with point charges {Z_i} at positions {r_i} in a
// cell with lattice vectors, the Coulomb energy diverges if computed
// naively. Ewald summation splits the conditionally convergent sum into
// three absolutely convergent parts:
//
// 1. Real-space sum: E_real = 0.5 * sum_{i,j} sum_{n} Z_i Z_j *
//        erfc(alpha * |r_ij + n*L|) / |r_ij + n*L|
//    (converges fast for large alpha; erfc suppresses long-range)
//
// 2. Reciprocal-space sum: E_recip = 0.5 * (4*pi/V) * sum_{G!=0}
//        exp(-G^2/(4*alpha^2)) / G^2 * |sum_i Z_i exp(-i G.r_i)|^2
//    (converges fast for small alpha)
//
// 3. Self correction: E_self = -0.5 * sum_i Z_i^2 * alpha / sqrt(pi)
//    (removes the self-interaction introduced by the Gaussian screening)
//
// For charged systems (sum Z_i != 0), a neutralizing background is added:
//    E_bg = -0.5 * (pi / (alpha^2 * V)) * (sum Z_i)^2
//
// The parameter alpha controls the real/reciprocal balance. Optimal alpha
// minimizes total cost: alpha ~ sqrt(pi) * (N/V)^(1/3) * (n_real/n_recip)^(1/6).
//
// Observable: Ewald energy of NaCl crystal matches known reference to <= 1e-6 Ha.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace tides::scf {

// Ewald summation result.
struct EwaldResult {
  double energy = 0.0;          // total Ewald energy
  double real_space = 0.0;      // real-space contribution
  double reciprocal = 0.0;      // reciprocal-space contribution
  double self_correction = 0.0; // self-interaction correction
  double background = 0.0;      // neutralizing background (if charged)
  bool ok = false;
};

// Ewald summation for periodic ion-ion electrostatics.
class EwaldSum {
 public:
  // Compute the Ewald ion-ion energy for a periodic system.
  //
  //   positions: flat array [x0,y0,z0, x1,y1,z1, ...] (Bohr)
  //   charges:   array of nuclear charges Z_i
  //   cell_vectors: 3x3 matrix, rows are lattice vectors [a1, a2, a3] (Bohr)
  //   alpha:     Ewald screening parameter (Bohr^-1)
  //   n_real:    number of real-space lattice vector shells
  //   n_recip:   number of reciprocal-space G-vector shells
  static EwaldResult Compute(
      const std::vector<double>& positions,
      const std::vector<double>& charges,
      const std::array<double, 9>& cell_vectors,
      double alpha = 0.0,
      int n_real = 0,
      int n_recip = 0) {
    EwaldResult result;
    const std::size_t n_atoms = charges.size();
    if (n_atoms == 0 || positions.size() != 3 * n_atoms) return result;

    // Compute cell volume and reciprocal lattice vectors.
    const auto& a1 = std::array<double, 3>{cell_vectors[0], cell_vectors[1], cell_vectors[2]};
    const auto& a2 = std::array<double, 3>{cell_vectors[3], cell_vectors[4], cell_vectors[5]};
    const auto& a3 = std::array<double, 3>{cell_vectors[6], cell_vectors[7], cell_vectors[8]};

    const double volume = TripleProduct(a1, a2, a3);
    if (volume <= 0) return result;

    // Reciprocal lattice vectors: b_i = 2*pi * (a_j x a_k) / V
    const auto b1 = CrossProduct(a2, a3, 2.0 * M_PI / volume);
    const auto b2 = CrossProduct(a3, a1, 2.0 * M_PI / volume);
    const auto b3 = CrossProduct(a1, a2, 2.0 * M_PI / volume);

    // Auto-select alpha and cutoffs if not provided.
    double eff_alpha = alpha;
    if (eff_alpha <= 0) {
      // Balance real and reciprocal convergence.
      // For small cells, use a conservative alpha that doesn't over-suppress
      // the real-space sum. alpha ~ 1.5 / r_nn where r_nn ~ (V/N)^(1/3).
      const double r_nn = std::cbrt(volume / static_cast<double>(n_atoms));
      eff_alpha = 1.5 / r_nn;
    }
    int eff_n_real = n_real;
    if (eff_n_real <= 0) {
      // Need erfc(alpha * r_cutoff) < 1e-10 => r_cutoff ~ 5/alpha
      const double cell_len = std::cbrt(volume);
      eff_n_real = static_cast<int>(std::ceil(5.0 / eff_alpha / cell_len)) + 1;
      eff_n_real = std::max(eff_n_real, 4);
      eff_n_real = std::min(eff_n_real, 15);
    }
    int eff_n_recip = n_recip;
    if (eff_n_recip <= 0) {
      // Need exp(-G^2/(4*alpha^2)) < 1e-10 => G_max ~ 10*alpha
      const double cell_len = std::cbrt(volume);
      eff_n_recip = static_cast<int>(std::ceil(10.0 * eff_alpha / cell_len)) + 1;
      eff_n_recip = std::max(eff_n_recip, 4);
      eff_n_recip = std::min(eff_n_recip, 25);
    }

    // 1. Real-space sum.
    result.real_space = RealSpaceSum(positions, charges, a1, a2, a3,
                                      eff_alpha, eff_n_real);

    // 2. Reciprocal-space sum.
    result.reciprocal = ReciprocalSum(positions, charges, b1, b2, b3,
                                       volume, eff_alpha, eff_n_recip);

    // 3. Self correction: -sum_i Z_i^2 * alpha / sqrt(pi)
    // (No 0.5 factor — self-interaction is not a pair term.)
    result.self_correction = 0.0;
    for (std::size_t i = 0; i < n_atoms; ++i)
      result.self_correction -= charges[i] * charges[i];
    result.self_correction *= eff_alpha / std::sqrt(M_PI);

    // 4. Background correction for charged systems.
    double total_charge = 0.0;
    for (std::size_t i = 0; i < n_atoms; ++i) total_charge += charges[i];
    if (std::fabs(total_charge) > 1e-10) {
      result.background = -0.5 * M_PI / (eff_alpha * eff_alpha * volume) *
          total_charge * total_charge;
    }

    result.energy = result.real_space + result.reciprocal +
                    result.self_correction + result.background;
    result.ok = true;
    return result;
  }

  // Direct (non-periodic) Coulomb sum for comparison.
  static double DirectCoulomb(const std::vector<double>& positions,
                               const std::vector<double>& charges) {
    const std::size_t n = charges.size();
    double E = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = i + 1; j < n; ++j) {
        const double dx = positions[3 * i] - positions[3 * j];
        const double dy = positions[3 * i + 1] - positions[3 * j + 1];
        const double dz = positions[3 * i + 2] - positions[3 * j + 2];
        const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (r > 1e-10) E += charges[i] * charges[j] / r;
      }
    return E;
  }

 private:
  static double TripleProduct(const std::array<double, 3>& a,
                               const std::array<double, 3>& b,
                               const std::array<double, 3>& c) {
    return a[0] * (b[1] * c[2] - b[2] * c[1]) -
           a[1] * (b[0] * c[2] - b[2] * c[0]) +
           a[2] * (b[0] * c[1] - b[1] * c[0]);
  }

  static std::array<double, 3> CrossProduct(const std::array<double, 3>& a,
                                             const std::array<double, 3>& b,
                                             double scale) {
    return {
      scale * (a[1] * b[2] - a[2] * b[1]),
      scale * (a[2] * b[0] - a[0] * b[2]),
      scale * (a[0] * b[1] - a[1] * b[0])
    };
  }

  static double RealSpaceSum(
      const std::vector<double>& positions,
      const std::vector<double>& charges,
      const std::array<double, 3>& a1,
      const std::array<double, 3>& a2,
      const std::array<double, 3>& a3,
      double alpha, int n_shell) {
    const std::size_t n_atoms = charges.size();
    double E = 0.0;
    const double r_cutoff = 5.0 / alpha;  // erfc(5) ~ 1e-12

    for (std::size_t i = 0; i < n_atoms; ++i) {
      for (std::size_t j = 0; j < n_atoms; ++j) {
        const double dx0 = positions[3 * i] - positions[3 * j];
        const double dy0 = positions[3 * i + 1] - positions[3 * j + 1];
        const double dz0 = positions[3 * i + 2] - positions[3 * j + 2];

        for (int n1 = -n_shell; n1 <= n_shell; ++n1) {
          for (int n2 = -n_shell; n2 <= n_shell; ++n2) {
            for (int n3 = -n_shell; n3 <= n_shell; ++n3) {
              // Skip self-image at (0,0,0) when i==j.
              if (i == j && n1 == 0 && n2 == 0 && n3 == 0) continue;

              const double dx = dx0 + n1 * a1[0] + n2 * a2[0] + n3 * a3[0];
              const double dy = dy0 + n1 * a1[1] + n2 * a2[1] + n3 * a3[1];
              const double dz = dz0 + n1 * a1[2] + n2 * a2[2] + n3 * a3[2];
              const double r2 = dx * dx + dy * dy + dz * dz;
              if (r2 > r_cutoff * r_cutoff) continue;
              const double r = std::sqrt(r2);
              E += charges[i] * charges[j] * std::erfc(alpha * r) / r;
            }
          }
        }
      }
    }
    return 0.5 * E;
  }

  static double ReciprocalSum(
      const std::vector<double>& positions,
      const std::vector<double>& charges,
      const std::array<double, 3>& b1,
      const std::array<double, 3>& b2,
      const std::array<double, 3>& b3,
      double volume, double alpha, int n_shell) {
    const std::size_t n_atoms = charges.size();
    double E = 0.0;
    const double g_cutoff = 10.0 * alpha;  // exp(-25) ~ 1e-11

    for (int n1 = -n_shell; n1 <= n_shell; ++n1) {
      for (int n2 = -n_shell; n2 <= n_shell; ++n2) {
        for (int n3 = -n_shell; n3 <= n_shell; ++n3) {
          if (n1 == 0 && n2 == 0 && n3 == 0) continue;

          const double gx = n1 * b1[0] + n2 * b2[0] + n3 * b3[0];
          const double gy = n1 * b1[1] + n2 * b2[1] + n3 * b3[1];
          const double gz = n1 * b1[2] + n2 * b2[2] + n3 * b3[2];
          const double g2 = gx * gx + gy * gy + gz * gz;
          if (g2 > g_cutoff * g_cutoff) continue;

          // Structure factor: S(G) = sum_i Z_i * exp(-i G.r_i)
          // |S(G)|^2 = (sum Z_i cos(G.r_i))^2 + (sum Z_i sin(G.r_i))^2
          double sf_cos = 0.0, sf_sin = 0.0;
          for (std::size_t i = 0; i < n_atoms; ++i) {
            const double gr = gx * positions[3 * i] +
                              gy * positions[3 * i + 1] +
                              gz * positions[3 * i + 2];
            sf_cos += charges[i] * std::cos(gr);
            sf_sin += charges[i] * std::sin(gr);
          }
          const double sf2 = sf_cos * sf_cos + sf_sin * sf_sin;

          E += std::exp(-g2 / (4.0 * alpha * alpha)) * sf2 / g2;
        }
      }
    }
    return 0.5 * (4.0 * M_PI / volume) * E;
  }
};

}  // namespace tides::scf
