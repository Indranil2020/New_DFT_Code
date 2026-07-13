#pragma once

// Atom-centered (Becke) integration grid for exchange-correlation quadrature.
//
// AUDIT B1-B5 ROOT CAUSE: XC energy/potential were integrated on a single
// uniform Cartesian grid (molecule_driver / nao_driver), which cannot resolve
// the nuclear cusp of an all-electron density (rho ~ e^{-Zr}). That is why the
// H atom was off by ~0.08 Ha, H2O by ~2.8 Ha, and Ne by ~66 Ha. Real molecular
// DFT codes (PySCF, FHI-aims, ...) integrate XC on an atom-centered grid:
//   - a radial quadrature per atom (points cluster near the nucleus), times
//   - an angular (spherical) quadrature per radial shell, with
//   - Becke fuzzy-cell weights that partition space among atoms so the union of
//     per-atom spherical grids integrates the molecular density.
//
// This module produces {position, weight} points; the weight already folds in
// r^2 dr (radial), dOmega (angular), and the Becke partition. An integral is
// then just  I = sum_g w_g f(r_g).  The grid is basis-agnostic; the caller
// evaluates orbitals at the points.
//
// Scheme choices (all standard, all generated at runtime — no large tables):
//   Radial : Mura-Knowles Log3 mapping (J. Chem. Phys. 104, 9848 (1996)).
//   Angular: Gauss-Legendre in cos(theta) x uniform in phi. A product spherical
//            quadrature; correct and generatable at any order. (Lebedev grids
//            are more point-efficient and can replace this later; the interface
//            does not change.)
//   Partition: Becke's iterated-polynomial fuzzy cells with Bragg-Slater
//            size adjustment (J. Chem. Phys. 88, 2547 (1988)).

#include <cmath>
#include <cstddef>
#include <vector>

namespace tides::grid {

// One quadrature point: Cartesian position (Bohr) and its integration weight
// (Bohr^3), where weight = w_radial * w_angular * becke_cell_weight.
struct AtomGridPoint {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double w = 0.0;
};

namespace detail {

// Gauss-Legendre nodes/weights on [-1, 1] via Newton-Raphson on P_n.
// nodes/weights are sized to n. Standard algorithm; converges to ~1e-15.
inline void GaussLegendre(int n, std::vector<double>& nodes,
                          std::vector<double>& weights) {
  nodes.assign(static_cast<std::size_t>(n), 0.0);
  weights.assign(static_cast<std::size_t>(n), 0.0);
  const double kPi = 3.14159265358979323846;
  const int m = (n + 1) / 2;  // roots are symmetric; compute half
  for (int i = 1; i <= m; ++i) {
    // Initial guess for the i-th root.
    double x = std::cos(kPi * (i - 0.25) / (n + 0.5));
    double pp = 0.0;
    for (int iter = 0; iter < 100; ++iter) {
      // Evaluate P_n(x) and P_{n-1}(x) by recurrence.
      double p0 = 1.0, p1 = x;
      for (int j = 2; j <= n; ++j) {
        double p2 = ((2.0 * j - 1.0) * x * p1 - (j - 1.0) * p0) / j;
        p0 = p1;
        p1 = p2;
      }
      // Derivative: P_n'(x) = n (x P_n - P_{n-1}) / (x^2 - 1).
      pp = n * (x * p1 - p0) / (x * x - 1.0);
      double dx = -p1 / pp;
      x += dx;
      if (std::fabs(dx) < 1e-15) break;
    }
    const double w = 2.0 / ((1.0 - x * x) * pp * pp);
    nodes[i - 1] = -x;
    nodes[n - i] = x;
    weights[i - 1] = w;
    weights[n - i] = w;
  }
}

// Bragg-Slater atomic radii (Bohr) for Becke cell size adjustment.
// Values from Slater, J. Chem. Phys. 41, 3199 (1964) (H set to 0.35 Angstrom
// per Becke's original prescription), converted to Bohr. Only Z=1..18 needed
// for the current light-element scope; falls back to 1.5 Bohr above that.
inline double BraggSlaterRadius(int Z) {
  // Angstrom, index by Z (0 unused).
  static const double kA[19] = {
      0.00,  // 0
      0.35,  // H  (Becke's special value)
      0.28,  // He (Slater gives none; small)
      1.45,  // Li
      1.05,  // Be
      0.85,  // B
      0.70,  // C
      0.65,  // N
      0.60,  // O
      0.50,  // F
      0.45,  // Ne (small closed shell)
      1.80,  // Na
      1.50,  // Mg
      1.25,  // Al
      1.10,  // Si
      1.00,  // P
      1.00,  // S
      1.00,  // Cl
      0.90   // Ar
  };
  const double ang = (Z >= 1 && Z <= 18) ? kA[Z] : 1.50 * 0.529177210903;
  return ang / 0.529177210903;  // Angstrom -> Bohr
}

// Becke's smoothing step function s(mu) = 0.5 (1 - f(f(f(mu)))),
// with f(mu) = 1.5 mu - 0.5 mu^3 (iteration order k = 3).
inline double BeckeStep(double mu) {
  double f = mu;
  for (int k = 0; k < 3; ++k) f = 1.5 * f - 0.5 * f * f * f;
  return 0.5 * (1.0 - f);
}

}  // namespace detail

// Build the molecular Becke grid.
//   positions       : 3*n_atoms Cartesian coordinates (Bohr), atom-major.
//   atomic_numbers  : Z per atom (for radial scale + Bragg size adjustment).
//   n_radial        : radial points per atom (Mura-Knowles).
//   n_theta, n_phi  : angular resolution (Gauss-Legendre x uniform).
// The number of raw points is n_atoms * n_radial * n_theta * n_phi; points with
// negligible Becke weight are dropped.
inline std::vector<AtomGridPoint> BuildBeckeGrid(
    const std::vector<double>& positions, const std::vector<int>& atomic_numbers,
    int n_radial = 60, int n_theta = 16, int n_phi = 32) {
  const std::size_t n_atoms = atomic_numbers.size();
  std::vector<AtomGridPoint> grid;
  if (n_atoms == 0 || n_radial <= 0 || n_theta <= 0 || n_phi <= 0) return grid;

  const double kPi = 3.14159265358979323846;

  // Angular quadrature (shared by all radial shells and all atoms).
  std::vector<double> ct, wt;  // cos(theta) nodes and weights on [-1,1]
  detail::GaussLegendre(n_theta, ct, wt);
  struct Ang {
    double dx, dy, dz, w;
  };
  std::vector<Ang> ang;
  ang.reserve(static_cast<std::size_t>(n_theta) * n_phi);
  const double dphi_w = 2.0 * kPi / n_phi;  // uniform azimuthal weight
  for (int j = 0; j < n_theta; ++j) {
    const double cth = ct[j];
    const double sth = std::sqrt(std::max(0.0, 1.0 - cth * cth));
    for (int k = 0; k < n_phi; ++k) {
      const double phi = 2.0 * kPi * k / n_phi;
      ang.push_back({sth * std::cos(phi), sth * std::sin(phi), cth,
                     wt[j] * dphi_w});  // sum of w over sphere = 4*pi
    }
  }

  // Precompute inter-atomic distances and Becke size-adjustment coefficients.
  std::vector<double> Rij(n_atoms * n_atoms, 0.0);
  std::vector<double> aij(n_atoms * n_atoms, 0.0);  // clamped size adjustment
  for (std::size_t i = 0; i < n_atoms; ++i) {
    for (std::size_t j = 0; j < n_atoms; ++j) {
      if (i == j) continue;
      const double dx = positions[3 * i] - positions[3 * j];
      const double dy = positions[3 * i + 1] - positions[3 * j + 1];
      const double dz = positions[3 * i + 2] - positions[3 * j + 2];
      Rij[i * n_atoms + j] = std::sqrt(dx * dx + dy * dy + dz * dz);
      const double chi = detail::BraggSlaterRadius(atomic_numbers[i]) /
                         detail::BraggSlaterRadius(atomic_numbers[j]);
      const double u = (chi - 1.0) / (chi + 1.0);
      double a = u / (u * u - 1.0);
      if (a > 0.5) a = 0.5;
      if (a < -0.5) a = -0.5;
      aij[i * n_atoms + j] = a;
    }
  }

  // Reusable per-point distance buffer.
  std::vector<double> rI(n_atoms, 0.0);

  for (std::size_t a = 0; a < n_atoms; ++a) {
    const double Rax = positions[3 * a];
    const double Ray = positions[3 * a + 1];
    const double Raz = positions[3 * a + 2];
    // Mura-Knowles Log3 radial scale: 5.0 Bohr for light elements
    // (7.0 recommended for Li,Be,Na,Mg,K,Ca).
    const int Z = atomic_numbers[a];
    const double alpha =
        (Z == 3 || Z == 4 || Z == 11 || Z == 12 || Z == 19 || Z == 20) ? 7.0
                                                                        : 5.0;

    for (int ir = 1; ir <= n_radial; ++ir) {
      const double xr = static_cast<double>(ir) / (n_radial + 1.0);  // (0,1)
      const double one_m_x3 = 1.0 - xr * xr * xr;
      const double r = -alpha * std::log(one_m_x3);
      const double drdx = alpha * 3.0 * xr * xr / one_m_x3;
      // Radial weight includes r^2 (spherical volume element), dr/dx, and the
      // uniform spacing dx = 1/(n_radial+1).
      const double w_rad = (1.0 / (n_radial + 1.0)) * drdx * r * r;

      for (const Ang& ad : ang) {
        const double px = Rax + r * ad.dx;
        const double py = Ray + r * ad.dy;
        const double pz = Raz + r * ad.dz;

        // Becke fuzzy-cell weight for the parent atom `a` at this point.
        double cell_w = 1.0;
        if (n_atoms > 1) {
          for (std::size_t i = 0; i < n_atoms; ++i) {
            const double dx = px - positions[3 * i];
            const double dy = py - positions[3 * i + 1];
            const double dz = pz - positions[3 * i + 2];
            rI[i] = std::sqrt(dx * dx + dy * dy + dz * dz);
          }
          double Psum = 0.0, Pa = 0.0;
          for (std::size_t i = 0; i < n_atoms; ++i) {
            double Pi = 1.0;
            for (std::size_t j = 0; j < n_atoms; ++j) {
              if (i == j) continue;
              double mu = (rI[i] - rI[j]) / Rij[i * n_atoms + j];
              // Size adjustment: nu = mu + a_ij (1 - mu^2).
              const double aval = aij[i * n_atoms + j];
              mu = mu + aval * (1.0 - mu * mu);
              Pi *= detail::BeckeStep(mu);
            }
            Psum += Pi;
            if (i == a) Pa = Pi;
          }
          cell_w = (Psum > 0.0) ? (Pa / Psum) : 0.0;
        }

        const double w = w_rad * ad.w * cell_w;
        if (w > 1e-14) grid.push_back({px, py, pz, w});
      }
    }
  }
  return grid;
}

}  // namespace tides::grid
